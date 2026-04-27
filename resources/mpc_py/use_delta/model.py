from . import costs as c
from .. import params as p

from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
import casadi as ca
import numpy as np
from pathlib import Path

def build_model():
    model = AcadosModel()
    model.name = "use_delta"

    # Model state
    theta           = ca.SX.sym('theta',  3) # [rad]
    omega           = ca.SX.sym('omega',  3) # [rad/s]
    x     = ca.vertcat(theta, omega)
    x_dot = ca.SX.sym('x_dot', x.size1())
    model.x = x
    model.xdot = x_dot

    # Model control input(u)
    delta_theta_cmd = ca.SX.sym('delta_theta_cmd', 3) # [rad]
    model.u = delta_theta_cmd

    # Model parameter
    R_raw = ca.SX.sym('R_raw', 3, 3)    # desired attitude SO3 matrix
    W_raw = ca.SX.sym('W_raw', 3)       # desired angular rate [rad/s]
    Wdot_raw = ca.SX.sym('Wdot_raw', 3) # desired angular accel [rad/s^2]
    R_0   = ca.SX.sym('R_0', 3, 3)      # initial attitude SO3 matrix
    f_0 = ca.SX.sym('f_0')              # [N]
    d_hat = ca.SX.sym('d_hat', 3)       # torque disturbance [N.m]
    model.p  = ca.vertcat(ca.reshape(R_raw, 9, 1), W_raw, Wdot_raw, ca.reshape(R_0, 9, 1), f_0, d_hat)

    # Constants
    J = ca.DM(p.J_TENSOR)
    J_inv = ca.inv(J)
    zeta = float(p.ZETA)
    KR = ca.reshape(ca.DM(np.asarray(p.KR, dtype=np.float64)), 3, 1)
    KW = ca.reshape(ca.DM(np.asarray(p.KW, dtype=np.float64)), 3, 1)
    l = float(p.L_DIST)
    b_F0 = ca.vertcat(0.0, 0.0, -f_0)
    g_F0 = R_0 @ b_F0

    # ---------- math utils ----------
    def euler_zyx_to_R(theta: ca.SX) -> ca.SX:
        phi, th, psi = theta[0], theta[1], theta[2]

        cphi, sphi = ca.cos(phi), ca.sin(phi)
        cth,  sth  = ca.cos(th),  ca.sin(th)
        cpsi, spsi = ca.cos(psi), ca.sin(psi)

        Rz = ca.vertcat(ca.horzcat(cpsi, -spsi, 0.0),
                        ca.horzcat(spsi,  cpsi, 0.0),
                        ca.horzcat(0.0,   0.0,  1.0),)

        Ry = ca.vertcat(ca.horzcat(cth,  0.0,  sth),
                        ca.horzcat(0.0,  1.0,  0.0),
                        ca.horzcat(-sth, 0.0,  cth),)

        Rx = ca.vertcat(ca.horzcat(1.0,  0.0,   0.0),
                        ca.horzcat(0.0,  cphi, -sphi),
                        ca.horzcat(0.0,  sphi,  cphi),)

        return Rz @ Ry @ Rx

    def euler_zyx_rate_map(theta: ca.SX) -> ca.SX:
        phi, th = theta[0], theta[1]

        cphi, sphi = ca.cos(phi), ca.sin(phi)
        cth,  tth  = ca.cos(th), ca.tan(th)

        return ca.vertcat(ca.horzcat(1.0, sphi*tth, cphi*tth),
                          ca.horzcat(0.0,     cphi,    -sphi),
                          ca.horzcat(0.0, sphi/cth, cphi/cth),)
    
    def hat(w: ca.SX) -> ca.SX:
        return ca.vertcat(ca.horzcat(  0.0, -w[2],  w[1]),
                          ca.horzcat( w[2],   0.0, -w[0]),
                          ca.horzcat(-w[1],  w[0],  0.0))

    def expm_hat(w: ca.SX) -> ca.SX:
        th2 = ca.dot(w, w)          # theta^2
        th  = ca.sqrt(th2 + 1e-12)  # theta

        A = ca.sin(th) / th
        B = (1.0 - ca.cos(th)) / (th2 + 1e-12)

        K = hat(w)
        I = ca.SX.eye(3)
        return I + A*K + B*(K @ K)
    
    def vee(R: ca.SX) -> ca.SX:
        return ca.vertcat(R[2, 1], R[0, 2], R[1, 0])
    
    # ---------- Dynamics ----------
    # attitude (theta)
    theta_dot = euler_zyx_rate_map(theta) @ omega # (body->global)

    # angular rate (omega)
    R = euler_zyx_to_R(theta)  # (body->global)
    Rd = R_raw @ expm_hat(delta_theta_cmd)
    Wd = expm_hat(-delta_theta_cmd) @ W_raw
    Wd_dot = expm_hat(-delta_theta_cmd) @ Wdot_raw
    RtRd = R.T @ Rd
    e_R = 0.5 * vee(RtRd.T - RtRd)
    e_w = omega - RtRd @ Wd
    tau_d = - KR * e_R - KW * e_w + J@(hat(omega)@RtRd@Wd + RtRd@Wd_dot)
    omega_dot = J_inv@(tau_d + d_hat - ca.cross(omega, J@omega))

    f_expl = ca.vertcat(theta_dot, omega_dot)
    model.f_expl_expr = f_expl
    model.f_impl_expr = x_dot - f_expl

    # ---------- Propeller thrust expression ----------
    A = ca.vertcat(ca.horzcat(    l,     l,    -l,    -l),
                   ca.horzcat(    l,    -l,    -l,     l),
                   ca.horzcat(-zeta,  zeta, -zeta,  zeta),
                   ca.horzcat( -1.0,  -1.0,  -1.0,  -1.0))

    b_F = R.T @ g_F0
    w_d = ca.vertcat(tau_d, b_F[2])
    F_expr = ca.solve(A, w_d)
    model.con_h_expr   = F_expr

    return model

def build_ocp():
    model = build_model()

    ocp = AcadosOcp()
    ocp.model = model

    # ---------- horizon ----------
    ocp.solver_options.N_horizon = p.N
    ocp.solver_options.tf        = p.N * p.DT

    # ---------- costs ----------
    delta_theta_cmd = model.u
    
    model.cost_y_expr   = ca.vertcat(delta_theta_cmd) # 1~k-1 ref

    ocp.dims.ny   = 3
    
    ocp.cost.W = np.diag(np.asarray(c.Q_THETA, dtype=np.float64))

    ocp.cost.cost_type   = "NONLINEAR_LS"

    # cost reference default value
    ocp.cost.yref   = np.zeros((model.cost_y_expr.size()[0],))
    ocp.parameter_values = np.zeros((model.p.size()[0],))
    ocp.constraints.x0 = np.zeros(model.x.size()[0])

    # ---------- h_expr constraints ----------
    ocp.constraints.lh   = np.asarray(c.F_MIN, dtype=np.float64)
    ocp.constraints.uh   = np.asarray(c.F_MAX, dtype=np.float64)
    ocp.dims.nh   = 4

    # ---------- solver options ----------
    ocp.solver_options.qp_solver        = "PARTIAL_CONDENSING_HPIPM" # or "FULL_CONDENSING_HPIPM(5ms)" "PARTIAL_CONDENSING_HPIPM"(3ms) "FULL_CONDENSING_QPOASES(6ms)"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON" # "GAUSS_NEWTON" or "EXACT"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI" # "SQP" or "SQP_RTI" or "DDP" or "SQP_WITH_FEASIBLE_QP"
    ocp.solver_options.qp_solver_cond_N = p.N
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    # ocp.solver_options.print_level = 4

    # codegen dir
    codegen_dir = Path(__file__).resolve().parent / "generated"
    ocp.code_export_directory = str(codegen_dir)
    codegen_dir.mkdir(parents=True, exist_ok=True)

    return ocp