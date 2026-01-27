from . import params as p

from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
import casadi as ca
import numpy as np

def build_model():
    model = AcadosModel()
    model.name = "yes_cot_MRG"

    # Model state
    theta           = ca.SX.sym('theta',  3) # [rad]
    omega           = ca.SX.sym('omega',  3) # [rad/s]
    r_cot           = ca.SX.sym('r_cot',  2) # [m]
    delta_theta_cmd = ca.SX.sym('delta_theta_cmd',  3) # [rad], Augmented state(command input)
    r_cot_cmd       = ca.SX.sym('r_cot_cmd',  2)       # [m], Augmented state(command input)
    x     = ca.vertcat(theta, omega, r_cot, delta_theta_cmd, r_cot_cmd)
    x_dot = ca.SX.sym('x_dot', x.size1())
    model.x = x
    model.xdot = x_dot

    # Model control input(u-rate)
    u_rate = ca.SX.sym('u_rate', 5) # 3: delta_theta_cmd_rate [rad/s] / 2: r_cot_cmd_rate[m/s]
    model.u = u_rate

    # Model parameter
    R_raw = ca.SX.sym('R_raw', 3, 3) # SO3 matrix
    l     = ca.SX.sym('l')           # [m]
    T_des = ca.SX.sym('T_des')       # [N]
    model.p  = ca.vertcat(ca.reshape(R_raw, 9, 1), l, T_des)

    # Constants
    J = ca.DM(p.J_TENSOR)
    J_inv = ca.inv(J)
    tau_inv = 1.0 / p.TAU
    zeta = p.ZETA

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
    R_d = R_raw @ expm_hat(delta_theta_cmd)
    e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
    tau_d = -p.KR * e_R - p.KW * omega
    omega_dot = J_inv @ (tau_d - ca.cross(omega, J @ omega))

    # CoT (r_cot, 1st-order)
    r_cot_dot = tau_inv * (r_cot_cmd - r_cot)

    # Augmented dynamics
    u_cmd_dot = u_rate

    f_expl = ca.vertcat(theta_dot, omega_dot, r_cot_dot, u_cmd_dot)
    model.f_expl_expr = f_expl
    model.f_impl_expr = x_dot - f_expl

    # ---------- Propeller thrust expression ----------
    dx, dy = r_cot[0], r_cot[1]
    A = ca.vertcat(ca.horzcat( l-dy,  l-dy, -l-dy, -l-dy),
                   ca.horzcat( l+dx, -l+dx, -l+dx,  l+dx),
                   ca.horzcat(-zeta,  zeta, -zeta,  zeta),
                   ca.horzcat(  1.0,   1.0,   1.0,   1.0))

    w_d = ca.vertcat(tau_d, T_des)
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
    omega           = model.x[3:6]
    delta_theta_cmd = model.x[8:11]
    r_cot_cmd       = model.x[11:13]
    delta_theta_cmd_rate = model.u[0:3]
    r_cot_cmd_rate       = model.u[3:5]
    
    model.cost_y_expr   = ca.vertcat(omega, delta_theta_cmd, r_cot_cmd, delta_theta_cmd_rate, r_cot_cmd_rate) # 1~k-1 ref
    model.cost_y_expr_e = ca.vertcat(omega, delta_theta_cmd, r_cot_cmd) # terminal(k) ref

    ocp.dims.ny   = 13
    ocp.dims.ny_e = 8
    
    ocp.cost.W = np.diag(np.concatenate([p.Q_OMEGA, p.Q_THETA, p.Q_COT, p.R_THETA, p.R_COT]).astype(np.float64))
    ocp.cost.W_e = np.diag(np.concatenate([p.Q_OMEGA, p.Q_THETA, p.Q_COT]).astype(np.float64))

    ocp.cost.cost_type   = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # cost reference default value
    ocp.cost.yref   = np.zeros((model.cost_y_expr.size()[0],))
    ocp.cost.yref_e = np.zeros((model.cost_y_expr_e.size()[0],))
    ocp.parameter_values = np.zeros((model.p.size()[0],))
    ocp.constraints.x0 = np.zeros(model.x.size()[0])

    # ---------- h_expr constraints ----------
    ocp.constraints.lh   = p.F_MIN
    ocp.constraints.uh   = p.F_MAX
    ocp.dims.nh   = 4

    # ---------- x(augmented-state) constraints ----------
    idx_r_cot_cmd = np.array([11, 12], dtype=np.int64)

    ocp.constraints.idxbx   = idx_r_cot_cmd
    ocp.constraints.lbx     = p.COT_MIN
    ocp.constraints.ubx     = p.COT_MAX
    ocp.constraints.idxbx_e = idx_r_cot_cmd
    ocp.constraints.lbx_e   = p.COT_MIN
    ocp.constraints.ubx_e   = p.COT_MAX

    ocp.dims.nbx            = idx_r_cot_cmd.size
    ocp.dims.nbx_e          = idx_r_cot_cmd.size

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
    ocp.code_export_directory = "generated"
    return ocp