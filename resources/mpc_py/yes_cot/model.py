from . import costs as c
from .. import params as p

from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
import casadi as ca
import numpy as np
from pathlib import Path

def build_model():
    model = AcadosModel()
    model.name = "yes_cot_MRG"

    # Model state
    theta           = ca.SX.sym('theta',  3) # [rad]
    omega           = ca.SX.sym('omega',  3) # [rad/s]
    r_rotor         = ca.SX.sym('r_rotor',  8) # (r1x, r2x, r3x, r4x, r1y, r2y, r3y, r4y) [m]
    delta_theta_cmd = ca.SX.sym('delta_theta_cmd',  3) # [rad], Augmented state(command input)
    r_rotor_cmd     = ca.SX.sym('r_rotor_cmd',  8)     # [m], Augmented state(command input)
    x     = ca.vertcat(theta, omega, r_rotor, delta_theta_cmd, r_rotor_cmd)
    x_dot = ca.SX.sym('x_dot', x.size1())
    model.x = x
    model.xdot = x_dot

    # Model control input(u-rate)
    u_rate = ca.SX.sym('u_rate', 11) # 3: delta_theta_cmd_rate [rad/s] / 8: r_rotor_cmd_rate[m/s]
    model.u = u_rate

    # Model parameter
    R_raw = ca.SX.sym('R_raw', 3, 3) # desired attitude SO3 matrix
    omega_raw = ca.SX.sym('omega_raw', 3) # desired angular rate [rad/s]
    T_des = ca.SX.sym('T_des')       # [N]
    model.p  = ca.vertcat(ca.reshape(R_raw, 9, 1), omega_raw, T_des)

    # Constants
    J = ca.DM(p.J_TENSOR)
    J_inv = ca.inv(J)
    tau_inv = 1.0 / p.TAU
    zeta = p.ZETA
    KR = ca.DM(p.KR).reshape((3, 1))
    KW = ca.DM(p.KW).reshape((3, 1))
    r_off_x = ca.DM(p.R_OFF_X).reshape((4, 1))
    r_off_y = ca.DM(p.R_OFF_Y).reshape((4, 1))
    
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
    RtRd = R.T @ R_d
    e_R = 0.5 * vee(RtRd.T - RtRd)
    e_w = omega - RtRd @ omega_raw
    tau_d = - KR * e_R - KW * e_w
    omega_dot = J_inv @ (tau_d - ca.cross(omega, J @ omega))

    # rotor position (r_rotor, 1st-order)
    r_rotor_dot = tau_inv * (r_rotor_cmd - r_rotor)

    # Augmented dynamics
    u_cmd_dot = u_rate

    f_expl = ca.vertcat(theta_dot, omega_dot, r_rotor_dot, u_cmd_dot)
    model.f_expl_expr = f_expl
    model.f_impl_expr = x_dot - f_expl

    # ---------- Propeller thrust expression ----------
    A = ca.vertcat(ca.horzcat(-r_rotor[4], -r_rotor[5], -r_rotor[6], -r_rotor[7]),
                   ca.horzcat( r_rotor[0],  r_rotor[1],  r_rotor[2],  r_rotor[3]),
                   ca.horzcat(-zeta,  zeta, -zeta,  zeta),
                   ca.horzcat( -1.0,  -1.0,  -1.0,  -1.0))

    w_d = ca.vertcat(tau_d, T_des)
    F_expr = ca.solve(A, w_d)

    F_mean = (ca.sum1(F_expr) / 4.0)
    model.thrust_dev  = F_expr - F_mean * ca.SX.ones(4, 1)

    # ---------- Workspace & Rotor collision expression ----------
    # h_expr stacking order:
    #   [0:4)   : F1, F2, F3, F4
    #   [4:8)   : workspace radius constraints
    #   [8:16)  : workspace sign constraints
    #   [16:20) : collision distance constraints

    ws1 = []
    ws2 = []
    ws1_cmd = []
    ws2_cmd = []

    # workspace constraints:
    for i in range(4):
        rx = r_rotor[i]     - r_off_x[i]
        ry = r_rotor[4 + i] - r_off_y[i]

        rx_cmd = r_rotor_cmd[i]     - r_off_x[i]
        ry_cmd = r_rotor_cmd[4 + i] - r_off_y[i]

        ws1.append(rx*rx + ry*ry)
        ws2.extend([r_rotor[i], r_rotor[4 + i]])

        ws1_cmd.append(rx_cmd*rx_cmd + ry_cmd*ry_cmd)
        ws2_cmd.extend([r_rotor_cmd[i], r_rotor_cmd[4 + i]])

    # collision distance constraints:
    def dist2(i, j):
        dx = r_rotor[i] - r_rotor[j]
        dy = r_rotor[4 + i] - r_rotor[4 + j]
        return dx*dx + dy*dy
    
    def dist2_cmd(i, j):
        dx = r_rotor_cmd[i] - r_rotor_cmd[j]
        dy = r_rotor_cmd[4 + i] - r_rotor_cmd[4 + j]
        return dx*dx + dy*dy

    collision     = [dist2(0, 1),         dist2(1, 2),     dist2(2, 3),     dist2(3, 0),]
    collision_cmd = [dist2_cmd(0, 1), dist2_cmd(1, 2), dist2_cmd(2, 3), dist2_cmd(3, 0),]

    model.con_h_expr = ca.vertcat(F_expr, ca.vertcat(*ws1), ca.vertcat(*ws2), ca.vertcat(*collision), ca.vertcat(*ws1_cmd), ca.vertcat(*ws2_cmd), ca.vertcat(*collision_cmd),)

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
    delta_theta_cmd = model.x[14:17]
    delta_theta_cmd_rate = model.u[0:3]
    r_rotor_cmd_rate       = model.u[3:11]
    thrust_dev      = model.thrust_dev
    
    model.cost_y_expr   = ca.vertcat(omega, delta_theta_cmd, thrust_dev, delta_theta_cmd_rate, r_rotor_cmd_rate) # 1~k-1 ref
    model.cost_y_expr_e = ca.vertcat(omega, delta_theta_cmd, thrust_dev) # terminal(k) ref

    ocp.dims.ny   = 21
    ocp.dims.ny_e = 10
    
    ocp.cost.W   = np.diag(np.concatenate([c.Q_OMEGA, c.Q_THETA, c.Q_FDEV, c.R_THETA, c.R_ROTOR]).astype(np.float64))
    ocp.cost.W_e = np.diag(np.concatenate([c.Q_OMEGA, c.Q_THETA, c.Q_FDEV]).astype(np.float64))

    ocp.cost.cost_type   = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # cost reference default value
    ocp.cost.yref   = np.zeros((model.cost_y_expr.size()[0],))
    ocp.cost.yref_e = np.zeros((model.cost_y_expr_e.size()[0],))
    ocp.parameter_values = np.zeros((model.p.size()[0],))
    ocp.constraints.x0 = np.zeros(model.x.size()[0])

    # ---------- h_expr constraints ----------
    r_min_sq = p.R_MIN * p.R_MIN
    r_max_sq = p.R_MAX * p.R_MAX
    workspace_expr1_lb = np.array([r_min_sq, r_min_sq, r_min_sq, r_min_sq])
    workspace_expr1_ub = np.array([r_max_sq, r_max_sq, r_max_sq, r_max_sq])

    r_min_slk_sq = p.R_MIN_SLK * p.R_MIN_SLK
    r_max_slk_sq = p.R_MAX_SLK * p.R_MAX_SLK
    workspace_expr11_lb = np.array([r_min_slk_sq, r_min_slk_sq, r_min_slk_sq, r_min_slk_sq])
    workspace_expr11_ub = np.array([r_max_slk_sq, r_max_slk_sq, r_max_slk_sq, r_max_slk_sq])

    workspace_expr2_lb = np.array([0.0, -1e12, -1e12, -1e12, -1e12, 0.0, 0.0, 0.0], dtype=np.float64)
    workspace_expr2_ub = np.array([1e12, 0.0, 0.0, 0.0, 0.0, 1e12, 1e12, 1e12], dtype=np.float64)

    four_r_rotor_sq = 4.0 * p.R_ROTOR *p.R_ROTOR
    collision_expr_lb = np.array([four_r_rotor_sq, four_r_rotor_sq, four_r_rotor_sq, four_r_rotor_sq], dtype=np.float64)
    collision_expr_ub = np.array([1e12, 1e12, 1e12, 1e12], dtype=np.float64)

    four_r_rotor_slk_sq = 4.0 * p.R_ROTOR_SLK *p.R_ROTOR_SLK
    collision_expr2_lb = np.array([four_r_rotor_slk_sq, four_r_rotor_slk_sq, four_r_rotor_slk_sq, four_r_rotor_slk_sq], dtype=np.float64)

    ocp.constraints.lh = np.concatenate([c.F_MIN, workspace_expr1_lb, workspace_expr2_lb, collision_expr_lb, workspace_expr11_lb, workspace_expr2_lb, collision_expr2_lb]).astype(np.float64)
    ocp.constraints.uh = np.concatenate([c.F_MAX, workspace_expr1_ub, workspace_expr2_ub, collision_expr_ub, workspace_expr11_ub, workspace_expr2_ub, collision_expr_ub]).astype(np.float64)
    ocp.dims.nh   = 20

    # ---------- solver options ----------
    ocp.solver_options.qp_solver        = "PARTIAL_CONDENSING_HPIPM" # or "FULL_CONDENSING_HPIPM(5ms)" "PARTIAL_CONDENSING_HPIPM"(3ms) "FULL_CONDENSING_QPOASES(6ms)"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON" # "GAUSS_NEWTON" or "EXACT"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI" # "SQP" or "SQP_RTI" or "DDP" or "SQP_WITH_FEASIBLE_QP"
    ocp.solver_options.qp_solver_cond_N = p.N
    ocp.solver_options.qp_solver_iter_max = 10
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    # ocp.solver_options.print_level = 4

    # codegen dir
    codegen_dir = Path(__file__).resolve().parent / "generated"
    ocp.code_export_directory = str(codegen_dir)
    codegen_dir.mkdir(parents=True, exist_ok=True)
    
    return ocp