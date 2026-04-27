from . import costs as c
from .. import params as p

from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
import casadi as ca
import numpy as np
from pathlib import Path

def build_model():
    model = AcadosModel()
    model.name = "use_arm"

    # Model state
    theta           = ca.SX.sym('theta',  3) # [rad]
    omega           = ca.SX.sym('omega',  3) # [rad/s]
    r1              = ca.SX.sym('r1',  2)    # [m, rad]
    r2              = ca.SX.sym('r2',  2)    # [m, rad]
    r3              = ca.SX.sym('r3',  2)    # [m, rad]
    r4              = ca.SX.sym('r4',  2)    # [m, rad]
    x     = ca.vertcat(theta, omega, r1, r2, r3, r4)
    x_dot = ca.SX.sym('x_dot', x.size1())
    model.x = x
    model.xdot = x_dot

    # Model control input(u)
    u_cmd = ca.SX.sym('u_cmd', 8) # r_rotor_cmd [m, rad]
    model.u = u_cmd
    r1_cmd = u_cmd[0:2]
    r2_cmd = u_cmd[2:4]
    r3_cmd = u_cmd[4:6]
    r4_cmd = u_cmd[6:8]

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
    t_arm_inv  = 1.0 / p.TAU_ARM
    t_base_inv = 1.0 / p.TAU_BASE
    zeta = float(p.ZETA)
    KR = ca.reshape(ca.DM(np.asarray(p.KR, dtype=np.float64)), 3, 1)
    KW = ca.reshape(ca.DM(np.asarray(p.KW, dtype=np.float64)), 3, 1)
    r_off_x = ca.reshape(ca.DM(np.asarray(p.R_OFF_X, dtype=np.float64)), 4, 1)
    r_off_y = ca.reshape(ca.DM(np.asarray(p.R_OFF_Y, dtype=np.float64)), 4, 1)
    b_F0 = ca.vertcat(0.0, 0.0, -f_0)
    g_F0 = R_0 @ b_F0
    
    m_link = ca.reshape(ca.DM(np.asarray(p.M_LINK, dtype=np.float64)), 5, 1)
    m_link_sum = ca.sum1(m_link)
    inv_m_tot = 1.0 / (ca.DM(float(p.M_CENTER)) + 4.0 * m_link_sum)
    a1 = float(p.A_LINK[0])
    a2 = float(p.A_LINK[1])
    a3 = float(p.A_LINK[2])
    d1 = float(p.D_LINK[0])
    d2 = float(p.D_LINK[1])
    d3 = float(p.D_LINK[2])
    rz = float(p.R_Z)
    
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

    def vee(R: ca.SX) -> ca.SX:
        return ca.vertcat(R[2, 1], R[0, 2], R[1, 0])
    
    # ---------- Dynamics ----------
    # attitude (theta)
    theta_dot = euler_zyx_rate_map(theta) @ omega # (body->global)

    # angular rate (omega)
    R = euler_zyx_to_R(theta)  # (body->global)
    RtRraw = R.T @ R_raw
    e_R = 0.5 * vee(RtRraw.T - RtRraw)
    e_w = omega - RtRraw @ W_raw
    tau_d = - KR * e_R - KW * e_w + J@(hat(omega)@RtRraw@W_raw + RtRraw@Wdot_raw)
    omega_dot = J_inv@(tau_d + d_hat - ca.cross(omega, J@omega))

    # body->rotor pos in cartesian coordinate (state)
    r_pol = [r1, r2, r3, r4]
    r = []
    for a in range(4):
        ra = ca.vertcat(
            r_off_x[a, 0] + r_pol[a][0] * ca.cos(r_pol[a][1]),
            r_off_y[a, 0] + r_pol[a][0] * ca.sin(r_pol[a][1]))
        r.append(ra)

    # body->rotor pos in cartesian coordinate (cmd)
    r_pol_cmd = [r1_cmd, r2_cmd, r3_cmd, r4_cmd]
    r_cmd = []
    for a in range(4):
        ra_cmd = ca.vertcat(
            r_off_x[a, 0] + r_pol_cmd[a][0] * ca.cos(r_pol_cmd[a][1]),
            r_off_y[a, 0] + r_pol_cmd[a][0] * ca.sin(r_pol_cmd[a][1]))
        r_cmd.append(ra_cmd)

    # rotor position (r, 1st-order)
    r_dot = ca.SX.zeros(8, 1)
    r_dot[0] = t_arm_inv  * (r1_cmd[0] - r1[0]); r_dot[1] = t_base_inv * (r1_cmd[1] - r1[1])
    r_dot[2] = t_arm_inv  * (r2_cmd[0] - r2[0]); r_dot[3] = t_base_inv * (r2_cmd[1] - r2[1])
    r_dot[4] = t_arm_inv  * (r3_cmd[0] - r3[0]); r_dot[5] = t_base_inv * (r3_cmd[1] - r3[1])
    r_dot[6] = t_arm_inv  * (r4_cmd[0] - r4[0]); r_dot[7] = t_base_inv * (r4_cmd[1] - r4[1])
    model.r_dot = r_dot

    f_expl = ca.vertcat(theta_dot, omega_dot, r_dot)
    model.f_expl_expr = f_expl
    model.f_impl_expr = x_dot - f_expl

    # ---------- CoM position ----------
    pc = ca.SX.zeros(2, 1)
    for a in range(4):
        x_a = r_pol[a][0] - a1
        D_a = (x_a * x_a + rz * rz - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)

        # IK - elbow-up solution
        q3 = ca.atan2(ca.sqrt(1.0 - D_a * D_a), D_a)
        q2 = ca.atan2(rz, x_a) - ca.atan2(a3 * ca.sin(q3), a2 + a3 * ca.cos(q3))

        rho_c_a = (
            m_link[0, 0] * (a1 + d1)
          + m_link[1, 0] * (a1 + (a2 + d2) * ca.cos(q2))
          + m_link[2, 0] * (a1 + a2 * ca.cos(q2) + (a3 + d3) * ca.cos(q2 + q3))
          + (m_link[3, 0] + m_link[4, 0]) * r_pol[a][0]
        )
        pc = pc + ca.vertcat(m_link_sum * (r_off_x[a, 0] + rho_c_a * ca.cos(r_pol[a][1])), m_link_sum * (r_off_y[a, 0] + rho_c_a * ca.sin(r_pol[a][1])))
    pc = pc * inv_m_tot

    # ---------- Propeller thrust expression ----------
    A = ca.vertcat(ca.horzcat(-(r[0][1]-pc[1]), -(r[1][1]-pc[1]), -(r[2][1]-pc[1]), -(r[3][1]-pc[1])),
                   ca.horzcat( (r[0][0]-pc[0]),  (r[1][0]-pc[0]),  (r[2][0]-pc[0]),  (r[3][0]-pc[0])),
                   ca.horzcat(-zeta,  zeta, -zeta,  zeta),
                   ca.horzcat(-1.0,  -1.0,  -1.0,  -1.0))

    b_F = R.T @ g_F0
    w_d = ca.vertcat(tau_d, b_F[2])
    F_expr = ca.solve(A, w_d)

    # ---------- Thrust deviation expression ----------
    F_mean = (ca.sum1(F_expr) / 4.0)
    model.thrust_dev  = F_expr - F_mean * ca.SX.ones(4, 1)

    # ---------- Workspace & Rotor collision expression ----------
    def dist2_cart(rr, i, j):
        dx = rr[i][0] - rr[j][0]
        dy = rr[i][1] - rr[j][1]
        return dx*dx + dy*dy

    collision_cmd = ca.vertcat(
        dist2_cart(r_cmd, 0, 1),
        dist2_cart(r_cmd, 1, 2),
        dist2_cart(r_cmd, 2, 3),
        dist2_cart(r_cmd, 3, 0),
    )

    model.con_h_expr = ca.vertcat(F_expr, collision_cmd, r_dot)

    return model

def build_ocp():
    model = build_model()

    ocp = AcadosOcp()
    ocp.model = model

    # ---------- horizon ----------
    ocp.solver_options.N_horizon = p.N
    ocp.solver_options.tf        = p.N * p.DT

    # ---------- costs ----------
    thrust_dev           = model.thrust_dev
    r_dot                = model.r_dot
    
    model.cost_y_expr   = ca.vertcat(thrust_dev, r_dot) # 1~k-1 ref
    model.cost_y_expr_e = ca.vertcat(thrust_dev) # terminal(k) ref

    ocp.dims.ny   = 12
    ocp.dims.ny_e = 4
    
    ocp.cost.W   = np.diag(np.concatenate([c.Q_FDEV, c.R_ROTOR]).astype(np.float64))
    ocp.cost.W_e = np.diag(np.concatenate([c.Q_FDEV]).astype(np.float64))

    ocp.cost.cost_type   = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # cost reference default value
    ocp.cost.yref   = np.zeros((model.cost_y_expr.size()[0],))
    ocp.cost.yref_e = np.zeros((model.cost_y_expr_e.size()[0],))
    ocp.parameter_values = np.zeros((model.p.size()[0],))
    ocp.constraints.x0 = np.zeros(model.x.size()[0])

    idx_rho_cmd     = np.array([0, 2, 4, 6], dtype=np.int64)
    idx_alpha_cmd   = np.array([1, 3, 5, 7], dtype=np.int64)
    ocp.constraints.idxbu = np.concatenate([idx_rho_cmd, idx_alpha_cmd])

    ocp.constraints.lbu = np.concatenate([np.full(4, p.RHO_MIN,     dtype=np.float64),
                                          np.array(p.ALPHA_MIN,     dtype=np.float64)])

    ocp.constraints.ubu = np.concatenate([np.full(4, p.RHO_MAX,     dtype=np.float64),
                                          np.array(p.ALPHA_MAX,     dtype=np.float64)])
    ocp.dims.nbu = ocp.constraints.idxbu.size

    # ---------- h_expr constraints ----------
    col_cmd_lb   = np.full(4, (2.0 * p.R_ROTOR)**2, dtype=np.float64)
    
    r_dot_lb = np.array([
        c.RHO_DOT_MIN[0],   c.ALPHA_DOT_MIN[0],
        c.RHO_DOT_MIN[1],   c.ALPHA_DOT_MIN[1],
        c.RHO_DOT_MIN[2],   c.ALPHA_DOT_MIN[2],
        c.RHO_DOT_MIN[3],   c.ALPHA_DOT_MIN[3],
    ], dtype=np.float64)

    r_dot_ub = np.array([
        c.RHO_DOT_MAX[0],   c.ALPHA_DOT_MAX[0],
        c.RHO_DOT_MAX[1],   c.ALPHA_DOT_MAX[1],
        c.RHO_DOT_MAX[2],   c.ALPHA_DOT_MAX[2],
        c.RHO_DOT_MAX[3],   c.ALPHA_DOT_MAX[3],
    ], dtype=np.float64)

    ocp.constraints.lh = np.concatenate([c.F_MIN, col_cmd_lb, r_dot_lb]).astype(np.float64)
    ocp.constraints.uh = np.concatenate([c.F_MAX, np.full(4, 1e12), r_dot_ub]).astype(np.float64)
    ocp.dims.nh = ocp.constraints.lh.size

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