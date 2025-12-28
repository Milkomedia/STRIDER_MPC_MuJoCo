from acados_template import AcadosModel, AcadosOcp
import casadi as ca
import numpy as np

def build_model():
    from .params import MOTOR_LAMBDA, J_TENSOR, MASS, G_ACCEL, LINK_MASS, LINK_COM_DIST, DH_PARAMS_ARM, DH_PARAMS_BASE

    model = AcadosModel()
    model.name = "strider"

    # Model state
    p_cot = ca.SX.sym('p_cot',  3) # cot pos
    v_cot = ca.SX.sym('v_cot',  3) # cot vel
    theta = ca.SX.sym('theta',  3) # global attitude
    omega = ca.SX.sym('omega',  3) # cot angular vel
    q     = ca.SX.sym('q',     20) # arm pos
    x     = ca.vertcat(p_cot, v_cot, theta, omega, q)
    x_dot = ca.SX.sym('x_dot', x.size1())
    model.x = x
    model.xdot = x_dot

    # Model Control input
    f_t   = ca.SX.sym('f_t')     # overall thrust
    q_d   = ca.SX.sym('q_d', 20) # arm desired pos
    u     = ca.vertcat(f_t, q_d)
    model.u = u

    # Model parameter
    delta    = ca.SX.sym('delta', 3)      # cot -> com [m] (estimated - nominal value)
    cot_R_b  = ca.SX.sym('cot_R_b', 3, 3) # body tilt Rotation matrix command
    th_tilt  = ca.SX.sym('th_tilt')       # tilt angle [rad](sequential control allocation)
    l        = ca.SX.sym('l')             # distance between thruster [m]
    model.p  = ca.vertcat(delta, ca.reshape(cot_R_b, 9, 1), th_tilt, l)

    # DH parameter & link mass & link com position
    dh_params_arm  = ca.DM(DH_PARAMS_ARM)
    dh_params_base = ca.DM(DH_PARAMS_BASE)
    m_link         = ca.DM(LINK_MASS)      # [m1, m2, m3, m4, m5]
    d_link_com     = ca.DM(LINK_COM_DIST)  # [d1, d2, d3, d4, d5]

    # Constants
    J = ca.DM(J_TENSOR)              # inertia tensor read in {cot}
    Lambda_vec = ca.DM(MOTOR_LAMBDA) # motor constants
    Jinv = ca.inv(J)
    InvLambda = ca.diag(1.0 / Lambda_vec)

    m = MASS                 # total mass
    g = G_ACCEL              # gravitational acc
    e3 = ca.SX([0., 0., 1.]) # unit z vector

    def dh_transform(a_link, alpha_joint, d_link, theta_joint):
        cos_t, sin_t = ca.cos(theta_joint), ca.sin(theta_joint)
        cos_a, sin_a = ca.cos(alpha_joint), ca.sin(alpha_joint)

        T = ca.SX(4, 4)
        T[0, 0] = cos_t
        T[0, 1] = -sin_t * cos_a
        T[0, 2] = sin_t * sin_a
        T[0, 3] = a_link * cos_t
        T[1, 0] = sin_t
        T[1, 1] = cos_t * cos_a
        T[1, 2] = -cos_t * sin_a
        T[1, 3] = a_link * sin_t
        T[2, 0] = 0.0
        T[2, 1] = sin_a
        T[2, 2] = cos_a
        T[2, 3] = d_link
        T[3, 0] = 0.0
        T[3, 1] = 0.0
        T[3, 2] = 0.0
        T[3, 3] = 1.0
        return T

    def FK(q_arm, arm_idx):
        # Calculates transform matrix of thrust point wrt. {b}
        a0     = dh_params_base[arm_idx, 0]
        theta0 = dh_params_base[arm_idx, 1]
        T = dh_transform(a0, 0.0, 0.0, theta0)  # {b}->{0}

        T_list = []
        for i in range(dh_params_arm.shape[0]): # {b}->{5}
            a = dh_params_arm[i, 0]
            alpha = dh_params_arm[i, 1]
            T = T @ dh_transform(a, alpha, 0.0, q_arm[i])
            T_list.append(T)

        return T_list # [{b}->{1}, {b}->{2}, {b}->{3}, {b}->{4}, {b}->{5}]

    # ---------- Rotation matrix ----------
    phi, th, psi = theta[0], theta[1], theta[2]
    cphi, sphi   = ca.cos(phi),  ca.sin(phi)
    cth,  sth    = ca.cos(th),   ca.sin(th)
    cpsi, spsi   = ca.cos(psi),  ca.sin(psi)

    Rz = ca.vertcat(ca.horzcat(cpsi, -spsi, 0.0), ca.horzcat(spsi,  cpsi, 0.0),  ca.horzcat(0.0,   0.0,  1.0),)
    Ry = ca.vertcat(ca.horzcat(cth,  0.0,  sth),  ca.horzcat(0.0,  1.0,  0.0),   ca.horzcat(-sth, 0.0,  cth),)
    Rx = ca.vertcat(ca.horzcat(1.0,  0.0,   0.0), ca.horzcat(0.0,  cphi, -sphi), ca.horzcat(0.0,  sphi,  cphi),)
    R = Rz @ Ry @ Rx  # (cot->global)

    # ---------- Euler-rate mapping ----------
    L = ca.SX.zeros(3,3)
    L[0,0] = 1.
    L[0,1] = sphi*ca.tan(th)
    L[0,2] = cphi*ca.tan(th)
    L[1,0] = 0.
    L[1,1] = cphi
    L[1,2] = -sphi
    L[2,0] = 0.
    L[2,1] = sphi/cth
    L[2,2] = cphi/cth
    theta_dot = L @ omega # (cot->global)

    # ---------- motor 1st-order dynamics ----------
    q_dot = InvLambda @ (q_d - q)

    # ---------- get b_p_cot via FK ----------
    BT1 = FK(q[0:5],   0)
    BT2 = FK(q[5:10],  1)
    BT3 = FK(q[10:15], 2)
    BT4 = FK(q[15:20], 3)
    b_p_15, b_e1_15 = BT1[-1][0:3, 3], BT1[-1][0:3, 0]
    b_p_25, b_e1_25 = BT2[-1][0:3, 3], BT2[-1][0:3, 0]
    b_p_35, b_e1_35 = BT3[-1][0:3, 3], BT3[-1][0:3, 0]
    b_p_45, b_e1_45 = BT4[-1][0:3, 3], BT4[-1][0:3, 0]
    b_p_cot = (b_p_15 + b_p_25 + b_p_35 + b_p_45) / 4.0

    # ---------- get cot_p_c ----------
    bpc_acc = ca.SX.zeros(3)
    for j in range(5):
        bpc_acc += m_link[j] * (BT1[j][0:3, 3] - d_link_com[j] * BT1[j][0:3, 0])
        bpc_acc += m_link[j] * (BT2[j][0:3, 3] - d_link_com[j] * BT2[j][0:3, 0])
        bpc_acc += m_link[j] * (BT3[j][0:3, 3] - d_link_com[j] * BT3[j][0:3, 0])
        bpc_acc += m_link[j] * (BT4[j][0:3, 3] - d_link_com[j] * BT4[j][0:3, 0])

    bpc_nom = bpc_acc / m
    cot_p_c = delta + cot_R_b @ (bpc_nom - b_p_cot)

    # ---------- cot rotational dynamics ----------
    f_g       = -m * g * e3
    wrench    = ca.cross(cot_p_c, R.T @ f_g)
    Jomega    = J @ omega
    cori      = ca.cross(omega, Jomega)
    omega_dot = Jinv @ (wrench - cori)

    # ---------- cot position dynamics ----------
    # a_cot   = (f_t / m) * R @ e3 - g * e3
    a_cot   = (f_t / m) * R @ e3 - g * e3 - (ca.cross(omega_dot, cot_p_c) + ca.cross(omega, ca.cross(omega, cot_p_c)))

    f_expl = ca.vertcat(v_cot, a_cot, theta_dot, omega_dot, q_dot)
    model.f_expl_expr = f_expl
    model.f_impl_expr = x_dot - f_expl

    # ---------- constraint expression ----------
    sin_t, cos_t = ca.sin(th_tilt), ca.cos(th_tilt)
    sqrt2 = ca.sqrt(2.0)
    heading_err1 = ca.acos(ca.dot(cot_R_b @ b_e1_15, ca.vertcat( sin_t/sqrt2, -sin_t/sqrt2,  cos_t)))
    heading_err2 = ca.acos(ca.dot(cot_R_b @ b_e1_25, ca.vertcat( sin_t/sqrt2,  sin_t/sqrt2,  cos_t)))
    heading_err3 = ca.acos(ca.dot(cot_R_b @ b_e1_35, ca.vertcat(-sin_t/sqrt2,  sin_t/sqrt2,  cos_t)))
    heading_err4 = ca.acos(ca.dot(cot_R_b @ b_e1_45, ca.vertcat(-sin_t/sqrt2, -sin_t/sqrt2,  cos_t)))
    heading_errs = ca.vertcat(ca.norm_2(heading_err1), ca.norm_2(heading_err2), ca.norm_2(heading_err3), ca.norm_2(heading_err4))

    b_R_cot = cot_R_b.T
    pos_err1 = (b_p_15 - b_p_cot) - 0.5*l*( b_R_cot[:, 0] + b_R_cot[:, 1])
    pos_err2 = (b_p_25 - b_p_cot) - 0.5*l*(-b_R_cot[:, 0] + b_R_cot[:, 1])
    pos_err3 = (b_p_35 - b_p_cot) - 0.5*l*(-b_R_cot[:, 0] - b_R_cot[:, 1])
    pos_err4 = (b_p_45 - b_p_cot) - 0.5*l*( b_R_cot[:, 0] - b_R_cot[:, 1])
    pos_errs = ca.vertcat(ca.norm_2(pos_err1), ca.norm_2(pos_err2), ca.norm_2(pos_err3), ca.norm_2(pos_err4))

    model.con_h_expr = ca.vertcat(pos_errs, heading_errs)

    return model

def build_ocp():
    from .params import N, DT, ARM_MIN, ARM_MAX, F_MIN, F_MAX, COST_POS_ERR, COST_ANG_ERR, COST_F_THRUST

    model = build_model()

    ocp = AcadosOcp()
    ocp.model = model

    # ---------- horizon ----------
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = N * DT

    # ---------- costs ----------

    # states & inputs
    p_cot = model.x[0:3] # position
    yaw   = model.x[8]   # yaw angle
    f_t   = model.u[0]   # overall thrust
    
    model.cost_y_expr   = ca.vertcat(p_cot, yaw, f_t) # 1~k-1 ref = [p_d, theta_d, f_d]
    model.cost_y_expr_e = ca.vertcat(p_cot, yaw)      # terminal ref = [p_d, theta_d]
    
    ocp.cost.W   = np.diag(np.concatenate([COST_POS_ERR, COST_ANG_ERR, COST_F_THRUST,]))
    ocp.cost.W_e = np.diag(np.concatenate([COST_POS_ERR, COST_ANG_ERR,]))

    # reference default value
    ocp.cost.yref   = np.zeros((model.cost_y_expr.size()[0],))
    ocp.cost.yref_e = np.zeros((model.cost_y_expr_e.size()[0],))
    ocp.parameter_values = np.zeros((model.p.size()[0],))
    ocp.constraints.x0 = np.zeros(model.x.size()[0])

    ocp.cost.cost_type   = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # ---------- constraints ----------
    ocp.constraints.lbu   = np.concatenate([[F_MIN], np.tile(ARM_MIN, 4)])
    ocp.constraints.ubu   = np.concatenate([[F_MAX], np.tile(ARM_MAX, 4)])
    ocp.constraints.idxbu = np.arange(model.u.size()[0])   # u[0]~u[20] all

    # ---------- orientation&position constraint ----------
    nh = model.con_h_expr.size()[0]
    eps_pos  = [2.0 * 1e-3]*4          # tolerance in [m]
    eps_head = [4.0 * np.pi/180.0]*4   # tolerance in [rad]
    ocp.constraints.lh = np.zeros((nh,))
    ocp.constraints.uh = np.array(eps_pos + eps_head, dtype=float)

    # ---------- solver options ----------
    ocp.solver_options.qp_solver        = "PARTIAL_CONDENSING_HPIPM" # or "FULL_CONDENSING_HPIPM" "PARTIAL_CONDENSING_HPIPM" "FULL_CONDENSING_QPOASES"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI"
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1

    # codegen dir
    ocp.code_export_directory = "generated"
    return ocp