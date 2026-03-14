import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 18.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input constraint
DTHETA_DOT_MIN = -1.0 * np.array([1.0, 1.0, 0.25]) # [rad/s]
DTHETA_DOT_MAX =  1.0 * np.array([1.0, 1.0, 0.25]) # [rad/s]

RHO_DOT_MIN    = -0.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -1.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  1.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# input cost
Q_THETA = 0.5 * np.array([1.0, 1.0, 10.0])

# thurst deviation cost
Q_FDEV  = 0.1 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
R_THETA = 1.0  * np.array([1.0, 1.0, 5.0])
c_a = 1.0  # arm cost (m/s scale)
c_b = 0.025 # base cost (rad/s scale)
R_ROTOR = 0.01 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])