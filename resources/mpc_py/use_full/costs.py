import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 18.9 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input constraint
DTHETA_MIN     = -45.0 * np.array([1.0, 1.0, 0.25]) * np.pi / 180.0
DTHETA_MAX     =  45.0 * np.array([1.0, 1.0, 0.25]) * np.pi / 180.0

RHO_DOT_MIN    = -0.25 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.25 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -1.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  1.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# input cost
Q_THETA = 2500.0 * np.array([1.0, 1.0, 10.0]) * 100.0

# thurst deviation cost
Q_FDEV  = 12.5 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
c_a = 1.0  # arm cost (m/s scale)
c_b = 0.01 # base cost (rad/s scale)
R_ROTOR = 1.5 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b]) * 300.0