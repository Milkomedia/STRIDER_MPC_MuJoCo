import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 19.8 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

RHO_DOT_MIN    = -0.25 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.25 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -1.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  1.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# thurst deviation cost
Q_FDEV  = 1.25 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
c_a = 1.0  # arm cost (m/s scale)
c_b = 0.01 # base cost (rad/s scale)
R_ROTOR = 1.0 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b]) * 300.0