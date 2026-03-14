import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 18.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input constraint
RHO_DOT_MIN    = -2.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  2.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -5.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  5.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# thurst deviation cost
Q_FDEV  = 1.0 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
c_a = 1.0  # arm cost (m/s scale)
c_b = 0.035 # base cost (rad/s scale)
R_ROTOR = 0.85 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])