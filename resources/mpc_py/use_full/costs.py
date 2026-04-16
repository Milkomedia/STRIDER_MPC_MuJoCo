import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 18.4 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input constraint
RHO_DOT_MIN    = -0.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -0.9 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  0.9 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# input cost
Q_THETA = 7.5 * np.array([1.0, 1.0, 10.0])

# thurst deviation cost
Q_FDEV  = 25.0 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
c_a = 1.0  # arm cost (m/s scale)
c_b = 0.025 # base cost (rad/s scale)
R_ROTOR = 0.75 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])