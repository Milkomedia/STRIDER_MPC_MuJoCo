import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 18. * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input constraint
RHO_DOT_MIN    = -0.2 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.2 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -0.1 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  0.1 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# thurst deviation cost
Q_FDEV  = 3. * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
c_a = 1.0  # arm cost (m/s scale)
c_b = 0.035 # base cost (rad/s scale)
R_ROTOR = 1.6 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])