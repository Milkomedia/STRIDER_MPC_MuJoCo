import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 17.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input constraint
DTHETA_DOT_MIN = -10.0 * np.array([1.0, 1.0, 0.25]) # [rad/s]
DTHETA_DOT_MAX =  10.0 * np.array([1.0, 1.0, 0.25]) # [rad/s]

# input cost
Q_THETA = 0.1 * np.array([1.0, 1.0, 1.0])

# rate cost
R_THETA = 2.0 * np.array([1.0, 1.0, 1.0])