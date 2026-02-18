import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 20.75 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 100. * np.array([1.0, 1.0, 1.0])

# state cost
Q_OMEGA = 10.0  * np.array([1.0, 1.0, 1.0])

# rate cost
R_THETA = 0.001  * np.array([1.0, 1.0, 1.0])