import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 21.2 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 0.25 * np.array([1.0, 1.0, 1.0])

# thurst deviation cost
Q_FDEV  = 0.1 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
R_THETA = 0.07 * np.array([1.0, 1.0, 1.0])
R_ROTOR = 1. * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

"""

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 20.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 5000. * np.array([1.0, 1.0, 1.0])

# thurst deviation cost
Q_FDEV  = 0.9 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
R_THETA = 0.1  * np.array([1.0, 1.0, 1.0])
R_ROTOR = 4.0  * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

"""