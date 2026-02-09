import numpy as np

# state constraint
COT_MIN = -0.047 * np.array([1.0, 1.0]) # CoT box bound (x,y)
COT_MAX =  0.047 * np.array([1.0, 1.0]) # [m]

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 27.36 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 10000. * np.array([1.0, 1.0, 1.0])

# state cost
Q_OMEGA = 10.0  * np.array([1.0, 1.0, 1.0])

# thurst deviation cost
Q_FDEV  = 20.0 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
R_THETA = 0.001  * np.array([1.0, 1.0, 1.0])
R_COT   = 8.0   * np.array([1.0, 1.0])