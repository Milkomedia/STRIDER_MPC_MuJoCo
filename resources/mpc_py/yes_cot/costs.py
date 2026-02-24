import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 23.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA_CMD = 5000. * np.array([1.0, 1.0, 1.0])
Q_OMEGA_CMD = 1000. * np.array([1.0, 1.0, 1.0])

# state cost
Q_OMEGA = 0.5  * np.array([1.0, 1.0, 1.0])

# thurst deviation cost
Q_FDEV  = 1.0 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
R_THETA_CMD = 0.01  * np.array([1.0, 1.0, 1.0])
R_OMEGA_CMD = 0.0001 * np.array([1.0, 1.0, 1.0])
R_ROTOR = 4.0   * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])