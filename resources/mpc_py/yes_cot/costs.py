import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 20.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 10. * np.array([1.0, 1.0, 1.0])

# thurst deviation cost
Q_FDEV  = 1.0 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
R_THETA = 3.0  * np.array([1.0, 1.0, 1.0])
c_a = 1.0  # arm  cost (m/s scale)
c_b = 0.1 * 0.23 # base cost (rad/s scale)
R_ROTOR = 2.0 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])