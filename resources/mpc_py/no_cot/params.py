import numpy as np

# MPC horizon
N  = 60     # number of steps
DT = 1.0 / 400  # [s] (of each step)

# ---------- model parameters ----------
# Inertia tensor
J_TENSOR = np.array([
    [0.27, 0.00, 0.00],
    [0.00, 0.27, 0.00],
    [0.00, 0.00, 0.54]
], dtype=np.float64)

# GAC controller gain
KR = np.array([50., 50., 5.5])
KW = np.array([ 5.,  5., 2.5])

# control allocation
ZETA = 0.02

# Center of mass offset (x,y)
COM_OFFSET = np.array([0.0271, 0.0])

# ---------- Constraints & Costs ----------
# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 22.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 50. * np.array([1.0, 1.0, 1.0])

# state cost
Q_OMEGA = 10.0  * np.array([1.0, 1.0, 1.0])

# rate cost
R_THETA = 0.01  * np.array([1.0, 1.0, 1.0])