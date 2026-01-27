import numpy as np

# MPC horizon
N  = 60     # number of steps
DT = 0.005  # [s] (of each step)

# ---------- model parameters ----------
# CoT actuator time constant
TAU = 0.1

# Inertia tensor
J_TENSOR = 0.5 * np.array([
    [0.27, 0.00, 0.00],
    [0.00, 0.27, 0.00],
    [0.00, 0.00, 0.53]
], dtype=np.float64)

# GAC controller gain
KR = 13.0 * 1.5
KW = 2.5 * 1.5

# control allocation
ZETA = 0.02 * 0.5

# ---------- Constraints & Costs ----------
# state constraint
COT_MIN = -0.05 * np.array([1.0, 1.0]) # CoT box bound (x,y)
COT_MAX =  0.05 * np.array([1.0, 1.0]) # [m]

# h_expr constraint
F_MIN   = 6.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 12.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 500. * np.array([1.0, 1.0, 1.0])
Q_COT   = 0.001 * np.array([1.0, 1.0])

# state cost
Q_OMEGA = 10.0  * np.array([1.0, 1.0, 1.0])

# rate cost
R_THETA = 2.0   * np.array([1.0, 1.0, 1.0])
R_COT   = 0.01   * np.array([1.0, 1.0])