import numpy as np

# MPC horizon
N  = 120     # number of steps
DT = 1.0 / 400  # [s] (of each step)

# ---------- model parameters ----------
# CoT actuator time constant
TAU = 0.2

# Inertia tensor
J_TENSOR = np.array([
    [0.27, 0.00, 0.00],
    [0.00, 0.27, 0.00],
    [0.00, 0.00, 0.54]
], dtype=np.float64)

# GAC controller gain
KR = np.array([40., 40., 5.5])
KW = np.array([ 5.,  5., 2.5])

# control allocation
ZETA = 0.02

# Center of mass offset (x,y)
COM_OFFSET = np.array([0.0271, 0.0])

# ---------- Constraints & Costs ----------
# state constraint
COT_MIN = -0.08 * np.array([1.0, 1.0]) # CoT box bound (x,y)
COT_MAX =  0.08 * np.array([1.0, 1.0]) # [m]

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 21.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 10000. * np.array([1.0, 1.0, 1.0])

# state cost
Q_OMEGA = 10.0  * np.array([1.0, 1.0, 1.0])

# thurst deviation cost
Q_FDEV  = 10.0 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
R_THETA = 0.001  * np.array([1.0, 1.0, 1.0])
R_COT   = 1.0   * np.array([1.0, 1.0])