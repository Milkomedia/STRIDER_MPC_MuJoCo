import numpy as np

# MPC horizon
N  = 50     # number of steps
DT = 0.01  # [s] (of each step)

# ---------- model parameters ----------
# CoT actuator time constant
TAU = 2.0

# Inertia tensor @body frame
J_TENSOR = np.array([
    [ 0.386,   -0.0006, -0.0006],
    [-0.0006,    0.386,  0.0006],
    [-0.0006,   0.0006,  0.5318]
], dtype=np.float64)

# Mass
MASS    = 5.09495 # [kg]
G_ACCEL = 9.80665 # [m/s^2]

# GAC controller gain
KR = 8.0
KW = 5.0
KP_Z = 15.0
KV_Z = 15.0

# control allocation
ZETA = 0.02

# ---------- Constraints & Costs ----------
# input constraint
COT_MIN = -0.05 * np.array([1.0, 1.0]) # CoT box bound (x,y)
COT_MAX =  0.05 * np.array([1.0, 1.0]) # [m]

# h_expr constraint
F_MIN   = 6.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 13.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

# input cost
Q_THETA = 100. * np.array([1.0, 1.0, 1.0])
Q_COT   = 0.1  * np.array([1.0, 1.0])

# state cost
Q_OMEGA = 1.  * np.array([1.0, 1.0, 1.0])

# rate cost
R_THETA = 1.   * np.array([1.0, 1.0, 1.0])
R_COT   = 0.1   * np.array([1.0, 1.0])