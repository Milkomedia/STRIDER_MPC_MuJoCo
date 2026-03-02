import numpy as np

# MPC horizon
N  = 75     # number of steps
DT = 1.0 / 200  # [s] (of each step)

# ---------- model parameters ----------
# CoT actuator time constant
TAU = 0.17

# Inertia tensor
J_TENSOR = np.array([
    [0.27, 0.00, 0.00],
    [0.00, 0.77, 0.00],
    [0.00, 0.00, 1.04]
], dtype=np.float64)

# GAC controller gain
KR = np.array([45., 45., 5.5])
KW = np.array([20., 20., 2.5])

# control allocation
# real model uses thrust-based yaw (sequential allocation). but mpc model uses reaction-based yaw.
# To compensate this, mpc's allocation thinks that less thrust deviation produces more reaction torque.
ZETA = 0.04

# ---------- yes-cot parameters ----------
R_OFF_X = np.array([ 0.12, -0.12, -0.12,  0.12])/np.sqrt(2)
R_OFF_Y = np.array([-0.12, -0.12,  0.12,  0.12])/np.sqrt(2)

R_MIN = 0.1506
R_MAX = 0.2925
R_MIN_SLK = 0.1506 - 0.03
R_MAX_SLK = 0.2925 + 0.03
R_ROTOR = 0.22
R_ROTOR_SLK = 0.22 - 0.04

# ---------- no-cot parameters ----------
# default l
L_DIST = 0.5 * 0.48
