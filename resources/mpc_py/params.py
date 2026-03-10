import numpy as np

# MPC horizon
N  = 200     # number of steps
DT = 1.0 / 400  # [s] (of each step)

# ---------- model parameters ----------
# Inertia tensor
J_TENSOR = np.array([
    [0.27, 0.00, 0.00],
    [0.00, 0.62, 0.00],
    [0.00, 0.00, 0.89]
], dtype=np.float64)

# GAC controller gain
KR = np.array([75., 75., 5.5])
KW = np.array([15., 15., 2.5])

# control allocation
# real model uses thrust-based yaw (sequential allocation). but mpc model uses reaction-based yaw.
# To compensate this, mpc's allocation thinks that less thrust deviation produces more reaction torque.
ZETA = 0.1

# ---------- yes-cot parameters ----------
# CoT actuator time constant
TAU_BASE = 0.25
TAU_ARM  = 0.15

M_ARM  = np.array([0.8205656, 0.8205656, 0.8205656, 0.8205656])
M_BODY = 3.5177376

R_OFF_X = np.array([ 0.12, -0.12, -0.12,  0.12])/np.sqrt(2)
R_OFF_Y = np.array([-0.12, -0.12,  0.12,  0.12])/np.sqrt(2)

RHO_MIN = 0.1506
RHO_MAX = 0.2925
ALPHA_MIN = np.array([-105.0, -195.0,  75.0, -15.0]) * np.pi / 180.0
ALPHA_MAX = np.array([  15.0,  -75.0, 195.0, 105.0]) * np.pi / 180.0
R_ROTOR = 0.22

RHO_MIN_SLK = 0.1506 - 0.05
RHO_MAX_SLK = 0.2925 + 0.05
R_ROTOR_SLK = 0.22 - 0.07

# ---------- no-cot parameters ----------
# default l
L_DIST = 0.5 * 0.48
