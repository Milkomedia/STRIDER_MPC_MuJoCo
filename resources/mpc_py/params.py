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

# ---------- use_arm & use_full parameters ----------
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

RHO_MIN_SLK = 0.1506 - 0.015
RHO_MAX_SLK = 0.2925 + 0.015
R_ROTOR_SLK = 0.22 - 0.03

# IK & CoM estimate
A_LINK   = np.array([0.1395, 0.115, 0.110, 0.024, 0.070])             # link length [m]
R_Z = 0.24 - A_LINK[4] - A_LINK[3]                                    # z-distance body<->3-th joint frame [m]
M_LINK   = np.array([0.374106, 0.13658, 0.0415148, 0.102003, 0.3734]) # each link mass [kg]
M_CENTER = 2.6845345                                                  # center body + load mass [kg]
D_LINK   = np.array([-0.040, -0.031, -0.055, -0.012, -0.020])         # link com distance [m]

# ---------- use_delta parameters ----------
# default l
L_DIST = 0.5 * 0.48
