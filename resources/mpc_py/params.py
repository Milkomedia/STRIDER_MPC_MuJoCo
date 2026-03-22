import numpy as np

# MPC horizon
N  = 150     # number of steps
DT = 1.0 / 100  # [s] (of each step)

# ---------- model parameters ----------
J_TENSOR = np.array([
    [0.27, 0.00, 0.00],
    [0.00, 0.41,0.00],
    [0.00, 0.00, 0.68]
], dtype=np.float64)

# GAC controller gain
KR = np.array([85., 85., 5.5])
KW = np.array([15., 15., 2.5])

# control allocation
# real model uses thrust-based yaw (sequential allocation). but mpc model uses reaction-based yaw.
# To compensate this, mpc's allocation thinks that less thrust deviation produces more reaction torque.
ZETA = 0.1

# IK & CoM estimate
M_LINK   = np.array([0.374106, 0.13658, 0.0415148, 0.102003, 0.3734]) # each link mass [kg]
M_CENTER = 2.6845345                                                  # center body + load mass [kg]
MAX_COM_BIAS_OF_LOAD = 0.18                                           # load-link length * load wieght [kg*m]

# ---------- use_arm & use_full parameters ----------
# CoT actuator time constant
TAU_BASE = 0.25
TAU_ARM  = 0.15

R_OFF_X = np.array([ 0.12, -0.12, -0.12,  0.12])/np.sqrt(2)
R_OFF_Y = np.array([-0.12, -0.12,  0.12,  0.12])/np.sqrt(2)

RHO_MIN = 0.1506 + 0.01
RHO_MAX = 0.2925 - 0.01
ALPHA_MIN = np.array([-105.0, -195.0,  75.0, -15.0]) * np.pi / 180.0
ALPHA_MAX = np.array([  15.0,  -75.0, 195.0, 105.0]) * np.pi / 180.0
R_ROTOR = 0.22 + 0.01

# IK & CoM estimate
A_LINK   = np.array([0.1395, 0.115, 0.110, 0.024, 0.070])             # link length [m]
R_Z = 0.24 - A_LINK[4] - A_LINK[3]                                    # z-distance body<->3-th joint frame [m]
D_LINK   = np.array([-0.040, -0.031, -0.055, -0.012, -0.020])         # link com distance [m]

# ---------- use_delta parameters ----------
# default l
L_DIST = 0.5 * 0.48
