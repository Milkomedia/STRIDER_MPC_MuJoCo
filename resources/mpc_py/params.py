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
# real model uses thrust-based yaw (sequential allocation). but mpc model uses reaction-based yaw.
# To compensate this, mpc's allocation thinks that less thrust deviation produces more reaction torque.
ZETA = 0.04

# Center of mass offset (x,y)
COM_OFFSET = np.array([0.0, 0.0])

# The rate at which CoM moves as CoT moves. (x,y)
COT2COM_COEFF = np.array([0.3569, 0.3569])