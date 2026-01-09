#ifndef PARAMS_H
#define PARAMS_H

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <chrono>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>

namespace param {

// ===== Frequencies & timesteps =====
static constexpr double CTRL_HZ   = 400.0;  // flight contol
static constexpr double SIM_HZ    = 1000.0; // mujoco simul
static constexpr double VIEWER_HZ = 30.0;   // mujoco viewer
static const std::chrono::steady_clock::duration VIEWER_DT = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / VIEWER_HZ));
static constexpr double CTRL_DT   = 1.0 / CTRL_HZ;
static constexpr double SIM_DT    = 1.0 / SIM_HZ;

// ===== Motor model =====
static constexpr double PWM_A    = 45.0;  // propeller thrust[N] = A * pwm^2 + B
static constexpr double PWM_B    = 8.0;
static constexpr double PWM_ZETA = 0.02;  // propeller torque[Nm] = zeta * thrust
static constexpr double rotor_dir[4] = {1.0, -1.0, 1.0, -1.0};       // propeller torque direction

// ===== SE3 controlelr gains ====
// Control Parameters
static constexpr double kX[3] = {25.0, 25.0, 15.0};  // Position gain [x, y, z]
static constexpr double kV[3] = {17.0, 17.0, 15.0};  // Velocity gain [x, y, z]
static constexpr double kR[3] = { 8.0,  8.0, 0.75};  // Rotational gain [roll, pitch, yaw]
static constexpr double kW[3] = { 5.0,  5.0,  2.0};  // angular Velocity gain [roll, pitch, yaw]

// Integral Parameters
static constexpr bool use_integral = true;
static constexpr double kIX = 3.0;
static constexpr double ki  = 0.0;
static constexpr double kIR = 0.0;
static constexpr double kI  = 0.0;
static constexpr double kyI = 0.0;
static constexpr double c1  = 1.0;
static constexpr double c2  = 0.0;
static constexpr double c3  = 0.0;

// ===== UAV Parameters =====
static constexpr double J[9] = {0.271587936842000, 0.000146304434327, 0.000394623073964,
                                0.000146304434327, 0.292342580590000, 0.001041120426640,
                                0.000394623073964, 0.001041120426640, 0.525751256781000};  // Inertia tensor @ cot frame
static constexpr double M  = 5.09495;
static constexpr double G  = 9.80665;

// Allocation parameters
static constexpr double SERVO_DELAY_ALPHA = 0.01;  // yaw trimming
static constexpr double SERVO_DELAY_BETA  = 1.0 - SERVO_DELAY_ALPHA;
static constexpr double TAUZ_MIN = -5.0; // saturation ref [Nm]
static constexpr double TAUZ_MAX =  5.0;

// CoM estimate parameters
static constexpr double COM_CUTOFF_FREQ = 0.5; // [Hz] Butterworth cutoff
static constexpr double COM_GAMMA = 0.0003;

// yaw wrench conversion

// ===== Trajectory (figure-8 on XY) =====
static constexpr double TRAJ_Z     = 1.0;               // desired altitude [m]
static constexpr double TRAJ_AX    = 2.0;               // lobe half-width in X [m]
static constexpr double TRAJ_FREQ  = 0.15;              // [Hz]

static constexpr double TRAJ_AY    = TRAJ_AX / 2.0;     // lobe half-height in Y [m]
static constexpr double FREQ_RAD_S = 2.*M_PI*TRAJ_FREQ; // [rad/s]
static constexpr double PATH_SEC   = 1.0 / TRAJ_FREQ;   // history length [sec]

// ===== DH parameters =====
static constexpr double B2BASE_THETA[4] = {0.25*M_PI, 0.75*M_PI, -0.75*M_PI, -0.25*M_PI};
static constexpr double B2BASE_A[4]     = {0.120, 0.120, 0.120, 0.120};
static constexpr double DH_ARM_A[5]     = {0.1395, 0.115, 0.110, 0.024, 0.068};
static constexpr double DH_ARM_ALPHA[5] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, 0.0};

// desired inter-rotor distance
static constexpr double L_DIST = 0.55; // [m]

// ===== MPC parameters (Must be the same as prams.py) =====
static constexpr double MPC_COMPUTE_HZ = 100.0; // [Hz]
static const std::chrono::steady_clock::duration MPC_COMPUTE_DT = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / MPC_COMPUTE_HZ));

constexpr std::size_t N_STEPS  = 30;
constexpr std::size_t NX       = 13;
constexpr std::size_t NU_AUG   = 5;
constexpr std::size_t NU       = 5;
constexpr std::size_t NP       = 11;

// ===== MuJoCo viewer parameters =====
static constexpr float SIZE_DOT  = 0.003f; // size(radious) of dot
static constexpr float SIZE_PATH = 0.005f; // size(radious) of path

static constexpr float RGBA_DOT[4]   = {1.00f, 0.00f, 0.00f, 0.95f}; // current pos color
static constexpr float RGBA_PATH[4]  = {0.20f, 0.80f, 0.90f, 0.60f}; // current path color
static constexpr float RGBA_DPATH[4] = {0.60f, 0.60f, 0.60f, 0.60f}; // desired path color

} // namespace param

#endif // PARAMS_H