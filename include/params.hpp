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
static constexpr double PWM_B    = 8.0;   // propeller thrust[N] = A * pwm^2 + B
static constexpr double PWM_ZETA = 0.02;  // propeller torque[Nm] = zeta * thrust
static constexpr double rotor_dir[4] = {1.0, -1.0, 1.0, -1.0}; // propeller torque direction

// ===== SE3 controlelr gains ====
// Control Parameters
static constexpr double kX[3] = { 8.0,  8.0,  5.0}; // Position gain [x, y, z]
static constexpr double kV[3] = {12.0, 12.0, 10.0}; // Velocity gain [x, y, z]
static constexpr double kR[3] = {13.0, 13.0,  0.5}; // Rotational gain [roll, pitch, yaw]
static constexpr double kW[3] = { 2.5,  2.5,  0.5}; // angular Velocity gain [roll, pitch, yaw]

// Integral Parameters
static constexpr double kI  = 0.0;  /**< Attitude integral gain for roll and pitch */
static constexpr double kyI = 0.0;  /**< Attitude integral gain for yaw */
static constexpr double kIX = 0.1;  /**< Position integral gains */

// ===== UAV Parameters =====
static constexpr double J[9] = {0.27, 0.00, 0.00,
                                0.00, 0.27, 0.00,
                                0.00, 0.00, 0.53};
static constexpr double M  = 4.8;
static constexpr double G  = 9.80665;

// Allocation parameters
static constexpr double SERVO_DELAY_ALPHA = 0.01;  // yaw trimming
static constexpr double SERVO_DELAY_BETA  = 1.0 - SERVO_DELAY_ALPHA;
static constexpr double TAUZ_MIN = -5.0; // saturation ref [Nm]
static constexpr double TAUZ_MAX =  5.0;

// ===== DH parameters =====
static constexpr double B2BASE_THETA[4] = {-0.25*M_PI, -0.75*M_PI, 0.75*M_PI, 0.25*M_PI};
static constexpr double B2BASE_ALPHA[4] = {M_PI, M_PI, M_PI, M_PI};
static constexpr double B2BASE_A[4]     = {0.120, 0.120, 0.120, 0.120};
static constexpr double DH_ARM_A[5]     = {0.1395, 0.115, 0.110, 0.024, 0.068};
static constexpr double DH_ARM_ALPHA[5] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, 0.0};

// desired inter-rotor distance
static constexpr double L_DIST = 0.55; // [m]

// ===== MPC parameters  =====
static constexpr double MPC_COMPUTE_HZ = 200.0; // [Hz]
static constexpr double MPC_COMPUTE_DT_DOUBLE = 1.0 / MPC_COMPUTE_HZ; // [sec]
static const std::chrono::steady_clock::duration MPC_COMPUTE_DT = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / MPC_COMPUTE_HZ));

constexpr std::size_t N_STEPS  = 60; // (Must be the same as prams.py)
constexpr std::size_t NX       = 13;
constexpr std::size_t NU_AUG   = 5;
constexpr std::size_t NU       = 5;
constexpr std::size_t NP       = 11;

// ===== MuJoCo viewer parameters =====
static constexpr double PATH_SEC = 10.0;   // history length [sec]
static constexpr float SIZE_DOT  = 0.03f; // size(radious) of dot
static constexpr float SIZE_PATH = 0.005f; // size(radious) of path

static constexpr float RGBA_DOT[4]   = {1.00f, 0.00f, 0.00f, 0.95f}; // current pos color
static constexpr float RGBA_PATH[4]  = {0.20f, 0.80f, 0.90f, 0.60f}; // current path color
static constexpr float RGBA_DPATH[4] = {0.60f, 0.60f, 0.60f, 0.60f}; // desired path color

// ===== MPC input noise (sim->real validation) =====
static constexpr bool MPC_NOISE_ON = false;
static constexpr std::uint64_t MPC_NOISE_SEED = 42;

// 1. rpy noise/bias
static constexpr double MPC_RPY_SIGMA_RP    = 0.50 * M_PI / 180.0; // white noise std (roll/pitch), [rad]
static constexpr double MPC_RPY_SIGMA_YAW   = 0.80 * M_PI / 180.0; // white noise std (yaw), [rad]
static constexpr double MPC_RPY_BIAS_RW_RP  = 0.06 * M_PI / 180.0; // bias random-walk, [rad/sqrt(s)]
static constexpr double MPC_RPY_BIAS_RW_YAW = 0.10 * M_PI / 180.0; // [rad/sqrt(s)]

// 2. omega noise/bias
static constexpr double MPC_OMEGA_SIGMA   = 0.012;  // white noise std, [rad/s]
static constexpr double MPC_OMEGA_BIAS_RW = 0.005; // bias random-walk, [rad/s/sqrt(s)]

// 3. r_cot noise/bias
static constexpr double MPC_RCOT_SIGMA   = 0.0010;  // white noise std, [m]
static constexpr double MPC_RCOT_BIAS_RW = 0.0003; // bias random-walk, [m/sqrt(s)]

// 4. R_raw perturbation noise/bias
static constexpr double MPC_RRAW_SIGMA_RP    = 1.00 * M_PI / 180.0;  // white noise std (roll/pitch), [rad]
static constexpr double MPC_RRAW_SIGMA_YAW   = 1.50 * M_PI / 180.0;  // white noise std (yaw), [rad]
static constexpr double MPC_RRAW_BIAS_RW_RP  = 0.05 * M_PI / 180.0;  // [rad/sqrt(s)]
static constexpr double MPC_RRAW_BIAS_RW_YAW = 0.10 * M_PI / 180.0;  // [rad/sqrt(s)]

// 5. T_des bias/noise
static constexpr double MPC_T_SCALE_RW  = 0.01;  // scale random-walk (1.0% per sqrt(s)) [1/sqrt(s)]
static constexpr double MPC_T_ADD_RW    = 0.10;  // additive random-walk [N/sqrt(s)]
static constexpr double MPC_T_NOISE_REL = 0.01;  // [-] relative white noise (1.0%)
static constexpr double MPC_T_NOISE_ABS = 0.50;  // [N] minimum abs noise

} // namespace param

#endif // PARAMS_H