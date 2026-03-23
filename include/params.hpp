#ifndef PARAMS_H
#define PARAMS_H

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <chrono>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>

static inline constexpr double inv_sqrt2 = 0.7071067811865474617150084668537601828575;  // 1/sqrt(2)

namespace param {

// ===== Frequencies & timesteps =====
inline constexpr double CTRL_HZ   = 400.0;  // flight contol
inline constexpr double SIM_HZ    = 1000.0; // mujoco simul
inline constexpr double VIEWER_HZ = 30.0;   // mujoco viewer
inline const std::chrono::steady_clock::duration VIEWER_DT = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / VIEWER_HZ));
inline constexpr double CTRL_DT   = 1.0 / CTRL_HZ;
inline constexpr double SIM_DT    = 1.0 / SIM_HZ;

// ===== Motor model =====
inline constexpr double PWM_A    = 45.0;  // propeller thrust[N] = A * pwm^2 + B
inline constexpr double PWM_B    = 8.0;   // propeller thrust[N] = A * pwm^2 + B
inline constexpr double PWM_ZETA = 0.02;  // propeller torque[Nm] = zeta * thrust
inline constexpr double rotor_dir[4] = {1.0, -1.0, 1.0, -1.0}; // propeller torque direction

// ===== SE3 controlelr gains ====
// Control Parameters
inline constexpr double kX[3] = {50.0, 50.0, 65.0}; // Position gain [x, y, z]
inline constexpr double kV[3] = {27.5, 27.5, 60.0}; // Velocity gain [x, y, z]
inline constexpr double kR[3] = {110.0, 110.0, 5.5}; // Rotational gain [roll, pitch, yaw]
inline constexpr double kW[3] = {12.0, 12.0,  2.5}; // angular Velocity gain [roll, pitch, yaw]

// Integral Parameters
inline constexpr double kI  = 0.0;  /**< Attitude integral gain for roll and pitch */
inline constexpr double kyI = 0.0;  /**< Attitude integral gain for yaw */
inline constexpr double kIX = 0.3;  /**< Position integral gains */

// ===== UAV Parameters =====
inline constexpr double J[9] = {0.27, 0.00, 0.00,
                                0.00, 0.49, 0.00,
                                0.00, 0.00, 0.76};
inline constexpr double M  = 6.8;
inline constexpr double G  = 9.80665;

inline constexpr double VIRTUAL_MARGIN    = 2.0; // thrust margin of each thruster [N]
inline constexpr double SATURATION_THRUST = M * G / 4.0 + VIRTUAL_MARGIN;

// Allocation parameters
inline constexpr double SERVO_DELAY_ALPHA = 0.093158;  // yaw trimming
inline constexpr double SERVO_DELAY_BETA  = 1.0 - SERVO_DELAY_ALPHA;
inline constexpr double TAUZ_MIN = -3.0; // saturation ref [Nm]
inline constexpr double TAUZ_MAX =  3.0;

// ===== DH parameters =====
inline constexpr double B2BASE_THETA[4] = {-0.25*M_PI, -0.75*M_PI, 0.75*M_PI, 0.25*M_PI};
inline constexpr double B2BASE_ALPHA[4] = {M_PI, M_PI, M_PI, M_PI};
inline constexpr double B2BASE_A[4]     = {0.120, 0.120, 0.120, 0.120};
inline constexpr double DH_ARM_A[5]     = {0.1395, 0.115, 0.110, 0.024, 0.068};
inline constexpr double DH_ARM_ALPHA[5] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, 0.0};

// ===== Workspace constraint =====
inline constexpr double MAX_STRETCH       = 0.2925; // Maximum distance arm can extend from the base [m]
inline constexpr double MIN_STRETCH       = 0.1506; // Minimum distance arm can extend from the base [m]
inline constexpr double ROTOR_DIAMETER    = 0.44;   // propeller diameter [m]

inline constexpr double STRETCH_FAIL_MARGIN    = 0.2; // [m]
inline constexpr double COLLISION_FAIL_MARGIN  = 0.2; // [m]
inline constexpr double GUARD_MOVE_MARGIN      = 0.2; // [m]
inline constexpr double B2BASE_X[4]            = { 0.12*inv_sqrt2, -0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2}; // x-distance from the body frame to each base frame [m]
inline constexpr double B2BASE_Y[4]            = {-0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2,  0.12*inv_sqrt2}; // y-distance from the body frame to each base frame [m]
inline constexpr double ALPHA_MIN[4] = {-105.0 * M_PI/180.0, -195.0 * M_PI/180.0,  75.0 * M_PI/180.0, -15.0 * M_PI/180.0};
inline constexpr double ALPHA_MAX[4] = {  15.0 * M_PI/180.0,  -75.0 * M_PI/180.0, 195.0 * M_PI/180.0, 105.0 * M_PI/180.0};
inline const     Eigen::Vector3d r1_init       = Eigen::Vector3d( 0.24, -0.24, -0.24); // rotor-1 inital position
inline const     Eigen::Vector3d r2_init       = Eigen::Vector3d(-0.24, -0.24, -0.24); // rotor-2 inital position
inline const     Eigen::Vector3d r3_init       = Eigen::Vector3d(-0.24,  0.24, -0.24); // rotor-3 inital position
inline const     Eigen::Vector3d r4_init       = Eigen::Vector3d( 0.24,  0.24, -0.24); // rotor-4 inital position

inline constexpr double MPC_OFF_TIME_CONSTANT = 0.8; // [sec] each arm goes to initial position when MPC-off or Solve-failed
inline const     double GOES_2_ZERO_A         = std::exp(-CTRL_DT / MPC_OFF_TIME_CONSTANT); // not a tunable parameter
inline const     double GOES_2_ZERO_B         = 1.0 - GOES_2_ZERO_A;                        // not a tunable parameter

// ===== CoM estimating parameter =====
inline constexpr double LINK_MASS[5] = {0.374106, 0.13658, 0.0415148, 0.102003, 0.3734}; // link mass [kg]
inline constexpr double CENTER_MASS  = 2.6845345;   // center body + load mass [kg]
inline constexpr double TOTAL_MASS   = CENTER_MASS + 4.0*(LINK_MASS[0]+LINK_MASS[1]+LINK_MASS[2]+LINK_MASS[3]+LINK_MASS[4]); // strider mass (same as M) [kg]
inline constexpr double BIAS_WEIGHT_MAX_COM = 0.1875; // load-link length * load wieght [kg*m]
inline constexpr double LINK_COM_DIST[5] = {-0.040, -0.031, -0.055, -0.012, -0.020};     // link com distance [m]

// ===== MPC parameters  =====
inline constexpr double COT_DELAY_TAU   = 0.17; // MuJoCo actuator delay [sec]
inline const     double COT_DELAY_ALPHA = std::exp(-CTRL_DT / COT_DELAY_TAU); // not a tunable parameter
inline const     double COT_DELAY_BETA  = 1.0 - COT_DELAY_ALPHA;              // not a tunable parameter

inline constexpr double      MPC_STEP_DT = 1.0 / 100.0; // This value must be same as >> DT << on params.py
inline constexpr std::size_t N_STEPS_REQ = 20; // This value must be less than >> N << on params.py
inline constexpr std::size_t MPC_NX      = 22; // This value must be same as >> self.use_full_nx << on solver.py
inline constexpr std::size_t MPC_NU      = 11; // This value must be same as >> self.use_full_nu << on solver.py
inline constexpr std::size_t MPC_NP      = 26; // This value must be same as >> self.use_full_np << on solver.py
inline constexpr std::chrono::steady_clock::duration MPC_TIMEOUT_DURATUION = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(static_cast<double>(N_STEPS_REQ-1) * MPC_STEP_DT));

// ===== MuJoCo viewer parameters =====
inline constexpr double PATH_SEC = 10.0;   // history length [sec]
inline constexpr float SIZE_DOT  = 0.03f;  // size(radious) of dot
inline constexpr float SIZE_PATH = 0.005f; // size(radious) of path

inline constexpr float RGBA_DOT[4]   = {1.00f, 0.00f, 0.00f, 0.95f}; // current pos color
inline constexpr float RGBA_PATH[4]  = {0.20f, 0.80f, 0.90f, 0.60f}; // current path color
inline constexpr float RGBA_DPATH[4] = {0.60f, 0.60f, 0.60f, 0.60f}; // desired path color

// ===== state noise input (sim->real validation) =====
inline constexpr bool NOISE_ON = false;
inline constexpr std::uint64_t NOISE_SEED = 42;

inline constexpr double POS_NOISE_SIGMA   = 0.006; // white noise std,  [m]
inline constexpr double POS_BIAS_RW       = 0.00005; // bias random-walk, [m/sqrt(s)]
inline constexpr double VEL_NOISE_SIGMA   = 0.010; // white noise std, [m/s]
inline constexpr double ACC_NOISE_SIGMA   = 0.020; // white noise std, [m/s^2]

inline constexpr double RP_NOISE_SIGMA    = 1.00 * M_PI/180.0;  // white noise std (roll/pitch), [rad]
inline constexpr double YAW_NOISE_SIGMA   = 1.50 * M_PI/180.0;  // white noise std (yaw), [rad]
inline constexpr double RP_NOISE_BIAS_RW  = 0.002 * M_PI/180.0;  // [rad/sqrt(s)]
inline constexpr double YAW_NOISE_BIAS_RW = 0.005 * M_PI/180.0;  // [rad/sqrt(s)]

inline constexpr double OMEGA_NOISE_SIGMA = 4.0 * M_PI/180.0;  // white noise std, [rad/s]

inline constexpr double ARM_NOISE_SIGMA   = 0.40 * M_PI/180.0;  // white noise std,  [rad/s]
inline constexpr double ARM_BIAS_RW       = 0.03 * M_PI/180.0;  // bias random-walk, [rad/s/sqrt(s)]

} // namespace param

#endif // PARAMS_H