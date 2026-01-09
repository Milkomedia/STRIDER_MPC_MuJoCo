#ifndef FLIGHT_CONTROL_HPP
#define FLIGHT_CONTROL_HPP

#include "params.hpp"
#include "fdcl_control.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

#define DOB             1  // Mode parameter
#define COM_ESTIMATING  2  // Mode parameter

namespace FC {

struct ControlInput {
  Eigen::Vector3d des_pos;      // desired position vector  [m]
  Eigen::Vector3d des_heading;  // desired heading vector(normalized)
  Eigen::Vector3d lin_pos;      // current position vector  [m]
  Eigen::Vector3d lin_vel;      // current velocity vector  [m/s]
  Eigen::Vector3d lin_acc;      // current accel vector     [m/s^2]
  Eigen::Quaterniond quat;      // current orientation      [wxyz]
  Eigen::Vector3d ang_vel;      // current angular velocity [rad/s]
  Eigen::Matrix<double, 4, 5> arm_pos;  // current joint angle [rad]
};
struct ControlOutput {
  Eigen::Vector4d pwm;              // pwm for arm1~4 [0.0 ~ 1.0]
  Eigen::Vector4d thrust;           // thrust for arm1~4 [N]
  Eigen::Vector4d tilt_rad;         // tilt cmd for arm 1~4 [rad]
  Eigen::Vector4d wrench;           // control wrench by controller [Mx My Mz Fz] in N, N.m
  Eigen::Vector3d d_hat;            // [N.m]
  Eigen::Vector3d bpc_hat;          // [m]
  double          tau_zt;           // [N.m]
};
struct ThirdOrderState {
  double x1=0, x2=0, x3=0;
};
struct DOBState {
  ThirdOrderState xr, xp, xy; // Q*s^2*J*q
  ThirdOrderState yr, yp, yy; // Q*tau_tilde
};
struct COMState {
  ThirdOrderState ax, ay, az;
};

class ControllerGeom {
public:
  ControllerGeom();
  ~ControllerGeom();

  Eigen::Matrix3d position_control(const ControlInput& in);
  void attitude_control(const Eigen::Matrix3d& R_d, ControlOutput& out);
  void set_mode(const uint8_t mode);

private:

  fdcl::state_t * state_{nullptr};
  fdcl::command_t * command_{nullptr};
  fdcl::control fdcl_controller; 

  double arm_pos_[4][5]{};

  uint8_t estimator_state_ = 0; // 0->conventional / 1->dob / 2->com

  // -------------- Control Allocation part -------------- 
  Eigen::Vector4d Sequential_Allocation(const Eigen::Vector4d wrench, const Eigen::Vector3d bpc);
  
  // Body-to-Arm base z-rotation
  Eigen::Vector4d C1_des_;              // calculated thrust f_1234 [N]
  Eigen::Vector4d C2_des_;              // calculated tilted angle [rad]
  double tauz_bar_ = 0.0;

  // --------------       Q-filter      --------------
  const double wc_ = 2.0 * M_PI * param::COM_CUTOFF_FREQ;
  const double w2_ = wc_ * wc_;
  const double w3_ = w2_ * wc_;

  const double k1_ = 2.0 * wc_;
  const double k2_ = 2.0 * w2_;
  const double k3_ =       w3_;

  // --------------       DOB part      --------------
  Eigen::Vector3d DoB_update(const Eigen::Vector3d rpy, const Eigen::Vector3d tau_tilde_star);
  Eigen::Vector3d d_hat_ = Eigen::Vector3d::Zero(); // [N.m]
  DOBState dob_;

  // -------------- CoM Estimation part --------------
  Eigen::Vector3d CoM_update();
  Eigen::Vector3d Pc_hat_ = Eigen::Vector3d::Zero(); // [m]
  COMState com_;

  // -------------- Math utils --------------
  static inline Eigen::Vector3d R_to_rpy(const Eigen::Matrix3d& R) {
    // input  R  -> z-down
    // output rpy -> z-down
    // ZYX Tait–Bryan angles (roll, pitch, yaw)

    const double sy = std::sqrt(R(0,0) * R(0,0) + R(1,0) * R(1,0));
    const bool singular = sy < 1e-9;

    Eigen::Vector3d rpy;
    if (!singular) {
      rpy(0)     = std::atan2(R(2,1), R(2,2));
      rpy(1) = -std::asin(-R(2,0));
      rpy(2)   = -std::atan2(R(1,0), R(0,0));
    }
    else {
      rpy(0)     = std::atan2(-R(1,2), R(1,1));
      rpy(1) = -std::asin(-R(2,0));
      rpy(2)   = 0.0;
    }
    return rpy;
  }
  static inline Eigen::Matrix4d compute_DH(double a, double alpha, double d, double theta) {
    Eigen::Matrix4d T;
    const double c_th = cos(theta);
    const double s_th = sin(theta);
    const double c_a = cos(alpha);
    const double s_a = sin(alpha);
    
    T << c_th, -s_th * c_a,  s_th * s_a, a * c_th,
        s_th,   c_th * c_a, -c_th * s_a, a * s_th,
        0.0,         s_a,        c_a,        d,
        0.0,         0.0,        0.0,       1.0;
    return T;
  }
  static inline void step_third_order(ThirdOrderState& s, double u, double dt, double k1, double k2, double k3) {
    const double x1_dot = -k1*s.x1 - k2*s.x2 - k3*s.x3 + u;
    const double x2_dot =  s.x1;
    const double x3_dot =  s.x2;

    s.x1 += x1_dot * dt;
    s.x2 += x2_dot * dt;
    s.x3 += x3_dot * dt;
  } 
  static inline Eigen::Matrix3d quat_to_R(const Eigen::Quaterniond q) {
    // Quaternion to SO3 map (z-up input → z-down output)
    const double xx = q.x()*q.x(), yy = q.y()*q.y(), zz = q.z()*q.z();
    const double xy = q.x()*q.y(), xz = q.x()*q.z(), yz = q.y()*q.z();
    const double wx = q.w()*q.x(), wy = q.w()*q.y(), wz = q.w()*q.z();

    Eigen::Matrix3d R;
    R(0,0) =  1.0 - 2.0 * (yy + zz);
    R(0,1) = -2.0 * (xy - wz);
    R(0,2) = -2.0 * (xz + wy);
    R(1,0) = -2.0 * (xy + wz);
    R(1,1) =  1.0 - 2.0 * (xx + zz);
    R(1,2) =  2.0 * (yz - wx);
    R(2,0) = -2.0 * (xz - wy);
    R(2,1) =  2.0 * (yz + wx);
    R(2,2) =  1.0 - 2.0 * (xx + yy);
    return R;
  }
  
  const double inv_sqrt2 = 0.7071067811865474617150084668537601828575;  // 1/sqrt(2)
  const double sqrt2 = 1.4142135623730951454746218587388284504414;      // sqrt(2)
  const Eigen::Matrix3d Rx_180 = (Eigen::Matrix3d() << 1, 0, 0, 0, -1, 0, 0, 0, -1).finished();
};


} // namespace FC

#endif // FLIGHT_CONTROL_HPP