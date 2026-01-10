#ifndef FLIGHT_CONTROL_HPP
#define FLIGHT_CONTROL_HPP

#include "params.hpp"
#include "fdcl_control.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

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
  double          tau_zt;           // [N.m]
};

class ControllerGeom {
public:
  ControllerGeom();
  ~ControllerGeom();

  Eigen::Matrix3d position_control(const ControlInput& in);
  void attitude_control(const Eigen::Matrix3d& R_d, ControlOutput& out);

private:

  fdcl::state_t * state_{nullptr};
  fdcl::command_t * command_{nullptr};
  fdcl::control fdcl_controller; 

  double arm_pos_[4][5]{};

  // -------------- Control Allocation part -------------- 
  Eigen::Vector4d Sequential_Allocation(const Eigen::Vector4d wrench);
  
  // Body-to-Arm base z-rotation
  Eigen::Vector4d C1_des_;  // calculated thrust f_1234 [N]
  Eigen::Vector4d C2_des_;  // calculated tilted angle [rad]
  double tauz_bar_ = 0.0;

  // -------------- Math utils --------------
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