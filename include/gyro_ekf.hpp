#ifndef GYRO_EKF_H
#define GYRO_EKF_H

#include "params.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

class GyroEKF {
 public:
  GyroEKF() {
    dt_ = param::CTRL_DT;

    xhat_.setZero(); // [phi, theta, psi, p, q, r]

    P_ = 1e-2 * Eigen::Matrix<double, 6, 6>::Identity();

    Q_.setIdentity(); // Process Noise
    Q_.diagonal() << 5e-6, 5e-6, 1e-5, 2e-3, 2e-3, 2e-2;

    R_.setIdentity(); // Measurement Noise
    R_.diagonal() << 5e-4, 5e-4, 2e-3, 4e-2, 4e-2, 1e-2;

    J_ << param::J[0], param::J[1], param::J[2], 
          param::J[3], param::J[4], param::J[5], 
          param::J[6], param::J[7], param::J[8];

    J_inv_ = J_.inverse();
  }

  Eigen::Vector3d step(const Eigen::Vector3d& tau, const Eigen::Vector3d& angle_meas, const Eigen::Vector3d& gyro_meas) {
    Eigen::Matrix<double, 6, 1> y_meas;
    y_meas << angle_meas, gyro_meas;

    predict(tau);
    update(y_meas);

    return xhat_.segment<3>(3);
  }

 private:
  double dt_;
  Eigen::Matrix<double, 6, 1> xhat_;
  Eigen::Matrix<double, 6, 6> P_;
  Eigen::Matrix<double, 6, 6> Q_;
  Eigen::Matrix<double, 6, 6> R_;
  Eigen::Matrix3d J_;
  Eigen::Matrix3d J_inv_;

  static double wrapPi(double a) {
    while (a > M_PI) {a -= 2.0 * M_PI;}
    while (a <= -M_PI) {a += 2.0 * M_PI;}
    return a;
  }

  static Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    Eigen::Matrix3d M;
    M << 0.0, -v(2),  v(1),
         v(2),  0.0, -v(0),
        -v(1),  v(0),  0.0;
    return M;
  }

  static double safeCos(double a) {
    constexpr double eps = 1e-6;
    const double c = std::cos(a);
    if (std::abs(c) < eps) {return (c >= 0.0) ? eps : -eps;}
    return c;
  }

  Eigen::Matrix3d calcL(const Eigen::Vector3d& euler) const {
    const double phi = euler(0);
    const double th  = euler(1);

    const double sphi = std::sin(phi);
    const double cphi = std::cos(phi);
    const double cth  = safeCos(th);
    const double tth  = std::sin(th) / cth;
    const double secth = 1.0 / cth;

    Eigen::Matrix3d L;
    L << 1.0, sphi * tth, cphi * tth,
         0.0, cphi,      -sphi,
         0.0, sphi * secth, cphi * secth;
    return L;
  }

  Eigen::Matrix3d calc_dL_dphi(const Eigen::Vector3d& euler) const {
    const double phi = euler(0);
    const double th  = euler(1);

    const double sphi = std::sin(phi);
    const double cphi = std::cos(phi);
    const double cth  = safeCos(th);
    const double tth  = std::sin(th) / cth;
    const double secth = 1.0 / cth;

    Eigen::Matrix3d M;
    M << 0.0, cphi * tth, -sphi * tth,
         0.0, -sphi,      -cphi,
         0.0, cphi * secth, -sphi * secth;
    return M;
  }

  Eigen::Matrix3d calc_dL_dtheta(const Eigen::Vector3d& euler) const {
    const double phi = euler(0);
    const double th  = euler(1);

    const double sphi = std::sin(phi);
    const double cphi = std::cos(phi);
    const double cth  = safeCos(th);
    const double secth = 1.0 / cth;
    const double sec2th = secth * secth;
    const double tth = std::sin(th) / cth;

    Eigen::Matrix3d M;
    M << 0.0, sphi * sec2th, cphi * sec2th,
         0.0, 0.0,           0.0,
         0.0, sphi * secth * tth, cphi * secth * tth;
    return M;
  }
  
  void predict(const Eigen::Vector3d& tau) {
    const Eigen::Vector3d euler = xhat_.segment<3>(0);
    const Eigen::Vector3d omega = xhat_.segment<3>(3);
    const Eigen::Matrix3d L = calcL(euler);

    Eigen::Matrix<double, 6, 1> x_pred;
    x_pred.segment<3>(0) = euler + dt_ * L * omega;
    x_pred.segment<3>(3) = omega + dt_ * J_inv_ * (tau - omega.cross(J_ * omega));
    x_pred(0) = wrapPi(x_pred(0));
    x_pred(1) = wrapPi(x_pred(1));
    x_pred(2) = wrapPi(x_pred(2));

    Eigen::Matrix3d A_theta = Eigen::Matrix3d::Zero();
    A_theta.col(0) = calc_dL_dphi(euler) * omega;
    A_theta.col(1) = calc_dL_dtheta(euler) * omega;
    A_theta.col(2).setZero();
    const Eigen::Matrix3d A_omega = J_inv_ * hat(J_ * omega) - J_inv_ * hat(omega) * J_;

    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
    F.block<3, 3>(0, 0) += dt_ * A_theta;
    F.block<3, 3>(0, 3)  = dt_ * L;
    F.block<3, 3>(3, 3) += dt_ * A_omega;

    xhat_ = x_pred;
    P_ = F * P_ * F.transpose() + Q_;
    P_ = 0.5 * (P_ + P_.transpose());
  }

  void update(const Eigen::Matrix<double, 6, 1>& y_meas) {
    // Measurement model: y = Hx + v, H = I
    Eigen::Matrix<double, 6, 1> innov = y_meas - xhat_;

    // Wrap angle residuals
    innov(0) = wrapPi(innov(0));
    innov(1) = wrapPi(innov(1));
    innov(2) = wrapPi(innov(2));

    const Eigen::Matrix<double, 6, 6> K = P_ * (P_ + R_).inverse();
    xhat_ = xhat_ + K * innov;
    xhat_(0) = wrapPi(xhat_(0));
    xhat_(1) = wrapPi(xhat_(1));
    xhat_(2) = wrapPi(xhat_(2));

    const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
    P_ = (I - K) * P_ * (I - K).transpose() + K * R_ * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
  }
};

#endif // GYRO_EKF_H