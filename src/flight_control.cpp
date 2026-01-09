#include "flight_control.hpp"

namespace FC {

ControllerGeom::ControllerGeom()
 : state_(new fdcl::state_t()),
  command_(new fdcl::command_t()),
  fdcl_controller(state_, command_) {

  command_->xd << 0.0, 0.0, 0.0;
  command_->xd_dot.setZero();
  command_->xd_2dot.setZero();
  command_->xd_3dot.setZero();
  command_->xd_4dot.setZero();

  command_->b1d << 1.0, 0.0, 0.0;
  command_->b1d_dot.setZero();
  command_->b1d_ddot.setZero();
}

Eigen::Matrix3d ControllerGeom::position_control(const ControlInput& in) {
  command_->xd = in.des_pos;
  command_->xd_dot.setZero();
  command_->xd_2dot.setZero();
  command_->xd_3dot.setZero();
  command_->xd_4dot.setZero();
  command_->b1d = in.des_heading;
  command_->b1d_dot.setZero();
  command_->b1d_ddot.setZero();

  state_->x = in.lin_pos;
  state_->v = in.lin_vel;
  state_->a = in.lin_acc;
  state_->R = quat_to_R(in.quat);
  state_->W = in.ang_vel;

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 5; ++j)
      arm_pos_[i][j] = in.arm_pos(i, j);

  fdcl_controller.position_control();

  return Rx_180 * command_->Rd * Rx_180; // world z-down & body z-down => world z-up & body z-up
}

void ControllerGeom::attitude_control(const Eigen::Matrix3d& R_d, ControlOutput& out) {
  // Flight controller (Geometry control in SE(3))
  fdcl_controller.attitude_control(Rx_180 * R_d * Rx_180);

  double Fz_geom; Eigen::Vector3d M_geom;
  fdcl_controller.output_fM(Fz_geom, M_geom);

  Eigen::Vector3d M_out;
  switch (estimator_state_) {
    case DOB:
      M_geom -= d_hat_;
      M_out = M_geom;
      break;

    case COM_ESTIMATING: {
      Eigen::Vector3d Pc_hat_dot = CoM_update();
      Pc_hat_ += Pc_hat_dot * param::CTRL_DT;
      M_geom -= d_hat_;
      M_out = M_geom;
      break;
    }

    default:
      M_out = M_geom;
      break;
  }
  
  // DOB update
  Eigen::Vector3d RPY = R_to_rpy(state_->R);
  d_hat_ = DoB_update(RPY, M_geom);

  // FC/DOB/Estimator: z-down | ControlAllocation: z-up
  Eigen::Vector4d Wrench; Eigen::Vector3d bPc;
  Wrench << M_out(0), -M_out(1), -M_out(2), Fz_geom;
  bPc << Pc_hat_(0), -Pc_hat_(1), -Pc_hat_(2);

  // Sequential Allocation: update thrust & tilt angle
  Eigen::Vector4d pwm = Sequential_Allocation(Wrench, bPc);

  // copy out
  out.pwm      = pwm;
  out.thrust   = C1_des_;
  out.tilt_rad = C2_des_;
  out.wrench   = Wrench;
  out.d_hat    = d_hat_;
  out.bpc_hat  = bPc;
  out.tau_zt   = tauz_bar_;
}

Eigen::Vector3d ControllerGeom::DoB_update(const Eigen::Vector3d rpy, const Eigen::Vector3d tau_tilde_star) {
  const double Jxx = state_->J(0,0), Jyy = state_->J(1,1), Jzz = state_->J(2,2);

  // ---- Block A: Q*s^2*J*q  ----
  step_third_order(dob_.xr, rpy(0), param::CTRL_DT, k1_, k2_, k3_);
  step_third_order(dob_.xp, rpy(1), param::CTRL_DT, k1_, k2_, k3_);
  step_third_order(dob_.xy, rpy(2), param::CTRL_DT, k1_, k2_, k3_);

  // tau_hat = J(ii) * (w3_ * x1)
  const double tau_hat_r = Jxx * (w3_ * dob_.xr.x1);
  const double tau_hat_p = Jyy * (w3_ * dob_.xp.x1);
  const double tau_hat_y = Jzz * (w3_ * dob_.xy.x1);

  // ---- Block B: Q * tau_tilde  ----
  step_third_order(dob_.yr, tau_tilde_star(0), param::CTRL_DT, k1_, k2_, k3_);
  step_third_order(dob_.yp, tau_tilde_star(1), param::CTRL_DT, k1_, k2_, k3_);
  step_third_order(dob_.yy, tau_tilde_star(2), param::CTRL_DT, k1_, k2_, k3_);

  // Q*tau_tilde = w3_ * y3
  const double Qtau_r = w3_ * dob_.yr.x3;
  const double Qtau_p = w3_ * dob_.yp.x3;
  const double Qtau_y = w3_ * dob_.yy.x3;

  Eigen::Vector3d d_hat(tau_hat_r - Qtau_r, tau_hat_p - Qtau_p, tau_hat_y - Qtau_y);

  d_hat = (d_hat.cwiseMax(Eigen::Vector3d::Constant(-5.0))).cwiseMin(Eigen::Vector3d::Constant(5.0)); // saturation
  return d_hat;
}

Eigen::Vector3d ControllerGeom::CoM_update() {
  Eigen::Vector3d acc = state_->a - param::G*state_->R.col(2);
  const double k1 = 3.0 * wc_;
  const double k2 = 3.0 * w2_;
  const double k3 =       w3_;
  
  step_third_order(com_.ax, acc(0), param::CTRL_DT, k1_, k2_, k3_);
  step_third_order(com_.ay, acc(1), param::CTRL_DT, k1_, k2_, k3_);
  step_third_order(com_.az, acc(2), param::CTRL_DT, k1_, k2_, k3_);

  const Eigen::Vector3d Q_acc(w3_*com_.ax.x3, w3_*com_.ay.x3, w3_*com_.az.x3);
  const Eigen::Vector3d Pc_hat_dot = -param::COM_GAMMA * param::G * (Q_acc.cross(d_hat_));
  return Pc_hat_dot;
}

Eigen::Vector4d ControllerGeom::Sequential_Allocation(const Eigen::Vector4d wrench, const Eigen::Vector3d bpc) {
  // yaw wrench conversion
  tauz_bar_ = param::SERVO_DELAY_ALPHA*wrench(2) + param::SERVO_DELAY_BETA*tauz_bar_;
  double tauz_r = wrench(2) - tauz_bar_;
  double tauz_r_sat = std::clamp(tauz_r, param::TAUZ_MIN, param::TAUZ_MAX);
  double tauz_t = tauz_bar_ + tauz_r - tauz_r_sat;

  // FK for each arm
  Eigen::Matrix<double, 3, 4> r_mea;   // calculated position vect of each arm [m]
  Eigen::Vector4d C2_mea;              // calculated tilted angle [rad]
  for (uint i = 0; i < 4; ++i) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T *= compute_DH(param::B2BASE_A[i], 0.0, 0.0, param::B2BASE_THETA[i]);
    for (int j = 0; j < 5; ++j) T *= compute_DH(param::DH_ARM_A[j], param::DH_ARM_ALPHA[j], 0.0, arm_pos_[i][j]);
    r_mea.col(i) = T.block<3,1>(0,3);
    
    const Eigen::Vector3d heading = T.block<3,3>(0,0).col(0);
    C2_mea(i) = std::asin(std::clamp(heading.head<2>().cwiseAbs().sum() * inv_sqrt2, -0.5, 0.5));
  }

  double s1 = std::sin(C2_mea(0)); double s2 = std::sin(C2_mea(1)); double s3 = std::sin(C2_mea(2)); double s4 = std::sin(C2_mea(3));
  double c1 = std::cos(C2_mea(0)); double c2 = std::cos(C2_mea(1)); double c3 = std::cos(C2_mea(2)); double c4 = std::cos(C2_mea(3));

  // thrust allocation
  Eigen::Matrix4d A1;
  A1(0,0) = inv_sqrt2 * (param::PWM_ZETA + r_mea(2, 0) - bpc(2)) * s1  +  (r_mea(1, 0) - bpc(1)) * c1;
  A1(0,1) = inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 1) + bpc(2)) * s2 +  (r_mea(1, 1) - bpc(1)) * c2;
  A1(0,2) = inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 2) + bpc(2)) * s3 +  (r_mea(1, 2) - bpc(1)) * c3;
  A1(0,3) = inv_sqrt2 * (param::PWM_ZETA + r_mea(2, 3) - bpc(2)) * s4  +  (r_mea(1, 3) - bpc(1)) * c4;
  A1(1,0) = inv_sqrt2 * (-param::PWM_ZETA + r_mea(2, 0) - bpc(2)) * s1 + (-r_mea(0, 0) + bpc(0)) * c1;
  A1(1,1) = inv_sqrt2 * (-param::PWM_ZETA + r_mea(2, 1) - bpc(2)) * s2 + (-r_mea(0, 1) + bpc(0)) * c2;
  A1(1,2) = inv_sqrt2 * (param::PWM_ZETA - r_mea(2, 2) + bpc(2)) * s3  + (-r_mea(0, 2) + bpc(0)) * c3;
  A1(1,3) = inv_sqrt2 * (param::PWM_ZETA - r_mea(2, 3) + bpc(2)) * s4  + (-r_mea(0, 3) + bpc(0)) * c4;
  A1(2,0) =  param::PWM_ZETA * c1;
  A1(2,1) = -param::PWM_ZETA * c2;
  A1(2,2) =  param::PWM_ZETA * c3;
  A1(2,3) = -param::PWM_ZETA * c4;
  A1(3,0) = c1;
  A1(3,1) = c2;
  A1(3,2) = c3;
  A1(3,3) = c4;
  Eigen::Vector4d B1(wrench(0), wrench(1), tauz_r_sat, wrench(3));
  Eigen::FullPivLU<Eigen::Matrix4d> lu_1(A1);
  if (lu_1.isInvertible()) {C1_des_ = lu_1.solve(B1);}
  else {C1_des_ = (A1.transpose()*A1 + 1e-8*Eigen::Matrix4d::Identity()).ldlt().solve(A1.transpose()*B1);}

  // tilt allocation
  Eigen::Matrix4d A2;
  A2(0,0) =  inv_sqrt2 * C1_des_(0);
  A2(0,1) =  inv_sqrt2 * C1_des_(1);
  A2(0,2) = -inv_sqrt2 * C1_des_(2);
  A2(0,3) = -inv_sqrt2 * C1_des_(3);
  A2(1,0) = -inv_sqrt2 * C1_des_(0);
  A2(1,1) =  inv_sqrt2 * C1_des_(1);
  A2(1,2) =  inv_sqrt2 * C1_des_(2);
  A2(1,3) = -inv_sqrt2 * C1_des_(3);
  A2(2,0) = inv_sqrt2 * (bpc(0) + bpc(1)) * s1  + inv_sqrt2 * (-r_mea(0, 0) - r_mea(1, 0)) * C1_des_(0);
  A2(2,1) = inv_sqrt2 * (-bpc(0) + bpc(1)) * s2 + inv_sqrt2 * ( r_mea(0, 1) - r_mea(1, 1)) * C1_des_(1);
  A2(2,2) = inv_sqrt2 * (-bpc(0) -bpc(1)) * s3  + inv_sqrt2 * ( r_mea(0, 2) + r_mea(1, 2)) * C1_des_(2);
  A2(2,3) = inv_sqrt2 * (bpc(0)  - bpc(1)) * s4 + inv_sqrt2 * (-r_mea(0, 3) + r_mea(1, 3)) * C1_des_(3);
  A2(3,0) = inv_sqrt2 * (-r_mea(0, 0) - r_mea(1, 0)) * C1_des_(0);
  A2(3,1) = inv_sqrt2 * (-r_mea(0, 1) + r_mea(1, 1)) * C1_des_(1);
  A2(3,2) = inv_sqrt2 * ( r_mea(0, 2) + r_mea(1, 2)) * C1_des_(2);
  A2(3,3) = inv_sqrt2 * ( r_mea(0, 3) - r_mea(1, 3)) * C1_des_(3);
  Eigen::Vector4d B2(0.0, 0.0, tauz_t, 0.0);
  Eigen::FullPivLU<Eigen::Matrix4d> lu_2(A2);
  if (lu_2.isInvertible()) {C2_des_ = lu_2.solve(B2);}
  else {C2_des_ = (A2.transpose()*A2 + 1e-8*Eigen::Matrix4d::Identity()).ldlt().solve(A2.transpose()*B2);}

  Eigen::Vector4d pwm;
  for (int i = 0; i < 4; ++i) {
    const double val = std::max(0.0, (C1_des_(i) - param::PWM_B) / param::PWM_A);
    pwm(i) = std::sqrt(val);
    pwm(i) = std::clamp(pwm(i), 0.0, 1.0);
  }

  return pwm;
}

void ControllerGeom::set_mode(const uint8_t mode) {
  if      (mode == 1){
    estimator_state_ = 1;
    std::cout << "[geometry controller] : DOB" << std::endl;
  }
  else if (mode == 2){
    estimator_state_ = 2;
    std::cout << "[geometry controller] : CoM" << std::endl;
  }
  else {
    estimator_state_ = 0;
    std::cout << "[geometry controller] : Conventional" << std::endl;
  }
}

ControllerGeom::~ControllerGeom() {
  delete state_;
  delete command_;
}

} // namespace FC