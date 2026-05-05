#ifndef GRADIENT_ASCENT_HPP
#define GRADIENT_ASCENT_HPP

#include <cmath>
#include <cstdio>
#include <array>
#include <Eigen/Dense>

struct GRADIENT_ASCENT {
  Eigen::Vector2d p1_prev = Eigen::Vector2d::Zero();
  Eigen::Vector2d p2_prev = Eigen::Vector2d::Zero();
  Eigen::Vector2d p3_prev = Eigen::Vector2d::Zero();
  Eigen::Vector2d p4_prev = Eigen::Vector2d::Zero();

  inline void reset(const Eigen::Vector2d& r1_init, const Eigen::Vector2d& r2_init, const Eigen::Vector2d& r3_init, const Eigen::Vector2d& r4_init) {
    polar2cart(r1_init, r2_init, r3_init, r4_init, p1_prev, p2_prev, p3_prev, p4_prev);
  }

inline Eigen::Matrix4d build_A1_matrix(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3, const Eigen::Vector2d& p4, const Eigen::Vector2d& Pc, const Eigen::Vector4d& tilt_des) {
  Eigen::Matrix4d A1_out;

  const double pcx = Pc(0), pcy = Pc(1);
  const double s1 = std::sin(tilt_des(0)), c1 = std::cos(tilt_des(0));
  const double s2 = std::sin(tilt_des(1)), c2 = std::cos(tilt_des(1));
  const double s3 = std::sin(tilt_des(2)), c3 = std::cos(tilt_des(2));
  const double s4 = std::sin(tilt_des(3)), c4 = std::cos(tilt_des(3));

  A1_out(0,0) = -inv_sqrt2*param::PWM_ZETA*s1 + (pcy - p1(1))*c1;
  A1_out(0,1) = -inv_sqrt2*param::PWM_ZETA*s2 + (pcy - p2(1))*c2;
  A1_out(0,2) = -inv_sqrt2*param::PWM_ZETA*s3 + (pcy - p3(1))*c3;
  A1_out(0,3) = -inv_sqrt2*param::PWM_ZETA*s4 + (pcy - p4(1))*c4;

  A1_out(1,0) = -inv_sqrt2*param::PWM_ZETA*s1 + (p1(0) - pcx)*c1;
  A1_out(1,1) = -inv_sqrt2*param::PWM_ZETA*s2 + (p2(0) - pcx)*c2;
  A1_out(1,2) = -inv_sqrt2*param::PWM_ZETA*s3 + (p3(0) - pcx)*c3;
  A1_out(1,3) = -inv_sqrt2*param::PWM_ZETA*s4 + (p4(0) - pcx)*c4;

  A1_out(2,0) = -param::PWM_ZETA * c1;
  A1_out(2,1) =  param::PWM_ZETA * c2;
  A1_out(2,2) = -param::PWM_ZETA * c3;
  A1_out(2,3) =  param::PWM_ZETA * c4;

  A1_out(3,0) = -c1;
  A1_out(3,1) = -c2;
  A1_out(3,2) = -c3;
  A1_out(3,3) = -c4;

  return A1_out;
}

inline double eta(const Eigen::Matrix4d& A1) {

  //  η (energy efficiency)  — eq.(7)
  Eigen::FullPivLU<Eigen::Matrix4d> lu(A1);
  if (!lu.isInvertible()) return 0.0;

  Eigen::Vector4d T_hover(0.0, 0.0, 0.0, -param::TOTAL_MASS * param::G);
  Eigen::Vector4d f = lu.solve(T_hover);

  double sum_f = f.sum();
  double sum_power = 0.0;

  for (int i = 0; i < 4; ++i) {
    if (f(i) > 0) {
      double power_i = param::POWER_GAMMA * std::sqrt(f(i)*f(i)*f(i) / (2.0 * param::AIR_DENSITY * param::PROP_DISK_AREA));
      sum_power += power_i;
    }
  }

  return (sum_power > 0.0) ? sum_f / sum_power : 0.0;
}

inline double controllability(const Eigen::Matrix4d& A1) {

  //  C (controllability)  — eq.(8)
  Eigen::FullPivLU<Eigen::Matrix4d> lu(A1);
  if (!lu.isInvertible()) return 0.0;

  Eigen::Matrix4d A1inv = lu.inverse();

  const Eigen::Map<const Eigen::Matrix3d> J_full(param::J);
  Eigen::Matrix<double, 3, 2> J_sub;
  J_sub << J_full(0,0), -J_full(0,1),
           J_full(1,0),  J_full(1,1),
           J_full(2,0), -J_full(2,1);

  double max_norm = 0.0;
  for (int i = 0; i < 4; ++i) {
    Eigen::RowVector3d Gi = A1inv.row(i).segment(1, 3);
    Eigen::RowVector2d Si = Gi * J_sub;
    max_norm = std::max(max_norm, Si.norm());
  }

  return (max_norm > 0.0) ? 1.0 / max_norm : 0.0;
}

inline Eigen::Vector2d estimate_CoM_Hover(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, const Eigen::Vector3d& p4, const Eigen::Vector4d& thrust_des) {
  const double sum_f = thrust_des.sum();
  if (std::abs(sum_f) < 1e-6) {(p1 + p2 + p3 + p4) * 0.25;}

  Eigen::Vector2d CoM = Eigen::Vector2d::Zero();
  CoM(0) = (p1(0)*thrust_des(0) + p2(0)*thrust_des(1) + p3(0)*thrust_des(2) + p4(0)*thrust_des(3)) / sum_f;
  CoM(1) = (p1(1)*thrust_des(0) + p2(1)*thrust_des(1) + p3(1)*thrust_des(2) + p4(1)*thrust_des(3)) / sum_f;

  return CoM;
}

inline void gradients(Eigen::Vector2d& p1, Eigen::Vector2d& p2, Eigen::Vector2d& p3, Eigen::Vector2d& p4,
                             const Eigen::Vector2d& Pc, const Eigen::Vector4d& tilt_des,
                             Eigen::Matrix<double, 4, 2>& grad_eta_out, Eigen::Matrix<double, 4, 2>& grad_C_out) {

  Eigen::Matrix4d A1_base = build_A1_matrix(p1, p2, p3, p4, Pc, tilt_des);
  const double eta_base = eta(A1_base);
  const double C_base   = controllability(A1_base);

  Eigen::Vector2d* p[4] = {&p1, &p2, &p3, &p4};
  for (int i = 0; i < 4; ++i) {
    for (int ax = 0; ax < 2; ++ax) {
      (*p[i])(ax) += param::ARM_OPT_EPS;

      Eigen::Matrix4d A1_perturbed = build_A1_matrix(p1, p2, p3, p4, Pc, tilt_des);

      const double eta_pert = eta(A1_perturbed);
      const double C_pert   = controllability(A1_perturbed);

      (*p[i])(ax) -= param::ARM_OPT_EPS;
      //  Numerical gradients  ∇η, ∇C  (finite differences)
      grad_eta_out(i, ax) = (eta_pert - eta_base) / param::ARM_OPT_EPS;
      grad_C_out(i, ax)   = (C_pert   - C_base)   / param::ARM_OPT_EPS;
    }
  }
}

inline void arm_cmd(Eigen::Vector2d& r1_cmd, Eigen::Vector2d& r2_cmd, Eigen::Vector2d& r3_cmd, Eigen::Vector2d& r4_cmd, const Eigen::Vector3d& p1_mea, const Eigen::Vector3d& p2_mea, const Eigen::Vector3d& p3_mea, const Eigen::Vector3d& p4_mea, const Eigen::Vector4d& tilt_des, const Eigen::Vector4d& thrust_des) {
  const Eigen::Vector2d com_hat = estimate_CoM_Hover(p1_mea, p2_mea, p3_mea, p4_mea, thrust_des);

  Eigen::Matrix<double, 4, 2> grad_eta, grad_C;
  gradients(p1_prev, p2_prev, p3_prev, p4_prev, com_hat, tilt_des, grad_eta, grad_C);

  // --- project ∇C onto orthogonal complement of ∇η ---
  double dot_product = 0.0, norm_eta_sq = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int ax = 0; ax < 2; ++ax) {
      dot_product += grad_C(i, ax) * grad_eta(i, ax);
      norm_eta_sq += grad_eta(i, ax) * grad_eta(i, ax);
    }
  }
  const double proj_scalar = (norm_eta_sq > 1e-12) ? dot_product / norm_eta_sq : 0.0;
  const Eigen::Matrix<double, 4, 2> grad_C_orth = grad_C - proj_scalar * grad_eta;

  //  θ_{k+1} = θ_k + β₁·∇η + β₂·(∇C − proj_{∇η}∇C)       — eq.(10)
  p1_prev += param::ARM_OPT_BETA1 * grad_eta.row(0).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(0).transpose();
  p2_prev += param::ARM_OPT_BETA1 * grad_eta.row(1).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(1).transpose();
  p3_prev += param::ARM_OPT_BETA1 * grad_eta.row(2).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(2).transpose();
  p4_prev += param::ARM_OPT_BETA1 * grad_eta.row(3).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(3).transpose();

  // --- convert back to polar & enforce feasibility ---
  const Eigen::Vector3d p1_prev_3d(p1_prev(0), p1_prev(1), 0.0); const Eigen::Vector3d p2_prev_3d(p2_prev(0), p2_prev(1), 0.0); const Eigen::Vector3d p3_prev_3d(p3_prev(0), p3_prev(1), 0.0); const Eigen::Vector3d p4_prev_3d(p4_prev(0), p4_prev(1), 0.0);
  std::array<Eigen::Vector2d, 4> opt_r;
  cart2polar(p1_prev_3d, p2_prev_3d, p3_prev_3d, p4_prev_3d, opt_r[0], opt_r[1], opt_r[2], opt_r[3]);

  if (make_feasible(opt_r)) {
    r1_cmd = opt_r[0];
    r2_cmd = opt_r[1];
    r3_cmd = opt_r[2];
    r4_cmd = opt_r[3];
  }
}

}; // struct GA

#endif // GRADIENT_ASCENT_HPP