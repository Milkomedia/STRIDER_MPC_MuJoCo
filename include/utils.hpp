#include "params.hpp"

#include <filesystem>
#include <string>

#if defined(__APPLE__)
  #include <mach-o/dyld.h>
#elif defined(__linux__)
  #include <unistd.h>
#endif

static inline constexpr double inv_sqrt2 = 0.7071067811865474617150084668537601828575;  // 1/sqrt(2)
static inline constexpr double sqrt2 = 1.4142135623730951454746218587388284504414;      // sqrt(2)
struct SimState { // use for copy snapshot in thread lock
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R   = Eigen::Matrix3d::Identity();
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  double arm_q[20] = {0.0};
};

static inline Eigen::Vector3d fig8_point(double t_sec){
  // Gerono lemniscate trajectory: x = A sin(wt), y = A sin(wt)cos(wt) = 0.5A sin(2wt)
  constexpr double x    = 2.0;               // lobe half-width in X [m]
  constexpr double y    = x / 1.5;           // lobe half-height in Y [m]
  constexpr double freq = 2.0 * M_PI * 0.16; // [rad/s]

  const double s = std::sin(freq * t_sec);
  const double c = std::cos(freq * t_sec);
  return Eigen::Vector3d(x*s, y*s*c, -1.0);
}

static inline void fig8_point_pva(double t_sec, Eigen::Vector3d& p_d, Eigen::Vector3d& v_d, Eigen::Vector3d& a_d){
  constexpr double l = 2.0;               // lobe half-width in X [m]
  constexpr double d = 0.0;               // lobe half-height in Y [m]
  constexpr double f = 2.0 * M_PI / 3.5;  // [rad/s]

  const double s = std::sin(f * t_sec);
  const double c = std::cos(f * t_sec);

  p_d = Eigen::Vector3d(l*s, d*s*c, -1.0);
  v_d = Eigen::Vector3d(l*f*c, d*f*(1.0-2.0*s*s), 0.0);
  a_d = Eigen::Vector3d(-l*f*f*s, -4.0*d*f*f*s*c, 0.0);
}

static inline void circle_pva(double t_sec, Eigen::Vector3d& p_d, Eigen::Vector3d& v_d, Eigen::Vector3d& a_d){
  constexpr double r = 1.5;              // circle radious [m]
  constexpr double f = 2.0 * M_PI * 0.3; // [rad/s]

  const double s = std::sin(f * t_sec);
  const double c = std::cos(f * t_sec);

  p_d = Eigen::Vector3d(r*s, r*c, -1.0);
  v_d = Eigen::Vector3d(r*f*c, -r*f*s, 0.0);
  a_d = Eigen::Vector3d(-r*f*f*s, -r*f*f*c, 0.0);
}

static inline void l_traj_pva(double t_sec, Eigen::Vector3d& p_d, Eigen::Vector3d& v_d, Eigen::Vector3d& a_d){
  constexpr double lx_ = 1.5;            // width in X [m]
  constexpr double ly_ = 0.0;            // width in Y [m]
  constexpr double T_  = 4.0;            // period [sec]
  constexpr double f   = 2.0 * M_PI / T_;  // [rad/s]

  const double r = 2.0 / T_ * std::fmod(t_sec, 3.0 * T_);  // [0,6)

  if (r > 0.0 && r <= 1.0) {
    const double s = std::sin(f * t_sec); const double s2 = std::sin(2.0 *f * t_sec);
    const double c = std::cos(f * t_sec);
    p_d = Eigen::Vector3d(-lx_*c, -ly_*c, -1.0);
    v_d = Eigen::Vector3d( lx_*f*s, ly_*f*s, 0.0);
    a_d = Eigen::Vector3d( lx_*f*f*s2, ly_*f*f*s2, 0.0);
    return;
  }

  if (r > 1.0 && r <= 3.0) {
    p_d = Eigen::Vector3d( lx_, ly_, -1.0);
    v_d = Eigen::Vector3d::Zero();
    a_d = Eigen::Vector3d::Zero();
    return;
  }

  if (r > 3.0 && r <= 4.0) {
    const double s = std::sin(f * t_sec); const double s2 = std::sin(2.0 *f * t_sec);
    const double c = std::cos(f * t_sec);
    p_d = Eigen::Vector3d(-lx_*c, -ly_*c, -1.0);
    v_d = Eigen::Vector3d( lx_*f*s, ly_*f*s, 0.0);
    a_d = Eigen::Vector3d(-lx_*f*f*s2,-ly_*f*f*s2, 0.0);
    return;
  }

  // 4 < r < 6
  p_d = Eigen::Vector3d(-lx_, -ly_, -1.0);
  v_d = Eigen::Vector3d::Zero();
  a_d = Eigen::Vector3d::Zero();
}

static inline Eigen::Vector3d goes_to(const Eigen::Vector3d& p_d, const double t, const double t_term){
  Eigen::Vector3d out = p_d * t / t_term;
  return out;
}

static inline Eigen::Vector3d square4_point(double t_sec) {
  constexpr double T = 4.0;
  constexpr double L = 1.0;

  std::int64_t k = static_cast<std::int64_t>(std::floor(t_sec / T));
  int phase = static_cast<int>(k % 4);
  if (phase < 0) phase += 4;

  double sx = -L, sy = L;
  switch (phase) {
    case 0: sx = -L; sy =  L; break;
    case 1: sx = -L; sy = -L; break;
    case 2: sx =  L; sy = -L; break;
    default:sx =  L; sy =  L; break;
  }

  return Eigen::Vector3d(sx, sy, -2.0);
}

static inline Eigen::Vector3d quat_to_RPY(const Eigen::Quaterniond q) {
  // Quaternion to Euler angle map
  double w = q.w(); double x = q.x(); double y = q.y(); double z = q.z();

  const double xx = x*x, yy = y*y, zz = z*z;
  const double xy = x*y, xz = x*z, yz = y*z;
  const double wx = w*x, wy = w*y, wz = w*z;
  
  const double phi = std::atan2(2.0*(wx + yz), 1.0 - 2.0*(xx + yy));
  double sinp = 2.0*(wy - xz);
  sinp = std::max(-1.0, std::min(1.0, sinp));
  const double th = std::asin(sinp);
  const double psi = std::atan2(2.0*(wz + xy), 1.0 - 2.0*(yy + zz));

  return Eigen::Vector3d(phi, th, psi);
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

static inline Eigen::Vector3d R_to_rpy(const Eigen::Matrix3d& R) {
  const double r11 = R(0,0), r21 = R(1,0);
  const double r31 = R(2,0), r32 = R(2,1), r33 = R(2,2);
  const double r12 = R(0,1), r22 = R(1,1);

  // th = asin(-r31)
  double sin_th = -r31;
  sin_th = std::max(-1.0, std::min(1.0, sin_th));
  const double th = std::asin(sin_th);

  // If cos(th) is near zero => Choose phi=0
  const double cth2 = 1.0 - sin_th*sin_th; // = cos(th)^2
  if (cth2 > 1e-12) {
    const double phi = std::atan2(r32, r33);
    const double psi = std::atan2(r21, r11);
    return Eigen::Vector3d(phi, th, psi);
  }
  else {
    const double phi = 0.0;
    const double psi = std::atan2(-r12, r22);
    return Eigen::Vector3d(phi, th, psi);
  }
}

static inline Eigen::Matrix3d expm_hat(const Eigen::Vector3d& w) {
  constexpr double eps = 1e-12;

  const double th2 = w.dot(w);                 // theta^2
  const double th  = std::sqrt(th2 + eps);     // theta

  const double A = std::sin(th) / th;
  const double B = (1.0 - std::cos(th)) / (th2 + eps);

  const Eigen::Matrix3d K = hat(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  return I + A * K + B * (K * K);
}

static inline void onearm_IK(const Eigen::Vector3d& pos, const Eigen::Vector3d& heading, double out5[5]) {
  Eigen::Vector3d heading_in = heading;
  const double a2_sq = param::DH_ARM_A[1]*param::DH_ARM_A[1];
  const double a3_sq = param::DH_ARM_A[2]*param::DH_ARM_A[2];
  const double a2a3_2 = 2.0 * param::DH_ARM_A[1] * param::DH_ARM_A[2];

  const double hn = heading_in.norm();
  if (hn > 1e-12) heading_in /= hn;

  const Eigen::Vector3d p04 = pos - param::DH_ARM_A[4] * heading_in;

  const double th1 = std::atan2(p04.y(), p04.x());
  const double c1 = std::cos(th1), s1 = std::sin(th1);

  const double cross_z  = p04.x() * heading_in.y() - p04.y() * heading_in.x();
  const double denom_xy = std::hypot(p04.x(), p04.y()) + 1e-12;
  double th5 = -std::acos(std::clamp(std::abs(cross_z) / denom_xy, -1.0, 1.0));
  if (th5 <= M_PI/2.0) th5 += M_PI/2.0;
  if (p04.x() * pos.y() - p04.y() * pos.x() > 0.0) th5 = -th5;

  Eigen::Vector3d heading_projected = heading_in - std::sin(th5) * Eigen::Vector3d(s1, -c1, 0.0);
  const double hp_n = heading_projected.norm();
  if (hp_n > 1e-12) heading_projected /= hp_n;

  const Eigen::Vector3d p01(param::DH_ARM_A[0] * c1, param::DH_ARM_A[0] * s1, 0.0);
  const Eigen::Vector3d p34 = param::DH_ARM_A[3] * heading_projected;
  const Eigen::Vector3d p03 = p04 - p34;
  const Eigen::Vector3d p31 = p03 - p01;

  const double r = std::hypot(p31.x(), p31.y());
  const double s = p31.z();
  double D = (r*r + s*s - (a2_sq + a3_sq)) / a2a3_2;
  D = std::clamp(D, -1.0, 1.0);
  const double th3 = std::acos(D);

  const double alpha = std::atan2(s, r);
  const double beta  = std::atan2(param::DH_ARM_A[2] * std::sin(th3), param::DH_ARM_A[1] + param::DH_ARM_A[2] * std::cos(th3));
  const double th2   = alpha - beta;

  const double th23 = th2 + th3;
  const double c23  = std::cos(th23), s23 = std::sin(th23);
  const Eigen::Vector3d x3(c1 * c23,  s1 * c23,  s23);
  const Eigen::Vector3d z3(    s1,      -c1,    0.0);

  Eigen::Vector3d x4_des = p34;
  const double x4n = x4_des.norm();
  if (x4n > 1e-12) x4_des /= x4n;

  const double c4 = std::clamp(x3.dot(x4_des), -1.0, 1.0);
  const double s4 = z3.dot(x3.cross(x4_des));
  const double th4 = std::atan2(s4, c4);

  out5[0] = th1;
  out5[1] = th2;
  out5[2] = th3;
  out5[3] = th4;
  out5[4] = th5;
}

static inline void IK(const Eigen::Vector3d& bPcot, const Eigen::Matrix3d& bRcot, const Eigen::Vector4d& th_tvc, const double l, double q[20]) {
  std::array<Eigen::Vector3d, 4> bodyParm;
  const Eigen::Vector3d e1 = bRcot.col(0);
  const Eigen::Vector3d e2 = bRcot.col(1);
  bodyParm[0] = bPcot + (0.5 * l) * ( e1 - e2);
  bodyParm[1] = bPcot + (0.5 * l) * (-e1 - e2);
  bodyParm[2] = bPcot + (0.5 * l) * (-e1 + e2);
  bodyParm[3] = bPcot + (0.5 * l) * ( e1 + e2);
  std::array<Eigen::Vector3d, 4> bodyE3arm;
  const double s1 = std::sin(th_tvc(0)); const double c1 = std::cos(th_tvc(0));
  const double s2 = std::sin(th_tvc(1)); const double c2 = std::cos(th_tvc(1));
  const double s3 = std::sin(th_tvc(2)); const double c3 = std::cos(th_tvc(2));
  const double s4 = std::sin(th_tvc(3)); const double c4 = std::cos(th_tvc(3));
  bodyE3arm[0] = bRcot * Eigen::Vector3d( s1*M_SQRT1_2,  s1*M_SQRT1_2, -c1);
  bodyE3arm[1] = bRcot * Eigen::Vector3d( s2*M_SQRT1_2, -s2*M_SQRT1_2, -c2);
  bodyE3arm[2] = bRcot * Eigen::Vector3d(-s3*M_SQRT1_2, -s3*M_SQRT1_2, -c3);
  bodyE3arm[3] = bRcot * Eigen::Vector3d(-s4*M_SQRT1_2,  s4*M_SQRT1_2, -c4);
  
  for (uint8_t i = 0; i < 4; ++i) {
    const double s = std::sin(param::B2BASE_THETA[i]);
    const double c = std::cos(param::B2BASE_THETA[i]);
    const double a = param::B2BASE_A[i];

    const double px = bodyParm[i].x();
    const double py = bodyParm[i].y();
    const double ezx = bodyE3arm[i].x();
    const double ezy = bodyE3arm[i].y();

    const double x_base =  c * px  + s * py - a;
    const double y_base =  s * px  - c * py;
    const double ex_base = c * ezx + s * ezy;
    const double ey_base = s * ezx - c * ezy;

    Eigen::Vector3d baseParm(x_base, y_base, -bodyParm[i].z());
    Eigen::Vector3d baseE3arm(ex_base, ey_base, -bodyE3arm[i].z());

    onearm_IK(baseParm, baseE3arm, &q[5 * i]);
  }
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

static inline Eigen::Vector3d FK(const double q[20]) {
  // returns {b}->{cot} position and z-directional heading vector
  Eigen::Vector3d bpcot = Eigen::Vector3d::Zero();
  for (uint8_t i = 0; i < 4; ++i) {
    Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
    T_i *= compute_DH(param::B2BASE_A[i], param::B2BASE_ALPHA[i], 0.0, param::B2BASE_THETA[i]);
    for (int j = 0; j < 5; ++j) {T_i *= compute_DH(param::DH_ARM_A[j], param::DH_ARM_ALPHA[j], 0.0, q[5*i+j]);}
    bpcot += T_i.block<3, 1>(0, 3);
  }
  bpcot *= 0.25;

  return bpcot;
}

static inline void Sequential_Allocation(const double& thrust_d, const Eigen::Vector3d& tau_d, double& tauz_bar, const double arm_q[20], const Eigen::Vector3d& Pc, Eigen::Vector4d& C1_des, Eigen::Vector4d& C2_des) {
  // yaw wrench conversion
  tauz_bar = param::SERVO_DELAY_ALPHA*tau_d(2) + param::SERVO_DELAY_BETA*tauz_bar;
  double tauz_r = tau_d(2) - tauz_bar;
  double tauz_r_sat = std::clamp(tauz_r, param::TAUZ_MIN, param::TAUZ_MAX);
  double tauz_t = tauz_bar + tauz_r - tauz_r_sat;

  // FK for each arm
  Eigen::Matrix<double, 3, 4> r_mea;   // calculated position vect of each arm [m]
  Eigen::Vector4d C2_mea;              // calculated tilted angle [rad]
  for (uint i = 0; i < 4; ++i) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T *= compute_DH(param::B2BASE_A[i], param::B2BASE_ALPHA[i], 0.0, param::B2BASE_THETA[i]);
    for (int j = 0; j < 5; ++j) {T *= compute_DH(param::DH_ARM_A[j], param::DH_ARM_ALPHA[j], 0.0, arm_q[5*i+j]);}
    r_mea.col(i) = T.block<3,1>(0,3);
    
    const Eigen::Vector3d heading = T.block<3,3>(0,0).col(0);
    C2_mea(i) = std::asin(std::clamp(heading.head<2>().cwiseAbs().sum() * inv_sqrt2, -0.5, 0.5));
  }

  double s1 = std::sin(C2_mea(0)); double s2 = std::sin(C2_mea(1)); double s3 = std::sin(C2_mea(2)); double s4 = std::sin(C2_mea(3));
  double c1 = std::cos(C2_mea(0)); double c2 = std::cos(C2_mea(1)); double c3 = std::cos(C2_mea(2)); double c4 = std::cos(C2_mea(3));

  double pcx = Pc(0); double pcy = Pc(1); double pcz = Pc(2);

  // thrust allocation
  Eigen::Matrix4d A1;
  A1(0,0) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 0) - pcz) * s1 + (pcy - r_mea(1, 0)) * c1;
  A1(0,1) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 1) + pcz) * s2 + (pcy - r_mea(1, 1)) * c2;
  A1(0,2) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 2) + pcz) * s3 + (pcy - r_mea(1, 2)) * c3;
  A1(0,3) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 3) - pcz) * s4 + (pcy - r_mea(1, 3)) * c4;
  A1(1,0) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 0) + pcz) * s1 + (r_mea(0, 0) - pcx) * c1;
  A1(1,1) = -inv_sqrt2 * (-param::PWM_ZETA - r_mea(2, 1) + pcz) * s2 + (r_mea(0, 1) - pcx) * c2;
  A1(1,2) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 2) - pcz) * s3 + (r_mea(0, 2) - pcx) * c3;
  A1(1,3) = -inv_sqrt2 * ( param::PWM_ZETA + r_mea(2, 3) - pcz) * s4 + (r_mea(0, 3) - pcx) * c4;
  A1(2,0) = -param::PWM_ZETA * c1;
  A1(2,1) =  param::PWM_ZETA * c2;
  A1(2,2) = -param::PWM_ZETA * c3;
  A1(2,3) =  param::PWM_ZETA * c4;
  A1(3,0) = -c1;
  A1(3,1) = -c2;
  A1(3,2) = -c3;
  A1(3,3) = -c4;
  Eigen::Vector4d B1(tau_d(0), tau_d(1), tauz_r_sat, thrust_d);
  Eigen::FullPivLU<Eigen::Matrix4d> lu_1(A1);
  if (lu_1.isInvertible()) {C1_des = lu_1.solve(B1);}
  else {C1_des = (A1.transpose()*A1 + 1e-8*Eigen::Matrix4d::Identity()).ldlt().solve(A1.transpose()*B1);}

  // Thrust clamp
  for (uint8_t i = 0; i < 4; ++i) {C1_des(i) = std::clamp(C1_des(i), 8.0, 52.0);}

  // tilt allocation
  Eigen::Matrix4d A2;
  A2(0,0) =  inv_sqrt2 * C1_des(0);
  A2(0,1) =  inv_sqrt2 * C1_des(1);
  A2(0,2) = -inv_sqrt2 * C1_des(2);
  A2(0,3) = -inv_sqrt2 * C1_des(3);
  A2(1,0) =  inv_sqrt2 * C1_des(0);
  A2(1,1) = -inv_sqrt2 * C1_des(1);
  A2(1,2) = -inv_sqrt2 * C1_des(2);
  A2(1,3) =  inv_sqrt2 * C1_des(3);
  A2(2,0) = inv_sqrt2 * (-pcx + pcy + r_mea(0, 0) - r_mea(1, 0)) * C1_des(0);
  A2(2,1) = inv_sqrt2 * ( pcx + pcy - r_mea(0, 1) - r_mea(1, 1)) * C1_des(1);
  A2(2,2) = inv_sqrt2 * ( pcx - pcy - r_mea(0, 2) + r_mea(1, 2)) * C1_des(2);
  A2(2,3) = inv_sqrt2 * (-pcx - pcy + r_mea(0, 3) + r_mea(1, 3)) * C1_des(3);
  A2(3,0) = inv_sqrt2 * ( r_mea(0, 0) - r_mea(1, 0)) * C1_des(0);
  A2(3,1) = inv_sqrt2 * ( r_mea(0, 1) + r_mea(1, 1)) * C1_des(1);
  A2(3,2) = inv_sqrt2 * (-r_mea(0, 2) + r_mea(1, 2)) * C1_des(2);
  A2(3,3) = inv_sqrt2 * (-r_mea(0, 3) - r_mea(1, 3)) * C1_des(3);
  Eigen::Vector4d B2(0.0, 0.0, tauz_t, 0.0);
  Eigen::FullPivLU<Eigen::Matrix4d> lu_2(A2);
  if (lu_2.isInvertible()) {C2_des = lu_2.solve(B2);}
  else {C2_des = (A2.transpose()*A2 + 1e-8*Eigen::Matrix4d::Identity()).ldlt().solve(A2.transpose()*B2);}

  // Tilt angle clamp
  for (uint8_t i = 0; i < 4; ++i) {C2_des(i) = std::clamp(C2_des(i), -0.175, 0.175);}
}

static inline void Control_Allocation(const double& F_d, const Eigen::Vector3d& tau_d, const Eigen::Vector3d& r_cot, const Eigen::Vector3d& Pc, Eigen::Vector4d& F1234) {
  constexpr double l = param::L_DIST / 2.0;

  const double dx = r_cot(0) - Pc(0);
  const double dy = r_cot(1) - Pc(1);

  Eigen::Matrix4d A_d;
  A_d(0,0) =  l - dy;
  A_d(0,1) =  l - dy;
  A_d(0,2) = -l - dy;
  A_d(0,3) = -l - dy;
  A_d(1,0) =  l + dx;
  A_d(1,1) = -l + dx;
  A_d(1,2) = -l + dx;
  A_d(1,3) =  l + dx;
  A_d(2,0) = -param::PWM_ZETA;
  A_d(2,1) =  param::PWM_ZETA;
  A_d(2,2) = -param::PWM_ZETA;
  A_d(2,3) =  param::PWM_ZETA;
  A_d(3,0) = -1.0;
  A_d(3,1) = -1.0;
  A_d(3,2) = -1.0;
  A_d(3,3) = -1.0;

  Eigen::Vector4d Wrench(tau_d(0), tau_d(1), tau_d(2), F_d);
  Eigen::FullPivLU<Eigen::Matrix4d> lu(A_d);
  if (lu.isInvertible()) {F1234 = lu.solve(Wrench);}
  else {F1234 = (A_d.transpose()*A_d + 1e-8*Eigen::Matrix4d::Identity()).ldlt().solve(A_d.transpose()*Wrench);}
}

namespace NOISE {
struct Rng {
  std::uint64_t s = 88172645463325252ull; // non-zero seed

  inline std::uint64_t next_u64() { // return random number
    // xorshift64*
    std::uint64_t x = s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    s = x;
    return x * 2685821657736338717ull;
  }

  inline double uni01() { // return random number [0.1)
    // Convert top 53 bits to [0,1)
    const std::uint64_t r = next_u64();
    const std::uint64_t u = (r >> 11);
    return static_cast<double>(u) * (1.0 / 9007199254740992.0); // 2^53
  }

  inline double randn_fast() {
    // Approx N(0,1) using sum of 12 uniforms - 6 (CLT), no trig/log
    double s12 = 0.0;
    for (int i = 0; i < 12; ++i) s12 += uni01();
    return s12 - 6.0;
  }
};

struct State {
  std::chrono::steady_clock::time_point last_t;

  Rng rng;

  // Bias states (random-walk)
  Eigen::Vector3d pos_bias   = Eigen::Vector3d::Zero();
  Eigen::Vector3d theta_bias = Eigen::Vector3d::Zero();
  double q_bias[20] = {0.0};
};

// Reset/init the noise state
static inline void reset(State& st, std::uint64_t seed, const std::chrono::steady_clock::time_point& now) {
  st.last_t = now;

  st.rng.s = (seed == 0) ? 1ull : seed;

  st.pos_bias.setZero();
  st.theta_bias.setZero();
  for (uint8_t i=0; i<20; ++i) {st.q_bias[i] = 0.0;}
}

static inline void apply(State& st, const std::chrono::steady_clock::time_point& now, SimState& raw) {

  double dt = std::chrono::duration<double>(now - st.last_t).count();
  dt = std::clamp(dt, 0.002, 0.01);
  const double sdt = std::sqrt(dt);
  st.last_t = now;

  // bias update
  st.pos_bias.x() += param::POS_BIAS_RW * sdt * st.rng.randn_fast();
  st.pos_bias.y() += param::POS_BIAS_RW * sdt * st.rng.randn_fast();
  st.pos_bias.z() += param::POS_BIAS_RW * sdt * st.rng.randn_fast();
  
  st.theta_bias.x() += param::RP_NOISE_BIAS_RW  * sdt * st.rng.randn_fast();
  st.theta_bias.y() += param::RP_NOISE_BIAS_RW  * sdt * st.rng.randn_fast();
  st.theta_bias.z() += param::YAW_NOISE_BIAS_RW * sdt * st.rng.randn_fast();

  for (uint8_t i=0; i<20; ++i) {st.q_bias[i] += param::ARM_BIAS_RW * sdt * st.rng.randn_fast();}

  // white noise & bias apply
  raw.pos.x() += st.pos_bias.x() + param::POS_NOISE_SIGMA  * st.rng.randn_fast();
  raw.pos.y() += st.pos_bias.y() + param::POS_NOISE_SIGMA  * st.rng.randn_fast();
  raw.pos.z() += st.pos_bias.z() + param::POS_NOISE_SIGMA * st.rng.randn_fast();

  raw.vel.x() += param::VEL_NOISE_SIGMA * st.rng.randn_fast();
  raw.vel.y() += param::VEL_NOISE_SIGMA * st.rng.randn_fast();
  raw.vel.z() += param::VEL_NOISE_SIGMA * st.rng.randn_fast();

  raw.acc.x() += param::ACC_NOISE_SIGMA * st.rng.randn_fast();
  raw.acc.y() += param::ACC_NOISE_SIGMA * st.rng.randn_fast();
  raw.acc.z() += param::ACC_NOISE_SIGMA * st.rng.randn_fast();

  Eigen::Vector3d dtheta;
  dtheta.x() = st.theta_bias.x() + param::RP_NOISE_SIGMA * st.rng.randn_fast();
  dtheta.y() = st.theta_bias.y() + param::RP_NOISE_SIGMA * st.rng.randn_fast();
  dtheta.z() = st.theta_bias.z() + param::YAW_NOISE_SIGMA * st.rng.randn_fast();
  raw.R = raw.R * expm_hat(dtheta);

  raw.omega.x() += param::OMEGA_NOISE_SIGMA * st.rng.randn_fast();
  raw.omega.y() += param::OMEGA_NOISE_SIGMA * st.rng.randn_fast();
  raw.omega.z() += param::OMEGA_NOISE_SIGMA * st.rng.randn_fast();

  for (uint8_t i=0; i<20; ++i) {raw.arm_q[i] += st.q_bias[i] + param::ARM_NOISE_SIGMA  * st.rng.randn_fast();}
}

// static inline void apply_time_delay(State& st,

// )

} // namespace NOISE

inline std::filesystem::path get_executable_path() {
#if defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string buf(size, '\0');
  if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
  return std::filesystem::weakly_canonical(buf.c_str());
#elif defined(__linux__)
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return {};
  buf[n] = '\0';
  return std::filesystem::weakly_canonical(buf);
#else
  return {};
#endif
}