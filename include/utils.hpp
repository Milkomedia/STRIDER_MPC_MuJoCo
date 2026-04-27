#include "params.hpp"

#include <filesystem>
#include <string>

#if defined(__APPLE__)
  #include <mach-o/dyld.h>
#elif defined(__linux__)
  #include <unistd.h>
#endif

enum class Phase : uint8_t {
  READY          = 0,  // program started
  ARMED          = 1,  // all sanity checked
  IDLE           = 2,  // all propellers are idling
  RISING         = 3,  // propeller thrust increasing
  GAC_ONLY       = 4,  // flight with only geometry controller
  USE_DTHETA     = 5,  // flight with delta-theta filtering
  USE_ARM        = 6,  // flight with arm-moving
  USE_FULL       = 7,  // flight with arm-moving and delta-theta filtering
  KILLED         = 99, // killed; (It's not used as a trigger, just a state representation)
};

struct State {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();       // current linear position [m]
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();       // current linear velocity [m/s]
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();       // current linear acceleration [m/s^2]
  Eigen::Matrix3d R   = Eigen::Matrix3d::Identity();   // current Rotation matrix [SO3]
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();     // current angular velocity [rad/s]
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero();     // current angular acceleration [rad/s^2] (not used, just logging)
  Eigen::Vector3d d_hat = Eigen::Vector3d::Zero();     // estimated torque disturbance [N.m]
  Eigen::Vector3d r_cot = Eigen::Vector3d::Zero();     // current b_p_Cot position [m]
  Eigen::Vector3d r1  = Eigen::Vector3d::Zero();       // current rotor1 position [m]
  Eigen::Vector3d r2  = Eigen::Vector3d::Zero();       // current rotor2 position [m]
  Eigen::Vector3d r3  = Eigen::Vector3d::Zero();       // current rotor3 position [m]
  Eigen::Vector3d r4  = Eigen::Vector3d::Zero();       // current rotor4 position [m]
  double arm_q[20]    = {0.0};                         // current joint angle [rad]
  Eigen::Vector3d r_com = Eigen::Vector3d::Zero();     // current estimated CoM position [m] 
};

struct Command {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();        // desired linear position [m]
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();        // desired linear velocity [m/s]
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();        // desired linear acceleration [m/s^2]
  Eigen::Vector3d heading = Eigen::Vector3d(1,0,0);     // desired heading vector [unit vector]
  // This can only be changed by Control Allocation
  double tauz_bar  = 0.0;                               // current yaw thrust torque [N.m] (Sequential control allocation)
  // These can only be changed by MRG
  Eigen::Vector3d d_theta = Eigen::Vector3d::Zero();    // desired delta theta [rad]
  Eigen::Vector2d r1 = Eigen::Vector2d::Zero();         // desired rotor1 polar position [m, rad], z-element is not updated
  Eigen::Vector2d r2 = Eigen::Vector2d::Zero();         // desired rotor2 polar position [m, rad], z-element is not updated
  Eigen::Vector2d r3 = Eigen::Vector2d::Zero();         // desired rotor3 polar position [m, rad], z-element is not updated
  Eigen::Vector2d r4 = Eigen::Vector2d::Zero();         // desired rotor4 polar position [m, rad], z-element is not updated
};

static inline Eigen::Vector3d dob_update(const Eigen::Vector3d& rpy, const Eigen::Vector3d& tau_cmd, Eigen::Matrix<double, 3, 6>& state) {
  constexpr double fc = 0.5;        // [Hz]
  constexpr double d_hat_lim = 3.0; // [Nm]

  constexpr int X1 = 0;
  constexpr int X2 = 1;
  constexpr int X3 = 2;
  constexpr int Y1 = 3;
  constexpr int Y2 = 4;
  constexpr int Y3 = 5;

  constexpr double wc = 2.0 * M_PI * fc;
  constexpr double wc2 = wc * wc;
  constexpr double wc3 = wc2 * wc;
  constexpr double a1 = -2.0 * wc;
  constexpr double a2 = -2.0 * wc2;
  constexpr double a3 = -wc3;
  constexpr int J_IDX[3] = {0, 4, 8};

  const double dt = param::CTRL_DT;

  Eigen::Vector3d d_hat = Eigen::Vector3d::Zero();

  for (int i = 0; i < 3; ++i) {
    double& x1 = state(i, X1);
    double& x2 = state(i, X2);
    double& x3 = state(i, X3);
    double& y1 = state(i, Y1);
    double& y2 = state(i, Y2);
    double& y3 = state(i, Y3);

    const double x1_prev = x1;
    const double x2_prev = x2;
    const double x3_prev = x3;
    const double y1_prev = y1;
    const double y2_prev = y2;
    const double y3_prev = y3;

    // Angle-driven filter
    const double dx1 = a1 * x1_prev + a2 * x2_prev + a3 * x3_prev + rpy(i);
    const double dx2 = x1_prev;
    const double dx3 = x2_prev;

    // Torque-driven filter
    const double dy1 = a1 * y1_prev + a2 * y2_prev + a3 * y3_prev + tau_cmd(i);
    const double dy2 = y1_prev;
    const double dy3 = y2_prev;

    x1 = x1_prev + dt * dx1;
    x2 = x2_prev + dt * dx2;
    x3 = x3_prev + dt * dx3;

    y1 = y1_prev + dt * dy1;
    y2 = y2_prev + dt * dy2;
    y3 = y3_prev + dt * dy3;

    const double tau_hat = param::J[J_IDX[i]] * wc3 * x1;
    const double q_tau   = wc3 * y3;
    const double d_i     = tau_hat - q_tau;

    d_hat(i) = std::clamp(d_i, -d_hat_lim, d_hat_lim);
  }

  return d_hat;
}

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

static inline void l_traj_pva(double t_sec, Eigen::Vector3d& p_d, Eigen::Vector3d& v_d, Eigen::Vector3d& a_d) {
  constexpr double lx_ = 1.2;              // width in X [m]
  constexpr double ly_ = 0.0;              // width in Y [m]
  constexpr double T_  = 2.5;             // base period [sec]
  constexpr double f   = 2.0 * M_PI / T_;  // [rad/s]

  double tau = std::fmod(t_sec, 6.0 * T_);
  if (tau < 0.0) tau += 6.0 * T_;

  // 0 ~ 0.5T : move from (-lx, -ly) to (+lx, +ly)
  if (tau >= 0.0 && tau <= 0.5 * T_) {
    const double u  = tau;
    const double s  = std::sin(f * u);
    const double c  = std::cos(f * u);

    p_d = Eigen::Vector3d(-lx_ * c, -ly_ * c, -1.3);
    v_d = Eigen::Vector3d( lx_ * f * s,  ly_ * f * s, 0.0);
    a_d = Eigen::Vector3d( lx_ * f * f * c, ly_ * f * f * c, 0.0);
    return;
  }

  // 0.5T ~ 3.0T : hold at (+lx, +ly)
  if (tau > 0.5 * T_ && tau <= 3.0 * T_) {
    p_d = Eigen::Vector3d(lx_, ly_, -1.3);
    v_d = Eigen::Vector3d::Zero();
    a_d = Eigen::Vector3d::Zero();
    return;
  }

  // 3.0T ~ 3.5T : move from (+lx, +ly) to (-lx, -ly)
  if (tau > 3.0 * T_ && tau <= 3.5 * T_) {
    const double u  = tau - 3.0 * T_;
    const double s  = std::sin(f * u);
    const double c  = std::cos(f * u);

    p_d = Eigen::Vector3d(lx_ * c, ly_ * c, -1.3);
    v_d = Eigen::Vector3d(-lx_ * f * s, -ly_ * f * s, 0.0);
    a_d = Eigen::Vector3d(-lx_ * f * f * c, -ly_ * f * f * c, 0.0);
    return;
  }

  // 3.5T ~ 6.0T : hold at (-lx, -ly)
  p_d = Eigen::Vector3d(-lx_, -ly_, -1.3);
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
    std::fprintf(stderr, "[R_to_rpy] near gimbal lock: cth2=%f => forcing phi=0.\n", cth2); std::fflush(stderr);
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

static inline Eigen::Vector3d diff(const Eigen::Vector3d& x_cur, const Eigen::Vector3d& x_prev, const double& t_cur, const double& t_prev) {
  constexpr double MinDt_ = 0.0005;   // 0.5 ms  -> max 2000 Hz
  constexpr double MaxDt_ = 0.005;    // 5.0 ms  -> min 200 Hz

  if (t_cur <= t_prev) return Eigen::Vector3d::Zero();

  double dt = t_cur - t_prev;
  dt = std::max(dt, MinDt_);
  if (dt > MaxDt_) return Eigen::Vector3d::Zero();

  const double inv_dt = 1.0 / dt;
  return (x_cur - x_prev) * inv_dt;
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

static inline void IK(const Eigen::Vector2d& r1, const Eigen::Vector2d& r2, const Eigen::Vector2d& r3, const Eigen::Vector2d& r4, const Eigen::Vector4d& th_tvc, double q[20]) {
  const std::array<Eigen::Vector2d, 4> polar_pos{r1, r2, r3, r4};
  std::array<Eigen::Vector3d, 4> bodyE3arm;
  const double s1 = std::sin(th_tvc(0)); const double c1 = std::cos(th_tvc(0));
  const double s2 = std::sin(th_tvc(1)); const double c2 = std::cos(th_tvc(1));
  const double s3 = std::sin(th_tvc(2)); const double c3 = std::cos(th_tvc(2));
  const double s4 = std::sin(th_tvc(3)); const double c4 = std::cos(th_tvc(3));
  bodyE3arm[0] = Eigen::Vector3d( s1*M_SQRT1_2,  s1*M_SQRT1_2, -c1);
  bodyE3arm[1] = Eigen::Vector3d( s2*M_SQRT1_2, -s2*M_SQRT1_2, -c2);
  bodyE3arm[2] = Eigen::Vector3d(-s3*M_SQRT1_2, -s3*M_SQRT1_2, -c3);
  bodyE3arm[3] = Eigen::Vector3d(-s4*M_SQRT1_2,  s4*M_SQRT1_2, -c4);
  
  for (uint8_t i = 0; i < 4; ++i) {
    const double s = std::sin(param::B2BASE_THETA[i]);
    const double c = std::cos(param::B2BASE_THETA[i]);
    const double a = param::B2BASE_A[i];

    const double rho = polar_pos[i].x();
    const double alpha = polar_pos[i].y();
    const double px = param::B2BASE_X[i] + rho * std::cos(alpha);
    const double py = param::B2BASE_Y[i] + rho * std::sin(alpha);
    const double ezx = bodyE3arm[i].x();
    const double ezy = bodyE3arm[i].y();

    const double x_base =  c * px  + s * py - a;
    const double y_base =  s * px  - c * py;
    const double ex_base = c * ezx + s * ezy;
    const double ey_base = s * ezx - c * ezy;

    Eigen::Vector3d baseParm(x_base, y_base, -param::r_z_position);
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

static inline void FK(const double q[20], const double& load_angle, Eigen::Vector3d& bpcot, Eigen::Vector3d& bpc, Eigen::Vector3d& r1, Eigen::Vector3d& r2, Eigen::Vector3d& r3, Eigen::Vector3d& r4) {
  std::array<Eigen::Vector3d*, 4> bparm = {&r1, &r2, &r3, &r4};
  Eigen::Vector3d mass_weighted_sum = Eigen::Vector3d::Zero();
  bpc = Eigen::Vector3d::Zero();
  bpcot = Eigen::Vector3d::Zero();

  for (uint8_t i = 0; i < 4; ++i) {
    Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
    T_i *= compute_DH(param::B2BASE_A[i], param::B2BASE_ALPHA[i], 0.0, param::B2BASE_THETA[i]);

    for (int j = 0; j < 5; ++j) {
      T_i *= compute_DH(param::DH_ARM_A[j], param::DH_ARM_ALPHA[j], 0.0, q[5*i+j]);
      mass_weighted_sum += param::LINK_MASS[j] * (T_i.block<3,1>(0,3) + param::LINK_COM_DIST[j] * T_i.block<3,1>(0,0));
    }
    *(bparm[i]) = T_i.block<3,1>(0,3);
    bpcot += *(bparm[i]);
  }
  bpcot *= 0.25;
  bpc = mass_weighted_sum / param::TOTAL_MASS;
}

static inline double spin_360(double a, double amin, double amax) {
  constexpr double two_pi = 2.0 * M_PI;

  double best = a;
  double best_err = 1e100;

  for (int k = -2; k <= 2; ++k) {
    const double ak = a + static_cast<double>(k) * two_pi;
    double err = 0.0;
    if (ak < amin) err = amin - ak;
    else if (ak > amax) err = ak - amax;

    if (err < best_err) {
      best_err = err;
      best = ak;
    }
    if (err <= 1e-12) break; // already inside
  }
  return best;
}

static inline void cart2polar(const Eigen::Vector3d& r1, const Eigen::Vector3d& r2, const Eigen::Vector3d& r3, const Eigen::Vector3d& r4, Eigen::Vector2d& p1, Eigen::Vector2d& p2, Eigen::Vector2d& p3, Eigen::Vector2d& p4) {
  constexpr double eps = 1e-12;

  const Eigen::Vector3d* cart[4] = {&r1, &r2, &r3, &r4};
  Eigen::Vector2d* polar[4] = {&p1, &p2, &p3, &p4};

  for (int a = 0; a < 4; ++a) {
    const double dx = (*cart[a])(0) - param::B2BASE_X[a];
    const double dy = (*cart[a])(1) - param::B2BASE_Y[a];

    const double rho = std::sqrt(dx * dx + dy * dy);
    double alpha = (rho > eps) ? std::atan2(dy, dx) : 0.0;

    // Make alpha compatible with your box bounds which may exceed [-pi, pi].
    alpha = spin_360(alpha, param::ALPHA_MIN[a], param::ALPHA_MAX[a]);

    (*polar[a])(0) = rho;
    (*polar[a])(1) = alpha;
  }
}

static inline void polar2cart(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3, const Eigen::Vector2d& p4, Eigen::Vector2d& r1, Eigen::Vector2d& r2, Eigen::Vector2d& r3, Eigen::Vector2d& r4) {
  const Eigen::Vector2d* polar[4] = {&p1, &p2, &p3, &p4};
  Eigen::Vector2d* cart[4] = {&r1, &r2, &r3, &r4};

  for (int a = 0; a < 4; ++a) {
    const double rho   = (*polar[a])(0);
    const double alpha = (*polar[a])(1);
    (*cart[a])(0) = param::B2BASE_X[a] + rho * std::cos(alpha);
    (*cart[a])(1) = param::B2BASE_Y[a] + rho * std::sin(alpha);
  }
}

static inline bool make_feasible(std::array<Eigen::Vector2d, 4>& r) {
  constexpr double sq_min_stretch_fail    = (param::MIN_STRETCH - param::STRETCH_FAIL_MARGIN) * (param::MIN_STRETCH - param::STRETCH_FAIL_MARGIN);
  constexpr double sq_max_stretch_fail    = (param::MAX_STRETCH + param::STRETCH_FAIL_MARGIN) * (param::MAX_STRETCH + param::STRETCH_FAIL_MARGIN);
  constexpr double sq_rotor_diameter_fail = (param::ROTOR_DIAMETER - param::COLLISION_FAIL_MARGIN) * (param::ROTOR_DIAMETER - param::COLLISION_FAIL_MARGIN);
  constexpr double sq_max_stretch         = param::MAX_STRETCH * param::MAX_STRETCH;
  constexpr double sq_min_stretch         = param::MIN_STRETCH * param::MIN_STRETCH;
  constexpr double sq_rotor_diameter      = param::ROTOR_DIAMETER * param::ROTOR_DIAMETER;

  constexpr int nbr_a[4] = {3, 0, 1, 2}; // nearby rotor a index
  constexpr int nbr_b[4] = {1, 2, 3, 0}; // nearby rotor b index

  constexpr double PI = 3.14159265358979323846;

  auto wrap_pi = [&](double a) -> double {
    while (a >  PI) {a -= 2.0 * PI;}
    while (a < -PI) {a += 2.0 * PI;}
    return a;
  };

  auto clamp_angle_in_sector = [&](double a, int i) -> double {
    a = spin_360(a, param::ALPHA_MIN[i], param::ALPHA_MAX[i]);
    if (a < param::ALPHA_MIN[i]) a = param::ALPHA_MIN[i];
    if (a > param::ALPHA_MAX[i]) a = param::ALPHA_MAX[i];
    return a;
  };

  auto pol_to_cart = [&](double rho, double alpha, int i) -> Eigen::Vector2d {
    return Eigen::Vector2d(
      param::B2BASE_X[i] + rho * std::cos(alpha),
      param::B2BASE_Y[i] + rho * std::sin(alpha)
    );
  };

  auto cart_to_pol_from_base = [&](const Eigen::Vector2d& p, int i) -> Eigen::Vector2d {
    const double dx = p.x() - param::B2BASE_X[i];
    const double dy = p.y() - param::B2BASE_Y[i];
    const double rho = std::sqrt(dx * dx + dy * dy);
    const double alpha = std::atan2(dy, dx);
    return Eigen::Vector2d(rho, wrap_pi(alpha));
  };

  for (int it = 0; it < 3; ++it) { // A few iterations to resolve "workspace <-> collision" coupling
    for (int i = 0; i < 4; ++i) { // per each rotor
      { // ------------ [ 1. Workspace guard ] ------------
        double rho   = r[i].x();
        double alpha = r[i].y();

        // Angle clamp (replacing old sign clamp)
        alpha = clamp_angle_in_sector(alpha, i);

        // Hard-fail if stretch is too far outside
        const double sqd_r = rho * rho;
        if (sqd_r < sq_min_stretch_fail || sqd_r > sq_max_stretch_fail) {
          std::fprintf(stderr, "[arm-cmd check] WARNING-iter[%d]: stretch too far.\n", it);
          std::fflush(stderr);
          return false;
        }

        if (sqd_r < sq_min_stretch || sqd_r > sq_max_stretch) {
          const double target = (sqd_r < sq_min_stretch) ? param::MIN_STRETCH : param::MAX_STRETCH;
          const double scale_factor = target / rho;
          rho *= scale_factor;
          std::fprintf(stderr, "[arm-cmd check] iter[%d]: rotor[%d] scaled by %f\n", it, i + 1, scale_factor);
          std::fflush(stderr);
        }

        r[i].x() = rho;
        r[i].y() = alpha;
      }

      { // ------------ [ 2. Rotor collision guard ] ------------
        const Eigen::Vector2d p0 = pol_to_cart(r[i].x(), r[i].y(), i);
        const Eigen::Vector2d pa = pol_to_cart(r[nbr_a[i]].x(), r[nbr_a[i]].y(), nbr_a[i]);
        const Eigen::Vector2d pb = pol_to_cart(r[nbr_b[i]].x(), r[nbr_b[i]].y(), nbr_b[i]);
        Eigen::Vector2d p = p0; // new target xy to compute

        const Eigen::Vector2d ap0 = p0 - pa;
        const Eigen::Vector2d bp0 = p0 - pb;
        const double sqrd_ap0 = ap0.squaredNorm();
        const double sqrd_bp0 = bp0.squaredNorm();

        // (1) collision occurred too deep -> fail
        if (sqrd_ap0 < sq_rotor_diameter_fail || sqrd_bp0 < sq_rotor_diameter_fail) {
          std::fprintf(stderr, "[arm-cmd check] WARNING-iter[%d]: collision occured too deep.\n", it);
          std::fflush(stderr);
          return false;
        }

        // (2) both already safe -> skip
        if (sqrd_ap0 >= sq_rotor_diameter && sqrd_bp0 >= sq_rotor_diameter) {
          continue;
        }

        // (3) collision occurred -> move target p
        const bool closeA = (sqrd_ap0 < sq_rotor_diameter);
        const bool closeB = (sqrd_bp0 < sq_rotor_diameter);
        if (closeA && closeB) { // both are closer than D -> |p - pa| = D, |p - pb| = D  (two solutions)
          const Eigen::Vector2d dc = pb - pa;
          const double d2 = dc.squaredNorm();
          const double d = std::sqrt(d2);

          const Eigen::Vector2d m = 0.5 * (pa + pb);
          double h2 = sq_rotor_diameter - 0.25 * d2;
          if (h2 < 0.0) {
            std::fprintf(stderr, "[arm-cmd check] WARNING-iter[%d]: negative h2 detected.\n", it);
            std::fflush(stderr);
            return false;
          }
          const double h = std::sqrt(h2);
          const Eigen::Vector2d n(-dc.y() / d, dc.x() / d); // unit perpendicular (rotate +90deg)

          // Original code chose one branch only. Keep same style.
          p = m + n * h;
        }
        else { // only one side is closer than D
          const Eigen::Vector2d Close = closeA ? pa : pb;
          const Eigen::Vector2d Far   = closeA ? pb : pa;
          const double r1_keep2 = closeA ? sqrd_bp0 : sqrd_ap0; // keep original distance to the other rotor

          const Eigen::Vector2d dc = Far - Close;
          const double d2 = dc.squaredNorm();
          const double d = std::sqrt(d2);

          const double a = (sq_rotor_diameter + d2 - r1_keep2) / (2.0 * d);
          const double h = std::sqrt(sq_rotor_diameter - a * a);

          const Eigen::Vector2d n(-dc.y() / d, dc.x() / d); // unit perpendicular (rotate +90deg)

          p = Close + a * dc / d + n * h;
        }

        // Commit as polar
        Eigen::Vector2d pol = cart_to_pol_from_base(p, i);

        // Re-apply polar workspace constraint after collision projection
        pol.y() = clamp_angle_in_sector(pol.y(), i);

        const double sqd_r = pol.x() * pol.x();
        if (sqd_r < sq_min_stretch_fail || sqd_r > sq_max_stretch_fail) {
          std::fprintf(stderr, "[arm-cmd check] WARNING-iter[%d]: stretch too far after collision projection.\n", it);
          std::fflush(stderr);
          return false;
        }

        if (sqd_r < sq_min_stretch || sqd_r > sq_max_stretch) {
          const double target = (sqd_r < sq_min_stretch) ? param::MIN_STRETCH : param::MAX_STRETCH;
          const double scale_factor = target / pol.x();
          pol.x() *= scale_factor;
          std::fprintf(stderr, "[arm-cmd check] iter[%d]: rotor[%d] scaled by %f after collision projection\n", it, i + 1, scale_factor);
          std::fflush(stderr);
        }

        r[i] = pol;

        { // NaN/Inf guard
          if (!r[i].allFinite()) {
            std::fprintf(stderr, "[arm-cmd check] WARNING-iter[%d]: r[i]=r[%d] NaN/Inf detected.\n", it, i);
            std::fflush(stderr);
            return false;
          }
          const int ia = nbr_a[i];
          const int ib = nbr_b[i];
          if (!r[ia].allFinite()) {
            std::fprintf(stderr, "[arm-cmd check] WARNING-iter[%d]: r[ia]=r[%d] NaN/Inf detected.\n", it, ia);
            std::fflush(stderr);
            return false;
          }
          if (!r[ib].allFinite()) {
            std::fprintf(stderr, "[arm-cmd check] WARNING-iter[%d]: r[ib]=r[%d] NaN/Inf detected.\n", it, ib);
            std::fflush(stderr);
            return false;
          }
        }

        std::fprintf(stderr, "[arm-cmd check] iter[%d]: collision removed.\n", it);
        std::fflush(stderr);
      }
    } // rotor 1234
  } // iter

  // ------------ [ 3. Final pass check (just check) ] ------------
  for (int i = 0; i < 4; ++i) {
    const double rho   = r[i].x();
    const double alpha = wrap_pi(r[i].y());

    // Angle-sector check
    const double alpha_clamped = clamp_angle_in_sector(alpha, i);
    if (std::abs(wrap_pi(alpha - alpha_clamped)) > 1e-12) {
      std::fprintf(stderr, "[arm-cmd check] WARNING final-pass: rotor-%d angle out of sector.\n", i);
      std::fflush(stderr);
      return false;
    }

    const double n2 = rho * rho;
    if (n2 < sq_min_stretch || n2 > sq_max_stretch) {
      std::fprintf(stderr, "[arm-cmd check] WARNING final-pass: arm-%d too much stretch.\n", i);
      std::fflush(stderr);
      return false;
    }
  }

  return true;
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
  constexpr double l = 0.48 / 2.0;

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

static inline Eigen::Vector3d Forward_Allocate(const Eigen::Vector4d& F1234, const Eigen::Vector3d& r1, const Eigen::Vector3d& r2, const Eigen::Vector3d& r3, const Eigen::Vector3d& r4, const Eigen::Vector3d& Pcot) {
  Eigen::Matrix<double, 3, 4> A;
  A(0,0) = -r1(1) + Pcot(1);
  A(0,1) = -r2(1) + Pcot(1);
  A(0,2) = -r3(1) + Pcot(1);
  A(0,3) = -r4(1) + Pcot(1);
  A(1,0) =  r1(0) - Pcot(0);
  A(1,1) =  r2(0) - Pcot(0);
  A(1,2) =  r3(0) - Pcot(0);
  A(1,3) =  r4(0) - Pcot(0);
  A(2,0) = -param::PWM_ZETA;
  A(2,1) =  param::PWM_ZETA;
  A(2,2) = -param::PWM_ZETA;
  A(2,3) =  param::PWM_ZETA;
  return A * F1234;
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

struct nState {
  std::chrono::steady_clock::time_point last_t;

  Rng rng;

  // Bias states (random-walk)
  Eigen::Vector3d pos_bias   = Eigen::Vector3d::Zero();
  Eigen::Vector3d theta_bias = Eigen::Vector3d::Zero();
  double q_bias[20] = {0.0};
};

// Reset/init the noise state
static inline void reset(nState& st, std::uint64_t seed, const std::chrono::steady_clock::time_point& now) {
  st.last_t = now;

  st.rng.s = (seed == 0) ? 1ull : seed;

  st.pos_bias.setZero();
  st.theta_bias.setZero();
  for (uint8_t i=0; i<20; ++i) {st.q_bias[i] = 0.0;}
}

static inline void apply(nState& st, const std::chrono::steady_clock::time_point& now, State& raw) {

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

} // namespace NOISE

static inline int sensor_adr_check(const mjModel* m, const char* name, int expect_dim) {
  const int sid = mj_name2id(m, mjOBJ_SENSOR, name);
  if (sid < 0) { std::printf("[ERROR] sensor '%s' not found\n", name); std::abort(); }
  if (m->sensor_dim[sid] != expect_dim) {std::printf("[ERROR] sensor '%s' dim mismatch: got %d expect %d\n", name, m->sensor_dim[sid], expect_dim); std::abort();}
  return m->sensor_adr[sid];
}

static inline void set_bong_tip_load_enabled(mjModel* m, mjData* d, int body_id, int geom_id, bool enabled) {
  if (body_id < 0) {return;}

  const mjtNum mass = enabled ? param::BONG_TIP_LOAD_MASS : 1e-9;
  const mjtNum inertia = enabled ? param::BONG_TIP_LOAD_INERTIA : 1e-12;

  m->body_mass[body_id] = mass;
  m->body_inertia[3 * body_id + 0] = inertia;
  m->body_inertia[3 * body_id + 1] = inertia;
  m->body_inertia[3 * body_id + 2] = inertia;

  if (geom_id >= 0) {m->geom_rgba[4 * geom_id + 3] = enabled ? 1.0 : 0.0;}

  // Recompute mjModel derived constants using scratch mjData.
  // Do not pass live mjData here: mj_setConst works around qpos0 and may disturb d->qpos.
  mjData* d_scratch = mj_makeData(m);
  if (d_scratch != nullptr) {
    mj_setConst(m, d_scratch);
    mj_deleteData(d_scratch);
  }

  mj_forward(m, d); // Refresh live mjData using the current qpos/qvel.
}

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