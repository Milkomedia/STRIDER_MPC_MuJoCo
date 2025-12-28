#include "params.hpp"

#include <filesystem>
#include <string>

#if defined(__APPLE__)
  #include <mach-o/dyld.h>
#elif defined(__linux__)
  #include <unistd.h>
#endif

struct SimStae { // use for copy snapshot in thread lock
  double qpos_xyz[3];
  double quat_wxyz[4];
  double qvel_lin[3];
  double qvel_ang[3];
  double qacc_lin[3];
  double arm_q[20];
};

static inline Eigen::Matrix<double, 3 * param::N_STEPS, 1> fig8_traj(double t0_sec){
  // Gerono lemniscate trajectory: x = A sin(wt), y = A sin(wt)cos(wt) = 0.5A sin(2wt)
  Eigen::Matrix<double, 3*param::N_STEPS, 1> out;
  out.setZero();
  
  for (int k = 0; k < param::N_STEPS; ++k) {
    const double t = t0_sec + static_cast<double>(k)*param::MPC_MODEL_DT_D;
    const double s = std::sin(param::FREQ_RAD_S * t);
    const double c = std::cos(param::FREQ_RAD_S * t);
    out.template segment<3>(3 * k) << param::TRAJ_AX * s, param::TRAJ_AY * s*c, param::TRAJ_Z;
  }
  return out;
}

static inline Eigen::Vector3d fig8_point(double t_sec){
  // Gerono lemniscate trajectory: x = A sin(wt), y = A sin(wt)cos(wt) = 0.5A sin(2wt)
  const double s = std::sin(param::FREQ_RAD_S * t_sec);
  const double c = std::cos(param::FREQ_RAD_S * t_sec);
  return Eigen::Vector3d(param::TRAJ_AX * s, param::TRAJ_AY * s*c, param::TRAJ_Z);
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

static inline void IK(const Eigen::Vector3d& bPcot, const Eigen::Matrix3d& cotRb, const Eigen::Vector4d& th_tvc, const double l, double q[20]) {
  std::array<Eigen::Vector3d,4> bodyParm;
  const Eigen::Vector3d e1 = cotRb.col(0);
  const Eigen::Vector3d e2 = cotRb.col(1);
  bodyParm[0] = bPcot + (0.5 * l) * ( e1 + e2);
  bodyParm[1] = bPcot + (0.5 * l) * (-e1 + e2);
  bodyParm[2] = bPcot + (0.5 * l) * (-e1 - e2);
  bodyParm[3] = bPcot + (0.5 * l) * ( e1 - e2);

  std::array<Eigen::Vector3d,4> bodyE3arm;
  const Eigen::Matrix3d bRcot = cotRb.transpose();
  const double s1 = std::sin(th_tvc(0)); const double c1 = std::cos(th_tvc(0));
  const double s2 = std::sin(th_tvc(1)); const double c2 = std::cos(th_tvc(1));
  const double s3 = std::sin(th_tvc(2)); const double c3 = std::cos(th_tvc(2));
  const double s4 = std::sin(th_tvc(3)); const double c4 = std::cos(th_tvc(3));
  bodyE3arm[0] = bRcot * Eigen::Vector3d( s1*M_SQRT1_2, -s1*M_SQRT1_2,  c1);
  bodyE3arm[1] = bRcot * Eigen::Vector3d( s2*M_SQRT1_2,  s2*M_SQRT1_2,  c2);
  bodyE3arm[2] = bRcot * Eigen::Vector3d(-s3*M_SQRT1_2,  s3*M_SQRT1_2,  c3);
  bodyE3arm[3] = bRcot * Eigen::Vector3d(-s4*M_SQRT1_2, -s4*M_SQRT1_2,  c4);

  for (uint8_t i = 0; i < 4; ++i) {
    const double s = std::sin(param::B2BASE_THETA[i]);
    const double c = std::cos(param::B2BASE_THETA[i]);
    const double a = param::B2BASE_A[i];

    const double px = bodyParm[i].x();
    const double py = bodyParm[i].y();
    const double ezx = bodyE3arm[i].x();
    const double ezy = bodyE3arm[i].y();

    const double x_base =  c * px + s * py - a;
    const double y_base = -s * px + c * py;
    const double ex_base =  c * ezx + s * ezy;
    const double ey_base = -s * ezx + c * ezy;

    Eigen::Vector3d baseParm (x_base,  y_base,  bodyParm[i].z());
    Eigen::Vector3d baseE3arm(ex_base, ey_base, bodyE3arm[i].z());

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

static inline void FK(const double q[20], Eigen::Vector3d& bpcot) {
  bpcot = Eigen::Vector3d::Zero();
  // returns {b}->{cot} position and z-directional heading vector
  for (uint8_t i = 0; i < 4; ++i) {
    Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
    T_i *= compute_DH(param::B2BASE_A[i], 0.0, 0.0, param::B2BASE_THETA[i]);
    for (int j = 0; j < 5; ++j){
      T_i *= compute_DH(param::DH_ARM_A[j], param::DH_ARM_ALPHA[j], 0.0, q[5*i + j]);
    }
    bpcot += T_i.block<3, 1>(0, 3);
  }
  bpcot *= 0.25;
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