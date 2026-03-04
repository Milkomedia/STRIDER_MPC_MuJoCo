#ifndef COMMON_TYPE_HPP
#define COMMON_TYPE_HPP

#include <Eigen/Dense>

typedef Eigen::Matrix<double, 2, 1> Vector2;
typedef Eigen::Matrix<double, 3, 1> Vector3;
typedef Eigen::Matrix<double, 4, 1> Vector4;
typedef Eigen::Matrix<double, 3, 3> Matrix3;

namespace fdcl{

struct state_t
{
  Vector3 x = Vector3::Zero(); /**< Position */
  Vector3 v = Vector3::Zero(); /**< Velocity */
  Matrix3 R = Matrix3::Identity(); /**< Attitude in SO(3) */
  Vector3 W = Vector3::Zero(); /**< Body angular velocity */
  Matrix3 J = Matrix3::Zero();  /**< Inertia matrix for the rover */
};

class command_t{
public:
  Matrix3 Rd = Matrix3::Identity();
  Vector3 Wd = Vector3::Zero();
  Vector3 Wd_dot = Vector3::Zero();
  Vector3 xd = Vector3::Zero();
  Vector3 xd_dot = Vector3::Zero();
  Vector3 xd_2dot = Vector3::Zero();
  Vector3 b1d = Vector3::Zero();
};
}

#endif