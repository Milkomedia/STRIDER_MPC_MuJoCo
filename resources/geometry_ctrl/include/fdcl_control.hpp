#ifndef FDCL_CONTROL_HPP
#define FDCL_CONTROL_HPP

#include "params.hpp"
#include "fdcl_common_types.hpp"
#include "fdcl_integral_utils.hpp"
#include "fdcl_matrix_utils.hpp"

#include <Eigen/Dense>

namespace fdcl {

class control
{
public:
  double dt = param::CTRL_DT;  /**< Time step size in seconds */

  // for integral controller
  fdcl::integral_error_vec3 eIR; /**< Attitude integral error */
  fdcl::integral_error_vec3 eIX; /**< Position integral error */

  Vector3 eR = Vector3::Zero(); /**< Attitude error */
  Vector3 eW = Vector3::Zero(); /**< Angular rate error */
  Vector3 M = Vector3::Zero();  /**< Control moments */

  Vector3 eX = Vector3::Zero(); /**< Position error */
  Vector3 eV = Vector3::Zero(); /**< Velocity error */

  double f_total = 0.0;  /**< Total propeller thrust */

  Matrix3 Rddot = Matrix3::Zero();
  Matrix3 Rdddot = Matrix3::Zero();
   
  control(
    fdcl::state_t *&state_, /**< Pointer to the current states */
    fdcl::command_t *&command_ /**< Pointer to the desired states */
  );
  control(void); // Default constructor (this is should not be used)
  ~control(void);
  
  void load_config(void);
  void position_control(void);
  Vector3 attitude_control(const Eigen::Matrix3d& Rd, const Eigen::Vector3d& Wd, const Eigen::Vector3d& Wd_dot);
  void integral_reset();

private:
  const double eX_norm_max_ = 1.0; /**< Saturating term of position error */
  const double eR_norm_max_ = 40.0 * M_PI / 180.0; /**< Saturating term of attitude error */

  fdcl::state_t *state = nullptr; /**< Pointer to the current states */
  fdcl::command_t *command = nullptr; /**< Pointer to the desired states */

  Vector3 e3; /**< Direction of the third axis of the fixed frame */

  double m = 0.0;  /**< Mass of the rover (kg) */
  double g = 0.0;  /**< Gravitational acceleration (m/s^2) */

  // Attitude gains
  Matrix3 kR = Matrix3::Zero();  /**< Attitude gains */
  Matrix3 kW = Matrix3::Zero();  /**< Angular rate gains */

  // Position gains
  Matrix3 kX = Matrix3::Zero(); /**< Position gains */
  Matrix3 kV = Matrix3::Zero(); /**< Velocity gains */

  // Integral gains
  double kI = 0.0;  /**< Attitude integral gain for roll and pitch */
  double kIX = 0.0;  /**< Position integral gains */
};

}
#endif