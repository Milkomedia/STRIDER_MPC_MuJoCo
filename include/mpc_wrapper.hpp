#ifndef MPC_WRAPPER_H
#define MPC_WRAPPER_H

#include "params.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace strider_mpc {

// MPC Input data
struct MPCInput {
  Eigen::Matrix<double, param::MPC_NX, 1> x_0;
  Eigen::Matrix<double, param::MPC_NU, 1> u_0;
  Eigen::Matrix<double, param::MPC_NP, 1> p;
  bool use_arm = false;
  bool use_delta = false;
  uint16_t steps_req = param::N_STEPS_REQ;
  std::chrono::steady_clock::time_point t;
  uint32_t key = 0;
  uint32_t epoch = 0; // session/epoch for ON/OFF safety
  bool has = false;
};

// MPC Output data
struct MPCOutput {
  Eigen::Matrix<double, param::MPC_NX, param::N_STEPS_REQ> x_stage;
  Eigen::Matrix<double, param::MPC_NU, param::N_STEPS_REQ> u_stage;
  double solve_ms = 0.0;
  std::uint8_t state = 255;
  std::chrono::steady_clock::time_point t;
  uint32_t key = 0;
  uint32_t epoch = 0; // must match input epoch
  bool has = false;
};

class acados_wrapper {
public:
  acados_wrapper();
  ~acados_wrapper();
  acados_wrapper(acados_wrapper&&) noexcept;
  acados_wrapper& operator=(acados_wrapper&&) noexcept;

  MPCOutput compute(const MPCInput& in);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace strider_mpc

#endif // MPC_WRAPPER_H