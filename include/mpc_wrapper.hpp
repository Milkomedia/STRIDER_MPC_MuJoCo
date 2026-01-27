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
  Eigen::Matrix<double, param::NX, 1> x_0;
  Eigen::Matrix<double, param::NU, 1> u_0;
  Eigen::Matrix<double, param::NP, 1> p;
  Eigen::Matrix<double, 10, 1> log;
  bool use_cot = false;
  std::chrono::steady_clock::time_point t;
  uint32_t key = 0;
  bool has = false;
};

// MPC Output data
struct MPCOutput {
  Eigen::Matrix<double, param::NU,     1> u_opt;
  Eigen::Matrix<double, param::NU_AUG, 1> u_rate;
  double solve_ms = 0.0;
  std::uint8_t state = 255;
  std::chrono::steady_clock::time_point t;
  uint32_t key = 0;
  bool has = false;
};

class acados_wrapper {
public:
  acados_wrapper();
  ~acados_wrapper();
  acados_wrapper(acados_wrapper&&) noexcept;
  acados_wrapper& operator=(acados_wrapper&&) noexcept;

  MPCOutput compute(const MPCInput& in);
  void print_last_debug();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace strider_mpc

#endif // MPC_WRAPPER_H