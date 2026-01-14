#include "fdcl_control.hpp"

fdcl::control::control(
  fdcl::state_t *&state_,
  fdcl::command_t *&command_) : 
  state(state_), command(command_)
{
  // init uninitialized parameters
  e1 << 1.0, 0.0, 0.0;
  e2 << 0.0, 1.0, 0.0;
  e3 << 0.0, 0.0, 1.0;

  state->J << param::J[0], param::J[1], param::J[2], 
             param::J[3], param::J[4], param::J[5], 
             param::J[6], param::J[7], param::J[8];

  // load parameters from the config file
  fdcl::control::load_config();

  command_->xd << 0.0, 0.0, 0.0;
  command_->xd_dot.setZero();
  command_->xd_2dot.setZero();
  command_->xd_3dot.setZero();
  command_->xd_4dot.setZero();

  command_->b1d << 1.0, 0.0, 0.0;
  command_->b1d_dot.setZero();
  command_->b1d_ddot.setZero();
};

fdcl::control::control(void){
  // This should not happen as this leads to uninitialized pointers.
  std::cout << "CTRL: control class is initiated without required parameters."
            << "\n\tThis might lead to undefined behavoirs."
            << "\n\tCalling the destructor .."
            << std::endl;
    this->~control();
}

fdcl::control::~control(void){};

void fdcl::control::position_control(void){
  // translational error functions
  eX = state->x - command->xd;     // position error - eq (11)
  eV = state->v - command->xd_dot; // velocity error - eq (12)

  // if norm(eX) exceeds limit, scale it back
  double eX_norm = eX.norm();
  if(eX_norm > eX_norm_max_) {eX = eX * (eX_norm_max_ / eX_norm);}

  // position integral terms
  eIX.integrate(eX + eV, dt); // eq (13)
  double sat_sigma = 20.0/kIX; // 20N saturation
  saturate(eIX.error, -sat_sigma, sat_sigma);
  eIX.error(0)=0.0;
  eIX.error(1)=0.0;

  // force 'f' along negative b3-axis - eq (14)
  // this term equals to R.e3
  Vector3 A = -kX*eX - kV*eV - kIX*eIX.error - m*g*e3 + m*command->xd_2dot;

  Vector3 b3 = state->R * e3;
  Vector3 b3_dot = state->R * hat(state->W) * e3; // eq (22)
  f_total = -A.dot(b3);

  // intermediate terms for rotational errors
  Vector3 ea = g * e3 - f_total / m * b3 - command->xd_2dot;
  Vector3 A_dot = -kX * eV - kV * ea + m * command->xd_3dot;

  double fdot = -A_dot.dot(b3) - A.dot(b3_dot);
  Vector3 eb = -fdot / m * b3 - f_total / m * b3_dot - command->xd_3dot;
  Vector3 A_ddot = -kX * ea - kV * eb + m * command->xd_4dot;

  Vector3 b3c, b3c_dot, b3c_ddot;
  deriv_unit_vector(-A, -A_dot, -A_ddot, b3c, b3c_dot, b3c_ddot);

  Vector3 A2 = -hat(command->b1d) * b3c;
  Vector3 A2_dot = -hat(command->b1d_dot) * b3c - hat(command->b1d) * b3c_dot;
  Vector3 A2_ddot = -hat(command->b1d_ddot)*b3c - 2.0*hat(command->b1d_dot)*b3c_dot - hat(command->b1d)*b3c_ddot;

  Vector3 b2c, b2c_dot, b2c_ddot;
  deriv_unit_vector(A2, A2_dot, A2_ddot, b2c, b2c_dot, b2c_ddot);

  Vector3 b1c = hat(b2c) * b3c;
  Vector3 b1c_dot = hat(b2c_dot) * b3c + hat(b2c) * b3c_dot;
  Vector3 b1c_ddot = hat(b2c_ddot)*b3c + 2.0*hat(b2c_dot)*b3c_dot + hat(b2c)*b3c_ddot;

  Matrix3 Rddot, Rdddot;

  command->Rd << b1c, b2c, b3c;
  Rddot << b1c_dot, b2c_dot, b3c_dot;
  Rdddot << b1c_ddot, b2c_ddot, b3c_ddot;

  command->Wd = vee(command->Rd.transpose()*Rddot);
  command->Wd_dot = vee(command->Rd.transpose()*Rdddot - hat(command->Wd)*hat(command->Wd));

  // roll / pitch
  command->b3d = b3c;
  command->b3d_dot = b3c_dot;
  command->b3d_ddot = b3c_ddot;

  // yaw
  command->b1c = b1c;
  command->wc3 = e3.dot(state->R.transpose() * command->Rd * command->Wd);
  command->wc3_dot = (e3).dot(state->R.transpose()*command->Rd * command->Wd_dot) - e3.dot(hat(state->W)*state->R.transpose()*command->Rd*command->Wd);
}

Vector3 fdcl::control::attitude_control(const Eigen::Matrix3d& R_d){
  command->Rd = R_d; // use MRG calculated R_d

  Matrix3 RdtR = command->Rd.transpose() * state->R;
  eR = 0.5 * vee(RdtR - RdtR.transpose());

  // if norm(eR) exceeds limit, scale it back
  double eR_norm = eR.norm();
  if(eR_norm > eR_norm_max_) {eR = eR * (eR_norm_max_ / eR_norm);}

  eW = state->W - state->R.transpose() * command->Rd * command->Wd;

  eIR.integrate(eW + eR, dt);

  M = - kR * eR \
      - kW * eW \
      - kI * eIR.error \
      + hat(state->R.transpose() * command->Rd * command->Wd) * state->J * \
            state->R.transpose() * command->Rd * command->Wd \
      + state->J * state->R.transpose() * command->Rd * command->Wd_dot;

  return M;
}

void fdcl::control::integral_reset(){
  eIR.set_zero();
  eI1.set_zero();
  eI2.set_zero();
  eIy.set_zero();
  eIX.set_zero();
}

void fdcl::control::load_config(void){

  kX.setZero();
  kX(0,0) = param::kX[0];
  kX(1,1) = param::kX[1];
  kX(2,2) = param::kX[2];
  kV.setZero();
  kV(0,0) = param::kV[0];
  kV(1,1) = param::kV[1];
  kV(2,2) = param::kV[2];
  kR.setZero();
  kR(0,0) = param::kR[0];
  kR(1,1) = param::kR[1];
  kR(2,2) = param::kR[2];
  kW.setZero();
  kW(0,0) = param::kW[0];
  kW(1,1) = param::kW[1];
  kW(2,2) = param::kW[2];

  kIX = param::kIX;
  kI  = param::kI;
  kyI = param::kyI;

  this->m = param::M;
  this->g = param::G;
}