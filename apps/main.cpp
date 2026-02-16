#include "mj_viewer_helper.hpp"
#include "mmap_manager.hpp"
#include "mpc_wrapper.hpp"
#include "fdcl_control.hpp"
#include "utils.hpp"

#include <thread>
#include <condition_variable>
#include <csignal>
#include <pybind11/embed.h>

static std::atomic<bool> g_stop{false};
static void sigint_handler(int) {g_stop.store(true);}

std::mutex scene_mtx;
static double g_pos_des[3] = {0.0, 0.0, 0.0};  // desired point for viewer
static double g_pos_cur[3] = {0.0, 0.0, 0.0};  // current point for viewer
static std::atomic<bool> g_mpc_activated{false};

static std::mutex mpc_mtx;
static std::condition_variable mpc_cv;
static strider_mpc::MPCInput g_mpc_input;
static strider_mpc::MPCOutput g_mpc_output;
static std::atomic<uint32_t> g_mpc_epoch{1}; // increments on ON/OFF transitions
static bool g_mpc_busy = false;

static inline void mpc_reset_locked(uint32_t& mpc_key) {
  g_mpc_epoch.fetch_add(1, std::memory_order_relaxed);
  g_mpc_input.has = false;
  g_mpc_output.has = false;
  g_mpc_busy = false;
  mpc_key += 1; // force key++
}

// ===== Main =====
int main() {
  std::signal(SIGINT, sigint_handler); // SIGINT handler(ctrl+C)

  // Load MuJoCo model
  std::string xml_path;
  {
    const std::filesystem::path exe = get_executable_path();
    if (exe.empty()) return 1;
    const std::filesystem::path root = exe.parent_path().parent_path();
    xml_path = (root / "resources" / "mujoco" / "scene.xml").string();
  }

  char error_[1024] = {0};
  mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error_, sizeof(error_));
  mjData*  d = mj_makeData(m);
  m->opt.timestep = param::SIM_DT;

  const int adr_quat = sensor_adr_check(m, "imu_quat",   4);
  const int adr_gyro = sensor_adr_check(m, "imu_gyro",   3);
  const int adr_pos  = sensor_adr_check(m, "imu_pos",    3);
  const int adr_vel  = sensor_adr_check(m, "imu_linvel", 3);
  const int adr_acc  = sensor_adr_check(m, "imu_linacc", 3);

  // ---------------- [ MPC thread ] ----------------
  std::thread th_mpc([&]() {

    pybind11::scoped_interpreter py_guard{false};
    strider_mpc::acados_wrapper mpc; // compile acados before start.
    std::printf("\n\n ||----------------------------------||\n ||       Acados compile done.       ||\n || press the [space] to MPC on/off. ||\n ||----------------------------------||\n\n\n");

    while (!g_stop.load()) {
      strider_mpc::MPCInput in_local;
      // std::printf(" [mpc]->waiting   ");
      {
        std::unique_lock<std::mutex> lk(mpc_mtx);
        mpc_cv.wait(lk, [&]{
          if (g_stop.load()) return true;
          if (!g_mpc_activated.load(std::memory_order_relaxed)) return false;
          return g_mpc_input.has;
        });
        if (g_stop.load()) break;
        if (g_mpc_input.has == true) {
          in_local = g_mpc_input;
          g_mpc_input.has = false;
          g_mpc_busy = true;
        }
      }
      // std::printf(" [mpc]->get   ");

      strider_mpc::MPCOutput out_local;
      try { out_local = mpc.compute(in_local); }
      catch (const std::exception&) { out_local.solve_ms = 0.0; out_local.state = 99; std::printf("OH SHIT");}
      // std::printf("[mpc]->solved:%f  ", out_local.solve_ms);

      {
        std::lock_guard<std::mutex> lk(mpc_mtx);
        g_mpc_output = out_local;
        g_mpc_output.t = in_local.t;
        g_mpc_output.key = in_local.key;
        g_mpc_output.epoch = in_local.epoch;
        g_mpc_output.has = true;
        g_mpc_busy = false;
      }
    }
  });

  // -------------- [ Control thread ] --------------
  std::thread th_ctrl([&]() {
    // --- geometry SO3 controller definition ---
    fdcl::state_t   gac_state;
    fdcl::command_t gac_cmd;
    fdcl::state_t*   gac_state_ptr = &gac_state;
    fdcl::command_t* gac_cmd_ptr   = &gac_cmd;
    fdcl::control geometry_ctrl(gac_state_ptr, gac_cmd_ptr);

    // --- state definition ---
    Phase   phase = Phase::GAC_FLIGHT;
    State   s{};
    Command cmd{};
    cmd.r1 = param::r1_init;
    cmd.r2 = param::r2_init;
    cmd.r3 = param::r3_init;
    cmd.r4 = param::r4_init;
    Eigen::Vector3d prev_omega = Eigen::Vector3d::Zero();
    double prev_elapsed_double = 0.0;

    // --- MRG parameters ---
    uint32_t mpc_key = 1;
    strider_mpc::MPCOutput l_mpc_output;
    l_mpc_output.u_rate.setZero();
    l_mpc_output.u_opt.setZero();
    l_mpc_output.t = std::chrono::steady_clock::time_point::max();
    bool prev_mpc_on = false; // for ON/OFF edge detection
    bool next_use_cot = true;

    // --- noise injection ---
    static NOISE::nState noise_state;
    NOISE::reset(noise_state, param::NOISE_SEED, std::chrono::steady_clock::now());

    // --- timedelay ---
    State delayed_s;
    double delayed_q_d[20] = {0};
    double smoothed_q_d[20] = {0};
    Eigen::Vector4d smoothed_F   = Eigen::Vector4d::Zero();
    Eigen::Vector4d smoothed_Tau = Eigen::Vector4d::Zero();

    // --- logging ---
    mmap_manager::MMapLogger logger("/tmp/strider_log.mmap", /*reset=*/true);
    logger.open();

    // --- time scope definition ---
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point next_tick = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::duration ctrl_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(param::CTRL_DT));
    const double steps_per_ctrl = param::SIM_HZ / param::CTRL_HZ;
    double substep_accum = 0.0;

    { // Model warm-up
      double arm_angles[20]; // Initial arm joint angles
      Eigen::Vector4d tvc_angle = Eigen::Vector4d::Zero();
      IK(cmd.r1, cmd.r2, cmd.r3, cmd.r4, tvc_angle, arm_angles);
      for (uint8_t i=0; i<20; ++i) {delayed_q_d[i] = arm_angles[i]; smoothed_q_d[i] = arm_angles[i];}
      {
        std::lock_guard<std::mutex> scene_lk(scene_mtx);
        // spawn and 3 second do nothing
        for (int k = 0; k < 3*static_cast<int>(param::SIM_HZ); ++k) {
          for (int i = 0; i < 8 && i < m->nu; ++i) {d->ctrl[i] = 0.0;}
          for (int i = 0; i < 20 && (8 + i) < m->nu; ++i) {d->ctrl[8 + i] = arm_angles[i];}
          mj_step(m, d);
        }
      }
    }

    while (!g_stop.load()) {
      // --- time count ---
      const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
      const double elapsed_double = std::chrono::duration<double>(now - t0).count();

      // Handle MPC ON/OFF transitions
      const bool mpc_on = g_mpc_activated.load(std::memory_order_relaxed);
      if (mpc_on != prev_mpc_on) {
        std::lock_guard<std::mutex> lk(mpc_mtx);
        mpc_reset_locked(mpc_key);
        if (mpc_on) {
          if (next_use_cot) {phase = Phase::MRG_ACTIVE_COT; std::printf("MPC->YES-COT\n");}
          else {phase = Phase::MRG_FLIGHT; std::printf("MPC->NO-COT\n");}
          next_use_cot = !next_use_cot;
        }
        else {phase = Phase::GAC_FLIGHT;}
        prev_mpc_on = mpc_on;
      }

      // --- mujoco measurement ---
      {
        std::lock_guard<std::mutex> scene_lk(scene_mtx);
        const mjtNum* sd = d->sensordata;
        s.pos << sd[adr_pos], -sd[adr_pos+1], -sd[adr_pos+2];
        s.vel << sd[adr_vel], -sd[adr_vel+1], -sd[adr_vel+2];
        s.acc << sd[adr_acc], -sd[adr_acc+1], -sd[adr_acc+2];
        const Eigen::Quaterniond q(sd[adr_quat], sd[adr_quat+1], sd[adr_quat+2], sd[adr_quat+3]);
        s.R = quat_to_R(q);
        s.omega << sd[adr_gyro], -sd[adr_gyro + 1], -sd[adr_gyro + 2];
        for (int i = 0; i < 20; ++i) {s.arm_q[i] = d->qpos[7 + i];}
      }
      s.alpha = diff(s.omega, prev_omega, elapsed_double, prev_elapsed_double);
      prev_omega = s.omega; prev_elapsed_double = elapsed_double;

      // --- sensor noise injection ---
      if (param::NOISE_ON) {NOISE::apply(noise_state, now, s);}

      // --- position control ---
      // if (elapsed_double >= 4.0) {cmd.pos = square4_point(elapsed_double);} // option: [fig8_point/square4_point]
      // else {cmd.pos = Eigen::Vector3d(0.0, 0.0, -1.0);}
      
      if (elapsed_double >= 6.0) {l_traj_pva(elapsed_double, cmd.pos, cmd.vel, cmd.acc);} // option: [fig8_point_pva/circle_pva/l_traj_pva]
      else if (elapsed_double <= 2.0) {cmd.pos = goes_to(Eigen::Vector3d(1.0,0.0,-1.0), elapsed_double, 2.0);}
      else {cmd.pos = Eigen::Vector3d(1.0,0.0,-1.0);}

      gac_cmd.xd = cmd.pos;
      gac_cmd.xd_dot = cmd.vel;
      gac_cmd.xd_2dot = cmd.acc;
      gac_cmd.b1d = cmd.heading;
      gac_state.x = delayed_s.pos;
      gac_state.v = delayed_s.vel;
      gac_state.a = delayed_s.acc;
      gac_state.R = delayed_s.R;
      gac_state.W = delayed_s.omega;
      geometry_ctrl.position_control();
      const Eigen::Matrix3d R_raw = gac_cmd.Rd;
      const Eigen::Vector3d omega_raw = gac_cmd.Wd;
      const double F_des = -geometry_ctrl.f_total; // (f_total > 0)

      { // MPC get
        std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
        if (g_mpc_output.has) {
          const bool epoch_ok = (g_mpc_output.epoch == g_mpc_epoch.load(std::memory_order_relaxed));
          const bool key_ok = (g_mpc_output.key == mpc_key);
          const bool solve_ok = (g_mpc_output.state == 0);

          // l_mpc_output updated only when solve succeed.
          if (mpc_on && epoch_ok && key_ok && solve_ok) {l_mpc_output = g_mpc_output;}
          else if (mpc_on && epoch_ok && !key_ok) {mpc_reset_locked(mpc_key);}
          g_mpc_output.has = false;
        }
      }

      if (mpc_on) { // MPC unpack
        const bool epoch_ok = (l_mpc_output.epoch == g_mpc_epoch.load(std::memory_order_relaxed));
        const bool time_ok = ((now - l_mpc_output.t) < param::MPC_TIMEOUT_DURATUION);
        if (epoch_ok && time_ok) {
          const std::size_t idx = static_cast<std::size_t>(std::floor(std::chrono::duration<double>(now - l_mpc_output.t).count() / param::MPC_STEP_DT));
          cmd.d_theta = l_mpc_output.u_opt.col(idx).head<3>();
          cmd.r1(0) = l_mpc_output.u_opt(3, idx); cmd.r1(1) = l_mpc_output.u_opt(7, idx);
          cmd.r2(0) = l_mpc_output.u_opt(4, idx); cmd.r2(1) = l_mpc_output.u_opt(8, idx);
          cmd.r3(0) = l_mpc_output.u_opt(5, idx); cmd.r3(1) = l_mpc_output.u_opt(9, idx);
          cmd.r4(0) = l_mpc_output.u_opt(6, idx); cmd.r4(1) = l_mpc_output.u_opt(10, idx);
          workspace_guard(cmd.r1, cmd.r2, cmd.r3, cmd.r4);
        }
        else { // solve failed timeout
          cmd.d_theta *= param::GOES_2_ZERO_A;
          cmd.r1 = param::GOES_2_ZERO_A*cmd.r1 + param::GOES_2_ZERO_B*param::r1_init;
          cmd.r2 = param::GOES_2_ZERO_A*cmd.r2 + param::GOES_2_ZERO_B*param::r2_init;
          cmd.r3 = param::GOES_2_ZERO_A*cmd.r3 + param::GOES_2_ZERO_B*param::r3_init;
          cmd.r4 = param::GOES_2_ZERO_A*cmd.r4 + param::GOES_2_ZERO_B*param::r4_init;
          l_mpc_output.u_rate.setZero();
        }
      }
      else { // only GAC flight
        cmd.d_theta *= param::GOES_2_ZERO_A;
        cmd.r1 = param::GOES_2_ZERO_A*cmd.r1 + param::GOES_2_ZERO_B*param::r1_init;
        cmd.r2 = param::GOES_2_ZERO_A*cmd.r2 + param::GOES_2_ZERO_B*param::r2_init;
        cmd.r3 = param::GOES_2_ZERO_A*cmd.r3 + param::GOES_2_ZERO_B*param::r3_init;
        cmd.r4 = param::GOES_2_ZERO_A*cmd.r4 + param::GOES_2_ZERO_B*param::r4_init;
        l_mpc_output.u_rate.setZero();
      }

      const Eigen::Vector3d euler_rpy = R_to_rpy(delayed_s.R);
      FK(delayed_s.arm_q, s.r_cot, s.r1, s.r2, s.r3, s.r4);
      s.r_com(0) = param::COM_OFF_X + param::COT_2_COM_X * s.r_cot(0);
      s.r_com(1) = param::COM_OFF_Y + param::COT_2_COM_Y * s.r_cot(1);
      { // MPC send
        if (mpc_on) {
          std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
          if (!g_mpc_busy && !g_mpc_input.has && !g_mpc_output.has) { // push next solve immediately after the previous output
            mpc_key += 1;

            int k = 0; // fill initial state(x)
            g_mpc_input.x_0(k++) = euler_rpy(0); g_mpc_input.x_0(k++) = euler_rpy(1); g_mpc_input.x_0(k++) = euler_rpy(2); // theta(0,1,2)
            g_mpc_input.x_0(k++) = delayed_s.omega(0); g_mpc_input.x_0(k++) = delayed_s.omega(1); g_mpc_input.x_0(k++) = delayed_s.omega(2); // omega(3,4,5)
            g_mpc_input.x_0(k++) = s.r1(0); g_mpc_input.x_0(k++) = s.r2(0); g_mpc_input.x_0(k++) = s.r3(0); g_mpc_input.x_0(k++) = s.r4(0); // r_rotor_x(6,7,8,9)
            g_mpc_input.x_0(k++) = s.r1(1); g_mpc_input.x_0(k++) = s.r2(1); g_mpc_input.x_0(k++) = s.r3(1); g_mpc_input.x_0(k++) = s.r4(1); // r_rotor_y(10,11,12,13)
            g_mpc_input.x_0(k++) = cmd.d_theta(0); g_mpc_input.x_0(k++) = cmd.d_theta(1); g_mpc_input.x_0(k++) = cmd.d_theta(2); // delta_theta(14,15,16)
            g_mpc_input.x_0(k++) = cmd.r1(0); g_mpc_input.x_0(k++) = cmd.r2(0); g_mpc_input.x_0(k++) = cmd.r3(0); g_mpc_input.x_0(k++) = cmd.r4(0); // r_rotor_cmd_x(17,18,19,20)
            g_mpc_input.x_0(k++) = cmd.r1(1); g_mpc_input.x_0(k++) = cmd.r2(1); g_mpc_input.x_0(k++) = cmd.r3(1); g_mpc_input.x_0(k++) = cmd.r4(1); // r_rotor_cmd_y(21,22,23,24)

            // fill initial control input(u)
            for (int l=0; l<11; ++l) {g_mpc_input.u_0(l) = l_mpc_output.u_rate(l, 0);}

            int m = 0; // fill initial parameter(p)
            for (int j=0; j<3; ++j) {for (int i=0; i<3; ++i) {g_mpc_input.p(m++) = R_raw(i, j);}} // R_raw(0~8), column-major order to match CasADi reshape
            g_mpc_input.p(m++) = omega_raw(0); g_mpc_input.p(m++) = omega_raw(1); g_mpc_input.p(m++) = omega_raw(2); // omega_raw(9~11)
            g_mpc_input.p(m++) = -geometry_ctrl.f_total; // T_des(12)

            if (phase==Phase::MRG_ACTIVE_COT) {g_mpc_input.use_cot = true;}
            else {g_mpc_input.use_cot = false;}

            g_mpc_input.steps_req = param::N_STEPS_REQ;
            g_mpc_input.t = now;
            g_mpc_input.key = mpc_key;
            g_mpc_input.epoch = g_mpc_epoch.load(std::memory_order_relaxed);
            g_mpc_input.has = true;
            g_mpc_busy = true;
            mpc_cv.notify_one();
          }
        }
      }

      // --- attitude control --- 
      const Eigen::Matrix3d R_d = R_raw * expm_hat(cmd.d_theta);
      Eigen::Vector3d tau_des = geometry_ctrl.attitude_control(R_d);
      
      // --- (Sequential) Control Allocation ---
      Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
      Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
      Sequential_Allocation(F_des, tau_des, cmd.tauz_bar, delayed_s.arm_q, s.r_com, thrust_des, tilt_ang_des);

      // // --- (Normal) Control Allocation ---
      // Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
      // Control_Allocation(F_des, tau_des, bPcot_cur, s.r_com, thrust_des);
      // Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
      // thrust_des_log = thrust_des;

      // --- resolve r_cot_d to q_d  ---
      double q_d[20] = {0};
      IK(cmd.r1, cmd.r2, cmd.r3, cmd.r4, tilt_ang_des, q_d);

      // --- thrust to pwm ---
      Eigen::Vector4d pwm;
      for (int i = 0; i < 4; ++i) {
        pwm(i) = std::sqrt(std::max(0.0, (thrust_des(i) - param::PWM_B) / param::PWM_A));
        pwm(i) = std::clamp(pwm(i), 0.0, 1.0);
      }

      // ------ (PLANT) ------------------------------------------------------------------------------------
      // --- pwm -> thrust & torque ---
      const Eigen::Vector4d F = param::PWM_A * pwm.array().square() + param::PWM_B;
      const Eigen::Map<const Eigen::Vector4d> ROTOR_DIR(param::rotor_dir);
      const Eigen::Vector4d Tau = (param::PWM_ZETA * F.array() * ROTOR_DIR.array()).matrix();

      // --- thruster force&torque smoothing [25ms-timeconstant] ---
      smoothed_F   = 0.9 * smoothed_F   + 0.1 * F;
      smoothed_Tau = 0.9 * smoothed_Tau + 0.1 * Tau;

      // --- joint actuator delay [2.5ms one-step delay] ---
      for (uint8_t i=0; i<20; ++i) {smoothed_q_d[i] =  param::COT_DELAY_ALPHA * smoothed_q_d[i] + param::COT_DELAY_BETA * delayed_q_d[i];}

      // --- MAX thrust constraint ---
      if (elapsed_double >= 10.0) {for (uint8_t i=0; i<4; ++i) {if (smoothed_F(i) > param::SATURATION_THRUST) {smoothed_F(i) = param::SATURATION_THRUST;}}}

      // --- Step simulation at SIM_HZ using ZOH ---
      substep_accum += steps_per_ctrl;
      const int n_sub = static_cast<int>(substep_accum);
      substep_accum -= n_sub;

      { // Apply controls to MuJoCo
        std::lock_guard<std::mutex> scene_lk(scene_mtx);

        // save desired/current positions for viewer
        g_pos_cur[0]=d->qpos[0]; g_pos_des[0]=cmd.pos(0);
        g_pos_cur[1]=d->qpos[1]; g_pos_des[1]=-cmd.pos(1);
        g_pos_cur[2]=d->qpos[2]; g_pos_des[2]=-cmd.pos(2);
        Eigen::Map<Eigen::Matrix<mjtNum,4,1>>(d->ctrl) = smoothed_F.cast<mjtNum>();
        Eigen::Map<Eigen::Matrix<mjtNum,4,1>>(d->ctrl + 4) = smoothed_Tau.cast<mjtNum>();
        for (int i = 0; i < 20; ++i) d->ctrl[8 + i] = smoothed_q_d[i];

        for (int s = 0; s < n_sub; ++s) {mj_step(m, d);}
      }

      // time_delay store
      delayed_s = s;
      for (uint8_t i=0; i<20; ++i) {delayed_q_d[i] = q_d[i];}

      // ------ (Data logging) -----------------------------------------------------------------------------
      {
        mmap_manager::LogData ld;

        ld.t = static_cast<float>(elapsed_double);

        ld.pos_d[0] = static_cast<float>(cmd.pos(0));
        ld.pos_d[1] = static_cast<float>(cmd.pos(1));
        ld.pos_d[2] = static_cast<float>(cmd.pos(2));
        ld.vel_d[0] = static_cast<float>(cmd.vel(0));
        ld.vel_d[1] = static_cast<float>(cmd.vel(1));
        ld.vel_d[2] = static_cast<float>(cmd.vel(2));
        ld.acc_d[0] = static_cast<float>(cmd.acc(0));
        ld.acc_d[1] = static_cast<float>(cmd.acc(1));
        ld.acc_d[2] = static_cast<float>(cmd.acc(2));

        ld.pos[0] = static_cast<float>(s.pos(0));
        ld.pos[1] = static_cast<float>(s.pos(1));
        ld.pos[2] = static_cast<float>(s.pos(2));
        ld.vel[0] = static_cast<float>(s.vel(0));
        ld.vel[1] = static_cast<float>(s.vel(1));
        ld.vel[2] = static_cast<float>(s.vel(2));
        ld.acc[0] = static_cast<float>(s.acc(0));
        ld.acc[1] = static_cast<float>(s.acc(1));
        ld.acc[2] = static_cast<float>(s.acc(2));

        {
          const Eigen::Vector3d rpy_raw = R_to_rpy(R_raw);
          ld.rpy_raw[0] = static_cast<float>(rpy_raw(0));
          ld.rpy_raw[1] = static_cast<float>(rpy_raw(1));
          ld.rpy_raw[2] = static_cast<float>(rpy_raw(2));
        }
        {
          const Eigen::Vector3d rpy_d = R_to_rpy(R_d);
          ld.rpy_d[0] = static_cast<float>(rpy_d(0));
          ld.rpy_d[1] = static_cast<float>(rpy_d(1));
          ld.rpy_d[2] = static_cast<float>(rpy_d(2));
        }
        ld.omega_d[0] = static_cast<float>(omega_raw(0));
        ld.omega_d[1] = static_cast<float>(omega_raw(1));
        ld.omega_d[2] = static_cast<float>(omega_raw(2));
        ld.alpha_d[0] = static_cast<float>(gac_cmd.Wd_dot(0));
        ld.alpha_d[1] = static_cast<float>(gac_cmd.Wd_dot(1));
        ld.alpha_d[2] = static_cast<float>(gac_cmd.Wd_dot(2));

        ld.rpy[0]   = static_cast<float>(euler_rpy(0));
        ld.rpy[1]   = static_cast<float>(euler_rpy(1));
        ld.rpy[2]   = static_cast<float>(euler_rpy(2));
        ld.omega[0] = static_cast<float>(s.omega(0));
        ld.omega[1] = static_cast<float>(s.omega(1));
        ld.omega[2] = static_cast<float>(s.omega(2));
        ld.alpha[0] = static_cast<float>(s.alpha(0));
        ld.alpha[1] = static_cast<float>(s.alpha(1));
        ld.alpha[2] = static_cast<float>(s.alpha(2));

        ld.f_total  = static_cast<float>(geometry_ctrl.f_total);
        ld.tau_d[0] = static_cast<float>(tau_des(0));
        ld.tau_d[1] = static_cast<float>(tau_des(1));
        ld.tau_d[2] = static_cast<float>(tau_des(2));

        ld.tau_z_t        = static_cast<float>(cmd.tauz_bar);
        ld.tilt_rad[0]    = static_cast<float>(tilt_ang_des(0));
        ld.tilt_rad[1]    = static_cast<float>(tilt_ang_des(1));
        ld.tilt_rad[2]    = static_cast<float>(tilt_ang_des(2));
        ld.tilt_rad[3]    = static_cast<float>(tilt_ang_des(3));
        ld.f_thrst[0]     = static_cast<float>(thrust_des(0));
        ld.f_thrst[1]     = static_cast<float>(thrust_des(1));
        ld.f_thrst[2]     = static_cast<float>(thrust_des(2));
        ld.f_thrst[3]     = static_cast<float>(thrust_des(3));
        ld.f_thrst_con[0] = static_cast<float>(smoothed_F(0));
        ld.f_thrst_con[1] = static_cast<float>(smoothed_F(1));
        ld.f_thrst_con[2] = static_cast<float>(smoothed_F(2));
        ld.f_thrst_con[3] = static_cast<float>(smoothed_F(3));

        {
          const Eigen::Vector2d tau_off(F_des*(s.r_cot(1)-s.r_com(1)), -F_des*(s.r_cot(0)-s.r_com(0)));
          ld.tau_off[0] = static_cast<float>(tau_off(0));
          ld.tau_off[1] = static_cast<float>(tau_off(1));
        }
        {
          const Eigen::Vector3d tau_thrust = Forward_Allocate(smoothed_F, s.r1, s.r2, s.r3, s.r4, s.r_com);
          ld.tau_thrust[0] = static_cast<float>(tau_thrust(0));
          ld.tau_thrust[1] = static_cast<float>(tau_thrust(1));
          ld.tau_thrust[2] = static_cast<float>(tau_thrust(2));
        }

        ld.r_rotor1[0] = static_cast<float>(s.r1(0));
        ld.r_rotor1[1] = static_cast<float>(s.r1(1));
        ld.r_rotor2[0] = static_cast<float>(s.r2(0));
        ld.r_rotor2[1] = static_cast<float>(s.r2(1));
        ld.r_rotor3[0] = static_cast<float>(s.r3(0));
        ld.r_rotor3[1] = static_cast<float>(s.r3(1));
        ld.r_rotor4[0] = static_cast<float>(s.r4(0));
        ld.r_rotor4[1] = static_cast<float>(s.r4(1));
        ld.r_cot[0] = static_cast<float>(s.r_cot(0));
        ld.r_cot[1] = static_cast<float>(s.r_cot(1));

        ld.r_rotor1_d[0] = static_cast<float>(cmd.r1(0));
        ld.r_rotor1_d[1] = static_cast<float>(cmd.r1(1));
        ld.r_rotor2_d[0] = static_cast<float>(cmd.r2(0));
        ld.r_rotor2_d[1] = static_cast<float>(cmd.r2(1));
        ld.r_rotor3_d[0] = static_cast<float>(cmd.r3(0));
        ld.r_rotor3_d[1] = static_cast<float>(cmd.r3(1));
        ld.r_rotor4_d[0] = static_cast<float>(cmd.r4(0));
        ld.r_rotor4_d[1] = static_cast<float>(cmd.r4(1));
        {
          const Eigen::Vector3d r_cot_d = (cmd.r1 + cmd.r2 + cmd.r3 + cmd.r4) / 4.0;
          ld.r_cot_d[0] = static_cast<float>(r_cot_d(0));
          ld.r_cot_d[1] = static_cast<float>(r_cot_d(1));
        }

        for (uint8_t i=0; i<20; ++i){ld.q[i]     = static_cast<float>(s.arm_q[i]);}
        for (uint8_t i=0; i<20; ++i){ld.q_cmd[i] = static_cast<float>(q_d[i]);}

        ld.solve_ms = static_cast<float>(l_mpc_output.solve_ms);
        ld.solve_status = static_cast<int32_t>(l_mpc_output.state);

        ld.phase = static_cast<uint8_t>(phase);

        logger.push(ld);
      }

      // delay for real-time calculation
      const auto now_ = std::chrono::steady_clock::now();
      if (now_ < next_tick) {std::this_thread::sleep_until(next_tick);}
      next_tick += ctrl_period;
    }
  });

  // -------- [ Viewer thread (main thread) ] --------
  {
    mj_viewer::ViewerCtx v;
    mj_viewer::viewer_init(v, m);
    v.mpc_activated = &g_mpc_activated;

    std::deque<std::pair<double, Eigen::Vector3d>> path;
    std::deque<std::pair<double, Eigen::Vector3d>> dpath;
    std::chrono::steady_clock::time_point tv0 = std::chrono::steady_clock::now();

    while (!g_stop.load() && !glfwWindowShouldClose(v.window)) {
      std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

      Eigen::Vector3d pnow;
      Eigen::Vector3d pdes;
      { // MuJoCo Veiwer
        std::lock_guard<std::mutex> scene_lk(scene_mtx);
        mjv_updateScene(m, d, &v.opt, &v.pert, &v.cam, mjCAT_ALL, &v.scn);

        pnow(0)=g_pos_cur[0], pnow(1)=g_pos_cur[1], pnow(2)=g_pos_cur[2];
        pdes(0)=g_pos_des[0], pdes(1)=g_pos_des[1], pdes(2)=g_pos_des[2];
      }

      const double tv = std::chrono::duration<double>(now - tv0).count();

      // draw current position path
      path.emplace_back(tv, pnow);
      while (!path.empty() && (tv - path.front().first > param::PATH_SEC)) {path.pop_front();}
      for (size_t i = 1; i < path.size(); ++i) {
        const auto& a = path[i - 1].second;
        const auto& b = path[i].second;
        double p0[3] = {a.x(), a.y(), a.z()};
        double p1[3] = {b.x(), b.y(), b.z()};
        mj_viewer::add_capsule_segment(&v.scn, p0, p1, param::SIZE_PATH, param::RGBA_PATH);
      }

      // draw desired position path
      dpath.emplace_back(tv, pdes);
      while (!dpath.empty() && (tv - dpath.front().first > param::PATH_SEC)) {dpath.pop_front();}
      for (size_t i = 1; i < dpath.size(); ++i) {
        const auto& a = dpath[i - 1].second;
        const auto& b = dpath[i].second;
        double p0[3] = {a.x(), a.y(), a.z()};
        double p1[3] = {b.x(), b.y(), b.z()};
        mj_viewer::add_capsule_segment(&v.scn, p0, p1, param::SIZE_PATH, param::RGBA_DPATH);
      }

      // draw current position marker
      double l_pos_des[3] = {pdes.x(), pdes.y(), pdes.z()};
      mj_viewer::add_sphere_marker(&v.scn, l_pos_des, param::SIZE_DOT, param::RGBA_DOT);

      // Render
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(v.window, &viewport.width, &viewport.height);
      mjr_render(viewport, &v.scn, &v.con);
      glfwSwapBuffers(v.window);
      glfwPollEvents();
      std::this_thread::sleep_until(now + param::VIEWER_DT);
    }
    
    mj_viewer::viewer_close(v);
  }

  g_stop.store(true);
  mpc_cv.notify_all();

  if (th_ctrl.joinable()) th_ctrl.join();
  if (th_mpc.joinable())  th_mpc.join();

  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}
