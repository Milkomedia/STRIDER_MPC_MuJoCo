#include "mj_viewer_helper.hpp"
#include "mmap_manager.hpp"
#include "mpc_wrapper.hpp"
#include "fdcl_control.hpp"
#include "utils.hpp"
#include "gyro_ekf.hpp"
#include "gradient_ascent.hpp"

#include <thread>
#include <condition_variable>
#include <csignal>
#include <pybind11/embed.h>

static std::atomic<bool> g_stop{false};
static void sigint_handler(int) {g_stop.store(true);}

std::mutex scene_mtx;
static double g_pos_des[3] = {0.0, 0.0, 0.0};  // desired point for viewer
static double g_pos_cur[3] = {0.0, 0.0, 0.0};  // current point for viewer
static std::atomic<uint8_t> g_phase_cmd{static_cast<uint8_t>(Phase::GAC_ONLY)};

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

  std::array<int, 4> thrust_act_ids{};
  std::array<int, 4> torque_act_ids{};
  for (int i = 0; i < 4; ++i) {
    const std::string thrust_name = "thrust" + std::to_string(i + 1);
    const std::string torque_name = "torque" + std::to_string(i + 1);
    thrust_act_ids[i] = mj_name2id(m, mjOBJ_ACTUATOR, thrust_name.c_str());
    torque_act_ids[i] = mj_name2id(m, mjOBJ_ACTUATOR, torque_name.c_str());
  }

  std::array<int, 20> arm_act_ids{};
  for (int arm = 0; arm < 4; ++arm) {
    for (int joint = 0; joint < 5; ++joint) {
      const int idx = 5 * arm + joint;
      const std::string name = "servo_Arm" + std::to_string(arm + 1) + "_joint" + std::to_string(joint + 1);
      arm_act_ids[idx] = mj_name2id(m, mjOBJ_ACTUATOR, name.c_str());
    }
  }

  std::array<int, 20> arm_joint_ids{};
  std::array<int, 20> arm_qpos_adrs{};
  for (int arm = 0; arm < 4; ++arm) {
    for (int joint = 0; joint < 5; ++joint) {
      const int idx = 5 * arm + joint;
      const std::string name = "Arm" + std::to_string(arm + 1) + "_joint" + std::to_string(joint + 1);
      arm_joint_ids[idx] = mj_name2id(m, mjOBJ_JOINT, name.c_str());
      arm_qpos_adrs[idx] = (arm_joint_ids[idx] != -1) ? m->jnt_qposadr[arm_joint_ids[idx]] : -1;
    }
  }

  int load_act_id = -1;
  load_act_id = mj_name2id(m, mjOBJ_ACTUATOR, "servo_load_joint");
  const int load_joint_id = mj_name2id(m, mjOBJ_JOINT, "load_joint");
  const int load_qpos_adr = (load_joint_id != -1) ? m->jnt_qposadr[load_joint_id] : -1;
  const int bong_tip_load_body_id = mj_name2id(m, mjOBJ_BODY, "bong_tip_load");
  const int bong_tip_load_geom_id = mj_name2id(m, mjOBJ_GEOM, "bong_tip_load_geom");
  set_bong_tip_load_enabled(m, d, bong_tip_load_body_id, bong_tip_load_geom_id, false);

  // ---------------- [ MPC thread ] ----------------
  std::thread th_mpc([&]() {

    pybind11::scoped_interpreter py_guard{false};
    strider_mpc::acados_wrapper mpc; // compile acados before start.
    std::printf("\n\n ||----------------------------------||\n ||       Acados compile done.       ||\n || press the [space] to MPC on/off. ||\n ||----------------------------------||\n\n\n");

    while (!g_stop.load()) {
      strider_mpc::MPCInput in_local;
      {
        std::unique_lock<std::mutex> lk(mpc_mtx);
        mpc_cv.wait(lk, [&]{
          if (g_stop.load()) return true;
          if (static_cast<Phase>(g_phase_cmd.load(std::memory_order_relaxed)) == Phase::GAC_ONLY) {return false;}
          return g_mpc_input.has;
        });
        if (g_stop.load()) break;
        if (g_mpc_input.has == true) {
          in_local = g_mpc_input;
          g_mpc_input.has = false;
          g_mpc_busy = true;
        }
      }

      strider_mpc::MPCOutput out_local;
      try { out_local = mpc.compute(in_local); }
      catch (const std::exception& e) {
        out_local.solve_ms = 0.0;
        out_local.state = 99;
        std::fprintf(stderr, "[MPC EXCEPTION] %s\n", e.what());
        std::fflush(stderr);
      }

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

    // Gyro estimation
    Eigen::Vector3d prev_tau = Eigen::Vector3d::Zero();
    GyroEKF ekf;

    // --- state definition ---
    Phase   phase = Phase::GAC_ONLY;
    State   s{};
    Command cmd{};
    cmd.r1 = param::r1_init;
    cmd.r2 = param::r2_init;
    cmd.r3 = param::r3_init;
    cmd.r4 = param::r4_init;
    Eigen::Vector2d smoothed_cmd_r1 = param::r1_init;
    Eigen::Vector2d smoothed_cmd_r2 = param::r2_init;
    Eigen::Vector2d smoothed_cmd_r3 = param::r3_init;
    Eigen::Vector2d smoothed_cmd_r4 = param::r4_init;
    Eigen::Vector3d smoothed_d_theta = Eigen::Vector3d::Zero();
    Eigen::Vector3d prev_omega = Eigen::Vector3d::Zero();
    double prev_elapsed_double = 0.0;

    Eigen::Matrix<double, 3, 6> dob_state = Eigen::Matrix<double, 3, 6>::Zero();

    // --- com-bias inducing load param ---
    double servo_load_angle_cmd = 0.0; // 0.5*M_PI; 0.0;
    double servo_load_angle = 0.0; // 0.5*M_PI; 0.0;

    // --- auto-phase start ---
    bool auto_phase_started = false;
    constexpr Phase AUTO_PHASE = Phase::USE_FULL; // choose GAC_ONLY or USE_ARM or USE_DTHETA or USE_FULL or GRADIENT_ASCENT

    // --- MRG parameters ---
    uint32_t mpc_key = 1;
    strider_mpc::MPCOutput l_mpc_output;
    l_mpc_output.x_stage.setZero();
    l_mpc_output.u_stage.setZero();
    l_mpc_output.t = std::chrono::steady_clock::time_point::max();

    // --- Gradient Ascent parameters ---
    GRADIENT_ASCENT gradient_ascent;
    gradient_ascent.reset(param::r1_init, param::r2_init, param::r3_init, param::r4_init);

    // --- noise injection ---
    static NOISE::nState noise_state;
    NOISE::reset(noise_state, param::NOISE_SEED, std::chrono::steady_clock::now());

    // --- timedelay ---
    State delayed_s;
    double delayed_q_d[20] = {0};
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
      for (uint8_t i=0; i<20; ++i) {delayed_q_d[i] = arm_angles[i];}
      {
        std::lock_guard<std::mutex> scene_lk(scene_mtx);
        // spawn and 3 second do nothing
        for (int k = 0; k < 3*static_cast<int>(param::SIM_HZ); ++k) {
          for (int i = 0; i < 4; ++i) {
            if (thrust_act_ids[i] != -1) {d->ctrl[thrust_act_ids[i]] = 0.0;}
            if (torque_act_ids[i] != -1) {d->ctrl[torque_act_ids[i]] = 0.0;}
          }
          for (int i = 0; i < 20; ++i) {if (arm_act_ids[i] != -1) {d->ctrl[arm_act_ids[i]] = arm_angles[i];}}
          if (load_act_id != -1) {d->ctrl[load_act_id] = 0.0;}
          mj_step(m, d);
        }
      }
    }

    while (!g_stop.load()) {
      // --- time count ---
      const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
      const double elapsed_double = std::chrono::duration<double>(now - t0).count();

      Phase requested_phase = static_cast<Phase>(g_phase_cmd.load(std::memory_order_relaxed));

      if (!auto_phase_started && elapsed_double >= 15.05) {
        {
          std::lock_guard<std::mutex> scene_lk(scene_mtx);
          set_bong_tip_load_enabled(m, d, bong_tip_load_body_id, bong_tip_load_geom_id, true);
        }
        requested_phase = AUTO_PHASE;
        g_phase_cmd.store(static_cast<uint8_t>(AUTO_PHASE), std::memory_order_relaxed);
        auto_phase_started = true;
        mpc_cv.notify_all();
        std::printf("[AUTO PHASE] elapsed=%.3f, request=%d\n", elapsed_double, static_cast<int>(AUTO_PHASE));
        std::fflush(stdout);
      }

      if (requested_phase != phase) {
        std::lock_guard<std::mutex> lk(mpc_mtx);
        mpc_reset_locked(mpc_key);
        phase = requested_phase;
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
        for (int i = 0; i < 20; ++i) {s.arm_q[i] = (arm_qpos_adrs[i] != -1) ? d->qpos[arm_qpos_adrs[i]] : 0.0;}
        servo_load_angle = (load_qpos_adr != -1) ? d->qpos[load_qpos_adr] : 0.0;
      }
      s.alpha = diff(s.omega, prev_omega, elapsed_double, prev_elapsed_double);
      prev_omega = s.omega; prev_elapsed_double = elapsed_double;

      // --- sensor noise injection ---
      if (param::NOISE_ON) {NOISE::apply(noise_state, now, s);}

      // --- estimate gyro ---
      const Eigen::Vector3d euler_rpy = R_to_rpy(delayed_s.R);
      Eigen::Vector3d omega_hat = ekf.step(prev_tau, euler_rpy, delayed_s.omega);

      // // --- load angle cmd update ---
      // if (elapsed_double > 15.0) {
      //   servo_load_angle_cmd -= 1.0/400.0 * 0.5*M_PI / 3.0;
      //   if (servo_load_angle_cmd <= 0.0) {servo_load_angle_cmd = 0.0;}
      //   // servo_load_angle_cmd = 0.5*M_PI*std::sin(elapsed_double / M_2_PI)+0.5*M_PI;
      // }

      // --- position control ---
      if (elapsed_double >= 14.0) {l_traj_pva(elapsed_double+13.05, cmd.pos, cmd.vel, cmd.acc);} // option: [fig8_point_pva/circle_pva/l_traj_pva]
      else if (elapsed_double <= 2.0) {cmd.pos = goes_to(Eigen::Vector3d(-1.4,0.0,-1.3), elapsed_double, 2.0);}
      else {cmd.pos = Eigen::Vector3d(-1.4,0.0,-1.3);}
      cmd.vel = Eigen::Vector3d::Zero(); // not-use velocity command
      cmd.acc = Eigen::Vector3d::Zero(); // not-use velocity command

      gac_cmd.xd = cmd.pos;
      gac_cmd.xd_dot = cmd.vel;
      gac_cmd.xd_2dot = cmd.acc;
      gac_cmd.b1d = cmd.heading;
      gac_state.x = delayed_s.pos;
      gac_state.v = delayed_s.vel;
      gac_state.R = delayed_s.R;
      gac_state.W = omega_hat;
      geometry_ctrl.position_control();
      const Eigen::Matrix3d R_raw = gac_cmd.Rd;
      const Eigen::Vector3d omega_raw = gac_cmd.Wd;
      const Eigen::Vector3d alpha_raw = gac_cmd.Wd_dot;
      const double f_sum = -geometry_ctrl.f_total; // (f_total > 0)

      { // MPC get
        std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
        if (g_mpc_output.has) {
          const bool epoch_ok = (g_mpc_output.epoch == g_mpc_epoch.load(std::memory_order_relaxed));
          const bool key_ok = (g_mpc_output.key == mpc_key);
          const bool solve_ok = (g_mpc_output.state == 0);

          // l_mpc_output updated only when solve succeed.
          if ((phase != Phase::GAC_ONLY) && epoch_ok && key_ok && solve_ok) {l_mpc_output = g_mpc_output;}
          else if ((phase != Phase::GAC_ONLY) && epoch_ok && !key_ok) {mpc_reset_locked(mpc_key);}
          l_mpc_output.state = g_mpc_output.state; // *BUT l_mpc_output state indicates previous solve state(for logging)*
          g_mpc_output.has = false;
        }
      }

      bool mpc_applied = false;
      if (phase == Phase::USE_DTHETA || phase == Phase::USE_ARM || phase == Phase::USE_FULL) { // MPC unpack
        const bool epoch_ok = (l_mpc_output.epoch == g_mpc_epoch.load(std::memory_order_relaxed));
        const bool time_ok = ((now - l_mpc_output.t) < param::MPC_TIMEOUT_DURATUION);
        const bool solve_ok = (l_mpc_output.state == 0);
        
        if (epoch_ok && time_ok && solve_ok) {
          const std::size_t idx_raw = static_cast<std::size_t>(std::floor(std::chrono::duration<double>(now - l_mpc_output.t).count() / param::MPC_STEP_DT));
          const std::size_t idx = (idx_raw < param::N_STEPS_REQ) ? idx_raw : (param::N_STEPS_REQ+1);

          std::array<Eigen::Vector2d, 4> opt_r; // polar opt r_cmd
          opt_r[0] << l_mpc_output.u_stage(3, idx),  l_mpc_output.u_stage(4, idx);
          opt_r[1] << l_mpc_output.u_stage(5, idx),  l_mpc_output.u_stage(6, idx);
          opt_r[2] << l_mpc_output.u_stage(7, idx),  l_mpc_output.u_stage(8, idx);
          opt_r[3] << l_mpc_output.u_stage(9, idx),  l_mpc_output.u_stage(10, idx);
          const bool is_feasible = make_feasible(opt_r); // check workspace & collision

          if (is_feasible) {
            cmd.d_theta = l_mpc_output.u_stage.col(idx).segment<3>(0);
            std::array<Eigen::Vector2d, 4> r; // cartesian opt r_cmd
            polar2cart(opt_r[0], opt_r[1], opt_r[2], opt_r[3], r[0], r[1], r[2], r[3]);
            cmd.r1 = opt_r[0];
            cmd.r2 = opt_r[1];
            cmd.r3 = opt_r[2];
            cmd.r4 = opt_r[3];
            mpc_applied = true;
          }
        }
      }
      // [GAC flight] or [solve failed timeout] or [cannot make_feasible]
      if (!mpc_applied) { 
        cmd.d_theta *= param::GOES_2_ZERO_A;
        cmd.r1 = param::GOES_2_ZERO_A*cmd.r1 + param::GOES_2_ZERO_B*param::r1_init;
        cmd.r2 = param::GOES_2_ZERO_A*cmd.r2 + param::GOES_2_ZERO_B*param::r2_init;
        cmd.r3 = param::GOES_2_ZERO_A*cmd.r3 + param::GOES_2_ZERO_B*param::r3_init;
        cmd.r4 = param::GOES_2_ZERO_A*cmd.r4 + param::GOES_2_ZERO_B*param::r4_init; 
        l_mpc_output.x_stage.setZero();
        l_mpc_output.u_stage.setZero();
      }

      FK(delayed_s.arm_q, s.r_com, s.moi, s.r1, s.r2, s.r3, s.r4); // FK updates rotor position, CoM position, and inertia
      gac_state.J = s.moi;
      { // MPC send
        if (phase == Phase::USE_DTHETA || phase == Phase::USE_ARM || phase == Phase::USE_FULL) {
          Eigen::Vector2d s_p1, s_p2, s_p3, s_p4;
          Eigen::Vector2d c_p1, c_p2, c_p3, c_p4;
          cart2polar(s.r1, s.r2, s.r3, s.r4, s_p1, s_p2, s_p3, s_p4);
          std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
          if (!g_mpc_busy && !g_mpc_input.has && !g_mpc_output.has) { // push next solve immediately after the previous output
            mpc_key += 1;

            int k = 0; // fill initial state(x)
            g_mpc_input.x_0(k++) = euler_rpy(0); g_mpc_input.x_0(k++) = euler_rpy(1); g_mpc_input.x_0(k++) = euler_rpy(2); // theta(0,1,2)
            g_mpc_input.x_0(k++) = omega_hat(0); g_mpc_input.x_0(k++) = omega_hat(1); g_mpc_input.x_0(k++) = omega_hat(2); // omega(3,4,5)
            g_mpc_input.x_0(k++) = s_p1(0); g_mpc_input.x_0(k++) = s_p1(1);
            g_mpc_input.x_0(k++) = s_p2(0); g_mpc_input.x_0(k++) = s_p2(1);
            g_mpc_input.x_0(k++) = s_p3(0); g_mpc_input.x_0(k++) = s_p3(1);
            g_mpc_input.x_0(k++) = s_p4(0); g_mpc_input.x_0(k++) = s_p4(1); // r_rotor(6~13)

            // fill initial control input
            g_mpc_input.u_0(0) = cmd.d_theta(0); // delta_theta_cmd(0:3)
            g_mpc_input.u_0(1) = cmd.d_theta(1);
            g_mpc_input.u_0(2) = cmd.d_theta(2);
            g_mpc_input.u_0(3) = cmd.r1(0); g_mpc_input.u_0(4) = cmd.r1(1);
            g_mpc_input.u_0(5) = cmd.r2(0); g_mpc_input.u_0(6) = cmd.r2(1);
            g_mpc_input.u_0(7) = cmd.r3(0); g_mpc_input.u_0(8) = cmd.r3(1);
            g_mpc_input.u_0(9) = cmd.r4(0); g_mpc_input.u_0(10) = cmd.r4(1); // r_rotor_cmd(3:11)

            int m = 0; // fill initial parameter(p)
            for (int j=0; j<3; ++j) {for (int i=0; i<3; ++i) {g_mpc_input.p(m++) = R_raw(i, j);}} // R_raw(0~8), column-major order to match CasADi reshape
            g_mpc_input.p(m++) = omega_raw(0); g_mpc_input.p(m++) = omega_raw(1); g_mpc_input.p(m++) = omega_raw(2); // omega_raw(9~11)
            g_mpc_input.p(m++) = alpha_raw(0); g_mpc_input.p(m++) = alpha_raw(1); g_mpc_input.p(m++) = alpha_raw(2); // alpha_raw(12~14)
            for (int j=0; j<3; ++j) {for (int i=0; i<3; ++i) {g_mpc_input.p(m++) = delayed_s.R(i, j);}} // R_0(15~23), column-major order to match CasADi reshape
            // g_mpc_input.p(m++) = -f_sum; // positive, f_sum(24)
            g_mpc_input.p(m++) = std::clamp(-f_sum, 4.0*param::PWM_B, 4.0*(param::SATURATION_THRUST-0.3)); // positive, f_sum(24)
            g_mpc_input.p(m++) = s.d_hat(0); g_mpc_input.p(m++) = s.d_hat(1); g_mpc_input.p(m++) = s.d_hat(2); // disturbance torque(25~27)
            for (int j=0; j<3; ++j) {for (int i=0; i<3; ++i) {g_mpc_input.p(m++) = s.moi(i, j);}} // J(28~36), column-major order to match CasADi reshape

            if (phase==Phase::USE_FULL)        {g_mpc_input.use_delta = true;  g_mpc_input.use_arm = true; }
            else if (phase==Phase::USE_DTHETA) {g_mpc_input.use_delta = true;  g_mpc_input.use_arm = false;}
            else if (phase==Phase::USE_ARM)    {g_mpc_input.use_delta = false; g_mpc_input.use_arm = true; }
            else                               {g_mpc_input.use_delta = false; g_mpc_input.use_arm = false;}

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
      smoothed_d_theta = param::DTHETA_LPF_ALPHA*smoothed_d_theta + param::DTHETA_LPF_BETA*cmd.d_theta;
      const Eigen::Matrix3d Et = expm_hat(-smoothed_d_theta);
      const Eigen::Matrix3d Rd = R_raw * Et.transpose();
      const Eigen::Vector3d Wd = Et * omega_raw;
      const Eigen::Vector3d Wd_dot = Et * alpha_raw;
      const Eigen::Vector3d tau_des = geometry_ctrl.attitude_control(Rd, Wd, Wd_dot);

      // --- torque estimated ---
      prev_tau = tau_des + s.d_hat;
      if (auto_phase_started && elapsed_double >= 15.05) {s.d_hat = dob_update(euler_rpy, tau_des, dob_state);}

      // --- (Sequential) Control Allocation ---
      Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
      Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
      Sequential_Allocation(f_sum, tau_des, cmd.tauz_bar, delayed_s.arm_q, s.r_com, thrust_des, tilt_ang_des);
      if (phase == Phase::GRADIENT_ASCENT) {gradient_ascent.arm_cmd(cmd.r1, cmd.r2, cmd.r3, cmd.r4, s.r1, s.r2, s.r3, s.r4, tilt_ang_des, thrust_des);}

      // // --- (Normal) Control Allocation ---
      // Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
      // Control_Allocation(F_des, tau_des, bPcot_cur, s.r_com, thrust_des);
      // Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
      // thrust_des_log = thrust_des;

      // --- resolve r_cot_d to q_d  ---
      double q_d[20] = {0};
      smoothed_cmd_r1(0) = param::ARM_DELAY_ALPHA*smoothed_cmd_r1(0) + param::ARM_DELAY_BETA*cmd.r1(0); smoothed_cmd_r1(1) = param::BASE_DELAY_ALPHA*smoothed_cmd_r1(1) + param::BASE_DELAY_BETA*cmd.r1(1);
      smoothed_cmd_r2(0) = param::ARM_DELAY_ALPHA*smoothed_cmd_r2(0) + param::ARM_DELAY_BETA*cmd.r2(0); smoothed_cmd_r2(1) = param::BASE_DELAY_ALPHA*smoothed_cmd_r2(1) + param::BASE_DELAY_BETA*cmd.r2(1);
      smoothed_cmd_r3(0) = param::ARM_DELAY_ALPHA*smoothed_cmd_r3(0) + param::ARM_DELAY_BETA*cmd.r3(0); smoothed_cmd_r3(1) = param::BASE_DELAY_ALPHA*smoothed_cmd_r3(1) + param::BASE_DELAY_BETA*cmd.r3(1);
      smoothed_cmd_r4(0) = param::ARM_DELAY_ALPHA*smoothed_cmd_r4(0) + param::ARM_DELAY_BETA*cmd.r4(0); smoothed_cmd_r4(1) = param::BASE_DELAY_ALPHA*smoothed_cmd_r4(1) + param::BASE_DELAY_BETA*cmd.r4(1);
      IK(smoothed_cmd_r1, smoothed_cmd_r2, smoothed_cmd_r3, smoothed_cmd_r4, tilt_ang_des, q_d);

      // --- virtual thrust clipping (tightening starts at 10s, finishes at 15s)---
      double thrust_sat = 1e12;
      Eigen::Vector4d thrust_clamped = Eigen::Vector4d::Zero(); // (f_1234 > 0)
      if (elapsed_double >= 15.0)      {thrust_sat = param::SATURATION_THRUST;}
      else if (elapsed_double >= 10.0) {thrust_sat = param::SATURATION_THRUST + (1.0 - 0.2*param::CTRL_DT) * 5.0;}
      else                             {thrust_sat = param::SATURATION_THRUST + 5.0;}
      for (uint8_t i=0; i<4; ++i) {thrust_clamped(i) = (thrust_des(i) > thrust_sat) ? thrust_sat : thrust_des(i);}
      
      // --- thrust to pwm ---
      Eigen::Vector4d pwm;
      for (int i = 0; i < 4; ++i) {
        pwm(i) = std::sqrt(std::max(0.0, (thrust_clamped(i) - param::PWM_B) / param::PWM_A));
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
        for (int i = 0; i < 4; ++i) {
          if (thrust_act_ids[i] != -1) {d->ctrl[thrust_act_ids[i]] = static_cast<mjtNum>(smoothed_F(i));}
          if (torque_act_ids[i] != -1) {d->ctrl[torque_act_ids[i]] = static_cast<mjtNum>(smoothed_Tau(i));}
        }
        for (int i = 0; i < 20; ++i) {if (arm_act_ids[i] != -1) {d->ctrl[arm_act_ids[i]] = static_cast<mjtNum>(delayed_q_d[i]);}}
        if (load_act_id != -1) {d->ctrl[load_act_id] = servo_load_angle_cmd;}

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
          const Eigen::Vector3d rpy_d = R_to_rpy(Rd);
          ld.rpy_d[0] = static_cast<float>(rpy_d(0));
          ld.rpy_d[1] = static_cast<float>(rpy_d(1));
          ld.rpy_d[2] = static_cast<float>(rpy_d(2));
        }
        ld.omega_raw[0] = static_cast<float>(omega_raw(0));
        ld.omega_raw[1] = static_cast<float>(omega_raw(1));
        ld.omega_raw[2] = static_cast<float>(omega_raw(2));
        ld.omega_d[0] = static_cast<float>(Wd(0));
        ld.omega_d[1] = static_cast<float>(Wd(1));
        ld.omega_d[2] = static_cast<float>(Wd(2));
        
        ld.alpha_raw[0] = static_cast<float>(alpha_raw(0));
        ld.alpha_raw[1] = static_cast<float>(alpha_raw(1));
        ld.alpha_raw[2] = static_cast<float>(alpha_raw(2));
        ld.alpha_d[0] = static_cast<float>(Wd_dot(0));
        ld.alpha_d[1] = static_cast<float>(Wd_dot(1));
        ld.alpha_d[2] = static_cast<float>(Wd_dot(2));

        ld.rpy[0]   = static_cast<float>(euler_rpy(0));
        ld.rpy[1]   = static_cast<float>(euler_rpy(1));
        ld.rpy[2]   = static_cast<float>(euler_rpy(2));
        ld.omega[0] = static_cast<float>(omega_hat(0));
        ld.omega[1] = static_cast<float>(omega_hat(1));
        ld.omega[2] = static_cast<float>(omega_hat(2));
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
          const Eigen::Matrix<double, 3, 4> A = A_Matrix(s.r1, s.r2, s.r3, s.r4, s.r_com);

          const Eigen::Vector4d f_mean = 0.25 * smoothed_F.sum() * Eigen::Vector4d::Ones();
          const Eigen::Vector3d tau_arm = A * f_mean;
          ld.tau_off[0] = static_cast<float>(tau_arm(0));
          ld.tau_off[1] = static_cast<float>(tau_arm(1));
          
          const Eigen::Vector4d f_dev = smoothed_F - f_mean;
          const Eigen::Vector3d tau_dev = A * f_dev;
          ld.tau_thrust[0] = static_cast<float>(tau_dev(0));
          ld.tau_thrust[1] = static_cast<float>(tau_dev(1));
          ld.tau_thrust[2] = static_cast<float>(tau_dev(2));

          ld.r_rotor1[0] = static_cast<float>(s.r1(0));
          ld.r_rotor1[1] = static_cast<float>(s.r1(1));
          ld.r_rotor2[0] = static_cast<float>(s.r2(0));
          ld.r_rotor2[1] = static_cast<float>(s.r2(1));
          ld.r_rotor3[0] = static_cast<float>(s.r3(0));
          ld.r_rotor3[1] = static_cast<float>(s.r3(1));
          ld.r_rotor4[0] = static_cast<float>(s.r4(0));
          ld.r_rotor4[1] = static_cast<float>(s.r4(1));

          const Eigen::Vector3d s_cot = (s.r1 + s.r2 + s.r3 + s.r4) / 4.0;
          ld.r_cot[0] = static_cast<float>(s_cot(0));
          ld.r_cot[1] = static_cast<float>(s_cot(1));
          
          Eigen::Vector2d r1_d_xy, r2_d_xy, r3_d_xy, r4_d_xy;
          polar2cart(cmd.r1, cmd.r2, cmd.r3, cmd.r4, r1_d_xy, r2_d_xy, r3_d_xy, r4_d_xy);
          const Eigen::Vector2d cot_cmd = (r1_d_xy + r2_d_xy + r3_d_xy + r4_d_xy) / 4.0;

          ld.r_rotor1_d[0] = static_cast<float>(r1_d_xy(0));
          ld.r_rotor1_d[1] = static_cast<float>(r1_d_xy(1));
          ld.r_rotor2_d[0] = static_cast<float>(r2_d_xy(0));
          ld.r_rotor2_d[1] = static_cast<float>(r2_d_xy(1));
          ld.r_rotor3_d[0] = static_cast<float>(r3_d_xy(0));
          ld.r_rotor3_d[1] = static_cast<float>(r3_d_xy(1));
          ld.r_rotor4_d[0] = static_cast<float>(r4_d_xy(0));
          ld.r_rotor4_d[1] = static_cast<float>(r4_d_xy(1));

          ld.r_cot_d[0] = static_cast<float>(cot_cmd(0));
          ld.r_cot_d[1] = static_cast<float>(cot_cmd(1));
        }

        for (uint8_t i=0; i<20; ++i){ld.q[i]     = static_cast<float>(s.arm_q[i]);}
        for (uint8_t i=0; i<20; ++i){ld.q_cmd[i] = static_cast<float>(q_d[i]);}

        ld.d_hat[0] = static_cast<float>(s.d_hat(0));
        ld.d_hat[1] = static_cast<float>(s.d_hat(1));
        ld.d_hat[2] = static_cast<float>(s.d_hat(2));

        ld.pc_hat[0] = static_cast<float>(s.r_com(0));
        ld.pc_hat[1] = static_cast<float>(s.r_com(1));
        ld.pc_hat[2] = static_cast<float>(s.r_com(2));

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
    v.phase_cmd = &g_phase_cmd;

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
