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
        g_mpc_output.has = true;
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

    // --- parameter definition ---
    const Eigen::Vector3d bPcot_init(0.0, 0.0, -0.24);    // [m]
    Eigen::Vector3d bPcot_des = bPcot_init;              // [m]
    Eigen::Vector3d bPc_hat   = Eigen::Vector3d::Zero(); // [m]
    Eigen::Matrix3d bRcot_des        = Eigen::Matrix3d::Identity(); // Body tilt
    double tauz_bar                  = 0.0;                     // Sequential control allocation
    Eigen::Vector3d delta_theta_opt  = Eigen::Vector3d::Zero(); // MRG optimal state
    Eigen::Vector2d r_cot_opt        = Eigen::Vector2d::Zero(); // MRG optimal state
    Eigen::Vector3d delta_theta_rate = Eigen::Vector3d::Zero(); // MRG optimal input
    Eigen::Vector2d r_cot_rate       = Eigen::Vector2d::Zero(); // MRG optimal input
    bool mpc_in_solving = false;
    uint32_t mpc_key = 1;
    Eigen::Vector3d bPcot_cur = Eigen::Vector3d::Zero();
    Eigen::Vector3d euler_rpy = Eigen::Vector3d::Zero();
    bool prev_mpc_on = false;
    uint8_t mpc_mod = COT_ACTIVATED;

    // --- noise injection ---
    static NOISE::State noise_state;
    NOISE::reset(noise_state, param::NOISE_SEED, std::chrono::steady_clock::now());

    // --- timedelay ---
    SimState delayed_s;
    double delayed_q_d[20] = {0};
    Eigen::Vector4d smoothed_F   = Eigen::Vector4d::Zero();
    Eigen::Vector4d smoothed_Tau = Eigen::Vector4d::Zero();

    // --- logging ---
    mmap_manager::MMapLogger logger("/tmp/strider_log.mmap", /*reset=*/true);
    logger.open();
    Eigen::Vector4d thrust_des_log = Eigen::Vector4d::Zero();
    float last_solve_ms = 0.0f;
    int32_t last_solve_status = -1;

    // --- time scope definition ---
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point next_tick = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point next_mpc_tick = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::duration ctrl_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(param::CTRL_DT));
    const double steps_per_ctrl = param::SIM_HZ / param::CTRL_HZ;
    double substep_accum = 0.0;

    { // Model warm-up
      double arm_angles[20]; // Initial arm joint angles
      Eigen::Vector4d tvc_angle = Eigen::Vector4d::Zero();
      IK(bPcot_des, bRcot_des, tvc_angle, param::L_DIST, arm_angles);
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

      // --- mujoco measurement ---
      SimState s; // mujoco d: z-up -> SimState s: z-down
      {
        std::lock_guard<std::mutex> scene_lk(scene_mtx);
        s.pos << d->qpos[0], -d->qpos[1], -d->qpos[2];
        s.vel << d->qvel[0], -d->qvel[1], -d->qvel[2];
        s.acc << d->qacc[0], -d->qacc[1], -d->qacc[2];
        const Eigen::Quaterniond q(d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]);
        s.R << quat_to_R(q);
        s.omega << d->qvel[3], -d->qvel[4], -d->qvel[5];
        for (int i=0; i<20; ++i) {s.arm_q[i] = d->qpos[7 + i];}
      }

      // --- sensor noise injection ---
      if (param::NOISE_ON) {NOISE::apply(noise_state, now, delayed_s);}

      // --- position control ---
      // Eigen::Vector3d pos_des;
      // if (elapsed_double >= 4.0) {pos_des = square4_point(elapsed_double);} // option: [fig8_point/square4_point]
      // else {pos_des = Eigen::Vector3d(0.0, 0.0, -1.0);}
      
      Eigen::Vector3d pos_des;
      Eigen::Vector3d vel_des = Eigen::Vector3d::Zero();
      Eigen::Vector3d acc_des = Eigen::Vector3d::Zero();
      if (elapsed_double >= 6.0) {l_traj_pva(elapsed_double, pos_des, vel_des, acc_des);} // option: [fig8_point_pva/circle_pva/l_traj_pva]
      else if (elapsed_double <= 2.0) {pos_des = goes_to(Eigen::Vector3d(1.5,0.0,-1.0), elapsed_double, 2.0);}
      else {pos_des = Eigen::Vector3d(1.5,0.0,-1.0);}

      gac_cmd.xd = pos_des;
      gac_cmd.xd_dot = vel_des;
      gac_cmd.xd_2dot = acc_des;
      gac_cmd.b1d = Eigen::Vector3d(1.0, 0.0 ,0.0);
      gac_state.x = delayed_s.pos;
      gac_state.v = delayed_s.vel;
      gac_state.a = delayed_s.acc;
      gac_state.R = delayed_s.R;
      gac_state.W = delayed_s.omega;
      geometry_ctrl.position_control();
      const Eigen::Matrix3d R_raw = gac_cmd.Rd;
      const Eigen::Vector3d w_raw = gac_cmd.Wd;
      const double T_des = -geometry_ctrl.f_total; // (f_total > 0)
      
      euler_rpy = R_to_rpy(delayed_s.R);
      bPcot_cur = FK(delayed_s.arm_q);
      bPc_hat(0) = param::COM_OFF_X + param::COT_2_COM_X * bPcot_cur(0);
      bPc_hat(1) = param::COM_OFF_Y + param::COT_2_COM_Y * bPcot_cur(1);
      { // MPC send
        std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
        if (g_mpc_activated.load(std::memory_order_relaxed)) {
          if (!mpc_in_solving) {
            if (now >= next_mpc_tick) {
              if (!g_mpc_output.has) {
                next_mpc_tick += param::MPC_COMPUTE_DT;
                mpc_key += 1;
                
                int k = 0; // fill initial state(x)
                g_mpc_input.x_0(k++) = euler_rpy(0); g_mpc_input.x_0(k++) = euler_rpy(1); g_mpc_input.x_0(k++) = euler_rpy(2); // theta(0,1,2)
                g_mpc_input.x_0(k++) = delayed_s.omega(0); g_mpc_input.x_0(k++) = delayed_s.omega(1); g_mpc_input.x_0(k++) = delayed_s.omega(2); // omega(3,4,5)
                g_mpc_input.x_0(k++) = bPcot_cur(0); g_mpc_input.x_0(k++) = bPcot_cur(1); // r_cot(6,7)
                g_mpc_input.x_0(k++) = delta_theta_opt(0); g_mpc_input.x_0(k++) = delta_theta_opt(1); g_mpc_input.x_0(k++) = delta_theta_opt(2); // delta_theta(8,9,10)
                g_mpc_input.x_0(k++) = bPcot_des(0); g_mpc_input.x_0(k++) = bPcot_des(1); // r_cot_cmd(11,12)

                int l = 0; // fill initial control input(u)
                g_mpc_input.u_0(l++) = delta_theta_rate(0); g_mpc_input.u_0(l++) = delta_theta_rate(1); g_mpc_input.u_0(l++) = delta_theta_rate(2); // delta_theta_cmd_rate(0,1,2)
                g_mpc_input.u_0(l++) = r_cot_rate(0); g_mpc_input.u_0(l++) = r_cot_rate(1); // r_cot_cmd_rate(3,4)

                int m = 0; // fill initial parameter(p)
                for (int j=0; j<3; ++j) {for (int i=0; i<3; ++i) {g_mpc_input.p(m++) = R_raw(i, j);}} // R_raw(0~8), column-major order to match CasADi reshape
                g_mpc_input.p(m++) = w_raw(0); g_mpc_input.p(m++) = w_raw(1); g_mpc_input.p(m++) = w_raw(2); // omega_raw(9~11)
                g_mpc_input.p(m++) = 0.5 * param::L_DIST; // l(12)
                g_mpc_input.p(m++) = -geometry_ctrl.f_total; // T_des(13)

                int n = 0;
                g_mpc_input.log(n++) = delayed_s.pos(0); g_mpc_input.log(n++) = delayed_s.pos(1); g_mpc_input.log(n++) = delayed_s.pos(2); // pos_cur
                g_mpc_input.log(n++) = pos_des(0); g_mpc_input.log(n++) = pos_des(1); g_mpc_input.log(n++) = pos_des(2); // pos_des
                g_mpc_input.log(n++) = thrust_des_log(0); g_mpc_input.log(n++) = thrust_des_log(1); g_mpc_input.log(n++) = thrust_des_log(2); g_mpc_input.log(n++) = thrust_des_log(3); // F1234
                
                if (!prev_mpc_on) {
                  if (mpc_mod==COT_ACTIVATED){
                    std::printf("MPC->NO-COT\n");
                    mpc_mod=COT_DISABLED;
                  }
                  else {
                    std::printf("MPC->YES-COT\n");
                    mpc_mod=COT_ACTIVATED;
                  }
                }

                if (mpc_mod==COT_ACTIVATED) {g_mpc_input.use_cot = true;}
                else {g_mpc_input.use_cot = false;}
                g_mpc_input.t = now;
                g_mpc_input.key = mpc_key;
                g_mpc_input.has = true;
                mpc_cv.notify_one();
              }
            }
          }
          prev_mpc_on = true;
        }
        else {
          // MPC OFF -> bPcot_des goes to inital value.
          bPcot_des = 0.005 * bPcot_init + 0.995 * bPcot_des;

          // reset prev solve value
          delta_theta_opt  = Eigen::Vector3d::Zero();
          r_cot_opt        = Eigen::Vector2d::Zero();
          delta_theta_rate = Eigen::Vector3d::Zero();
          r_cot_rate       = Eigen::Vector2d::Zero();
          prev_mpc_on = false;
        }
      }

      { // MPC get
        std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
        if (g_mpc_output.has) {
          if (g_mpc_output.key == mpc_key) {
            if (g_mpc_output.state == 0) { // solve success
              delta_theta_opt  = g_mpc_output.u_opt.template head<3>();   // [0,1,2]
              r_cot_opt        = g_mpc_output.u_opt.template tail<2>();   // [3,4]
              delta_theta_rate = g_mpc_output.u_rate.template head<3>();  // [0,1,2]
              r_cot_rate       = g_mpc_output.u_rate.template tail<2>();  // [3,4]
            }
            else { // solve failed
              delta_theta_opt  *= 0.995;
              r_cot_opt        *= 0.9995;
              delta_theta_rate  = Eigen::Vector3d::Zero();
              r_cot_rate        = Eigen::Vector2d::Zero();
            }
            g_mpc_output.has = false;

            // next time to solve
            if (now >= g_mpc_output.t + param::MPC_COMPUTE_DT) {next_mpc_tick = now;}
            else {next_mpc_tick = g_mpc_output.t + param::MPC_COMPUTE_DT;}

            // log
            last_solve_ms = static_cast<float>(g_mpc_output.solve_ms);
            last_solve_status = static_cast<int32_t>(g_mpc_output.state);
          }
          // else{std::printf("\n\n[MPC KEY WRONG ERROR. RESTART NOW]\n\n");}
        }
      }
      bPcot_des(0) = param::COT_DELAY_ALPHA * bPcot_des(0) + param::COT_DELAY_BETA * r_cot_opt(0);
      bPcot_des(1) = param::COT_DELAY_ALPHA * bPcot_des(1) + param::COT_DELAY_BETA * r_cot_opt(1);

      // --- attitude control --- 
      const Eigen::Matrix3d R_d = R_raw * expm_hat(delta_theta_opt);
      Eigen::Vector3d tau_des = geometry_ctrl.attitude_control(R_d);
      
      // --- (Sequential) Control Allocation ---
      Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
      Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
      Sequential_Allocation(T_des, tau_des, tauz_bar, delayed_s.arm_q, bPc_hat, thrust_des, tilt_ang_des);
      thrust_des_log = thrust_des;

      // // --- (Normal) Control Allocation ---
      // Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
      // Control_Allocation(T_des, tau_des, bPcot_cur, bPc_hat, thrust_des);
      // Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
      // thrust_des_log = thrust_des;

      // --- thrust to pwm ---
      Eigen::Vector4d pwm;
      for (int i = 0; i < 4; ++i) {
        pwm(i) = std::sqrt(std::max(0.0, (thrust_des(i) - param::PWM_B) / param::PWM_A));
        pwm(i) = std::clamp(pwm(i), 0.0, 1.0);
      }

      // --- resolve r_cot_cmd to q_d  ---
      double q_d[20] = {0};
      IK(bPcot_des, bRcot_des, tilt_ang_des, param::L_DIST, q_d);

      // ------ (PLANT) ------------------------------------------------------------------------------------
      // --- pwm -> thrust & torque ---
      const Eigen::Vector4d F = param::PWM_A * pwm.array().square() + param::PWM_B;
      const Eigen::Map<const Eigen::Vector4d> ROTOR_DIR(param::rotor_dir);
      const Eigen::Vector4d Tau = (param::PWM_ZETA * F.array() * ROTOR_DIR.array()).matrix();

      // --- thruster force&torque smoothing ---
      smoothed_F   = 0.882 * smoothed_F   + 0.118 * F;
      smoothed_Tau = 0.882 * smoothed_Tau + 0.118 * Tau;

      // // --- HW thrust constraint ---
      // if (elapsed_double >= 30.0) {for (uint8_t i=0; i<4; ++i) {if (smoothed_F(i) > 22.0) {smoothed_F(i) = 22.0;}}}

      // --- Step simulation at SIM_HZ using ZOH ---
      substep_accum += steps_per_ctrl;
      const int n_sub = static_cast<int>(substep_accum);
      substep_accum -= n_sub;

      { // Apply controls to MuJoCo
        std::lock_guard<std::mutex> scene_lk(scene_mtx);

        // save desired/current positions for viewer
        g_pos_cur[0]=d->qpos[0]; g_pos_des[0]=pos_des(0);
        g_pos_cur[1]=d->qpos[1]; g_pos_des[1]=-pos_des(1);
        g_pos_cur[2]=d->qpos[2]; g_pos_des[2]=-pos_des(2);
        Eigen::Map<Eigen::Matrix<mjtNum,4,1>>(d->ctrl) = smoothed_F.cast<mjtNum>();
        Eigen::Map<Eigen::Matrix<mjtNum,4,1>>(d->ctrl + 4) = smoothed_Tau.cast<mjtNum>();
        for (int i = 0; i < 20; ++i) d->ctrl[8 + i] = delayed_q_d[i];

        for (int s = 0; s < n_sub; ++s) {mj_step(m, d);}
      }

      // time_delay store
      delayed_s = s;
      for (uint8_t i=0; i<20; ++i) {delayed_q_d[i] = q_d[i];}

      // ------ (Data logging) -----------------------------------------------------------------------------
      {
        mmap_manager::LogData ld;

        ld.t = static_cast<float>(elapsed_double);

        ld.pos_d[0] = static_cast<float>(pos_des(0));
        ld.pos_d[1] = static_cast<float>(pos_des(1));
        ld.pos_d[2] = static_cast<float>(pos_des(2));

        ld.pos[0] = static_cast<float>(s.pos(0));
        ld.pos[1] = static_cast<float>(s.pos(1));
        ld.pos[2] = static_cast<float>(s.pos(2));

        ld.rpy[0] = static_cast<float>(euler_rpy(0));
        ld.rpy[1] = static_cast<float>(euler_rpy(1));
        ld.rpy[2] = static_cast<float>(euler_rpy(2));

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

        ld.tau_d[0] = static_cast<float>(tau_des(0));
        ld.tau_d[1] = static_cast<float>(tau_des(1));
        ld.tau_d[2] = static_cast<float>(tau_des(2));

        // CoT-offset torque around x,y
        const double f_total = geometry_ctrl.f_total;
        const Eigen::Vector2d tau_off(-f_total*(bPcot_cur(1)-bPc_hat(1)), f_total*(bPcot_cur(0)-bPc_hat(0)));
        ld.tau_off[0] = static_cast<float>(tau_off(0));
        ld.tau_off[1] = static_cast<float>(tau_off(1));

        // Thrust-diff torque = total tau_xy - cot-offset tau_xy
        ld.tau_thrust[0] = static_cast<float>(tau_des(0) - tau_off(0));
        ld.tau_thrust[1] = static_cast<float>(tau_des(1) - tau_off(1));

        for (int i = 0; i < 4; ++i) {
          ld.tilt_rad[i] = static_cast<float>(tilt_ang_des(i));
          ld.f_thrust[i] = static_cast<float>(smoothed_F(i));
        }

        ld.f_total = static_cast<float>(geometry_ctrl.f_total);

        ld.r_cot[0] = static_cast<float>(bPcot_cur(0));
        ld.r_cot[1] = static_cast<float>(bPcot_cur(1));

        ld.r_cot_cmd[0] = static_cast<float>(bPcot_des(0));
        ld.r_cot_cmd[1] = static_cast<float>(bPcot_des(1));

        ld.solve_ms = last_solve_ms;
        ld.solve_status = last_solve_status;

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
