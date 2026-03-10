#include "mj_viewer_helper.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

namespace mj_viewer {

namespace {
// ===== Internal helper & GLFW callbacks =====
int cam_action_for_buttons(const ViewerCtx* v) {
  if (v->btn_left  && !v->btn_middle && !v->btn_right) return mjMOUSE_ROTATE_V;
  if (v->btn_right && !v->btn_left   && !v->btn_middle) return mjMOUSE_MOVE_H;
  if (v->btn_middle && !v->btn_left  && !v->btn_right)  return mjMOUSE_MOVE_V;
  return -1;
}

void cb_mouse_button(GLFWwindow* win, int button, int action, int /*mods*/) {
  auto* v = reinterpret_cast<ViewerCtx*>(glfwGetWindowUserPointer(win));
  if (!v) return;

  const bool down = (action == GLFW_PRESS || action == GLFW_REPEAT);
  if (button == GLFW_MOUSE_BUTTON_LEFT)   v->btn_left   = down;
  if (button == GLFW_MOUSE_BUTTON_MIDDLE) v->btn_middle = down;
  if (button == GLFW_MOUSE_BUTTON_RIGHT)  v->btn_right  = down;

  if (action == GLFW_PRESS) {
    v->dragging = true;
    glfwGetCursorPos(win, &v->last_x, &v->last_y);
  } else if (action == GLFW_RELEASE) {
    v->dragging = v->btn_left || v->btn_middle || v->btn_right;
  }
}

void cb_cursor_pos(GLFWwindow* win, double xpos, double ypos) {
  auto* v = reinterpret_cast<ViewerCtx*>(glfwGetWindowUserPointer(win));
  if (!v || !v->dragging) return;

  const double dx = xpos - v->last_x;
  const double dy = ypos - v->last_y;
  v->last_x = xpos;
  v->last_y = ypos;

  const int act = cam_action_for_buttons(v);
  if (act >= 0) {
    const double scale = 0.0005;
    mjv_moveCamera(v->model, act, scale * dx, scale * dy, &v->scn, &v->cam);
  }
}

void cb_scroll(GLFWwindow* win, double /*xoff*/, double yoff) {
  auto* v = reinterpret_cast<ViewerCtx*>(glfwGetWindowUserPointer(win));
  if (!v) return;

  const double scale = 0.02;
  mjv_moveCamera(v->model, mjMOUSE_ZOOM, 0.0, -scale * yoff, &v->scn, &v->cam);
}
}

void cb_key(GLFWwindow* win, int key, int /*scancode*/, int action, int /*mods*/) {
  auto* v = reinterpret_cast<ViewerCtx*>(glfwGetWindowUserPointer(win));
  if (!v) return;

  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
    if (v->phase_cmd) {
      const uint8_t cur = v->phase_cmd->load(std::memory_order_relaxed);
      uint8_t next = 4;  // GAC_ONLY

      switch (cur) {
        case 4:  next = 5; break;  // GAC_ONLY   -> USE_DTHETA
        case 5:  next = 6; break;  // USE_DTHETA -> USE_ARM
        case 6:  next = 7; break;  // USE_ARM    -> USE_BOTH
        case 7:  next = 4; break;  // USE_BOTH   -> GAC_ONLY
        default: next = 4; break;
      }

      v->phase_cmd->store(next, std::memory_order_relaxed);

      std::printf("[PHASE] ");
      switch (next) {
          case 4:  std::printf("GAC_ONLY\n");   break;
          case 5:  std::printf("USE_DTHETA\n"); break;
          case 6:  std::printf("USE_ARM\n");    break;
          case 7:  std::printf("USE_BOTH\n");   break;
          default: std::printf("UNKNOWN\n");    break;
      }
    }
  }
}

// ===== Viewer init / close =====
void viewer_init(ViewerCtx& v, const mjModel* m) {
  v.model = m;

  mjv_defaultOption(&v.opt);
  mjv_defaultScene(&v.scn);
  mjr_defaultContext(&v.con);
  mjv_defaultCamera(&v.cam);
  mjv_defaultPerturb(&v.pert);

  if (!glfwInit()) mju_error("Could not initialize GLFW");

  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  v.window = glfwCreateWindow(1200, 900, "MuJoCo Viewer", nullptr, nullptr);
  if (!v.window) mju_error("Could not create GLFW window");
  glfwMakeContextCurrent(v.window);
  glfwSwapInterval(1);

  glfwSetWindowUserPointer(v.window, &v);
  glfwSetMouseButtonCallback(v.window, cb_mouse_button);
  glfwSetCursorPosCallback(v.window, cb_cursor_pos);
  glfwSetScrollCallback(v.window, cb_scroll);

  mjv_makeScene(m, &v.scn, 2000);
  mjr_makeContext(m, &v.con, mjFONTSCALE_150);

  v.cam.type = mjCAMERA_FREE;

  glfwSetKeyCallback(v.window, cb_key);
}

void viewer_close(ViewerCtx& v) {
  mjr_freeContext(&v.con);
  mjv_freeScene(&v.scn);
  if (v.window) {
    glfwDestroyWindow(v.window);
    v.window = nullptr;
  }
  glfwTerminate();
}

// ===== Marker utilities =====
void add_sphere_marker(mjvScene* scn, const double pos[3], double radius, const float rgba[4]) {
  if (!scn || scn->ngeom >= scn->maxgeom) return;

  mjtNum size[3] = { (mjtNum)radius, (mjtNum)radius, (mjtNum)radius };
  mjtNum p[3]    = { (mjtNum)pos[0], (mjtNum)pos[1], (mjtNum)pos[2] };
  mjtNum mat[9]  = { (mjtNum)1,0,0,  0,(mjtNum)1,0,  0,0,(mjtNum)1 };

  mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_SPHERE, size, p, mat, rgba);
  scn->geoms[scn->ngeom].category = mjCAT_DECOR;
  scn->ngeom += 1;
}

// Draw a solid segment (capsule) between p0 and p1; used for path history
void add_capsule_segment(mjvScene* scn, const double p0[3], const double p1[3], double radius, const float rgba[4]) {
  if (!scn || scn->ngeom >= scn->maxgeom) return;

  Eigen::Vector3d a(p0[0], p0[1], p0[2]);
  Eigen::Vector3d b(p1[0], p1[1], p1[2]);
  Eigen::Vector3d v = b - a;
  const double len = v.norm();
  if (len < 1e-6) return;

  Eigen::Vector3d z = v / len;
  Eigen::Vector3d tmp =
      (std::abs(z.z()) < 0.999) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  Eigen::Vector3d x = z.cross(tmp).normalized();
  Eigen::Vector3d y = z.cross(x);
  Eigen::Vector3d c = 0.5 * (a + b);

  mjtNum size[3] = { (mjtNum)radius, (mjtNum)(0.5 * len), (mjtNum)0 };
  mjtNum pos[3]  = { (mjtNum)c.x(), (mjtNum)c.y(), (mjtNum)c.z() };
  mjtNum mat[9]  = { (mjtNum)x.x(), (mjtNum)y.x(), (mjtNum)z.x(),
                     (mjtNum)x.y(), (mjtNum)y.y(), (mjtNum)z.y(),
                     (mjtNum)x.z(), (mjtNum)y.z(), (mjtNum)z.z() };

  mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_CAPSULE, size, pos, mat, rgba);
  scn->geoms[scn->ngeom].category = mjCAT_DECOR;
  scn->ngeom += 1;
}

} // namespace mj_viewer