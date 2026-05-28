#ifndef MJ_VIEWER_HELPER_H
#define MJ_VIEWER_HELPER_H

#include <cstdint>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <atomic>

namespace mj_viewer {

inline constexpr double PATH_SEC = 10.0;   // history length [sec]
inline constexpr float SIZE_DOT  = 0.03f;  // size(radious) of dot
inline constexpr float SIZE_PATH = 0.005f; // size(radious) of path
inline constexpr float RGBA_DOT[4]   = {1.00f, 0.00f, 0.00f, 0.95f}; // current pos color
inline constexpr float RGBA_PATH[4]  = {0.20f, 0.80f, 0.90f, 0.60f}; // current path color
inline constexpr float RGBA_DPATH[4] = {0.60f, 0.60f, 0.60f, 0.60f}; // desired path color

// Viewer context for MuJoCo + GLFW
struct ViewerCtx {
  mjvOption  opt;
  mjvScene   scn;
  mjrContext con;
  mjvCamera  cam;
  mjvPerturb pert;
  GLFWwindow* window{nullptr};
  const mjModel* model{nullptr};

  double last_x{0.0}, last_y{0.0};
  bool btn_left{false}, btn_middle{false}, btn_right{false};
  bool dragging{false};
  std::atomic<uint8_t>* phase_cmd = nullptr;
};

void viewer_init(ViewerCtx& v, const mjModel* m);
void viewer_close(ViewerCtx& v);

void add_sphere_marker(mjvScene* scn, const double pos[3], double radius, const float rgba[4]);
void add_capsule_segment(mjvScene* scn, const double p0[3], const double p1[3], double radius, const float rgba[4]);

} // namespace mj_viewer


#endif // MJ_VIEWER_HELPER_H