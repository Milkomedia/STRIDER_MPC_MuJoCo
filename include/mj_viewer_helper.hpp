#ifndef MJ_VIEWER_HELPER_H
#define MJ_VIEWER_HELPER_H

#include <cstdint>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <atomic>

namespace mj_viewer {

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