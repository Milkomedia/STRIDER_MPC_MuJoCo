from __future__ import annotations

import os
import sys
import time
import signal
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtWidgets import (
  QApplication,
  QGroupBox,
  QHBoxLayout,
  QLabel,
  QMainWindow,
  QSizePolicy,
  QVBoxLayout,
  QWidget,
  QSplitter,
)

import pyqtgraph as pg
from vispy import scene
from vispy.scene import visuals

# ----------------------------
# Robust package-style import
# ----------------------------
THIS_DIR = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":
  PARENT_DIR = THIS_DIR.parent
  if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
  __package__ = THIS_DIR.name

from .mmap_manager import MMapReader

DebugFrame = Dict[str, Optional[np.ndarray]]

def _get(frame: DebugFrame, key: str) -> Optional[np.ndarray]:
  return frame.get(key, None)

def _rgb_hex(rgba: tuple[float, float, float, float]) -> str:
  r = int(np.clip(rgba[0], 0.0, 1.0) * 255)
  g = int(np.clip(rgba[1], 0.0, 1.0) * 255)
  b = int(np.clip(rgba[2], 0.0, 1.0) * 255)
  return f"#{r:02x}{g:02x}{b:02x}"

def _make_legend_html(
  c_ref: tuple[float, float, float, float],
  c_cot: tuple[float, float, float, float],
  arm_colors: list[tuple[float, float, float, float]],
) -> str:
  return (
    f"<span style='color:{_rgb_hex(c_ref)};'>■</span> ref traj<br/>"
    f"<span style='color:{_rgb_hex(c_cot)};'>■</span> cot traj(dot)<br/>"
    f"<span style='color:{_rgb_hex(arm_colors[0])};'>■</span> a1\t"
    f"<span style='color:{_rgb_hex(arm_colors[1])};'>■</span> a2<br/>"
    f"<span style='color:{_rgb_hex(arm_colors[2])};'>■</span> a3\t"
    f"<span style='color:{_rgb_hex(arm_colors[3])};'>■</span> a4"
  )

# ----------------------------
# math utils
# ----------------------------
def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
  # English comments only.
  cr = np.cos(roll); sr = np.sin(roll)
  cp = np.cos(pitch); sp = np.sin(pitch)
  cy = np.cos(yaw);   sy = np.sin(yaw)
  Rz = np.array([[cy, -sy, 0.0], [sy,  cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
  Ry = np.array([[ cp, 0.0,  sp], [0.0, 1.0, 0.0], [-sp, 0.0,  cp]], dtype=np.float64)
  Rx = np.array([[1.0, 0.0, 0.0], [0.0,  cr, -sr], [0.0,  sr,  cr]], dtype=np.float64)
  return Rz @ Ry @ Rx

def dh_transform_np(a_link: float, alpha_joint: float, d_link: float, theta_joint: float) -> np.ndarray:
  # English comments only.
  ct, st = np.cos(theta_joint), np.sin(theta_joint)
  ca_, sa_ = np.cos(alpha_joint), np.sin(alpha_joint)
  Tm = np.zeros((4, 4), dtype=np.float64)
  Tm[0, 0] = ct
  Tm[0, 1] = -st * ca_
  Tm[0, 2] = st * sa_
  Tm[0, 3] = a_link * ct
  Tm[1, 0] = st
  Tm[1, 1] = ct * ca_
  Tm[1, 2] = -ct * sa_
  Tm[1, 3] = a_link * st
  Tm[2, 0] = 0.0
  Tm[2, 1] = sa_
  Tm[2, 2] = ca_
  Tm[2, 3] = d_link
  Tm[3, 3] = 1.0
  return Tm

def FK_chain_np(q_arm: np.ndarray, arm_idx: int, dh_arm: np.ndarray, dh_base: np.ndarray) -> np.ndarray:
  # English comments only.
  a0     = float(dh_base[arm_idx, 0])
  theta0 = float(dh_base[arm_idx, 1])
  Tm = dh_transform_np(a0, 0.0, 0.0, theta0)

  joint_pos = np.zeros((6, 3), dtype=np.float64)
  joint_pos[0, :] = Tm[0:3, 3]

  for i in range(dh_arm.shape[0]):
    a     = float(dh_arm[i, 0])
    alpha = float(dh_arm[i, 1])
    Tm = Tm @ dh_transform_np(a, alpha, 0.0, float(q_arm[i]))
    joint_pos[i + 1, :] = Tm[0:3, 3]
  return joint_pos

def _as_2d(arr: Any, rows: int, cols: int) -> np.ndarray:
  # English comments only.
  a = np.asarray(arr)
  if a.ndim == 2:
    return a
  if a.ndim == 1 and a.size == rows * cols:
    return a.reshape(rows, cols)
  return a.reshape(rows, cols)  # will raise if incompatible

def _as_1d(arr: Any, size: Optional[int] = None) -> np.ndarray:
  # English comments only.
  a = np.asarray(arr).ravel()
  if size is not None and a.size != size:
    # Keep as-is; caller may min() lengths.
    return a
  return a

def compute_frame_from_pkt(pkt: Any) -> DebugFrame:
  # English comments only.
  from .params import (
    DH_PARAMS_ARM,
    DH_PARAMS_BASE,
    MASS,
    G_ACCEL,
    COST_POS_ERR,
    COST_ANG_ERR,
    COST_F_THRUST,
  )

  N = int(pkt.N)
  nx = int(pkt.nx)
  nu = int(pkt.nu)
  np_ = int(pkt.np)

  # Time step: prefer params.DT, fallback env, then 1.0
  dt = float(getattr(__import__(f"{__package__}.params", fromlist=["DT"]), "DT", None) or
             os.environ.get("STRIDER_MPC_DT", "0.04"))

  T = N + 1
  t = np.arange(T, dtype=np.float64) * dt

  x_all = _as_2d(pkt.x_all, T, nx).astype(np.float64, copy=False)
  u_all = _as_2d(pkt.u_all, N, nu).astype(np.float64, copy=False)
  p_all = _as_2d(pkt.p_all, max(1, N), np_).astype(np.float64, copy=False)

  pos_d = np.asarray(pkt.pos_d, dtype=np.float64)
  if pos_d.ndim == 1:
    pos_d = pos_d.reshape(-1, 3)
  yaw_d = _as_1d(pkt.yaw_d)

  # State slices (same indexing as your print_last_debug)
  p = x_all[:, 0:3]
  v = x_all[:, 3:6]
  th = x_all[:, 6:9]     # rad
  om = x_all[:, 9:12]
  q  = x_all[:, 12:32]   # 20 joints (rad)

  # Parameter (same as solver.get(0,"p"))
  p_param = p_all[0, :].ravel()
  cot_p_c_hat = p_param[0:3].copy()

  dh_arm = np.array(DH_PARAMS_ARM, dtype=np.float64)
  dh_base = np.array(DH_PARAMS_BASE, dtype=np.float64)

  e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
  m = float(MASS)
  g = float(G_ACCEL)

  # Outputs to feed GUI
  cot_p_c_mm = np.full((T, 3), np.nan, dtype=np.float64)
  wrench_xyz = np.full((T, 3), np.nan, dtype=np.float64)
  arm_tips = np.full((T, 4, 3), np.nan, dtype=np.float64)

  # FK-based quantities
  for k in range(T):
    qk = q[k, :]

    q1 = qk[0:5]
    q2 = qk[5:10]
    q3 = qk[10:15]
    q4 = qk[15:20]

    b_j1 = FK_chain_np(q1, 0, dh_arm, dh_base)
    b_j2 = FK_chain_np(q2, 1, dh_arm, dh_base)
    b_j3 = FK_chain_np(q3, 2, dh_arm, dh_base)
    b_j4 = FK_chain_np(q4, 3, dh_arm, dh_base)

    b_p15 = b_j1[-1, :]
    b_p25 = b_j2[-1, :]
    b_p35 = b_j3[-1, :]
    b_p45 = b_j4[-1, :]

    b_p_cot = (b_p15 + b_p25 + b_p35 + b_p45) / 4.0
    cot_p_c = cot_p_c_hat - b_p_cot
    cot_p_c_mm[k, :] = 1000.0 * cot_p_c

    Rk = rpy_to_R(th[k, 0], th[k, 1], th[k, 2])

    # global arm tips = p_cot + R*(b_tip - b_p_cot)
    arm_tips[k, 0, :] = p[k, :] + Rk @ (b_p15 - b_p_cot)
    arm_tips[k, 1, :] = p[k, :] + Rk @ (b_p25 - b_p_cot)
    arm_tips[k, 2, :] = p[k, :] + Rk @ (b_p35 - b_p_cot)
    arm_tips[k, 3, :] = p[k, :] + Rk @ (b_p45 - b_p_cot)

    wrench_xyz[k, :] = np.cross(cot_p_c, Rk.T @ (-m * g * e3))

  # Controls: thrust (pad to T for convenience)
  thrust_N = np.full(T, np.nan, dtype=np.float64)
  if u_all.shape[0] >= 1:
    f_hist = u_all[:, 0].ravel()
    thrust_N[:min(T, f_hist.size)] = f_hist[:min(T, f_hist.size)]
    if T == f_hist.size + 1:
      thrust_N[-1] = f_hist[-1]

  # Costs
  Wp = np.diag(np.asarray(COST_POS_ERR, dtype=np.float64))
  Wy = float(np.asarray(COST_ANG_ERR, dtype=np.float64).ravel()[0])
  Wf = float(np.asarray(COST_F_THRUST, dtype=np.float64).ravel()[0])
  f_ref = float(m * g)

  # Stage count is limited by available arrays
  N_cost = int(min(
    max(0, pos_d.shape[0]),
    max(0, yaw_d.size),
    max(0, u_all.shape[0]),
    max(0, T - 1),
  ))

  Jp = np.full(N_cost, np.nan, dtype=np.float64)
  Jy = np.full(N_cost, np.nan, dtype=np.float64)
  Jf = np.full(N_cost, np.nan, dtype=np.float64)
  Jtotal = np.full(N_cost + 1, np.nan, dtype=np.float64)

  for k in range(N_cost):
    pk = p[k, :]
    yaw_k = th[k, 2]
    f_k = float(u_all[k, 0])

    e_p = pk - pos_d[k, :]
    e_y = yaw_k - float(yaw_d[k])
    e_f = f_k - f_ref

    Jp[k] = float(e_p.T @ Wp @ e_p)
    Jy[k] = float(Wy * e_y * e_y)
    Jf[k] = float(Wf * e_f * e_f)
    Jtotal[k] = Jp[k] + Jy[k] + Jf[k]

  if N_cost >= 1:
    pN = p[N_cost, :]
    yawN = th[N_cost, 2]
    e_pN = pN - pos_d[-1, :]
    e_yN = yawN - float(yaw_d[-1])
    Jp_N = float(e_pN.T @ Wp @ e_pN)
    Jy_N = float(Wy * e_yN * e_yN)
    Jtotal[N_cost] = Jp_N + Jy_N

  # rpy in deg for GUI
  rpy_deg = (th * (180.0 / np.pi)).astype(np.float64, copy=False)

  # Show arm1 joint angles in the "Arm joints" 5-curve plot (deg)
  joints_deg = (q[:, 0:5] * (180.0 / np.pi)).astype(np.float64, copy=False)

  # X-axes for plots that are naturally stage-based
  t_u = np.arange(N, dtype=np.float64) * dt
  t_cost = np.arange(N_cost, dtype=np.float64) * dt
  t_total = np.arange(N_cost + 1, dtype=np.float64) * dt

  # Solve-time history: build a short x for GUI (viewer will supply y history)
  solve_ms = float(pkt.solve_ms)

  return {
    "t": t,

    # 3D
    "p_ref": pos_d,     # (N,3)
    "p_cot": p,         # (T,3)
    "arm_tips": arm_tips,  # (T,4,3)

    # 2D (state-based)
    "v_xyz": v,
    "rpy_deg": rpy_deg,
    "omega_xyz": om,
    "cot_p_c_mm": cot_p_c_mm,
    "wrench_xyz": wrench_xyz,
    "joints_deg": joints_deg,

    # 2D (control/cost axes)
    "t_u": t_u,
    "thrust_N": f_hist if u_all.shape[0] >= 1 else None,  # (N,)
    "t_cost": t_cost,
    "t_total": t_total,
    "J_pos": Jp,
    "J_yaw": Jy,
    "J_thrust": Jf,
    "J_total": Jtotal,

    # scalar solve time
    "solve_ms_scalar": np.array([solve_ms], dtype=np.float64),
  }

class DebugViewerMainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("STRIDER MPC Viewer")
    self.resize(1800, 600)

    # Global pg config
    pg.setConfigOptions(antialias=True)

    # Solve-time history buffer (viewer-side, like your original ring)
    self._solve_hist_size = 200
    self._solve_ms_hist = np.full(self._solve_hist_size, np.nan, dtype=np.float64)
    self._solve_count = 0

    central = QWidget()
    self.setCentralWidget(central)

    root = QHBoxLayout(central)
    root.setContentsMargins(0,0,0,0)

    splitter = QSplitter(Qt.Horizontal)
    root.addWidget(splitter)

    # ----------------------------
    # Left: 3D plot group (Vispy)
    # ----------------------------
    self.gb_3d = QGroupBox()
    self.gb_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    l3 = QVBoxLayout(self.gb_3d)
    l3.setContentsMargins(0,0,0,0)

    self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="white")
    l3.addWidget(self.canvas.native)

    self.view = self.canvas.central_widget.add_view()
    self.view.camera = "turntable"
    self.view.camera.fov = 5
    self.view.camera.distance = 50.0
    self.view.camera.elevation = 30
    self.view.camera.azimuth = 15

    # Grid
    grid = visuals.GridLines(color=(0.1, 0.1, 0.1, 1.0), parent=self.view.scene)
    grid.set_gl_state('translucent', depth_test=False)
    
    # Axis params
    axis_len = 0.2     # meters
    axis_w   = 2       # pixels

    # X axis (red)
    self._axis_x = visuals.Line(
        pos=np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=np.float32),
        color=(1, 0, 0, 1),
        width=axis_w,
        parent=self.view.scene,
        method="gl",
    )

    # Y axis (green)
    self._axis_y = visuals.Line(
        pos=np.array([[0, 0, 0], [0, axis_len, 0]], dtype=np.float32),
        color=(0, 1, 0, 1),
        width=axis_w,
        parent=self.view.scene,
        method="gl",
    )

    # Z axis (blue)
    self._axis_z = visuals.Line(
        pos=np.array([[0, 0, 0], [0, 0, axis_len]], dtype=np.float32),
        color=(0, 0, 1, 1),
        width=axis_w,
        parent=self.view.scene,
        method="gl",
    )

    # Colors
    self._c_ref = (0.10, 0.30, 0.90, 1.0)   # blue
    self._c_cot = (1.00, 0.20, 0.20, 0.60)  # red-ish
    self._arm_colors = [
      (0.95, 0.55, 0.10, 1.0),  # arm1 tip
      (0.15, 0.70, 0.20, 1.0),  # arm2 tip
      (0.80, 0.20, 0.85, 1.0),  # arm3 tip
      (0.85, 0.15, 0.15, 1.0),  # arm4 tip
    ]

    # Trajectory visuals
    self._ref_line = visuals.Line(
      pos=np.zeros((2, 3), dtype=np.float32),
      parent=self.view.scene,
      width=2,
      color=self._c_ref,
      connect="strip",
      method="gl",
    )

    self._cot_markers = visuals.Markers(parent=self.view.scene)
    self._cot_markers.set_data(
      pos=np.zeros((1, 3), dtype=np.float32),
      face_color=self._c_cot,
      size=4,
    )

    self._arm_lines: list[visuals.Line] = []
    for c in self._arm_colors:
      ln = visuals.Line(
        pos=np.zeros((2, 3), dtype=np.float32),
        parent=self.view.scene,
        width=2,
        color=c,
        connect="strip",
        method="gl",
      )
      self._arm_lines.append(ln)

    # Legend overlay
    self._legend = QLabel(self.canvas.native)
    self._legend.setTextFormat(Qt.RichText)
    self._legend.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    self._legend.setStyleSheet(
      "QLabel {"
      "  background: rgba(255,255,255,220);"
      "  border: 1px solid rgba(0,0,0,70);"
      "  border-radius: 4px;"
      "  padding: 6px 8px;"
      "  font-size: 12px;"
      "}"
    )
    self._legend.setText(_make_legend_html(self._c_ref, (1.0, 0.2, 0.2, 1.0), self._arm_colors))
    self._legend.adjustSize()

    self.canvas.native.installEventFilter(self)
    self._reposition_legend()

    splitter.addWidget(self.gb_3d)

    # ----------------------------
    # Right: 3x3 plot group (PyQtGraph)
    # ----------------------------
    self.gb_2d = QGroupBox()
    self.gb_2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    l2 = QVBoxLayout(self.gb_2d)
    l2.setContentsMargins(0,0,0,0)

    gridw = QWidget()
    gl = pg.QtWidgets.QGridLayout(gridw)
    gl.setContentsMargins(0,0,0,0)
    gl.setSpacing(6)

    pens = [
      pg.mkPen(color=(220, 50, 50), width=2),
      pg.mkPen(color=(50, 160, 70), width=2),
      pg.mkPen(color=(50, 90, 220), width=2),
      pg.mkPen(color=(120, 60, 200), width=2),
      pg.mkPen(color=(0, 0, 0), width=2),
    ]

    def make_plot(title: str, y_label: str, curve_names: list[str]):
      w = pg.PlotWidget()
      w.setBackground("w")
      w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

      pi = w.getPlotItem()
      pi.setTitle(title)
      pi.showGrid(x=True, y=True, alpha=0.25)
      pi.getAxis("left").setPen(pg.mkPen("k"))
      pi.getAxis("bottom").setPen(pg.mkPen("k"))
      pi.getAxis("left").setTextPen(pg.mkPen("k"))
      pi.getAxis("bottom").setTextPen(pg.mkPen("k"))
      pi.setLabel("left", y_label)

      legend = pi.addLegend()
      legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

      curves = []
      for i, name in enumerate(curve_names):
        curves.append(w.plot(name=name, pen=pens[i % len(pens)]))
      return w, curves

    self._plots: Dict[str, Tuple[pg.PlotWidget, list[Any]]] = {}
    self._plots["vel"] = make_plot("Velocity", "[m/s]", ["vx", "vy", "vz"])
    self._plots["cotpc"] = make_plot("cot_p_c {b}", "[mm]", ["x", "y", "z"])
    self._plots["joints"] = make_plot("Arm joints (arm1)", "[deg]", ["q11", "q12", "q13", "q14", "q15"])

    self._plots["rpy"] = make_plot("RPY", "[deg]", ["roll", "pitch", "yaw"])
    self._plots["wrench"] = make_plot("Arm wrench", "[Nm]", ["x", "y", "z"])
    self._plots["thrust"] = make_plot("Thrust command f_t", "[N]", ["f_t"])

    self._plots["omega"] = make_plot("Angular velocity", "[rad/s]", ["wx", "wy", "wz"])
    self._plots["cost"] = make_plot("Cost", "cost", ["J_pos", "J_yaw", "J_thrust", "J_total"])
    self._plots["solve"] = make_plot("Solve time", "[ms]", ["ms"])

    cells = [
      ["vel", "cotpc", "joints"],
      ["rpy", "wrench", "thrust"],
      ["omega", "cost", "solve"],
    ]
    for r in range(3):
      for c in range(3):
        w, _ = self._plots[cells[r][c]]
        gl.addWidget(w, r, c)

    l2.addWidget(gridw)
    splitter.addWidget(self.gb_2d)

    splitter.setStretchFactor(0, 5)
    splitter.setStretchFactor(1, 2)

  def eventFilter(self, obj, event) -> bool:
    if obj is self.canvas.native and event.type() == QEvent.Resize:
      self._reposition_legend()
      return False
    return super().eventFilter(obj, event)

  def _reposition_legend(self) -> None:
    pad = 10
    w = self.canvas.native.width()
    self._legend.adjustSize()
    x = max(pad, w - self._legend.width() - pad)
    y = pad
    self._legend.move(x, y)

  def _push_solve_ms(self, solve_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    idx = self._solve_count % self._solve_hist_size
    self._solve_ms_hist[idx] = solve_ms
    self._solve_count += 1

    count = min(self._solve_count, self._solve_hist_size)
    idx_end = self._solve_count
    idx_start = max(0, idx_end - count)
    indices = np.array([(i % self._solve_hist_size) for i in range(idx_start, idx_end)], dtype=np.int64)
    ms_vals = self._solve_ms_hist[indices]
    x_idx = np.arange(-count + 1, 1, dtype=np.float64)
    return x_idx, ms_vals

  # Main update: frame -> plots
  def update_from_frame(self, frame: DebugFrame) -> None:
    # -------- 3D --------
    p_ref = _get(frame, "p_ref")
    p_cot = _get(frame, "p_cot")
    arm_tips = _get(frame, "arm_tips")

    if p_ref is not None and p_ref.ndim == 2 and p_ref.shape[1] == 3 and p_ref.shape[0] >= 2:
      self._ref_line.set_data(pos=p_ref.astype(np.float32, copy=False))

    if p_cot is not None and p_cot.ndim == 2 and p_cot.shape[1] == 3 and p_cot.shape[0] >= 1:
      pts = p_cot.astype(np.float32, copy=False)
      self._cot_markers.set_data(pos=pts, face_color=self._c_cot, size=3)

    if arm_tips is not None and arm_tips.ndim == 3 and arm_tips.shape[1] == 4 and arm_tips.shape[2] == 3:
      tips = arm_tips.astype(np.float32, copy=False)
      for i in range(4):
        self._arm_lines[i].set_data(pos=tips[:, i, :])

    # # Camera auto-center
    # pts_all = []
    # if p_cot is not None:
    #   pts_all.append(p_cot)
    # if p_ref is not None:
    #   pts_all.append(p_ref)
    # if arm_tips is not None:
    #   pts_all.append(arm_tips.reshape(-1, 3))

    # if pts_all:
    #   P = np.concatenate(pts_all, axis=0)
    #   if P.shape[0] >= 2:
    #     mn = P.min(axis=0)
    #     mx = P.max(axis=0)
    #     center = 0.5 * (mn + mx)
    #     span = float(np.max(mx - mn))
    #     if np.isfinite(span) and span > 1e-6:
    #       self.view.camera.center = tuple(center.tolist())
    #       self.view.camera.distance = max(1.5, 2.2 * span)

    # -------- 2D --------
    t = _get(frame, "t")
    if t is None:
      return

    def set_plot(key: str, xs: Any, ys: list[np.ndarray]) -> None:
      # English comments only.
      _, curves = self._plots[key]
      for i, (c, y) in enumerate(zip(curves, ys)):
        if isinstance(xs, (list, tuple)):
          x = xs[i]
        else:
          x = xs
        c.setData(x, y)

    v_xyz = _get(frame, "v_xyz")
    if v_xyz is not None and v_xyz.ndim == 2 and v_xyz.shape[1] == 3:
      set_plot("vel", t, [v_xyz[:, 0], v_xyz[:, 1], v_xyz[:, 2]])

    cot_p_c_mm = _get(frame, "cot_p_c_mm")
    if cot_p_c_mm is not None and cot_p_c_mm.ndim == 2 and cot_p_c_mm.shape[1] == 3:
      set_plot("cotpc", t, [cot_p_c_mm[:, 0], cot_p_c_mm[:, 1], cot_p_c_mm[:, 2]])

    joints_deg = _get(frame, "joints_deg")
    if joints_deg is not None and joints_deg.ndim == 2 and joints_deg.shape[1] == 5:
      set_plot("joints", t, [joints_deg[:, 0], joints_deg[:, 1], joints_deg[:, 2], joints_deg[:, 3], joints_deg[:, 4]])

    rpy_deg = _get(frame, "rpy_deg")
    if rpy_deg is not None and rpy_deg.ndim == 2 and rpy_deg.shape[1] == 3:
      set_plot("rpy", t, [rpy_deg[:, 0], rpy_deg[:, 1], rpy_deg[:, 2]])

    wrench_xyz = _get(frame, "wrench_xyz")
    if wrench_xyz is not None and wrench_xyz.ndim == 2 and wrench_xyz.shape[1] == 3:
      set_plot("wrench", t, [wrench_xyz[:, 0], wrench_xyz[:, 1], wrench_xyz[:, 2]])

    omega_xyz = _get(frame, "omega_xyz")
    if omega_xyz is not None and omega_xyz.ndim == 2 and omega_xyz.shape[1] == 3:
      set_plot("omega", t, [omega_xyz[:, 0], omega_xyz[:, 1], omega_xyz[:, 2]])

    # Thrust (stage axis)
    t_u = _get(frame, "t_u")
    thrust_stage = _get(frame, "thrust_N")
    if t_u is not None and thrust_stage is not None:
      set_plot("thrust", t_u, [thrust_stage])

    # Cost (stage axis + total axis)
    t_cost = _get(frame, "t_cost")
    t_total = _get(frame, "t_total")
    J_pos = _get(frame, "J_pos")
    J_yaw = _get(frame, "J_yaw")
    J_th = _get(frame, "J_thrust")
    J_tot = _get(frame, "J_total")
    if t_cost is not None and t_total is not None and J_pos is not None and J_yaw is not None and J_th is not None and J_tot is not None:
      set_plot("cost", [t_cost, t_cost, t_cost, t_total], [J_pos, J_yaw, J_th, J_tot])

    # Solve-time history (viewer-side ring)
    solve_ms_scalar = _get(frame, "solve_ms_scalar")
    if solve_ms_scalar is not None and solve_ms_scalar.size >= 1:
      x_idx, ms_vals = self._push_solve_ms(float(solve_ms_scalar[0]))
      set_plot("solve", x_idx, [ms_vals])

def main() -> int:

  app = QApplication(sys.argv)

  # White theme
  app.setStyleSheet("""
  QWidget { background: #ffffff; color: #111111; }
  QMainWindow { background: #ffffff; }
  QGroupBox {
    background: #ffffff;
    border: 1px solid #cfcfcf;
    border-radius: 6px;
    margin-top: 10px;
  }
  QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #111111;
  }
  QSplitter::handle {
    background: #ffffff;
  }
  QSplitter::handle:horizontal {
    width: 3px;
  }
  QSplitter::handle:vertical {
    height: 3px;
  }
  QLabel { background: transparent; }
  """)

  def _cleanup() -> None:
    nonlocal reader
    if reader is not None:
      try: reader.close()
      except Exception: pass
      reader = None

  def _on_sigint(*_args) -> None:
    app.quit()

  # Route Ctrl+C (SIGINT) to Qt quit.
  signal.signal(signal.SIGINT, _on_sigint)

  # Some platforms need a small timer tick so Python can process signals during Qt event loop.
  sig_timer = QTimer()
  sig_timer.timeout.connect(lambda: None)
  sig_timer.start(100)

  win = DebugViewerMainWindow()
  win.show()

  path = os.environ.get("STRIDER_MPC_MMAP", "/tmp/strider_mpc_debug.mmap")
  reader: Optional[MMapReader] = None
  last_seq: Optional[int] = None

  def poll_mmap() -> None:
    nonlocal reader, last_seq
    try:
      if reader is None:
        if not os.path.exists(path):
          return
        reader = MMapReader(path)

      pkt = reader.read_latest()
      if pkt is None:
        return

      seq = int(pkt.seq)
      if last_seq is not None and seq == last_seq:
        return
      last_seq = seq

      frame = compute_frame_from_pkt(pkt)
      win.update_from_frame(frame)

    except Exception:
      # English comments only.
      print("[viewer] poll_mmap error:")
      traceback.print_exc()

  tm = QTimer()
  tm.timeout.connect(poll_mmap)
  tm.start(30)  # ~33Hz polling; update is skipped if seq unchanged

  def _cleanup() -> None:
    nonlocal reader
    if reader is not None:
      try:
        reader.close()
      except Exception:
        pass
      reader = None

  app.aboutToQuit.connect(_cleanup)
  return int(app.exec_())

if __name__ == "__main__":
  raise SystemExit(main())
