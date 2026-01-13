# mpc_viewer.py
from __future__ import annotations

import os
import sys
import time
import signal
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
  QApplication,
  QGroupBox,
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
  __package__ = THIS_DIR.name  # e.g. "mpc_py"

from .mmap_manager import MMapReader, MMapPacket
from . import params as p

DebugFrame = Dict[str, Optional[np.ndarray]]

def _get(frame: DebugFrame, key: str) -> Optional[np.ndarray]:
  return frame.get(key, None)

# ----------------------------
# Math utils (NumPy)
# ----------------------------
def euler_zyx_to_R_np(theta: np.ndarray) -> np.ndarray:
  phi, th, psi = float(theta[0]), float(theta[1]), float(theta[2])

  cphi, sphi = np.cos(phi), np.sin(phi)
  cth,  sth  = np.cos(th),  np.sin(th)
  cpsi, spsi = np.cos(psi), np.sin(psi)

  Rz = np.array([[cpsi, -spsi, 0.0],
                 [spsi,  cpsi, 0.0],
                 [0.0,   0.0,  1.0]], dtype=np.float64)
  Ry = np.array([[ cth, 0.0,  sth],
                 [0.0,  1.0,  0.0],
                 [-sth, 0.0,  cth]], dtype=np.float64)
  Rx = np.array([[1.0, 0.0,   0.0],
                 [0.0, cphi, -sphi],
                 [0.0, sphi,  cphi]], dtype=np.float64)

  return Rz @ Ry @ Rx

def R_to_euler_zyx_np(R: np.ndarray) -> np.ndarray:
  R = np.asarray(R, dtype=np.float64).reshape(3, 3)

  s = -float(R[2, 0])
  s = float(np.clip(s, -1.0, 1.0))
  pitch = np.arcsin(s)

  c_pitch = float(np.cos(pitch))
  if abs(c_pitch) < 1e-8:
    yaw = np.arctan2(-float(R[0, 1]), float(R[1, 1]))
    roll = 0.0
  else:
    roll = np.arctan2(float(R[2, 1]), float(R[2, 2]))
    yaw  = np.arctan2(float(R[1, 0]), float(R[0, 0]))

  return np.array([roll, pitch, yaw], dtype=np.float64)

def hat_np(w: np.ndarray) -> np.ndarray:
  wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
  return np.array([
    [0.0, -wz,  wy],
    [wz,  0.0, -wx],
    [-wy,  wx,  0.0],
  ], dtype=np.float64)

def expm_hat_np(w: np.ndarray) -> np.ndarray:
  w = np.asarray(w, dtype=np.float64).reshape(3,)
  th2 = float(w @ w)
  th = np.sqrt(th2 + 1e-12)

  A = np.sin(th) / th
  B = (1.0 - np.cos(th)) / (th2 + 1e-12)

  K = hat_np(w)
  I = np.eye(3, dtype=np.float64)
  return I + A * K + B * (K @ K)

def vee_np(M: np.ndarray) -> np.ndarray:
  return np.array([float(M[2, 1]), float(M[0, 2]), float(M[1, 0])], dtype=np.float64)

# ----------------------------
# Debug-frame builder
# ----------------------------
def compute_frame_from_pkt(pkt: MMapPacket) -> DebugFrame:
  N = int(pkt.N)
  nx = int(pkt.nx)
  nu = int(pkt.nu)
  np_ = int(pkt.np)

  dt = float(p.DT)
  T = N + 1

  x_all = np.asarray(pkt.x_all, dtype=np.float64).reshape(T, nx)
  u_all = np.asarray(pkt.u_all, dtype=np.float64).reshape(N, nu) if N > 0 else np.zeros((0, nu), dtype=np.float64)
  p_all = np.asarray(pkt.p_all, dtype=np.float64).reshape(T, np_)

  # log_param (10): [pos_cur(3), pos_des(3), F1..F4(4)]
  log_param = None
  if hasattr(pkt, "log_param"):
    try:
      log_param = np.asarray(pkt.log_param, dtype=np.float64).ravel()
    except Exception:
      log_param = None

  pos_cur = np.full((3,), np.nan, dtype=np.float64)
  pos_des = np.full((3,), np.nan, dtype=np.float64)
  thrust_log = np.full((4,), np.nan, dtype=np.float64)
  if log_param is not None and log_param.size >= 10:
    pos_cur[:] = log_param[0:3]
    pos_des[:] = log_param[3:6]
    thrust_log[:] = log_param[6:10]

  # Current convention (augmented):
  # x = [theta(3), omega(3), r_cot(2), delta_theta_cmd(3), r_cot_cmd(2)] => nx=13
  # u = u_rate = [delta_theta_cmd_rate(3), r_cot_cmd_rate(2)] => nu=5
  #
  # Legacy fallback:
  # x = [theta(3), omega(3), r_cot(2)] => nx=8
  # u = [delta_theta_cmd(3), r_cot_cmd(2)] => nu=5
  use_aug = (nx >= 13)

  theta = x_all[:, 0:3]
  omega = x_all[:, 3:6]
  r_cot = x_all[:, 6:8]

  if use_aug:
    delta_theta_cmd = x_all[:, 8:11]
    r_cot_cmd = x_all[:, 11:13]
  else:
    u_cmd_T = np.zeros((T, 5), dtype=np.float64)
    if u_all.shape[0] == N and u_all.shape[1] >= 5:
      u_cmd_T[:N, :] = u_all[:, 0:5]
      u_cmd_T[N, :] = u_cmd_T[N - 1, :] if N > 0 else 0.0
    delta_theta_cmd = u_cmd_T[:, 0:3]
    r_cot_cmd = u_cmd_T[:, 3:5]

  # Stage-axis r_cot_cmd for plotting (same placement as before)
  r_cot_cmd_stage = np.zeros((N, 2), dtype=np.float64)
  if N > 0:
    r_cot_cmd_stage[:, :] = r_cot_cmd[:N, :]

  # Stage-axis u_rate (for cost)
  u_rate_stage = np.zeros((N, 5), dtype=np.float64)
  if use_aug and (N > 0) and (u_all.shape[0] == N) and (u_all.shape[1] >= 5):
    u_rate_stage[:, :] = u_all[:, 0:5]

  # p = [vec(R_raw)(9), l(1), T_des(1)] => np=11
  R_raw_all = np.zeros((T, 3, 3), dtype=np.float64)
  l_all = np.zeros((T,), dtype=np.float64)
  T_des_all = np.zeros((T,), dtype=np.float64)

  for k in range(T):
    pv = p_all[k, :].ravel()
    if pv.size < 11:
      continue
    Rk = pv[0:9].reshape(3, 3, order="F")
    R_raw_all[k, :, :] = Rk
    l_all[k] = float(pv[9])
    T_des_all[k] = float(pv[10])

  # Reference attitude from R_raw
  theta_ref = np.zeros_like(theta)
  for k in range(T):
    theta_ref[k, :] = R_to_euler_zyx_np(R_raw_all[k, :, :])

  # tau_d and R_d reconstruction
  KR = float(getattr(p, "KR", 8.0))
  KW = float(getattr(p, "KW", 5.0))
  zeta = float(getattr(p, "ZETA", 0.02))

  tau_d_all = np.full((T, 3), np.nan, dtype=np.float64)
  theta_d_all = np.full((T, 3), np.nan, dtype=np.float64)
  F_stage = np.full((N, 4), np.nan, dtype=np.float64)

  for k in range(T):
    R = euler_zyx_to_R_np(theta[k, :])
    R_d = R_raw_all[k, :, :] @ expm_hat_np(delta_theta_cmd[k, :])
    theta_d_all[k, :] = R_to_euler_zyx_np(R_d)

    e_R = 0.5 * vee_np(R_d.T @ R - R.T @ R_d)
    tau_d = -KR * e_R - KW * omega[k, :]
    tau_d_all[k, :] = tau_d

  # Thrust reconstruction (use r_cot state, consistent with con_h_expr)
  for k in range(N):
    dx = float(r_cot[k, 0])
    dy = float(r_cot[k, 1])
    l = float(l_all[k])
    T_des = float(T_des_all[k])
    tau_d = tau_d_all[k, :]

    A = np.array([
      [ l - dy,  l - dy, -l - dy, -l - dy],
      [ l + dx, -l + dx, -l + dx,  l + dx],
      [  -zeta,    zeta,   -zeta,    zeta],
      [    1.0,     1.0,     1.0,     1.0],
    ], dtype=np.float64)
    w_d = np.array([tau_d[0], tau_d[1], tau_d[2], T_des], dtype=np.float64)

    try:
      F_stage[k, :] = np.linalg.solve(A, w_d)
    except np.linalg.LinAlgError:
      F_stage[k, :] = np.linalg.lstsq(A, w_d, rcond=None)[0]

  # Cost breakdown (stage-wise)
  Q_OMEGA = np.asarray(getattr(p, "Q_OMEGA", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
  Q_THETA = np.asarray(getattr(p, "Q_THETA", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
  Q_COT   = np.asarray(getattr(p, "Q_COT",   [1.0, 1.0]), dtype=np.float64).ravel()
  R_THETA = np.asarray(getattr(p, "R_THETA", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
  R_COT   = np.asarray(getattr(p, "R_COT",   [1.0, 1.0]), dtype=np.float64).ravel()

  J_omega = np.zeros((N,), dtype=np.float64)
  J_u_ch = np.zeros((N, 5), dtype=np.float64)
  J_total = np.zeros((N,), dtype=np.float64)

  for k in range(N):
    om = omega[k, :]
    J_omega[k] = float(np.sum(Q_OMEGA * (om * om)))

    dtc = delta_theta_cmd[k, :]
    rcc = r_cot_cmd[k, :]

    ur = u_rate_stage[k, :]
    dtc_rate = ur[0:3]
    rcc_rate = ur[3:5]

    J_u_ch[k, 0] = float(Q_THETA[0] * dtc[0] * dtc[0] + R_THETA[0] * dtc_rate[0] * dtc_rate[0])
    J_u_ch[k, 1] = float(Q_THETA[1] * dtc[1] * dtc[1] + R_THETA[1] * dtc_rate[1] * dtc_rate[1])
    J_u_ch[k, 2] = float(Q_THETA[2] * dtc[2] * dtc[2] + R_THETA[2] * dtc_rate[2] * dtc_rate[2])
    J_u_ch[k, 3] = float(Q_COT[0]   * rcc[0] * rcc[0] + R_COT[0]   * rcc_rate[0] * rcc_rate[0])
    J_u_ch[k, 4] = float(Q_COT[1]   * rcc[1] * rcc[1] + R_COT[1]   * rcc_rate[1] * rcc_rate[1])

    J_total[k] = J_omega[k] + float(np.sum(J_u_ch[k, :]))

  # Time axes (horizon)
  t = np.arange(T, dtype=np.float64) * dt
  t_u = np.arange(N, dtype=np.float64) * dt

  # Bounds
  Fmin = float(np.min(np.asarray(getattr(p, "F_MIN", [0, 0, 0, 0]), dtype=np.float64).ravel()))
  Fmax = float(np.max(np.asarray(getattr(p, "F_MAX", [0, 0, 0, 0]), dtype=np.float64).ravel()))
  F_min_line = np.full((N,), Fmin, dtype=np.float64)
  F_max_line = np.full((N,), Fmax, dtype=np.float64)

  cot_min = np.asarray(getattr(p, "COT_MIN", [-0.2, -0.2]), dtype=np.float64).ravel()
  cot_max = np.asarray(getattr(p, "COT_MAX", [ 0.2,  0.2]), dtype=np.float64).ravel()
  cotx_min_line = np.full((T,), float(cot_min[0]), dtype=np.float64)
  cotx_max_line = np.full((T,), float(cot_max[0]), dtype=np.float64)
  coty_min_line = np.full((T,), float(cot_min[1]), dtype=np.float64)
  coty_max_line = np.full((T,), float(cot_max[1]), dtype=np.float64)

  # T_des on stage axis
  T_stage = np.full((N,), float(T_des_all[0]) if T_des_all.size else 0.0, dtype=np.float64)
  if T_des_all.size >= N:
    T_stage[:] = T_des_all[:N]

  # Plot conversions
  rad2deg = 180.0 / np.pi
  theta_plot = theta * rad2deg
  theta_ref_plot = theta_ref * rad2deg
  omega_plot = omega * rad2deg
  theta_d_plot = theta_d_all * rad2deg

  # ----------------------------
  # Top-right (cumulative) single-sample signals @ k=0
  # ----------------------------
  rpy_act0 = np.full((3,), np.nan, dtype=np.float64)
  rpy_raw0 = np.full((3,), np.nan, dtype=np.float64)
  rpy_des0 = np.full((3,), np.nan, dtype=np.float64)
  cot0 = np.full((2,), np.nan, dtype=np.float64)
  F0 = np.full((4,), np.nan, dtype=np.float64)

  if T >= 1:
    rpy_act0[:] = theta_plot[0, :]
    rpy_raw0[:] = theta_ref_plot[0, :]
    rpy_des0[:] = theta_d_plot[0, :]
    cot0[:] = r_cot[0, :]

  # Prefer thrust_log for top-right F0 (and thus tau plots).
  F0_src = None
  if (log_param is not None) and (log_param.size >= 10):
    Flog = log_param[6:10]
    if np.isfinite(Flog).all():
      F0_src = Flog

  # Fallback to reconstructed F if log is missing/invalid.
  if F0_src is None and N >= 1 and np.isfinite(F_stage[0, :]).all():
    F0_src = F_stage[0, :]

  if F0_src is not None:
    F0[:] = np.asarray(F0_src, dtype=np.float64).reshape(4,)

  # tau_thrust (A_thrust @ F0) xy + tau_cot (r_cot x [0,0,sum(F0)]) xy
  tau_thrust0_xy = np.full((2,), np.nan, dtype=np.float64)
  tau_cot0_xy = np.full((2,), np.nan, dtype=np.float64)
  tau_total0_xy = np.full((2,), np.nan, dtype=np.float64)

  l0 = float(l_all[0]) if l_all.size >= 1 else float("nan")
  if np.isfinite(l0) and np.isfinite(zeta) and np.isfinite(F0).all():
    A_th = np.array([
      [  l0,   l0,  -l0,  -l0],
      [  l0,  -l0,  -l0,   l0],
      [-zeta, zeta, -zeta, zeta],
      [ 1.0,  1.0,  1.0,  1.0],
    ], dtype=np.float64)

    w = A_th @ F0.reshape(4,)
    tau_thrust0_xy[:] = w[0:2]

    Ttot = float(np.sum(F0))
    if np.isfinite(cot0).all() and np.isfinite(Ttot):
      dx0, dy0 = float(cot0[0]), float(cot0[1])
      # r x F, with r=[dx,dy,0], F=[0,0,Ttot]
      tau_cot0_xy[0] = dy0 * Ttot
      tau_cot0_xy[1] = -dx0 * Ttot

    if np.isfinite(tau_cot0_xy).all():
      tau_total0_xy[:] = tau_thrust0_xy + tau_cot0_xy

  return {
    "t": t,
    "t_u": t_u,

    "theta_plot": theta_plot,
    "theta_ref_plot": theta_ref_plot,
    "theta_d_plot": theta_d_plot,

    "r_cot": r_cot,
    "r_cot_cmd": r_cot_cmd,
    "cotx_min_line": cotx_min_line,
    "cotx_max_line": cotx_max_line,
    "coty_min_line": coty_min_line,
    "coty_max_line": coty_max_line,

    "omega_plot": omega_plot,

    "r_cot_cmd_stage": r_cot_cmd_stage,

    "F_stage": F_stage,
    "F_min_line": F_min_line,
    "F_max_line": F_max_line,

    "J_omega": J_omega,
    "J_u_ch": J_u_ch,     # (N,5)
    "J_total": J_total,   # (N,)

    "tau_d": tau_d_all,
    "T_des": T_stage,

    "solve_ms_scalar": np.array([float(pkt.solve_ms)], dtype=np.float64),

    # log-derived (single sample per packet; viewer will build history)
    "log_pos_cur": pos_cur,
    "log_pos_des": pos_des,
    "log_thrust": thrust_log,

    # top-right cumulative (single sample per packet)
    "top_rpy_act0": rpy_act0,
    "top_rpy_raw0": rpy_raw0,
    "top_rpy_des0": rpy_des0,
    "top_cot0": cot0,
    "top_F0": F0,
    "top_tau_thrust0_xy": tau_thrust0_xy,
    "top_tau_cot0_xy": tau_cot0_xy,
    "top_tau_total0_xy": tau_total0_xy,
  }

# ----------------------------
# Main window
# ----------------------------
class DebugViewerMainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("STRIDER MRG Viewer")
    self.resize(1900, 1000)

    pg.setConfigOptions(antialias=True)

    # History buffers (solve_ms)
    self._solve_hist_size = 200
    self._solve_ms_hist = np.full(self._solve_hist_size, np.nan, dtype=np.float64)
    self._solve_count = 0

    # History buffers (Top-right cumulative time-series)
    self._tr_hist_size = 200
    self._tr_count = 0
    self._tr_t0 = time.perf_counter()
    self._tr_t = np.full((self._tr_hist_size,), np.nan, dtype=np.float64)

    self._tr_pos_cur = np.full((self._tr_hist_size, 3), np.nan, dtype=np.float64)
    self._tr_pos_des = np.full((self._tr_hist_size, 3), np.nan, dtype=np.float64)

    self._tr_rpy_act = np.full((self._tr_hist_size, 3), np.nan, dtype=np.float64)
    self._tr_rpy_raw = np.full((self._tr_hist_size, 3), np.nan, dtype=np.float64)
    self._tr_rpy_des = np.full((self._tr_hist_size, 3), np.nan, dtype=np.float64)

    self._tr_F = np.full((self._tr_hist_size, 4), np.nan, dtype=np.float64)
    self._tr_cot = np.full((self._tr_hist_size, 2), np.nan, dtype=np.float64)

    self._tr_tau_thrust = np.full((self._tr_hist_size, 2), np.nan, dtype=np.float64)
    self._tr_tau_cot = np.full((self._tr_hist_size, 2), np.nan, dtype=np.float64)
    self._tr_tau_total = np.full((self._tr_hist_size, 2), np.nan, dtype=np.float64)

    central = QWidget()
    self.setCentralWidget(central)

    root = QVBoxLayout(central)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    v_splitter = QSplitter(Qt.Vertical)
    root.addWidget(v_splitter)

    # ----------------------------
    # Top region: Horizontal splitter
    # ----------------------------
    top_container = QWidget()
    top_layout = QVBoxLayout(top_container)
    top_layout.setContentsMargins(0, 0, 0, 0)

    h_splitter = QSplitter(Qt.Horizontal)
    top_layout.addWidget(h_splitter)

    # Top-left: 3D shell (VisPy)
    self.gb_3d = QGroupBox()
    self.gb_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    l3 = QVBoxLayout(self.gb_3d)
    l3.setContentsMargins(0, 0, 0, 0)

    self.canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="white")
    l3.addWidget(self.canvas.native)

    self.view = self.canvas.central_widget.add_view()
    self.view.camera = "turntable"
    self.view.camera.fov = 6
    self.view.camera.center = (0, 0, -1)
    self.view.camera.distance = 28.0
    self.view.camera.elevation = 40
    self.view.camera.azimuth = 15

    grid = visuals.GridLines(color=(0.1, 0.1, 0.1, 1.0), parent=self.view.scene)
    grid.set_gl_state("translucent", depth_test=False)

    axis_len = 0.2
    axis_w = 3
    visuals.Line(
      pos=np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=np.float32),
      color=(1, 0, 0, 1),
      width=axis_w,
      parent=self.view.scene,
      method="gl",
    )
    visuals.Line(
      pos=np.array([[0, 0, 0], [0, axis_len, 0]], dtype=np.float32),
      color=(0, 1, 0, 1),
      width=axis_w,
      parent=self.view.scene,
      method="gl",
    )
    visuals.Line(
      pos=np.array([[0, 0, 0], [0, 0, axis_len]], dtype=np.float32),
      color=(0, 0, 1, 1),
      width=axis_w,
      parent=self.view.scene,
      method="gl",
    )

    # 3D trajectories: cur (blue solid), des (black dashed)
    self._traj_cur = visuals.Line(
      pos=np.zeros((0, 3), dtype=np.float32),
      color=(0.2, 0.35, 0.95, 1.0),
      width=2,
      parent=self.view.scene,
      method="gl",
    )
    self._traj_des = visuals.Line(
      pos=np.zeros((0, 3), dtype=np.float32),
      color=(0.0, 0.0, 0.0, 1.0),
      width=2,
      parent=self.view.scene,
      method="gl",
    )

    self._pt_cur = visuals.Markers(parent=self.view.scene)
    self._pt_des = visuals.Markers(parent=self.view.scene)

    # Top-right: 3x3 plots (cumulative time-series)
    self.gb_top_empty = QGroupBox()
    self.gb_top_empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    top_r_layout = QVBoxLayout(self.gb_top_empty)
    top_r_layout.setContentsMargins(0, 0, 0, 0)

    top_r_gridw = QWidget()
    top_r_gl = pg.QtWidgets.QGridLayout(top_r_gridw)
    top_r_gl.setContentsMargins(6, 6, 6, 6)
    top_r_gl.setSpacing(6)
    top_r_layout.addWidget(top_r_gridw)

    h_splitter.addWidget(self.gb_3d)
    h_splitter.addWidget(self.gb_top_empty)
    h_splitter.setStretchFactor(0, 6)
    h_splitter.setStretchFactor(1, 4)

    v_splitter.addWidget(top_container)

    self.gb_3d.setMinimumWidth(1)
    self.gb_top_empty.setMinimumWidth(1)
    self.canvas.native.setMinimumWidth(1)
    top_r_gridw.setMinimumWidth(1)

    # ----------------------------
    # Bottom region: 5x3 plot group
    # ----------------------------
    self.gb_under = QGroupBox()
    self.gb_under.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    l2 = QVBoxLayout(self.gb_under)
    l2.setContentsMargins(0, 0, 0, 0)

    gridw = QWidget()
    gl = pg.QtWidgets.QGridLayout(gridw)
    gl.setContentsMargins(0, 0, 0, 0)
    gl.setSpacing(6)

    lw = 3  # thicker lines globally

    # RPY style requirement:
    # act = blue solid, raw = black dashed, des = red dashed
    pen_rpy = pg.mkPen(color=(50, 90, 220), width=lw)
    pen_raw = pg.mkPen(color=(0, 0, 0), width=lw, style=Qt.DashLine)
    pen_des = pg.mkPen(color=(220, 50, 50), width=lw, style=Qt.DashLine)

    pens = [
      pg.mkPen(color=(220, 50, 50), width=lw),   # red
      pg.mkPen(color=(50, 160, 70), width=lw),   # green
      pg.mkPen(color=(50, 90, 220), width=lw),   # blue
      pg.mkPen(color=(120, 60, 200), width=lw),  # purple
      pg.mkPen(color=(0, 0, 0), width=lw),       # black
      pg.mkPen(color=(255, 140, 0), width=lw),   # orange
      pg.mkPen(color=(0, 150, 150), width=lw),   # teal
    ]
    pen_dash = pg.mkPen(color=(0, 0, 0), width=lw, style=Qt.DashLine)
    pen_dash_red = pg.mkPen(color=(220, 50, 50), width=lw, style=Qt.DashLine)

    # Top-right pos pens (cur: blue solid, des: black dashed)
    pen_pos_cur = pg.mkPen(color=(50, 90, 220), width=lw)
    pen_pos_des = pg.mkPen(color=(0, 0, 0), width=lw, style=Qt.DashLine)

    # Top-right tau pens
    pen_tau_thrust = pens[0]  # red
    pen_tau_cot = pens[1]     # green
    pen_tau_total = pg.mkPen(color=(0, 0, 0), width=lw, style=Qt.DashLine)

    def make_plot(
      title: str,
      y_label: str,
      curve_specs: List[Tuple[Optional[str], Any]],
      y_range: Optional[Tuple[float, float]] = None,
    ):
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

      if y_range is not None:
        pi.setYRange(float(y_range[0]), float(y_range[1]), padding=0.0)

      legend = pi.addLegend()
      legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

      curves = []
      for name, pen in curve_specs:
        if name is None:
          curves.append(w.plot(pen=pen))          # no legend entry
        else:
          curves.append(w.plot(name=name, pen=pen))
      return w, curves

    self._plots: Dict[str, Tuple[pg.PlotWidget, list[Any]]] = {}

    # ----------------------------
    # Top-right plots (3x3, cumulative time-series)
    # Row0: posx posy posz
    # Row1: roll pitch F1234
    # Row2: tau_x tau_y cot_xy
    # ----------------------------
    self._plots["posx"] = make_plot("pos_x", "[m]", [("cur", pen_pos_cur), ("des", pen_pos_des)])
    self._plots["posy"] = make_plot("pos_y", "[m]", [("cur", pen_pos_cur), ("des", pen_pos_des)])
    self._plots["posz"] = make_plot("pos_z", "[m]", [("cur", pen_pos_cur), ("des", pen_pos_des)])

    self._plots["roll_tr"] = make_plot(
      "Roll", "[deg]",
      [("act", pen_rpy), ("raw", pen_raw), ("des", pen_des)],
      y_range=(-50.0, 50.0),
    )
    self._plots["pitch_tr"] = make_plot(
      "Pitch", "[deg]",
      [("act", pen_rpy), ("raw", pen_raw), ("des", pen_des)],
      y_range=(-50.0, 50.0),
    )

    self._plots["F_tr"] = make_plot(
      "F1234", "[N]",
      [("F1", pens[0]), ("F2", pens[1]), ("F3", pens[2]), ("F4", pens[3])],
      y_range=(10.0, 15.0),
    )

    self._plots["tau_x_tr"] = make_plot(
      "torque_x", "[Nm]",
      [("tau_thrust", pen_tau_thrust), ("tau_off", pen_tau_cot), ("total", pen_tau_total)],
    )
    self._plots["tau_y_tr"] = make_plot(
      "tau_y", "[Nm]",
      [("tau_thrust", pen_tau_thrust), ("tau_off", pen_tau_cot), ("total", pen_tau_total)],
    )

    self._plots["cotxy_tr"] = make_plot(
      "r_cot", "[m]",
      [("x", pens[2]), ("y", pens[0])],
    )

    top_r_gl.addWidget(self._plots["posx"][0], 0, 0)
    top_r_gl.addWidget(self._plots["posy"][0], 0, 1)
    top_r_gl.addWidget(self._plots["posz"][0], 0, 2)

    top_r_gl.addWidget(self._plots["roll_tr"][0], 1, 0)
    top_r_gl.addWidget(self._plots["pitch_tr"][0], 1, 1)
    top_r_gl.addWidget(self._plots["F_tr"][0], 1, 2)

    top_r_gl.addWidget(self._plots["tau_x_tr"][0], 2, 0)
    top_r_gl.addWidget(self._plots["tau_y_tr"][0], 2, 1)
    top_r_gl.addWidget(self._plots["cotxy_tr"][0], 2, 2)

    # ---- Bottom grid (existing) ----
    # ---- Row 1 ----
    self._plots["roll"]  = make_plot(
      "Roll",  "[deg]",
      [("r", pen_rpy), ("r_raw", pen_raw), ("r_des", pen_des)],
      y_range=(-30.0, 30.0),
    )
    self._plots["pitch"] = make_plot(
      "Pitch", "[deg]",
      [("p", pen_rpy), ("p_raw", pen_raw), ("p_des", pen_des)],
      y_range=(-30.0, 30.0),
    )
    self._plots["yaw"]   = make_plot(
      "Yaw",   "[deg]",
      [("y", pen_rpy), ("y_raw", pen_raw), ("y_des", pen_des)],
      y_range=(-30.0, 30.0),
    )

    self._plots["cotx"]  = make_plot(
      "r_cot_x", "[m]",
      [("x", pens[2]), ("cmd", pens[0]), (None, pen_dash), (None, pen_dash_red)]
    )
    self._plots["coty"]  = make_plot(
      "r_cot_y", "[m]",
      [("y", pens[2]), ("cmd", pens[0]), (None, pen_dash), (None, pen_dash_red)]
    )

    # ---- Row 2 ----
    self._plots["wx"]    = make_plot("omega_x", "[deg/s]", [("wx", pens[2])], y_range=(-25., 25.))
    self._plots["wy"]    = make_plot("omega_y", "[deg/s]", [("wy", pens[2])], y_range=(-25., 25.))
    self._plots["wz"]    = make_plot("omega_z", "[deg/s]", [("wz", pens[2])], y_range=(-25., 25.))

    self._plots["cotux"] = make_plot("r_cot_cmd_x (u)", "[m]", [("u_x", pens[2])], y_range=(-0.05, 0.05))
    self._plots["cotuy"] = make_plot("r_cot_cmd_y (u)", "[m]", [("u_y", pens[2])], y_range=(-0.05, 0.05))

    # ---- Row 3 ----
    self._plots["F"] = make_plot(
      "F1234 (recon)", "[N]",
      [("F1", pens[0]), ("F2", pens[1]), ("F3", pens[2]), ("F4", pens[3]),
       (None, pen_dash), (None, pen_dash_red)]
    )

    pen_total = pg.mkPen(color=(0, 0, 0), width=lw, style=Qt.DashLine)
    self._plots["cost"] = make_plot(
      "Cost breakdown", "cost",
      [("J_omega", pens[2]),
       ("J_dth_x", pens[0]),
       ("J_dth_y", pens[1]),
       ("J_dth_z", pens[3]),
       ("J_rcot_x", pens[5]),
       ("J_rcot_y", pens[6]),
       ("J_total", pen_total)]
    )

    self._plots["solve"] = make_plot("Solve time", "[ms]", [("ms", pens[2])])

    self._plots["tau"] = make_plot(
      "tau_des (tau_d)", "[Nm]",
      [("tau_x", pens[0]), ("tau_y", pens[1]), ("tau_z", pens[2])],
      y_range=(-2.0, 2.0),
    )

    self._plots["T"] = make_plot(
      "T_des", "[N]",
      [("T_des", pens[2])],
      y_range=(40.0, 55.0),
    )

    cells = [
      ["roll", "pitch", "yaw",  "cotx",  "coty"],
      ["wx",   "wy",    "wz",   "cotux", "cotuy"],
      ["F",    "cost",  "solve","tau",   "T"],
    ]
    for r in range(3):
      for c in range(5):
        key = cells[r][c]
        w, _ = self._plots[key]
        gl.addWidget(w, r, c)

    l2.addWidget(gridw)
    v_splitter.addWidget(self.gb_under)

    v_splitter.setStretchFactor(0, 10)
    v_splitter.setStretchFactor(1, 4)

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

  def _push_tr(
    self,
    pos_cur: np.ndarray,
    pos_des: np.ndarray,
    rpy_act: np.ndarray,
    rpy_raw: np.ndarray,
    rpy_des: np.ndarray,
    F0: np.ndarray,
    cot0: np.ndarray,
    tau_thrust_xy: np.ndarray,
    tau_cot_xy: np.ndarray,
    tau_total_xy: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = self._tr_count % self._tr_hist_size

    t_now = time.perf_counter() - self._tr_t0
    self._tr_t[idx] = float(t_now)

    self._tr_pos_cur[idx, :] = np.asarray(pos_cur, dtype=np.float64).reshape(3,)
    self._tr_pos_des[idx, :] = np.asarray(pos_des, dtype=np.float64).reshape(3,)

    self._tr_rpy_act[idx, :] = np.asarray(rpy_act, dtype=np.float64).reshape(3,)
    self._tr_rpy_raw[idx, :] = np.asarray(rpy_raw, dtype=np.float64).reshape(3,)
    self._tr_rpy_des[idx, :] = np.asarray(rpy_des, dtype=np.float64).reshape(3,)

    self._tr_F[idx, :] = np.asarray(F0, dtype=np.float64).reshape(4,)
    self._tr_cot[idx, :] = np.asarray(cot0, dtype=np.float64).reshape(2,)

    self._tr_tau_thrust[idx, :] = np.asarray(tau_thrust_xy, dtype=np.float64).reshape(2,)
    self._tr_tau_cot[idx, :] = np.asarray(tau_cot_xy, dtype=np.float64).reshape(2,)
    self._tr_tau_total[idx, :] = np.asarray(tau_total_xy, dtype=np.float64).reshape(2,)

    self._tr_count += 1

    count = min(self._tr_count, self._tr_hist_size)
    idx_end = self._tr_count
    idx_start = max(0, idx_end - count)
    indices = np.array([(i % self._tr_hist_size) for i in range(idx_start, idx_end)], dtype=np.int64)

    tt = self._tr_t[indices]

    pc = self._tr_pos_cur[indices, :]
    pd = self._tr_pos_des[indices, :]

    ra = self._tr_rpy_act[indices, :]
    rr = self._tr_rpy_raw[indices, :]
    rd = self._tr_rpy_des[indices, :]

    FF = self._tr_F[indices, :]
    cc = self._tr_cot[indices, :]

    tth = self._tr_tau_thrust[indices, :]
    tco = self._tr_tau_cot[indices, :]
    tto = self._tr_tau_total[indices, :]

    return tt, pc, pd, ra, rr, rd, FF, cc, tth, tco, tto

  def _update_3d_traj(self, cur_pts: np.ndarray, des_pts: np.ndarray) -> None:
    cur_pts = np.asarray(cur_pts, dtype=np.float64).reshape(-1, 3)
    des_pts = np.asarray(des_pts, dtype=np.float64).reshape(-1, 3)

    # Filter finite rows
    mcur = np.isfinite(cur_pts).all(axis=1)
    mdes = np.isfinite(des_pts).all(axis=1)
    cur = cur_pts[mcur, :]
    des = des_pts[mdes, :]

    # Update current (solid)
    if cur.shape[0] >= 2:
      self._traj_cur.set_data(pos=cur.astype(np.float32), connect="strip")
    else:
      self._traj_cur.set_data(pos=np.zeros((0, 3), dtype=np.float32), connect="strip")

    # Update desired (dashed via connect mask)
    if des.shape[0] >= 2:
      M = des.shape[0]
      pat = np.array([True, False], dtype=bool)
      conn = np.tile(pat, int(np.ceil((M - 1) / pat.size)))[:(M - 1)]
      self._traj_des.set_data(pos=des.astype(np.float32), connect=conn)
    else:
      self._traj_des.set_data(pos=np.zeros((0, 3), dtype=np.float32), connect="strip")

    # Update last-point markers
    if cur.shape[0] >= 1:
      self._pt_cur.set_data(
        pos=cur[-1:, :].astype(np.float32),
        face_color=(0.0, 0.0, 1.0, 1.0),
        edge_color=(0.0, 0.0, 0.0, 1.0),
        size=6,
        symbol="disc",
      )
    else:
      self._pt_cur.set_data(pos=np.zeros((0, 3), dtype=np.float32))

    if des.shape[0] >= 1:
      self._pt_des.set_data(
        pos=des[-1:, :].astype(np.float32),
        face_color=(0.0, 0.0, 0.0, 1.0),
        edge_color=(0.0, 0.0, 0.0, 1.0),
        size=6,
        symbol="disc",
      )
    else:
      self._pt_des.set_data(pos=np.zeros((0, 3), dtype=np.float32))

  def update_from_frame(self, frame: DebugFrame) -> None:
    t = _get(frame, "t")
    if t is None:
      return

    def set_plot(key: str, xs: Any, ys: list[np.ndarray]) -> None:
      _, curves = self._plots[key]
      for c, y in zip(curves, ys):
        c.setData(xs, y)

    # ----------------------------
    # Top-right (cumulative) + Top-left 3D
    # ----------------------------
    def _nan3() -> np.ndarray:
      return np.full((3,), np.nan, dtype=np.float64)

    def _nan4() -> np.ndarray:
      return np.full((4,), np.nan, dtype=np.float64)

    def _nan2() -> np.ndarray:
      return np.full((2,), np.nan, dtype=np.float64)

    pos_cur = _get(frame, "log_pos_cur")
    pos_des = _get(frame, "log_pos_des")
    if pos_cur is None:
      pos_cur = _nan3()
    if pos_des is None:
      pos_des = _nan3()

    rpy_act0 = _get(frame, "top_rpy_act0")
    rpy_raw0 = _get(frame, "top_rpy_raw0")
    rpy_des0 = _get(frame, "top_rpy_des0")
    if rpy_act0 is None:
      rpy_act0 = _nan3()
    if rpy_raw0 is None:
      rpy_raw0 = _nan3()
    if rpy_des0 is None:
      rpy_des0 = _nan3()

    F0 = _get(frame, "log_thrust")
    if F0 is not None:
      F0 = np.asarray(F0, dtype=np.float64).ravel()
      if F0.size >= 4:
        F0 = F0[:4]
      else:
        F0 = None

    # Fallback to reconstructed F0 if log_thrust is missing/invalid.
    if F0 is None or (not np.isfinite(F0).any()):
      F0 = _get(frame, "top_F0")
      if F0 is not None:
        F0 = np.asarray(F0, dtype=np.float64).ravel()
        if F0.size >= 4:
          F0 = F0[:4]
        else:
          F0 = None

    if F0 is None:
      F0 = _nan4()


    cot0 = _get(frame, "top_cot0")
    if cot0 is None:
      cot0 = _nan2()

    tau_thrust0_xy = _get(frame, "top_tau_thrust0_xy")
    tau_cot0_xy = _get(frame, "top_tau_cot0_xy")
    tau_total0_xy = _get(frame, "top_tau_total0_xy")
    if tau_thrust0_xy is None:
      tau_thrust0_xy = _nan2()
    if tau_cot0_xy is None:
      tau_cot0_xy = _nan2()
    if tau_total0_xy is None:
      tau_total0_xy = _nan2()

    (
      tt, pc, pd, ra, rr, rd, FF, cc, tth, tco, tto
    ) = self._push_tr(
      pos_cur=pos_cur,
      pos_des=pos_des,
      rpy_act=rpy_act0,
      rpy_raw=rpy_raw0,
      rpy_des=rpy_des0,
      F0=F0,
      cot0=cot0,
      tau_thrust_xy=tau_thrust0_xy,
      tau_cot_xy=tau_cot0_xy,
      tau_total_xy=tau_total0_xy,
    )

    # pos plots (time axis)
    set_plot("posx", tt, [pc[:, 0], pd[:, 0]])
    set_plot("posy", tt, [pc[:, 1], pd[:, 1]])
    set_plot("posz", tt, [pc[:, 2], pd[:, 2]])

    # Roll/Pitch (time axis)
    set_plot("roll_tr", tt, [ra[:, 0], rr[:, 0], rd[:, 0]])
    set_plot("pitch_tr", tt, [ra[:, 1], rr[:, 1], rd[:, 1]])

    # F1234 (time axis)
    set_plot("F_tr", tt, [FF[:, 0], FF[:, 1], FF[:, 2], FF[:, 3]])

    # tau_x, tau_y (time axis)
    set_plot("tau_x_tr", tt, [tth[:, 0], tco[:, 0], tto[:, 0]])
    set_plot("tau_y_tr", tt, [tth[:, 1], tco[:, 1], tto[:, 1]])

    # cot_xy (time axis)
    set_plot("cotxy_tr", tt, [cc[:, 0], cc[:, 1]])

    # 3D trajectories (cur: blue solid, des: black dashed)
    self._update_3d_traj(pc, pd)

    # ----------------------------
    # Bottom plots (existing horizon data)
    # ----------------------------
    theta_plot = _get(frame, "theta_plot")
    theta_ref_plot = _get(frame, "theta_ref_plot")
    theta_d_plot = _get(frame, "theta_d_plot")
    if (
      theta_plot is not None and theta_ref_plot is not None and theta_d_plot is not None
      and theta_plot.shape[1] == 3 and theta_ref_plot.shape[1] == 3 and theta_d_plot.shape[1] == 3
    ):
      set_plot("roll",  t, [theta_plot[:, 0], theta_ref_plot[:, 0], theta_d_plot[:, 0]])
      set_plot("pitch", t, [theta_plot[:, 1], theta_ref_plot[:, 1], theta_d_plot[:, 1]])
      set_plot("yaw",   t, [theta_plot[:, 2], theta_ref_plot[:, 2], theta_d_plot[:, 2]])

    r_cot = _get(frame, "r_cot")
    r_cot_cmd = _get(frame, "r_cot_cmd")
    cotx_min_line = _get(frame, "cotx_min_line")
    cotx_max_line = _get(frame, "cotx_max_line")
    coty_min_line = _get(frame, "coty_min_line")
    coty_max_line = _get(frame, "coty_max_line")
    if (
      r_cot is not None and r_cot_cmd is not None and r_cot.shape[1] == 2 and r_cot_cmd.shape[1] == 2
      and cotx_min_line is not None and cotx_max_line is not None
      and coty_min_line is not None and coty_max_line is not None
    ):
      set_plot("cotx", t, [r_cot[:, 0], r_cot_cmd[:, 0], cotx_min_line, cotx_max_line])
      set_plot("coty", t, [r_cot[:, 1], r_cot_cmd[:, 1], coty_min_line, coty_max_line])

    omega_plot = _get(frame, "omega_plot")
    if omega_plot is not None and omega_plot.shape[1] == 3:
      set_plot("wx", t, [omega_plot[:, 0]])
      set_plot("wy", t, [omega_plot[:, 1]])
      set_plot("wz", t, [omega_plot[:, 2]])

    t_u = _get(frame, "t_u")
    r_cot_cmd_stage = _get(frame, "r_cot_cmd_stage")
    if t_u is not None and r_cot_cmd_stage is not None and r_cot_cmd_stage.shape[1] == 2:
      set_plot("cotux", t_u, [r_cot_cmd_stage[:, 0]])
      set_plot("cotuy", t_u, [r_cot_cmd_stage[:, 1]])

    F_stage = _get(frame, "F_stage")
    F_min_line = _get(frame, "F_min_line")
    F_max_line = _get(frame, "F_max_line")
    if t_u is not None and F_stage is not None and F_stage.shape[1] == 4 and F_min_line is not None and F_max_line is not None:
      set_plot("F", t_u, [F_stage[:, 0], F_stage[:, 1], F_stage[:, 2], F_stage[:, 3], F_min_line, F_max_line])

    J_omega = _get(frame, "J_omega")
    J_u_ch = _get(frame, "J_u_ch")
    J_total = _get(frame, "J_total")
    if t_u is not None and J_omega is not None and J_u_ch is not None and J_total is not None and J_u_ch.shape[1] == 5:
      set_plot("cost", t_u, [
        J_omega,
        J_u_ch[:, 0], J_u_ch[:, 1], J_u_ch[:, 2], J_u_ch[:, 3], J_u_ch[:, 4],
        J_total,
      ])

    solve_ms_scalar = _get(frame, "solve_ms_scalar")
    if solve_ms_scalar is not None and solve_ms_scalar.size >= 1:
      x_idx, ms_vals = self._push_solve_ms(float(solve_ms_scalar[0]))
      set_plot("solve", x_idx, [ms_vals])

    tau_d = _get(frame, "tau_d")
    if tau_d is not None and tau_d.shape[1] == 3:
      set_plot("tau", t, [tau_d[:, 0], tau_d[:, 1], tau_d[:, 2]])

    T_des = _get(frame, "T_des")
    if t_u is not None and T_des is not None:
      set_plot("T", t_u, [T_des])

# ----------------------------
# App main
# ----------------------------
def main() -> int:
  app = QApplication(sys.argv)

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
  QSplitter::handle { background: #ffffff; }
  QSplitter::handle:horizontal { width: 3px; }
  QSplitter::handle:vertical { height: 3px; }
  """)

  def _on_sigint(*_args) -> None:
    app.quit()

  signal.signal(signal.SIGINT, _on_sigint)

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
      print("[viewer] poll_mmap error:")
      traceback.print_exc()

  tm = QTimer()
  tm.timeout.connect(poll_mmap)
  tm.start(10)

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
