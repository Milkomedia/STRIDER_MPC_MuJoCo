from __future__ import annotations

import os
import sys
import signal
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
  QApplication,
  QLabel,
  QMainWindow,
  QSizePolicy,
  QVBoxLayout,
  QWidget,
  QGridLayout,
)

import pyqtgraph as pg


# ----------------------------
# Robust package-style import
# ----------------------------
THIS_DIR = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":
  PARENT_DIR = THIS_DIR.parent
  if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
  __package__ = THIS_DIR.name

from .mmap_manager import MMapReader, MMapPacket
from . import params as p
from .use_full import costs as c


ViewerFrame = Dict[str, np.ndarray]

FULL_NX = 14
FULL_NU = 11
FULL_NP = 29


def _gain3(x: Any) -> np.ndarray:
  a = np.asarray(x, dtype=np.float64).reshape(-1)
  if a.size == 1:
    return np.repeat(a.item(), 3)
  if a.size != 3:
    raise ValueError(f"Expected scalar or length-3 gain, got size={a.size}")
  return a


def _safe_clip(x: float, lo: float, hi: float) -> float:
  return float(np.clip(float(x), float(lo), float(hi)))


def _euler_zyx_to_R_np(theta: np.ndarray) -> np.ndarray:
  th = np.asarray(theta, dtype=np.float64).reshape(3)
  phi, pitch, psi = float(th[0]), float(th[1]), float(th[2])

  cphi, sphi = np.cos(phi), np.sin(phi)
  cth, sth = np.cos(pitch), np.sin(pitch)
  cpsi, spsi = np.cos(psi), np.sin(psi)

  Rz = np.array([
    [cpsi, -spsi, 0.0],
    [spsi,  cpsi, 0.0],
    [0.0,   0.0,  1.0],
  ], dtype=np.float64)

  Ry = np.array([
    [cth,  0.0, sth],
    [0.0,  1.0, 0.0],
    [-sth, 0.0, cth],
  ], dtype=np.float64)

  Rx = np.array([
    [1.0,  0.0,   0.0],
    [0.0,  cphi, -sphi],
    [0.0,  sphi,  cphi],
  ], dtype=np.float64)

  return Rz @ Ry @ Rx


def _R_to_euler_zyx_np(R: np.ndarray) -> np.ndarray:
  R = np.asarray(R, dtype=np.float64).reshape(3, 3)

  pitch = np.arcsin(_safe_clip(-R[2, 0], -1.0, 1.0))
  cth = np.cos(pitch)

  if abs(cth) < 1e-8:
    roll = np.arctan2(-R[1, 2], R[1, 1])
    yaw = 0.0
  else:
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

  return np.array([roll, pitch, yaw], dtype=np.float64)


def _hat_np(w: np.ndarray) -> np.ndarray:
  w = np.asarray(w, dtype=np.float64).reshape(3)
  return np.array([
    [0.0,   -w[2],  w[1]],
    [w[2],   0.0,  -w[0]],
    [-w[1],  w[0],  0.0],
  ], dtype=np.float64)


def _expm_hat_np(w: np.ndarray) -> np.ndarray:
  w = np.asarray(w, dtype=np.float64).reshape(3)
  th2 = float(np.dot(w, w))
  th = np.sqrt(th2 + 1e-12)

  A = np.sin(th) / th
  B = (1.0 - np.cos(th)) / (th2 + 1e-12)

  K = _hat_np(w)
  I = np.eye(3, dtype=np.float64)
  return I + A * K + B * (K @ K)


def _vee_np(S: np.ndarray) -> np.ndarray:
  S = np.asarray(S, dtype=np.float64).reshape(3, 3)
  return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def _polar_state_to_cartesian(r_pol: np.ndarray) -> np.ndarray:
  r_pol = np.asarray(r_pol, dtype=np.float64).reshape(4, 2)
  r_off_x = np.asarray(p.R_OFF_X, dtype=np.float64).reshape(4)
  r_off_y = np.asarray(p.R_OFF_Y, dtype=np.float64).reshape(4)

  r_xy = np.zeros((4, 2), dtype=np.float64)
  for a in range(4):
    rho = float(r_pol[a, 0])
    alpha = float(r_pol[a, 1])
    r_xy[a, 0] = r_off_x[a] + rho * np.cos(alpha)
    r_xy[a, 1] = r_off_y[a] + rho * np.sin(alpha)
  return r_xy


def _compute_pc_exact(r_pol: np.ndarray, load_angle: float) -> np.ndarray:
  """
  Reconstruct pc with the same formula used in use_full/model.py.
  No clipping is applied to D_a so that the behavior matches the model.
  """
  r_pol = np.asarray(r_pol, dtype=np.float64).reshape(4, 2)

  m_link = np.asarray(p.M_LINK, dtype=np.float64).reshape(5)
  m_link_sum = float(np.sum(m_link))
  m_tot = float(p.M_CENTER) + 4.0 * m_link_sum

  r_off_x = np.asarray(p.R_OFF_X, dtype=np.float64).reshape(4)
  r_off_y = np.asarray(p.R_OFF_Y, dtype=np.float64).reshape(4)

  a1 = float(p.A_LINK[0])
  a2 = float(p.A_LINK[1])
  a3 = float(p.A_LINK[2])
  d1 = float(p.D_LINK[0])
  d2 = float(p.D_LINK[1])
  d3 = float(p.D_LINK[2])
  rz = float(p.R_Z)

  center_body_com = np.array([
    float(p.MAX_COM_BIAS_OF_LOAD) * np.cos(float(load_angle)),
    0.0,
  ], dtype=np.float64)

  pc = np.zeros(2, dtype=np.float64)

  with np.errstate(invalid="ignore"):
    for a in range(4):
      rho = float(r_pol[a, 0])
      alpha = float(r_pol[a, 1])

      x_a = rho - a1
      D_a = (x_a * x_a + rz * rz - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)

      q3 = np.arctan2(np.sqrt(1.0 - D_a * D_a), D_a)
      q2 = np.arctan2(rz, x_a) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))

      rho_c_a = (
        m_link[0] * (a1 + d1)
        + m_link[1] * (a1 + (a2 + d2) * np.cos(q2))
        + m_link[2] * (a1 + a2 * np.cos(q2) + (a3 + d3) * np.cos(q2 + q3))
        + (m_link[3] + m_link[4]) * rho
      )

      pc += np.array([
        m_link_sum * (r_off_x[a] + rho_c_a * np.cos(alpha)),
        m_link_sum * (r_off_y[a] + rho_c_a * np.sin(alpha)),
      ], dtype=np.float64)

  pc = (pc + center_body_com) / m_tot
  return pc


def _solve_F_exact(A: np.ndarray, w_d: np.ndarray) -> np.ndarray:
  """
  Reconstruct F_expr as in the model.
  If A is singular or invalid, return NaNs instead of a least-squares fallback,
  because the model itself does not use a fallback solve.
  """
  A = np.asarray(A, dtype=np.float64).reshape(4, 4)
  w_d = np.asarray(w_d, dtype=np.float64).reshape(4)

  if not np.all(np.isfinite(A)) or not np.all(np.isfinite(w_d)):
    return np.full(4, np.nan, dtype=np.float64)

  try:
    return np.linalg.solve(A, w_d)
  except np.linalg.LinAlgError:
    return np.full(4, np.nan, dtype=np.float64)


def _decompose_tau_xy(r_xy: np.ndarray, pc_xy: np.ndarray, F_expr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Decompose tau_xy into:
    tau_tot   = exact xy torque about pc from F_expr
    tau_thrust = xy torque about r_mean
    tau_off    = offset term from r_mean -> pc
  """
  if not np.all(np.isfinite(r_xy)) or not np.all(np.isfinite(pc_xy)) or not np.all(np.isfinite(F_expr)):
    nan2 = np.full(2, np.nan, dtype=np.float64)
    return nan2, nan2, nan2, np.full(2, np.nan, dtype=np.float64)

  r_mean_xy = np.mean(r_xy, axis=0)

  A_pc_xy = np.array([
    [-(r_xy[0, 1] - pc_xy[1]), -(r_xy[1, 1] - pc_xy[1]), -(r_xy[2, 1] - pc_xy[1]), -(r_xy[3, 1] - pc_xy[1])],
    [ (r_xy[0, 0] - pc_xy[0]),  (r_xy[1, 0] - pc_xy[0]),  (r_xy[2, 0] - pc_xy[0]),  (r_xy[3, 0] - pc_xy[0])],
  ], dtype=np.float64)

  A_mean_xy = np.array([
    [-(r_xy[0, 1] - r_mean_xy[1]), -(r_xy[1, 1] - r_mean_xy[1]), -(r_xy[2, 1] - r_mean_xy[1]), -(r_xy[3, 1] - r_mean_xy[1])],
    [ (r_xy[0, 0] - r_mean_xy[0]),  (r_xy[1, 0] - r_mean_xy[0]),  (r_xy[2, 0] - r_mean_xy[0]),  (r_xy[3, 0] - r_mean_xy[0])],
  ], dtype=np.float64)

  tau_tot_xy = A_pc_xy @ F_expr
  tau_thrust_xy = A_mean_xy @ F_expr

  F_sum = float(np.sum(F_expr))
  dr = r_mean_xy - pc_xy
  tau_off_xy = np.array([
    -dr[1] * F_sum,
     dr[0] * F_sum,
  ], dtype=np.float64)

  return tau_thrust_xy, tau_off_xy, tau_tot_xy, r_mean_xy


def _solve_viewer_frame(pkt: MMapPacket) -> ViewerFrame:
  """
  Parse mmap packet strictly as the upcasted use_full layout:

    x = [
      theta(0:3), omega(3:6),
      r1(6:8), r2(8:10), r3(10:12), r4(12:14)
    ]

    u = [
      delta_theta_cmd(0:3),
      r1_cmd(3:5), r2_cmd(5:7), r3_cmd(7:9), r4_cmd(9:11)
    ]

    p = [
      R_raw(0:9), W_raw(9:12), Wdot_raw(12:15),
      R_0(15:24), f_0(24), load_angle(25)
    ]
  """
  N = int(pkt.N)
  nx = int(pkt.nx)
  nu = int(pkt.nu)
  np_ = int(pkt.np)

  if nx < FULL_NX or nu < FULL_NU or np_ < FULL_NP:
    raise ValueError(
      f"Packet dims are smaller than full-layout expectation: "
      f"got (nx, nu, np)=({nx}, {nu}, {np_}), "
      f"expected at least ({FULL_NX}, {FULL_NU}, {FULL_NP})"
    )

  x_all = np.asarray(pkt.x_all, dtype=np.float64).reshape(N + 1, nx)[:, :FULL_NX]
  p_all = np.asarray(pkt.p_all, dtype=np.float64).reshape(N + 1, np_)[:, :FULL_NP]
  u_all = np.asarray(pkt.u_all, dtype=np.float64).reshape(N, nu)[:, :FULL_NU] if N > 0 else np.zeros((0, FULL_NU), dtype=np.float64)

  KR = _gain3(p.KR)
  KW = _gain3(p.KW)
  zeta = float(p.ZETA)

  steps = np.arange(N, dtype=np.float64)

  F_stage = np.full((N, 4), np.nan, dtype=np.float64)

  tau_d_stage = np.full((N, 3), np.nan, dtype=np.float64)
  tau_thrust_stage = np.full((N, 2), np.nan, dtype=np.float64)
  tau_off_stage = np.full((N, 2), np.nan, dtype=np.float64)
  tau_tot_stage = np.full((N, 2), np.nan, dtype=np.float64)

  bFz_pos_stage = np.full(N, np.nan, dtype=np.float64)

  roll_raw_stage = np.full(N, np.nan, dtype=np.float64)
  roll_des_stage = np.full(N, np.nan, dtype=np.float64)
  roll_cur_stage = np.full(N, np.nan, dtype=np.float64)

  pitch_raw_stage = np.full(N, np.nan, dtype=np.float64)
  pitch_des_stage = np.full(N, np.nan, dtype=np.float64)
  pitch_cur_stage = np.full(N, np.nan, dtype=np.float64)

  r_state_stage = np.full((N, 4, 2), np.nan, dtype=np.float64)
  r_cmd_stage = np.full((N, 4, 2), np.nan, dtype=np.float64)
  r_mean_stage = np.full((N, 2), np.nan, dtype=np.float64)
  r_cmd_mean_stage = np.full((N, 2), np.nan, dtype=np.float64)

  pc_stage = np.full((N, 2), np.nan, dtype=np.float64)

  delta_theta_cmd_stage = np.full((N, 3), np.nan, dtype=np.float64)
  u_cmd_rotor_stage = np.full((N, 8), np.nan, dtype=np.float64)

  for k in range(N):
    xk = x_all[k, :]
    pk = p_all[k, :]
    uk = u_all[k, :] if k < u_all.shape[0] else np.zeros((FULL_NU,), dtype=np.float64)

    theta = xk[0:3]
    omega = xk[3:6]
    r_pol = xk[6:14].reshape(4, 2)
    r_pol_cmd = uk[3:11].reshape(4, 2)
    delta_theta_cmd = uk[0:3]

    R_raw = pk[0:9].reshape(3, 3, order="F")
    W_raw = pk[9:12]
    Wdot_raw = pk[12:15]  # kept for completeness
    R_0 = pk[15:24].reshape(3, 3, order="F")
    f_0 = float(pk[24])
    load_angle = float(pk[25])

    _ = Wdot_raw  # keep exact p-unpack symmetry with model

    R = _euler_zyx_to_R_np(theta)
    Rd = R_raw @ _expm_hat_np(delta_theta_cmd)
    Wd = _expm_hat_np(-delta_theta_cmd) @ W_raw

    raw_euler = _R_to_euler_zyx_np(R_raw)
    des_euler = _R_to_euler_zyx_np(Rd)

    RtRd = R.T @ Rd
    e_R = 0.5 * _vee_np(RtRd.T - RtRd)
    e_w = omega - RtRd @ Wd
    tau_d = -(KR * e_R) - (KW * e_w)

    r_xy = _polar_state_to_cartesian(r_pol)
    r_cmd_xy = _polar_state_to_cartesian(r_pol_cmd)
    r_cmd_mean_xy = np.mean(r_cmd_xy, axis=0)

    pc_xy = _compute_pc_exact(r_pol, load_angle)

    b_F0 = np.array([0.0, 0.0, -f_0], dtype=np.float64)
    g_F0 = R_0 @ b_F0
    b_F = R.T @ g_F0

    A = np.array([
      [-(r_xy[0, 1] - pc_xy[1]), -(r_xy[1, 1] - pc_xy[1]), -(r_xy[2, 1] - pc_xy[1]), -(r_xy[3, 1] - pc_xy[1])],
      [ (r_xy[0, 0] - pc_xy[0]),  (r_xy[1, 0] - pc_xy[0]),  (r_xy[2, 0] - pc_xy[0]),  (r_xy[3, 0] - pc_xy[0])],
      [-zeta, zeta, -zeta, zeta],
      [-1.0, -1.0, -1.0, -1.0],
    ], dtype=np.float64)

    w_d = np.array([tau_d[0], tau_d[1], tau_d[2], b_F[2]], dtype=np.float64)
    F_expr = _solve_F_exact(A, w_d)

    tau_thrust_xy, tau_off_xy, tau_tot_xy, r_mean_xy = _decompose_tau_xy(r_xy, pc_xy, F_expr)

    F_stage[k, :] = F_expr

    tau_d_stage[k, :] = tau_d
    tau_thrust_stage[k, :] = tau_thrust_xy
    tau_off_stage[k, :] = tau_off_xy
    tau_tot_stage[k, :] = tau_tot_xy

    bFz_pos_stage[k] = -b_F[2]

    roll_raw_stage[k] = raw_euler[0]
    roll_des_stage[k] = des_euler[0]
    roll_cur_stage[k] = theta[0]

    pitch_raw_stage[k] = raw_euler[1]
    pitch_des_stage[k] = des_euler[1]
    pitch_cur_stage[k] = theta[1]

    r_state_stage[k, :, :] = r_xy
    r_cmd_stage[k, :, :] = r_cmd_xy
    r_mean_stage[k, :] = r_mean_xy
    r_cmd_mean_stage[k, :] = r_cmd_mean_xy
    pc_stage[k, :] = pc_xy

    delta_theta_cmd_stage[k, :] = delta_theta_cmd
    u_cmd_rotor_stage[k, :] = uk[3:11]

  F_min = np.asarray(getattr(c, "F_MIN", np.full(4, np.nan)), dtype=np.float64).reshape(-1)
  F_max = np.asarray(getattr(c, "F_MAX", np.full(4, np.nan)), dtype=np.float64).reshape(-1)

  if F_min.size == 1:
    F_min = np.repeat(F_min.item(), 4)
  if F_max.size == 1:
    F_max = np.repeat(F_max.item(), 4)

  if F_min.size != 4:
    F_min = np.full(4, np.nan, dtype=np.float64)
  if F_max.size != 4:
    F_max = np.full(4, np.nan, dtype=np.float64)

  return {
    "steps": steps,
    "F_stage": F_stage,
    "F_min": F_min,
    "F_max": F_max,
    "tau_d": tau_d_stage,
    "tau_thrust": tau_thrust_stage,
    "tau_off": tau_off_stage,
    "tau_tot": tau_tot_stage,
    "bFz_pos": bFz_pos_stage,
    "roll_raw": roll_raw_stage,
    "roll_des": roll_des_stage,
    "roll_cur": roll_cur_stage,
    "pitch_raw": pitch_raw_stage,
    "pitch_des": pitch_des_stage,
    "pitch_cur": pitch_cur_stage,
    "r_state": r_state_stage,
    "r_cmd": r_cmd_stage,
    "r_mean": r_mean_stage,
    "r_cmd_mean": r_cmd_mean_stage,
    "pc": pc_stage,
    "delta_theta_cmd": delta_theta_cmd_stage,
    "u_cmd_rotor": u_cmd_rotor_stage,
  }


class DualAxisPlot:
  def __init__(
    self,
    title: str,
    left_label: str,
    right_label: str,
    left_specs: List[Tuple[str, Any]],
    right_specs: List[Tuple[str, Any]],
  ) -> None:
    self.widget = pg.PlotWidget()
    self.widget.setBackground("w")
    self.widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    self.plot_item = self.widget.getPlotItem()
    self.plot_item.setTitle(title)
    self.plot_item.showGrid(x=True, y=True, alpha=0.25)
    self.plot_item.getAxis("left").setPen(pg.mkPen("k"))
    self.plot_item.getAxis("bottom").setPen(pg.mkPen("k"))
    self.plot_item.getAxis("left").setTextPen(pg.mkPen("k"))
    self.plot_item.getAxis("bottom").setTextPen(pg.mkPen("k"))
    self.plot_item.setLabel("bottom", "step k")
    self.plot_item.setLabel("left", left_label)
    self.plot_item.showAxis("right")
    self.plot_item.getAxis("right").setPen(pg.mkPen("k"))
    self.plot_item.getAxis("right").setTextPen(pg.mkPen("k"))
    self.plot_item.setLabel("right", right_label)

    self.legend = self.plot_item.addLegend()
    self.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

    self.right_view = pg.ViewBox()
    self.plot_item.scene().addItem(self.right_view)
    self.plot_item.getAxis("right").linkToView(self.right_view)
    self.right_view.setXLink(self.plot_item.vb)

    self.left_curves: List[Any] = []
    self.right_curves: List[Any] = []

    for name, pen in left_specs:
      curve = self.plot_item.plot(name=name, pen=pen)
      self.left_curves.append(curve)

    for name, pen in right_specs:
      curve = pg.PlotDataItem(name=name, pen=pen)
      self.right_view.addItem(curve)
      self.legend.addItem(curve, name)
      self.right_curves.append(curve)

    self.plot_item.vb.sigResized.connect(self._update_views)
    self._update_views()

  def _update_views(self) -> None:
    self.right_view.setGeometry(self.plot_item.vb.sceneBoundingRect())
    self.right_view.linkedViewChanged(self.plot_item.vb, self.right_view.XAxis)

  def set_left_data(self, x: np.ndarray, ys: List[np.ndarray]) -> None:
    for curve, y in zip(self.left_curves, ys):
      curve.setData(x, y)

  def set_right_data(self, x: np.ndarray, ys: List[np.ndarray]) -> None:
    for curve, y in zip(self.right_curves, ys):
      curve.setData(x, y)


class ViewerMainWindow(QMainWindow):
  def __init__(self) -> None:
    super().__init__()
    self.setWindowTitle("MPC Horizon Viewer (full-layout reconstruction)")

    lw = 2.0
    dash = Qt.DashLine

    pen_r = pg.mkPen(color=(220, 50, 50), width=lw)
    pen_g = pg.mkPen(color=(50, 160, 70), width=lw)
    pen_b = pg.mkPen(color=(50, 90, 220), width=lw)
    pen_p = pg.mkPen(color=(120, 60, 200), width=lw)
    pen_k = pg.mkPen(color=(20, 20, 20), width=lw)

    pen_r_d = pg.mkPen(color=(220, 50, 50), width=lw, style=dash)
    pen_g_d = pg.mkPen(color=(50, 160, 70), width=lw, style=dash)
    pen_b_d = pg.mkPen(color=(50, 90, 220), width=lw, style=dash)
    pen_p_d = pg.mkPen(color=(120, 60, 200), width=lw, style=dash)

    pen_gray = pg.mkPen(color=(100, 100, 100), width=1.5, style=dash)
    pen_black = pg.mkPen(color=(0, 0, 0), width=1.5, style=dash)

    self._plots: Dict[str, Tuple[pg.PlotWidget, List[Any]]] = {}
    self._dual_plots: Dict[str, DualAxisPlot] = {}

    def make_plot(
      title: str,
      y_label: str,
      curve_specs: List[Tuple[str, Any]],
      fixed_y_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[pg.PlotWidget, List[Any]]:
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
      pi.setLabel("bottom", "step k")
      pi.setLabel("left", y_label)

      if fixed_y_range is not None:
        pi.enableAutoRange(axis="y", enable=False)
        pi.setYRange(float(fixed_y_range[0]), float(fixed_y_range[1]), padding=0.0)

      legend = pi.addLegend()
      legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

      curves: List[Any] = []
      for name, pen in curve_specs:
        curves.append(w.plot(name=name, pen=pen))
      return w, curves

    central = QWidget()
    self.setCentralWidget(central)

    root = QVBoxLayout(central)
    root.setContentsMargins(8, 8, 8, 8)
    root.setSpacing(8)

    self.info_label = QLabel("Waiting for mmap packet...")
    self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    root.addWidget(self.info_label)

    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(6)
    root.addLayout(grid)

    fmin = np.asarray(getattr(c, "F_MIN", np.zeros(4)), dtype=np.float64).reshape(-1)
    fmax = np.asarray(getattr(c, "F_MAX", np.zeros(4)), dtype=np.float64).reshape(-1)

    if fmin.size == 0 or not np.any(np.isfinite(fmin)):
      y_lo = -1.0
    else:
      y_lo = float(np.nanmin(fmin)) - 2.0

    if fmax.size == 0 or not np.any(np.isfinite(fmax)):
      y_hi = 20.0
    else:
      y_hi = float(np.nanmax(fmax)) + 2.0

    # Row 1
    self._plots["F"] = make_plot(
      "F1234 (reconstructed from full-layout x,u,p)",
      "[N]",
      [("F1", pen_r), ("F2", pen_g), ("F3", pen_b), ("F4", pen_p), ("Fmin", pen_gray), ("Fmax", pen_black)],
      fixed_y_range=(y_lo, y_hi),
    )
    self._plots["tau_x"] = make_plot(
      "tau_x",
      "[N·m]",
      [("tau_d_x", pen_k), ("tau_thrust_x", pen_b), ("tau_off_x", pen_r), ("tau_tot_x", pen_g)],
    )
    self._plots["tau_y"] = make_plot(
      "tau_y",
      "[N·m]",
      [("tau_d_y", pen_k), ("tau_thrust_y", pen_b), ("tau_off_y", pen_r), ("tau_tot_y", pen_g)],
    )
    self._plots["bFz"] = make_plot(
      "(-b_F[2])",
      "[N]",
      [("-b_F[2]", pen_k)],
    )

    # Row 2
    self._plots["roll"] = make_plot(
      "roll raw / des / cur",
      "[deg]",
      [("roll_raw", pen_k), ("roll_des", pen_b), ("roll_cur", pen_r)],
    )
    self._plots["pitch"] = make_plot(
      "pitch raw / des / cur",
      "[deg]",
      [("pitch_raw", pen_k), ("pitch_des", pen_b), ("pitch_cur", pen_r)],
    )
    self._dual_plots["u_rotor"] = DualAxisPlot(
      title="u_cmd(r1234 rho alpha)",
      left_label="alpha [deg]",
      right_label="rho [mm]",
      left_specs=[
        ("r1_alpha_cmd", pen_r_d),
        ("r2_alpha_cmd", pen_g_d),
        ("r3_alpha_cmd", pen_b_d),
        ("r4_alpha_cmd", pen_p_d),
      ],
      right_specs=[
        ("r1_rho_cmd", pen_r),
        ("r2_rho_cmd", pen_g),
        ("r3_rho_cmd", pen_b),
        ("r4_rho_cmd", pen_p),
      ],
    )

    # Row 3
    self._plots["r_x"] = make_plot(
      "r1,2,3,4 (cmd, cur) x",
      "[mm]",
      [
        ("r1_x", pen_r), ("r1_cmd_x", pen_r_d),
        ("r2_x", pen_g), ("r2_cmd_x", pen_g_d),
        ("r3_x", pen_b), ("r3_cmd_x", pen_b_d),
        ("r4_x", pen_p), ("r4_cmd_x", pen_p_d),
      ],
    )
    self._plots["r_y"] = make_plot(
      "r1,2,3,4 (cmd, cur) y",
      "[mm]",
      [
        ("r1_y", pen_r), ("r1_cmd_y", pen_r_d),
        ("r2_y", pen_g), ("r2_cmd_y", pen_g_d),
        ("r3_y", pen_b), ("r3_cmd_y", pen_b_d),
        ("r4_y", pen_p), ("r4_cmd_y", pen_p_d),
      ],
    )
    self._plots["r_mean"] = make_plot(
      "r1234 mean",
      "[mm]",
      [("mean_x", pen_r), ("mean_y", pen_b), ("cmd_mean_x", pen_r_d), ("cmd_mean_y", pen_b_d)],
    )
    self._plots["pc"] = make_plot(
      "pc xy",
      "[mm]",
      [("pc_x", pen_r), ("pc_y", pen_b)],
    )

    grid.addWidget(self._plots["F"][0], 0, 0)
    grid.addWidget(self._plots["tau_x"][0], 0, 1)
    grid.addWidget(self._plots["tau_y"][0], 0, 2)
    grid.addWidget(self._plots["bFz"][0], 0, 3)

    grid.addWidget(self._plots["roll"][0], 1, 0)
    grid.addWidget(self._plots["pitch"][0], 1, 1)
    grid.addWidget(self._dual_plots["u_rotor"].widget, 1, 2, 1, 2)

    grid.addWidget(self._plots["r_x"][0], 2, 0)
    grid.addWidget(self._plots["r_y"][0], 2, 1)
    grid.addWidget(self._plots["r_mean"][0], 2, 2)
    grid.addWidget(self._plots["pc"][0], 2, 3)

    base_plot = self._plots["F"][0]
    for key, (plot_widget, _) in self._plots.items():
      if key != "F":
        plot_widget.setXLink(base_plot)
    for dual in self._dual_plots.values():
      dual.widget.setXLink(base_plot)

    screen = QApplication.primaryScreen()
    if screen is not None:
      avail = screen.availableGeometry()
      w0 = int(avail.width() * 0.95)
      h0 = int(avail.height() * 0.92)
      self.resize(w0, h0)
    else:
      self.resize(1850, 1200)

  def _set_plot(self, key: str, x: np.ndarray, ys: List[np.ndarray]) -> None:
    _, curves = self._plots[key]
    for curve, y in zip(curves, ys):
      curve.setData(x, y)

  def update_from_packet(self, pkt: MMapPacket) -> None:
    frame = _solve_viewer_frame(pkt)

    steps = frame["steps"]
    if steps.size == 0:
      return

    F_stage = frame["F_stage"]
    F_min = frame["F_min"]
    F_max = frame["F_max"]

    tau_d = frame["tau_d"]
    tau_thrust = frame["tau_thrust"]
    tau_off = frame["tau_off"]
    tau_tot = frame["tau_tot"]
    bFz_pos = frame["bFz_pos"]

    roll_raw_deg = np.rad2deg(frame["roll_raw"])
    roll_des_deg = np.rad2deg(frame["roll_des"])
    roll_cur_deg = np.rad2deg(frame["roll_cur"])

    pitch_raw_deg = np.rad2deg(frame["pitch_raw"])
    pitch_des_deg = np.rad2deg(frame["pitch_des"])
    pitch_cur_deg = np.rad2deg(frame["pitch_cur"])

    r_state_mm = 1000.0 * frame["r_state"]
    r_cmd_mm = 1000.0 * frame["r_cmd"]
    r_mean_mm = 1000.0 * frame["r_mean"]
    r_cmd_mean_mm = 1000.0 * frame["r_cmd_mean"]
    pc_mm = 1000.0 * frame["pc"]

    delta_theta_cmd_deg = np.rad2deg(frame["delta_theta_cmd"])

    u_cmd_rotor = np.asarray(frame["u_cmd_rotor"], dtype=np.float64)
    rho_mm_s = 1000.0 * u_cmd_rotor[:, 0::2]
    alpha_deg_s = np.rad2deg(u_cmd_rotor[:, 1::2])

    if np.any(np.isfinite(F_min)):
      fmin_line = np.full_like(steps, float(np.nanmin(F_min)), dtype=np.float64)
    else:
      fmin_line = np.full_like(steps, np.nan, dtype=np.float64)

    if np.any(np.isfinite(F_max)):
      fmax_line = np.full_like(steps, float(np.nanmax(F_max)), dtype=np.float64)
    else:
      fmax_line = np.full_like(steps, np.nan, dtype=np.float64)

    self._set_plot(
      "F",
      steps,
      [F_stage[:, 0], F_stage[:, 1], F_stage[:, 2], F_stage[:, 3], fmin_line, fmax_line],
    )

    self._set_plot(
      "tau_x",
      steps,
      [tau_d[:, 0], tau_thrust[:, 0], tau_off[:, 0], tau_tot[:, 0]],
    )
    self._set_plot(
      "tau_y",
      steps,
      [tau_d[:, 1], tau_thrust[:, 1], tau_off[:, 1], tau_tot[:, 1]],
    )
    self._set_plot("bFz", steps, [bFz_pos])

    self._set_plot("roll", steps, [roll_raw_deg, roll_des_deg, roll_cur_deg])
    self._set_plot("pitch", steps, [pitch_raw_deg, pitch_des_deg, pitch_cur_deg])

    self._dual_plots["u_rotor"].set_left_data(
      steps,
      [alpha_deg_s[:, 0], alpha_deg_s[:, 1], alpha_deg_s[:, 2], alpha_deg_s[:, 3]],
    )
    self._dual_plots["u_rotor"].set_right_data(
      steps,
      [rho_mm_s[:, 0], rho_mm_s[:, 1], rho_mm_s[:, 2], rho_mm_s[:, 3]],
    )

    self._set_plot(
      "r_x",
      steps,
      [
        r_state_mm[:, 0, 0], r_cmd_mm[:, 0, 0],
        r_state_mm[:, 1, 0], r_cmd_mm[:, 1, 0],
        r_state_mm[:, 2, 0], r_cmd_mm[:, 2, 0],
        r_state_mm[:, 3, 0], r_cmd_mm[:, 3, 0],
      ],
    )
    self._set_plot(
      "r_y",
      steps,
      [
        r_state_mm[:, 0, 1], r_cmd_mm[:, 0, 1],
        r_state_mm[:, 1, 1], r_cmd_mm[:, 1, 1],
        r_state_mm[:, 2, 1], r_cmd_mm[:, 2, 1],
        r_state_mm[:, 3, 1], r_cmd_mm[:, 3, 1],
      ],
    )
    self._set_plot(
      "r_mean",
      steps,
      [r_mean_mm[:, 0], r_mean_mm[:, 1], r_cmd_mean_mm[:, 0], r_cmd_mean_mm[:, 1]],
    )
    self._set_plot("pc", steps, [pc_mm[:, 0], pc_mm[:, 1]])

    f_now = F_stage[0, :]
    tau_d_now = tau_d[0, :]
    tau_thrust_now = tau_thrust[0, :]
    tau_off_now = tau_off[0, :]
    tau_tot_now = tau_tot[0, :]
    pc_now = pc_mm[0, :]
    bFz_now = float(bFz_pos[0])
    dtheta0_deg = delta_theta_cmd_deg[0, :]

    self.info_label.setText(
      f"seq={pkt.seq}   status={pkt.status}   solve_ms={pkt.solve_ms:.3f}   "
      f"F0=[{f_now[0]:.3f}, {f_now[1]:.3f}, {f_now[2]:.3f}, {f_now[3]:.3f}]   "
      f"tau_d_xy0=[{tau_d_now[0]:.3f}, {tau_d_now[1]:.3f}]   "
      f"tau_thrust_xy0=[{tau_thrust_now[0]:.3f}, {tau_thrust_now[1]:.3f}]   "
      f"tau_off_xy0=[{tau_off_now[0]:.3f}, {tau_off_now[1]:.3f}]   "
      f"tau_tot_xy0=[{tau_tot_now[0]:.3f}, {tau_tot_now[1]:.3f}]   "
      f"(-bF[2])0={bFz_now:.3f}   "
      f"delta_theta_cmd0[deg]=[{dtheta0_deg[0]:.2f}, {dtheta0_deg[1]:.2f}, {dtheta0_deg[2]:.2f}]   "
      f"pc[mm]=[{pc_now[0]:.2f}, {pc_now[1]:.2f}]"
    )


def main() -> int:
  pg.setConfigOptions(antialias=True)

  app = QApplication(sys.argv)
  app.setStyleSheet("""
  QWidget { background: #ffffff; color: #111111; }
  QMainWindow { background: #ffffff; }
  QLabel {
    font-family: monospace;
    font-size: 12px;
    padding: 2px 4px 2px 4px;
  }
  """)

  win = ViewerMainWindow()
  win.show()

  path = os.environ.get("MRG_MMAP", "/tmp/MRG_debug.mmap")
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

      win.update_from_packet(pkt)

    except Exception:
      traceback.print_exc()

  timer = QTimer()
  timer.timeout.connect(poll_mmap)
  timer.start(100)

  def _sigint_handler(*_args: Any) -> None:
    app.quit()

  signal.signal(signal.SIGINT, _sigint_handler)

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