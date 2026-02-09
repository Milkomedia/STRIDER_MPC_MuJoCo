# mpc_viewer.py
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
  QGroupBox,
  QMainWindow,
  QSizePolicy,
  QVBoxLayout,
  QWidget,
  QGridLayout,
  QSplitter,
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
from . import params as p  # Always use yes_cot params

DebugFrame = Dict[str, Optional[np.ndarray]]


def _get(frame: DebugFrame, key: str) -> Optional[np.ndarray]:
  v = frame.get(key, None)
  if v is None:
    return None
  a = np.asarray(v)
  return a


# ----------------------------
# SO(3) math utils (NumPy)
# ----------------------------
def euler_zyx_to_R_np(theta: np.ndarray) -> np.ndarray:
  """Euler ZYX (roll-pitch-yaw) to rotation matrix.
  theta = [phi, th, psi] in rad.
  Returns R (3,3) mapping body->global (same convention as model.py).
  """
  th = np.asarray(theta, dtype=np.float64).ravel()
  if th.size != 3:
    return np.eye(3, dtype=np.float64)

  phi, t, psi = float(th[0]), float(th[1]), float(th[2])

  cphi, sphi = np.cos(phi), np.sin(phi)
  ct, st = np.cos(t), np.sin(t)
  cpsi, spsi = np.cos(psi), np.sin(psi)

  Rx = np.array([
    [1.0, 0.0, 0.0],
    [0.0, cphi, -sphi],
    [0.0, sphi, cphi],
  ], dtype=np.float64)

  Ry = np.array([
    [ct, 0.0, st],
    [0.0, 1.0, 0.0],
    [-st, 0.0, ct],
  ], dtype=np.float64)

  Rz = np.array([
    [cpsi, -spsi, 0.0],
    [spsi, cpsi, 0.0],
    [0.0, 0.0, 1.0],
  ], dtype=np.float64)

  return Rz @ Ry @ Rx

def R_to_euler_zyx_np(R: np.ndarray) -> np.ndarray:
  """Rotation matrix to Euler ZYX (roll-pitch-yaw), rad."""
  R = np.asarray(R, dtype=np.float64).reshape(3, 3)

  # Robust-ish ZYX extraction (consistent with existing viewer).
  s = -float(R[2, 0])
  s = float(np.clip(s, -1.0, 1.0))
  th = np.arcsin(s)

  if abs(np.cos(th)) < 1e-8:
    # Gimbal lock
    phi = 0.0
    psi = np.arctan2(-R[0, 1], R[1, 1])
  else:
    phi = np.arctan2(R[2, 1], R[2, 2])
    psi = np.arctan2(R[1, 0], R[0, 0])

  return np.array([phi, th, psi], dtype=np.float64)

def hat_np(w: np.ndarray) -> np.ndarray:
  w = np.asarray(w, dtype=np.float64).ravel()
  if w.size != 3:
    return np.zeros((3, 3), dtype=np.float64)
  wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
  return np.array([
    [0.0, -wz, wy],
    [wz, 0.0, -wx],
    [-wy, wx, 0.0],
  ], dtype=np.float64)

def expm_hat_np(w: np.ndarray) -> np.ndarray:
  """Exponential map exp(hat(w)) for rotation vector w (rad)."""
  w = np.asarray(w, dtype=np.float64).ravel()
  if w.size != 3:
    return np.eye(3, dtype=np.float64)

  th2 = float(np.dot(w, w))
  th = float(np.sqrt(th2))

  K = hat_np(w)
  I = np.eye(3, dtype=np.float64)

  if th < 1e-10:
    # First-order approximation is enough.
    return I + K

  A = np.sin(th) / th
  B = (1.0 - np.cos(th)) / (th2 + 1e-12)
  return I + A * K + B * (K @ K)

def vee_np(S: np.ndarray) -> np.ndarray:
  """vee map for a skew-symmetric matrix (3x3)."""
  S = np.asarray(S, dtype=np.float64).reshape(3, 3)
  return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)

def gain3_np(x) -> np.ndarray:
  a = np.asarray(x, dtype=np.float64).reshape(-1)
  if a.size == 1:
    a = np.repeat(a.item(), 3)
  elif a.size != 3:
    raise ValueError(f"gain must be scalar or length-3, got size={a.size}, value={x}")
  return a

# ----------------------------
# Frame computation (one MPC horizon)
# ----------------------------
def compute_frame_from_pkt(pkt: MMapPacket) -> DebugFrame:
  N = int(pkt.N)
  nx = int(pkt.nx)
  nu = int(pkt.nu)
  np_ = int(pkt.np)

  T = N + 1
  dt = float(getattr(p, "DT", 0.015))

  x_all = np.asarray(pkt.x_all, dtype=np.float64).reshape(T, nx)
  u_all = np.asarray(pkt.u_all, dtype=np.float64).reshape(N, nu) if N > 0 else np.zeros((0, nu), dtype=np.float64)
  p_all = np.asarray(pkt.p_all, dtype=np.float64).reshape(T, np_)

  # Current convention (augmented):
  # x = [theta(3), omega(3), r_cot(2), delta_theta_cmd(3), r_cot_cmd(2)] => nx=13
  # u = u_rate = [delta_theta_cmd_rate(3), r_cot_cmd_rate(2)] => nu=5
  use_aug = (nx >= 13)

  theta = x_all[:, 0:3] if nx >= 3 else np.zeros((T, 3), dtype=np.float64)
  omega = x_all[:, 3:6] if nx >= 6 else np.zeros((T, 3), dtype=np.float64)
  r_cot = x_all[:, 6:8] if nx >= 8 else np.zeros((T, 2), dtype=np.float64)

  if use_aug:
    delta_theta_cmd = x_all[:, 8:11]
    r_cot_cmd = x_all[:, 11:13]
  else:
    # Legacy fallback: treat u_all as command and replicate last for terminal.
    u_cmd_T = np.zeros((T, 5), dtype=np.float64)
    if (N > 0) and (u_all.shape[0] == N) and (u_all.shape[1] >= 5):
      u_cmd_T[:N, :] = u_all[:, 0:5]
      u_cmd_T[N, :] = u_cmd_T[N - 1, :]
    delta_theta_cmd = u_cmd_T[:, 0:3]
    r_cot_cmd = u_cmd_T[:, 3:5]

  # u_rate (stage axis)
  u_rate_stage = np.zeros((N, 5), dtype=np.float64)
  if use_aug and (N > 0) and (u_all.shape[0] == N) and (u_all.shape[1] >= 5):
    u_rate_stage[:, :] = u_all[:, 0:5]

  # p = [vec(R_raw)(9), omega_raw(3), l(1), T_des(1)] => np=14
  R_raw_all = np.zeros((T, 3, 3), dtype=np.float64)
  omega_raw_all = np.zeros((T, 3), dtype=np.float64)
  l_all = np.zeros((T,), dtype=np.float64)
  T_des_all = np.zeros((T,), dtype=np.float64)

  for k in range(T):
    pv = p_all[k, :].ravel()
    if pv.size < 14:
      continue
    # NOTE: CasADi reshape is column-major.
    R_raw_all[k, :, :] = pv[0:9].reshape(3, 3, order="F")
    omega_raw_all[k, :] = pv[9:12].astype(np.float64, copy=False)
    l_all[k] = float(pv[12])
    T_des_all[k] = float(pv[13])

  # Euler from R_raw
  theta_raw = np.zeros((T, 3), dtype=np.float64)
  for k in range(T):
    theta_raw[k, :] = R_to_euler_zyx_np(R_raw_all[k, :, :])

  # r_mrg = (r_raw + delta_theta_cmd) per requirement (small-angle assumption).
  theta_mrg = theta_raw + delta_theta_cmd

  # Unit conversions
  RAD2DEG = 180.0 / np.pi
  M2MM = 1000.0

  theta_deg = theta * RAD2DEG
  raw_deg = theta_raw * RAD2DEG
  mrg_deg = theta_mrg * RAD2DEG

  omega_deg_s = omega * RAD2DEG

  r_cot_mm = r_cot * M2MM
  r_cot_cmd_mm = r_cot_cmd * M2MM

  u_dth_rate_deg_s = u_rate_stage[:, 0:3] * RAD2DEG
  u_rcot_rate_mm_s = u_rate_stage[:, 3:5] * M2MM

  # Bounds (constant lines)
  cot_min = np.asarray(getattr(p, "COT_MIN", [-0.05, -0.05]), dtype=np.float64).ravel()
  cot_max = np.asarray(getattr(p, "COT_MAX", [0.05, 0.05]), dtype=np.float64).ravel()
  if cot_min.size < 2:
    cot_min = np.array([-0.05, -0.05], dtype=np.float64)
  if cot_max.size < 2:
    cot_max = np.array([0.05, 0.05], dtype=np.float64)

  cot_min_line = np.tile((cot_min * M2MM).reshape(1, 2), (T, 1))
  cot_max_line = np.tile((cot_max * M2MM).reshape(1, 2), (T, 1))

  Fmin = float(np.min(np.asarray(getattr(p, "F_MIN", [0, 0, 0, 0]), dtype=np.float64).ravel()))
  Fmax = float(np.max(np.asarray(getattr(p, "F_MAX", [0, 0, 0, 0]), dtype=np.float64).ravel()))
  F_min_line = np.full((N,), Fmin, dtype=np.float64)
  F_max_line = np.full((N,), Fmax, dtype=np.float64)

  # ----------------------------
  # Derived (right group): tau_d, F1234, cost breakdown
  # ----------------------------
  KR = gain3_np(getattr(p, "KR", 1.0))   # (3,)
  KW = gain3_np(getattr(p, "KW", 1.0))   # (3,)
  zeta = float(getattr(p, "ZETA", 0.02))

  # Stage-wise (k=0..N-1) per requirement.
  tau_d = np.zeros((N, 3), dtype=np.float64)
  for k in range(N):
    R = euler_zyx_to_R_np(theta[k, :])  # body->global
    R_d = R_raw_all[k, :, :] @ expm_hat_np(delta_theta_cmd[k, :])
    e_R = 0.5 * vee_np(R_d.T @ R - R.T @ R_d)
    omega_k = omega[k, :]
    omega_raw_k = omega_raw_all[k, :]
    e_w = omega_k - (R.T @ R_d @ omega_raw_k)
    tau_d[k, :] = -(KR * e_R) - (KW * e_w)

  # F1234 solve stage-wise
  F_stage = np.full((N, 4), np.nan, dtype=np.float64)
  for k in range(N):
    dx = float(r_cot[k, 0])
    dy = float(r_cot[k, 1])
    l = float(l_all[k])
    A = np.array([
      [ l - dy,  l - dy, -l - dy, -l - dy],
      [ l + dx, -l + dx, -l + dx,  l + dx],
      [-zeta,    zeta,   -zeta,    zeta ],
      [ -1.0,   -1.0,    -1.0,    -1.0 ],
    ], dtype=np.float64)
    w_d = np.array([tau_d[k, 0], tau_d[k, 1], tau_d[k, 2], float(T_des_all[k])], dtype=np.float64)

    try:
      F_stage[k, :] = np.linalg.solve(A, w_d)
    except np.linalg.LinAlgError:
      F_stage[k, :] = np.linalg.lstsq(A, w_d, rcond=None)[0]

  T_tot = np.sum(F_stage, axis=1)

  # tau_cot = r_cot x [0,0,T_des] (3D with z=0) per requirement.
  tau_cot = np.zeros((N, 3), dtype=np.float64)
  for k in range(N):
    x = float(r_cot[k, 0])
    y = float(r_cot[k, 1])
    Tdes = float(T_des_all[k])
    tau_cot[k, 0] = y * Tdes
    tau_cot[k, 1] = -x * Tdes
    tau_cot[k, 2] = 0.0

  # Cost breakdown (stage-wise)
  Q_OMEGA = np.asarray(getattr(p, "Q_OMEGA", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
  Q_THETA = np.asarray(getattr(p, "Q_THETA", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
  Q_COT = np.asarray(getattr(p, "Q_COT", [1.0, 1.0]), dtype=np.float64).ravel()
  R_THETA = np.asarray(getattr(p, "R_THETA", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
  R_COT = np.asarray(getattr(p, "R_COT", [1.0, 1.0]), dtype=np.float64).ravel()

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

    # NOTE: Stage-wise only; no terminal stage contribution here.
    J_u_ch[k, 0] = float(Q_THETA[0] * dtc[0] * dtc[0] + R_THETA[0] * dtc_rate[0] * dtc_rate[0])
    J_u_ch[k, 1] = float(Q_THETA[1] * dtc[1] * dtc[1] + R_THETA[1] * dtc_rate[1] * dtc_rate[1])
    J_u_ch[k, 2] = float(Q_THETA[2] * dtc[2] * dtc[2] + R_THETA[2] * dtc_rate[2] * dtc_rate[2])
    J_u_ch[k, 3] = float(Q_COT[0] * rcc[0] * rcc[0] + R_COT[0] * rcc_rate[0] * rcc_rate[0])
    J_u_ch[k, 4] = float(Q_COT[1] * rcc[1] * rcc[1] + R_COT[1] * rcc_rate[1] * rcc_rate[1])

    J_total[k] = J_omega[k] + float(np.sum(J_u_ch[k, :]))

  # Time axes
  t_x = np.arange(T, dtype=np.float64) * dt
  t_u = np.arange(N, dtype=np.float64) * dt

  # T_des on stage axis (for T_tot plot)
  T_des_stage = T_des_all[:N].copy()

  frame: DebugFrame = {
    "t_x": t_x,
    "t_u": t_u,

    # Left group (raw horizon)
    "theta_deg": theta_deg,
    "raw_deg": raw_deg,
    "mrg_deg": mrg_deg,
    "omega_deg_s": omega_deg_s,
    "r_cot_mm": r_cot_mm,
    "r_cot_cmd_mm": r_cot_cmd_mm,
    "cot_min_line": cot_min_line,
    "cot_max_line": cot_max_line,
    "u_dth_rate_deg_s": u_dth_rate_deg_s,
    "u_rcot_rate_mm_s": u_rcot_rate_mm_s,

    # Right group (derived)
    "F_stage": F_stage,
    "F_min_line": F_min_line,
    "F_max_line": F_max_line,

    "J_omega": J_omega,
    "J_u_ch": J_u_ch,
    "J_total": J_total,

    "tau_d": tau_d,
    "tau_cot": tau_cot,
    "T_des_stage": T_des_stage,
    "T_tot": T_tot,
  }
  return frame

# ----------------------------
# GUI
# ----------------------------
class MPCViewerMainWindow(QMainWindow):
  def __init__(self) -> None:
    super().__init__()
    self.setWindowTitle("MPC one horizon breakdown")

    # Pen configuration (keep similar style to existing viewer).
    lw = 2.0
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
          curves.append(w.plot(pen=pen))  # no legend entry
        else:
          curves.append(w.plot(name=name, pen=pen))
      return w, curves

    self._plots: Dict[str, Tuple[pg.PlotWidget, list[Any]]] = {}

    # ----------------------------
    # Layout: left groupbox + right groupbox
    # ----------------------------
    central = QWidget()
    self.setCentralWidget(central)

    root = QVBoxLayout(central)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    h_splitter = QSplitter(Qt.Horizontal)
    root.addWidget(h_splitter)

    # Left group box (raw horizon)
    gb_l = QGroupBox()
    gb_l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    gl = QGridLayout(gb_l)
    gl.setContentsMargins(8, 8, 8, 8)
    gl.setSpacing(6)

    # Row 1: Roll/Pitch/Yaw (r, raw, r_mrg)
    self._plots["roll"] = make_plot("Roll", "[deg]", [("r", pens[2]), ("raw", pen_dash), ("r_mrg", pen_dash_red)])
    self._plots["pitch"] = make_plot("Pitch", "[deg]", [("p", pens[2]), ("raw", pen_dash), ("p_mrg", pen_dash_red)])
    self._plots["yaw"] = make_plot("Yaw", "[deg]", [("y", pens[2]), ("raw", pen_dash), ("y_mrg", pen_dash_red)])

    for k in ("roll", "pitch", "yaw"):
      w = self._plots[k][0]
      pi = w.getPlotItem()
      pi.enableAutoRange(axis='y', enable=False)
      pi.setYRange(-25.0, 25.0, padding=0.0)

    # Row 2: wx/wy/wz (omega)
    self._plots["wx"] = make_plot("wx", "[deg/s]", [("wx", pens[2])])
    self._plots["wy"] = make_plot("wy", "[deg/s]", [("wy", pens[2])])
    self._plots["wz"] = make_plot("wz", "[deg/s]", [("wz", pens[2])])

    for k in ("wx", "wy", "wz"):
      w = self._plots[k][0]
      pi = w.getPlotItem()
      pi.enableAutoRange(axis='y', enable=False)
      pi.setYRange(-10.0, 10.0, padding=0.0)

    # Row 3: cot_x/cot_y (r_cot, r_cot_cmd, bounds)
    self._plots["cotx"] = make_plot("cot_x", "[mm]", [("cot", pens[2]), ("cmd", pens[0]), (None, pen_dash), (None, pen_dash_red)])
    self._plots["coty"] = make_plot("cot_y", "[mm]", [("cot", pens[2]), ("cmd", pens[0]), (None, pen_dash), (None, pen_dash_red)])

    # Row 4: u_theta/u_cot (u_rate)
    self._plots["u_theta"] = make_plot(
      "u_theta_rate", "[deg/s]",
      [("dth_x", pens[0]), ("dth_y", pens[1]), ("dth_z", pens[3])]
    )
    self._plots["u_cot"] = make_plot(
      "u_cot_rate", "[mm/s]",
      [("dcot_x", pens[5]), ("dcot_y", pens[6])]
    )

    gl.addWidget(self._plots["roll"][0], 0, 0)
    gl.addWidget(self._plots["pitch"][0], 0, 1)
    gl.addWidget(self._plots["yaw"][0], 0, 2)

    gl.addWidget(self._plots["wx"][0], 1, 0)
    gl.addWidget(self._plots["wy"][0], 1, 1)
    gl.addWidget(self._plots["wz"][0], 1, 2)

    gl.addWidget(self._plots["cotx"][0], 2, 0, 1, 1)
    gl.addWidget(self._plots["coty"][0], 2, 1, 1, 1)

    gl.addWidget(self._plots["u_theta"][0], 3, 0, 1, 1)
    gl.addWidget(self._plots["u_cot"][0], 3, 1, 1, 1)

    h_splitter.addWidget(gb_l)

    # Right group box (derived horizon)
    gb_r = QGroupBox()
    gb_r.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    gr = QGridLayout(gb_r)
    gr.setContentsMargins(8, 8, 8, 8)
    gr.setSpacing(6)

    # Row 1: Thruster forces (F1..F4 + bounds)
    self._plots["F"] = make_plot(
      "Thruster", "[N]",
      [("F1", pens[0]), ("F2", pens[1]), ("F3", pens[2]), ("F4", pens[3]), (None, pen_dash), (None, pen_dash_red)]
    )

    # Row 2: Cost breakdown (stage-wise)
    pen_total = pg.mkPen(color=(0, 0, 0), width=lw, style=Qt.DashLine)
    self._plots["cost"] = make_plot(
      "Cost", "cost",
      [("J_omega", pens[2]),
       ("J_dth_x", pens[0]),
       ("J_dth_y", pens[1]),
       ("J_dth_z", pens[3]),
       ("J_rcot_x", pens[5]),
       ("J_rcot_y", pens[6]),
       ("J_total", pen_total)]
    )

    # Row 3: tau_x / tau_y (tau_thrust, tau_cot)
    self._plots["tau_x"] = make_plot("tau_x", "[Nm]", [("tau_thrust", pens[0]), ("tau_cot", pens[1])])
    self._plots["tau_y"] = make_plot("tau_y", "[Nm]", [("tau_thrust", pens[0]), ("tau_cot", pens[1])])

    for k in ("tau_x", "tau_y"):
      w = self._plots[k][0]
      pi = w.getPlotItem()
      pi.enableAutoRange(axis='y', enable=False)
      pi.setYRange(-2.0, 2.0, padding=0.0)

    # Row 4: tau_z and T_tot
    self._plots["tau_z"] = make_plot("tau_z", "[Nm]", [("tau_thrust", pens[2])])
    self._plots["tau_z"][0].enableAutoRange(axis='y', enable=False)
    self._plots["tau_z"][0].setYRange(-1.0, 1.0, padding=0.0)

    self._plots["T_tot"] = make_plot("total thrust", "[N]", [("sum_F1234", pens[4]), ("T_des", pen_dash_red)])
    self._plots["T_tot"][0].enableAutoRange(axis='y', enable=False)
    self._plots["T_tot"][0].setYRange(50.0, 70.0, padding=0.0)

    gr.addWidget(self._plots["F"][0], 0, 0, 1, 2)
    gr.addWidget(self._plots["cost"][0], 1, 0, 1, 2)
    gr.addWidget(self._plots["tau_x"][0], 2, 0, 1, 1)
    gr.addWidget(self._plots["tau_y"][0], 2, 1, 1, 1)
    gr.addWidget(self._plots["tau_z"][0], 3, 0, 1, 1)
    gr.addWidget(self._plots["T_tot"][0], 3, 1, 1, 1)

    h_splitter.addWidget(gb_r)

    h_splitter.setStretchFactor(0, 1)
    h_splitter.setStretchFactor(1, 1)

    # ---- Initial window size: 3:2 aspect ratio ----
    screen = QApplication.primaryScreen()
    if screen is not None:
      avail = screen.availableGeometry()
      # Use a conservative fraction of available width
      w0 = int(avail.width() * 0.75)
      h0 = int(w0 * 2 / 3)  # 3:2 => h = (2/3) w

      # Clamp to available height (keep 3:2 as much as possible)
      if h0 > int(avail.height() * 0.85):
        h0 = int(avail.height() * 0.85)
        w0 = int(h0 * 3 / 2)

      self.resize(w0, h0)
    else:
      self.resize(1200, 800)  # fallback: 3:2

  def update_from_frame(self, frame: DebugFrame) -> None:
    t_x = _get(frame, "t_x")
    t_u = _get(frame, "t_u")
    if t_x is None or t_u is None:
      return

    def set_plot(key: str, xs: Any, ys: list[np.ndarray]) -> None:
      _, curves = self._plots[key]
      for c, y in zip(curves, ys):
        c.setData(xs, y)

    # Left group
    theta_deg = _get(frame, "theta_deg")
    raw_deg = _get(frame, "raw_deg")
    mrg_deg = _get(frame, "mrg_deg")
    if theta_deg is not None and raw_deg is not None and mrg_deg is not None:
      set_plot("roll", t_x, [theta_deg[:, 0], raw_deg[:, 0], mrg_deg[:, 0]])
      set_plot("pitch", t_x, [theta_deg[:, 1], raw_deg[:, 1], mrg_deg[:, 1]])
      set_plot("yaw", t_x, [theta_deg[:, 2], raw_deg[:, 2], mrg_deg[:, 2]])

    omega_deg_s = _get(frame, "omega_deg_s")
    if omega_deg_s is not None:
      set_plot("wx", t_x, [omega_deg_s[:, 0]])
      set_plot("wy", t_x, [omega_deg_s[:, 1]])
      set_plot("wz", t_x, [omega_deg_s[:, 2]])

    r_cot_mm = _get(frame, "r_cot_mm")
    r_cot_cmd_mm = _get(frame, "r_cot_cmd_mm")
    cot_min_line = _get(frame, "cot_min_line")
    cot_max_line = _get(frame, "cot_max_line")
    if (r_cot_mm is not None) and (r_cot_cmd_mm is not None) and (cot_min_line is not None) and (cot_max_line is not None):
      set_plot("cotx", t_x, [r_cot_mm[:, 0], r_cot_cmd_mm[:, 0], cot_min_line[:, 0], cot_max_line[:, 0]])
      set_plot("coty", t_x, [r_cot_mm[:, 1], r_cot_cmd_mm[:, 1], cot_min_line[:, 1], cot_max_line[:, 1]])

    u_dth_rate_deg_s = _get(frame, "u_dth_rate_deg_s")
    if u_dth_rate_deg_s is not None and u_dth_rate_deg_s.shape[0] == t_u.shape[0]:
      set_plot("u_theta", t_u, [u_dth_rate_deg_s[:, 0], u_dth_rate_deg_s[:, 1], u_dth_rate_deg_s[:, 2]])

    u_rcot_rate_mm_s = _get(frame, "u_rcot_rate_mm_s")
    if u_rcot_rate_mm_s is not None and u_rcot_rate_mm_s.shape[0] == t_u.shape[0]:
      set_plot("u_cot", t_u, [u_rcot_rate_mm_s[:, 0], u_rcot_rate_mm_s[:, 1]])

    # Right group
    F_stage = _get(frame, "F_stage")
    F_min_line = _get(frame, "F_min_line")
    F_max_line = _get(frame, "F_max_line")
    if F_stage is not None and F_min_line is not None and F_max_line is not None and F_stage.shape[0] == t_u.shape[0]:
      set_plot("F", t_u, [F_stage[:, 0], F_stage[:, 1], F_stage[:, 2], F_stage[:, 3], F_min_line, F_max_line])

    J_omega = _get(frame, "J_omega")
    J_u_ch = _get(frame, "J_u_ch")
    J_total = _get(frame, "J_total")
    if J_omega is not None and J_u_ch is not None and J_total is not None and J_omega.shape[0] == t_u.shape[0]:
      set_plot(
        "cost", t_u,
        [J_omega,
         J_u_ch[:, 0], J_u_ch[:, 1], J_u_ch[:, 2],
         J_u_ch[:, 3], J_u_ch[:, 4],
         J_total]
      )

    tau_d = _get(frame, "tau_d")
    tau_cot = _get(frame, "tau_cot")
    if tau_d is not None and tau_cot is not None and tau_d.shape[0] == t_u.shape[0]:
      set_plot("tau_x", t_u, [tau_d[:, 0], tau_cot[:, 0]])
      set_plot("tau_y", t_u, [tau_d[:, 1], tau_cot[:, 1]])
      set_plot("tau_z", t_u, [tau_d[:, 2]])

    T_des_stage = _get(frame, "T_des_stage")
    T_tot = _get(frame, "T_tot")
    if T_des_stage is not None and T_tot is not None and T_tot.shape[0] == t_u.shape[0]:
      # Plot order: sumF (solid), T_des (dashed red)
      set_plot("T_tot", t_u, [T_tot, -T_des_stage]) # F1234 is positive, T_des_stage is negative

# ----------------------------
# App main
# ----------------------------
def main() -> int:
  app = QApplication(sys.argv)

  app.setStyleSheet("""
  QWidget { background: #ffffff; color: #111111; }
  QMainWindow { background: #ffffff; }
  QGroupBox {
    border: 1px solid #cfcfcf;
    border-radius: 6px;
    margin-top: 8px;
    font-weight: bold;
  }
  QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px 0 4px;
  }
  """)

  win = MPCViewerMainWindow()
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

      frame = compute_frame_from_pkt(pkt)
      win.update_from_frame(frame)

    except Exception:
      traceback.print_exc()

  timer = QTimer()
  timer.timeout.connect(poll_mmap)
  timer.start(100)

  # Allow Ctrl+C in terminal to quit cleanly.
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
