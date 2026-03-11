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
  QGridLayout,
  QLabel,
  QMainWindow,
  QSizePolicy,
  QVBoxLayout,
  QWidget,
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


ForceFrame = Dict[str, np.ndarray]


def _gain3(x: Any) -> np.ndarray:
  a = np.asarray(x, dtype=np.float64).reshape(-1)
  if a.size == 1:
    return np.repeat(a.item(), 3)
  if a.size != 3:
    raise ValueError(f"Expected scalar or length-3 gain, got size={a.size}")
  return a


def _euler_zyx_to_R_np(theta: np.ndarray) -> np.ndarray:
  th = np.asarray(theta, dtype=np.float64).ravel()
  if th.size != 3:
    return np.eye(3, dtype=np.float64)

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


def _hat_np(w: np.ndarray) -> np.ndarray:
  w = np.asarray(w, dtype=np.float64).ravel()
  if w.size != 3:
    return np.zeros((3, 3), dtype=np.float64)

  return np.array([
    [0.0,   -w[2],  w[1]],
    [w[2],   0.0,  -w[0]],
    [-w[1],  w[0],  0.0],
  ], dtype=np.float64)


def _expm_hat_np(w: np.ndarray) -> np.ndarray:
  w = np.asarray(w, dtype=np.float64).ravel()
  if w.size != 3:
    return np.eye(3, dtype=np.float64)

  th2 = float(np.dot(w, w))
  th = float(np.sqrt(th2))
  K = _hat_np(w)
  I = np.eye(3, dtype=np.float64)

  if th < 1e-10:
    return I + K

  A = np.sin(th) / th
  B = (1.0 - np.cos(th)) / (th2 + 1e-12)
  return I + A * K + B * (K @ K)


def _vee_np(S: np.ndarray) -> np.ndarray:
  S = np.asarray(S, dtype=np.float64).reshape(3, 3)
  return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def _solve_force_horizon(pkt: MMapPacket) -> ForceFrame:
  """
  Reconstruct thrust horizon from the current mmap packet.

  Assumed model convention:
    x = [
      theta(0:3), omega(3:6),
      r1(6:8), r2(8:10), r3(10:12), r4(12:14),
      delta_theta_cmd(14:17),
      r1_cmd(17:19), r2_cmd(19:21), r3_cmd(21:23), r4_cmd(23:25)
    ]

    p = [
      R_raw(0:9), W_raw(9:12), Wdot_raw(12:15),
      R_0(15:24), f_0(24)
    ]
  """
  N = int(pkt.N)
  nx = int(pkt.nx)
  np_ = int(pkt.np)

  x_all = np.asarray(pkt.x_all, dtype=np.float64).reshape(N + 1, nx)
  p_all = np.asarray(pkt.p_all, dtype=np.float64).reshape(N + 1, np_)

  KR = _gain3(getattr(p, "KR"))
  KW = _gain3(getattr(p, "KW"))
  zeta = float(getattr(p, "ZETA"))

  m_arm = np.asarray(getattr(p, "M_ARM"), dtype=np.float64).reshape(4)
  m_body = float(getattr(p, "M_BODY"))
  m_tot = m_body + float(np.sum(m_arm))

  r_off_x = np.asarray(getattr(p, "R_OFF_X"), dtype=np.float64).reshape(4)
  r_off_y = np.asarray(getattr(p, "R_OFF_Y"), dtype=np.float64).reshape(4)

  F_stage = np.full((N, 4), np.nan, dtype=np.float64)
  tau_d_stage = np.full((N, 3), np.nan, dtype=np.float64)
  bFz_stage = np.full((N,), np.nan, dtype=np.float64)

  for k in range(N):
    xk = x_all[k, :]
    pk = p_all[k, :]

    if xk.size < 25 or pk.size < 25:
      continue

    theta = xk[0:3]
    omega = xk[3:6]
    delta_theta_cmd = xk[14:17]

    # Rotor positions in body x-y plane
    r_xy = np.zeros((4, 2), dtype=np.float64)
    for a in range(4):
      rho = float(xk[6 + 2 * a])
      alpha = float(xk[7 + 2 * a])
      r_xy[a, 0] = r_off_x[a] + rho * np.cos(alpha)
      r_xy[a, 1] = r_off_y[a] + rho * np.sin(alpha)

    pc = (m_arm[:, None] * r_xy).sum(axis=0) / m_tot

    # CasADi reshape is column-major
    R_raw = pk[0:9].reshape(3, 3, order="F")
    W_raw = pk[9:12]
    R_0 = pk[15:24].reshape(3, 3, order="F")
    f_0 = float(pk[24])

    R = _euler_zyx_to_R_np(theta)
    Rd = R_raw @ _expm_hat_np(delta_theta_cmd)
    Wd = _expm_hat_np(-delta_theta_cmd) @ W_raw

    RtRd = R.T @ Rd
    e_R = 0.5 * _vee_np(RtRd.T - RtRd)
    e_w = omega - RtRd @ Wd
    tau_d = -(KR * e_R) - (KW * e_w)

    b_F0 = np.array([0.0, 0.0, -f_0], dtype=np.float64)
    g_F0 = R_0 @ b_F0
    b_F = R.T @ g_F0

    A = np.array([
      [-(r_xy[0, 1] - pc[1]), -(r_xy[1, 1] - pc[1]), -(r_xy[2, 1] - pc[1]), -(r_xy[3, 1] - pc[1])],
      [ (r_xy[0, 0] - pc[0]),  (r_xy[1, 0] - pc[0]),  (r_xy[2, 0] - pc[0]),  (r_xy[3, 0] - pc[0])],
      [-zeta, zeta, -zeta, zeta],
      [-1.0, -1.0, -1.0, -1.0],
    ], dtype=np.float64)

    w_d = np.array([tau_d[0], tau_d[1], tau_d[2], b_F[2]], dtype=np.float64)

    try:
      F_stage[k, :] = np.linalg.solve(A, w_d)
    except np.linalg.LinAlgError:
      F_stage[k, :] = np.linalg.lstsq(A, w_d, rcond=None)[0]

    tau_d_stage[k, :] = tau_d
    bFz_stage[k] = b_F[2]

  steps = np.arange(N, dtype=np.float64)

  F_min = np.asarray(getattr(c, "F_MIN", np.zeros(4)), dtype=np.float64).reshape(-1)
  F_max = np.asarray(getattr(c, "F_MAX", np.zeros(4)), dtype=np.float64).reshape(-1)

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
    "bFz": bFz_stage,
  }


class ThrusterViewerMainWindow(QMainWindow):
  def __init__(self) -> None:
    super().__init__()
    self.setWindowTitle("MPC Thruster Horizon Viewer")

    lw = 2.0
    pen_f1 = pg.mkPen(color=(220, 50, 50), width=lw)
    pen_f2 = pg.mkPen(color=(50, 160, 70), width=lw)
    pen_f3 = pg.mkPen(color=(50, 90, 220), width=lw)
    pen_f4 = pg.mkPen(color=(120, 60, 200), width=lw)
    pen_lo = pg.mkPen(color=(80, 80, 80), width=1.5, style=Qt.DashLine)
    pen_hi = pg.mkPen(color=(0, 0, 0), width=1.5, style=Qt.DashLine)

    self._pens = [pen_f1, pen_f2, pen_f3, pen_f4]
    self._plots: Dict[str, Tuple[pg.PlotWidget, List[Any]]] = {}

    def make_plot(
      title: str,
      y_label: str,
      curve_specs: List[Tuple[Optional[str], Any]],
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
        if name is None:
          curves.append(w.plot(pen=pen))
        else:
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

    # Top combined plot
    gb_top = QGroupBox()
    gb_top.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    gl_top = QGridLayout(gb_top)
    gl_top.setContentsMargins(8, 8, 8, 8)
    gl_top.setSpacing(6)

    fmin = np.asarray(getattr(c, "F_MIN", np.zeros(4)), dtype=np.float64).reshape(-1)
    fmax = np.asarray(getattr(c, "F_MAX", np.zeros(4)), dtype=np.float64).reshape(-1)
    y_lo = float(np.nanmin(fmin)) - 2.0 if fmin.size > 0 else -1.0
    y_hi = float(np.nanmax(fmax)) + 2.0 if fmax.size > 0 else 20.0

    self._plots["all"] = make_plot(
      "Thruster horizon",
      "[N]",
      [("F1", pen_f1), ("F2", pen_f2), ("F3", pen_f3), ("F4", pen_f4),
       ("Fmin", pen_lo), ("Fmax", pen_hi)],
      fixed_y_range=(y_lo, y_hi),
    )
    gl_top.addWidget(self._plots["all"][0], 0, 0)
    root.addWidget(gb_top, stretch=2)

    # Bottom individual plots
    gb_bot = QGroupBox()
    gb_bot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    gl_bot = QGridLayout(gb_bot)
    gl_bot.setContentsMargins(8, 8, 8, 8)
    gl_bot.setSpacing(6)

    self._plots["F1"] = make_plot("F1", "[N]", [("F1", pen_f1), ("min", pen_lo), ("max", pen_hi)], fixed_y_range=(y_lo, y_hi))
    self._plots["F2"] = make_plot("F2", "[N]", [("F2", pen_f2), ("min", pen_lo), ("max", pen_hi)], fixed_y_range=(y_lo, y_hi))
    self._plots["F3"] = make_plot("F3", "[N]", [("F3", pen_f3), ("min", pen_lo), ("max", pen_hi)], fixed_y_range=(y_lo, y_hi))
    self._plots["F4"] = make_plot("F4", "[N]", [("F4", pen_f4), ("min", pen_lo), ("max", pen_hi)], fixed_y_range=(y_lo, y_hi))

    gl_bot.addWidget(self._plots["F1"][0], 0, 0)
    gl_bot.addWidget(self._plots["F2"][0], 0, 1)
    gl_bot.addWidget(self._plots["F3"][0], 1, 0)
    gl_bot.addWidget(self._plots["F4"][0], 1, 1)

    # Link x-axis
    self._plots["F1"][0].setXLink(self._plots["all"][0])
    self._plots["F2"][0].setXLink(self._plots["all"][0])
    self._plots["F3"][0].setXLink(self._plots["all"][0])
    self._plots["F4"][0].setXLink(self._plots["all"][0])

    root.addWidget(gb_bot, stretch=2)

    screen = QApplication.primaryScreen()
    if screen is not None:
      avail = screen.availableGeometry()
      w0 = int(avail.width() * 0.70)
      h0 = int(avail.height() * 0.75)
      self.resize(w0, h0)
    else:
      self.resize(1200, 900)

  def _set_plot(self, key: str, x: np.ndarray, ys: List[np.ndarray]) -> None:
    _, curves = self._plots[key]
    for curve, y in zip(curves, ys):
      curve.setData(x, y)

  def update_from_packet(self, pkt: MMapPacket) -> None:
    frame = _solve_force_horizon(pkt)

    steps = frame["steps"]
    F_stage = frame["F_stage"]
    F_min = frame["F_min"]
    F_max = frame["F_max"]

    if steps.size == 0 or F_stage.shape[0] == 0:
      return

    fmin_line = np.full_like(steps, float(np.nanmin(F_min)), dtype=np.float64)
    fmax_line = np.full_like(steps, float(np.nanmax(F_max)), dtype=np.float64)

    self._set_plot(
      "all",
      steps,
      [F_stage[:, 0], F_stage[:, 1], F_stage[:, 2], F_stage[:, 3], fmin_line, fmax_line],
    )

    self._set_plot("F1", steps, [F_stage[:, 0], np.full_like(steps, F_min[0]), np.full_like(steps, F_max[0])])
    self._set_plot("F2", steps, [F_stage[:, 1], np.full_like(steps, F_min[1]), np.full_like(steps, F_max[1])])
    self._set_plot("F3", steps, [F_stage[:, 2], np.full_like(steps, F_min[2]), np.full_like(steps, F_max[2])])
    self._set_plot("F4", steps, [F_stage[:, 3], np.full_like(steps, F_min[3]), np.full_like(steps, F_max[3])])

    f_now = F_stage[0, :]
    self.info_label.setText(
      f"seq={pkt.seq}   status={pkt.status}   solve_ms={pkt.solve_ms:.3f}   "
      f"F[k=0]=[{f_now[0]:.3f}, {f_now[1]:.3f}, {f_now[2]:.3f}, {f_now[3]:.3f}]"
    )


def main() -> int:
  pg.setConfigOptions(antialias=True)

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
  QLabel {
    font-family: monospace;
    font-size: 12px;
    padding: 2px 4px 2px 4px;
  }
  """)

  win = ThrusterViewerMainWindow()
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