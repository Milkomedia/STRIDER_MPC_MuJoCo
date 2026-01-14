import os
import mmap
import struct
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


# -----------------------------
# Must match C++ mmap_manager.hpp
# -----------------------------
HEADER_SIZE = 64
LOGDATA_SIZE = 152
SLOT_SIZE = 160  # seq(u64)=8 + LogData(152)=160

MAGIC = b"STRLOG2\x00"
VERSION = 2


@dataclass
class Header:
  magic: bytes
  version: int
  header_size: int
  capacity: int
  slot_size: int
  write_count: int
  start_time_ns: int

  @staticmethod
  def parse(buf: bytes) -> "Header":
    if len(buf) < HEADER_SIZE:
      raise ValueError("Header buffer too small")

    magic = buf[0:8]
    version, header_size, capacity, slot_size = struct.unpack_from("<IIII", buf, 8)
    write_count, start_time_ns = struct.unpack_from("<QQ", buf, 24)
    return Header(magic, version, header_size, capacity, slot_size, write_count, start_time_ns)


class MMapReader:
  def __init__(self, path: str = "/tmp/strider_log.mmap"):
    self.path = path
    self.fd: Optional[int] = None
    self.mm: Optional[mmap.mmap] = None
    self.header: Optional[Header] = None

  def open(self) -> None:
    if self.mm is not None:
      return

    self.fd = os.open(self.path, os.O_RDONLY)
    st = os.fstat(self.fd)
    self.mm = mmap.mmap(self.fd, st.st_size, access=mmap.ACCESS_READ)

    self.header = Header.parse(self.mm[0:HEADER_SIZE])

    if self.header.magic != MAGIC:
      raise RuntimeError(f"Bad magic: {self.header.magic}")
    if self.header.version != VERSION:
      raise RuntimeError(f"Unsupported version: {self.header.version}")
    if self.header.header_size != HEADER_SIZE:
      raise RuntimeError(f"Header size mismatch: {self.header.header_size}")
    if self.header.slot_size != SLOT_SIZE:
      raise RuntimeError(f"Slot size mismatch: {self.header.slot_size}")

  def close(self) -> None:
    if self.mm is not None:
      self.mm.close()
      self.mm = None
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None

  def _u64(self, offset: int) -> int:
    return struct.unpack_from("<Q", self.mm, offset)[0]

  def read_all(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cap = self.header.capacity

    wc = int(self._u64(24))
    n = int(min(wc, cap))
    if n <= 0:
      return np.zeros((0,), dtype=np.float32), {}

    start = wc - n
    base = HEADER_SIZE

    t = np.empty((n,), dtype=np.float32)
    pos_d = np.empty((n, 3), dtype=np.float32)
    pos = np.empty((n, 3), dtype=np.float32)

    rpy = np.empty((n, 3), dtype=np.float32)
    rpy_raw = np.empty((n, 3), dtype=np.float32)
    rpy_d = np.empty((n, 3), dtype=np.float32)

    tau_d = np.empty((n, 3), dtype=np.float32)
    tau_off = np.empty((n, 2), dtype=np.float32)
    tau_thrust = np.empty((n, 2), dtype=np.float32)

    tilt = np.empty((n, 4), dtype=np.float32)
    f_thrust = np.empty((n, 4), dtype=np.float32)
    f_total = np.empty((n,), dtype=np.float32)

    r_cot = np.empty((n, 2), dtype=np.float32)
    r_cot_cmd = np.empty((n, 2), dtype=np.float32)

    solve_ms = np.empty((n,), dtype=np.float32)
    solve_status = np.empty((n,), dtype=np.int32)

    # Offsets in LogData (packed)
    OFF_T = 0
    OFF_POS_D = 4
    OFF_POS = 16
    OFF_RPY = 28
    OFF_RPY_RAW = 40
    OFF_RPY_D = 52
    OFF_TAU_D = 64
    OFF_TAU_OFF = 76
    OFF_TAU_THRUST = 84
    OFF_TILT = 92
    OFF_F_THRUST = 108
    OFF_F_TOTAL = 124
    OFF_R_COT = 128
    OFF_R_COT_CMD = 136
    OFF_SOLVE_MS = 144
    OFF_SOLVE_STATUS = 148

    for i in range(n):
      logical = start + i
      idx = logical % cap
      slot_off = base + idx * SLOT_SIZE

      for _ in range(10):
        seq_a = self._u64(slot_off + 0)
        if seq_a & 1:
          continue

        dbuf = self.mm[slot_off + 8: slot_off + 8 + LOGDATA_SIZE]

        seq_b = self._u64(slot_off + 0)
        if seq_a != seq_b or (seq_b & 1):
          continue

        t[i] = struct.unpack_from("<f", dbuf, OFF_T)[0]
        pos_d[i, :] = struct.unpack_from("<fff", dbuf, OFF_POS_D)
        pos[i, :] = struct.unpack_from("<fff", dbuf, OFF_POS)

        rpy[i, :] = struct.unpack_from("<fff", dbuf, OFF_RPY)
        rpy_raw[i, :] = struct.unpack_from("<fff", dbuf, OFF_RPY_RAW)
        rpy_d[i, :] = struct.unpack_from("<fff", dbuf, OFF_RPY_D)

        tau_d[i, :] = struct.unpack_from("<fff", dbuf, OFF_TAU_D)
        tau_off[i, :] = struct.unpack_from("<ff", dbuf, OFF_TAU_OFF)
        tau_thrust[i, :] = struct.unpack_from("<ff", dbuf, OFF_TAU_THRUST)

        tilt[i, :] = struct.unpack_from("<ffff", dbuf, OFF_TILT)
        f_thrust[i, :] = struct.unpack_from("<ffff", dbuf, OFF_F_THRUST)
        f_total[i] = struct.unpack_from("<f", dbuf, OFF_F_TOTAL)[0]

        r_cot[i, :] = struct.unpack_from("<ff", dbuf, OFF_R_COT)
        r_cot_cmd[i, :] = struct.unpack_from("<ff", dbuf, OFF_R_COT_CMD)

        solve_ms[i] = struct.unpack_from("<f", dbuf, OFF_SOLVE_MS)[0]
        solve_status[i] = struct.unpack_from("<i", dbuf, OFF_SOLVE_STATUS)[0]
        break

    ch = {
      "pos_d": pos_d, "pos": pos,
      "rpy": rpy, "rpy_raw": rpy_raw, "rpy_d": rpy_d,
      "tau_d": tau_d, "tau_off": tau_off, "tau_thrust": tau_thrust,
      "tilt": tilt, "f_thrust": f_thrust, "f_total": f_total,
      "r_cot": r_cot, "r_cot_cmd": r_cot_cmd,
      "solve_ms": solve_ms, "solve_status": solve_status,
      "write_count": np.int64(wc),
    }
    return t, ch


# -----------------------------
# UI helpers
# -----------------------------
def _deg(x: np.ndarray) -> np.ndarray:
  return x * (180.0 / np.pi)

def _mm(x: np.ndarray) -> np.ndarray:
  return x * 1000.0

def _style(plot: pg.PlotItem) -> None:
  plot.showGrid(x=True, y=True)
  plot.getAxis("bottom").setPen(pg.mkPen("k"))
  plot.getAxis("left").setPen(pg.mkPen("k"))
  plot.getAxis("bottom").setTextPen(pg.mkPen("k"))
  plot.getAxis("left").setTextPen(pg.mkPen("k"))

def _mk_pen(style: str, width: int = 2, color=None) -> pg.mkPen:
  if style == "solid":
    return pg.mkPen(color=color, width=width)
  if style == "dash":
    return pg.mkPen(color=color, width=width, style=QtCore.Qt.DashLine)
  if style == "dot":
    return pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
  if style == "dashdot":
    return pg.mkPen(color=color, width=width, style=QtCore.Qt.DashDotLine)
  return pg.mkPen(color=color, width=width)


class LoggerWindow(QtWidgets.QMainWindow):
  def __init__(self, path="/tmp/strider_log.mmap", update_ms: int = 100):
    super().__init__()
    self.setWindowTitle("STRIDER Logger (mmap)")

    self.reader = MMapReader(path)
    self.update_ms = int(update_ms)

    self._curves: Dict[str, pg.PlotDataItem] = {}

    # plot linking
    self._all_plots: List[pg.PlotItem] = []
    self._x_master: Optional[pg.PlotItem] = None

    # status scatter + legend colors
    self._status_plot: Optional[pg.PlotItem] = None
    self._status_scatter: Optional[pg.ScatterPlotItem] = None
    self._status_colors = {
      0: (0, 200, 0, 220),     # green
      1: (255, 230, 0, 220),   # yellow
      2: (255, 140, 0, 220),   # orange
      3: (255, 0, 0, 220),     # red
      4: (160, 0, 255, 220),   # purple
    }
    self._status_names = {
      0: "SUCCESS",
      1: "NAN_DETECTED",
      2: "MAXITER",
      3: "MINSTEP",
      4: "QP_FAIL",
    }
    self._status_bars: Dict[int, pg.BarGraphItem] = {}

    self._init_ui()

    self.timer = QtCore.QTimer(self)
    self.timer.timeout.connect(self.on_timer)
    self.timer.start(self.update_ms)

  def _init_ui(self) -> None:
    cw = QtWidgets.QWidget()
    self.setCentralWidget(cw)
    layout = QtWidgets.QVBoxLayout(cw)

    top = QtWidgets.QHBoxLayout()
    self.lbl_path = QtWidgets.QLabel("path: " + self.reader.path)
    self.lbl_stat = QtWidgets.QLabel("waiting...")
    self.lbl_stat.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    top.addWidget(self.lbl_path, 1)
    top.addWidget(self.lbl_stat, 1)
    layout.addLayout(top)

    self.glw = pg.GraphicsLayoutWidget()
    layout.addWidget(self.glw, 1)

    # -------------------------
    # Pens (explicit colors)
    # -------------------------
    pen_act   = _mk_pen("solid", width=3, color="b")                  # blue solid
    pen_des   = _mk_pen("dash",  width=3, color="r")                  # red dashed
    pen_raw   = _mk_pen("dot",   width=3, color="k")                  # black dotted
    pen_total = _mk_pen("dot",   width=3, color="k")                  # black dotted
    pen_cot   = _mk_pen("solid", width=3, color=(255, 105, 180))      # pink solid
    pen_thr   = _mk_pen("solid", width=3, color="b")                  # blue solid
    pen_rcot_x_cmd = _mk_pen("dash",  width=3, color="r")  # red dashed
    pen_rcot_x_act = _mk_pen("solid", width=3, color="r")  # red solid
    pen_rcot_y_cmd = _mk_pen("dash",  width=3, color="b")  # blue dashed
    pen_rcot_y_act = _mk_pen("solid", width=3, color="b")  # blue solid
    rotor_colors = ["b", "g", "m", "c"]

    # Helper: create plot and link X to master
    def _mk_plot(row: int, col: int, title: str,
                add_legend: bool = True,
                y_range: Optional[Tuple[float, float]] = None) -> pg.PlotItem:
      p = self.glw.addPlot(row=row, col=col, title=title)
      _style(p)
      p.setLabel("bottom", "t", units="sec")
      p.enableAutoRange(axis="y", enable=False)
      if y_range is not None:
        p.setYRange(float(y_range[0]), float(y_range[1]), padding=0.0)
      if add_legend:
        leg = p.addLegend(offset=(10, 10))
        leg.setZValue(1000)

      # collect & link
      self._all_plots.append(p)
      if self._x_master is None:
        self._x_master = p
      else:
        p.setXLink(self._x_master)

      return p

    # ========== Row 1: Position (y, x, z) ==========
    p1c1 = _mk_plot(0, 0, "pos_y [mm]", y_range=(-1200., 1200.))
    p1c2 = _mk_plot(0, 1, "pos_x [mm]", y_range=(-1200., 1200.))
    p1c3 = _mk_plot(0, 2, "pos_z [mm]", y_range=(-1200., 1200.))

    self._curves["pos_y_des"] = p1c1.plot(pen=pen_des, name="des")
    self._curves["pos_y_act"] = p1c1.plot(pen=pen_act, name="act")
    self._curves["pos_x_des"] = p1c2.plot(pen=pen_des, name="des")
    self._curves["pos_x_act"] = p1c2.plot(pen=pen_act, name="act")
    self._curves["pos_z_des"] = p1c3.plot(pen=pen_des, name="des")
    self._curves["pos_z_act"] = p1c3.plot(pen=pen_act, name="act")

    # ========== Row 2: RPY ==========
    p2c1 = _mk_plot(1, 0, "roll [deg]", y_range=(-30., 30.))
    p2c2 = _mk_plot(1, 1, "pitch [deg]", y_range=(-30., 30.))
    p2c3 = _mk_plot(1, 2, "yaw [deg]", y_range=(-20., 20.))

    self._curves["roll_raw"]  = p2c1.plot(pen=pen_raw, name="raw")
    self._curves["roll_mrg"]  = p2c1.plot(pen=pen_des, name="MRG ref")
    self._curves["roll_act"]  = p2c1.plot(pen=pen_act, name="act")

    self._curves["pitch_raw"] = p2c2.plot(pen=pen_raw, name="raw")
    self._curves["pitch_mrg"] = p2c2.plot(pen=pen_des, name="MRG ref")
    self._curves["pitch_act"] = p2c2.plot(pen=pen_act, name="act")

    self._curves["yaw_raw"]   = p2c3.plot(pen=pen_raw, name="raw")
    self._curves["yaw_mrg"]   = p2c3.plot(pen=pen_des, name="MRG ref")
    self._curves["yaw_act"]   = p2c3.plot(pen=pen_act, name="act")

    # ========== Row 3: tau ==========
    p3c1 = _mk_plot(2, 0, "tau_x [N·m]", y_range=(-3., 3.))
    p3c2 = _mk_plot(2, 1, "tau_y [N·m]", y_range=(-3., 3.))
    p3c3 = _mk_plot(2, 2, "tau_z [N·m]", y_range=(-1., 1.))

    self._curves["tau_x_total"]  = p3c1.plot(pen=pen_total, name="total")
    self._curves["tau_x_cot"]    = p3c1.plot(pen=pen_cot,   name="cot")
    self._curves["tau_x_thrust"] = p3c1.plot(pen=pen_thr,   name="thrust")

    self._curves["tau_y_total"]  = p3c2.plot(pen=pen_total, name="total")
    self._curves["tau_y_cot"]    = p3c2.plot(pen=pen_cot,   name="cot")
    self._curves["tau_y_thrust"] = p3c2.plot(pen=pen_thr,   name="thrust")

    self._curves["tau_z_total"]  = p3c3.plot(pen=pen_act, name="total")

    # ========== Row 4: f1234 / tilt / f_total ==========
    p4c1 = _mk_plot(3, 0, "f_thrust [N]", y_range=(11., 13.))
    p4c2 = _mk_plot(3, 1, "tilt [deg]", y_range=(-30., 30.))
    p4c3 = _mk_plot(3, 2, "f_total [N]", y_range=(0., 80.))

    for i in range(4):
      self._curves[f"F{i+1}"] = p4c1.plot(
        pen=_mk_pen("solid", width=2, color=rotor_colors[i]),
        name=f"F{i+1}"
      )
      self._curves[f"tilt{i+1}"] = p4c2.plot(
        pen=_mk_pen("solid", width=2, color=rotor_colors[i]),
        name=f"tilt{i+1}"
      )

    self._curves["f_total"] = p4c3.plot(pen=pen_act, name="act")

    # ========== Row 5: r_cot / solve_ms / solve_status ==========
    p5c1 = _mk_plot(4, 0, "r_cot [mm]", y_range=(-50., 50.))
    p5c2 = _mk_plot(4, 1, "solve_ms [ms]", y_range=(0., 5.))
    p5c3 = _mk_plot(4, 2, "solve_status", add_legend=True)

    self._curves["rcot_x_cmd"] = p5c1.plot(pen=pen_rcot_x_cmd, name="MRG cmd x")
    self._curves["rcot_x_act"] = p5c1.plot(pen=pen_rcot_x_act, name="act x")

    self._curves["rcot_y_cmd"] = p5c1.plot(pen=pen_rcot_y_cmd, name="MRG cmd y")
    self._curves["rcot_y_act"] = p5c1.plot(pen=pen_rcot_y_act, name="act y")


    self._curves["solve_ms"] = p5c2.plot(pen=pen_act, name="solve_ms")

    # status plot: fixed y ticks + scatter squares
    self._status_plot = p5c3
    self._status_plot.setLabel("left", "status")
    self._status_plot.setYRange(-1.1, 4.5, padding=0.05)
    self._status_plot.getAxis("left").setTicks([[(i, str(i)) for i in range(5)]])

    leg = self._status_plot.legend

    for s in range(5):
      c = self._status_colors[s]
      dummy = pg.PlotDataItem(
        [np.nan], [np.nan],
        pen=None,
        symbol="s",
        symbolSize=10,
        symbolBrush=pg.mkBrush(*c),
        symbolPen=pg.mkPen(c[0], c[1], c[2], 255),
      )
      leg.addItem(dummy, f"{s}: {self._status_names[s]}")

  @QtCore.pyqtSlot()
  def on_timer(self) -> None:
    try:
      if self.reader.mm is None:
        self.reader.open()
        self.lbl_stat.setText(f"opened (cap={self.reader.header.capacity})")

      t, ch = self.reader.read_all()
      if t.size == 0:
        self.lbl_stat.setText("no data yet")
        return

      tt = (t - t[0]).astype(np.float64)

      # conversions
      pos_act_mm = _mm(ch["pos"])
      pos_des_mm = _mm(ch["pos_d"])

      rpy_act_deg = _deg(ch["rpy"])
      rpy_raw_deg = _deg(ch["rpy_raw"])
      rpy_d_deg   = _deg(ch["rpy_d"])

      tau_d      = ch["tau_d"]
      tau_off    = ch["tau_off"]
      tau_thrust = ch["tau_thrust"]

      f_thrust = ch["f_thrust"]
      tilt_deg = _deg(ch["tilt"])
      f_total  = ch["f_total"]

      r_cot_cmd_mm = _mm(ch["r_cot_cmd"])
      r_cot_act_mm = _mm(ch["r_cot"])

      solve_ms = ch["solve_ms"]
      solve_status = ch["solve_status"].astype(np.int32)
      wc = int(ch["write_count"])

      # Row 1
      self._curves["pos_y_act"].setData(tt, pos_act_mm[:, 1])
      self._curves["pos_y_des"].setData(tt, pos_des_mm[:, 1])
      self._curves["pos_x_act"].setData(tt, pos_act_mm[:, 0])
      self._curves["pos_x_des"].setData(tt, pos_des_mm[:, 0])
      self._curves["pos_z_act"].setData(tt, pos_act_mm[:, 2])
      self._curves["pos_z_des"].setData(tt, pos_des_mm[:, 2])

      # Row 2
      self._curves["roll_raw"].setData(tt,  rpy_raw_deg[:, 0])
      self._curves["roll_mrg"].setData(tt,  rpy_d_deg[:, 0])
      self._curves["roll_act"].setData(tt,  rpy_act_deg[:, 0])

      self._curves["pitch_raw"].setData(tt, rpy_raw_deg[:, 1])
      self._curves["pitch_mrg"].setData(tt, rpy_d_deg[:, 1])
      self._curves["pitch_act"].setData(tt, rpy_act_deg[:, 1])

      self._curves["yaw_raw"].setData(tt,   rpy_raw_deg[:, 2])
      self._curves["yaw_mrg"].setData(tt,   rpy_d_deg[:, 2])
      self._curves["yaw_act"].setData(tt,   rpy_act_deg[:, 2])

      # Row 3
      self._curves["tau_x_total"].setData(tt,  tau_d[:, 0])
      self._curves["tau_x_cot"].setData(tt,    tau_off[:, 0])
      self._curves["tau_x_thrust"].setData(tt, tau_thrust[:, 0])

      self._curves["tau_y_total"].setData(tt,  tau_d[:, 1])
      self._curves["tau_y_cot"].setData(tt,    tau_off[:, 1])
      self._curves["tau_y_thrust"].setData(tt, tau_thrust[:, 1])

      self._curves["tau_z_total"].setData(tt,  tau_d[:, 2])

      # Row 4
      for i in range(4):
        self._curves[f"F{i+1}"].setData(tt, f_thrust[:, i])
        self._curves[f"tilt{i+1}"].setData(tt, tilt_deg[:, i])
      self._curves["f_total"].setData(tt, f_total)

      # Row 5
      self._curves["rcot_x_cmd"].setData(tt, r_cot_cmd_mm[:, 0])
      self._curves["rcot_y_cmd"].setData(tt, r_cot_cmd_mm[:, 1])
      self._curves["rcot_x_act"].setData(tt, r_cot_act_mm[:, 0])
      self._curves["rcot_y_act"].setData(tt, r_cot_act_mm[:, 1])

      self._curves["solve_ms"].setData(tt, solve_ms)

      # Status bar plot (y = -1 -> status)
      dt = float(np.median(np.diff(tt))) if tt.size >= 2 else 0.01
      x = tt
      y = solve_status.astype(np.int32)

      for s in range(5):
        mask = (y == s)
        xs = x[mask]

        y0 = -1.0
        heights = np.full(xs.shape, float(s + 1), dtype=np.float32)
        width = dt if s == 0 else 6.0 * dt

        if s not in self._status_bars:
          bar = pg.BarGraphItem(
            x=xs,
            y0=y0,
            height=heights,
            width=width,
            brush=pg.mkBrush(*self._status_colors[s]),
            pen=None,
          )
          self._status_plot.addItem(bar)
          self._status_bars[s] = bar
        else:
          self._status_bars[s].setOpts(
            x=xs,
            y0=y0,
            height=heights,
            width=width,
            brush=pg.mkBrush(*self._status_colors[s]),
            pen=None,
          )


      self.lbl_stat.setText(
        f"wc={wc} | samples={tt.size} | last solve_ms={float(solve_ms[-1]):.3f} | last status={int(solve_status[-1])}"
      )

    except FileNotFoundError:
      self.lbl_stat.setText(f"waiting: {self.reader.path}")
    except Exception as e:
      self.lbl_stat.setText(f"error: {e}")


def main():
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("--path", default="/tmp/strider_log.mmap")
  ap.add_argument("--update_ms", type=int, default=100)
  args = ap.parse_args()

  app = QtWidgets.QApplication([])

  app.setStyleSheet("""
    QWidget { background: #ffffff; color: #111111; }
    QMainWindow { background: #ffffff; }
  """)

  win = LoggerWindow(path=args.path, update_ms=args.update_ms)
  win.resize(1600, 1200)
  win.show()
  app.exec_()


if __name__ == "__main__":
  main()