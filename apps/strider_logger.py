import os
import mmap
import struct
import shutil
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

# -----------------------------
# Must match C++ mmap_manager.hpp
# -----------------------------
HEADER_SIZE = 64
LOGDATA_SIZE = 493  # sizeof(LogData) with #pragma pack(1)
_SLOT_PAD = (8 - (LOGDATA_SIZE % 8)) % 8
SLOT_SIZE = 8 + LOGDATA_SIZE + _SLOT_PAD  # seq(u64)=8 + LogData + pad -> multiple of 8

MAGIC = b"STRLOG2\x00"
VERSION = 2

# LogData offsets (packed, little-endian) float32 = 4 bytes, int32 = 4 bytes, uint8 = 1 byte
OFF_T          = 0    # float t
OFF_POS_D      = 4    # float pos_d[3]
OFF_VEL_D      = 16   # float vel_d[3]
OFF_ACC_D      = 28   # float acc_d[3]
OFF_POS        = 40   # float pos[3]
OFF_VEL        = 52   # float vel[3]
OFF_ACC        = 64   # float acc[3]
OFF_RPY_RAW    = 76   # float rpy_raw[3]
OFF_RPY_D      = 88   # float rpy_d[3]
OFF_OMEGA_D    = 100  # float omega_d[3]
OFF_ALPHA_D    = 112  # float alpha_d[3]
OFF_RPY        = 124  # float rpy[3]
OFF_OMEGA      = 136  # float omega[3]
OFF_ALPHA      = 148  # float alpha[3]
OFF_F_TOTAL    = 160  # float f_total
OFF_TAU_D      = 164  # float tau_d[3]
OFF_TAU_Z_T    = 176  # float tau_z_t
OFF_TILT       = 180  # float tilt_rad[4]
OFF_F_THRST    = 196  # float f_thrst[4]
OFF_F_THRST_CON= 212  # float f_thrst_con[4]
OFF_TAU_OFF    = 228  # float tau_off[2]
OFF_TAU_THRUST = 236  # float tau_thrust[2]
OFF_R_ROTOR1   = 244  # float r_rotor1[2]
OFF_R_ROTOR2   = 252  # float r_rotor2[2]
OFF_R_ROTOR3   = 260  # float r_rotor3[2]
OFF_R_ROTOR4   = 268  # float r_rotor4[2]
OFF_R_COT      = 276  # float r_cot[2]
OFF_R_ROTOR1_D = 284  # float r_rotor1_d[2]
OFF_R_ROTOR2_D = 292  # float r_rotor2_d[2]
OFF_R_ROTOR3_D = 300  # float r_rotor3_d[2]
OFF_R_ROTOR4_D = 308  # float r_rotor4_d[2]
OFF_R_COT_D    = 316  # float r_cot_d[2]
OFF_Q          = 324  # float q[20]
OFF_Q_CMD      = 404  # float q_cmd[20]
OFF_SOLVE_MS   = 484  # float solve_ms
OFF_SOLVE_STATUS = 488 # int32 solve_status
OFF_PHASE      = 492  # uint8 phase
LOGDATA_SIZE   = 493

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
    if len(buf) < HEADER_SIZE: raise ValueError("Header buffer too small")

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
    if self.mm is not None: return

    self.fd = os.open(self.path, os.O_RDONLY)
    st = os.fstat(self.fd)
    self.mm = mmap.mmap(self.fd, st.st_size, access=mmap.ACCESS_READ)

    self.header = Header.parse(self.mm[0:HEADER_SIZE])

    if self.header.magic != MAGIC: raise RuntimeError(f"Bad magic: {self.header.magic}")
    if self.header.version != VERSION: raise RuntimeError(f"Unsupported version: {self.header.version}")
    if self.header.header_size != HEADER_SIZE: raise RuntimeError(f"Header size mismatch: {self.header.header_size}")
    if self.header.slot_size != SLOT_SIZE: raise RuntimeError(f"Slot size mismatch: {self.header.slot_size} (expected {SLOT_SIZE})")

  def close(self) -> None:
    if self.mm is not None:
      self.mm.close()
      self.mm = None
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None

  def _u64(self, offset: int) -> int:
    return struct.unpack_from("<Q", self.mm, offset)[0]

  def write_count(self) -> int:
    # write_count is at header offset 24
    return int(self._u64(24))

  def capacity(self) -> int:
    return int(self.header.capacity)

  def _read_one_logical(self, logical: int,
                        out_t: np.ndarray,
                        out_ch: Dict[str, np.ndarray],
                        i: int) -> None:
    cap = self.header.capacity
    base = HEADER_SIZE

    idx = logical % cap
    slot_off = base + idx * SLOT_SIZE

    # Default fill (if seqlock fails)
    out_t[i] = np.nan
    out_ch["pos_d"][i, :] = np.nan
    out_ch["pos"][i, :] = np.nan
    out_ch["rpy"][i, :] = np.nan
    out_ch["rpy_raw"][i, :] = np.nan
    out_ch["rpy_d"][i, :] = np.nan
    out_ch["tau_d"][i, :] = np.nan
    out_ch["tau_off"][i, :] = np.nan
    out_ch["tau_thrust"][i, :] = np.nan
    out_ch["tilt"][i, :] = np.nan
    out_ch["f_thrst"][i, :] = np.nan
    out_ch["f_total"][i] = np.nan
    out_ch["r_cot"][i, :] = np.nan
    out_ch["r_cot_d"][i, :] = np.nan
    out_ch["solve_ms"][i] = np.nan
    out_ch["solve_status"][i] = -1

    for _ in range(10):
      seq_a = self._u64(slot_off + 0)
      if seq_a & 1: continue

      dbuf = self.mm[slot_off + 8: slot_off + 8 + LOGDATA_SIZE]  # bytes slice (packed LogData)

      seq_b = self._u64(slot_off + 0)
      if seq_a != seq_b or (seq_b & 1): continue

      out_t[i] = struct.unpack_from("<f", dbuf, OFF_T)[0]
      out_ch["pos_d"][i, :] = struct.unpack_from("<fff", dbuf, OFF_POS_D)
      out_ch["pos"][i, :] = struct.unpack_from("<fff", dbuf, OFF_POS)

      out_ch["rpy"][i, :] = struct.unpack_from("<fff", dbuf, OFF_RPY)
      out_ch["rpy_raw"][i, :] = struct.unpack_from("<fff", dbuf, OFF_RPY_RAW)
      out_ch["rpy_d"][i, :] = struct.unpack_from("<fff", dbuf, OFF_RPY_D)

      out_ch["tau_d"][i, :] = struct.unpack_from("<fff", dbuf, OFF_TAU_D)
      out_ch["tau_off"][i, :] = struct.unpack_from("<ff", dbuf, OFF_TAU_OFF)
      out_ch["tau_thrust"][i, :] = struct.unpack_from("<ff", dbuf, OFF_TAU_THRUST)

      out_ch["tilt"][i, :] = struct.unpack_from("<ffff", dbuf, OFF_TILT)
      out_ch["f_thrst"][i, :] = struct.unpack_from("<ffff", dbuf, OFF_F_THRST)
      out_ch["f_total"][i] = struct.unpack_from("<f", dbuf, OFF_F_TOTAL)[0]

      out_ch["r_cot"][i, :] = struct.unpack_from("<ff", dbuf, OFF_R_COT)
      out_ch["r_cot_d"][i, :] = struct.unpack_from("<ff", dbuf, OFF_R_COT_D)

      out_ch["solve_ms"][i] = struct.unpack_from("<f", dbuf, OFF_SOLVE_MS)[0]
      out_ch["solve_status"][i] = struct.unpack_from("<i", dbuf, OFF_SOLVE_STATUS)[0]
      return

  def read_range(self, wc_from: int, wc_to: int) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, int]:
    """
    Read samples in [wc_from, wc_to) by logical write_count index.
    Returns: t, ch, dropped, effective_wc_from
    dropped: how many samples were lost due to ring overwrite.
    """
    cap = int(self.header.capacity)
    wc_from = int(wc_from)
    wc_to = int(wc_to)

    if wc_to <= wc_from:
      t = np.zeros((0,), dtype=np.float32)
      ch: Dict[str, np.ndarray] = {}
      return t, ch, 0, wc_from

    n_req = wc_to - wc_from
    dropped = 0
    eff_from = wc_from

    # If requester is behind more than cap, older samples already overwritten.
    if n_req > cap:
      dropped = n_req - cap
      eff_from = wc_to - cap

    n = wc_to - eff_from
    if n <= 0:
      t = np.zeros((0,), dtype=np.float32)
      ch = {}
      return t, ch, dropped, eff_from

    t = np.empty((n,), dtype=np.float32)
    ch = {
      "pos_d": np.empty((n, 3), dtype=np.float32),
      "pos": np.empty((n, 3), dtype=np.float32),
      "rpy": np.empty((n, 3), dtype=np.float32),
      "rpy_raw": np.empty((n, 3), dtype=np.float32),
      "rpy_d": np.empty((n, 3), dtype=np.float32),
      "tau_d": np.empty((n, 3), dtype=np.float32),
      "tau_off": np.empty((n, 2), dtype=np.float32),
      "tau_thrust": np.empty((n, 2), dtype=np.float32),
      "tilt": np.empty((n, 4), dtype=np.float32),
      "f_thrst": np.empty((n, 4), dtype=np.float32),
      "f_total": np.empty((n,), dtype=np.float32),
      "r_cot": np.empty((n, 2), dtype=np.float32),
      "r_cot_d": np.empty((n, 2), dtype=np.float32),
      "solve_ms": np.empty((n,), dtype=np.float32),
      "solve_status": np.empty((n,), dtype=np.int32),
    }

    for i in range(n):
      logical = eff_from + i
      self._read_one_logical(logical, t, ch, i)

    return t, ch, dropped, eff_from

  def read_all(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cap = self.header.capacity

    wc = self.write_count()
    n = int(min(wc, cap))
    if n <= 0:
      return np.zeros((0,), dtype=np.float32), {}

    start = wc - n
    t, ch, _, _ = self.read_range(start, wc)
    ch["write_count"] = np.int64(wc)
    return t, ch

# -----------------------------
# Recording (persistent logging)
# -----------------------------
class LogRecorder:
  """
  Records all samples during the Python program lifetime by polling the ring buffer.
  Saves to:
    <log_dir>/npz/log_YYYYmmdd_HHMMSS.npz
    <log_dir>/mmap/snapshot_YYYYmmdd_HHMMSS.mmap
  """
  def __init__(self, mmap_path: str, log_dir: Path):
    self.mmap_path = str(mmap_path)
    self.log_dir = Path(log_dir)  # base dir
    self.npz_dir = self.log_dir / "npz"
    self.mmap_dir = self.log_dir / "mmap"

    self.started = False
    self.wall_start: Optional[datetime] = None
    self.wc_start: int = 0
    self.wc_last: int = 0
    self.wc_end: int = 0
    self.dropped_total: int = 0

    # Store blocks to avoid O(n^2) concat during runtime
    self._t_blocks: List[np.ndarray] = []
    self._ch_blocks: Dict[str, List[np.ndarray]] = {}

  def _ensure_keys(self, ch: Dict[str, np.ndarray]) -> None:
    for k in ch.keys():
      if k == "write_count": continue
      if k not in self._ch_blocks: self._ch_blocks[k] = []

  def start(self, reader: MMapReader) -> None:
    if self.started: return
    # Ensure base + subdirs exist
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.npz_dir.mkdir(parents=True, exist_ok=True)
    self.mmap_dir.mkdir(parents=True, exist_ok=True)
    self.wall_start = datetime.now()
    wc_now = reader.write_count()
    self.wc_start = wc_now
    self.wc_last = wc_now
    self.wc_end = wc_now
    self.started = True

  def poll(self, reader: MMapReader) -> None:
    if not self.started: self.start(reader)

    wc_now = reader.write_count()
    self.wc_end = wc_now

    if wc_now <= self.wc_last: return

    t, ch, dropped, eff_from = reader.read_range(self.wc_last, wc_now)
    self.dropped_total += int(dropped)

    if t.size > 0:
      self._t_blocks.append(t.copy())
      self._ensure_keys(ch)
      for k, v in ch.items():
        if k == "write_count": continue
        self._ch_blocks[k].append(v.copy())

    # Move forward: we advance to wc_now (even if ring overwrote older part)
    self.wc_last = wc_now

  def finalize_and_save(self, reader: Optional[MMapReader]) -> Optional[Path]:
    if not self.started: return None

    # One last poll to capture tail
    if reader is not None and reader.mm is not None:
      try: self.poll(reader)
      except Exception: pass

    # Concatenate
    if len(self._t_blocks) == 0:
      t_all = np.zeros((0,), dtype=np.float32)
      ch_all: Dict[str, np.ndarray] = {}
    else:
      t_all = np.concatenate(self._t_blocks, axis=0)
      ch_all = {}
      for k, blocks in self._ch_blocks.items():
        if len(blocks) == 0: continue
        ch_all[k] = np.concatenate(blocks, axis=0)

    ts = (self.wall_start or datetime.now()).strftime("%m%d_%H%M_%S")
    # Ensure dirs exist (in case start() was skipped due to edge cases)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.npz_dir.mkdir(parents=True, exist_ok=True)
    self.mmap_dir.mkdir(parents=True, exist_ok=True)

    npz_path = self.npz_dir / f"{ts}.npz" 

    meta = {
      "mmap_path": self.mmap_path,
      "wall_start_iso": (self.wall_start or datetime.now()).isoformat(timespec="seconds"),
      "wc_start": np.int64(self.wc_start),
      "wc_end": np.int64(self.wc_end),
      "dropped_total": np.int64(self.dropped_total),
      "n_samples": np.int64(int(t_all.size)),
    }

    # Save as .npz (compressed)
    np.savez_compressed(npz_path, t=t_all, **ch_all, **meta)

    # Also snapshot the mmap file for reference (best-effort)
    snap_path = self.mmap_dir / f"{ts}.mmap"
    try:
      if os.path.exists(self.mmap_path):
        shutil.copy2(self.mmap_path, snap_path)
    except Exception:
      pass

    return npz_path


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
  if style == "solid": return pg.mkPen(color=color, width=width)
  if style == "dash": return pg.mkPen(color=color, width=width, style=QtCore.Qt.DashLine)
  if style == "dot": return pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
  if style == "dashdot": return pg.mkPen(color=color, width=width, style=QtCore.Qt.DashDotLine)
  return pg.mkPen(color=color, width=width)


class LoggerWindow(QtWidgets.QMainWindow):
  def __init__(self,
               live_mmap_path: str = "/tmp/strider_log.mmap",
               update_ms: int = 100,
               replay_path: Optional[str] = None,
               log_dir: Optional[Path] = None):
    super().__init__()
    self.setWindowTitle("STRIDER Logger (mmap)")

    self.live_mmap_path = str(live_mmap_path)
    self.update_ms = int(update_ms)
    self.replay_path = replay_path

    self.reader = MMapReader(self.live_mmap_path)

    self._curves: Dict[str, pg.PlotDataItem] = {}

    # plot linking
    self._all_plots: List[pg.PlotItem] = []
    self._x_master: Optional[pg.PlotItem] = None

    # status plot
    self._status_plot: Optional[pg.PlotItem] = None
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

    # recorder (live only)
    if log_dir is None:
      base_dir = Path(__file__).resolve().parent
      log_dir = base_dir / "log"
    self.log_dir = Path(log_dir)
    self.recorder: Optional[LogRecorder] = None
    if self.replay_path is None:
      self.recorder = LogRecorder(self.live_mmap_path, self.log_dir)

    self._init_ui()

    # Timer only for live mode
    self.timer: Optional[QtCore.QTimer] = None
    if self.replay_path is None:
      self.timer = QtCore.QTimer(self)
      self.timer.timeout.connect(self.on_timer)
      self.timer.start(self.update_ms)
    else:
      self._load_replay(self.replay_path)

  def closeEvent(self, event) -> None:
    # Save recording on exit (live mode)
    if self.recorder is not None:
      try:
        out = self.recorder.finalize_and_save(self.reader if self.reader.mm is not None else None)
        if out is not None:
          self.lbl_stat.setText(f"saved: {str(out)}")
      except Exception as e:
        self.lbl_stat.setText(f"save error: {e}")

    # Close mmap
    try: self.reader.close()
    except Exception: pass

    super().closeEvent(event)

  def _init_ui(self) -> None:
    cw = QtWidgets.QWidget()
    self.setCentralWidget(cw)
    layout = QtWidgets.QVBoxLayout(cw)

    top = QtWidgets.QHBoxLayout()
    self.lbl_path = QtWidgets.QLabel("path: " + (self.replay_path if self.replay_path else self.reader.path))
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
      if self._x_master is None: self._x_master = p
      else: p.setXLink(self._x_master)

      return p

    # ========== Row 1: Position (y, x, z) ==========
    p1c1 = _mk_plot(0, 0, "pos_y [m]", y_range=(-2.5, 2.5))
    p1c2 = _mk_plot(0, 1, "pos_x [m]", y_range=(-2.5, 2.5))
    p1c3 = _mk_plot(0, 2, "pos_z [m]", y_range=(-1.5, 0.5))

    self._curves["pos_y_des"] = p1c1.plot(pen=pen_des, name="des")
    self._curves["pos_y_act"] = p1c1.plot(pen=pen_act, name="act")
    self._curves["pos_x_des"] = p1c2.plot(pen=pen_des, name="des")
    self._curves["pos_x_act"] = p1c2.plot(pen=pen_act, name="act")
    self._curves["pos_z_des"] = p1c3.plot(pen=pen_des, name="des")
    self._curves["pos_z_act"] = p1c3.plot(pen=pen_act, name="act")

    # ========== Row 2: RPY ==========
    p2c1 = _mk_plot(1, 0, "roll [deg]", y_range=(-40., 40.))
    p2c2 = _mk_plot(1, 1, "pitch [deg]", y_range=(-40., 40.))
    p2c3 = _mk_plot(1, 2, "yaw [deg]", y_range=(-10., 10.))

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
    p3c1 = _mk_plot(2, 0, "tau_x [N·m]", y_range=(-1.5, 1.5))
    p3c2 = _mk_plot(2, 1, "tau_y [N·m]", y_range=(-6., 11.))
    p3c3 = _mk_plot(2, 2, "tau_z [N·m]", y_range=(-1., 1.))

    self._curves["tau_x_gac"]    = p3c1.plot(pen=pen_total, name="gac")
    self._curves["tau_x_off"]    = p3c1.plot(pen=pen_cot,   name="off-d")
    self._curves["tau_x_thrust"] = p3c1.plot(pen=pen_thr,   name="thrust")

    self._curves["tau_y_gac"]    = p3c2.plot(pen=pen_total, name="gac")
    self._curves["tau_y_off"]    = p3c2.plot(pen=pen_cot,   name="off-d")
    self._curves["tau_y_thrust"] = p3c2.plot(pen=pen_thr,   name="thrust")

    self._curves["tau_z_gac"]    = p3c3.plot(pen=pen_act, name="gac")

    # ========== Row 4: f1234 / tilt / f_total ==========
    p4c1 = _mk_plot(3, 0, "f_thrst [N]", y_range=(10. , 30.))
    p4c2 = _mk_plot(3, 1, "tilt [deg]", y_range=(-20., 20.))
    p4c3 = _mk_plot(3, 2, "f_total [N]", y_range=(40., 100.))

    for i in range(4):
      self._curves[f"F{i+1}"] = p4c1.plot(pen=_mk_pen("solid", width=2, color=rotor_colors[i]), name=f"F{i+1}")
      self._curves[f"tilt{i+1}"] = p4c2.plot(pen=_mk_pen("solid", width=2, color=rotor_colors[i]), name=f"tilt{i+1}")

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

    # Status plot: fixed y ticks + legend only
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

  def _update_plots(self, t: np.ndarray, ch: Dict[str, np.ndarray], wc: int) -> None:
    if t.size == 0:
      self.lbl_stat.setText("no data")
      return

    tt = (t - t[0]).astype(np.float64)

    # conversions
    pos_act = ch["pos"]
    pos_des = ch["pos_d"]

    rpy_act_deg = _deg(ch["rpy"])
    rpy_raw_deg = _deg(ch["rpy_raw"])
    rpy_d_deg   = _deg(ch["rpy_d"])

    tau_d      = ch["tau_d"]
    tau_off    = ch["tau_off"]
    tau_thrust = ch["tau_thrust"]

    f_thrst = ch["f_thrst"]
    tilt_deg = _deg(ch["tilt"])
    f_total  = ch["f_total"]

    r_cot_d_mm = _mm(ch["r_cot_d"])
    r_cot_act_mm = _mm(ch["r_cot"])

    solve_ms = ch["solve_ms"]
    solve_status = ch["solve_status"].astype(np.int32)

    # Row 1
    self._curves["pos_y_act"].setData(tt, pos_act[:, 1])
    self._curves["pos_y_des"].setData(tt, pos_des[:, 1])
    self._curves["pos_x_act"].setData(tt, pos_act[:, 0])
    self._curves["pos_x_des"].setData(tt, pos_des[:, 0])
    self._curves["pos_z_act"].setData(tt, pos_act[:, 2])
    self._curves["pos_z_des"].setData(tt, pos_des[:, 2])

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
    self._curves["tau_x_gac"].setData(tt, tau_d[:, 0])
    self._curves["tau_x_off"].setData(tt, tau_off[:, 0])
    self._curves["tau_x_thrust"].setData(tt, tau_thrust[:, 0])

    self._curves["tau_y_gac"].setData(tt, tau_d[:, 1])
    self._curves["tau_y_off"].setData(tt, tau_off[:, 1])
    self._curves["tau_y_thrust"].setData(tt, tau_thrust[:, 1])

    self._curves["tau_z_gac"].setData(tt, tau_d[:, 2])

    # Row 4
    for i in range(4):
      self._curves[f"F{i+1}"].setData(tt, f_thrst[:, i])
      self._curves[f"tilt{i+1}"].setData(tt, tilt_deg[:, i])
    self._curves["f_total"].setData(tt, f_total)

    # Row 5
    self._curves["rcot_x_cmd"].setData(tt, r_cot_d_mm[:, 0])
    self._curves["rcot_y_cmd"].setData(tt, r_cot_d_mm[:, 1])
    self._curves["rcot_x_act"].setData(tt, r_cot_act_mm[:, 0])
    self._curves["rcot_y_act"].setData(tt, r_cot_act_mm[:, 1])

    self._curves["solve_ms"].setData(tt, solve_ms)

    # Status bars: top = status value (y0=-1, height=s+1)
    dt = float(np.median(np.diff(tt))) if tt.size >= 2 else 0.01
    x = tt
    y = solve_status

    for s in range(5):
      mask = (y == s)
      xs = x[mask]
      y0 = -1.0
      heights = np.full(xs.shape, float(s + 1), dtype=np.float32)
      width = dt if s == 0 else 6.0 * dt

      if s not in self._status_bars:
        bar = pg.BarGraphItem(x=xs, y0=y0, height=heights, width=width, brush=pg.mkBrush(*self._status_colors[s]), pen=None,)
        self._status_plot.addItem(bar)
        self._status_bars[s] = bar
      else:
        self._status_bars[s].setOpts(x=xs, y0=y0, height=heights, width=width, brush=pg.mkBrush(*self._status_colors[s]), pen=None,)

    last_ms = float(solve_ms[-1]) if solve_ms.size > 0 else float("nan")
    last_st = int(solve_status[-1]) if solve_status.size > 0 else -1

    extra = ""
    if self.recorder is not None and self.recorder.started: extra = f" | rec_samples={sum(b.size for b in self.recorder._t_blocks)} | dropped={self.recorder.dropped_total}"

    self.lbl_stat.setText(f"wc={int(wc)} | samples={tt.size} | last solve_ms={last_ms:.3f} | last status={last_st}{extra}")

  def _load_replay(self, replay_path: str) -> None:
    rp = Path(replay_path)
    self.lbl_path.setText("path: " + str(rp))

    try:
      if rp.suffix.lower() == ".npz":
        data = np.load(str(rp), allow_pickle=False)
        t = data["t"].astype(np.float32)

        ch = {
          "pos_d": data["pos_d"].astype(np.float32),
          "pos": data["pos"].astype(np.float32),
          "rpy": data["rpy"].astype(np.float32),
          "rpy_raw": data["rpy_raw"].astype(np.float32),
          "rpy_d": data["rpy_d"].astype(np.float32),
          "tau_d": data["tau_d"].astype(np.float32),
          "tau_off": data["tau_off"].astype(np.float32),
          "tau_thrust": data["tau_thrust"].astype(np.float32),
          "tilt": data["tilt"].astype(np.float32),
          "f_thrst": data["f_thrst"].astype(np.float32),
          "f_total": data["f_total"].astype(np.float32),
          "r_cot": data["r_cot"].astype(np.float32),
          "r_cot_d": data["r_cot_d"].astype(np.float32),
          "solve_ms": data["solve_ms"].astype(np.float32),
          "solve_status": data["solve_status"].astype(np.int32),
        }
        wc_end = int(data["wc_end"]) if "wc_end" in data else int(t.size)
        self._update_plots(t, ch, wc_end)
        self.lbl_stat.setText(f"replay loaded: {rp.name} | samples={int(t.size)}")

      else:
        # Treat as snapshot mmap
        r = MMapReader(str(rp))
        r.open()
        t, ch = r.read_all()
        wc = int(ch.get("write_count", 0))
        r.close()

        # remove write_count from channel dict
        ch.pop("write_count", None)

        self._update_plots(t, ch, wc)
        self.lbl_stat.setText(f"replay mmap loaded: {rp.name} | samples={int(t.size)}")

    except Exception as e: self.lbl_stat.setText(f"replay error: {e}")

  @QtCore.pyqtSlot()
  def on_timer(self) -> None:
    try:
      if self.reader.mm is None:
        self.reader.open()
        self.lbl_stat.setText(f"opened (cap={self.reader.header.capacity})")
        if self.recorder is not None:
          self.recorder.start(self.reader)

      # Recording poll (captures all samples, not just last ring window)
      if self.recorder is not None:
        self.recorder.poll(self.reader)

      # Viewer uses last ring window (fast, bounded)
      t, ch = self.reader.read_all()
      if t.size == 0:
        self.lbl_stat.setText("no data yet")
        return

      wc = int(ch.get("write_count", 0))
      ch.pop("write_count", None)
      self._update_plots(t, ch, wc)

    except FileNotFoundError:
      self.lbl_stat.setText(f"waiting: {self.reader.path}")
    except Exception as e:
      self.lbl_stat.setText(f"error: {e}")


def main():
  import argparse
  ap = argparse.ArgumentParser()

  # Replay mode (viewer-only):
  # - Provide either positional <replay> or --path <replay>.
  # - Supported: .npz recorded log or .mmap snapshot.
  ap.add_argument("replay", nargs="?", default=None, help="Replay file (.npz recorded log or .mmap snapshot). If provided, runs viewer-only.",)
  ap.add_argument("--path", dest="replay_path", default=None, help="Replay file path (.npz recorded log or .mmap snapshot). Same as positional replay.",)

  args = ap.parse_args()

  replay_path = args.replay_path if args.replay_path is not None else args.replay

  app = QtWidgets.QApplication([])

  app.setStyleSheet("""
    QWidget { background: #ffffff; color: #111111; }
    QMainWindow { background: #ffffff; }
  """)

  # Log dir: same directory as this script
  base_dir = Path(__file__).resolve().parent
  log_dir = base_dir / "log"

  win = LoggerWindow(live_mmap_path="/tmp/strider_log.mmap", update_ms=100, replay_path=replay_path, log_dir=log_dir)
  win.resize(1600, 1200)
  win.show()
  app.exec_()


if __name__ == "__main__":
  main()
