import os
import mmap
import struct
import math
import shutil
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from pathlib import Path
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
import pyqtgraph as pg

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

class InteractiveViewBox(pg.ViewBox):
  """
  Emits user-interaction state for pan/zoom operations.
  - Drag start/finish is detected via mouseDragEvent (ev.isStart/isFinish).
  - Wheel zoom has no finish event, so we use a short singleShot timer.
  """
  sigUserInteracting = QtCore.pyqtSignal(bool)  # True: start, False: end

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._wheel_end = QtCore.QTimer(self)
    self._wheel_end.setSingleShot(True)
    self._wheel_end.timeout.connect(self._emit_end)

  def _emit_end(self):
    self.sigUserInteracting.emit(False)

  def mouseDragEvent(self, ev, axis=None):
    if ev.isStart():
      self.sigUserInteracting.emit(True)
    super().mouseDragEvent(ev, axis=axis)
    if ev.isFinish():
      self.sigUserInteracting.emit(False)

  def wheelEvent(self, ev, axis=None):
    # Wheel zoom: treat as interaction, end after a short quiet period.
    self.sigUserInteracting.emit(True)
    self._wheel_end.start(200)  # ms
    super().wheelEvent(ev, axis=axis)

# -----------------------------
# Manual range slider (no QSS, no superqt)
# -----------------------------
class ManualRangeSlider(QtWidgets.QWidget):
  """
  A minimal two-handle range selector implemented manually.
  - Drag left/right handle to resize selection.
  - Drag middle (selection body) to move selection.
  - Emits valueChanged((lo, hi)) with integer values.
  """
  valueChanged = QtCore.pyqtSignal(tuple)  # (lo, hi)

  def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
    super().__init__(parent)
    self._ori = orientation
    self._min = 0
    self._max = 1000
    self._lo = 200
    self._hi = 800

    self._min_span = 1  # in slider units (ms)
    self._drag_mode = None  # "left", "right", "mid", None
    self._press_x = 0
    self._press_lo = 0
    self._press_hi = 0

    # Visual params (pixels)
    self._pad = 10
    self._track_h = 10
    self._handle_w = 12
    self._handle_h = 18
    self._hit = 8  # extra hit slop around handles

    self.setMouseTracking(True)
    self.setMinimumHeight(26)

    # Colors (no stylesheet)
    self._c_bg = QColor("#ffffff")
    self._c_track = QColor("#d9d9d9")
    self._c_sel = QColor("#1677ff")
    self._c_handle = QColor("#ffffff")
    self._c_handle_border = QColor("#666666")
    self._c_border = QColor("#cfcfcf")

  def sizeHint(self):
    return QtCore.QSize(400, 26)

  def setRange(self, a: int, b: int) -> None:
    a = int(a)
    b = int(b)
    if b < a: a, b = b, a
    self._min = a
    self._max = b
    # Clamp current value
    self._lo = max(self._min, min(self._lo, self._max))
    self._hi = max(self._min, min(self._hi, self._max))
    if self._hi - self._lo < self._min_span:
      self._hi = min(self._max, self._lo + self._min_span)
      if self._hi - self._lo < self._min_span:
        self._lo = max(self._min, self._hi - self._min_span)
    self.update()

  def setValue(self, vals) -> None:
    try:
      lo = int(vals[0])
      hi = int(vals[1])
    except Exception:
      return
    if hi < lo: lo, hi = hi, lo
    lo = max(self._min, min(lo, self._max))
    hi = max(self._min, min(hi, self._max))
    if hi - lo < self._min_span:
      # Keep center, expand minimally
      mid = 0.5 * (lo + hi)
      lo = int(math.floor(mid - 0.5 * self._min_span))
      hi = lo + self._min_span
      lo = max(self._min, lo)
      hi = min(self._max, hi)
      if hi - lo < self._min_span:
        lo = max(self._min, hi - self._min_span)

    changed = (lo != self._lo) or (hi != self._hi)
    self._lo, self._hi = lo, hi
    self.update()
    if changed:
      self.valueChanged.emit((int(self._lo), int(self._hi)))

  def value(self) -> Tuple[int, int]:
    return (int(self._lo), int(self._hi))

  def _w(self) -> int:
    return max(1, int(self.width() - 2 * self._pad))

  def _x_from_val(self, v: int) -> int:
    if self._max <= self._min: return self._pad
    t = (float(v) - float(self._min)) / (float(self._max) - float(self._min))
    t = max(0.0, min(1.0, t))
    return int(self._pad + t * self._w())

  def _val_from_x(self, x: int) -> int:
    x = int(x)
    x0 = self._pad
    x1 = self._pad + self._w()
    if x1 <= x0: return int(self._min)
    x = max(x0, min(x, x1))
    t = (float(x - x0) / float(x1 - x0))
    v = float(self._min) + t * float(self._max - self._min)
    return int(round(v))

  def _geom(self):
    # Returns (track_rect, sel_rect, left_handle_rect, right_handle_rect)
    h = int(self.height())
    cx = int(h // 2)

    track_y = int(cx - self._track_h // 2)
    track = QtCore.QRect(self._pad, track_y, self._w(), self._track_h)

    x_lo = self._x_from_val(self._lo)
    x_hi = self._x_from_val(self._hi)
    if x_hi < x_lo: x_lo, x_hi = x_hi, x_lo

    sel = QtCore.QRect(x_lo, track_y, max(1, x_hi - x_lo), self._track_h)

    hh = self._handle_h
    hy = int(cx - hh // 2)
    l_handle = QtCore.QRect(int(x_lo - self._handle_w // 2), hy, self._handle_w, hh)
    r_handle = QtCore.QRect(int(x_hi - self._handle_w // 2), hy, self._handle_w, hh)
    return track, sel, l_handle, r_handle

  def paintEvent(self, ev):
    p = QPainter(self)
    p.setRenderHint(QPainter.Antialiasing, True)

    # Background
    p.fillRect(self.rect(), self._c_bg)

    # Outer border (subtle)
    p.setPen(QPen(self._c_border, 1))
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

    track, sel, l_handle, r_handle = self._geom()

    # Track
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QBrush(self._c_track))
    p.drawRoundedRect(track, 5, 5)

    # Selection
    p.setBrush(QBrush(self._c_sel))
    p.drawRoundedRect(sel, 5, 5)

    # Handles
    p.setPen(QPen(self._c_handle_border, 1))
    p.setBrush(QBrush(self._c_handle))
    p.drawRoundedRect(l_handle, 6, 6)
    p.drawRoundedRect(r_handle, 6, 6)

    p.end()

  def _hit_test(self, x: int) -> str:
    track, sel, l_handle, r_handle = self._geom()

    # Expand handle rects for easier hit
    lh = l_handle.adjusted(-self._hit, -self._hit, self._hit, self._hit)
    rh = r_handle.adjusted(-self._hit, -self._hit, self._hit, self._hit)

    pt = QtCore.QPoint(int(x), int(self.height() // 2))
    if lh.contains(pt): return "left"
    if rh.contains(pt): return "right"

    # Middle drag: inside selection body (expanded vertically)
    sel2 = sel.adjusted(0, -8, 0, 8)
    if sel2.contains(pt): return "mid"

    return "none"

  def mousePressEvent(self, ev):
    if ev.button() != QtCore.Qt.LeftButton:
      super().mousePressEvent(ev)
      return

    x = int(ev.pos().x())
    mode = self._hit_test(x)
    self._drag_mode = mode if mode != "none" else None

    self._press_x = x
    self._press_lo = int(self._lo)
    self._press_hi = int(self._hi)

    # Optional behavior: click outside selection -> move nearest edge
    if self._drag_mode is None:
      v = self._val_from_x(x)
      # Choose which handle is closer
      if abs(v - self._lo) <= abs(v - self._hi):
        self._drag_mode = "left"
      else:
        self._drag_mode = "right"
      # Apply immediately
      self._apply_drag(x)

    ev.accept()

  def mouseMoveEvent(self, ev):
    if self._drag_mode is not None and (ev.buttons() & QtCore.Qt.LeftButton):
      self._apply_drag(int(ev.pos().x()))
      ev.accept()
      return

    # Update cursor on hover
    x = int(ev.pos().x())
    mode = self._hit_test(x)
    if mode in ("left", "right"):
      self.setCursor(QtCore.Qt.SizeHorCursor)
    elif mode == "mid":
      self.setCursor(QtCore.Qt.OpenHandCursor)
    else:
      self.setCursor(QtCore.Qt.ArrowCursor)

    super().mouseMoveEvent(ev)

  def mouseReleaseEvent(self, ev):
    if ev.button() == QtCore.Qt.LeftButton:
      self._drag_mode = None
      self.setCursor(QtCore.Qt.ArrowCursor)
      ev.accept()
      return
    super().mouseReleaseEvent(ev)

  def _apply_drag(self, x_now: int) -> None:
    if self._drag_mode is None:
      return

    if self._drag_mode == "left":
      v = self._val_from_x(x_now)
      lo = int(v)
      hi = int(self._hi)
      lo = max(self._min, min(lo, hi - self._min_span))
      self.setValue((lo, hi))
      return

    if self._drag_mode == "right":
      v = self._val_from_x(x_now)
      lo = int(self._lo)
      hi = int(v)
      hi = min(self._max, max(hi, lo + self._min_span))
      self.setValue((lo, hi))
      return

    if self._drag_mode == "mid":
      dx = int(x_now - self._press_x)
      span = int(self._press_hi - self._press_lo)

      # If span is 0 or mapping degenerate, fallback to direct x->val shift
      if span <= 0 or self._max <= self._min:
        return

      # Map dx pixels to delta in value space
      # Use total width mapping for stability
      dv = int(round(float(dx) * float(self._max - self._min) / float(self._w())))
      lo = int(self._press_lo + dv)
      hi = int(self._press_hi + dv)

      # Clamp preserving span
      if lo < self._min:
        lo = int(self._min)
        hi = int(lo + span)
      if hi > self._max:
        hi = int(self._max)
        lo = int(hi - span)
      lo = max(self._min, lo)
      hi = min(self._max, hi)
      if hi - lo < self._min_span:
        hi = min(self._max, lo + self._min_span)
      self.setValue((lo, hi))
      return

# -----------------------------
# Fast unpack structs (avoid reparsing format strings)
# -----------------------------
_S_F32  = struct.Struct("<f")
_S_FF   = struct.Struct("<ff")
_S_FFF  = struct.Struct("<fff")
_S_FFFF = struct.Struct("<ffff")
_S_F20  = struct.Struct("<" + ("f" * 20))
_S_I32  = struct.Struct("<i")

_RAD2DEG = np.float32(180.0 / np.pi)
_M2MM    = np.float32(1000.0)

# -----------------------------
# Must match C++ mmap_manager.hpp
# -----------------------------
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
OFF_F_DES      = 160  # float f_total
OFF_TAU_D      = 164  # float tau_d[3]
OFF_TAU_Z_T    = 176  # float tau_z_t
OFF_TILT       = 180  # float tilt_rad[4]
OFF_F_THRST    = 196  # float f_thrst[4]
OFF_F_THRST_CON= 212  # float f_thrst_con[4]
OFF_TAU_OFF    = 228  # float tau_off[2]
OFF_TAU_THRUST = 236  # float tau_thrust[3]
OFF_R_ROTOR1   = 248  # float r_rotor1[2]
OFF_R_ROTOR2   = 256  # float r_rotor2[2]
OFF_R_ROTOR3   = 264  # float r_rotor3[2]
OFF_R_ROTOR4   = 272  # float r_rotor4[2]
OFF_R_COT      = 280  # float r_cot[2]
OFF_R_ROTOR1_D = 288  # float r_rotor1_d[2]
OFF_R_ROTOR2_D = 296  # float r_rotor2_d[2]
OFF_R_ROTOR3_D = 304  # float r_rotor3_d[2]
OFF_R_ROTOR4_D = 312  # float r_rotor4_d[2]
OFF_R_COT_D    = 320  # float r_cot_d[2]
OFF_Q          = 328  # float q[20]
OFF_Q_CMD      = 408  # float q_cmd[20]
OFF_SOLVE_MS   = 488  # float solve_ms
OFF_SOLVE_STATUS = 492 # int32 solve_status
OFF_PHASE      = 496  # uint8 phase

LOGDATA_SIZE = 497  # sizeof(LogData) with #pragma pack(1)
HEADER_SIZE = 64
_SLOT_PAD = (8 - (LOGDATA_SIZE % 8)) % 8
SLOT_SIZE = 8 + LOGDATA_SIZE + _SLOT_PAD  # seq(u64)=8 + LogData + pad -> multiple of 8

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
    self._mv: Optional[memoryview] = None  # zero-copy view into mmap
    self._inode: Optional[int] = None
    self._size: Optional[int] = None

  def open(self) -> None:
    if self.mm is not None: return

    self.fd = os.open(self.path, os.O_RDONLY)
    st = os.fstat(self.fd)
    self._inode = int(st.st_ino)
    self._size = int(st.st_size)
    self.mm = mmap.mmap(self.fd, st.st_size, access=mmap.ACCESS_READ)
    self._mv = memoryview(self.mm)

    self.header = Header.parse(self.mm[0:HEADER_SIZE])

    if self.header.magic != MAGIC: raise RuntimeError(f"Bad magic: {self.header.magic}")
    if self.header.version != VERSION: raise RuntimeError(f"Unsupported version: {self.header.version}")
    if self.header.header_size != HEADER_SIZE: raise RuntimeError(f"Header size mismatch: {self.header.header_size}")
    if self.header.slot_size != SLOT_SIZE: raise RuntimeError(f"Slot size mismatch: {self.header.slot_size} (expected {SLOT_SIZE})")

  def close(self) -> None:
    if self.mm is not None:
      self.mm.close()
      self.mm = None
    self._mv = None
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None
    self._inode = None
    self._size = None
    self.header = None

  def changed_on_disk(self) -> bool:
    """
    Returns True if the file at self.path was replaced (inode changed),
    truncated/expanded (size changed), or removed.
    """
    if self.mm is None: return False
    try: st = os.stat(self.path)
    except FileNotFoundError: return True
    except OSError: return True
    if self._inode is None or self._size is None: return True
    return (int(st.st_ino) != int(self._inode)) or (int(st.st_size) != int(self._size))

  def _u64(self, offset: int) -> int:
    return struct.unpack_from("<Q", self.mm, offset)[0]

  def write_count(self) -> int:
    # write_count is at header offset 24
    return int(self._u64(24))

  def capacity(self) -> int:
    return int(self.header.capacity)

  def _read_one_logical(self, logical: int, out_t: np.ndarray, out_ch: Dict[str, np.ndarray], i: int) -> None:
    cap = self.header.capacity
    base = HEADER_SIZE

    idx = logical % cap
    slot_off = base + idx * SLOT_SIZE

    for _ in range(10):
      seq_a = self._u64(slot_off + 0)
      if seq_a & 1: continue

      # NOTE: zero-copy view (avoid allocating/copying 497 bytes per slot)
      dbuf = self._mv[slot_off + 8: slot_off + 8 + LOGDATA_SIZE]

      seq_b = self._u64(slot_off + 0)
      if seq_a != seq_b or (seq_b & 1): continue

      out_t[i] = _S_F32.unpack_from(dbuf, OFF_T)[0]
      out_ch["pos_d"][i, :] = _S_FFF.unpack_from(dbuf, OFF_POS_D)
      out_ch["pos"][i, :] = _S_FFF.unpack_from(dbuf, OFF_POS)
      out_ch["vel_d"][i, :] = _S_FFF.unpack_from(dbuf, OFF_VEL_D)
      out_ch["vel"][i, :]   = _S_FFF.unpack_from(dbuf, OFF_VEL)
      out_ch["acc_d"][i, :] = _S_FFF.unpack_from(dbuf, OFF_ACC_D)
      out_ch["acc"][i, :]   = _S_FFF.unpack_from(dbuf, OFF_ACC)

      out_ch["rpy"][i, :] = _S_FFF.unpack_from(dbuf, OFF_RPY)
      out_ch["rpy_raw"][i, :] = _S_FFF.unpack_from(dbuf, OFF_RPY_RAW)
      out_ch["rpy_d"][i, :] = _S_FFF.unpack_from(dbuf, OFF_RPY_D)
      out_ch["omega_d"][i, :] = _S_FFF.unpack_from(dbuf, OFF_OMEGA_D)
      out_ch["omega"][i, :]   = _S_FFF.unpack_from(dbuf, OFF_OMEGA)
      out_ch["alpha_d"][i, :] = _S_FFF.unpack_from(dbuf, OFF_ALPHA_D)
      out_ch["alpha"][i, :]   = _S_FFF.unpack_from(dbuf, OFF_ALPHA)

      out_ch["tau_d"][i, :] = _S_FFF.unpack_from(dbuf, OFF_TAU_D)
      out_ch["tau_z_t"][i]  = _S_F32.unpack_from(dbuf, OFF_TAU_Z_T)[0]
      out_ch["tau_off"][i, :] = _S_FF.unpack_from(dbuf, OFF_TAU_OFF)
      out_ch["tau_thrust"][i, :] = _S_FFF.unpack_from(dbuf, OFF_TAU_THRUST)

      out_ch["tilt"][i, :] = _S_FFFF.unpack_from(dbuf, OFF_TILT)
      out_ch["f_thrst"][i, :] = _S_FFFF.unpack_from(dbuf, OFF_F_THRST)
      out_ch["f_thrst_con"][i, :] = _S_FFFF.unpack_from(dbuf, OFF_F_THRST_CON)
      out_ch["f_des"][i] = _S_F32.unpack_from(dbuf, OFF_F_DES)[0]

      out_ch["r_cot"][i, :] = _S_FF.unpack_from(dbuf, OFF_R_COT)
      out_ch["r_cot_d"][i, :] = _S_FF.unpack_from(dbuf, OFF_R_COT_D)

      out_ch["r_rotor1"][i, :]   = _S_FF.unpack_from(dbuf, OFF_R_ROTOR1)
      out_ch["r_rotor2"][i, :]   = _S_FF.unpack_from(dbuf, OFF_R_ROTOR2)
      out_ch["r_rotor3"][i, :]   = _S_FF.unpack_from(dbuf, OFF_R_ROTOR3)
      out_ch["r_rotor4"][i, :]   = _S_FF.unpack_from(dbuf, OFF_R_ROTOR4)
      out_ch["r_rotor1_d"][i, :] = _S_FF.unpack_from(dbuf, OFF_R_ROTOR1_D)
      out_ch["r_rotor2_d"][i, :] = _S_FF.unpack_from(dbuf, OFF_R_ROTOR2_D)
      out_ch["r_rotor3_d"][i, :] = _S_FF.unpack_from(dbuf, OFF_R_ROTOR3_D)
      out_ch["r_rotor4_d"][i, :] = _S_FF.unpack_from(dbuf, OFF_R_ROTOR4_D)

      # q[20], q_cmd[20] (stored as [arm1(5), arm2(5), arm3(5), arm4(5)])
      out_ch["q"][i, :]     = np.frombuffer(dbuf, dtype="<f4", count=20, offset=OFF_Q)
      out_ch["q_cmd"][i, :] = np.frombuffer(dbuf, dtype="<f4", count=20, offset=OFF_Q_CMD)

      out_ch["solve_ms"][i] = _S_F32.unpack_from(dbuf, OFF_SOLVE_MS)[0]
      out_ch["solve_status"][i] = _S_I32.unpack_from(dbuf, OFF_SOLVE_STATUS)[0]
      
      out_ch["phase"][i] = int(dbuf[OFF_PHASE])
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
      "vel_d": np.empty((n, 3), dtype=np.float32),
      "vel": np.empty((n, 3), dtype=np.float32),
      "acc_d": np.empty((n, 3), dtype=np.float32),
      "acc": np.empty((n, 3), dtype=np.float32),
      "rpy": np.empty((n, 3), dtype=np.float32),
      "rpy_raw": np.empty((n, 3), dtype=np.float32),
      "rpy_d": np.empty((n, 3), dtype=np.float32),
      "omega_d": np.empty((n, 3), dtype=np.float32),
      "omega": np.empty((n, 3), dtype=np.float32),
      "alpha_d": np.empty((n, 3), dtype=np.float32),
      "alpha": np.empty((n, 3), dtype=np.float32),
      "tau_d": np.empty((n, 3), dtype=np.float32),
      "tau_z_t": np.empty((n,), dtype=np.float32),
      "tau_off": np.empty((n, 2), dtype=np.float32),
      "tau_thrust": np.empty((n, 3), dtype=np.float32),
      "tilt": np.empty((n, 4), dtype=np.float32),
      "f_thrst": np.empty((n, 4), dtype=np.float32),
      "f_thrst_con": np.empty((n, 4), dtype=np.float32),
      "f_des": np.empty((n,), dtype=np.float32),
      "f_tot": np.empty((n,), dtype=np.float32),
      "r_cot": np.empty((n, 2), dtype=np.float32),
      "r_cot_d": np.empty((n, 2), dtype=np.float32),
      "r_rotor1": np.empty((n, 2), dtype=np.float32),
      "r_rotor2": np.empty((n, 2), dtype=np.float32),
      "r_rotor3": np.empty((n, 2), dtype=np.float32),
      "r_rotor4": np.empty((n, 2), dtype=np.float32),
      "r_rotor1_d": np.empty((n, 2), dtype=np.float32),
      "r_rotor2_d": np.empty((n, 2), dtype=np.float32),
      "r_rotor3_d": np.empty((n, 2), dtype=np.float32),
      "r_rotor4_d": np.empty((n, 2), dtype=np.float32),
      "q": np.empty((n, 20), dtype=np.float32),
      "q_cmd": np.empty((n, 20), dtype=np.float32),
      "solve_ms": np.empty((n,), dtype=np.float32),
      "solve_status": np.empty((n,), dtype=np.int32),
      "phase": np.empty((n,), dtype=np.uint8),
    }

    # NOTE: fill once per read_range (much cheaper than per-slot default fill)
    t.fill(np.nan)
    for k, v in ch.items():
      if k == "solve_status": v.fill(-1)
      elif k == "phase": v.fill(255)  # 255 = invalid/unknown
      else: v.fill(np.nan)

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
    <log_dir>/npz/mmdd_HHMMSS.npz
    <log_dir>/mmap/mmdd_HHMMSS.mmap
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
    self.n_samples_total: int = 0  # fast counter for UI

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
    self.n_samples_total = 0
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
      self.n_samples_total += int(t.size)
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
  def __init__(self, live_mmap_path: str = "/tmp/strider_log.mmap", update_ms: int = 100, replay_path: Optional[str] = None, log_dir: Optional[Path] = None):
    super().__init__()
    self.setWindowTitle("STRIDER Logger (mmap)")

    self.live_mmap_path = str(live_mmap_path)
    self.update_ms = int(update_ms)
    self.replay_path = replay_path

    self.reader = MMapReader(self.live_mmap_path)
    self._session_start_ns: Optional[int] = None  # header.start_time_ns latch

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

    # phase plot
    self._phase_plot: Optional[pg.PlotItem] = None
    self._phase_bars: Dict[int, pg.BarGraphItem] = {}
    self._phase_plot_t1: Optional[pg.PlotItem] = None
    self._phase_bars_t1: Dict[int, pg.BarGraphItem] = {}
    self._phase_plot_t3: Optional[pg.PlotItem] = None
    self._phase_bars_t3: Dict[int, pg.BarGraphItem] = {}

    # Compact display order (keeps y-range small)
    self._phase_order = [0, 1, 2, 3, 4, 5, 6, 99]
    # code -> (name, description)
    self._phase_info = {
      0: ("READY",          "program started"),
      1: ("ARMED",          "all sanity checked"),
      2: ("IDLE",           "propellers idling"),
      3: ("RISING",         "thrust increasing"),
      4: ("GAC_FLIGHT",     "only geometry control"),
      5: ("MRG_FLIGHT",     "no CoT moving"),
      6: ("MRG_ACTIVE_COT", "yes CoT moving"),
      99: ("KILLED",        "killed"),
    }
    # code -> RGBA (distinct colors)
    self._phase_colors = {
      0: (120, 120, 120, 220),   # gray
      1: (0, 120, 255, 220),     # blue
      2: (0, 200, 0, 220),       # green
      3: (255, 230, 0, 220),     # yellow
      4: (255, 165, 0, 220),     # orange
      5: (255, 0, 0, 220),       # red
      6: (160, 0, 255, 220),     # purple
      99: (0, 0, 0, 220),        # black
    }
    # LUT: uint8 phase -> compact idx (or -1)
    self._phase_lut = np.full((256,), -1, dtype=np.int16)
    for _i, _code in enumerate(self._phase_order): self._phase_lut[int(_code)] = int(_i)

    # Viewer performance knobs (does NOT affect recording)
    self.view_max_points = 6000  # set 0 to disable decimation
    self._last_wc_plotted: int = -1

    # recorder (live only)
    if log_dir is None:
      base_dir = Path(__file__).resolve().parent
      log_dir = base_dir / "log"
    self.log_dir = Path(log_dir)
    self.recorder: Optional[LogRecorder] = None
    if self.replay_path is None:
      self.recorder = LogRecorder(self.live_mmap_path, self.log_dir)

    # Pause heavy UI updates while user is panning/zooming
    self._ui_paused_by_user = False

    # Replay-only range slider state
    self._range_slider: Optional[ManualRangeSlider] = None
    self._range_slider_lbl: Optional[QtWidgets.QLabel] = None
    self._replay_full_t: Optional[np.ndarray] = None
    self._replay_full_ch: Optional[Dict[str, np.ndarray]] = None
    self._replay_full_tt: Optional[np.ndarray] = None  # t - t0 (sec)
    self._replay_time_base: Optional[float] = None     # t0
    self._replay_wc_end: int = 0
    self.glw4_joint: Optional[pg.GraphicsLayoutWidget] = None
    self.glw4_rotor: Optional[pg.GraphicsLayoutWidget] = None
    self._xy_map_plot: Optional[pg.PlotItem] = None
    self._xy_circles: Dict[int, QtWidgets.QGraphicsEllipseItem] = {}
    self._xy_border_items: List[pg.PlotDataItem] = []

    # Prevent feedback loops (slider <-> view range)
    self._block_slider_changed = False
    self._block_view_changed = False
    self._pending_slider_vals: Optional[Tuple[int, int]] = None
    self._pending_view_xrange: Optional[Tuple[float, float]] = None
    self._slider_apply_timer = QtCore.QTimer(self)
    self._slider_apply_timer.setSingleShot(True)
    self._slider_apply_timer.timeout.connect(self._apply_pending_slider_window)
    self._view_apply_timer = QtCore.QTimer(self)
    self._view_apply_timer.setSingleShot(True)
    self._view_apply_timer.timeout.connect(self._apply_pending_view_window)

    self._init_ui()

    # Timer only for live mode
    self.timer: Optional[QtCore.QTimer] = None
    if self.replay_path is None:
      self.timer = QtCore.QTimer(self)
      self.timer.timeout.connect(self.on_timer)
      self.timer.start(self.update_ms)
    else:
      self._load_replay(self.replay_path)

  def _clear_status_bars(self) -> None:
    if self._status_plot is None: return
    for _, bar in list(self._status_bars.items()):
      try: self._status_plot.removeItem(bar)
      except Exception: pass
    self._status_bars.clear()

  def _clear_phase_bars(self) -> None:
    # Clear phase bars in both plots (tab2 + tab3)
    if self._phase_plot_t1 is not None:
      for _, bar in list(self._phase_bars_t1.items()):
        try: self._phase_plot_t1.removeItem(bar)
        except Exception: pass
      self._phase_bars_t1.clear()
    if self._phase_plot is not None:
      for _, bar in list(self._phase_bars.items()):
        try: self._phase_plot.removeItem(bar)
        except Exception: pass
      self._phase_bars.clear()
    if self._phase_plot_t3 is not None:
      for _, bar in list(self._phase_bars_t3.items()):
        try: self._phase_plot_t3.removeItem(bar)
        except Exception: pass
      self._phase_bars_t3.clear()

  def _rotate_recording(self, reason: str) -> None:
    """
    Finish current recording (best-effort) and start a fresh recorder.
    Called when a new flight session is detected (mmap replaced / header reset).
    """
    if self.recorder is None: return
    try:
      out = self.recorder.finalize_and_save(self.reader if self.reader.mm is not None else None)
      if out is not None: self.lbl_stat.setText(f"saved ({reason}): {str(out)}")
    except Exception as e: self.lbl_stat.setText(f"save error ({reason}): {e}")

    # New recorder for next flight (live mode only)
    self.recorder = LogRecorder(self.live_mmap_path, self.log_dir)
    if self.reader.mm is not None:
      try: self.recorder.start(self.reader)
      except Exception: pass
    self._clear_status_bars()
    self._clear_phase_bars()

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

  @QtCore.pyqtSlot(bool)
  def _on_user_interacting(self, active: bool) -> None:
    # NOTE: only affects viewer updates; recording still runs.
    self._ui_paused_by_user = bool(active)

  # -----------------------------
  # Replay-only range slider glue
  # -----------------------------
  def _sec_to_slider(self, x_sec: float) -> int:
    # Use integer milliseconds to keep the slider stable.
    return int(round(float(x_sec) * 1000.0))

  def _slider_to_sec(self, x_ms: int) -> float:
    return float(int(x_ms)) / 1000.0

  def _set_slider_range(self, x0_sec: float, x1_sec: float) -> None:
    if self._range_slider is None: return
    lo = self._sec_to_slider(x0_sec)
    hi = self._sec_to_slider(x1_sec)
    if hi < lo: lo, hi = hi, lo
    self._block_slider_changed = True
    try:
      self._range_slider.setRange(lo, hi)
      self._range_slider.setValue((lo, hi))
    finally:
      self._block_slider_changed = False

  def _sync_slider_from_view_xrange(self, xmin: float, xmax: float) -> None:
    if self._range_slider is None: return
    if self._block_view_changed: return
    lo = self._sec_to_slider(xmin)
    hi = self._sec_to_slider(xmax)
    if hi < lo: lo, hi = hi, lo
    self._block_slider_changed = True
    try: self._range_slider.setValue((lo, hi))
    finally: self._block_slider_changed = False

  def _on_slider_changed(self, vals) -> None:
    # vals is typically a tuple (low_ms, high_ms)
    if self.replay_path is None: return
    if self._range_slider is None: return
    if self._block_slider_changed: return
    try: lo_ms, hi_ms = int(vals[0]), int(vals[1])
    except Exception: return
    if hi_ms < lo_ms: lo_ms, hi_ms = hi_ms, lo_ms
    self._pending_slider_vals = (lo_ms, hi_ms)
    # Throttle to avoid heavy updates while dragging
    self._slider_apply_timer.start(30)

  def _apply_pending_slider_window(self) -> None:
    if self.replay_path is None: return
    if self._pending_slider_vals is None: return
    lo_ms, hi_ms = self._pending_slider_vals
    self._pending_slider_vals = None
    xmin = self._slider_to_sec(lo_ms)
    xmax = self._slider_to_sec(hi_ms)
    self._apply_replay_window(xmin, xmax, source="slider")

  def _on_view_range_changed(self, vb: pg.ViewBox, ranges) -> None:
    # Triggered by pan/zoom; keep slider and plotted data window consistent.
    if self.replay_path is None: return
    if self._range_slider is None: return
    if self._block_view_changed: return
    try:
      # vb.viewRange() -> [[xmin, xmax], [ymin, ymax]]
      xr = vb.viewRange()[0]
      xmin = float(xr[0])
      xmax = float(xr[1])
    except Exception: return

    # Update slider immediately (cheap) and throttle heavy re-slicing.
    self._sync_slider_from_view_xrange(xmin, xmax)
    self._pending_view_xrange = (xmin, xmax)
    self._view_apply_timer.start(30)

  def _apply_pending_view_window(self) -> None:
    if self.replay_path is None: return
    if self._pending_view_xrange is None: return
    xmin, xmax = self._pending_view_xrange
    self._pending_view_xrange = None
    self._apply_replay_window(xmin, xmax, source="view")

  def _apply_replay_window(self, xmin: float, xmax: float, source: str) -> None:
    """
    Replay-only: slice the full replay arrays to [xmin, xmax] (sec, relative to t0),
    update all plots, and keep the view range coherent.
    """
    if self.replay_path is None: return
    if self._replay_full_t is None or self._replay_full_ch is None or self._replay_full_tt is None: return
    if self._x_master is None: return

    # Clamp and sanitize
    if not np.isfinite(xmin) or not np.isfinite(xmax): return
    if xmax < xmin: xmin, xmax = xmax, xmin
    tt_full = self._replay_full_tt
    n_full = int(tt_full.size)
    if n_full <= 0: return
    tmin_full = float(tt_full[0])
    tmax_full = float(tt_full[-1])
    xmin = max(tmin_full, float(xmin))
    xmax = min(tmax_full, float(xmax))
    if xmax <= xmin: xmax = min(tmax_full, xmin + 0.01) # Ensure a non-degenerate window

    # Fast window indices (assumes tt_full is sorted)
    lo = int(np.searchsorted(tt_full, xmin, side="left"))
    hi = int(np.searchsorted(tt_full, xmax, side="right"))
    lo = max(0, min(lo, n_full - 1))
    hi = max(lo + 2, min(hi, n_full))  # keep at least 2 samples

    t_sub = self._replay_full_t[lo:hi]
    ch_sub: Dict[str, np.ndarray] = {}
    for k, v in self._replay_full_ch.items(): ch_sub[k] = v[lo:hi]

    # Update plots with the sliced window
    self._update_plots(t_sub, ch_sub, int(self._replay_wc_end))

    # Keep the visible X range aligned with the selected window
    self._block_view_changed = True
    try:
      self._x_master.setXRange(float(xmin), float(xmax), padding=0.0)
    finally:
      self._block_view_changed = False

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

    # -------------------------
    # Tabs (3 panels) for future plot expansion
    # -------------------------
    self.tabs = QtWidgets.QTabWidget()
    layout.addWidget(self.tabs, 1)

    # Tab 1
    tab1 = QtWidgets.QWidget()
    tab1_lay = QtWidgets.QVBoxLayout(tab1)
    tab1_lay.setContentsMargins(0, 0, 0, 0)
    tab1_lay.setSpacing(0)
    self.glw1 = pg.GraphicsLayoutWidget()
    tab1_lay.addWidget(self.glw1, 1)
    self.tabs.addTab(tab1, "mrg")

    # Tab 2
    tab2 = QtWidgets.QWidget()
    tab2_lay = QtWidgets.QVBoxLayout(tab2)
    tab2_lay.setContentsMargins(0, 0, 0, 0)
    tab2_lay.setSpacing(0)
    self.glw2 = pg.GraphicsLayoutWidget()
    tab2_lay.addWidget(self.glw2, 1)
    self.tabs.addTab(tab2, "pos")

    # Tab 3
    tab3 = QtWidgets.QWidget()
    tab3_lay = QtWidgets.QVBoxLayout(tab3)
    tab3_lay.setContentsMargins(0, 0, 0, 0)
    tab3_lay.setSpacing(0)
    self.glw3 = pg.GraphicsLayoutWidget()
    tab3_lay.addWidget(self.glw3, 1)
    self.tabs.addTab(tab3, "att")

    # Tab 4
    tab4 = QtWidgets.QWidget()
    tab4_lay = QtWidgets.QVBoxLayout(tab4)
    tab4_lay.setContentsMargins(0, 0, 0, 0)
    tab4_lay.setSpacing(0)

    split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    split.setContentsMargins(0, 0, 0, 0)
    tab4_lay.addWidget(split, 1)

    self.glw4_joint = pg.GraphicsLayoutWidget()
    self.glw4_rotor = pg.GraphicsLayoutWidget()
    split.addWidget(self.glw4_joint)
    split.addWidget(self.glw4_rotor)
    try:
      split.setStretchFactor(0, 1)
      split.setStretchFactor(1, 1)
    except Exception:
      pass

    self.tabs.addTab(tab4, "arm")

    # Backward-compat alias: existing code continues to use glw (points to Plot 1)
    self.glw = self.glw1

    # Replay-only: manual range slider (viewer-only mode)
    if self.replay_path is not None:
      slider_wrap = QtWidgets.QWidget()
      slider_lay = QtWidgets.QHBoxLayout(slider_wrap)
      slider_lay.setContentsMargins(0,0,0,0)

      self._range_slider = ManualRangeSlider(QtCore.Qt.Horizontal)
      self._range_slider.valueChanged.connect(self._on_slider_changed)

      slider_lay.addWidget(self._range_slider, 1)
      layout.addWidget(slider_wrap, 0)

    # -------------------------
    # Pens (explicit colors)
    # -------------------------
    pen_act   = _mk_pen("solid", width=2, color="b")                  # blue solid
    pen_mrg   = _mk_pen("dash",  width=2, color="r")                  # red dashed
    pen_raw   = _mk_pen("dot",   width=2, color="k")                  # black dotted
    pen_des   = _mk_pen("dot",   width=2, color="k")                  # black dotted
    pen_total = _mk_pen("solid", width=1, color="k")                  # black solid
    pen_cot   = _mk_pen("solid", width=2, color=(255, 105, 180))      # pink solid
    pen_thr   = _mk_pen("solid", width=2, color="b")                  # blue solid
    pen_react = _mk_pen("solid", width=2, color=(255, 165, 0))        # orange solid
    pen_rcot_x_cmd = _mk_pen("dash",  width=2, color="r")  # red dashed
    pen_rcot_x_act = _mk_pen("solid", width=2, color="r")  # red solid
    pen_rcot_y_cmd = _mk_pen("dash",  width=2, color="b")  # blue dashed
    pen_rcot_y_act = _mk_pen("solid", width=2, color="b")  # blue solid
    rotor_colors = ["b", "g", "m", "c"]
    pen_q_cmd = _mk_pen("dash",  width=2, color="r")
    pen_q_act = _mk_pen("solid", width=2, color="b")
    pen_rx_cmd = _mk_pen("dash",  width=2, color="r")
    pen_rx_act = _mk_pen("solid", width=2, color="r")
    pen_ry_cmd = _mk_pen("dash",  width=2, color="b")
    pen_ry_act = _mk_pen("solid", width=2, color="b")

    # Helper: create plot and link X to master
    def _mk_plot(glw: pg.GraphicsLayoutWidget, row: int, col: int, title: str, add_legend: bool = True, y_range: Optional[Tuple[float, float]] = None) -> pg.PlotItem:
      vb = InteractiveViewBox()
      vb.sigUserInteracting.connect(self._on_user_interacting)
      if self.replay_path is not None: # Replay-only: keep slider synced with pan/zoom on any plot
        try: vb.sigRangeChanged.connect(self._on_view_range_changed)
        except Exception: pass
      p = glw.addPlot(row=row, col=col, title=title, viewBox=vb)
      _style(p)
      p.showLabel("bottom", False)
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

    # ========== 1-Row 1: Position (y, x, z) ==========
    p1c1 = _mk_plot(self.glw1, 0, 0, "pos_y [m]", y_range=(-2.5, 2.5))
    p1c2 = _mk_plot(self.glw1, 0, 1, "pos_x [m]", y_range=(-2.5, 2.5))
    p1c3 = _mk_plot(self.glw1, 0, 2, "pos_z [m]", y_range=(-1.5, 0.5))

    self._curves["pos_y_des"] = p1c1.plot(pen=pen_des, name="des")
    self._curves["pos_y_act"] = p1c1.plot(pen=pen_act, name="act")
    self._curves["pos_x_des"] = p1c2.plot(pen=pen_des, name="des")
    self._curves["pos_x_act"] = p1c2.plot(pen=pen_act, name="act")
    self._curves["pos_z_des"] = p1c3.plot(pen=pen_des, name="des")
    self._curves["pos_z_act"] = p1c3.plot(pen=pen_act, name="act")

    # ========== 1-Row 2: RPY ==========
    p2c1 = _mk_plot(self.glw1, 1, 0, "roll [deg]", y_range=(-40., 40.))
    p2c2 = _mk_plot(self.glw1, 1, 1, "pitch [deg]", y_range=(-40., 40.))
    p2c3 = _mk_plot(self.glw1, 1, 2, "yaw [deg]", y_range=(-10., 10.))

    self._curves["roll_raw"]  = p2c1.plot(pen=pen_raw, name="raw")
    self._curves["roll_mrg"]  = p2c1.plot(pen=pen_mrg, name="mrg")
    self._curves["roll_act"]  = p2c1.plot(pen=pen_act, name="act")

    self._curves["pitch_raw"] = p2c2.plot(pen=pen_raw, name="raw")
    self._curves["pitch_mrg"] = p2c2.plot(pen=pen_mrg, name="mrg")
    self._curves["pitch_act"] = p2c2.plot(pen=pen_act, name="act")

    self._curves["yaw_raw"]   = p2c3.plot(pen=pen_raw, name="raw")
    self._curves["yaw_mrg"]   = p2c3.plot(pen=pen_mrg, name="mrg")
    self._curves["yaw_act"]   = p2c3.plot(pen=pen_act, name="act")

    # ========== 1-Row 3: tau ==========
    p3c1 = _mk_plot(self.glw1, 2, 0, "tau_x [N·m]", y_range=(-1.5, 1.5))
    p3c2 = _mk_plot(self.glw1, 2, 1, "tau_y [N·m]", y_range=(-6., 11.))
    p3c3 = _mk_plot(self.glw1, 2, 2, "tau_z [N·m]", y_range=(-1., 1.))

    self._curves["tau_x_off"]    = p3c1.plot(pen=pen_cot,   name="off-d")
    self._curves["tau_x_thrust"] = p3c1.plot(pen=pen_thr,   name="thrust")
    self._curves["tau_x_total"]  = p3c1.plot(pen=pen_total, name="total")
    self._curves["tau_x_des"]    = p3c1.plot(pen=pen_des,   name="des")

    self._curves["tau_y_off"]    = p3c2.plot(pen=pen_cot,   name="off-d")
    self._curves["tau_y_thrust"] = p3c2.plot(pen=pen_thr,   name="thrust")
    self._curves["tau_y_total"]  = p3c2.plot(pen=pen_total, name="total")
    self._curves["tau_y_des"]    = p3c2.plot(pen=pen_des,   name="des")

    self._curves["tau_z_thrust"] = p3c3.plot(pen=pen_thr,   name="thrust")
    self._curves["tau_z_reaction"] = p3c3.plot(pen=pen_react, name="reaction")
    self._curves["tau_z_total"]  = p3c3.plot(pen=pen_total, name="total")
    self._curves["tau_z_des"]    = p3c3.plot(pen=pen_des,   name="des")

    # ========== 1-Row 4: f1234 / tilt / f_total ==========
    p4c1 = _mk_plot(self.glw1, 3, 0,  "r_cot [mm]", y_range=(-120., 120.))
    p4c2 = _mk_plot(self.glw1, 3, 1, "f_thrst [N]", y_range=(10. , 30.))
    p4c3 = _mk_plot(self.glw1, 3, 2, "f_total [N]", y_range=(40., 100.))

    self._curves["rcot_x_cmd"] = p4c1.plot(pen=pen_rcot_x_cmd, name="cmd x")
    self._curves["rcot_x_act"] = p4c1.plot(pen=pen_rcot_x_act, name="act x")
    self._curves["rcot_y_cmd"] = p4c1.plot(pen=pen_rcot_y_cmd, name="cmd y")
    self._curves["rcot_y_act"] = p4c1.plot(pen=pen_rcot_y_act, name="act y")

    for i in range(4):
      self._curves[f"F{i+1}"] = p4c2.plot(pen=_mk_pen("dash", width=1, color=rotor_colors[i]))
      self._curves[f"F{i+1}_con"] = p4c2.plot(pen=_mk_pen("solid", width=2, color=rotor_colors[i]), name=f"F{i+1}")

    self._curves["f_des"] = p4c3.plot(pen=pen_des, name="des")
    self._curves["f_tot"] = p4c3.plot(pen=pen_act, name="act")

    # ========== 1-Row 5: phase / solve_ms / solve_status ==========
    p5c1 = _mk_plot(self.glw1, 4, 0, "phase", add_legend=False, y_range=(-0.5, float(len(self._phase_order) - 0.5)))
    self._phase_plot_t1 = p5c1
    self._phase_plot_t1.setLabel("left", "phase")
    self._phase_plot_t1.getAxis("left").setTicks([[(i, str(int(code))) for i, code in enumerate(self._phase_order)]])
    p5c2 = _mk_plot(self.glw1, 4, 1, "solve_ms [ms]", y_range=(0., 40.))
    p5c3 = _mk_plot(self.glw1, 4, 2, "solve_status", add_legend=True)

    self._curves["solve_ms"] = p5c2.plot(pen=pen_act, name="solve_ms")

    # Status plot: fixed y ticks + legend only
    self._status_plot = p5c3
    self._status_plot.setLabel("left", "status")
    self._status_plot.setYRange(-1.1, 4.5, padding=0.05)
    self._status_plot.getAxis("left").setTicks([[(i, str(i)) for i in range(5)]])

    leg = self._status_plot.legend
    for s in range(5):
      c = self._status_colors[s]
      dummy = pg.PlotDataItem([np.nan], [np.nan], pen=None, symbol="s", symbolSize=10, symbolBrush=pg.mkBrush(*c), symbolPen=pg.mkPen(c[0], c[1], c[2], 255),)
      leg.addItem(dummy, f"{s}: {self._status_names[s]}")
    
    # ========== 2-Row 1 ==========
    p2r1c1 = _mk_plot(self.glw2, 0, 0, "pos_x [m]", y_range=(-2.5, 2.5))
    p2r1c2 = _mk_plot(self.glw2, 0, 1, "pos_y [m]", y_range=(-2.5, 2.5))
    p2r1c3 = _mk_plot(self.glw2, 0, 2, "pos_z [m]", y_range=(-1.5, 0.5))

    self._curves["t2_pos_x_des"] = p2r1c1.plot(pen=pen_des, name="des")
    self._curves["t2_pos_x_act"] = p2r1c1.plot(pen=pen_act, name="act")
    self._curves["t2_pos_y_des"] = p2r1c2.plot(pen=pen_des, name="des")
    self._curves["t2_pos_y_act"] = p2r1c2.plot(pen=pen_act, name="act")
    self._curves["t2_pos_z_des"] = p2r1c3.plot(pen=pen_des, name="des")
    self._curves["t2_pos_z_act"] = p2r1c3.plot(pen=pen_act, name="act")


    # ========== 2-Row 2 ==========
    p2r2c1 = _mk_plot(self.glw2, 1, 0, "vel_x [m/s]", y_range=(-3.0, 3.0))
    p2r2c2 = _mk_plot(self.glw2, 1, 1, "vel_y [m/s]", y_range=(-3.0, 3.0))
    p2r2c3 = _mk_plot(self.glw2, 1, 2, "vel_z [m/s]", y_range=(-3.0, 3.0))

    self._curves["t2_vel_x_des"] = p2r2c1.plot(pen=pen_des, name="des")
    self._curves["t2_vel_x_act"] = p2r2c1.plot(pen=pen_act, name="act")
    self._curves["t2_vel_y_des"] = p2r2c2.plot(pen=pen_des, name="des")
    self._curves["t2_vel_y_act"] = p2r2c2.plot(pen=pen_act, name="act")
    self._curves["t2_vel_z_des"] = p2r2c3.plot(pen=pen_des, name="des")
    self._curves["t2_vel_z_act"] = p2r2c3.plot(pen=pen_act, name="act")

    # ========== 2-Row 3 ==========
    p2r3c1 = _mk_plot(self.glw2, 2, 0, "acc_x [m/s^2]", y_range=(-20.0, 20.0))
    p2r3c2 = _mk_plot(self.glw2, 2, 1, "acc_y [m/s^2]", y_range=(-20.0, 20.0))
    p2r3c3 = _mk_plot(self.glw2, 2, 2, "acc_z [m/s^2]", y_range=(-20.0, 20.0))

    self._curves["t2_acc_x_des"] = p2r3c1.plot(pen=pen_des, name="des")
    self._curves["t2_acc_x_act"] = p2r3c1.plot(pen=pen_act, name="act")
    self._curves["t2_acc_y_des"] = p2r3c2.plot(pen=pen_des, name="des")
    self._curves["t2_acc_y_act"] = p2r3c2.plot(pen=pen_act, name="act")
    self._curves["t2_acc_z_des"] = p2r3c3.plot(pen=pen_des, name="des")
    self._curves["t2_acc_z_act"] = p2r3c3.plot(pen=pen_act, name="act")

    # ========== 2-Row 4 ==========
    p2r4c1 = _mk_plot(self.glw2, 3, 1, "roll [deg]", y_range=(-40., 40.))
    p2r4c2 = _mk_plot(self.glw2, 3, 0, "pitch [deg]", y_range=(-40., 40.))
    p2r4c3 = _mk_plot(self.glw2, 3, 2, "yaw [deg]", y_range=(-10., 10.))

    self._curves["t2_roll_raw"]  = p2r4c1.plot(pen=pen_raw, name="raw")
    self._curves["t2_roll_mrg"]  = p2r4c1.plot(pen=pen_mrg, name="mrg")
    self._curves["t2_roll_act"]  = p2r4c1.plot(pen=pen_act, name="act")

    self._curves["t2_pitch_raw"] = p2r4c2.plot(pen=pen_raw, name="raw")
    self._curves["t2_pitch_mrg"] = p2r4c2.plot(pen=pen_mrg, name="mrg")
    self._curves["t2_pitch_act"] = p2r4c2.plot(pen=pen_act, name="act")

    self._curves["t2_yaw_raw"]   = p2r4c3.plot(pen=pen_raw, name="raw")
    self._curves["t2_yaw_mrg"]   = p2r4c3.plot(pen=pen_mrg, name="mrg")
    self._curves["t2_yaw_act"]   = p2r4c3.plot(pen=pen_act, name="act")


    # ========== 2-Row 5 ==========
    # Left cell (row=4,col=0): phase legend text with colored squares
    p2_phase_leg = self.glw2.addPlot(row=4, col=0, title="phase description")
    p2_phase_leg.showGrid(x=False, y=False)
    p2_phase_leg.hideAxis("bottom")
    p2_phase_leg.hideAxis("left")
    p2_phase_leg.setMenuEnabled(False)
    p2_phase_leg.setMouseEnabled(x=False, y=False)
    p2_phase_leg.setXRange(0.0, 1.0, padding=0.0)
    p2_phase_leg.setYRange(0.0, 1.0, padding=0.0)
    try: p2_phase_leg.hideButtons()
    except Exception: pass

    # Build HTML legend (colored squares + description)
    lines = []
    for code in self._phase_order:
      rgba = self._phase_colors.get(int(code), (0, 0, 0, 220))
      hexcol = "#{:02x}{:02x}{:02x}".format(int(rgba[0]), int(rgba[1]), int(rgba[2]))
      name, desc = self._phase_info.get(int(code), ("?", ""))
      lines.append(f'<span style="color:{hexcol};">■</span> {int(code)}: {name} ({desc})')
    html = "<div style='font-size:11pt; line-height:130%; color:#111111;'>" + "<br>".join(lines) + "</div>"
    txt = pg.TextItem(html=html, anchor=(0, 0))
    p2_phase_leg.addItem(txt)
    txt.setPos(0.02, 0.98)

    # Middle cell (row=4,col=1): phase timeline plot (colored bars, no legend)
    p2_phase = _mk_plot(self.glw2, 4, 1, "phase", add_legend=False, y_range=(-0.5, float(len(self._phase_order) - 0.5)))
    self._phase_plot = p2_phase
    self._phase_plot.setLabel("left", "phase")
    # y ticks show the original phase codes (compact indices on y)
    self._phase_plot.getAxis("left").setTicks([[(i, str(int(code))) for i, code in enumerate(self._phase_order)]])

    # ========== 3-Row 1 ==========
    p3r1c1 = _mk_plot(self.glw3, 0, 0, "roll [deg]", y_range=(-40., 40.))
    p3r1c2 = _mk_plot(self.glw3, 0, 1, "pitch [deg]", y_range=(-40., 40.))
    p3r1c3 = _mk_plot(self.glw3, 0, 2, "yaw [deg]", y_range=(-10., 10.))

    self._curves["t3_roll_raw"]  = p3r1c1.plot(pen=pen_raw, name="raw")
    self._curves["t3_roll_mrg"]  = p3r1c1.plot(pen=pen_mrg, name="mrg")
    self._curves["t3_roll_act"]  = p3r1c1.plot(pen=pen_act, name="act")

    self._curves["t3_pitch_raw"] = p3r1c2.plot(pen=pen_raw, name="raw")
    self._curves["t3_pitch_mrg"] = p3r1c2.plot(pen=pen_mrg, name="mrg")
    self._curves["t3_pitch_act"] = p3r1c2.plot(pen=pen_act, name="act")

    self._curves["t3_yaw_raw"]   = p3r1c3.plot(pen=pen_raw, name="raw")
    self._curves["t3_yaw_mrg"]   = p3r1c3.plot(pen=pen_mrg, name="mrg")
    self._curves["t3_yaw_act"]   = p3r1c3.plot(pen=pen_act, name="act")

    # ========== 3-Row 2 ==========
    p3r2c1 = _mk_plot(self.glw3, 1, 0, "omega_x [deg/s]", y_range=(-150.0, 150.0))
    p3r2c2 = _mk_plot(self.glw3, 1, 1, "omega_y [deg/s]", y_range=(-150.0, 150.0))
    p3r2c3 = _mk_plot(self.glw3, 1, 2, "omega_z [deg/s]", y_range=( -10.0,  10.0))

    self._curves["t3_omega_x_des"] = p3r2c1.plot(pen=pen_des, name="des")
    self._curves["t3_omega_x_act"] = p3r2c1.plot(pen=pen_act, name="act")
    self._curves["t3_omega_y_des"] = p3r2c2.plot(pen=pen_des, name="des")
    self._curves["t3_omega_y_act"] = p3r2c2.plot(pen=pen_act, name="act")
    self._curves["t3_omega_z_des"] = p3r2c3.plot(pen=pen_des, name="des")
    self._curves["t3_omega_z_act"] = p3r2c3.plot(pen=pen_act, name="act")

    # ========== 3-Row 3 ==========
    p3r3c1 = _mk_plot(self.glw3, 2, 0, "alpha_x [deg/s^2]", y_range=(-700.0, 700.0))
    p3r3c2 = _mk_plot(self.glw3, 2, 1, "alpha_y [deg/s^2]", y_range=(-700.0, 700.0))
    p3r3c3 = _mk_plot(self.glw3, 2, 2, "alpha_z [deg/s^2]", y_range=( -20.0,  20.0))

    self._curves["t3_alpha_x_des"] = p3r3c1.plot(pen=pen_des, name="des")
    self._curves["t3_alpha_x_act"] = p3r3c1.plot(pen=pen_act, name="act")
    self._curves["t3_alpha_y_des"] = p3r3c2.plot(pen=pen_des, name="des")
    self._curves["t3_alpha_y_act"] = p3r3c2.plot(pen=pen_act, name="act")
    self._curves["t3_alpha_z_des"] = p3r3c3.plot(pen=pen_des, name="des")
    self._curves["t3_alpha_z_act"] = p3r3c3.plot(pen=pen_act, name="act")

    # ========== 3-Row 4 ==========
    p3r4c1 = _mk_plot(self.glw3, 3, 0, "tau_x [N·m]", y_range=(-1.5, 1.5))
    p3r4c2 = _mk_plot(self.glw3, 3, 1, "tau_y [N·m]", y_range=(-6., 11.))
    p3r4c3 = _mk_plot(self.glw3, 3, 2, "tau_z [N·m]", y_range=(-1., 1.))

    self._curves["t3_tau_x_off"]    = p3r4c1.plot(pen=pen_cot,   name="off-d")
    self._curves["t3_tau_x_thrust"] = p3r4c1.plot(pen=pen_thr,   name="thrust")
    self._curves["t3_tau_x_total"]  = p3r4c1.plot(pen=pen_total, name="total")
    self._curves["t3_tau_x_des"]    = p3r4c1.plot(pen=pen_des,   name="des")

    self._curves["t3_tau_y_off"]    = p3r4c2.plot(pen=pen_cot,   name="off-d")
    self._curves["t3_tau_y_thrust"] = p3r4c2.plot(pen=pen_thr,   name="thrust")
    self._curves["t3_tau_y_total"]  = p3r4c2.plot(pen=pen_total, name="total")
    self._curves["t3_tau_y_des"]    = p3r4c2.plot(pen=pen_des,   name="des")

    self._curves["t3_tau_z_thrust"] = p3r4c3.plot(pen=pen_thr,   name="thrust")
    self._curves["t3_tau_z_reaction"] = p3r4c3.plot(pen=pen_react, name="reaction")
    self._curves["t3_tau_z_total"]  = p3r4c3.plot(pen=pen_total, name="total")
    self._curves["t3_tau_z_des"]    = p3r4c3.plot(pen=pen_des,   name="des")


    # ========== 3-Row 5 ==========
    p3r5c1 = _mk_plot(self.glw3, 4, 0, "f_total [N]", y_range=(40., 100.))
    p3r5c3 = _mk_plot(self.glw3, 4, 2, "tilt_cmd [deg]", y_range=(-10., 10.))

    self._curves["t3_f_des"] = p3r5c1.plot(pen=pen_des, name="des")
    self._curves["t3_f_tot"] = p3r5c1.plot(pen=pen_act, name="act")

    for i in range(4): self._curves[f"t3_tilt{i+1}"] = p3r5c3.plot(pen=_mk_pen("solid", width=2, color=rotor_colors[i]), name=f"tilt{i+1}")

    p3_phase = _mk_plot(self.glw3, 4, 1, "phase", add_legend=False, y_range=(-0.5, float(len(self._phase_order) - 0.5)))
    self._phase_plot_t3 = p3_phase
    self._phase_plot_t3.setLabel("left", "phase")
    self._phase_plot_t3.getAxis("left").setTicks([[(i, str(int(code))) for i, code in enumerate(self._phase_order)]])

    # ========== 4-Tab (arm) ==========
    joint_y_ranges = {
      1: (-30.0,  30.0),
      2: (0.0,  50.0),
      3: (30.0,  100.0),
      4: (-45.0,  45.0),
      5: (-15.0, 15.0),
    }
    default_yr = (-180.0, 180.0)
    for j in range(5):
      yr = joint_y_ranges.get(j+1, default_yr)
      for a in range(4):
        title = f"a{a+1} j{j+1} [deg]"
        pj = _mk_plot(self.glw4_joint, j, a, title, add_legend=False, y_range=yr)
        k_act = f"arm{a+1}_j{j+1}_act"
        k_cmd = f"arm{a+1}_j{j+1}_cmd"
        self._curves[k_cmd] = pj.plot(pen=pen_q_cmd)
        self._curves[k_act] = pj.plot(pen=pen_q_act)

    if self.glw4_rotor is not None:
      # ---------------------------------------------------------
      # XY map (left): horizontal=y[mm], vertical=x[mm]
      # - rotor1..4 act: red dot
      # - rotor1..4 cmd: translucent blue dot
      # - rcot act: orange dot
      # - rcot cmd: translucent skyblue dot
      # - rotor1..4 act trajectory: translucent red line+dot
      # NOTE:
      # - Live: trajectory uses the currently rendered window (last ring window).
      # - Replay: _apply_replay_window slices to slider window, so trajectory matches it automatically.
      # ---------------------------------------------------------
      pxy = self.glw4_rotor.addPlot(row=0, col=0, rowspan=2, colspan=1, title="rotor XY map [mm]")
      _style(pxy)
      pxy.setLabel("bottom", "y [mm]")
      pxy.setLabel("left", "x [mm]")
      pxy.showLabel("bottom", True)
      pxy.enableAutoRange(axis="x", enable=False)
      pxy.enableAutoRange(axis="y", enable=False)
      pxy.setXRange(-400.0, 400.0, padding=0.0)
      pxy.setYRange(-400.0, 400.0, padding=0.0)
      try: pxy.setAspectLocked(True)
      except Exception: pass
      try: pxy.hideButtons()
      except Exception: pass
      self._xy_map_plot = pxy

      def _arc_xy(cx: float, cy: float, r: float, a0_deg: float, a1_deg: float, n: int = 64):
        th = np.linspace(np.deg2rad(a0_deg), np.deg2rad(a1_deg), int(max(8, n)), dtype=np.float32)
        xs = (cx + r * np.cos(th)).astype(np.float32, copy=False)
        ys = (cy + r * np.sin(th)).astype(np.float32, copy=False)
        return xs, ys

      def _add_border_segment(x: np.ndarray, y: np.ndarray, pen: pg.mkPen, z: float = 8.0):
        it = pxy.plot(x, y, pen=pen)
        try: it.setZValue(z)
        except Exception: pass
        self._xy_border_items.append(it)

      inv_sqrt2 = 1.0 / math.sqrt(2.0)
      B2BASE_X = [ 0.12*inv_sqrt2, -0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2 ]
      B2BASE_Y = [-0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2,  0.12*inv_sqrt2 ]

      R_IN  = 150.6
      R_OUT = 292.5

      pen_border = pg.mkPen(color=(0, 120, 255, 255), width=2)

      for i in range(4):
        # (x,y)[m] -> (x,y)[mm]
        x_mm = float(B2BASE_X[i]) * 1000.0
        y_mm = float(B2BASE_Y[i]) * 1000.0
        # plot coords: X= y_mm (horizontal), Y= x_mm (vertical)
        cx = y_mm
        cy = x_mm

        # Select the 90deg quadrant by center sign in *plot* coords
        if   (cx >= 0.0) and (cy >= 0.0): a0, a1 = 0.0,   90.0
        elif (cx <  0.0) and (cy >= 0.0): a0, a1 = 90.0,  180.0
        elif (cx <  0.0) and (cy <  0.0): a0, a1 = 180.0, 270.0
        else:                              a0, a1 = 270.0, 360.0

        # Outer quarter-arc
        xo, yo = _arc_xy(cx, cy, R_OUT, a0, a1, n=80)
        _add_border_segment(xo, yo, pen_border, z=8.0)

        # Inner quarter-arc
        xi, yi = _arc_xy(cx, cy, R_IN, a0, a1, n=80)
        _add_border_segment(xi, yi, pen_border, z=8.0)

        # Two straight segments connecting arc endpoints (a0 and a1)
        a0r = math.radians(a0)
        a1r = math.radians(a1)
        # at a0
        x0o = cx + R_OUT * math.cos(a0r)
        y0o = cy + R_OUT * math.sin(a0r)
        x0i = cx + R_IN  * math.cos(a0r)
        y0i = cy + R_IN  * math.sin(a0r)
        _add_border_segment(np.asarray([x0i, x0o], dtype=np.float32), np.asarray([y0i, y0o], dtype=np.float32), pen_border, z=8.0)
        # at a1
        x1o = cx + R_OUT * math.cos(a1r)
        y1o = cy + R_OUT * math.sin(a1r)
        x1i = cx + R_IN  * math.cos(a1r)
        y1i = cy + R_IN  * math.sin(a1r)
        _add_border_segment(np.asarray([x1i, x1o], dtype=np.float32), np.asarray([y1i, y1o], dtype=np.float32), pen_border, z=8.0)

      for i in range(4):
        circ = QtWidgets.QGraphicsEllipseItem()
        circ.setPen(QPen(QColor(0, 0, 0, 120), 1, QtCore.Qt.DashLine))
        circ.setBrush(QBrush(QtCore.Qt.NoBrush))
        circ.setZValue(5)
        circ.setVisible(False)
        pxy.addItem(circ)
        self._xy_circles[i+1] = circ

      # Curves for XY map
      pen_traj = pg.mkPen(255, 0, 0, 120, width=1)          # translucent red
      brush_traj = pg.mkBrush(255, 130, 0, 5)               # translucent orange dot
      brush_act = pg.mkBrush(255, 0, 0, 255)                # red dot
      brush_cmd = pg.mkBrush(0, 0, 255, 200)                # translucent blue dot

      for i in range(4):
        self._curves[f"xy_r{i+1}_traj"] = pxy.plot(pen=pen_traj, symbol="o", symbolSize=2, symbolBrush=brush_traj, symbolPen=None)
        self._curves[f"xy_r{i+1}_act"]  = pxy.plot(pen=None, symbol="o", symbolSize=6, symbolBrush=brush_act, symbolPen=None)
        self._curves[f"xy_r{i+1}_cmd"]  = pxy.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=brush_cmd, symbolPen=None)
      self._curves["xy_rcot_act"] = pxy.plot(pen=None, symbol="o", symbolSize=6, symbolBrush=brush_act, symbolPen=None)
      self._curves["xy_rcot_cmd"] = pxy.plot(pen=None, symbol="o", symbolSize=4, symbolBrush=brush_cmd, symbolPen=None)

      pr1 = _mk_plot(self.glw4_rotor, 0, 1, "rotor1 xy pos [mm]", add_legend=True, y_range=(-400., 400.))
      pr4 = _mk_plot(self.glw4_rotor, 0, 2, "rotor4 xy pos [mm]", add_legend=True, y_range=(40., 440.))
      pr2 = _mk_plot(self.glw4_rotor, 1, 1, "rotor2 xy pos [mm]", add_legend=True, y_range=(-440., -40.))
      pr3 = _mk_plot(self.glw4_rotor, 1, 2, "rotor3 xy pos [mm]", add_legend=True, y_range=(-400., 400.))

      def _add_rotor_curves(plot: pg.PlotItem, tag: str) -> None:
        self._curves[f"{tag}_x_cmd"] = plot.plot(pen=pen_rx_cmd, name="x_cmd")
        self._curves[f"{tag}_x_act"] = plot.plot(pen=pen_rx_act, name="x_act")
        self._curves[f"{tag}_y_cmd"] = plot.plot(pen=pen_ry_cmd, name="y_cmd")
        self._curves[f"{tag}_y_act"] = plot.plot(pen=pen_ry_act, name="y_act")

      _add_rotor_curves(pr1, "r1")
      _add_rotor_curves(pr2, "r2")
      _add_rotor_curves(pr3, "r3")
      _add_rotor_curves(pr4, "r4")

  def _update_plots(self, t: np.ndarray, ch: Dict[str, np.ndarray], wc: int) -> None:
    if t.size == 0:
      self.lbl_stat.setText("no data")
      return

    # NOTE: keep float32 for speed/memory; pyqtgraph accepts float32
    # Live: window is relative to its own start.
    # Replay: keep a fixed time base (t0 of full replay) so slider <-> plots are consistent.
    if self.replay_path is not None and self._replay_time_base is not None: tt = (t - float(self._replay_time_base)).astype(np.float32, copy=False)
    else: tt = (t - t[0]).astype(np.float32, copy=False)

    # Decimate ONLY for rendering (recording remains full resolution)
    n = int(tt.size)
    sl = slice(None)
    max_pts = int(getattr(self, "view_max_points", 0))
    if max_pts > 0 and n > max_pts:
      stride = (n + max_pts - 1) // max_pts
      sl = slice(None, None, int(stride))

    # Batch update to reduce repaint overhead during many setData calls
    # NOTE: update all tabs that have curves (glw1/glw2/glw3)
    self.glw1.setUpdatesEnabled(False)
    self.glw2.setUpdatesEnabled(False)
    self.glw3.setUpdatesEnabled(False)
    self.glw4_joint.setUpdatesEnabled(False)
    self.glw4_rotor.setUpdatesEnabled(False)
    try:
      x = tt[sl]

      # --- Pull/convert only what we render (decimated view) ---
      pos_act = ch["pos"][sl, :]
      pos_des = ch["pos_d"][sl, :]
      vel_act = ch["vel"][sl, :]
      vel_des = ch["vel_d"][sl, :]
      acc_act = ch["acc"][sl, :]
      acc_des = ch["acc_d"][sl, :]

      rpy_act_deg = ch["rpy"][sl, :] * _RAD2DEG
      rpy_raw_deg = ch["rpy_raw"][sl, :] * _RAD2DEG
      rpy_d_deg   = ch["rpy_d"][sl, :] * _RAD2DEG
      omega_act = ch["omega"][sl, :] * _RAD2DEG
      omega_des = ch["omega_d"][sl, :] * _RAD2DEG
      alpha_act = ch["alpha"][sl, :] * _RAD2DEG
      alpha_des = ch["alpha_d"][sl, :] * _RAD2DEG

      tau_d      = ch["tau_d"][sl, :]
      tau_z_t    = ch["tau_z_t"][sl]
      tau_off    = ch["tau_off"][sl, :]
      tau_thrust = ch["tau_thrust"][sl, :]

      f_thrst     = ch["f_thrst"][sl, :]
      f_thrst_con = ch["f_thrst_con"][sl, :]
      tilt_deg    = ch["tilt"][sl, :] * _RAD2DEG
      f_des       = ch["f_des"][sl]

      r_cot_d_mm   = ch["r_cot_d"][sl, :] * _M2MM
      r_cot_act_mm = ch["r_cot"][sl, :] * _M2MM

      q_act = ch.get("q", None)
      q_cmd = ch.get("q_cmd", None)
      if q_act is not None: q_act = q_act[sl, :] * _RAD2DEG
      if q_cmd is not None: q_cmd = q_cmd[sl, :] * _RAD2DEG

      r1 = ch.get("r_rotor1", None)
      r2 = ch.get("r_rotor2", None)
      r3 = ch.get("r_rotor3", None)
      r4 = ch.get("r_rotor4", None)
      r1d = ch.get("r_rotor1_d", None)
      r2d = ch.get("r_rotor2_d", None)
      r3d = ch.get("r_rotor3_d", None)
      r4d = ch.get("r_rotor4_d", None)
      if r1 is not None: r1 = r1[sl, :] * _M2MM
      if r2 is not None: r2 = r2[sl, :] * _M2MM
      if r3 is not None: r3 = r3[sl, :] * _M2MM
      if r4 is not None: r4 = r4[sl, :] * _M2MM
      if r1d is not None: r1d = r1d[sl, :] * _M2MM
      if r2d is not None: r2d = r2d[sl, :] * _M2MM
      if r3d is not None: r3d = r3d[sl, :] * _M2MM
      if r4d is not None: r4d = r4d[sl, :] * _M2MM

      solve_ms     = ch["solve_ms"][sl]
      solve_status = ch["solve_status"][sl].astype(np.int32, copy=False)

      phase_arr = ch.get("phase", None)
      phase_u8 = None
      if phase_arr is not None: phase_u8 = phase_arr[sl].astype(np.uint8, copy=False)

      # --- Row 1: position ---
      self._curves["pos_y_act"].setData(x, pos_act[:, 1])
      self._curves["pos_y_des"].setData(x, pos_des[:, 1])
      self._curves["pos_x_act"].setData(x, pos_act[:, 0])
      self._curves["pos_x_des"].setData(x, pos_des[:, 0])
      self._curves["pos_z_act"].setData(x, pos_act[:, 2])
      self._curves["pos_z_des"].setData(x, pos_des[:, 2])

      # --- Row 2: RPY ---
      self._curves["roll_raw"].setData(x,  rpy_raw_deg[:, 0])
      self._curves["roll_mrg"].setData(x,  rpy_d_deg[:, 0])
      self._curves["roll_act"].setData(x,  rpy_act_deg[:, 0])

      self._curves["pitch_raw"].setData(x, rpy_raw_deg[:, 1])
      self._curves["pitch_mrg"].setData(x, rpy_d_deg[:, 1])
      self._curves["pitch_act"].setData(x, rpy_act_deg[:, 1])

      self._curves["yaw_raw"].setData(x,   rpy_raw_deg[:, 2])
      self._curves["yaw_mrg"].setData(x,   rpy_d_deg[:, 2])
      self._curves["yaw_act"].setData(x,   rpy_act_deg[:, 2])

      # --- Row 3: tau ---
      self._curves["tau_x_des"].setData(x, tau_d[:, 0])
      self._curves["tau_x_off"].setData(x, tau_off[:, 0])
      self._curves["tau_x_thrust"].setData(x, tau_thrust[:, 0])
      self._curves["tau_x_total"].setData(x, tau_off[:, 0] + tau_thrust[:, 0])

      self._curves["tau_y_des"].setData(x, tau_d[:, 1])
      self._curves["tau_y_off"].setData(x, tau_off[:, 1])
      self._curves["tau_y_thrust"].setData(x, tau_thrust[:, 1])
      self._curves["tau_y_total"].setData(x, tau_off[:, 1] + tau_thrust[:, 1])

      self._curves["tau_z_des"].setData(x, tau_d[:, 2])
      self._curves["tau_z_total"].setData(x, tau_z_t + tau_thrust[:, 2])
      self._curves["tau_z_thrust"].setData(x, tau_z_t)
      self._curves["tau_z_reaction"].setData(x, tau_thrust[:, 2])

      # --- Row 4: thrust / tilt / total thrust ---
      for i in range(4):
        self._curves[f"F{i+1}"].setData(x, f_thrst[:, i])
        self._curves[f"F{i+1}_con"].setData(x, f_thrst_con[:, i])
        k = f"tilt{i+1}"
        if k in self._curves: self._curves[k].setData(x, tilt_deg[:, i])
      self._curves["f_des"].setData(x, f_des)
      self._curves["f_tot"].setData(x, f_thrst_con.sum(axis=1), dtype=np.float32)

      # --- Row 5: r_cot / solve_ms / solve_status ---
      self._curves["rcot_x_cmd"].setData(x, r_cot_d_mm[:, 0])
      self._curves["rcot_y_cmd"].setData(x, r_cot_d_mm[:, 1])
      self._curves["rcot_x_act"].setData(x, r_cot_act_mm[:, 0])
      self._curves["rcot_y_act"].setData(x, r_cot_act_mm[:, 1])

      self._curves["solve_ms"].setData(x, solve_ms)

      # --- Status bars: compress by segments (run-length encoding) ---
      # Keep visualization intent but reduce updates from O(N) to O(#segments).
      if x.size >= 2:
        dt = float(x[1] - x[0])
        if not np.isfinite(dt) or dt <= 0.0:
          dt = 0.01
      else:
        dt = 0.01

      y = solve_status
      if y.size > 0:
        # Indices where status changes
        changes = np.nonzero(y[1:] != y[:-1])[0] + 1
        edges = np.concatenate(([0], changes, [y.size]))

        # Collect per-status segments (center x + width)
        per_x: Dict[int, List[float]] = {s: [] for s in range(5)}
        per_w: Dict[int, List[float]] = {s: [] for s in range(5)}

        for k in range(edges.size - 1):
          a = int(edges[k])
          b = int(edges[k + 1])
          s = int(y[a])

          # Ignore invalid / missing samples (keeps behavior similar to old mask-based code)
          if s < 0 or s > 4:
            continue

          x0 = float(x[a])
          x1 = float(x[b - 1])
          w = max(dt, (x1 - x0) + dt)   # cover the segment span
          xc = 0.5 * (x0 + x1)

          per_x[s].append(xc)
          per_w[s].append(w)

        y0 = -1.0
        for s in range(5):
          xs = np.asarray(per_x[s], dtype=np.float32)
          ws = np.asarray(per_w[s], dtype=np.float32)
          hs = np.full(xs.shape, float(s + 1), dtype=np.float32)

          if s not in self._status_bars:
            bar = pg.BarGraphItem(x=xs, y0=y0, height=hs, width=ws, brush=pg.mkBrush(*self._status_colors[s]), pen=None,)
            self._status_plot.addItem(bar)
            self._status_bars[s] = bar
          else:
            self._status_bars[s].setOpts(x=xs, y0=y0, height=hs, width=ws, brush=pg.mkBrush(*self._status_colors[s]), pen=None,)

      # --- Status text ---
      last_ms = float(solve_ms[-1]) if solve_ms.size > 0 else float("nan")
      last_st = int(solve_status[-1]) if solve_status.size > 0 else -1

      extra = ""
      if self.recorder is not None and self.recorder.started: extra = f" | rec_samples={int(self.recorder.n_samples_total)} | dropped={int(self.recorder.dropped_total)}"

      self.lbl_stat.setText(f"wc={int(wc)} | samples={int(x.size)} | last solve_ms={last_ms:.3f} | last status={last_st}{extra}")

      # --- Tab 2  ---
      self._curves["t2_pos_y_act"].setData(x, pos_act[:, 1])
      self._curves["t2_pos_y_des"].setData(x, pos_des[:, 1])
      self._curves["t2_pos_x_act"].setData(x, pos_act[:, 0])
      self._curves["t2_pos_x_des"].setData(x, pos_des[:, 0])
      self._curves["t2_pos_z_act"].setData(x, pos_act[:, 2])
      self._curves["t2_pos_z_des"].setData(x, pos_des[:, 2])

      self._curves["t2_vel_x_act"].setData(x, vel_act[:, 0])
      self._curves["t2_vel_x_des"].setData(x, vel_des[:, 0])
      self._curves["t2_vel_y_act"].setData(x, vel_act[:, 1])
      self._curves["t2_vel_y_des"].setData(x, vel_des[:, 1])
      self._curves["t2_vel_z_act"].setData(x, vel_act[:, 2])
      self._curves["t2_vel_z_des"].setData(x, vel_des[:, 2])

      self._curves["t2_acc_x_act"].setData(x, acc_act[:, 0])
      self._curves["t2_acc_x_des"].setData(x, acc_des[:, 0])
      self._curves["t2_acc_y_act"].setData(x, acc_act[:, 1])
      self._curves["t2_acc_y_des"].setData(x, acc_des[:, 1])
      self._curves["t2_acc_z_act"].setData(x, acc_act[:, 2])
      self._curves["t2_acc_z_des"].setData(x, acc_des[:, 2])

      self._curves["t2_roll_raw"].setData(x,  rpy_raw_deg[:, 0])
      self._curves["t2_roll_mrg"].setData(x,  rpy_d_deg[:, 0])
      self._curves["t2_roll_act"].setData(x,  rpy_act_deg[:, 0])

      self._curves["t2_pitch_raw"].setData(x, rpy_raw_deg[:, 1])
      self._curves["t2_pitch_mrg"].setData(x, rpy_d_deg[:, 1])
      self._curves["t2_pitch_act"].setData(x, rpy_act_deg[:, 1])

      self._curves["t2_yaw_raw"].setData(x,   rpy_raw_deg[:, 2])
      self._curves["t2_yaw_mrg"].setData(x,   rpy_d_deg[:, 2])
      self._curves["t2_yaw_act"].setData(x,   rpy_act_deg[:, 2])

      # Phase timeline (tab2 + tab3). If phase is missing, clear bars and skip.
      if phase_u8 is None:
        self._clear_phase_bars()
      elif (x.size > 0) and (self._phase_plot is not None or self._phase_plot_t3 is not None or self._phase_plot_t1 is not None):
        if x.size >= 2: # dt estimate for bar width
          dtp = float(x[1] - x[0])
          if not np.isfinite(dtp) or dtp <= 0.0: dtp = 0.01
        else:
          dtp = 0.01

        # Map uint8 phase -> compact idx in [0..len(order)-1], else -1
        idx = self._phase_lut[phase_u8]  # int16

        # RLE boundaries
        changes = np.nonzero(idx[1:] != idx[:-1])[0] + 1
        edges = np.concatenate(([0], changes, [idx.size]))

        per_x: Dict[int, List[float]] = {int(code): [] for code in self._phase_order}
        per_w: Dict[int, List[float]] = {int(code): [] for code in self._phase_order}

        for k in range(edges.size - 1):
          a = int(edges[k])
          b = int(edges[k + 1])
          ii = int(idx[a])
          if ii < 0 or ii >= len(self._phase_order):
            continue
          code = int(self._phase_order[ii])

          x0 = float(x[a])
          x1 = float(x[b - 1])
          w = max(dtp, (x1 - x0) + dtp)
          xc = 0.5 * (x0 + x1)
          per_x[code].append(xc)
          per_w[code].append(w)

        # Update bars in both plots (tab2 + tab3)
        targets = []
        if self._phase_plot_t1 is not None:
          targets.append((self._phase_plot_t1, self._phase_bars_t1))
        if self._phase_plot is not None:
          targets.append((self._phase_plot, self._phase_bars))
        if self._phase_plot_t3 is not None:
          targets.append((self._phase_plot_t3, self._phase_bars_t3))

        for plot, bars in targets:
          for ii, code in enumerate(self._phase_order):
            code = int(code)
            xs = np.asarray(per_x[code], dtype=np.float32)
            ws = np.asarray(per_w[code], dtype=np.float32)
            # Bars centered at y=idx, with constant height
            y0 = np.full(xs.shape, float(ii) - 0.45, dtype=np.float32)
            hh = np.full(xs.shape, 0.90, dtype=np.float32)
            rgba = self._phase_colors.get(code, (0, 0, 0, 220))
            brush = pg.mkBrush(*rgba)

            if code not in bars:
              bar = pg.BarGraphItem(x=xs, y0=y0, height=hh, width=ws, brush=brush, pen=None)
              plot.addItem(bar)
              bars[code] = bar
            else:
              bars[code].setOpts(x=xs, y0=y0, height=hh, width=ws, brush=brush, pen=None)

      # --- Tab 3 ---
      self._curves["t3_roll_raw"].setData(x,  rpy_raw_deg[:, 0])
      self._curves["t3_roll_mrg"].setData(x,  rpy_d_deg[:, 0])
      self._curves["t3_roll_act"].setData(x,  rpy_act_deg[:, 0])

      self._curves["t3_pitch_raw"].setData(x, rpy_raw_deg[:, 1])
      self._curves["t3_pitch_mrg"].setData(x, rpy_d_deg[:, 1])
      self._curves["t3_pitch_act"].setData(x, rpy_act_deg[:, 1])

      self._curves["t3_yaw_raw"].setData(x,   rpy_raw_deg[:, 2])
      self._curves["t3_yaw_mrg"].setData(x,   rpy_d_deg[:, 2])
      self._curves["t3_yaw_act"].setData(x,   rpy_act_deg[:, 2])

      self._curves["t3_omega_x_act"].setData(x, omega_act[:, 0])
      self._curves["t3_omega_x_des"].setData(x, omega_des[:, 0])
      self._curves["t3_omega_y_act"].setData(x, omega_act[:, 1])
      self._curves["t3_omega_y_des"].setData(x, omega_des[:, 1])
      self._curves["t3_omega_z_act"].setData(x, omega_act[:, 2])
      self._curves["t3_omega_z_des"].setData(x, omega_des[:, 2])

      self._curves["t3_alpha_x_act"].setData(x, alpha_act[:, 0])
      self._curves["t3_alpha_x_des"].setData(x, alpha_des[:, 0])
      self._curves["t3_alpha_y_act"].setData(x, alpha_act[:, 1])
      self._curves["t3_alpha_y_des"].setData(x, alpha_des[:, 1])
      self._curves["t3_alpha_z_act"].setData(x, alpha_act[:, 2])
      self._curves["t3_alpha_z_des"].setData(x, alpha_des[:, 2])

      self._curves["t3_tau_x_des"].setData(x, tau_d[:, 0])
      self._curves["t3_tau_x_off"].setData(x, tau_off[:, 0])
      self._curves["t3_tau_x_thrust"].setData(x, tau_thrust[:, 0])
      self._curves["t3_tau_x_total"].setData(x, tau_off[:, 0] + tau_thrust[:, 0])

      self._curves["t3_tau_y_des"].setData(x, tau_d[:, 1])
      self._curves["t3_tau_y_off"].setData(x, tau_off[:, 1])
      self._curves["t3_tau_y_thrust"].setData(x, tau_thrust[:, 1])
      self._curves["t3_tau_y_total"].setData(x, tau_off[:, 1] + tau_thrust[:, 1])

      self._curves["t3_tau_z_des"].setData(x, tau_d[:, 2])
      self._curves["t3_tau_z_total"].setData(x, tau_z_t + tau_thrust[:, 2])
      self._curves["t3_tau_z_thrust"].setData(x, tau_z_t)
      self._curves["t3_tau_z_reaction"].setData(x, tau_thrust[:, 2])

      self._curves["t3_f_des"].setData(x, f_des)
      self._curves["t3_f_tot"].setData(x, f_thrst_con.sum(axis=1), dtype=np.float32)
      for i in range(4):
        k = f"t3_tilt{i+1}"
        if k in self._curves: self._curves[k].setData(x, tilt_deg[:, i])

      # --- Tab 4 (arm): joints + r_rotor ---
      # joints: 5 rows (joint 1..5), 4 cols (arm 1..4). q stored as [arm1, arm2, arm3, arm4] each 5.
      if (q_act is not None) and (q_cmd is not None) and (q_act.shape[1] >= 20) and (q_cmd.shape[1] >= 20):
        for a in range(4):
          for j in range(5):
            idx = int(5 * a + j)
            k_act = f"arm{a+1}_j{j+1}_act"
            k_cmd = f"arm{a+1}_j{j+1}_cmd"
            if k_cmd in self._curves: self._curves[k_cmd].setData(x, q_cmd[:, idx])
            if k_act in self._curves: self._curves[k_act].setData(x, q_act[:, idx])

      # r_rotor plots (mm): each plot shows x/y for cmd/act
      def _set_rotor(tag: str, act: Optional[np.ndarray], cmd: Optional[np.ndarray]) -> None:
        if act is None or cmd is None: return
        if act.shape[1] < 2 or cmd.shape[1] < 2: return
        kx_c = f"{tag}_x_cmd"
        kx_a = f"{tag}_x_act"
        ky_c = f"{tag}_y_cmd"
        ky_a = f"{tag}_y_act"
        if kx_c in self._curves: self._curves[kx_c].setData(x, cmd[:, 0])
        if kx_a in self._curves: self._curves[kx_a].setData(x, act[:, 0])
        if ky_c in self._curves: self._curves[ky_c].setData(x, cmd[:, 1])
        if ky_a in self._curves: self._curves[ky_a].setData(x, act[:, 1])

      _set_rotor("r1", r1, r1d)
      _set_rotor("r2", r2, r2d)
      _set_rotor("r3", r3, r3d)
      _set_rotor("r4", r4, r4d)

      if self._xy_map_plot is not None:
        def _set_point(key: str, xy: Optional[np.ndarray]) -> None:
          it = self._curves.get(key, None)
          if it is None: return
          if xy is None or xy.size < 2 or xy.shape[1] < 2:
            it.setData([], [])
            return
          x_mm = float(xy[-1, 0])
          y_mm = float(xy[-1, 1])
          if (not np.isfinite(x_mm)) or (not np.isfinite(y_mm)):
            it.setData([], [])
            return
          # horizontal=y, vertical=x
          it.setData([y_mm], [x_mm])

        def _set_traj(key: str, xy: Optional[np.ndarray]) -> None:
          it = self._curves.get(key, None)
          if it is None: return
          if xy is None or xy.ndim != 2 or xy.shape[1] < 2:
            it.setData([], [])
            return
          xx = xy[:, 0].astype(np.float32, copy=False)  # x [mm]
          yy = xy[:, 1].astype(np.float32, copy=False)  # y [mm]
          ok = np.isfinite(xx) & np.isfinite(yy)
          if not np.any(ok):
            it.setData([], [])
            return
          # horizontal=y, vertical=x
          it.setData(yy[ok], xx[ok])

        def _set_circle(idx: int, xy: Optional[np.ndarray]) -> None:
          it = self._xy_circles.get(idx, None)
          if it is None: return
          if xy is None or xy.size < 2 or xy.shape[1] < 2:
            it.setVisible(False)
            return
          x_mm = float(xy[-1, 0])  # x [mm]
          y_mm = float(xy[-1, 1])  # y [mm]
          if (not np.isfinite(x_mm)) or (not np.isfinite(y_mm)):
            it.setVisible(False)
            return

          r = 220.0  # radius [mm] (diameter 440mm)
          # Plot coordinates: horizontal = y_mm, vertical = x_mm
          it.setRect(y_mm - r, x_mm - r, 2.0 * r, 2.0 * r)
          it.setVisible(True)

        rot_act = [r1, r2, r3, r4]
        rot_cmd = [r1d, r2d, r3d, r4d]
        for i in range(4):
          _set_traj(f"xy_r{i+1}_traj", rot_act[i])
          _set_point(f"xy_r{i+1}_act", rot_act[i])
          _set_point(f"xy_r{i+1}_cmd", rot_cmd[i])
          _set_circle(i+1, rot_act[i])

        # rcot: use latest sample (already in mm)
        _set_point("xy_rcot_act", r_cot_act_mm)
        _set_point("xy_rcot_cmd", r_cot_d_mm)

    finally:
      self.glw1.setUpdatesEnabled(True)
      self.glw2.setUpdatesEnabled(True)
      self.glw3.setUpdatesEnabled(True)
      self.glw4_joint.setUpdatesEnabled(True)
      self.glw4_rotor.setUpdatesEnabled(True)

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
          "vel_d": data["vel_d"].astype(np.float32),
          "vel": data["vel"].astype(np.float32),
          "acc_d": data["acc_d"].astype(np.float32),
          "acc": data["acc"].astype(np.float32),
          "rpy": data["rpy"].astype(np.float32),
          "rpy_raw": data["rpy_raw"].astype(np.float32),
          "rpy_d": data["rpy_d"].astype(np.float32),
          "omega_d": data["omega_d"].astype(np.float32),
          "omega": data["omega"].astype(np.float32),
          "alpha_d": data["alpha_d"].astype(np.float32),
          "alpha": data["alpha"].astype(np.float32),
          "tau_d": data["tau_d"].astype(np.float32),
          "tau_z_t": data["tau_z_t"].astype(np.float32),
          "tau_off": data["tau_off"].astype(np.float32),
          "tau_thrust": data["tau_thrust"].astype(np.float32),
          "tilt": data["tilt"].astype(np.float32),
          "f_thrst": data["f_thrst"].astype(np.float32),
          "f_thrst_con": data["f_thrst_con"].astype(np.float32),
          "f_des": data["f_des"].astype(np.float32),
          "f_tot": data["f_tot"].astype(np.float32),
          "r_cot": data["r_cot"].astype(np.float32),
          "r_cot_d": data["r_cot_d"].astype(np.float32),
          "r_rotor1": data["r_rotor1"].astype(np.float32),
          "r_rotor2": data["r_rotor2"].astype(np.float32),
          "r_rotor3": data["r_rotor3"].astype(np.float32),
          "r_rotor4": data["r_rotor4"].astype(np.float32),
          "r_rotor1_d": data["r_rotor1_d"].astype(np.float32),
          "r_rotor2_d": data["r_rotor2_d"].astype(np.float32),
          "r_rotor3_d": data["r_rotor3_d"].astype(np.float32),
          "r_rotor4_d": data["r_rotor4_d"].astype(np.float32),
          "q": data["q"].astype(np.float32),
          "q_cmd": data["q_cmd"].astype(np.float32),
          "solve_ms": data["solve_ms"].astype(np.float32),
          "solve_status": data["solve_status"].astype(np.int32),
          "phase": data["phase"].astype(np.uint8),
        }
        wc_end = int(data["wc_end"]) if "wc_end" in data else int(t.size)

        # Store full replay buffers for window slicing
        self._replay_full_t = t
        self._replay_full_ch = ch
        self._replay_wc_end = int(wc_end)
        if t.size > 0:
          self._replay_time_base = float(t[0])
          self._replay_full_tt = (t - float(t[0])).astype(np.float32, copy=False)
          # Initialize slider to full range and render full window
          if self._range_slider is not None:
            self._set_slider_range(0.0, float(self._replay_full_tt[-1]))
          self._apply_replay_window(0.0, float(self._replay_full_tt[-1]), source="init")
          self.lbl_stat.setText(f"replay loaded: {rp.name} | samples={int(t.size)}")
        else:
          self._update_plots(t, ch, wc_end)
          self.lbl_stat.setText(f"replay loaded: {rp.name} | samples=0")

      else:
        # Treat as snapshot mmap
        r = MMapReader(str(rp))
        r.open()
        t, ch = r.read_all()
        wc = int(ch.get("write_count", 0))
        r.close()

        # remove write_count from channel dict
        ch.pop("write_count", None)

        # Backfill missing keys for older snapshots (keeps plotting safe)
        n = int(t.size)
        def _bf(key: str, shape: Tuple[int, ...], dtype, fill):
          if key not in ch: ch[key] = np.full(shape, fill, dtype=dtype)
        _bf("r_rotor1", (n, 2), np.float32, np.nan)
        _bf("r_rotor2", (n, 2), np.float32, np.nan)
        _bf("r_rotor3", (n, 2), np.float32, np.nan)
        _bf("r_rotor4", (n, 2), np.float32, np.nan)
        _bf("r_rotor1_d", (n, 2), np.float32, np.nan)
        _bf("r_rotor2_d", (n, 2), np.float32, np.nan)
        _bf("r_rotor3_d", (n, 2), np.float32, np.nan)
        _bf("r_rotor4_d", (n, 2), np.float32, np.nan)
        _bf("q", (n, 20), np.float32, np.nan)
        _bf("q_cmd", (n, 20), np.float32, np.nan)

        # Store full replay buffers for window slicing
        self._replay_full_t = t
        self._replay_full_ch = ch
        self._replay_wc_end = int(wc)
        if t.size > 0:
          self._replay_time_base = float(t[0])
          self._replay_full_tt = (t - float(t[0])).astype(np.float32, copy=False)
          if self._range_slider is not None:
            self._set_slider_range(0.0, float(self._replay_full_tt[-1]))
          self._apply_replay_window(0.0, float(self._replay_full_tt[-1]), source="init")
          self.lbl_stat.setText(f"replay mmap loaded: {rp.name} | samples={int(t.size)}")
        else:
          self._update_plots(t, ch, wc)
          self.lbl_stat.setText(f"replay mmap loaded: {rp.name} | samples=0")

    except Exception as e: self.lbl_stat.setText(f"replay error: {e}")

  @QtCore.pyqtSlot()
  def on_timer(self) -> None:
    try:
      # ---------
      # Detect mmap lifecycle changes (file replaced or removed)
      # ---------
      if self.reader.mm is not None and self.reader.changed_on_disk():
        # Finalize current flight before losing the old mapping
        if self.recorder is not None:
          self._rotate_recording("mmap replaced/removed")
        try: self.reader.close()
        except Exception: pass
        self._session_start_ns = None

        # If file is currently missing, just wait (next ticks will open when it reappears)
        if not os.path.exists(self.reader.path):
          self.lbl_stat.setText(f"waiting: {self.reader.path}")
          return

      # Open if needed
      if self.reader.mm is None:
        self.reader.open()
        self.lbl_stat.setText(f"opened (cap={self.reader.header.capacity})")

        if self.recorder is not None: # new flight begins => new recorder instance already exists in live mode
          self.recorder.start(self.reader)
        self._session_start_ns = int(self.reader.header.start_time_ns)
        self._clear_status_bars()

      # ---------
      # Detect "new flight" even if inode didn't change:
      # - header.start_time_ns changed
      # - write_count went backwards (reset)
      # ---------
      if self.replay_path is None and self.reader.header is not None:
        cur_start_ns = int(self.reader.header.start_time_ns)
        if self._session_start_ns is None: self._session_start_ns = cur_start_ns
        elif cur_start_ns != self._session_start_ns:
          self._rotate_recording("header start_time_ns changed")
          self._session_start_ns = cur_start_ns
      wc_now = self.reader.write_count()
      if (self.replay_path is None and self.recorder is not None and self.recorder.started and wc_now < int(self.recorder.wc_last)):
        self._rotate_recording("write_count reset")
        self._session_start_ns = int(self.reader.header.start_time_ns) if self.reader.header is not None else None

      # Recording poll (captures all samples, not just last ring window)
      if self.recorder is not None: self.recorder.poll(self.reader)

      # If user is interacting (pan/zoom), skip heavy read_all + setData updates.
      # NOTE: This keeps recording/session detection working.
      if self._ui_paused_by_user:
        # Optional lightweight status line
        extra = ""
        if self.recorder is not None and self.recorder.started:
          extra = f" | rec_samples={int(self.recorder.n_samples_total)} | dropped={int(self.recorder.dropped_total)}"
        self.lbl_stat.setText(f"wc={int(wc_now)} | UI paused (pan/zoom){extra}")
        return

      # Viewer uses last ring window (fast, bounded)
      t, ch = self.reader.read_all()
      if t.size == 0:
        self.lbl_stat.setText("no data yet")
        return

      wc = int(ch.get("write_count", 0))
      if wc == self._last_wc_plotted: return
      self._last_wc_plotted = wc
      ch.pop("write_count", None)
      self._update_plots(t, ch, wc)

    except FileNotFoundError: self.lbl_stat.setText(f"waiting: {self.reader.path}")
    except Exception as e: self.lbl_stat.setText(f"error: {e}")

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
    QTabWidget { background: #ffffff; }
    QTabWidget::pane {
      background: #ffffff;
      border: 1px solid #cfcfcf;
      border-radius: 5px;
      top: -10px;
    }
    QTabBar { background: #ffffff; }
    QTabBar::tab {
      background: #ffffff;
      border: 1px solid #cfcfcf;
      border-bottom: 1px solid #cfcfcf;
      padding: 2px 6px;
      margin-right: 2px;
      border-top-left-radius: 5px;
      border-top-right-radius: 5px;
      border-bottom-left-radius: 5px;
      border-bottom-right-radius: 5px;
    }
    QTabBar::tab:selected {
      background: #1677ff;
      color: #ffffff;
      border-color: #1677ff;
      border-bottom-color: #ffffff;
    }
    QTabBar::tab:!selected {
      background: #f2f2f2;
      color: #111111;
      border-color: #c0c0c0;
    }
    QSplitter::handle {
      background: #e6e6e6;
      border: 1px solid #cfcfcf;
    }
    QSplitter::handle:vertical {
      height: 1px;
    }
    QSplitter::handle:hover {
      background: #d8d8d8;
    }
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
