from __future__ import annotations

import os
import mmap
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_MAGIC = b"STRIDER\x00"  # 8 bytes

# Header layout (little-endian):
# magic(8s), seq(u64),
# N(i32), nx(i32), nu(i32), np(i32), nlog(i32), status(i32), solve_ms(f64),
# payload_bytes(u64)
_HDR_STRUCT = struct.Struct("<8sQiiiiiidQ")
_HDR_SIZE = _HDR_STRUCT.size

def _payload_bytes(N: int, nx: int, nu: int, np_: int,  nlog: int) -> int:
  # Payload order:
  # x: (N+1,nx), u: (N,nu), p: (N+1,np)
  n_x = (N + 1) * nx
  n_u = N * nu
  n_p = (N + 1) * np_
  return 8 * (n_x + n_u + n_p + nlog)

def _offsets(N: int, nx: int, nu: int, np_: int) -> Tuple[int, int, int, int]:
  # Returns offsets (bytes) within mmap for each block.
  # [HDR][x][u][p]
  off_x = _HDR_SIZE
  off_u = off_x + 8 * ((N + 1) * nx)
  off_p = off_u + 8 * (N * nu)
  off_log = off_p + 8 * ((N + 1) * np_)
  return off_x, off_u, off_p, off_log


class MMapWriter:
  def __init__(self, path: str, N: int, nx: int, nu: int, np_: int, nlog_: int):
    self.path = path
    self.N = int(N)
    self.nx = int(nx)
    self.nu = int(nu)
    self.np = int(np_)
    self.nlog = int(nlog_)
    self._seq_even = 0

    pay = _payload_bytes(self.N, self.nx, self.nu, self.np, self.nlog)
    self._total_size = _HDR_SIZE + pay

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
      os.ftruncate(fd, self._total_size)
      self._mm = mmap.mmap(fd, self._total_size, access=mmap.ACCESS_WRITE)
    finally:
      os.close(fd)

    # Initialize header once (seq=0, status=0, solve_ms=0.0).
    _HDR_STRUCT.pack_into(
      self._mm, 0,
      _MAGIC,
      0,  # seq
      self.N, self.nx, self.nu, self.np, self.nlog,
      0,      # status
      0.0,    # solve_ms
      pay,    # payload_bytes
    )

  def close(self) -> None:
    try:
      self._mm.flush()
      self._mm.close()
    except Exception:
      pass

  def write(
    self,
    x_all: np.ndarray,   # (N+1,nx) float64 C-contig
    u_all: np.ndarray,   # (N,nu) float64 C-contig
    p_all: np.ndarray,   # (N+1,np) float64 C-contig
    log_param: np.ndarray,
    solve_ms: float,
    status: int,
  ) -> None:
    N = self.N
    
    off_x, off_u, off_p, off_log = _offsets(self.N, self.nx, self.nu, self.np)

    # ---- seqlock begin: set seq odd first ----
    seq_begin = self._seq_even + 1  # odd
    struct.pack_into("<Q", self._mm, 8, int(seq_begin))  # seq offset = 8

    # Update only fields that change per write: status, solve_ms.
    # status offset = 36, solve_ms offset = 40 (see layout).
    struct.pack_into("<id", self._mm, 36, int(status), float(solve_ms))

    # ---- payload write ----
    mv = memoryview(self._mm)
    mv[off_x:   off_x   + x_all.nbytes] = x_all.view(np.uint8).reshape(-1)
    mv[off_u:   off_u   + u_all.nbytes] = u_all.view(np.uint8).reshape(-1)
    mv[off_p:   off_p   + p_all.nbytes] = p_all.view(np.uint8).reshape(-1)
    mv[off_log: off_log + log_param.nbytes]   = log_param.view(np.uint8).reshape(-1)

    # ---- seqlock commit: set seq even ----
    seq_end = seq_begin + 1
    struct.pack_into("<Q", self._mm, 8, int(seq_end))
    self._seq_even = seq_end

@dataclass
class MMapPacket:
  seq: int
  N: int
  nx: int
  nu: int
  np: int
  status: int
  solve_ms: float
  x_all: np.ndarray  # (N+1,nx) view into mmap
  u_all: np.ndarray  # (N,nu) view into mmap
  p_all: np.ndarray  # (N+1,np) view into mmap
  log_param: np.ndarray

class MMapReader:
  
  def __init__(self, path: str):
    self.path = path
    fd = os.open(path, os.O_RDONLY)
    try:
      st = os.fstat(fd)
      self._size = st.st_size
      self._mm = mmap.mmap(fd, self._size, access=mmap.ACCESS_READ)
    finally:
      os.close(fd)

    self._last_seq = -1

  def close(self) -> None:
    try:
      self._mm.close()
    except Exception:
      pass

  def read_latest(self) -> Optional[MMapPacket]:
    if self._size < _HDR_SIZE: return None

    seq0 = struct.unpack_from("<Q", self._mm, 8)[0]
    if (seq0 & 1) == 1: return None
    if seq0 == self._last_seq: return None

    if self._mm[0:8] != _MAGIC: return None

    # Read remaining header fields
    N, nx, nu, np_, nlog, status, solve_ms, pay = struct.unpack_from("<iiiiiidQ", self._mm, 16)

    # Compute offsets and create views into mmap.
    off_x, off_u, off_p, off_log = _offsets(N, nx, nu, np_)

    x_all = np.frombuffer(self._mm, dtype=np.float64, count=(N + 1) * nx, offset=off_x).reshape(N + 1, nx)
    u_all = np.frombuffer(self._mm, dtype=np.float64, count=N * nu, offset=off_u).reshape(N, nu)
    p_all = np.frombuffer(self._mm, dtype=np.float64, count=(N + 1) * np_, offset=off_p).reshape(N + 1, np_)
    log_param = np.frombuffer(self._mm, dtype=np.float64, count=int(nlog), offset=off_log)

    # Re-check seq to ensure a stable snapshot at read time.
    seq1 = struct.unpack_from("<Q", self._mm, 8)[0]
    if seq1 != seq0 or (seq1 & 1) == 1: return None

    self._last_seq = int(seq1)
    return MMapPacket(
      seq=int(seq1),
      N=int(N), nx=int(nx), nu=int(nu), np=int(np_),
      status=int(status),
      solve_ms=float(solve_ms),
      x_all=x_all, u_all=u_all, p_all=p_all,
      log_param=log_param,
    )
