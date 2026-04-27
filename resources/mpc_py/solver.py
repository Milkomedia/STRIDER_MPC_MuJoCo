from acados_template.acados_ocp_solver import AcadosOcpSolver

import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from .mmap_manager import MMapWriter

# Make codegen quieter.
os.environ.setdefault("MAKEFLAGS", "-s")

BASE_DIR = Path(__file__).resolve().parent

class StriderNMPC:
  def __init__(self):
    # Build USE-DELTA solver.
    from .use_delta.model import build_ocp as use_delta_build_ocp
    self.use_delta_ocp = use_delta_build_ocp()
    use_delta_json_path = BASE_DIR / "use_delta" / f"{self.use_delta_ocp.model.name}.json"
    self.use_delta_solver = AcadosOcpSolver(self.use_delta_ocp, json_file=str(use_delta_json_path))

    self.use_delta_nx = int(self.use_delta_ocp.model.x.size()[0])
    self.use_delta_nu = int(self.use_delta_ocp.model.u.size()[0])
    self.use_delta_np = int(self.use_delta_ocp.model.p.size()[0])

    # Build USE-ARM solver.
    from .use_arm.model import build_ocp as use_arm_build_ocp
    self.use_arm_ocp = use_arm_build_ocp()
    use_arm_json_path = BASE_DIR / "use_arm" / f"{self.use_arm_ocp.model.name}.json"
    self.use_arm_solver = AcadosOcpSolver(self.use_arm_ocp, json_file=str(use_arm_json_path))

    self.use_arm_nx = int(self.use_arm_ocp.model.x.size()[0])
    self.use_arm_nu = int(self.use_arm_ocp.model.u.size()[0])
    self.use_arm_np = int(self.use_arm_ocp.model.p.size()[0])

    # Build USE-FULL solver.
    from .use_full.model import build_ocp as use_full_build_ocp
    self.use_full_ocp = use_full_build_ocp()
    use_full_json_path = BASE_DIR / "use_full" / f"{self.use_full_ocp.model.name}.json"
    self.use_full_solver = AcadosOcpSolver(self.use_full_ocp, json_file=str(use_full_json_path))

    self.use_full_nx = int(self.use_full_ocp.model.x.size()[0])
    self.use_full_nu = int(self.use_full_ocp.model.u.size()[0])
    self.use_full_np = int(self.use_full_ocp.model.p.size()[0])

    from . import params as p
    self.N = int(p.N)

    mmap_path = os.environ.get("MRG_MMAP", "/tmp/MRG_debug.mmap")
    self._mmap_writer = MMapWriter(mmap_path, self.N, self.use_full_nx, self.use_full_nu, self.use_full_np)

    # Full-layout horizon buffers for mmap/debug output.
    self._xs_full = np.zeros((self.N + 1, self.use_full_nx), dtype=np.float64)
    self._us_full = np.zeros((self.N, self.use_full_nu), dtype=np.float64)
    self._ps_full = np.zeros((self.N + 1, self.use_full_np), dtype=np.float64)

    # Upcast buffers for reduced solvers.
    self._xs_up = np.zeros((self.N + 1, self.use_full_nx), dtype=np.float64)
    self._us_up = np.zeros((self.N, self.use_full_nu), dtype=np.float64)
    self._ps_up = np.zeros((self.N + 1, self.use_full_np), dtype=np.float64)

    # Returned full-layout stage-wise outputs.
    self._x_stage_steps = np.zeros((self.use_full_nx, self.N), dtype=np.float64, order="F")
    self._u_stage_steps = np.zeros((self.use_full_nu, self.N), dtype=np.float64, order="F")

    # Reduced-model initial-condition buffers.
    self._x0_delta = np.zeros((self.use_delta_nx,), dtype=np.float64)
    self._u0_delta = np.zeros((self.use_delta_nu,), dtype=np.float64)

    self._x0_arm = np.zeros((self.use_arm_nx,), dtype=np.float64)
    self._u0_arm = np.zeros((self.use_arm_nu,), dtype=np.float64)

    # Cached full-layout arm states for reduced solver upcasting.
    self._last_r_rotor = np.zeros(8, dtype=np.float64)
    self._last_r_rotor_cmd = np.zeros(8, dtype=np.float64)

  def _set_initial_guess_all_stages(self, solver: AcadosOcpSolver, x0: np.ndarray, u0: np.ndarray, p0: np.ndarray) -> None:
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    for k in range(self.N + 1):
      solver.set(k, "x", x0)
      solver.set(k, "p", p0)

    for k in range(self.N):
      solver.set(k, "u", u0)

  def use_delta_solve(self, x_0, u_0, p, steps_req: int):
    x_full = np.asarray(x_0, dtype=np.float64).ravel()
    u_full = np.asarray(u_0, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    # Full x layout:
    # [theta(0:3), omega(3:6), r_rotor(6:14)]
    # Full u layout:
    # [delta_theta_cmd(0:3), r_rotor_cmd(3:11)]
    self._last_r_rotor[:] = x_full[6:14]
    self._last_r_rotor_cmd[:] = u_full[3:11]

    # use_delta x = [theta(0:3), omega(3:6)]
    self._x0_delta[:] = x_full[0:6]

    # use_delta u = [delta_theta_cmd(0:3)]
    self._u0_delta[:] = u_full[0:3]

    self._set_initial_guess_all_stages(self.use_delta_solver, self._x0_delta, self._u0_delta, p)

    t0 = time.perf_counter()
    status = self.use_delta_solver.solve()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    xs, us, ps, x_stage, u_stage = self._extract_all_xup(full_model_using=False, arm_model_using=False, steps_req=steps_req)

    self._mmap_writer.write(x_all=xs, u_all=us, p_all=ps, solve_ms=float(solve_ms), status=int(status))

    return (
      x_stage[:, 0:steps_req],
      u_stage[:, 0:steps_req],
      float(solve_ms),
      int(status),
    )

  def use_arm_solve(self, x_0, u_0, p, steps_req: int):
    x_full = np.asarray(x_0, dtype=np.float64).ravel()
    u_full = np.asarray(u_0, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    # Full x layout:
    # [theta(0:3), omega(3:6), r_rotor(6:14)]

    # use_arm x = [theta(0:3), omega(3:6), r_rotor(6:14)]
    self._x0_arm[0:6] = x_full[0:6]
    self._x0_arm[6:14] = x_full[6:14]

    # use_arm u = [r_rotor_cmd(0:8)]
    self._u0_arm[:] = u_full[3:11]

    self._set_initial_guess_all_stages(self.use_arm_solver, self._x0_arm, self._u0_arm, p)

    t0 = time.perf_counter()
    status = self.use_arm_solver.solve()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    xs, us, ps, x_stage, u_stage = self._extract_all_xup(full_model_using=False, arm_model_using=True, steps_req=steps_req)

    self._mmap_writer.write(x_all=xs, u_all=us, p_all=ps, solve_ms=float(solve_ms), status=int(status))

    return (
      x_stage[:, 0:steps_req],
      u_stage[:, 0:steps_req],
      float(solve_ms),
      int(status),
    )

  def use_full_solve(self, x_0, u_0, p, steps_req: int):
    x_0 = np.asarray(x_0, dtype=np.float64).ravel()
    u_0 = np.asarray(u_0, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    self._set_initial_guess_all_stages(self.use_full_solver, x_0, u_0, p)

    t0 = time.perf_counter()
    status = self.use_full_solver.solve()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    xs, us, ps, x_stage, u_stage = self._extract_all_xup(full_model_using=True, arm_model_using=False, steps_req=steps_req)

    self._mmap_writer.write(x_all=xs, u_all=us, p_all=ps, solve_ms=float(solve_ms), status=int(status))

    return (
      x_stage[:, 0:steps_req],
      u_stage[:, 0:steps_req],
      float(solve_ms),
      int(status),
    )

  def compute_MPC(self, mpci: Dict[str, Any]) -> Dict[str, Any]:
    x_0 = np.asarray(mpci.get("x_0", np.zeros(self.use_full_nx)), dtype=np.float64).ravel()
    u_0 = np.asarray(mpci.get("u_0", np.zeros(self.use_full_nu)), dtype=np.float64).ravel()
    p = np.asarray(mpci.get("p", np.zeros(self.use_full_np)), dtype=np.float64).ravel()

    delta_using = bool(mpci.get("use_delta", False))
    arm_using = bool(mpci.get("use_arm", False))

    steps_req = int(mpci.get("steps_req", 1))
    if steps_req < 0 or steps_req > self.N:
      raise ValueError(f"steps_req out of range: got {steps_req}, valid=[0, {self.N}]")

    if x_0.size != self.use_full_nx:
      raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.use_full_nx}")
    if p.size != self.use_full_np:
      raise ValueError(f"p size mismatch: got {p.size}, expected {self.use_full_np}")
    if u_0.size != self.use_full_nu:
      u_0 = np.zeros(self.use_full_nu, dtype=np.float64)

    if delta_using and arm_using:
      x_stage, u_stage, solve_ms, status = self.use_full_solve(x_0, u_0, p, steps_req)
    elif (not delta_using) and arm_using:
      x_stage, u_stage, solve_ms, status = self.use_arm_solve(x_0, u_0, p, steps_req)
    elif delta_using and (not arm_using):
      x_stage, u_stage, solve_ms, status = self.use_delta_solve(x_0, u_0, p, steps_req)
    else:
      raise ValueError("At least one of 'use_delta' or 'use_arm' must be enabled.")

    return {
      "x_stage": x_stage,
      "u_stage": u_stage,
      "solve_ms": float(solve_ms),
      "state": int(status),
    }

  def _extract_all_xup(self, full_model_using: bool, arm_model_using: bool = False, steps_req: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = self.N

    self._x_stage_steps.fill(0.0)
    self._u_stage_steps.fill(0.0)

    if full_model_using:
      sol = self.use_full_solver
      xs = self._xs_full
      us = self._us_full
      ps = self._ps_full

      xs.fill(0.0)
      us.fill(0.0)
      ps.fill(0.0)

      for k in range(N + 1):
        xk = sol.get(k, "x").reshape(-1)
        pk = sol.get(k, "p").reshape(-1)

        xs[k, :] = xk
        ps[k, :] = pk

        if 1 <= k <= steps_req:
          self._x_stage_steps[:, k - 1] = xk

      for k in range(N):
        uk = sol.get(k, "u").reshape(-1)

        us[k, :] = uk

        if k < steps_req:
          self._u_stage_steps[:, k] = uk

      return xs, us, ps, self._x_stage_steps, self._u_stage_steps

    if arm_model_using:
      sol = self.use_arm_solver
      xs = self._xs_up
      us = self._us_up
      ps = self._ps_up

      xs.fill(0.0)
      us.fill(0.0)
      ps.fill(0.0)

      for k in range(N + 1):
        xk = sol.get(k, "x").reshape(-1)
        pk = sol.get(k, "p").reshape(-1)

        # Upcast use_arm x to full x:
        # use_arm x = [theta, omega, r_rotor]
        # full x    = [theta, omega, r_rotor]
        xs[k, :] = xk

        m = min(pk.size, self.use_full_np)
        ps[k, 0:m] = pk[0:m]

        if 1 <= k <= steps_req:
          self._x_stage_steps[:, k - 1] = xs[k, :]

      for k in range(N):
        uk = sol.get(k, "u").reshape(-1)

        # Upcast use_arm u to full u:
        # use_arm u = [r_rotor_cmd]
        # full u    = [delta_theta_cmd, r_rotor_cmd]
        us[k, 0:3] = 0.0
        us[k, 3:11] = uk

        if k < steps_req:
          self._u_stage_steps[:, k] = us[k, :]

      return xs, us, ps, self._x_stage_steps, self._u_stage_steps

    sol = self.use_delta_solver
    xs = self._xs_up
    us = self._us_up
    ps = self._ps_up

    xs.fill(0.0)
    us.fill(0.0)
    ps.fill(0.0)

    # Hold arm-related states constant across the horizon.
    xs[:, 6:14] = self._last_r_rotor.reshape(1, 8)

    for k in range(N + 1):
      xk = sol.get(k, "x").reshape(-1)
      pk = sol.get(k, "p").reshape(-1)

      # Upcast use_delta x to full x:
      # use_delta x = [theta, omega]
      # full x      = [theta, omega, r_rotor]
      xs[k, 0:6] = xk[0:6]

      m = min(pk.size, self.use_full_np)
      ps[k, 0:m] = pk[0:m]

      if 1 <= k <= steps_req:
        self._x_stage_steps[:, k - 1] = xs[k, :]

    for k in range(N):
      uk = sol.get(k, "u").reshape(-1)

      # Upcast use_delta u to full u:
      # use_delta u = [delta_theta_cmd]
      # full u      = [delta_theta_cmd, r_rotor_cmd]
      us[k, 0:3] = uk
      us[k, 3:11] = self._last_r_rotor_cmd

      if k < steps_req:
        self._u_stage_steps[:, k] = us[k, :]

    return xs, us, ps, self._x_stage_steps, self._u_stage_steps