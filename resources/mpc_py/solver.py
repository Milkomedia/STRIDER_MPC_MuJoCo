from acados_template.acados_ocp_solver import AcadosOcpSolver

import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .mmap_manager import MMapWriter

# Make codegen quieter.
os.environ.setdefault("MAKEFLAGS", "-s")

BASE_DIR = Path(__file__).resolve().parent


class StriderNMPC:
  def __init__(self):
    # 1. build "USE-DELTA"
    from .use_delta.model import build_ocp as use_delta_build_ocp
    self.use_delta_ocp = use_delta_build_ocp()
    use_delta_json_path = BASE_DIR / "use_delta" / f"{self.use_delta_ocp.model.name}.json"
    self.use_delta_solver = AcadosOcpSolver(self.use_delta_ocp, json_file=str(use_delta_json_path))

    self.use_delta_nx = self.use_delta_ocp.model.x.size()[0]
    self.use_delta_nu = self.use_delta_ocp.model.u.size()[0]
    self.use_delta_np = self.use_delta_ocp.model.p.size()[0]

    # 2. build "USE-ARM"
    from .use_arm.model import build_ocp as use_arm_build_ocp
    self.use_arm_ocp = use_arm_build_ocp()
    use_arm_json_path = BASE_DIR / "use_arm" / f"{self.use_arm_ocp.model.name}.json"
    self.use_arm_solver = AcadosOcpSolver(self.use_arm_ocp, json_file=str(use_arm_json_path))

    self.use_arm_nx = self.use_arm_ocp.model.x.size()[0]
    self.use_arm_nu = self.use_arm_ocp.model.u.size()[0]
    self.use_arm_np = self.use_arm_ocp.model.p.size()[0]

    # 3. build "USE-FULL"
    from .use_full.model import build_ocp as use_full_build_ocp
    self.use_full_ocp = use_full_build_ocp()
    use_full_json_path = BASE_DIR / "use_full" / f"{self.use_full_ocp.model.name}.json"
    self.use_full_solver = AcadosOcpSolver(self.use_full_ocp, json_file=str(use_full_json_path))

    self.use_full_nx = self.use_full_ocp.model.x.size()[0]
    self.use_full_nu = self.use_full_ocp.model.u.size()[0]
    self.use_full_np = self.use_full_ocp.model.p.size()[0]

    # 4. generate mmap logger using full-layout dims only
    from . import params as p
    self.N = int(p.N)

    mmap_path = os.environ.get("MRG_MMAP", "/tmp/MRG_debug.mmap")
    self._mmap_writer = MMapWriter(
      mmap_path,
      self.N,
      self.use_full_nx,
      self.use_full_nu,
      self.use_full_np,
    )

    # 5. RT-oriented buffers

    # Full-model buffers
    self._xs_full = np.zeros((self.N + 1, self.use_full_nx), dtype=np.float64)
    self._us_full = np.zeros((self.N,     self.use_full_nu), dtype=np.float64)
    self._ps_full = np.zeros((self.N + 1, self.use_full_np), dtype=np.float64)

    # Reduced-model upcast buffers
    self._xs_up = np.zeros((self.N + 1, self.use_full_nx), dtype=np.float64)
    self._us_up = np.zeros((self.N,     self.use_full_nu), dtype=np.float64)
    self._ps_up = np.zeros((self.N + 1, self.use_full_np), dtype=np.float64)

    # Returned step-wise outputs
    self._u_opt_steps = np.zeros((self.use_full_nu, self.N), dtype=np.float64, order="F")
    self._u_rate_steps = np.zeros((self.use_full_nu, self.N), dtype=np.float64, order="F")

    # Down-projection buffers
    self._x0_delta = np.zeros((self.use_delta_nx,), dtype=np.float64)
    self._u0_delta = np.zeros((self.use_delta_nu,), dtype=np.float64)

    self._x0_arm = np.zeros((self.use_arm_nx,), dtype=np.float64)
    self._u0_arm = np.zeros((self.use_arm_nu,), dtype=np.float64)

    # Cached rotor states used when upcasting use_delta results
    self._last_r_rotor = np.zeros(8, dtype=np.float64)
    self._last_r_rotor_cmd = np.zeros(8, dtype=np.float64)

  def _set_initial_guess_all_stages(self, solver, x0: np.ndarray, u0: np.ndarray, p0: np.ndarray) -> None:
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    for k in range(self.N + 1):
      solver.set(k, "x", x0)
      solver.set(k, "p", p0)

    for k in range(self.N):
      solver.set(k, "u", u0)

  def _reset_output_steps(self, steps_req: int, u_opt_out: Optional[np.ndarray], u_rate_out: Optional[np.ndarray]) -> None:
    if u_opt_out is not None:
      u_opt_out.fill(0.0)
    if u_rate_out is not None:
      u_rate_out.fill(0.0)

    if steps_req < 0 or steps_req > self.N:
      raise ValueError(f"steps_req out of range: got {steps_req}, valid=[0, {self.N}]")

  def use_delta_solve(self, x_0, u_0, p, steps_req: int):
    x_full = np.asarray(x_0, dtype=np.float64).ravel()
    u_full = np.asarray(u_0, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    # Cache rotor states from the current full packet.
    self._last_r_rotor[:] = x_full[6:14]
    self._last_r_rotor_cmd[:] = x_full[17:25]

    # use_delta x = [theta(0:3), omega(3:6), delta_theta_cmd(6:9)]
    self._x0_delta[0:6] = x_full[0:6]
    self._x0_delta[6:9] = x_full[14:17]

    # use_delta u = [delta_theta_cmd_rate(0:3)]
    self._u0_delta[0:3] = u_full[0:3]

    self._set_initial_guess_all_stages(self.use_delta_solver, self._x0_delta, self._u0_delta, p)

    t0 = time.perf_counter()
    status = self.use_delta_solver.solve()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    xs, us, ps = self._extract_all_xup(
      full_model_using=False,
      arm_model_using=False,
      steps_req=steps_req,
      u_opt_out=self._u_opt_steps,
      u_rate_out=self._u_rate_steps,
    )

    self._mmap_writer.write(
      x_all=xs,
      u_all=us,
      p_all=ps,
      solve_ms=float(solve_ms),
      status=int(status),
    )

    return self._u_opt_steps[:, 0:steps_req], self._u_rate_steps[:, 0:steps_req], float(solve_ms), int(status)

  def use_arm_solve(self, x_0, u_0, p, steps_req: int):
    x_full = np.asarray(x_0, dtype=np.float64).ravel()
    u_full = np.asarray(u_0, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    # use_arm x = [theta(0:3), omega(3:6), r_rotor(6:14), r_rotor_cmd(14:22)]
    self._x0_arm[0:6] = x_full[0:6]
    self._x0_arm[6:14] = x_full[6:14]
    self._x0_arm[14:22] = x_full[17:25]

    # use_arm u = full u[3:11]
    self._u0_arm[0:8] = u_full[3:11]

    self._set_initial_guess_all_stages(self.use_arm_solver, self._x0_arm, self._u0_arm, p)

    t0 = time.perf_counter()
    status = self.use_arm_solver.solve()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    xs, us, ps = self._extract_all_xup(
      full_model_using=False,
      arm_model_using=True,
      steps_req=steps_req,
      u_opt_out=self._u_opt_steps,
      u_rate_out=self._u_rate_steps,
    )

    self._mmap_writer.write(
      x_all=xs,
      u_all=us,
      p_all=ps,
      solve_ms=float(solve_ms),
      status=int(status),
    )

    return self._u_opt_steps[:, 0:steps_req], self._u_rate_steps[:, 0:steps_req], float(solve_ms), int(status)

  def use_full_solve(self, x_0, u_0, p, steps_req: int):
    x_0 = np.asarray(x_0, dtype=np.float64).ravel()
    u_0 = np.asarray(u_0, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    self._set_initial_guess_all_stages(self.use_full_solver, x_0, u_0, p)

    t0 = time.perf_counter()
    status = self.use_full_solver.solve()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    xs, us, ps = self._extract_all_xup(
      full_model_using=True,
      arm_model_using=False,
      steps_req=steps_req,
      u_opt_out=self._u_opt_steps,
      u_rate_out=self._u_rate_steps,
    )

    self._mmap_writer.write(
      x_all=xs,
      u_all=us,
      p_all=ps,
      solve_ms=float(solve_ms),
      status=int(status),
    )

    return self._u_opt_steps[:, 0:steps_req], self._u_rate_steps[:, 0:steps_req], float(solve_ms), int(status)

  def compute_MPC(self, mpci: Dict[str, Any]) -> Dict[str, Any]:
    x_0 = np.asarray(mpci.get("x_0", np.zeros(self.use_full_nx)), dtype=np.float64).ravel()
    u_0 = np.asarray(mpci.get("u_0", np.zeros(self.use_full_nu)), dtype=np.float64).ravel()
    p = np.asarray(mpci.get("p", np.zeros(self.use_full_np)), dtype=np.float64).ravel()

    delta_using = bool(mpci.get("use_delta", False))
    arm_using = bool(mpci.get("use_arm", False))

    steps_req = int(mpci.get("steps_req", 1))
    if steps_req > self.N:
      raise ValueError(f"steps_req too large: got {steps_req}, solver horizon N={self.N}")

    if x_0.size != self.use_full_nx:
      raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.use_full_nx}")
    if p.size != self.use_full_np:
      raise ValueError(f"p size mismatch: got {p.size}, expected {self.use_full_np}")
    if u_0.size != self.use_full_nu:
      u_0 = np.zeros(self.use_full_nu, dtype=np.float64)

    if delta_using and arm_using:
      u_opt_steps, u_rate_steps, solve_ms, status = self.use_full_solve(x_0, u_0, p, steps_req)
    elif (not delta_using) and arm_using:
      u_opt_steps, u_rate_steps, solve_ms, status = self.use_arm_solve(x_0, u_0, p, steps_req)
    elif delta_using and (not arm_using):
      u_opt_steps, u_rate_steps, solve_ms, status = self.use_delta_solve(x_0, u_0, p, steps_req)
    else:
      raise ValueError("At least one of 'use_delta' or 'use_arm' must be enabled.")

    return {
      "u_opt": u_opt_steps,
      "u_rate": u_rate_steps,
      "solve_ms": float(solve_ms),
      "state": int(status),
    }

  def _extract_all_xup(
    self,
    full_model_using: bool,
    arm_model_using: bool = False,
    steps_req: int = 0,
    u_opt_out: Optional[np.ndarray] = None,
    u_rate_out: Optional[np.ndarray] = None,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = self.N
    self._reset_output_steps(steps_req, u_opt_out, u_rate_out)

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

        if u_opt_out is not None and 1 <= k <= steps_req:
          u_opt_out[0:3,  k - 1] = xk[14:17]
          u_opt_out[3:11, k - 1] = xk[17:25]

      for k in range(N):
        uk = sol.get(k, "u").reshape(-1)
        us[k, :] = uk

        if u_rate_out is not None and k < steps_req:
          u_rate_out[:, k] = uk

      return xs, us, ps

    if arm_model_using:
      sol = self.use_arm_solver
      xs = self._xs_up
      us = self._us_up
      ps = self._ps_up

      xs.fill(0.0)
      us.fill(0.0)
      ps.fill(0.0)

      for k in range(N + 1):
        xk = sol.get(k, "x").reshape(-1)   # (22,)
        pk = sol.get(k, "p").reshape(-1)

        # upcast to full x:
        # [theta, omega, r_rotor, delta_theta_cmd=0, r_rotor_cmd]
        xs[k, 0:14] = xk[0:14]
        xs[k, 14:17] = 0.0
        xs[k, 17:25] = xk[14:22]

        m = min(pk.size, self.use_full_np)
        ps[k, 0:m] = pk[0:m]

        if u_opt_out is not None and 1 <= k <= steps_req:
          u_opt_out[0:3,  k - 1] = 0.0
          u_opt_out[3:11, k - 1] = xk[14:22]

      for k in range(N):
        uk = sol.get(k, "u").reshape(-1)   # (8,)

        # upcast to full u:
        # [delta_theta_cmd_rate=0, r_rotor_cmd_rate]
        us[k, 0:3] = 0.0
        us[k, 3:11] = uk

        if u_rate_out is not None and k < steps_req:
          u_rate_out[0:3, k] = 0.0
          u_rate_out[3:11, k] = uk

      return xs, us, ps

    # use_delta -> upcast to full layout
    sol = self.use_delta_solver
    xs = self._xs_up
    us = self._us_up
    ps = self._ps_up

    xs.fill(0.0)
    us.fill(0.0)
    ps.fill(0.0)

    # Hold rotor-related states constant across the horizon.
    xs[:, 6:14] = self._last_r_rotor.reshape(1, 8)
    xs[:, 17:25] = self._last_r_rotor_cmd.reshape(1, 8)

    for k in range(N + 1):
      xk = sol.get(k, "x").reshape(-1)     # (9,)
      pk = sol.get(k, "p").reshape(-1)

      # upcast to full x:
      # [theta, omega, r_rotor=hold, delta_theta_cmd, r_rotor_cmd=hold]
      xs[k, 0:6] = xk[0:6]
      xs[k, 14:17] = xk[6:9]

      m = min(pk.size, self.use_full_np)
      ps[k, 0:m] = pk[0:m]

      if u_opt_out is not None and 1 <= k <= steps_req:
        u_opt_out[0:3,  k - 1] = xk[6:9]
        u_opt_out[3:11, k - 1] = self._last_r_rotor_cmd

    for k in range(N):
      uk = sol.get(k, "u").reshape(-1)     # (3,)

      # upcast to full u:
      # [delta_theta_cmd_rate, r_rotor_cmd_rate=0]
      us[k, 0:3] = uk
      us[k, 3:11] = 0.0

      if u_rate_out is not None and k < steps_req:
        u_rate_out[0:3, k] = uk
        u_rate_out[3:11, k] = 0.0

    return xs, us, ps