from acados_template.acados_ocp_solver import AcadosOcpSolver

import numpy as np
from typing import Dict, Any
import time

# debugging
import os
from .mmap_manager import MMapWriter

# make silent (when model build)
os.environ.setdefault("MAKEFLAGS", "-s")

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

class StriderNMPC:
    def __init__(self):
        # 1. build "NO-COT"
        from .no_cot.model import build_ocp as no_cot_build_ocp
        self.no_cot_ocp  = no_cot_build_ocp()
        no_cot_json_path  = BASE_DIR / "no_cot" / f"{self.no_cot_ocp.model.name}.json"
        self.no_cot_solver  = AcadosOcpSolver(self.no_cot_ocp, json_file=str(no_cot_json_path))

        self.no_cot_nx = self.no_cot_ocp.model.x.size()[0]
        self.no_cot_nu = self.no_cot_ocp.model.u.size()[0]
        self.no_cot_np = self.no_cot_ocp.model.p.size()[0]

        # 2. build "YES-COT"
        from .yes_cot.model import build_ocp as yes_cot_build_ocp
        self.yes_cot_ocp = yes_cot_build_ocp()
        yes_cot_json_path = BASE_DIR / "yes_cot"  / f"{self.yes_cot_ocp.model.name}.json"
        self.yes_cot_solver = AcadosOcpSolver(self.yes_cot_ocp, json_file=str(yes_cot_json_path))

        self.yes_cot_nx = self.yes_cot_ocp.model.x.size()[0]
        self.yes_cot_nu = self.yes_cot_ocp.model.u.size()[0]
        self.yes_cot_np = self.yes_cot_ocp.model.p.size()[0]

        # 3. generate mmap logger
        from . import params as p
        self.N = int(p.N)

        mmap_path = os.environ.get("MRG_MMAP", "/tmp/MRG_debug.mmap")
        self._mmap_writer = MMapWriter(mmap_path, self.N, self.yes_cot_nx, self.yes_cot_nu, self.yes_cot_np)

        # 4. RT-oriented buffers (our glue code by reusing fixed-size buffers per tick)

        # Full-horizon debug buffers for mmap (C-contiguous)
        self._xs_yes = np.empty((self.N + 1, self.yes_cot_nx), dtype=np.float64)
        self._us_yes = np.empty((self.N,     self.yes_cot_nu), dtype=np.float64)
        self._ps_yes = np.empty((self.N + 1, self.yes_cot_np), dtype=np.float64)

        # no_cot results upcasted to yes_cot dims for mmap (C-contiguous)
        self._xs_up  = np.empty((self.N + 1, self.yes_cot_nx), dtype=np.float64)
        self._us_up  = np.zeros((self.N,     self.yes_cot_nu), dtype=np.float64)
        self._ps_up  = np.empty((self.N + 1, self.yes_cot_np), dtype=np.float64)

        # Per-call step outputs (Fortran-order helps Eigen column-major mapping)
        self._u_opt_steps  = np.empty((self.yes_cot_nu, self.N), dtype=np.float64, order="F")
        self._u_rate_steps = np.empty((self.yes_cot_nu, self.N), dtype=np.float64, order="F")

        # no_cot down-projection buffers (avoid np.concatenate/copy every tick).
        self._x0_no = np.empty((self.no_cot_nx,), dtype=np.float64)
        self._u0_no = np.empty((self.no_cot_nu,), dtype=np.float64)
        
        # no_cot param buffer (for size mismatch safety)
        self._p_no  = np.empty((self.no_cot_np,), dtype=np.float64)

        # Cached holds for no_cot upcast (avoid per-tick small allocations).
        self._last_r_rotor     = np.zeros(8, dtype=np.float64)
        self._last_r_rotor_cmd = np.zeros(8, dtype=np.float64)
    
    def no_cot_solve(self, x_0, u_0, p, steps_req: int):
        x_full = np.asarray(x_0, dtype=np.float64).ravel()
        u_full = np.asarray(u_0, dtype=np.float64).ravel()

        # Cache holds for no_cot -> yes_cot upcast
        self._last_r_rotor[:]     = x_full[6:14]
        self._last_r_rotor_cmd[:] = x_full[17:25]

        # Down-project yes_cot packet -> no_cot dimensions
        # no_cot x: [theta(0:3), omega(3:6), delta_theta_cmd(6:9)]
        self._x0_no[0:6] = x_full[0:6]
        self._x0_no[6:9] = x_full[14:17]
        # no_cot u: [delta_theta_cmd_rate(0:3)]
        self._u0_no[0:3] = u_full[0:3]
        p = np.asarray(p, dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.no_cot_solver.set(0, "lbx", self._x0_no)
        self.no_cot_solver.set(0, "ubx", self._x0_no)

        for k in range(self.N + 1):
            self.no_cot_solver.set(k, "x", self._x0_no)  # initial guess
            self.no_cot_solver.set(k, "p", p)            # feed parameter

        for k in range(self.N):
            self.no_cot_solver.set(k, "u", self._u0_no)  # initial guess

        # Solve
        t0 = time.perf_counter()
        status = self.no_cot_solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        
        # Extract full-horizon (for mmap) and requested steps in one pass.
        xs, us, ps = self._extract_all_xup(
            cot_using=False,
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

    def yes_cot_solve(self, x_0, u_0, p, steps_req: int):
        x_0   = np.asarray(x_0,   dtype=np.float64).ravel()
        u_0   = np.asarray(u_0,   dtype=np.float64).ravel()
        p     = np.asarray(p,     dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.yes_cot_solver.set(0, "lbx", x_0)
        self.yes_cot_solver.set(0, "ubx", x_0)

        for k in range(self.N + 1):
            self.yes_cot_solver.set(k, "x", x_0)  # initial guess
            self.yes_cot_solver.set(k, "p", p)    # feed parameter

        for k in range(self.N):
            self.yes_cot_solver.set(k, "u", u_0)  # initial guess

        # Solve
        t0 = time.perf_counter()
        status = self.yes_cot_solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract full-horizon (for mmap) and requested steps in one pass.
        xs, us, ps = self._extract_all_xup(
            cot_using=True,
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
        x_0    = np.asarray(mpci.get("x_0", np.zeros(self.yes_cot_nx)), dtype=np.float64).ravel()
        u_0    = np.asarray(mpci.get("u_0", np.zeros(self.yes_cot_nu)), dtype=np.float64).ravel()
        p      = np.asarray(mpci.get("p",   np.zeros(self.yes_cot_np)), dtype=np.float64).ravel()
        cot_using = bool(mpci.get("use_cot", False))

        steps_req = int(mpci.get("steps_req", 1))
        if steps_req > self.N: raise ValueError(f"steps_req too large: got {steps_req}, solver horizon N={self.N}")

        if x_0.size != self.yes_cot_nx: raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.yes_cot_nx}")
        if p.size != self.yes_cot_np: raise ValueError(f"p size mismatch: got {p.size}, expected {self.yes_cot_np}")
        if u_0.size != self.yes_cot_nu: u_0 = np.zeros(self.yes_cot_nu, dtype=np.float64)

        if cot_using: u_opt_steps, u_rate_steps, solve_ms, status = self.yes_cot_solve(x_0, u_0, p, steps_req)
        else: u_opt_steps, u_rate_steps, solve_ms, status = self.no_cot_solve(x_0, u_0, p, steps_req)

        return {"u_opt": u_opt_steps, "u_rate": u_rate_steps, "solve_ms": float(solve_ms), "state": int(status),}

    def _extract_all_xup(self, cot_using: bool, steps_req: int = 0, u_opt_out: np.ndarray | None = None, u_rate_out: np.ndarray | None = None,):
        N = self.N

        if cot_using:
            nx, nu, np_ = self.yes_cot_nx, self.yes_cot_nu, self.yes_cot_np
            sol = self.yes_cot_solver

            xs = self._xs_yes
            us = self._us_yes
            ps = self._ps_yes

            # x/p: (N+1) stages.
            # u_opt at step i is stored in augmented command states:
            #   delta_theta_cmd: x[14:17]
            #   r_rotor_cmd    : x[17:25]
            for k in range(N + 1):
                xk = sol.get(k, "x").reshape(-1)
                pk = sol.get(k, "p").reshape(-1)
                xs[k, :] = xk
                ps[k, :] = pk
                if u_opt_out is not None and steps_req > 0 and (1 <= k <= steps_req):
                    u_opt_out[0:3,  k - 1] = xk[14:17]
                    u_opt_out[3:11, k - 1] = xk[17:25]

            # u: (N) stages. u_rate at step i is u_i.
            for k in range(N):
                uk = sol.get(k, "u").reshape(-1)
                us[k, :] = uk
                if u_rate_out is not None and steps_req > 0 and (k < steps_req):
                    u_rate_out[:, k] = uk

            return xs, us, ps

        # ---------------- no_cot -> upcast to yes_cot dims ----------------
        sol = self.no_cot_solver

        xs = self._xs_up
        us = self._us_up
        ps = self._ps_up

        # Fill holds once (broadcast).
        # no_cot doesn't estimate rotor positions, so hold the last known values from input packet.
        xs[:, 6:14]   = self._last_r_rotor
        xs[:, 17:25]  = self._last_r_rotor_cmd

        # Pull each stage once, map into upcast arrays
        for k in range(N + 1):
            xk = sol.get(k, "x").reshape(-1)  # (9,)
            pk = sol.get(k, "p").reshape(-1)  # (14,)

            # no_cot x: [theta(0:3), omega(3:6), delta_theta_cmd(6:9)]
            xs[k, 0:6]  = xk[0:6]
            # yes_cot delta_theta_cmd is at [14:17]
            xs[k, 14:17] = xk[6:9]
            # Map p -> yes_cot_np for mmap (truncate/pad with zeros)
            ps[k, :].fill(0.0)
            m = min(pk.size, self.yes_cot_np)
            ps[k, 0:m] = pk[0:m]

            # u_opt at step i: [delta_theta_cmd(3), r_rotor_cmd(8)]
            if u_opt_out is not None and steps_req > 0 and (1 <= k <= steps_req):
                u_opt_out[0:3,  k - 1] = xk[6:9]
                u_opt_out[3:11, k - 1] = self._last_r_rotor_cmd

        for k in range(N):
            uk = sol.get(k, "u").reshape(-1)  # (3,)
            us[k, 0:3] = uk

            # u_rate at step i is u_i. Upcast to (11,) with last 8 = 0.
            if u_rate_out is not None and steps_req > 0 and (k < steps_req):
                u_rate_out[0:3, k] = uk
                u_rate_out[3:11, k] = 0.0

        return xs, us, ps
