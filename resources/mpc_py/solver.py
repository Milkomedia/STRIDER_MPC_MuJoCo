from acados_template.acados_ocp_solver import AcadosOcpSolver

import numpy as np
from typing import Dict, Any
import time

# debugging
import os
from .mmap_manager import MMapWriter

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

class StriderNMPC:
    def __init__(self):
        from .no_cot.model import build_ocp as no_cot_build_ocp
        self.no_cot_ocp  = no_cot_build_ocp()
        no_cot_json_path  = BASE_DIR / "no_cot" / f"{self.no_cot_ocp.model.name}.json"
        self.no_cot_solver  = AcadosOcpSolver(self.no_cot_ocp, json_file=str(no_cot_json_path))

        self.no_cot_nx = self.no_cot_ocp.model.x.size()[0]
        self.no_cot_nu = self.no_cot_ocp.model.u.size()[0]
        self.no_cot_np = self.no_cot_ocp.model.p.size()[0]

        from .yes_cot.model import build_ocp as yes_cot_build_ocp
        self.yes_cot_ocp = yes_cot_build_ocp()
        yes_cot_json_path = BASE_DIR / "yes_cot"  / f"{self.yes_cot_ocp.model.name}.json"
        self.yes_cot_solver = AcadosOcpSolver(self.yes_cot_ocp, json_file=str(yes_cot_json_path))

        self.yes_cot_nx = self.yes_cot_ocp.model.x.size()[0]
        self.yes_cot_nu = self.yes_cot_ocp.model.u.size()[0]
        self.yes_cot_np = self.yes_cot_ocp.model.p.size()[0]

        from . import params as p
        self.N = int(p.N)

        mmap_path = os.environ.get("MRG_MMAP", "/tmp/MRG_debug.mmap")
        self._mmap_writer = MMapWriter(mmap_path, self.N, self.yes_cot_nx, self.yes_cot_nu, self.yes_cot_np)
    
    def no_cot_solve(self, x_0, u_0, p):
        x_full = np.asarray(x_0, dtype=np.float64).ravel()
        u_full = np.asarray(u_0, dtype=np.float64).ravel()

        r_cot_in     = x_full[6:8].copy()
        r_cot_cmd_in = x_full[11:13].copy()

        # cache for _extract_all_xup up-cast
        self._last_r_cot     = r_cot_in
        self._last_r_cot_cmd = r_cot_cmd_in

        # down-project yes_cot packet -> no_cot dimensions
        x_0 = np.concatenate([x_full[0:6], x_full[8:11]], axis=0)
        u_0 = u_full[0:3].copy()
        p      = np.asarray(p,   dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.no_cot_solver.set(0, "lbx", x_0)
        self.no_cot_solver.set(0, "ubx", x_0)

        for k in range(self.N + 1):
            self.no_cot_solver.set(k, "x", x_0)  # initial guess
            self.no_cot_solver.set(k, "p", p)    # feed parameter

        for k in range(self.N):
            self.no_cot_solver.set(k, "u", u_0)  # initial guess

        # Solve
        t0 = time.perf_counter()
        status = self.no_cot_solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract augmented state (u_cmd) to apply + first control (u_rate)
        if status == 0:
            u_rate1 = self.no_cot_solver.get(0, "u").reshape(-1).copy()
            x_opt1  = self.no_cot_solver.get(1, "x").reshape(-1)
            u_opt1  = x_opt1[6:9].copy()
        else:
            u_rate1 = u_0.copy()
            u_opt1  = x_0[6:9].copy()

        # logging
        xs, us, ps = self._extract_all_xup(cot_using=False)
        self._mmap_writer.write(
            x_all=xs,
            u_all=us,
            p_all=ps,
            solve_ms=float(solve_ms),
            status=int(status),
        )

        # Reconstruct
        u_opt_full  = np.zeros(self.yes_cot_nu, dtype=np.float64)
        u_rate_full = np.zeros(self.yes_cot_nu, dtype=np.float64)

        u_opt_full[0:3]  = u_opt1
        u_opt_full[3:5]  = r_cot_cmd_in      # pass-through
        u_rate_full[0:3] = u_rate1
        return u_opt_full, u_rate_full, float(solve_ms), int(status)

    
    def yes_cot_solve(self, x_0, u_0, p):
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

        # Extract augmented state (u_cmd) to apply + first control (u_rate)
        if status == 0:
            u_rate1 = self.yes_cot_solver.get(0, "u").reshape(-1).copy()
            x_opt1  = self.yes_cot_solver.get(1, "x").reshape(-1)
            u_opt1  = x_opt1[8:13].copy()
        else:
            u_rate1 = u_0.copy()
            u_opt1  = x_0[8:13].copy()

        # logging
        xs, us, ps = self._extract_all_xup(cot_using=True)
        self._mmap_writer.write(
            x_all=xs,
            u_all=us,
            p_all=ps,
            solve_ms=float(solve_ms),
            status=int(status),
        )

        return u_opt1, u_rate1, float(solve_ms), int(status)
    
    def compute_MPC(self, mpci: Dict[str, Any]) -> Dict[str, Any]:
        x_0    = np.asarray(mpci.get("x_0", np.zeros(self.yes_cot_nx)), dtype=np.float64).ravel()
        u_0    = np.asarray(mpci.get("u_0", np.zeros(self.yes_cot_nu)), dtype=np.float64).ravel()
        p      = np.asarray(mpci.get("p",   np.zeros(self.yes_cot_np)), dtype=np.float64).ravel()
        cot_using = bool(mpci.get("use_cot", False))

        if x_0.size != self.yes_cot_nx: raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.yes_cot_nx}")
        if p.size != self.yes_cot_np: raise ValueError(f"p size mismatch: got {p.size}, expected {self.yes_cot_np}")
        if u_0.size != self.yes_cot_nu: u_0 = np.zeros(self.yes_cot_nu, dtype=np.float64)

        if cot_using: u_aug, u_rate, solve_ms, status = self.yes_cot_solve(x_0, u_0, p)
        else: u_aug, u_rate, solve_ms, status = self.no_cot_solve(x_0, u_0, p)

        return {"u_opt": u_aug.astype(np.float64), "u_rate": u_rate.astype(np.float64), "solve_ms": float(solve_ms), "state": int(status),}

    def _extract_all_xup(self, cot_using: bool):
        N = self.N

        if cot_using:
            nx, nu, np_ = self.yes_cot_nx, self.yes_cot_nu, self.yes_cot_np
            sol = self.yes_cot_solver

            xs = np.empty((N + 1, nx), dtype=np.float64)
            ps = np.empty((N + 1, np_), dtype=np.float64)
            us = np.empty((N,     nu), dtype=np.float64)

            for k in range(N + 1):
                xs[k] = np.asarray(sol.get(k, "x"), dtype=np.float64).ravel()
                ps[k] = np.asarray(sol.get(k, "p"), dtype=np.float64).ravel()
            for k in range(N):
                us[k] = np.asarray(sol.get(k, "u"), dtype=np.float64).ravel()

            return xs, us, ps

        # ---------------- no_cot -> upcast to yes_cot dims ----------------
        sol = self.no_cot_solver

        xs = np.zeros((N + 1, self.yes_cot_nx), dtype=np.float64)  # (N+1,13)
        us = np.zeros((N,     self.yes_cot_nu), dtype=np.float64)  # (N,5)
        ps = np.empty((N + 1, self.no_cot_np),  dtype=np.float64)  # (N+1,11)

        # Holds (broadcasted later)
        r_cot_hold     = getattr(self, "_last_r_cot",     None)
        r_cot_cmd_hold = getattr(self, "_last_r_cot_cmd", None)
        if r_cot_hold is None:
            r_cot_hold = np.zeros(2, dtype=np.float64)
        if r_cot_cmd_hold is None:
            r_cot_cmd_hold = np.zeros(2, dtype=np.float64)

        # Fill holds once (broadcast)
        xs[:, 6:8]   = r_cot_hold
        xs[:, 11:13] = r_cot_cmd_hold

        # Pull each stage once, map into upcast arrays
        for k in range(N + 1):
            xk = np.asarray(sol.get(k, "x"), dtype=np.float64).ravel()  # (9,)
            pk = np.asarray(sol.get(k, "p"), dtype=np.float64).ravel()  # (11,)

            # no_cot x: [theta(0:3), omega(3:6), delta_theta_cmd(6:9)]
            xs[k, 0:6]  = xk[0:6]
            xs[k, 8:11] = xk[6:9]
            ps[k] = pk

        for k in range(N):
            uk = np.asarray(sol.get(k, "u"), dtype=np.float64).ravel()  # (3,)
            us[k, 0:3] = uk

        return xs, us, ps
