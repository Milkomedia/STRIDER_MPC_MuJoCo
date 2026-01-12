from . import params as p
from acados_template import AcadosOcpSolver
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
        from .model import build_ocp
        self.ocp = build_ocp()

        json_path = BASE_DIR / f"{self.ocp.model.name}.json"
        self.solver = AcadosOcpSolver(self.ocp, json_file=str(json_path))
        self.nx = self.ocp.model.x.size()[0]
        self.nu = self.ocp.model.u.size()[0]
        self.np = self.ocp.model.p.size()[0]
        self.nlog = 10
        
        self.N = int(p.N)

        mmap_path = os.environ.get("STRIDER_MPC_MMAP", "/tmp/strider_mpc_debug.mmap")
        self._mmap_writer = MMapWriter(mmap_path, self.N, self.nx, self.nu, self.np, self.nlog)
    
    def solve(self, x_0, u_0, p, log, debug: bool = False):
        x_0   = np.asarray(x_0,   dtype=np.float64).ravel()
        u_0   = np.asarray(u_0,   dtype=np.float64).ravel()
        p     = np.asarray(p,     dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.solver.set(0, "lbx", x_0)
        self.solver.set(0, "ubx", x_0)

        for k in range(self.N + 1):
            self.solver.set(k, "x", x_0)  # initial guess
            self.solver.set(k, "p", p)    # feed parameter

        for k in range(self.N):
            self.solver.set(k, "u", u_0)  # initial guess

        # Solve
        t0 = time.perf_counter()
        status = self.solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract augmented state (u_cmd) to apply + first control (u_rate)
        if status == 0:
            u_rate1 = self.solver.get(0, "u").reshape(-1).copy()
            x_opt1  = self.solver.get(1, "x").reshape(-1)
            u_opt1  = x_opt1[8:13].copy()
        else:
            u_rate1 = u_0.copy()
            u_opt1  = x_0[8:13].copy()

        if debug and status==0:
            xs, us, ps = self._extract_all_xup()
            self._mmap_writer.write(
                x_all=xs,
                u_all=us,
                p_all=ps,
                log_param=log,
                solve_ms=float(solve_ms),
                status=int(status),
            )

        return u_opt1, u_rate1, float(solve_ms), int(status)
    
    def compute_MPC(self, mpci: Dict[str, Any]) -> Dict[str, Any]:
        x_0    = np.asarray(mpci.get("x_0",   np.zeros(self.nx)),  dtype=np.float64).ravel()
        u_0    = np.asarray(mpci.get("u_0",   np.zeros(self.nu)),  dtype=np.float64).ravel()
        p      = np.asarray(mpci.get("p",     np.zeros(self.np)),  dtype=np.float64).ravel()
        log    = np.asarray(mpci.get("log",   np.zeros(self.np)),  dtype=np.float64).ravel()
        debug  = bool(mpci.get("debug", False))
        
        if x_0.size != self.nx: raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.nx}")
        if p.size != self.np:   raise ValueError(f"p size mismatch: got {p.size}, expected {self.np}")
        if u_0.size != self.nu: u_0 = np.zeros(self.nu, dtype=np.float64)

        u_aug, u_rate, solve_ms, status = self.solve(x_0, u_0, p, log, debug=debug)

        return {"u_opt": u_aug.astype(np.float64), "u_rate": u_rate.astype(np.float64), "solve_ms": float(solve_ms), "state": int(status),}

    def _extract_all_xup(self):
        xs = np.empty((self.N + 1, self.nx), dtype=np.float64)
        us = np.empty((self.N, self.nu), dtype=np.float64)
        ps = np.empty((self.N + 1, self.np), dtype=np.float64)

        for k in range(self.N + 1):
            xs[k, :] = np.asarray(self.solver.get(k, "x"), dtype=np.float64).ravel()
            ps[k, :] = np.asarray(self.solver.get(k, "p"), dtype=np.float64).ravel()
        for k in range(self.N):
            us[k, :] = np.asarray(self.solver.get(k, "u"), dtype=np.float64).ravel()

        return xs, us, ps