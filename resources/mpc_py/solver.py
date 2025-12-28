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
        
        from .params import N, MASS, G_ACCEL
        self.N = int(N)
        self.f_ref = float(MASS*G_ACCEL)

        mmap_path = os.environ.get("STRIDER_MPC_MMAP", "/tmp/strider_mpc_debug.mmap")
        self._mmap_writer = MMapWriter(mmap_path, self.N, self.nx, self.nu, self.np)
    
    def solve(self, pos_d, yaw_d, x_0, u_0, p, debug: bool = False):
        x_0   = np.asarray(x_0,   dtype=np.float64).ravel()
        u_0   = np.asarray(u_0,   dtype=np.float64).ravel()
        p     = np.asarray(p,     dtype=np.float64).ravel()
        pos_d = np.asarray(pos_d, dtype=np.float64).ravel()
        yaw_d = np.asarray(yaw_d, dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.solver.set(0, "lbx", x_0)
        self.solver.set(0, "ubx", x_0)

        # initial guess
        for k in range(self.N + 1): self.solver.set(k, "x", x_0)
        for k in range(self.N): self.solver.set(k, "u", u_0)

        # feed parameter
        for k in range(self.N + 1): self.solver.set(k, "p", p)

        # feed input reference
        # [0,k-1]th step reference [p_d, theta_d, f_d]
        for k in range(self.N): self.solver.set(k, "yref", np.concatenate([pos_d[3*k : 3*(k+1)], np.array([float(yaw_d[k]), self.f_ref], dtype=np.float64)]))
        # [k]th terminal reference [p_d, theta_d]
        self.solver.set(self.N, "yref", np.concatenate([pos_d[3 * (self.N - 1) : 3 * self.N], np.array([float(yaw_d[self.N - 1])], dtype=np.float64)]))

        # Solve
        t0 = time.perf_counter()
        status = self.solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract first control
        if status == 0: u_opt0 = self.solver.get(0, "u").flatten()
        else: u_opt0 = u_0.copy()

        if debug and status==0:
            xs, us, ps = self._extract_all_xup()
            self._mmap_writer.write(
                pos_d=pos_d.reshape(self.N, 3),
                yaw_d=yaw_d.reshape(self.N,),
                x_all=xs,
                u_all=us,
                p_all=ps,
                solve_ms=float(solve_ms),
                status=int(status),
            )

        return u_opt0.astype(np.float64), float(solve_ms), int(status)
    
    def compute_MPC(self, mpci: Dict[str, Any]) -> Dict[str, Any]:
        pos_d  = np.asarray(mpci.get("pos_d", np.zeros(3*self.N)), dtype=np.float64).ravel()
        yaw_d  = np.asarray(mpci.get("yaw_d", np.zeros(self.N)),   dtype=np.float64).ravel()
        x_0    = np.asarray(mpci.get("x_0",   np.zeros(self.nx)),  dtype=np.float64).ravel()
        u_0    = np.asarray(mpci.get("u_0",   np.zeros(self.nu)),  dtype=np.float64).ravel()
        p      = np.asarray(mpci.get("p",     np.zeros(self.np)),  dtype=np.float64).ravel()
        debug  = bool(mpci.get("debug", False))
        
        if x_0.size != self.nx: raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.nx}")
        if p.size != self.np:   raise ValueError(f"p size mismatch: got {p.size}, expected {self.np}")
        if u_0.size != self.nu: u_0 = np.zeros(self.nu, dtype=np.float64)

        q_d, solve_ms, status = self.solve(pos_d, yaw_d, x_0, u_0, p, debug=debug)

        return {"u": q_d.astype(np.float64), "solve_ms": float(solve_ms), "state": int(status),}

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