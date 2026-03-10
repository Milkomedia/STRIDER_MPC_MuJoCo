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
        # 1. build "USE-DELTA"
        from .use_delta.model import build_ocp as use_delta_build_ocp
        self.use_delta_ocp  = use_delta_build_ocp()
        use_delta_json_path  = BASE_DIR / "use_delta" / f"{self.use_delta_ocp.model.name}.json"
        self.use_delta_solver  = AcadosOcpSolver(self.use_delta_ocp, json_file=str(use_delta_json_path))

        self.use_delta_nx = self.use_delta_ocp.model.x.size()[0]
        self.use_delta_nu = self.use_delta_ocp.model.u.size()[0]
        self.use_delta_np = self.use_delta_ocp.model.p.size()[0]

        # 2. build "USE-ARM"
        from .use_arm.model import build_ocp as use_arm_build_ocp
        self.use_arm_ocp  = use_arm_build_ocp()
        use_arm_json_path  = BASE_DIR / "use_arm" / f"{self.use_arm_ocp.model.name}.json"
        self.use_arm_solver  = AcadosOcpSolver(self.use_arm_ocp, json_file=str(use_arm_json_path))

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

        # 4. generate mmap logger
        from . import params as p
        self.N = int(p.N)

        mmap_path = os.environ.get("MRG_MMAP", "/tmp/MRG_debug.mmap")
        self._mmap_writer = MMapWriter(mmap_path, self.N, self.use_full_nx, self.use_full_nu, self.use_full_np)

        # 5. RT-oriented buffers

        # Full-horizon debug buffers for mmap (C-contiguous)
        self._xs_full = np.empty((self.N + 1, self.use_full_nx), dtype=np.float64)
        self._us_full = np.empty((self.N,     self.use_full_nu), dtype=np.float64)
        self._ps_full = np.empty((self.N + 1, self.use_full_np), dtype=np.float64)

        # Reduced-model results upcasted to use_full dims for mmap (C-contiguous)
        self._xs_up  = np.empty((self.N + 1, self.use_full_nx), dtype=np.float64)
        self._us_up  = np.zeros((self.N,     self.use_full_nu), dtype=np.float64)
        self._ps_up  = np.empty((self.N + 1, self.use_full_np), dtype=np.float64)

        # Per-call step outputs (Fortran-order helps Eigen column-major mapping)
        self._u_opt_steps  = np.empty((self.use_full_nu, self.N), dtype=np.float64, order="F")
        self._u_rate_steps = np.empty((self.use_full_nu, self.N), dtype=np.float64, order="F")

        # use_delta down-projection buffers (avoid np.concatenate/copy every tick).
        self._x0_delta = np.empty((self.use_delta_nx,), dtype=np.float64)
        self._u0_delta = np.empty((self.use_delta_nu,), dtype=np.float64)

        # use_arm down-projection buffers (avoid extra allocations every tick).
        self._x0_arm = np.empty((self.use_arm_nx,), dtype=np.float64)
        self._u0_arm = np.empty((self.use_arm_nu,), dtype=np.float64)

        # Cached holds for reduced-model -> use_full upcast (avoid per-tick small allocations).
        self._last_r_rotor     = np.zeros(8, dtype=np.float64)
        self._last_r_rotor_cmd = np.zeros(8, dtype=np.float64)
    
    def use_delta_solve(self, x_0, u_0, p, steps_req: int):
        x_full = np.asarray(x_0, dtype=np.float64).ravel()
        u_full = np.asarray(u_0, dtype=np.float64).ravel()

        # Cache holds for use_delta -> use_full upcast
        self._last_r_rotor[:]     = x_full[6:14]
        self._last_r_rotor_cmd[:] = x_full[17:25]

        # Down-project full-model packet -> use_delta dimensions
        # use_delta x: [theta(0:3), omega(3:6), delta_theta_cmd(6:9)]
        self._x0_delta[0:6] = x_full[0:6]
        self._x0_delta[6:9] = x_full[14:17]
        # use_delta u: [delta_theta_cmd_rate(0:3)]
        self._u0_delta[0:3] = u_full[0:3]
        p = np.asarray(p, dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.use_delta_solver.set(0, "lbx", self._x0_delta)
        self.use_delta_solver.set(0, "ubx", self._x0_delta)

        for k in range(self.N + 1):
            self.use_delta_solver.set(k, "x", self._x0_delta)  # initial guess
            self.use_delta_solver.set(k, "p", p)               # feed parameter

        for k in range(self.N): self.use_delta_solver.set(k, "u", self._u0_delta)  # initial guess

        # Solve
        t0 = time.perf_counter()
        status = self.use_delta_solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0
        
        # Extract full-horizon (for mmap) and requested steps in one pass.
        xs, us, ps = self._extract_all_xup(
            full_model_using=False,
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
        p      = np.asarray(p, dtype=np.float64).ravel()

        # Down-project full-model packet -> use_arm dimensions
        # use_arm x: [theta(0:3), omega(3:6), r_rotor(6:14), r_rotor_cmd(14:22)]
        self._x0_arm[0:6]   = x_full[0:6]
        self._x0_arm[6:14]  = x_full[6:14]
        self._x0_arm[14:22] = x_full[17:25]

        # use_arm u: [r_rotor_cmd_rate(0:8)] = full u[3:11]
        self._u0_arm[0:8] = u_full[3:11]

        # initial state condition (equality constraint)
        self.use_arm_solver.set(0, "lbx", self._x0_arm)
        self.use_arm_solver.set(0, "ubx", self._x0_arm)

        for k in range(self.N + 1):
            self.use_arm_solver.set(k, "x", self._x0_arm)  # initial guess
            self.use_arm_solver.set(k, "p", p)             # feed parameter

        for k in range(self.N):
            self.use_arm_solver.set(k, "u", self._u0_arm)  # initial guess

        # Solve
        t0 = time.perf_counter()
        status = self.use_arm_solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract full-horizon (for mmap) and requested steps in one pass.
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
        x_0   = np.asarray(x_0,   dtype=np.float64).ravel()
        u_0   = np.asarray(u_0,   dtype=np.float64).ravel()
        p     = np.asarray(p,     dtype=np.float64).ravel()

        # initial state condition (equality constraint)
        self.use_full_solver.set(0, "lbx", x_0)
        self.use_full_solver.set(0, "ubx", x_0)

        for k in range(self.N + 1):
            self.use_full_solver.set(k, "x", x_0)  # initial guess
            self.use_full_solver.set(k, "p", p)    # feed parameter

        for k in range(self.N): self.use_full_solver.set(k, "u", u_0)  # initial guess

        # Solve
        t0 = time.perf_counter()
        status = self.use_full_solver.solve()
        solve_ms = (time.perf_counter() - t0) * 1000.0

        # Extract full-horizon (for mmap) and requested steps in one pass.
        xs, us, ps = self._extract_all_xup(
            full_model_using=True,
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
        x_0    = np.asarray(mpci.get("x_0", np.zeros(self.use_full_nx)), dtype=np.float64).ravel()
        u_0    = np.asarray(mpci.get("u_0", np.zeros(self.use_full_nu)), dtype=np.float64).ravel()
        p      = np.asarray(mpci.get("p",   np.zeros(self.use_full_np)), dtype=np.float64).ravel()
        delta_using = bool(mpci.get("use_delta", False))
        arm_using = bool(mpci.get("use_arm", False))

        steps_req = int(mpci.get("steps_req", 1))
        if steps_req > self.N: raise ValueError(f"steps_req too large: got {steps_req}, solver horizon N={self.N}")

        if x_0.size != self.use_full_nx: raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.use_full_nx}")
        if p.size != self.use_full_np: raise ValueError(f"p size mismatch: got {p.size}, expected {self.use_full_np}")
        if u_0.size != self.use_full_nu: u_0 = np.zeros(self.use_full_nu, dtype=np.float64)

        if delta_using and arm_using:         u_opt_steps, u_rate_steps, solve_ms, status = self.use_full_solve(x_0, u_0, p, steps_req)
        elif (not delta_using) and arm_using: u_opt_steps, u_rate_steps, solve_ms, status = self.use_arm_solve(x_0, u_0, p, steps_req)
        elif delta_using and (not arm_using): u_opt_steps, u_rate_steps, solve_ms, status = self.use_delta_solve(x_0, u_0, p, steps_req)
        else:                                 raise ValueError("At least one of 'use_delta' or 'use_arm' must be enabled.")

        return {"u_opt": u_opt_steps, "u_rate": u_rate_steps, "solve_ms": float(solve_ms), "state": int(status),}

    def _extract_all_xup(self, full_model_using: bool, arm_model_using: bool = False, steps_req: int = 0, u_opt_out: np.ndarray | None = None, u_rate_out: np.ndarray | None = None,):
        N = self.N

        if full_model_using:
            nx, nu, np_ = self.use_full_nx, self.use_full_nu, self.use_full_np
            sol = self.use_full_solver

            xs = self._xs_full
            us = self._us_full
            ps = self._ps_full

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
        
        # ---------------- use_arm -> upcast to use_full dims ----------------
        
        if arm_model_using:
            sol = self.use_arm_solver

            xs = self._xs_up
            us = self._us_up
            ps = self._ps_up

            # Pull each stage once, map into upcast arrays
            for k in range(N + 1):
                xk = sol.get(k, "x").reshape(-1)  # (22,)
                pk = sol.get(k, "p").reshape(-1)

                # use_arm x: [theta(0:3), omega(3:6), r_rotor(6:14), r_rotor_cmd(14:22)]
                xs[k, 0:14] = xk[0:14]
                xs[k, 14:17] = 0.0
                xs[k, 17:25] = xk[14:22]

                # Map p -> use_full_np for mmap (truncate/pad with zeros)
                ps[k, :].fill(0.0)
                m = min(pk.size, self.use_full_np)
                ps[k, 0:m] = pk[0:m]

                # u_opt at step i: [delta_theta_cmd(3)=0, r_rotor_cmd(8)]
                if u_opt_out is not None and steps_req > 0 and (1 <= k <= steps_req):
                    u_opt_out[0:3,  k - 1] = 0.0
                    u_opt_out[3:11, k - 1] = xk[14:22]

            for k in range(N):
                uk = sol.get(k, "u").reshape(-1)  # (8,)
                us[k, 0:3] = 0.0
                us[k, 3:11] = uk

                # u_rate at step i: [delta_theta_cmd_rate(3)=0, r_rotor_cmd_rate(8)]
                if u_rate_out is not None and steps_req > 0 and (k < steps_req):
                    u_rate_out[0:3, k] = 0.0
                    u_rate_out[3:11, k] = uk

            return xs, us, ps

        # ---------------- use_delta -> upcast to use_full dims ----------------
        sol = self.use_delta_solver

        xs = self._xs_up
        us = self._us_up
        ps = self._ps_up

        # Fill holds once (broadcast).
        # use_delta does not estimate rotor positions, so hold the last known values from the input packet.
        xs[:, 6:14]   = self._last_r_rotor
        xs[:, 17:25]  = self._last_r_rotor_cmd

        # Pull each stage once, map into upcast arrays
        for k in range(N + 1):
            xk = sol.get(k, "x").reshape(-1)  # (9,)
            pk = sol.get(k, "p").reshape(-1)  # (14,)

            # use_delta x: [theta(0:3), omega(3:6), delta_theta_cmd(6:9)]
            xs[k, 0:6]  = xk[0:6]
            # full-model delta_theta_cmd is at [14:17]
            xs[k, 14:17] = xk[6:9]
            # Map p -> use_full_np for mmap (truncate/pad with zeros)
            ps[k, :].fill(0.0)
            m = min(pk.size, self.use_full_np)
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
