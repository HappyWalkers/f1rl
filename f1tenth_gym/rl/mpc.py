import numpy as np
from absl import logging
import cvxpy as cp


class MPCPolicy:
    """
    Kinematic MPC-like policy for F1Tenth gym environment.

    - State: [x, y, delta, v, yaw]
    - Control (internal): [delta_rate, acceleration]
    - Action to env: [steering_angle, desired_speed]

    Notes:
    - Observation from wrapper: [s, ey, vel, yaw] (+ optional lidar/params). We convert (s, ey) to
      (x, y) using the provided `Track` object.
    - The controller performs one linearization pass around the previous predicted trajectory
      (or zeros on the first step), solves a time-varying LQR/QP in closed form (unconstrained),
      and returns the first control converted to the required action interface.
    - To keep dependencies light, we solve the stacked least-squares system directly with numpy.
    """

    def __init__(self, track, horizon: int = 25, dt: float = 0.05):
        # Track object is required to convert Frenet to Cartesian and to fetch reference path
        self.track = track

        # Horizon and timestep
        self.N = int(horizon)
        self.dt = float(dt)

        # Cost weights (roughly based on the reference config)
        # Q on [x, y, delta, v, yaw]
        self.Q = np.diag([5.0, 5.0, 0.1, 0.2, 50.0])
        self.P = np.diag([5.0, 5.0, 0.1, 0.2, 50.0])  # terminal
        # R on [delta_rate, acceleration]
        self.R = np.diag([0.01, 0.1])
        # Rd on delta of inputs
        self.Rd = np.diag([0.1, 0.1])

        # Vehicle limits (match env defaults conservatively)
        self.steer_min = -0.4189
        self.steer_max = 0.4189
        # Keep speeds moderate (policy-level choice); env clamps to its own limits
        self.speed_min = 0.8
        self.speed_max = 2.5
        # Internal rate limits (roughly from env defaults)
        self.delta_rate_min = -3.2
        self.delta_rate_max = 3.2
        self.accel_min = -4.0
        self.accel_max = 4.0

        # Internal state memory
        self.last_delta = 0.0
        self.x_pred = None  # shape (5, N+1)
        self.u_pred = None  # shape (2, N)

    # ---------- Public API ----------
    def predict(self, observation, deterministic: bool = True):
        # Parse observation core components
        s = float(observation[0])
        ey = float(observation[1])
        v = float(observation[2])
        yaw = float(observation[3])  # global yaw

        # Convert to Cartesian using the track object
        try:
            x, y, _ = self.track.frenet_to_cartesian(s, ey, 0.0)
        except Exception:
            # Fallback if conversion fails
            logging.warning("Track conversion failed; using straight-line fallback")
            return np.array([0.0, max(self.speed_min, min(v, self.speed_max))]), None

        # Build reference trajectory in Cartesian along the centerline/raceline
        x_ref = self._build_reference(s, yaw0=yaw, lateral_offset=ey)

        # Initial state for the optimizer
        x0 = np.array([x, y, self.last_delta, v, yaw], dtype=np.float64)

        # Warm start or zeros
        if self.x_pred is None or self.u_pred is None:
            self.x_pred = np.repeat(x0.reshape(-1, 1), self.N + 1, axis=1)
            self.u_pred = np.zeros((2, self.N))

        # Solve stacked LTV tracking problem (one SQP-like pass) using CVXPY
        x_traj, u_traj = self._solve_mpc_cvxpy(x0, x_ref)

        # Store predictions for warm start next time
        self.x_pred = x_traj
        self.u_pred = u_traj

        # First control in trajectory is delta_rate, accel
        delta_rate, accel = float(u_traj[0, 0]), float(u_traj[1, 0])
        # Clamp rates
        delta_rate = float(np.clip(delta_rate, self.delta_rate_min, self.delta_rate_max))
        accel = float(np.clip(accel, self.accel_min, self.accel_max))

        # Convert to action directly from predicted trajectory
        delta_next = float(np.clip(x_traj[2, 1], self.steer_min, self.steer_max))
        v_next = float(np.clip(x_traj[3, 1], self.speed_min, self.speed_max))

        # Debug logs
        ref_xy = x_ref[:2, 1]
        pred_xy = x_traj[:2, 1]
        pos_err = float(np.linalg.norm(pred_xy - ref_xy))
        yaw_err = float(self._angle_diff(x_traj[4, 1], x_ref[4, 1]))
        logging.info(
            f"MPC obs s={s:.2f} ey={ey:.2f} v={v:.2f} yaw={yaw:.2f} | xy=({x:.2f},{y:.2f})"
        )
        logging.info(
            f"MPC cmd delta_rate={delta_rate:.3f} accel={accel:.3f} => delta={delta_next:.3f} v_next={v_next:.2f}"
        )
        logging.info(
            f"MPC step1 pos_err={pos_err:.3f} yaw_err={yaw_err:.3f} ref_v={x_ref[3,1]:.2f}"
        )

        # Update memory
        self.last_delta = delta_next

        return np.array([delta_next, v_next], dtype=np.float32), None

    def reset(self):
        self.last_delta = 0.0
        self.x_pred = None
        self.u_pred = None

    # ---------- MPC internals ----------
    def _build_reference(self, s0: float, yaw0: float, lateral_offset: float) -> np.ndarray:
        """
        Build reference trajectory (nx, N+1) over arc length using track geometry.
        Reference state vector is [x, y, delta_ref(=0), v_ref, yaw].
        """
        s_vals = np.zeros(self.N + 1, dtype=np.float64)
        x_vals = np.zeros(self.N + 1, dtype=np.float64)
        y_vals = np.zeros(self.N + 1, dtype=np.float64)
        yaw_vals = np.zeros(self.N + 1, dtype=np.float64)
        v_ref_vals = np.zeros(self.N + 1, dtype=np.float64)

        s_vals[0] = s0 % self.track.s_frame_max
        for k in range(self.N + 1):
            s_k = s_vals[k]
            x_k, y_k = self.track.centerline.calc_position(s_k)
            yaw_k = self.track.centerline.calc_yaw(s_k)
            try:
                v_ref_k = float(self.track.raceline.calc_velocity(s_k))
            except Exception:
                curvature = float(self.track.curvature(s_k))
                v_ref_k = self._curvature_speed(curvature)
            # Desired lateral offset decays to centerline along horizon
            decay = k / max(self.N, 1)
            ey_des_k = float(lateral_offset * (1.0 - decay))
            # Apply lateral offset along track normal to define reference XY
            x_k = x_k - ey_des_k * np.sin(yaw_k)
            y_k = y_k + ey_des_k * np.cos(yaw_k)

            # Mild curvature-based speed scaling only
            curvature = float(self.track.curvature(s_k))
            v_scale = 1.0 / (1.0 + 2.0 * abs(curvature))
            v_ref_k = float(np.clip(v_ref_k * v_scale, self.speed_min, self.speed_max))

            x_vals[k] = x_k
            y_vals[k] = y_k
            yaw_vals[k] = yaw_k
            v_ref_vals[k] = v_ref_k

            if k < self.N:
                # Advance s using a conservative step based on current reference speed
                ds = v_ref_k * self.dt
                s_vals[k + 1] = (s_k + ds) % self.track.s_frame_max

        delta_ref = np.zeros_like(x_vals)

        # Unwrap yaw to be continuous and near current yaw
        yaw_vals = self._unwrap_yaw_sequence(yaw_vals, yaw0)

        xref = np.vstack([x_vals, y_vals, delta_ref, v_ref_vals, yaw_vals])
        return xref

    def _curvature_speed(self, curvature: float) -> float:
        curvature = max(1e-4, min(abs(curvature), 1.0))
        return float(self.speed_max / (1.0 + 5.0 * curvature))

    @staticmethod
    def _linearize_kinematic(state: np.ndarray, dt: float) -> tuple:
        """
        Linearize the kinematic bicycle model around the provided state and zero input.
        Returns continuous-time A, B and then discretized Ad, Bd, and residual c.

        State: [x, y, delta, v, yaw]
        Control: [delta_rate, accel]
        f = [v*cos(yaw), v*sin(yaw), delta_rate, accel, (v/L)*tan(delta)]
        """
        x, y, delta, v, yaw = state
        L = 0.33  # approximate wheelbase (meters)

        # Continuous-time Jacobians
        A = np.zeros((5, 5), dtype=np.float64)
        A[0, 3] = np.cos(yaw)
        A[0, 4] = -v * np.sin(yaw)
        A[1, 3] = np.sin(yaw)
        A[1, 4] = v * np.cos(yaw)
        A[4, 2] = (v / L) * (1.0 / (np.cos(delta) ** 2))
        A[4, 3] = np.tan(delta) / L

        B = np.zeros((5, 2), dtype=np.float64)
        B[2, 0] = 1.0
        B[3, 1] = 1.0

        # Discretize (Euler)
        Ad = np.eye(5) + A * dt
        Bd = B * dt

        # Residual term for linearization about state, u=0
        fx = np.array([
            v * np.cos(yaw),
            v * np.sin(yaw),
            0.0,
            0.0,
            (v / L) * np.tan(delta),
        ])
        c = state + fx * dt - Ad @ state  # (x + f*dt) - Ad*x - Bd*u (u=0)

        return Ad, Bd, c

    def _build_stacked_system(self, x0: np.ndarray, xref: np.ndarray) -> tuple:
        """
        Build time-varying stacked matrices for the horizon using the previous predicted
        trajectory for linearization.
        Returns lists of (Ad_list, Bd_list, c_list).
        """
        Ad_list, Bd_list, c_list = [], [], []

        # Linearize around previous trajectory if available; otherwise around xref
        # Linearize around the reference trajectory for better tracking
        base_traj = xref
        for k in range(self.N):
            Ad, Bd, c = self._linearize_kinematic(base_traj[:, k], self.dt)
            Ad_list.append(Ad)
            Bd_list.append(Bd)
            c_list.append(c)
        return Ad_list, Bd_list, c_list

    def _stack_dynamics(self, Ad_list, Bd_list, c_list) -> tuple:
        """
        Construct Sx (nx*N x nx), Su (nx*N x nu*N), sc (nx*N,) for x1..xN.
        """
        nx = 5
        nu = 2
        N = self.N

        Sx = np.zeros((nx * N, nx), dtype=np.float64)
        Su = np.zeros((nx * N, nu * N), dtype=np.float64)
        sc = np.zeros((nx * N,), dtype=np.float64)

        # Recursions
        sc_prev = np.zeros(nx, dtype=np.float64)
        Phi = np.eye(nx)
        for k in range(N):
            Ad = Ad_list[k]
            # Update Sx row for x_{k+1}
            Phi = Ad @ Phi
            Sx[k * nx:(k + 1) * nx, :] = Phi

            # Build Su coefficients for controls u_0..u_k
            for j in range(0, k + 1):
                coeff = np.eye(nx, dtype=np.float64)
                # Product A_{j+1} ... A_k (left-most has largest index)
                for m in range(j + 1, k + 1):
                    coeff = Ad_list[m] @ coeff
                Su[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = coeff @ Bd_list[j]

            # Residual propagation to x_{k+1}
            sc_prev = Ad @ sc_prev + c_list[k]
            sc[k * nx:(k + 1) * nx] = sc_prev

        return Sx, Su, sc

    def _solve_mpc_cvxpy(self, x0: np.ndarray, xref: np.ndarray) -> tuple:
        nx, nu, N = 5, 2, self.N

        # Linearize along reference trajectory for this iteration
        Ad_list, Bd_list, c_list = self._build_stacked_system(x0, xref)

        # Variables
        x = cp.Variable((nx, N + 1))
        u = cp.Variable((nu, N))

        constraints = []
        # Initial condition
        constraints.append(x[:, 0] == x0)

        # Dynamics and bounds
        for k in range(N):
            Ad, Bd, c = Ad_list[k], Bd_list[k], c_list[k]
            constraints.append(x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k] + c)
            # Input bounds
            constraints.append(u[0, k] <= self.delta_rate_max)
            constraints.append(u[0, k] >= self.delta_rate_min)
            constraints.append(u[1, k] <= self.accel_max)
            constraints.append(u[1, k] >= self.accel_min)
            # State bounds (delta, v)
            constraints.append(x[2, k + 1] <= self.steer_max)
            constraints.append(x[2, k + 1] >= self.steer_min)
            constraints.append(x[3, k + 1] <= self.speed_max)
            constraints.append(x[3, k + 1] >= 0.0)

        # Input rate constraints
        for k in range(1, N):
            du = u[:, k] - u[:, k - 1]
            constraints.append(cp.abs(du[0]) <= 2.5)
            constraints.append(cp.abs(du[1]) <= 2.5)

        # Objective
        cost = 0
        for k in range(N):
            dx = x[:, k] - xref[:, k]
            cost += cp.quad_form(dx, self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            if k > 0:
                cost += cp.quad_form(u[:, k] - u[:, k - 1], self.Rd)
        dxN = x[:, N] - xref[:, N]
        cost += cp.quad_form(dxN, self.P)

        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Warm start if available
        if self.x_pred is not None and self.u_pred is not None:
            try:
                x.value = self.x_pred
                u.value = self.u_pred
            except Exception:
                pass

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=10_000)
        except Exception as e:
            logging.warning(f"CVXPY solve exception: {e}")

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            logging.warning(f"MPC solve status: {prob.status}; falling back to previous predictions")
            if self.x_pred is None or self.u_pred is None:
                # trivial fallback
                self.x_pred = np.repeat(x0.reshape(-1, 1), N + 1, axis=1)
                self.u_pred = np.zeros((nu, N))
            return self.x_pred, self.u_pred

        return x.value, u.value

    @staticmethod
    def _block_diag(mats):
        total_rows = sum(m.shape[0] for m in mats)
        total_cols = sum(m.shape[1] for m in mats)
        out = np.zeros((total_rows, total_cols), dtype=np.float64)
        r, c = 0, 0
        for M in mats:
            rr, cc = M.shape
            out[r:r + rr, c:c + cc] = M
            r += rr
            c += cc
        return out

    @staticmethod
    def _build_diff_operator(nu: int, N: int) -> np.ndarray:
        if N <= 1:
            return np.zeros((0, nu * N), dtype=np.float64)
        rows = nu * (N - 1)
        cols = nu * N
        D = np.zeros((rows, cols), dtype=np.float64)
        for k in range(N - 1):
            # u_{k+1} - u_k
            D[k * nu:(k + 1) * nu, (k + 1) * nu:(k + 2) * nu] = np.eye(nu)
            D[k * nu:(k + 1) * nu, k * nu:(k + 1) * nu] = -np.eye(nu)
        return D

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        return float((d + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _unwrap_yaw_sequence(yaws: np.ndarray, anchor: float) -> np.ndarray:
        out = yaws.copy()
        # Shift entire sequence to be near anchor for k=0
        out[0] = anchor + ((out[0] - anchor + np.pi) % (2 * np.pi) - np.pi)
        for k in range(1, len(out)):
            prev = out[k - 1]
            y = out[k]
            # choose k value minimizing difference to prev
            y_candidates = np.array([y - 2*np.pi, y, y + 2*np.pi])
            idx = np.argmin(np.abs(y_candidates - prev))
            out[k] = y_candidates[idx]
        return out


