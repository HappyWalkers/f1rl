from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
from absl import logging
import time

from .utils.Track import Track


@dataclass
class MPCConfig:
    """Configuration parameters for the MPC policy."""

    horizon: int = 22
    dt: float = 0.05
    wheelbase: float = 0.33
    min_speed: float = 1.2
    max_speed: float = 6.5
    steer_limit: float = 0.4189
    accel_limit: float = 4.5
    steer_rate_limit: float = 3.2
    input_rate_limit: float = 4.0
    q_weights: Tuple[float, float, float, float, float] = (14.0, 14.0, 1.0, 0.4, 12.0)
    r_weights: Tuple[float, float] = (0.35, 0.25)
    rd_weights: Tuple[float, float] = (0.1, 0.2)
    terminal_weights: Tuple[float, float, float, float, float] = (30.0, 30.0, 1.5, 0.8, 18.0)

    def __post_init__(self) -> None:
        self.q_matrix = np.diag(self.q_weights)
        self.r_matrix = np.diag(self.r_weights)
        self.rd_matrix = np.diag(self.rd_weights)
        self.p_matrix = np.diag(self.terminal_weights)


class MPCPolicy:
    """Linear time-varying MPC controller that closely tracks the raceline."""

    def __init__(self, track: Track, config: Optional[MPCConfig] = None) -> None:
        self.track = track
        self.config = config or MPCConfig()
        self.nx = 5
        self.nu = 2
        self.horizon = self.config.horizon

        self.last_delta = 0.0
        self.x_pred: Optional[np.ndarray] = None
        self.u_pred: Optional[np.ndarray] = None

        self.track_length = float(getattr(self.track, "s_frame_max", 0.0) or self.track.raceline.s[-1])
        self.speed_upper_bound = float(
            min(
                self.config.max_speed,
                max(
                    self.config.min_speed + 1.0,
                    self._estimate_track_speed(),
                ),
            )
        )

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        del deterministic
        if observation.shape[0] < 4:
            raise ValueError("Observation must contain at least [s, ey, vel, yaw].")

        s = float(observation[0])
        ey = float(observation[1])
        speed = float(observation[2])
        yaw = float(observation[3])

        logging.info(
            "MPC predict -- s: %.2f, ey: %.3f, v: %.2f, yaw: %.2f, last_delta: %.3f",
            s,
            ey,
            speed,
            yaw,
            self.last_delta,
        )

        x0 = self._state_from_observation(s, ey, speed, yaw)
        reference = self._build_reference(s, ey, speed, x0)

        solution = self._solve_ltv_mpc(x0, reference)
        if solution is None:
            logging.warning("MPC solver failed; falling back to stabilising controller.")
            return self._fallback_action(s, ey, speed, yaw), None

        x_traj, u_traj = solution
        self.x_pred = x_traj
        self.u_pred = u_traj

        delta_rate = float(np.clip(u_traj[0, 0], -self.config.steer_rate_limit, self.config.steer_rate_limit))
        accel = float(np.clip(u_traj[1, 0], -self.config.accel_limit, self.config.accel_limit))

        delta_next = float(np.clip(self.last_delta + delta_rate * self.config.dt, -self.config.steer_limit, self.config.steer_limit))
        speed_prediction = float(np.clip(speed + accel * self.config.dt, self.config.min_speed, self.speed_upper_bound))
        reference_speed = float(np.clip(reference[3, 1], self.config.min_speed, self.speed_upper_bound))
        desired_speed = float(np.clip(0.6 * speed_prediction + 0.4 * reference_speed, self.config.min_speed, self.speed_upper_bound))

        logging.info(
            "MPC action -- delta_next: %.3f, delta_rate: %.3f, accel: %.3f, desired_speed: %.2f",
            delta_next,
            delta_rate,
            accel,
            desired_speed,
        )

        self.last_delta = delta_next
        return np.array([delta_next, desired_speed], dtype=np.float32), None

    def reset(self) -> None:
        self.last_delta = 0.0
        self.x_pred = None
        self.u_pred = None

    def _build_reference(self, s: float, ey: float, current_speed: float, x_initial: np.ndarray) -> np.ndarray:
        ref = np.zeros((self.nx, self.horizon + 1), dtype=np.float64)
        s_vals = np.zeros(self.horizon + 1, dtype=np.float64)
        s_vals[0] = self._wrap_s(s)

        lateral_offset = float(np.clip(ey, -1.5, 1.5))
        current_speed = float(np.clip(current_speed, self.config.min_speed, self.speed_upper_bound))

        for k in range(self.horizon + 1):
            s_k = s_vals[k]
            x_k, y_k = self._sample_position(s_k)
            yaw_ref = self._sample_yaw(s_k)
            curvature = self._sample_curvature(s_k)
            speed_track = self._sample_track_speed(s_k)

            # Blend between the current speed and the raceline target so the optimisation has
            # enough slack to stabilise while still chasing the fast trajectory.
            blend = min(1.0, k / max(1, self.horizon // 2))
            v_ref = (1.0 - blend) * current_speed + blend * speed_track

            if abs(lateral_offset) > 1e-4:
                offset = lateral_offset * (1.0 - k / self.horizon)
                x_k -= offset * np.sin(yaw_ref)
                y_k += offset * np.cos(yaw_ref)

            delta_ref = np.clip(np.arctan(self.config.wheelbase * curvature), -self.config.steer_limit, self.config.steer_limit)

            ref[:, k] = np.array([x_k, y_k, delta_ref, v_ref, yaw_ref], dtype=np.float64)

            if k < self.horizon:
                ds = max(self.config.min_speed * 0.8, v_ref) * self.config.dt
                s_vals[k + 1] = self._wrap_s(s_vals[k] + ds)

        ref[:, 0] = x_initial
        return ref

    def _solve_ltv_mpc(self, x0: np.ndarray, reference: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        linearization = reference.copy()
        linearization[:, 0] = x0

        ad_list, bd_list, c_list = self._linearize_trajectory(linearization)

        x_var = cp.Variable((self.nx, self.horizon + 1))
        u_var = cp.Variable((self.nu, self.horizon))

        constraints = [x_var[:, 0] == x0]
        for k in range(self.horizon):
            constraints.append(x_var[:, k + 1] == ad_list[k] @ x_var[:, k] + bd_list[k] @ u_var[:, k] + c_list[k])
            constraints.append(x_var[2, k + 1] <= self.config.steer_limit)
            constraints.append(x_var[2, k + 1] >= -self.config.steer_limit)
            constraints.append(x_var[3, k + 1] <= self.speed_upper_bound)
            constraints.append(x_var[3, k + 1] >= self.config.min_speed)
            constraints.append(u_var[0, k] <= self.config.steer_rate_limit)
            constraints.append(u_var[0, k] >= -self.config.steer_rate_limit)
            constraints.append(u_var[1, k] <= self.config.accel_limit)
            constraints.append(u_var[1, k] >= -self.config.accel_limit)

        for k in range(1, self.horizon):
            du = u_var[:, k] - u_var[:, k - 1]
            constraints.append(cp.abs(du[0]) <= self.config.input_rate_limit)
            constraints.append(cp.abs(du[1]) <= self.config.input_rate_limit)

        cost_terms = []
        for k in range(self.horizon):
            dx = x_var[:, k] - reference[:, k]
            cost_terms.append(cp.quad_form(dx, self.config.q_matrix))
            cost_terms.append(cp.quad_form(u_var[:, k], self.config.r_matrix))
            if k > 0:
                du = u_var[:, k] - u_var[:, k - 1]
                cost_terms.append(cp.quad_form(du, self.config.rd_matrix))

        dx_terminal = x_var[:, self.horizon] - reference[:, self.horizon]
        cost_terms.append(cp.quad_form(dx_terminal, self.config.p_matrix))

        problem = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)

        if self.x_pred is not None and self.u_pred is not None:
            x_guess = np.hstack([self.x_pred[:, 1:], self.x_pred[:, -1:]])
            u_guess = np.hstack([self.u_pred[:, 1:], self.u_pred[:, -1:]])
            if x_guess.shape == x_var.shape:
                x_var.value = x_guess
            if u_guess.shape == u_var.shape:
                u_var.value = u_guess

        solve_start = time.perf_counter()
        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=5000,
            )
        except Exception as exc:
            logging.error("MPC optimisation crashed: %s", exc)
            return None
        solve_time = time.perf_counter() - solve_start

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            logging.warning("MPC optimisation status %s", problem.status)
            return None

        logging.info(
            "MPC solve -- status: %s, iterations: %s, time: %.3f s",
            problem.status,
            getattr(problem, "solver_stats", None) and problem.solver_stats.num_iters,
            solve_time,
        )

        x_sol = x_var.value
        u_sol = u_var.value
        if x_sol is None or u_sol is None:
            return None

        return np.asarray(x_sol, dtype=np.float64), np.asarray(u_sol, dtype=np.float64)

    def _linearize_trajectory(self, states: np.ndarray) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        ad_list, bd_list, c_list = [], [], []
        for k in range(self.horizon):
            ad, bd, c = self._linearize_step(states[:, k])
            ad_list.append(ad)
            bd_list.append(bd)
            c_list.append(c)
        return ad_list, bd_list, c_list

    def _linearize_step(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, _, delta, velocity, yaw = state
        dt = self.config.dt
        L = self.config.wheelbase

        ad = np.eye(self.nx, dtype=np.float64)
        bd = np.zeros((self.nx, self.nu), dtype=np.float64)

        ad[0, 3] = np.cos(yaw) * dt
        ad[0, 4] = -velocity * np.sin(yaw) * dt
        ad[1, 3] = np.sin(yaw) * dt
        ad[1, 4] = velocity * np.cos(yaw) * dt
        ad[4, 2] = velocity / max(1e-3, L) * (1.0 / (np.cos(delta) ** 2)) * dt
        ad[4, 3] = np.tan(delta) / max(1e-3, L) * dt

        bd[2, 0] = dt
        bd[3, 1] = dt

        fx = np.array(
            [
                velocity * np.cos(yaw),
                velocity * np.sin(yaw),
                0.0,
                0.0,
                velocity / max(1e-3, L) * np.tan(delta),
            ],
            dtype=np.float64,
        )
        c = state + fx * dt - ad @ state

        return ad, bd, c

    def _state_from_observation(self, s: float, ey: float, speed: float, yaw: float) -> np.ndarray:
        x_pos, y_pos, _ = self.track.frenet_to_cartesian(s, ey, 0.0)
        clamped_speed = float(np.clip(speed, self.config.min_speed, self.speed_upper_bound))
        return np.array([x_pos, y_pos, self.last_delta, clamped_speed, yaw], dtype=np.float64)

    def _fallback_action(self, s: float, ey: float, speed: float, yaw: float) -> np.ndarray:
        track_yaw = self._sample_yaw(self._wrap_s(s))
        yaw_error = self._wrap_angle(yaw - track_yaw)

        delta = float(
            np.clip(
                -1.8 * ey - 1.1 * yaw_error,
                -self.config.steer_limit,
                self.config.steer_limit,
            )
        )
        target_speed = float(
            np.clip(
                self._sample_track_speed(s) - 0.5 * abs(ey),
                self.config.min_speed,
                self.speed_upper_bound,
            )
        )
        logging.info(
            "MPC fallback -- delta: %.3f, target_speed: %.2f, ey: %.3f, yaw_error: %.3f",
            delta,
            target_speed,
            ey,
            yaw_error,
        )
        self.last_delta = delta
        self.x_pred = None
        self.u_pred = None
        return np.array([delta, target_speed], dtype=np.float32)

    def _sample_position(self, s: float) -> Tuple[float, float]:
        return self.track.raceline.calc_position(self._wrap_s(s))

    def _sample_yaw(self, s: float) -> float:
        return float(self.track.raceline.calc_yaw(self._wrap_s(s)))

    def _sample_curvature(self, s: float) -> float:
        curvature = float(self.track.raceline.calc_curvature(self._wrap_s(s)))
        return float(np.clip(curvature, -5.0, 5.0))

    def _sample_track_speed(self, s: float) -> float:
        try:
            speed = float(self.track.raceline.calc_velocity(self._wrap_s(s)))
        except Exception:
            speed = self.speed_upper_bound
        if not np.isfinite(speed):
            speed = self.speed_upper_bound
        return float(np.clip(speed, self.config.min_speed, self.speed_upper_bound))

    def _estimate_track_speed(self) -> float:
        candidate = float(self.config.max_speed)
        if hasattr(self.track, "vxs") and self.track.vxs is not None:
            data = np.asarray(self.track.vxs, dtype=np.float64)
            if data.size > 0:
                finite_data = data[np.isfinite(data)]
                if finite_data.size > 0:
                    candidate = float(np.percentile(finite_data, 95))
        return float(np.clip(candidate, self.config.min_speed + 0.5, self.config.max_speed))

    def _wrap_s(self, s: float) -> float:
        return float(s % self.track_length)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

