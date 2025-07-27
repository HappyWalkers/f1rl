#!/usr/bin/env python3
import math
import numpy as np
import cvxpy
from scipy.linalg import block_diag
from dataclasses import dataclass, field

from utils.Track import Track, nearest_point, get_reference_trajectory

@dataclass
class MPCConfig:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = [acceleration, steering]
    TK: int = 10  # finite time horizon length kinematic

    Rk: np.ndarray = field(
        default_factory=lambda: np.diag([0.1, 1.0])
    )  # input cost matrix, penalty for inputs - [accel, steering]
    Rdk: np.ndarray = field(
        default_factory=lambda: np.diag([0.1, 1.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering]
    Qk: np.ndarray = field(
        default_factory=lambda: np.diag([15.0, 15.0, 10.0, 15.0])
    )  # state error cost matrix, for the next (T) prediction time steps [x, y, v, yaw]
    Qfk: np.ndarray = field(
        default_factory=lambda: np.diag([15.0, 15.0, 10.0, 15.0])
    )  # final state error matrix, penalty for the final state constraints

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 7.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 4.0  # maximum acceleration [m/ss]

@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    steer: float = 0.0

class MPCPolicy:
    def __init__(self, track: Track, config: MPCConfig = MPCConfig()):
        self.track = track
        self.config = config
        self.ox = None
        self.oy = None
        self.oyaw = None
        self.ov = None
        self.oa = None
        self.od = None
        self.mpc_prob_init()

    def predict(self, observation, deterministic=True):
        current_s, current_ey, current_vel, current_yaw_car_global = observation[:4]
        
        current_x, current_y, _ = self.track.frenet_to_cartesian(current_s, current_ey, 0)
        
        vehicle_state = State(x=current_x, y=current_y, v=current_vel, yaw=current_yaw_car_global)

        ref_path = self.calc_ref_trajectory(vehicle_state)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        self.oa, self.od, self.ox, self.oy, self.oyaw, self.ov, path_predict = self.linear_mpc_control(ref_path, x0)
        
        if self.od is not None:
            steer_output = self.od[0]
            speed_output = vehicle_state.v + self.oa[0] * self.config.DTK
        else:
            steer_output = 0.0
            speed_output = vehicle_state.v

        return np.array([steer_output, speed_output]), None

    def mpc_prob_init(self):
        self.xk = cvxpy.Variable((self.config.NXK, self.config.TK + 1))
        self.uk = cvxpy.Variable((self.config.NU, self.config.TK))
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))

        self.A_params = [cvxpy.Parameter((self.config.NXK, self.config.NXK)) for _ in range(self.config.TK)]
        self.B_params = [cvxpy.Parameter((self.config.NXK, self.config.NU)) for _ in range(self.config.TK)]
        self.C_params = [cvxpy.Parameter((self.config.NXK,)) for _ in range(self.config.TK)]

        # Initialize parameters with a zero-state trajectory
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(path_predict[2,t], path_predict[3,t], 0.0)
            self.A_params[t].value = A
            self.B_params[t].value = B
            self.C_params[t].value = C

        objective = 0.0
        
        Q_block = block_diag(*([self.config.Qk] * (self.config.TK) + [self.config.Qfk]))
        R_block = block_diag(*([self.config.Rk] * self.config.TK))
        Rd_block = block_diag(*([self.config.Rdk] * (self.config.TK - 1)))
        
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        objective += cvxpy.quad_form(cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1]), Rd_block)
        
        constraints = [self.xk[:, 0] == self.x0k]

        for t in range(self.config.TK):
            constraints += [self.xk[:, t + 1] == self.A_params[t] @ self.xk[:, t] + self.B_params[t] @ self.uk[:, t] + self.C_params[t]]

        constraints += [
            self.uk[0, :] <= self.config.MAX_ACCEL,
            self.uk[0, :] >= -self.config.MAX_ACCEL,
            self.uk[1, :] <= self.config.MAX_STEER,
            self.uk[1, :] >= self.config.MIN_STEER,
            cvxpy.abs(self.uk[1, 1:] - self.uk[1, :-1]) <= self.config.MAX_DSTEER * self.config.DTK,
            self.xk[2, :] <= self.config.MAX_SPEED,
            self.xk[2, :] >= self.config.MIN_SPEED,
        ]

        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state):
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        
        track_points = np.vstack((self.track.centerline.xs, self.track.centerline.ys)).T
        _, _, _, _, ind = nearest_point(np.array([state.x, state.y]), track_points)
        
        ref_traj[0, 0] = self.track.centerline.xs[ind]
        ref_traj[1, 0] = self.track.centerline.ys[ind]
        ref_traj[2, 0] = self.track.centerline.vxs[ind] if self.track.centerline.vxs is not None else self.config.MAX_SPEED / 2
        ref_traj[3, 0] = self.track.centerline.calc_yaw(self.track.centerline.ss[ind])

        travel = abs(state.v) * self.config.DTK
        dind = int(round(travel / self.config.dlk))
        
        for i in range(1, self.config.TK + 1):
            ind = (ind + dind) % len(self.track.centerline.xs)
            ref_traj[0, i] = self.track.centerline.xs[ind]
            ref_traj[1, i] = self.track.centerline.ys[ind]
            ref_traj[2, i] = self.track.centerline.vxs[ind] if self.track.centerline.vxs is not None else self.config.MAX_SPEED / 2
            ref_traj[3, i] = self.track.centerline.calc_yaw(self.track.centerline.ss[ind])
            
        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], v=x0[2], yaw=x0[3])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):
        delta = np.clip(delta, self.config.MIN_STEER, self.config.MAX_STEER)
        state.x += state.v * math.cos(state.yaw) * self.config.DTK
        state.y += state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw += (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        state.v += a * self.config.DTK
        state.v = np.clip(state.v, self.config.MIN_SPEED, self.config.MAX_SPEED)
        return state

    def get_model_matrix(self, v, phi, delta):
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0
        
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(path_predict[2, t], path_predict[3, t], self.od[t] if self.od is not None else 0.0)
            self.A_params[t].value = A
            self.B_params[t].value = B
            self.C_params[t].value = C

        self.ref_traj_k.value = ref_traj
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if self.MPC_prob.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()
        else:
            # print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0):
        if self.oa is None or self.od is None:
            self.oa = [0.0] * self.config.TK
            self.od = [0.0] * self.config.TK

        path_predict = self.predict_motion(x0, self.oa, self.od, ref_path)

        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(ref_path, path_predict, x0)

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    def reset(self):
        self.ox = None
        self.oy = None
        self.oyaw = None
        self.ov = None
        self.oa = None
        self.od = None

