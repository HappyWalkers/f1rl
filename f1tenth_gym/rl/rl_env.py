import gymnasium
import gym  # Keep the original gym import
import numpy as np
import torch
from gymnasium import spaces
import random
from absl import logging
from utils.Track import Track # Added import
import collections


class F110GymWrapper(gymnasium.Env):
    def __init__(self, 
                 waypoints, 
                 seed, 
                 map_path, 
                 num_agents, 
                 track: Track,
                 max_episode_steps=10000,
                 # Domain Randomization Params
                 mu=1.0, 
                 C_Sf=4.718, 
                 C_Sr=5.4562, 
                 lf=0.15875, 
                 lr=0.17145, 
                 h=0.074, 
                 m=3.74, 
                 I=0.04712, 
                 s_min=-0.4189, 
                 s_max=0.4189, 
                 sv_min=-3.2, 
                 sv_max=3.2, 
                 v_switch=7.319, 
                 a_max=9.51, 
                 v_min=-5.0, 
                 v_max=20.0, 
                 width=0.31, 
                 length=0.58,
                 lidar_noise_stddev=0.0,
                 push_0_prob=0, # Probability of pushing 0 times
                 push_2_prob=0  # Probability of pushing 2 times
                 ):
        super().__init__()
        self.track = track # Store track object
        self.seed = seed
        self.np_random, _ = gymnasium.utils.seeding.np_random(seed)

        # Store DR params
        self.lidar_noise_stddev = lidar_noise_stddev
        self.push_0_prob = push_0_prob
        self.push_2_prob = push_2_prob
        self.push_1_prob = 1.0 - push_0_prob - push_2_prob
        assert 0 <= self.push_1_prob <= 1, f"Probabilities must sum to 1: p0={push_0_prob}, p2={push_2_prob}"
        self.action_buffer = collections.deque() # Initialize an unbounded deque

        # Construct params dict for the base env
        self.base_env_params = {
            'mu': mu, 'C_Sf': C_Sf, 'C_Sr': C_Sr, 'lf': lf, 'lr': lr, 'h': h,
            'm': m, 'I': I, 's_min': s_min, 's_max': s_max, 'sv_min': sv_min,
            'sv_max': sv_max, 'v_switch': v_switch, 'a_max': a_max, 'v_min': v_min,
            'v_max': v_max, 'width': width, 'length': length
        }

        self.env = gym.make(  # Use gym.make instead of gymnasium.make
            'f110_gym:f110-v0',
            model='dynamic_ST',
            params=self.base_env_params,
            map=map_path,
            seed=seed,
            num_agents=num_agents,
            # drive_control_mode='acc',
            # steering_control_mode='vel',
            waypoints=waypoints,
            timestep=0.01,
        )

        # Update observation space: [s, ey, vel, yaw_angle, yaw_rate] + lidar
        # low/high values are approximate, might need refinement
        self.observation_space = spaces.Box(
            low=np.concatenate(([-1000.0, -5.0, v_min, -np.pi, -10.0], np.zeros(1080))),
            high=np.concatenate(([1000.0, 5.0, v_max, np.pi, 10.0], np.full(1080, 30.0))),
            shape=(1085,), dtype=np.float32  # 5 state values + 1080 lidar points
        )
        # self.action_space = spaces.Box(
        #     low=np.array([-3.2, 0.0]), high=np.array([3.2, 9.51]), shape=(2,), dtype=np.float32
        # )
        # Action space: [Steering Angle, Speed]
        self.action_space = spaces.Box(
            low=np.array([s_min, 1.0]),  # Min speed slightly above 0
            high=np.array([s_max, v_max]), 
            shape=(2,), dtype=np.float32
        )
        self._max_episode_steps = max_episode_steps
        self.current_step = 0
        self.waypoints = waypoints
        self.last_frenet_arc_length = None
        self.last_lap_counts = 0

    def _process_observation(self, obs):
        """Processes the raw observation dict and applies lidar noise."""
        lidar_scan = obs['scans'][0]

        # Apply lidar noise
        if self.lidar_noise_stddev > 0:
            noise = self.np_random.normal(0, self.lidar_noise_stddev, lidar_scan.shape)
            lidar_scan += noise
            lidar_scan = np.clip(lidar_scan, 0, 30.0) # Assuming max range is 30

        # Observation: [s, ey, vel, yaw_angle, yaw_rate, scans]
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # state_frenet is [s, ey, steer_angle, vel_s, yaw_angle_frenet, yaw_rate_frenet, vel_ey]
        processed_obs = np.concatenate((
            obs["state_frenet"][0][0:2], # s, ey
            obs['state'][0][3:6],  # vel, yaw_angle, yaw_rate
            lidar_scan
        )).astype(np.float32)
        return processed_obs

    def reset(self, seed=None, options=None):
        poses = self.get_reset_pose()
        obs, reward, done, info = self.env.reset(initial_states=poses)
        self.current_step = 0
        self.last_frenet_arc_length = None
        self.action_buffer.clear() # Clear action buffer on reset

        processed_obs = self._process_observation(obs)
        return processed_obs, {}

    def get_reset_pose(self):
        starting_idx = random.sample(range(len(self.waypoints)), 1)
        x, y = self.waypoints[starting_idx][0, 1], self.waypoints[starting_idx][0, 2]
        theta_noise = (2*random.random() - 1) * 0.1
        theta = self.waypoints[starting_idx][0, 3] + theta_noise
        starting_pos = np.array([[x, y, theta]])
        return starting_pos

    def step(self, action):
        self.current_step += 1

        # --- Action Queuing Logic ---
        # Add the new action to the buffer based on queue state
        if not self.action_buffer: # If queue is empty
            self.action_buffer.append(action.copy())
            self.action_buffer.append(action.copy())
        else:
            # Push 0, 1, or 2 times with average 1
            num_pushes = self.np_random.choice([0, 1, 2], p=[self.push_0_prob, self.push_1_prob, self.push_2_prob])
            for _ in range(num_pushes):
                self.action_buffer.append(action.copy())
        
        # --- Action Execution Logic ---
        # Get the action to execute: oldest from queue or current if empty
        if self.action_buffer:
            action_to_execute = self.action_buffer.popleft()
        else:
            # Fallback: use the current action if queue is empty
            logging.warning("Action buffer empty, executing current action.")
            action_to_execute = action.copy()

        # --- Step Environment ---
        # Ensure action is the correct shape for the base env (expects batch dim)
        obs, reward, done, info = self.env.step(action_to_execute.reshape((1, 2)))

        truncated = False
        if self.current_step >= self._max_episode_steps:
            truncated = True

        linear_velocity = obs["linear_vels_x"][0]
        frenet_arc_length = obs["state_frenet"][0][0]
        frenet_lateral_offset = obs["state_frenet"][0][1]
        collision = obs["collisions"][0]
        angular_velocity = obs["ang_vels_z"][0]
        lap_counts = obs['lap_counts'][0]
        if (self.last_frenet_arc_length is not None and frenet_arc_length - self.last_frenet_arc_length < -10):
            self.last_frenet_arc_length = None
            
        # reward
        progress_reward = frenet_arc_length - self.last_frenet_arc_length if self.last_frenet_arc_length is not None else 0
        safety_distance_reward = 0.5 - abs(frenet_lateral_offset)
        linear_velocity_reward = abs(linear_velocity) - 1
        collision_punishment = -1 if collision else 0
        angular_velocity_punishment = - abs(angular_velocity)
        reward = progress_reward * 10 + safety_distance_reward * 0 + linear_velocity_reward * 1 + collision_punishment * 5000 + angular_velocity_punishment * 0 
        
        logging.debug(f"step: {self.current_step}")
        logging.debug(f"linear velocity: {linear_velocity:.2f}")
        logging.debug(f"angular velocity: {angular_velocity:.2f}")
        logging.debug(f"frenet arc length: {frenet_arc_length:.2f}")
        logging.debug(f"frenet lateral offset: {frenet_lateral_offset:.2f}")
        logging.debug(f"collision: {collision}")
        logging.debug(f"truncated: {truncated}")
        logging.debug(f"done: {done}")
        logging.debug(f"progress reward: {progress_reward:.2f}")
        logging.debug(f"safety distance reward: {safety_distance_reward:.2f}") # Changed name back
        logging.debug(f"linear velocity reward: {linear_velocity_reward:.2f}")
        logging.debug(f"collision punishment: {collision_punishment:.2f}")
        logging.debug(f"angular velocity punishment: {angular_velocity_punishment:.2f}")
        logging.debug(f"total reward: {reward:.2f}")

        self.last_frenet_arc_length = frenet_arc_length
        self.last_lap_counts = lap_counts

        # Process observation before returning
        processed_obs = self._process_observation(obs)

        # Done condition includes collision
        done = bool(done or collision)

        return processed_obs, reward, done, truncated, info # Gymnasium expects 5 return values

    def render(self, mode='human_fast'): # Default to faster rendering
        self.env.render(mode=mode)

    def close(self):
        self.env.close()