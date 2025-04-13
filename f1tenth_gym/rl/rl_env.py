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
                 mu=0.6,                         # Friction coefficient
                 C_Sf=4.718,                        # Cornering stiffness coefficient, front
                 C_Sr=5.4562,                       # Cornering stiffness coefficient, rear
                 lf=0.15875,                        # Distance from center of gravity to front axle
                 lr=0.17145,                        # Distance from center of gravity to rear axle
                 h=0.074,                           # Height of center of gravity
                 m=3.74,                            # Total mass of the vehicle
                 I=0.04712,                         # Moment of inertial of the entire vehicle about the z axis
                 s_min=-0.4189,                     # Minimum steering angle constraint
                 s_max=0.4189,                      # Maximum steering angle constraint
                 sv_min=-3.2,                       # Minimum steering velocity constraint
                 sv_max=3.2,                        # Maximum steering velocity constraint
                 v_switch=7.319,                    # Switching velocity
                 a_max=4,                        # Maximum acceleration
                 v_min=-5.0,                        # Minimum velocity
                 v_max=12.0,                        # Maximum velocity
                 width=0.31,                        # Width of the vehicle
                 length=0.58,                       # Length of the vehicle
                 lidar_noise_stddev=0.0,            # Noise std dev for lidar
                 s_noise_stddev=0.0,               # Noise std dev for s (arc length)
                 ey_noise_stddev=0.0,             # Noise std dev for ey (lateral offset)
                 vel_noise_stddev=0.0,             # Noise std dev for velocity
                 yaw_noise_stddev=0.0,            # Noise std dev for yaw angle (radians)
                 push_0_prob=0, # Probability of pushing 0 times
                 push_2_prob=0  # Probability of pushing 2 times
                 ):
        super().__init__()
        self.track = track # Store track object
        self.seed = seed
        self.np_random, _ = gymnasium.utils.seeding.np_random(seed)

        # Store DR params
        self.lidar_noise_stddev = lidar_noise_stddev
        self.s_noise_stddev = s_noise_stddev
        self.ey_noise_stddev = ey_noise_stddev
        self.vel_noise_stddev = vel_noise_stddev
        self.yaw_noise_stddev = yaw_noise_stddev
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
            timestep=0.03,
        )

        # Update observation space: [s, ey, vel, yaw_angle] + lidar
        # low/high values are approximate, might need refinement
        self.observation_space = spaces.Box(
            low=np.concatenate(([-1000.0, -5.0, v_min, -np.pi], np.zeros(1080))),
            high=np.concatenate(([1000.0, 5.0, v_max, np.pi], np.full(1080, 30.0))),
            shape=(1084,), dtype=np.float32  # 4 state values + 1080 lidar points
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
        """Processes the raw observation dict and applies noise."""
        lidar_scan = obs['scans'][0]

        # Get the state components
        s, ey = obs["state_frenet"][0][0:2]  # Frenet coordinates
        vel, yaw_angle = obs['state'][0][3:5]  # Velocity and yaw angle

        # Apply noise to state components
        if self.s_noise_stddev > 0:
            s += self.np_random.normal(0, self.s_noise_stddev) * s if s != 0 else self.np_random.normal(0, self.s_noise_stddev)
        
        if self.ey_noise_stddev > 0:
            ey += self.np_random.normal(0, self.ey_noise_stddev) * ey if ey != 0 else self.np_random.normal(0, self.ey_noise_stddev)
        
        if self.vel_noise_stddev > 0:
            vel += self.np_random.normal(0, self.vel_noise_stddev) * vel if vel != 0 else self.np_random.normal(0, self.vel_noise_stddev)
            vel = max(vel, 0.0)  # Ensure velocity doesn't go negative
        
        if self.yaw_noise_stddev > 0:
            yaw_angle += self.np_random.normal(0, self.yaw_noise_stddev) * yaw_angle if yaw_angle != 0 else self.np_random.normal(0, self.yaw_noise_stddev)
            # Keep yaw angle in range [-pi, pi]
            yaw_angle = (yaw_angle + np.pi) % (2 * np.pi) - np.pi

        # Apply lidar noise
        if self.lidar_noise_stddev > 0:
            noise = self.np_random.normal(0, self.lidar_noise_stddev, lidar_scan.shape)
            lidar_scan += noise * lidar_scan
            lidar_scan = np.clip(lidar_scan, 0, 30.0) # Assuming max range is 30

        # Combine all observation components with noise
        processed_obs = np.concatenate((
            [s, ey],  # Frenet coordinates with noise
            [vel, yaw_angle],  # Velocity and yaw angle with noise
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