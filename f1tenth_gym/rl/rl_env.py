import gymnasium
import gym  # Keep the original gym import
import numpy as np
import torch
from gymnasium import spaces
import random
from absl import logging


class F110GymWrapper(gymnasium.Env):
    def __init__(self, waypoints, seed, map_path, num_agents):
        super().__init__()
        self.env = gym.make(  # Use gym.make instead of gymnasium.make
            'f110_gym:f110-v0', 
            model='kinematic_ST',
            map=map_path, 
            seed=seed,
            num_agents=num_agents, 
            # drive_control_mode='acc',
            # steering_control_mode='vel', 
            waypoints=waypoints,
            timestep=0.01,
        )
        
        # Update observation space to match reduced dimensions (velocity, yaw, lidar)
        self.observation_space = spaces.Box(
            low=0, high=30, shape=(1082,), dtype=np.float32  # 2 state values + 1080 lidar points
        )
        # self.action_space = spaces.Box(
        #     low=np.array([-3.2, 0.0]), high=np.array([3.2, 9.51]), shape=(2,), dtype=np.float32
        # )
        self.action_space = spaces.Box(
            low=np.array([-0.4189, 1]), high=np.array([0.4189, 20]), shape=(2,), dtype=np.float32
        )
        self._max_episode_steps = 10000
        self.current_step = 0
        self.waypoints = waypoints
        self.last_frenet_arc_length = None
        self.last_lap_counts = 0

    def reset(self, seed=None, options=None):
        poses = self.get_reset_pose()
        obs, reward, done, info = self.env.reset(initial_states=poses)
        self.current_step = 0
        self.last_frenet_arc_length = None
        # Only include velocity, yaw angle, and lidar scans
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        return np.concatenate((
            obs['state'][0][3:5],  # vel, yaw_angle
            obs['scans'][0]
        )), {}
    
    def get_reset_pose(self):
        starting_idx = random.sample(range(len(self.waypoints)), 1)
        x, y = self.waypoints[starting_idx][0, 1], self.waypoints[starting_idx][0, 2]
        theta_noise = (2*random.random() - 1) * 0.1
        theta = self.waypoints[starting_idx][0, 3] + theta_noise
        starting_pos = np.array([[x, y, theta]])
        return starting_pos

    def step(self, action):
        self.current_step += 1
        obs, reward, done, info = self.env.step(action.reshape((1, 2)))
        
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
        reward = progress_reward * 10 + safety_distance_reward * 1 + linear_velocity_reward * 1 + collision_punishment * 1000 + angular_velocity_punishment * 0 
        
        logging.info(f"step: {self.current_step}")
        logging.info(f"linear velocity: {linear_velocity}")
        logging.info(f"angular velocity: {angular_velocity}")
        logging.info(f"frenet arc length: {frenet_arc_length}")
        logging.info(f"frenet lateral offset: {frenet_lateral_offset}")
        logging.info(f"collision: {collision}")
        logging.info(f"truncated: {truncated}")
        logging.info(f"done: {done}")
        logging.info(f"progress reward: {progress_reward}")
        logging.info(f"safety distance reward: {safety_distance_reward}")
        logging.info(f"linear velocity reward: {linear_velocity_reward}")
        logging.info(f"collision punishment: {collision_punishment}")
        logging.info(f"angular velocity punishment: {angular_velocity_punishment}")
        logging.info(f"reward: {reward}")
        
        self.last_frenet_arc_length = frenet_arc_length
        self.last_lap_counts = lap_counts
        
        # Return only velocity, yaw angle, and lidar scans
        return np.concatenate((
            obs['state'][0][3:5],  # vel, yaw_angle
            obs['scans'][0]
        )), float(reward), done or collision, truncated, info

    def render(self):
        self.env.render(mode='human_fast')

    def close(self):
        self.env.close()