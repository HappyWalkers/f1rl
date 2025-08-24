import gymnasium
import gym  # Keep the original gym import
import numpy as np
import torch
from gymnasium import spaces
import random
from absl import logging
from utils.Track import Track
import collections
from pure_pursuit import PurePursuitPolicy


class F110GymWrapper(gymnasium.Env):
    def __init__(self, 
                 waypoints, 
                 seed, 
                 map_path, 
                 num_agents, 
                 track: Track,
                 max_episode_steps=1000,
                 # Racing Mode
                 racing_mode=False,
                 opponent_policy_type="PURE_PURSUIT",
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
                 push_2_prob=0,  # Probability of pushing 2 times
                 include_params_in_obs=True,        # Whether to include parameters in observation
                 lidar_scan_in_obs_mode="FULL"     # Lidar mode: NONE, FULL, or DOWNSAMPLED
                 ):
        super().__init__()
        self.track = track # Store track object
        self.seed = seed
        self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        self.include_params_in_obs = include_params_in_obs
        self.include_lidar_in_obs = lidar_scan_in_obs_mode != "NONE"
        self.lidar_scan_in_obs_mode = lidar_scan_in_obs_mode
        
        # Set lidar dimensions based on mode
        if lidar_scan_in_obs_mode == "NONE":
            self.lidar_dim = 0
        elif lidar_scan_in_obs_mode == "FULL":
            self.lidar_dim = 1080
        elif lidar_scan_in_obs_mode == "DOWNSAMPLED":
            self.lidar_dim = 108  # 1080 / 10
        else:
            raise ValueError(f"Unknown lidar_scan_in_obs_mode: {lidar_scan_in_obs_mode}")
        
        # Racing mode settings
        self.racing_mode = racing_mode
        self.opponent_policy_type = opponent_policy_type
        self.opponent_policy = None
        self.opponent_obs = None
        self.min_distance_between_cars = 2.0  # Minimum safe distance between cars at reset
        self.rl_agent_idx = 0 if racing_mode else 0 # In racing mode, RL agent is idx 0
        
        # Simulation timestep
        self.timestep = 0.02
        
        # Set number of agents based on racing mode
        if racing_mode and num_agents < 2:
            logging.info("Setting num_agents to 2 for racing mode")
            self.num_agents = 2
        else:
            self.num_agents = num_agents

        # Variables to track previous values for acceleration and steering angle speed rewards
        self.last_linear_velocity = None
        self.last_steering_angle = None

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
            num_agents=self.num_agents,
            # drive_control_mode='acc',
            # steering_control_mode='vel',
            waypoints=waypoints,
            timestep=self.timestep,
        )

        # Define the parameter vector dimensions - only include domain randomized params
        self.num_params = 12
        
        # Calculate observation space dimensions
        state_dim = 4  # [s, ey, vel, yaw_angle]
        lidar_dim = self.lidar_dim if self.include_lidar_in_obs else 0
        params_dim = self.num_params if self.include_params_in_obs else 0
        total_obs_dim = state_dim + lidar_dim + params_dim
        
        # Update observation space: [s, ey, vel, yaw_angle] + lidar (optional) + params (optional)
        # low/high values are approximate, might need refinement
        low_values = [-1000.0, -5.0, v_min, -np.pi]
        high_values = [1000.0, 5.0, v_max, np.pi]
        
        if self.include_lidar_in_obs:
            low_values.extend(np.zeros(self.lidar_dim))
            high_values.extend(np.full(self.lidar_dim, 30.0))
            
        if self.include_params_in_obs:
            low_values.extend(np.zeros(self.num_params))
            high_values.extend(np.ones(self.num_params) * 10)
        
        self.observation_space = spaces.Box(
            low=np.array(low_values),
            high=np.array(high_values),
            shape=(total_obs_dim,), dtype=np.float32
        )
            
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
        
        # Initialize opponent policy if in racing mode
        if self.racing_mode:
            self._initialize_opponent_policy()

    def _initialize_opponent_policy(self):
        """Initialize the opponent policy based on the specified type"""
        if self.opponent_policy_type == "PURE_PURSUIT":
            self.opponent_policy = PurePursuitPolicy(self.track)
            logging.info("Initialized Pure Pursuit policy for opponent")
        else:
            logging.warning(f"Unknown opponent policy type: {self.opponent_policy_type}, defaulting to Pure Pursuit")
            self.opponent_policy = PurePursuitPolicy(self.track)

    def get_env_params_vector(self):
        """Returns a vector of the current environment parameters for inclusion in observation."""
        params = self.get_env_params()
        # Convert dictionary to vector in a consistent order - only include domain randomized params
        param_vector = np.array([
            params['mu'], params['C_Sf'], params['C_Sr'], 
            params['m'], params['I'],
            params['lidar_noise_stddev'], params['s_noise_stddev'],
            params['ey_noise_stddev'], params['vel_noise_stddev'],
            params['yaw_noise_stddev'], params['push_0_prob'],
            params['push_2_prob']
        ], dtype=np.float32)
        
        # Remove manual normalization - VecNormalize will handle this
        # param_vector = param_vector / np.array([
        #    1.1, 5.5, 5.5,     # mu, C_Sf, C_Sr
        #    4.5, 0.06,         # m, I
        #    0.01, 0.01, 0.01, 0.01, 0.01,  # noise stddevs
        #    0.01, 0.01         # push probabilities
        # ])
        
        return param_vector

    def get_env_params(self):
        """Returns a dictionary of the current environment parameters."""
        params = self.base_env_params.copy()
        params.update({
            'lidar_noise_stddev': self.lidar_noise_stddev,
            's_noise_stddev': self.s_noise_stddev,
            'ey_noise_stddev': self.ey_noise_stddev,
            'vel_noise_stddev': self.vel_noise_stddev,
            'yaw_noise_stddev': self.yaw_noise_stddev,
            'push_0_prob': self.push_0_prob,
            'push_2_prob': self.push_2_prob,
            'push_1_prob': self.push_1_prob,
            # Add any other parameters that might be randomized or relevant
            # 'max_episode_steps': self._max_episode_steps # Example
        })
        return params

    def _process_lidar_scan(self, lidar_scan):
        """
        Process lidar scan based on the observation mode.
        
        Args:
            lidar_scan: Raw lidar scan (1080 points)
            
        Returns:
            Processed lidar scan based on mode
        """
        if self.lidar_scan_in_obs_mode == "NONE":
            return np.array([])
        elif self.lidar_scan_in_obs_mode == "FULL":
            return lidar_scan
        elif self.lidar_scan_in_obs_mode == "DOWNSAMPLED":
            # Pick every 10th point (1080 / 10 = 108 points)
            return lidar_scan[::10]
        else:
            raise ValueError(f"Unknown lidar_scan_in_obs_mode: {self.lidar_scan_in_obs_mode}")

    def _process_observation(self, obs, agent_idx=None):
        """
        Processes the raw observation dict and applies noise.
        
        Args:
            obs: Raw observation dictionary
            agent_idx: Index of the agent whose observation to process. 
                       If None and in racing mode, uses self.rl_agent_idx.
                       If None and not in racing mode, uses 0.
        """
        if agent_idx is None:
            agent_idx = self.rl_agent_idx if self.racing_mode else 0
            
        lidar_scan = obs['scans'][agent_idx]

        # Get the state components
        s_raw, ey_raw = obs["state_frenet"][agent_idx][0:2]  # Frenet coordinates
        vel_raw = obs['state'][agent_idx][3] 
        yaw_angle_raw = obs['state'][agent_idx][4]

        # Clip raw state values to plausible physical limits before applying noise
        # This helps to prevent np.inf or extreme values from the simulator state from corrupting downstream processing
        s = np.clip(s_raw, -1000.0, 1000.0)  # Example: Large track arc length
        ey = np.clip(ey_raw, -5.0, 5.0)      # Example: Max lateral error (e.g., track width)
        
        # Use v_min/v_max from base_env_params for velocity clipping
        current_v_min = self.base_env_params.get('v_min', -5.0)
        current_v_max = self.base_env_params.get('v_max', 12.0)
        vel = np.clip(vel_raw, current_v_min - 5.0, current_v_max + 5.0) # Clip with some margin
        
        yaw_angle = np.clip(yaw_angle_raw, -np.pi - 0.5, np.pi + 0.5) # Clip with some margin before normalization

        # Apply noise to state components
        if self.s_noise_stddev > 0:
            s += self.np_random.normal(0, self.s_noise_stddev)
        
        if self.ey_noise_stddev > 0:
            ey += self.np_random.normal(0, self.ey_noise_stddev)
        
        if self.vel_noise_stddev > 0:
            vel += self.np_random.normal(0, self.vel_noise_stddev)
        
        if self.yaw_noise_stddev > 0:
            yaw_angle += self.np_random.normal(0, self.yaw_noise_stddev)
            # Keep yaw angle in range [-pi, pi]
            yaw_angle = (yaw_angle + np.pi) % (2 * np.pi) - np.pi

        # Apply lidar noise
        if self.lidar_noise_stddev > 0:
            # First add Gaussian noise
            noise = self.np_random.normal(0, self.lidar_noise_stddev, lidar_scan.shape)
            lidar_scan += noise * lidar_scan
            
            # # Then randomly mask out some values to zero
            # mask_probability = self.lidar_noise_stddev
            # mask = self.np_random.random(lidar_scan.shape) < mask_probability
            # lidar_scan[mask] = 0.0
            
            # Clip values to valid range
            lidar_scan = np.clip(lidar_scan, 0, 30.0) # Assuming max range is 30

        # Process lidar scan based on mode (downsample if needed)
        processed_lidar = self._process_lidar_scan(lidar_scan)

        # Base observation without environment parameters
        base_obs_components = [s, ey, vel, yaw_angle]
        
        if self.include_lidar_in_obs:
            base_obs_components.extend(processed_lidar)
            
        base_obs = np.array(base_obs_components, dtype=np.float32)
        
        # Add environment parameters to observation if enabled
        if self.include_params_in_obs:
            env_params = self.get_env_params_vector()
            processed_obs = np.concatenate((base_obs, env_params)).astype(np.float32)
        else:
            processed_obs = base_obs
            
        return processed_obs

    def reset(self, seed=None, options=None):
        poses = self.get_reset_pose()
        obs, reward, done, info = self.env.reset(initial_states=poses)
        self.current_step = 0
        self.last_frenet_arc_length = None
        self.last_lap_counts = 0
        
        # Reset previous velocity and steering angle
        self.last_linear_velocity = None
        self.last_steering_angle = None
        
        self.action_buffer.clear() # Clear action buffer on reset
        
        # Store observation for opponent in racing mode (now index 1)
        if self.racing_mode:
            self.opponent_obs = self._process_observation(obs, agent_idx=1)
        
        # Return processed observation for RL agent
        processed_obs = self._process_observation(obs)
        return processed_obs, {}

    def get_reset_pose(self):
        """
        Get the reset poses for all agents.
        
        In racing mode:
        - Randomly selects one of four starting scenarios:
          1. RL agent behind the opponent (original behavior)
          2. RL agent in front of the opponent
          3. RL agent at the side of opponent 
          4. Random positions for both cars
        
        In non-racing mode:
        - Places a single agent at a random position
        
        Returns:
            starting_poses: numpy array of shape (num_agents, 3) with [x, y, theta] for each agent
        """
        if not self.racing_mode:
            # Original single-agent code
            starting_idx = random.sample(range(len(self.waypoints)), 1)[0]
            x, y = self.waypoints[starting_idx][1], self.waypoints[starting_idx][2]
            theta_noise = (2*random.random() - 1) * 0.5
            theta = self.waypoints[starting_idx][3] + theta_noise
            starting_pos = np.array([[x, y, theta]])
            return starting_pos
        else:
            # Racing mode: place two cars
            starting_poses = np.zeros((self.num_agents, 3))
            
            # Choose scenario (4th scenario has lower probability)
            scenario = random.choices([1, 2, 3, 4], weights=[0.8, 0.1, 0.0, 0.1])[0]
            
            if scenario == 4:
                # Scenario 4: Random positions for both cars
                # Pick two random positions making sure they're not too close
                indices = []
                while len(indices) < 2:
                    idx = random.randrange(len(self.waypoints))
                    if not indices or abs(self.waypoints[idx][0] - self.waypoints[indices[0]][0]) > self.min_distance_between_cars:
                        indices.append(idx)
                
                # Set opponent position (now index 1)
                starting_poses[1] = [
                    self.waypoints[indices[0]][1],  # x
                    self.waypoints[indices[0]][2],  # y
                    self.waypoints[indices[0]][3]   # theta
                ]
                
                # Set RL agent position (now index 0)
                starting_poses[0] = [
                    self.waypoints[indices[1]][1],  # x
                    self.waypoints[indices[1]][2],  # y
                    self.waypoints[indices[1]][3]   # theta
                ]
                
                # Add small noise to prevent cars from being exactly aligned
                noise = lambda: (2*random.random() - 1) * 0.1
                starting_poses[1, 2] += noise() * 0.05  # Small theta noise for opponent
                starting_poses[0, 2] += noise() * 0.05  # Small theta noise for RL agent
                
                return starting_poses
            
            # For scenarios 1-3, first place opponent at random position (now index 1)
            opponent_idx = random.sample(range(len(self.waypoints)), 1)[0]
            x_opponent = self.waypoints[opponent_idx][1]
            y_opponent = self.waypoints[opponent_idx][2]
            theta_opponent = self.waypoints[opponent_idx][3]
            starting_poses[1] = [x_opponent, y_opponent, theta_opponent]
            
            # Get s-position of opponent
            s_opponent = self.waypoints[opponent_idx][0]
            
            if scenario == 1:
                # Scenario 1: RL agent behind opponent 
                s_agent = (s_opponent - self.min_distance_between_cars) % self.track.s_frame_max
                x_agent, y_agent, theta_agent = self.track.frenet_to_cartesian(s_agent, 0, 0)
                
            elif scenario == 2:
                # Scenario 2: RL agent in front of opponent
                s_agent = (s_opponent + self.min_distance_between_cars) % self.track.s_frame_max
                x_agent, y_agent, theta_agent = self.track.frenet_to_cartesian(s_agent, 0, 0)
                
            elif scenario == 3:
                # Scenario 3: RL agent at the side of opponent
                # Use same s-coordinate but with lateral offset
                s_agent = s_opponent
                lateral_offset = random.choice([-0.5, 0.5])  # Left or right side
                x_agent, y_agent, theta_agent = self.track.frenet_to_cartesian(s_agent, lateral_offset, 0)
            
            # Add small noise to prevent cars from being exactly aligned
            lateral_noise = (2*random.random() - 1) * 0.1
            theta_noise = (2*random.random() - 1) * 0.05
            
            # Apply noise to RL agent's position
            x_agent += lateral_noise * np.sin(theta_agent + np.pi/2)
            y_agent += lateral_noise * np.cos(theta_agent + np.pi/2)
            theta_agent += theta_noise
            
            # Set RL agent position (now index 0)
            starting_poses[0] = [x_agent, y_agent, theta_agent]
            
            return starting_poses

    def _update_action_buffer(self, action):
        """
        Updates the action buffer according to the domain randomization parameters
        and returns the next action to execute.
        
        Args:
            action: The current action from the agent
            
        Returns:
            The action to execute (from the buffer or the current action)
        """
        # Push 0, 1, or 2 times with average 1
        num_pushes = self.np_random.choice([0, 1, 2], p=[self.push_0_prob, self.push_1_prob, self.push_2_prob])
        for _ in range(num_pushes):
            self.action_buffer.append(action.copy())
        
        # Ensure buffer is not empty
        if not self.action_buffer:
            self.action_buffer.append(action.copy())
        
        # Get the action to execute from buffer
        return self.action_buffer.popleft()

    def step(self, action):
        self.current_step += 1

        if self.racing_mode:
            # In racing mode, we need to get the opponent's action
            opponent_action, _ = self.opponent_policy.predict(self.opponent_obs)
            
            # Combine RL agent's action with opponent's action
            combined_actions = np.zeros((self.num_agents, 2))
            combined_actions[1] = opponent_action  # Opponent (now index 1)
            
            # Update action buffer and get action to execute for RL agent
            combined_actions[0] = self._update_action_buffer(action)
            
            # Get current steering angle before stepping the environment
            current_steering_angle = combined_actions[0][0] if self.last_steering_angle is not None else None
            
            # Step the environment with combined actions
            try:
                obs, reward, done, info = self.env.step(combined_actions)
            except (ValueError, RuntimeError) as e:
                if "Invalid state detected" in str(e) or "inf" in str(e) or "nan" in str(e).lower():
                    logging.error(f"Numerical instability in simulation detected: {e}")
                    logging.error(f"Actions that caused the error: {combined_actions}")
                raise e
            
            # Update opponent observation for next step (now index 1)
            self.opponent_obs = self._process_observation(obs, agent_idx=1)
            
        else:
            # Original single-agent code
            # Update action buffer and get action to execute
            action_to_execute = self._update_action_buffer(action)
            
            # Get current steering angle before stepping the environment
            current_steering_angle = action_to_execute[0] if self.last_steering_angle is not None else None

            # --- Step Environment ---
            # Ensure action is the correct shape for the base env (expects batch dim)
            logging.debug(f"action_to_execute: {action_to_execute}")
            try:
                obs, reward, done, info = self.env.step(action_to_execute.reshape((1, 2)))
            except (ValueError, RuntimeError) as e:
                if "Invalid state detected" in str(e) or "inf" in str(e) or "nan" in str(e).lower():
                    logging.error(f"Numerical instability in simulation detected: {e}")
                    logging.error(f"Action that caused the error: {action_to_execute}")
                raise e

        logging.debug(f"obs: {obs}")

        truncated = False
        if self.current_step >= self._max_episode_steps:
            truncated = True

        # Get values for RL agent (agent 0)
        agent_idx = self.rl_agent_idx if self.racing_mode else 0
        linear_velocity = obs["linear_vels_x"][agent_idx]
        frenet_arc_length = obs["state_frenet"][agent_idx][0]
        frenet_lateral_offset = obs["state_frenet"][agent_idx][1]
        collision = obs["collisions"][agent_idx]
        angular_velocity = obs["ang_vels_z"][agent_idx]
        lap_counts = obs['lap_counts'][agent_idx]
        
        # Calculate acceleration and steering angle speed if we have previous values
        linear_acceleration = 0
        steering_angle_speed = 0
        
        if self.last_linear_velocity is not None:
            linear_acceleration = abs(linear_velocity - self.last_linear_velocity) / self.timestep
            
        if self.last_steering_angle is not None and current_steering_angle is not None:
            steering_angle_speed = abs(current_steering_angle - self.last_steering_angle) / self.timestep
        
        # Store current values for next step
        self.last_linear_velocity = linear_velocity
        self.last_steering_angle = current_steering_angle
        
        if (self.last_frenet_arc_length is not None and frenet_arc_length - self.last_frenet_arc_length < -10):
            self.last_frenet_arc_length = None
            
        # reward
        progress_reward = frenet_arc_length - self.last_frenet_arc_length if self.last_frenet_arc_length is not None else 0
        safety_distance_reward = 0.5 - abs(frenet_lateral_offset)
        linear_velocity_reward = abs(linear_velocity) - 1
        collision_punishment = -1 if collision else 0
        angular_velocity_punishment = - abs(angular_velocity)
        acceleration_punishment = - linear_acceleration
        steering_speed_punishment = - steering_angle_speed
        
        # Adjust reward for racing scenario
        if self.racing_mode:
            # reward for overtaking the opponent
            opponent_s = obs["state_frenet"][1][0] # Opponent is now index 1
            agent_s = frenet_arc_length 
            s_diff = agent_s - opponent_s
            overtake_reward = np.tanh(s_diff)
            reward = (progress_reward * 10 + 
                     safety_distance_reward * 0 + 
                     linear_velocity_reward * 1 + 
                     collision_punishment * 1000 + 
                     angular_velocity_punishment * 0 + 
                     overtake_reward * 0 +
                     acceleration_punishment * 0.01 +
                     steering_speed_punishment * 0.01)
        else:
            # Original reward function for single-agent with added punishments
            reward = (progress_reward * 20 + 
                     safety_distance_reward * 0 + 
                     linear_velocity_reward * 1 + 
                     collision_punishment * 1000 + 
                     angular_velocity_punishment * 0 +
                     acceleration_punishment * 0 +
                     steering_speed_punishment * 0.05)
        
        logging.debug(f"step: {self.current_step}")
        logging.debug(f"linear velocity: {linear_velocity:.2f}")
        logging.debug(f"angular velocity: {angular_velocity:.2f}")
        logging.debug(f"frenet arc length: {frenet_arc_length:.2f}")
        logging.debug(f"frenet lateral offset: {frenet_lateral_offset:.2f}")
        logging.debug(f"collision: {collision}")
        logging.debug(f"truncated: {truncated}")
        logging.debug(f"done: {done}")
        logging.debug(f"progress reward: {progress_reward:.2f}")
        logging.debug(f"safety distance reward: {safety_distance_reward:.2f}") 
        logging.debug(f"linear velocity reward: {linear_velocity_reward:.2f}")
        logging.debug(f"collision punishment: {collision_punishment:.2f}")
        logging.debug(f"angular velocity punishment: {angular_velocity_punishment:.2f}")
        logging.debug(f"acceleration punishment: {acceleration_punishment:.2f}")
        logging.debug(f"steering speed punishment: {steering_speed_punishment:.2f}")
        if self.racing_mode:
            logging.debug(f"opponent s: {opponent_s:.2f}")
            logging.debug(f"agent s: {agent_s:.2f}")
            logging.debug(f"s_diff: {s_diff:.2f}")
            logging.debug(f"overtake reward: {overtake_reward:.2f}")
        logging.debug(f"total reward: {reward:.2f}")

        self.last_frenet_arc_length = frenet_arc_length
        self.last_lap_counts = lap_counts

        # Process observation before returning
        processed_obs = self._process_observation(obs)

        # In racing mode, check for both cars' collisions
        if self.racing_mode:
            # Race is done if RL agent collides or either car completes the race
            done = bool(done or collision or np.any(obs["collisions"]))
        else:
            # Original done condition
            done = bool(done or collision)
            
        info["reward"] = reward
        info["done"] = done
        info["truncated"] = truncated
        info["current_step"] = self.current_step
        info["max_steps"] = self._max_episode_steps
        info["collision"] = collision

        return processed_obs, reward, done, truncated, info # Gymnasium expects 5 return values

    def render(self, mode='human_fast'): # Default to faster rendering
        self.env.render(mode=mode)

    def close(self):
        self.env.close()