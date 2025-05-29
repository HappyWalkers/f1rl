#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from stable_baselines3 import SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
import os
import time
from gymnasium import spaces
import sys
import argparse
import gymnasium
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import interp1d
import torch  # Added for device checking
from typing import Tuple
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.Track import Track
from utils import utils # Added import

class DummyEnv(gymnasium.Env):
    """
    Dummy Gymnasium environment for loading VecNormalize statistics
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.np_random = None
        
    def reset(self, seed=None, options=None):
        # Implementation of proper seeding
        super().reset(seed=seed)
        # Simple zeroed observation
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
    def step(self, action):
        # Return observation, reward, terminated, truncated, info
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}
        
    def render(self):
        pass
        
    def close(self):
        pass

class RLF1TenthController(Node):
    """
    ROS Node that uses a trained RL model to control an F1Tenth car
    """
    def __init__(self, algorithm="SAC", model_path="./logs/best_model/best_model.zip", 
                 vecnorm_path=None, map_index=63, map_dir_="./f1tenth_racetracks/",
                 lidar_scan_in_obs_mode="FULL", enable_lidar_plot=True):
        super().__init__('rl_f1tenth_controller')
        
        # Topics
        lidar_topic = '/scan'
        odom_topic = '/ego_racecar/odom'
        # odom_topic = '/pf/pose/odom'
        drive_topic = '/drive'
        
        # Configure QoS profiles for real-time, low-latency operation
        self._setup_qos_profiles()
        
        # Store configuration as attributes
        self.algorithm = algorithm
        self.lidar_scan_in_obs_mode = lidar_scan_in_obs_mode
        self.enable_lidar_plot = enable_lidar_plot
        model_path = os.path.expanduser(model_path)
        
        # Add last steering angle tracking
        self.last_steering_angle = 0.0
        
        # Enhanced drive message calculation variables
        self.control_loop_dt = 0.02  # Control loop frequency (50 Hz)
        self.previous_velocity = 0.0  # Track previous velocity for acceleration calculation
        self.previous_acceleration = 0.0  # Track previous acceleration for jerk calculation
        self.previous_steering_angle = 0.0  # Track previous steering angle for rate calculation
        self.previous_timestamp = None  # Track previous timestamp for accurate dt calculation
        
        # Track previous commanded values for acceleration limiting
        self.previous_commanded_speed = 0.0  # Track previous commanded speed
        self.previous_commanded_steering = 0.0  # Track previous commanded steering
        self.max_acceleration = 4.0  # Maximum allowed acceleration (m/s²)
        self.max_steering_velocity = 3.2  # Maximum allowed steering velocity (rad/s)
        
        # For recurrent policies (LSTM state tracking)
        self.is_recurrent = self.algorithm == "RECURRENT_PPO"
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)  # Mark the start of the episode
        
        # Initialize matplotlib figure for lidar visualization
        self.fig = plt.figure(figsize=(12, 10))
        
        # Create two subplots
        self.ax_polar = self.fig.add_subplot(211, projection='polar')
        self.ax_range = self.fig.add_subplot(212)
        
        plt.ion()  # Enable interactive mode
        self.lidar_plot = None
        self.show_lidar_plot = True
        
        # Try to infer vecnorm_path if not provided
        if vecnorm_path is None and model_path is not None:
            model_dir = os.path.dirname(model_path)
            potential_vecnorm_path = os.path.join(model_dir, "vec_normalize.pkl")
            if os.path.exists(potential_vecnorm_path):
                vecnorm_path = potential_vecnorm_path
                self.get_logger().info(f"Found VecNormalize statistics at {vecnorm_path}")
        
        self.vecnorm_path = vecnorm_path
        self.vec_normalize = None
        
        # Load the trained model
        self.get_logger().info(f"Loading {self.algorithm} model from {model_path}")
        try:
            # Create observation and action spaces matching training environment
            # Calculate observation space dimensions based on lidar inclusion
            state_dim = 4  # [s, ey, vel, yaw_angle]
            if self.lidar_scan_in_obs_mode == "NONE":
                lidar_dim = 0
            elif self.lidar_scan_in_obs_mode == "FULL":
                lidar_dim = 1080
            elif self.lidar_scan_in_obs_mode == "DOWNSAMPLED":
                lidar_dim = 108
            else:
                raise ValueError(f"Unknown lidar_scan_in_obs_mode: {self.lidar_scan_in_obs_mode}")
            total_obs_dim = state_dim + lidar_dim
            
            # Create observation space with appropriate dimensions
            low_values = [-1000.0, -5.0, -5.0, -np.pi]
            high_values = [1000.0, 5.0, 12.0, np.pi]
            
            if self.lidar_scan_in_obs_mode != "NONE":
                low_values.extend(np.zeros(lidar_dim))
                high_values.extend(np.full(lidar_dim, 30.0))
            
            observation_space = spaces.Box(
                low=np.array(low_values),
                high=np.array(high_values),
                shape=(total_obs_dim,), dtype=np.float32
            )
            action_space = spaces.Box(
                low=np.array([-0.4189, 1]), 
                high=np.array([0.4189, 20]), 
                shape=(2,), 
                dtype=np.float32
            )
            
            # Define custom objects to help with deserialization
            custom_objects = {
                "action_space": action_space,
                "observation_space": observation_space
            }
            
            # Load model based on algorithm type
            if self.algorithm == "RECURRENT_PPO":
                self.model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
                self.get_logger().info("RecurrentPPO model loaded successfully")
            else:
                self.model = SAC.load(model_path, custom_objects=custom_objects)
                self.get_logger().info("SAC model loaded successfully")
                
            # Double-check if model is recurrent based on its attributes
            if not self.is_recurrent and hasattr(self.model, 'policy') and hasattr(self.model.policy, '_initial_state'):
                self.get_logger().info("Detected recurrent policy, enabling LSTM state tracking")
                self.is_recurrent = True
                
            # Load VecNormalize statistics if available
            if self.vecnorm_path and os.path.exists(self.vecnorm_path):
                self.get_logger().info(f"Loading VecNormalize statistics from {self.vecnorm_path}")
                # Create a dummy environment with matching observation space
                from stable_baselines3.common.vec_env import DummyVecEnv
                from stable_baselines3.common.monitor import Monitor
                
                # Create a function to initialize the environment
                def make_dummy_env():
                    env = DummyEnv(observation_space, action_space)
                    env = Monitor(env)  # Monitor wrapper is required by SB3
                    return env
                
                # Create a vectorized environment
                dummy_vec_env = DummyVecEnv([make_dummy_env])
                
                # Load normalize statistics
                self.vec_normalize = VecNormalize.load(self.vecnorm_path, dummy_vec_env)
                # Disable training and reward normalization
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False
                self.get_logger().info("VecNormalize statistics loaded successfully")
            else:
                self.get_logger().warning("No VecNormalize statistics found, observations will not be normalized")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
        
        # State variables
        self.lidar_data = None
        self.odom_data = None
        self.state_ready = False
        self.s_guess = 0.0 # Initialize s_guess for Frenet conversion
        self.current_step = 0 # Track steps for debugging
        self.interpolation_mask = None  # Track which lidar points were interpolated

        # Load Track for Frenet Conversion
        class Config(utils.ConfigYAML): # Simple config mimicking main.py
            map_dir = map_dir_
            map_scale = 1
            map_ind = map_index

        config = Config()

        try:
            map_info_path = os.path.join(map_dir_, 'map_info.txt')
            if not os.path.exists(map_info_path):
                # Try relative to the file location if not found in cwd
                 script_dir = os.path.dirname(os.path.abspath(__file__))
                 map_info_path_alt = os.path.join(script_dir, '..', '..', '..', map_dir_, 'map_info.txt') # Adjust path as needed
                 if os.path.exists(map_info_path_alt):
                     map_info_path = map_info_path_alt
                     config.map_dir = os.path.dirname(map_info_path_alt) # Update config map_dir
                 else:
                     raise FileNotFoundError(f"map_info.txt not found in {map_dir_} or alternative path.")
            
            map_info = np.genfromtxt(map_info_path, delimiter='|', dtype='str')
            self.track, _ = Track.load_map(
                config.map_dir, map_info, config.map_ind, config, scale=config.map_scale, downsample_step=1
            )
            # s_frame_max is now handled within Track.load_map or Track.from_numpy
            self.get_logger().info(f"Track loaded successfully using map index {map_index}")
            
            # Store waypoints for reset functionality
            self.waypoints = self.track.waypoints  # Format: [s, x, y, psi, k, vx, ax]
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints for reset functionality")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load track using map index {map_index}: {e}")
            # Print traceback for more details
            import traceback
            traceback.print_exc()
            raise
        
        # Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, 
            lidar_topic, 
            self.lidar_callback, 
            self.sensor_qos  # Use sensor QoS for low-latency LiDAR data
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            self.odometry_qos  # Use odometry QoS for latest pose data
        )
        
        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            self.control_qos  # Use control QoS for reliable command delivery
        )
        
        # Create a timer for control loop
        self.timer = self.create_timer(0.02, self.control_loop)
        
        # QoS monitoring and statistics
        self._setup_qos_monitoring()
        
        # Collision detection and reset variables
        self.position_history = []  # Track recent positions
        self.velocity_history = []  # Track recent velocities
        self.command_history = []  # Track recent commands
        self.position_history_size = 20  # Number of recent positions to keep
        self.stuck_threshold = 0.1  # meters - if car moves less than this in N steps, it's stuck
        self.collision_threshold = 0.5  # cosine similarity threshold for velocity vs command alignment
        self.min_lidar_distance = 0.2  # meters - minimum safe distance from obstacles
        self.collision_detected = False
        self.reset_in_progress = False
        self.reset_timer = 0
        self.reset_duration = 100  # Number of control loops for reset sequence (increased for waypoint navigation)
        self.collision_count = 0
        self.last_collision_time = None
        self.consecutive_collision_threshold = 3  # Reset if stuck for this many consecutive checks
        self.consecutive_collision_count = 0
        
        # Reset controller state
        self.reset_target_waypoint = None
        self.reset_phase = 'backup'  # 'backup', 'align', 'complete'
        self.reset_backup_duration = 50  # Control loops for backing up
        self.reset_angle_tolerance = 0.2  # radians - tolerance for reaching target angle
        self.reset_kp_steering = 2.0  # Proportional gain for steering control during alignment
        
        # Reset pose publisher (for simulator reset)
        self.reset_pose_pub = self.create_publisher(
            Odometry,
            '/ego_racecar/reset_pose',
            self.reset_qos  # Use reset QoS for reliable reset commands
        )
        
        self.get_logger().info(f"RL F1Tenth Controller initialized with {self.algorithm} algorithm")
        
        # Print PyTorch device information at startup
        self.get_logger().info(f"PyTorch version: {torch.__version__}")
        self.get_logger().info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.get_logger().info(f"CUDA device count: {torch.cuda.device_count()}")
            self.get_logger().info(f"Current CUDA device: {torch.cuda.current_device()}")
            self.get_logger().info(f"CUDA device name: {torch.cuda.get_device_name()}")
        self.get_logger().info(f"Default device: {torch.tensor([1.0]).device}")

    def _setup_qos_profiles(self):
        """
        Setup QoS profiles optimized for real-time, low-latency autonomous racing.
        
        Based on ROS 2 QoS best practices for different message types:
        - Sensor data: Best effort, small queue for latest readings
        - Odometry: Best effort, keep last 1 for most recent pose
        - Control commands: Reliable delivery to ensure commands are received
        """
        # Sensor data QoS: Prioritize latest data over reliability
        # Use "best effort" for speed, small queue to minimize delay
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Only keep the latest reading
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Odometry QoS: Similar to sensor data but slightly more reliable
        # for critical state information
        self.odometry_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Only keep the latest pose
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Control command QoS: Ensure commands are delivered reliably
        # but with minimal buffering to avoid stale commands
        self.control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Only keep the latest command
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Reset pose QoS: Reliable delivery for critical reset commands
        self.reset_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.get_logger().info("QoS profiles configured for low-latency real-time control")
        self.get_logger().info("- LiDAR: Best effort, depth=1 (latest sensor data)")
        self.get_logger().info("- Odometry: Best effort, depth=1 (latest pose)")
        self.get_logger().info("- Drive commands: Reliable, depth=1 (ensure delivery)")
        self.get_logger().info("- Reset commands: Reliable, depth=1 (critical commands)")

    def _setup_qos_monitoring(self):
        """
        Setup QoS event monitoring to track message delivery performance.
        This helps identify any QoS-related issues in real-time.
        """
        # QoS statistics
        self.qos_stats = {
            'lidar_messages_received': 0,
            'odom_messages_received': 0,
            'drive_commands_sent': 0,
            'last_lidar_time': None,
            'last_odom_time': None,
            'max_lidar_interval': 0.0,
            'max_odom_interval': 0.0
        }
        
        # Create a timer for periodic QoS statistics reporting
        self.qos_stats_timer = self.create_timer(5.0, self._report_qos_statistics)
        
        self.get_logger().info("QoS monitoring initialized - statistics will be reported every 5 seconds")

    def _report_qos_statistics(self):
        """Report QoS and message delivery statistics periodically"""
        stats = self.qos_stats
        
        # Calculate message rates
        current_time = time.time()
        if stats['last_lidar_time'] and stats['last_odom_time']:
            lidar_rate = stats['lidar_messages_received'] / 5.0 if stats['lidar_messages_received'] > 0 else 0
            odom_rate = stats['odom_messages_received'] / 5.0 if stats['odom_messages_received'] > 0 else 0
            
            self.get_logger().info(
                f"QoS Stats - LiDAR: {lidar_rate:.1f}Hz (max gap: {stats['max_lidar_interval']:.3f}s), "
                f"Odom: {odom_rate:.1f}Hz (max gap: {stats['max_odom_interval']:.3f}s), "
                f"Drive commands: {stats['drive_commands_sent']}"
            )
        
        # Reset counters for next period
        stats['lidar_messages_received'] = 0
        stats['odom_messages_received'] = 0
        stats['drive_commands_sent'] = 0
        stats['max_lidar_interval'] = 0.0
        stats['max_odom_interval'] = 0.0

    def _process_lidar_scan(self, lidar_data):
        """
        Process lidar scan based on the observation mode.
        
        Args:
            lidar_data: Raw lidar scan (1080 points)
            
        Returns:
            Processed lidar scan based on mode
        """
        if self.lidar_scan_in_obs_mode == "NONE":
            return np.array([])
        
        # Common processing for both FULL and DOWNSAMPLED modes
        lidar_scan_processed = self._preprocess_lidar_data(lidar_data)
        self.lidar_data = lidar_scan_processed
        
        if self.lidar_scan_in_obs_mode == "FULL":
            return lidar_scan_processed
        elif self.lidar_scan_in_obs_mode == "DOWNSAMPLED":
            # Pick every 10th point (1080 / 10 = 108 points)
            return lidar_scan_processed[::10]
        else:
            raise ValueError(f"Unknown lidar_scan_in_obs_mode: {self.lidar_scan_in_obs_mode}")

    def _preprocess_lidar_data(self, lidar_data):
        """
        Common preprocessing steps for lidar data.
        
        Args:
            lidar_data: Raw lidar scan
            
        Returns:
            Preprocessed lidar scan with interpolated values
        """
        # Ensure lidar data has 1080 points, pad if necessary
        lidar_scan_processed = np.array(lidar_data[:1080], dtype=np.float32)
        if len(lidar_scan_processed) < 1080:
            lidar_scan_processed = np.pad(
                lidar_scan_processed, 
                (0, 1080 - len(lidar_scan_processed)), 
                'constant', 
                constant_values=30.0
            )
        
        # Handle NaNs or Infs in lidar data (replace with zeros for interpolation)
        lidar_scan_processed[np.isnan(lidar_scan_processed)] = 0
        lidar_scan_processed[np.isinf(lidar_scan_processed)] = 0
        
        # Interpolate small values using nearby valid readings and store interpolation mask
        lidar_scan_processed, self.interpolation_mask = self._interpolate_lidar_zeros(lidar_scan_processed)
        
        return lidar_scan_processed

    def _interpolate_lidar_zeros(self, lidar_scan):
        """
        Interpolate small values (<0.05) in lidar scan using scipy interpolation.
        
        Args:
            lidar_scan: numpy array of lidar readings
            
        Returns:
            tuple: (interpolated_scan, interpolation_mask)
                - interpolated_scan: lidar scan with interpolated values
                - interpolation_mask: boolean array indicating which points were interpolated
        """
        # Find indices of zero and non-zero values
        zero_mask = (lidar_scan < 0.05)
        valid_mask = ~zero_mask
        
        # If no zeros to interpolate, return original with empty interpolation mask
        if not np.any(zero_mask):
            self.get_logger().debug("All lidar values are valid - no need to interpolate")
            return lidar_scan, np.zeros_like(lidar_scan, dtype=bool)
        
        # If all values are zero, return original (can't interpolate)
        if not np.any(valid_mask):
            self.get_logger().warning("All lidar values are zero - cannot interpolate")
            return lidar_scan, np.zeros_like(lidar_scan, dtype=bool)
        
        # Create a copy to modify
        interpolated_scan = lidar_scan.copy()
        interpolation_mask = zero_mask.copy()  # Track which points were interpolated
        
        try:
            # Get indices for valid (non-zero) points
            valid_indices = np.where(valid_mask)[0]
            valid_values = lidar_scan[valid_indices]
            
            # Create interpolation function
            # Use 'linear' interpolation with bounds_error=False to handle edge cases
            interp_func = interp1d(
                valid_indices, 
                valid_values, 
                kind='linear', 
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Get indices where we need to interpolate
            zero_indices = np.where(zero_mask)[0]
            
            # Interpolate the zero values
            interpolated_values = interp_func(zero_indices)
            
            # Replace zero values with interpolated ones
            interpolated_scan[zero_indices] = interpolated_values
            
            # Ensure interpolated values are within reasonable bounds
            interpolated_scan = np.clip(interpolated_scan, 0.05, 30.0)
            
            # Log interpolation statistics occasionally
            num_interpolated = np.sum(zero_mask)
            if num_interpolated > 0:
                self.get_logger().debug(f"Interpolated {num_interpolated} zero lidar values")
            
        except Exception as e:
            self.get_logger().warning(f"Lidar interpolation failed: {e}, using original scan")
            # Fall back to replacing zeros with a reasonable default value
            interpolated_scan[zero_mask] = 0
            # Reset interpolation mask since we didn't actually interpolate
            interpolation_mask = np.zeros_like(lidar_scan, dtype=bool)
        
        return interpolated_scan, interpolation_mask

    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        current_time = time.time()
        
        # Update QoS statistics
        self.qos_stats['lidar_messages_received'] += 1
        if self.qos_stats['last_lidar_time'] is not None:
            interval = current_time - self.qos_stats['last_lidar_time']
            self.qos_stats['max_lidar_interval'] = max(self.qos_stats['max_lidar_interval'], interval)
        self.qos_stats['last_lidar_time'] = current_time
        
        self.lidar_data = msg.ranges
        self.check_state_ready()
    
    def odom_callback(self, msg):
        """Process odometry data"""
        current_time = time.time()
        
        # Update QoS statistics
        self.qos_stats['odom_messages_received'] += 1
        if self.qos_stats['last_odom_time'] is not None:
            interval = current_time - self.qos_stats['last_odom_time']
            self.qos_stats['max_odom_interval'] = max(self.qos_stats['max_odom_interval'], interval)
        self.qos_stats['last_odom_time'] = current_time
        
        pose = msg.pose.pose
        twist = msg.twist.twist
        
        # Extract position
        x = pose.position.x
        y = pose.position.y
        
        # Extract orientation
        orientation = pose.orientation
        # Convert quaternion to yaw angle
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Extract velocity
        linear_vel = twist.linear.x
        angular_vel = twist.angular.z
        
        # Initialize previous values on first odometry message
        if self.previous_timestamp is None:
            self.previous_velocity = linear_vel
            self.previous_acceleration = 0.0
            self.previous_steering_angle = 0.0
            self.previous_timestamp = current_time
            # Initialize commanded values
            self.previous_commanded_speed = linear_vel
            self.previous_commanded_steering = 0.0
            self.get_logger().info("Initialized derivative calculation variables from first odometry data")
        
        self.odom_data = {
            'x': x,
            'y': y,
            'yaw': yaw,
            'velocity': linear_vel,
            'yaw_rate': angular_vel
        }
        
        self.check_state_ready()
    
    def check_state_ready(self):
        """Check if we have received all required state information"""
        if self.lidar_data is not None and self.odom_data is not None:
            self.state_ready = True
    
    def prepare_observation(self):
        """Prepare observation in the format expected by the RL model"""
        if not self.state_ready:
            return None
        
        # Get Cartesian state from odometry
        x = self.odom_data['x']
        y = self.odom_data['y']
        yaw = self.odom_data['yaw']
        velocity = self.odom_data['velocity']
        
        # Convert to Frenet frame
        try:
            s, ey, ephi = self.track.cartesian_to_frenet(x, y, yaw, s_guess=self.s_guess)
            self.s_guess = s # Update s_guess for next iteration
        except Exception as e:
            self.get_logger().error(f"Frenet conversion failed: {e}. Using previous guess or zeros.")
            # Handle potential errors, e.g., if track not loaded or conversion fails
            s = self.s_guess
            ey = 0.0 # Defaulting ey might be risky, but needed for shape

        # Format observation: [s, ey, vel, yaw_angle, lidar_scan]
        state = [
            s,
            ey,
            velocity,
            yaw, # Keep global yaw for consistency with training env observation
        ]

        # Combine with lidar scans if enabled
        observation_components = state.copy()
        
        if self.lidar_scan_in_obs_mode != "NONE":
            lidar_scan_processed = self._process_lidar_scan(self.lidar_data)
            observation_components.extend(lidar_scan_processed)

        observation = np.array(observation_components, dtype=np.float32)

        # Normalize observation if VecNormalize is available
        original_obs = observation.copy()  # Keep a copy for debugging
        
        try:
            if self.vec_normalize is not None:
                # VecNormalize expects batch dimension
                observation = observation.reshape(1, -1)
                observation = self.vec_normalize.normalize_obs(observation)
                observation = observation.reshape(-1)
                # Log normalization impact occasionally (every 20 steps)
                if self.current_step % 20 == 0:
                    self.get_logger().debug(f"Observation normalized: original s={original_obs[0]:.4f}, normalized s={observation[0]:.4f}")
        except Exception as e:
            self.get_logger().error(f"Error during observation normalization: {e}")
            observation = original_obs  # Fallback to unnormalized observation
            
        # Final check for shape
        if self.lidar_scan_in_obs_mode == "NONE":
            expected_lidar_dim = 0
        elif self.lidar_scan_in_obs_mode == "FULL":
            expected_lidar_dim = 1080
        elif self.lidar_scan_in_obs_mode == "DOWNSAMPLED":
            expected_lidar_dim = 108
        else:
            expected_lidar_dim = 0
            
        expected_shape = 4 + expected_lidar_dim
        if observation.shape != (expected_shape,):
            self.get_logger().error(f"Observation shape mismatch: expected ({expected_shape},), got {observation.shape}")
            return None # Don't return incorrect shape

        return observation
    
    def _publish_drive_command(self, steering: float, speed: float, log_message: str = None) -> None:
        """
        Create and publish an AckermannDriveStamped message with all required fields.
        
        Args:
            steering: Steering angle in radians
            speed: Speed in m/s
            log_message: Optional additional message to log
        """
        # Get current timestamp for derivative calculations
        current_timestamp = time.time()
        current_velocity = self.odom_data['velocity'] if self.odom_data else 0.0
        
        # Apply acceleration and steering velocity limiting
        limited_steering, limited_speed = self._calculate_and_limit_commanded_values(
            steering, speed, current_timestamp
        )
        
        # Calculate enhanced drive message fields using the limited values
        acceleration, steering_angle_velocity, jerk = self._calculate_drive_derivatives(
            limited_steering, current_velocity, current_timestamp
        )
        
        # Create and populate drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        
        # Populate all AckermannDrive fields with limited values
        drive_msg.drive.steering_angle = limited_steering
        drive_msg.drive.steering_angle_velocity = steering_angle_velocity
        drive_msg.drive.speed = limited_speed
        drive_msg.drive.acceleration = acceleration
        drive_msg.drive.jerk = jerk
        
        # Publish the drive command
        self.drive_pub.publish(drive_msg)
        
        # Update QoS statistics for drive commands
        self.qos_stats['drive_commands_sent'] += 1
        
        # Log message if provided, showing both original and limited values
        if log_message:
            extra_info = ""
            if abs(limited_speed - speed) > 0.01 or abs(limited_steering - steering) > 0.001:
                extra_info = f" [LIMITED: orig_speed={speed:.2f}→{limited_speed:.2f}, orig_steer={steering:.3f}→{limited_steering:.3f}]"
            
            self.get_logger().debug(log_message + 
                f", Action: steering={limited_steering:.2f}, speed={limited_speed:.2f}"
                f", Derivatives: accel={acceleration:.2f}, steer_vel={steering_angle_velocity:.2f}, jerk={jerk:.2f}"
                f"{extra_info}")
        else:
            # Log acceleration limiting when it occurs
            if abs(limited_speed - speed) > 0.01:
                self.get_logger().debug(f"Speed limited: {speed:.2f} → {limited_speed:.2f} m/s")

    def control_loop(self):
        """Main control loop that gets predictions from the model and sends commands"""
        if not self.state_ready:
            self.get_logger().info("Waiting for sensor data...")
            return
            
        # Increment step counter
        self.current_step += 1
        
        # Update position and velocity history
        if self.odom_data:
            self.position_history.append((self.odom_data['x'], self.odom_data['y']))
            self.velocity_history.append(self.odom_data['velocity'])
            
            # Keep history size limited
            if len(self.position_history) > self.position_history_size:
                self.position_history.pop(0)
            if len(self.velocity_history) > self.position_history_size:
                self.velocity_history.pop(0)
        
        # Check for collision (only if not already in reset)
        if not self.reset_in_progress and self.check_collision():
            self.collision_detected = True
        
        # Handle reset sequence if collision was detected
        if self.collision_detected or self.reset_in_progress:
            reset_action = self.execute_reset()
            if reset_action is not None:
                # Still in reset sequence, send reset commands
                self.reset_timer += 1
                steering = float(reset_action[0])
                speed = float(reset_action[1])
                
                # Send reset command
                self._publish_drive_command(steering, speed, f"Reset phase: {self.reset_phase} ({self.reset_timer}/{self.reset_duration})")
                
                return
        
        # Normal operation - prepare observation
        obs = self.prepare_observation()
        
        if obs is None:
            self.get_logger().warn("Failed to prepare observation, skipping control loop iteration.")
            return
        
        # Get action from model
        start_time = time.time()
        
        # Handle differently based on whether the model is recurrent
        if self.is_recurrent:
            # For recurrent policies, we need to pass lstm_states and episode_starts
            action, self.lstm_states = self.model.predict(
                obs, 
                state=self.lstm_states, 
                episode_start=self.episode_starts, 
                deterministic=True
            )
            # Update episode_starts for next step
            self.episode_starts = np.zeros((1,), dtype=bool)  # After first step, no longer episode start
        else:
            # Standard prediction for non-recurrent models
            action, _ = self.model.predict(obs, deterministic=True)
        
        inference_time = time.time() - start_time
        
        # Convert action to drive command
        steering = float(action[0])
        speed = float(action[1])
        
        # Store command in history
        self.command_history.append((steering, speed))
        if len(self.command_history) > self.position_history_size:
            self.command_history.pop(0)
        
        # Update last steering angle
        self.last_steering_angle = steering
        
        # Send drive command
        self._publish_drive_command(steering, speed)
        
        # Plot lidar scan only if enabled
        if self.enable_lidar_plot:
            self.plot_lidar_scan()
        
        # Log info
        self.get_logger().info(
            f"State: s={obs[0]:.4f}, ey={obs[1]:.4f}, vel={obs[2]:.4f}, yaw={obs[3]:.4f}, " +
            f"Action: steering={steering:.4f}, speed={speed:.4f}, " +
            f"Inference time: {inference_time*1000:.2f}ms"
        )
    
    def plot_lidar_scan(self):
        """Plot lidar scan in polar coordinates and as range values"""
        if not self.show_lidar_plot or self.lidar_data is None:
            return
        
        # Use processed lidar data (with interpolation) instead of raw data
        lidar_array = self._preprocess_lidar_data(self.lidar_data)
        
        # Create angle array based on lidar data length
        scan_length = len(lidar_array)
        angles = np.linspace(0, 2*np.pi, scan_length)
        indices = np.arange(scan_length)
        
        # Clear previous plots
        self.ax_polar.clear()
        self.ax_range.clear()
        
        # Set polar plot limits
        self.ax_polar.set_rlim(0, 30)
        
        # Plot the lidar scan in polar coordinates with different colors for interpolated points
        if self.interpolation_mask is not None and len(self.interpolation_mask) == len(lidar_array):
            # Plot original points in blue
            original_mask = ~self.interpolation_mask
            if np.any(original_mask):
                self.ax_polar.scatter(angles[original_mask], lidar_array[original_mask], 
                                     s=2, c='blue', label='Original')
            
            # Plot interpolated points in red
            if np.any(self.interpolation_mask):
                self.ax_polar.scatter(angles[self.interpolation_mask], lidar_array[self.interpolation_mask], 
                                     s=4, c='red', marker='x', label='Interpolated')
                
            # Add legend if there are interpolated points
            if np.any(self.interpolation_mask):
                self.ax_polar.legend(loc='upper right', fontsize=8)
        else:
            # Fallback to single color if interpolation mask is not available
            self.ax_polar.scatter(angles, lidar_array, s=2, c='blue')
        
        # Set title and show ego vehicle position in polar plot
        title = 'LiDAR Scan - Polar View'
        if self.collision_detected or self.reset_in_progress:
            title += ' [COLLISION DETECTED - RESETTING]'
            self.ax_polar.set_facecolor('mistyrose')  # Light red background
        else:
            self.ax_polar.set_facecolor('white')
        self.ax_polar.set_title(title)
        self.ax_polar.plot(0, 0, 'ro')  # Red dot at origin representing the vehicle
        
        # Add car shape representation (simple triangle)
        car_angles = np.array([0, 2.5, -2.5]) * np.pi / 180  # Front and sides of car
        car_radius = np.array([0.3, 0.2, 0.2])  # Front longer than sides
        self.ax_polar.scatter(car_angles, car_radius, c='red', s=50)
        
        # Plot the lidar scan as range values with different colors for interpolated points
        if self.interpolation_mask is not None and len(self.interpolation_mask) == len(lidar_array):
            # Plot original points in blue
            original_mask = ~self.interpolation_mask
            if np.any(original_mask):
                self.ax_range.scatter(indices[original_mask], lidar_array[original_mask], 
                                     s=2, c='blue', label='Original')
            
            # Plot interpolated points in red
            if np.any(self.interpolation_mask):
                self.ax_range.scatter(indices[self.interpolation_mask], lidar_array[self.interpolation_mask], 
                                     s=4, c='red', marker='x', label='Interpolated')
                
            # Add legend if there are interpolated points
            if np.any(self.interpolation_mask):
                self.ax_range.legend(loc='upper right', fontsize=8)
        else:
            # Fallback to single color if interpolation mask is not available
            self.ax_range.scatter(indices, lidar_array, s=2, c='blue')
            
        self.ax_range.set_xlim(0, len(indices))
        self.ax_range.set_ylim(0, 30)
        self.ax_range.set_xlabel('Scan Index (0-1080)')
        self.ax_range.set_ylabel('Range (m)')
        self.ax_range.set_title('LiDAR Scan - Range View')
        self.ax_range.grid(True, linestyle='--', alpha=0.7)
        
        # Add collision statistics text and interpolation info
        collision_text = f"Collisions: {self.collision_count}"
        if self.collision_detected or self.reset_in_progress:
            collision_text += f" | Reset Phase: {self.reset_phase} ({self.reset_timer}/{self.reset_duration})"
            if self.reset_phase == 'align' and self.reset_target_waypoint and 'yaw' in self.reset_target_waypoint:
                collision_text += f"\nTarget Orientation: {self.reset_target_waypoint['yaw']:.2f} rad"
        
        # Add interpolation statistics
        if self.interpolation_mask is not None:
            num_interpolated = np.sum(self.interpolation_mask)
            total_points = len(self.interpolation_mask)
            interpolation_percentage = (num_interpolated / total_points) * 100 if total_points > 0 else 0
            collision_text += f"\nInterpolated: {num_interpolated}/{total_points} ({interpolation_percentage:.1f}%)"
        
        self.ax_range.text(0.02, 0.98, collision_text, 
                          transform=self.ax_range.transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat' if self.collision_detected else 'lightgreen', alpha=0.5))
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Draw and update plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        # Pause briefly to allow plot to update
        plt.pause(0.001)
    
    def detect_stuck(self):
        """Detect if the car is stuck by checking if it hasn't moved much recently"""
        if len(self.position_history) < self.position_history_size:
            return False
        
        # Calculate total distance moved in recent history
        total_distance = 0.0
        for i in range(1, len(self.position_history)):
            dx = self.position_history[i][0] - self.position_history[i-1][0]
            dy = self.position_history[i][1] - self.position_history[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Check if car is stuck
        avg_distance_per_step = total_distance / (len(self.position_history) - 1)
        is_stuck = avg_distance_per_step < self.stuck_threshold / self.position_history_size
        
        if is_stuck:
            self.get_logger().debug(f"Stuck detected: avg movement {avg_distance_per_step:.4f}m per step")
        
        return is_stuck
    
    def detect_velocity_command_mismatch(self):
        """Detect collision by checking if actual velocity matches commanded velocity"""
        if len(self.velocity_history) < 5 or len(self.command_history) < 5:
            return False
        
        # Get recent average commanded speed (ignore steering for now)
        avg_commanded_speed = np.mean([cmd[1] for cmd in self.command_history[-5:]])
        avg_actual_speed = np.mean(self.velocity_history[-5:])
        
        # If we're commanding significant speed but not moving
        if avg_commanded_speed > 1.0 and avg_actual_speed < 0.3:
            self.get_logger().debug(f"Velocity mismatch: commanded {avg_commanded_speed:.2f}, actual {avg_actual_speed:.2f}")
            return True
        
        return False
    
    def detect_lidar_collision(self):
        """Detect collision using lidar data"""
        if self.lidar_data is None:
            return False
        
        # Check front lidar beams (roughly -30 to +30 degrees)
        # Assuming 1080 beams covering 270 degrees, front is roughly indices 405 to 675
        front_start = 405
        front_end = 675
        
        # Convert to numpy array for vectorized operations
        lidar_array = np.array(self.lidar_data)
        front_lidar = lidar_array[front_start:front_end]
        
        # Filter out invalid readings
        valid_readings = front_lidar[
            (front_lidar > 0.05) & 
            (front_lidar < 30.0) & 
            (~np.isnan(front_lidar)) & 
            (~np.isinf(front_lidar))
        ]
        
        if len(valid_readings) > 0:
            min_distance = np.min(valid_readings)
            if min_distance < self.min_lidar_distance:
                self.get_logger().debug(f"Lidar collision detected: min distance {min_distance:.3f}m")
                return True
        
        return False
    
    def check_collision(self):
        """Combined collision detection"""
        # Check all collision conditions
        is_stuck = self.detect_stuck()
        velocity_mismatch = self.detect_velocity_command_mismatch()
        lidar_collision = self.detect_lidar_collision()
        
        # Collision is detected if any condition is met
        collision_detected = is_stuck or velocity_mismatch or lidar_collision
        
        if collision_detected:
            self.consecutive_collision_count += 1
            if self.consecutive_collision_count >= self.consecutive_collision_threshold:
                self.collision_count += 1
                self.last_collision_time = time.time()
                self.get_logger().warning(
                    f"Collision #{self.collision_count} detected! "
                    f"Stuck: {is_stuck}, Velocity mismatch: {velocity_mismatch}, "
                    f"Lidar collision: {lidar_collision}"
                )
                return True
        else:
            self.consecutive_collision_count = 0
        
        return False
    
    def find_reset_position(self):
        """Find a suitable waypoint for orientation reference during reset"""
        if self.odom_data is None or self.waypoints is None:
            return None
        
        # Get current position
        x = self.odom_data['x']
        y = self.odom_data['y']
        
        try:
            # Find current position on track
            s, ey, ephi = self.track.cartesian_to_frenet(x, y, self.odom_data['yaw'], 
                                                          s_guess=self.s_guess)
            
            # Find waypoint that's behind us by some distance
            reset_distance = 3.0  # meters back along the track
            target_s = s - reset_distance
            
            # Handle wrap-around
            if target_s < 0:
                target_s += self.track.s_frame_max
            
            # Find the closest waypoint to target_s
            waypoint_s_values = self.waypoints[:, 0]  # s values are in column 0
            closest_idx = np.argmin(np.abs(waypoint_s_values - target_s))
            
            # Get waypoint data: [s, x, y, psi, k, vx, ax]
            waypoint = self.waypoints[closest_idx]
            
            reset_position = {
                'waypoint_idx': closest_idx,
                's': waypoint[0],
                'yaw': waypoint[3]  # psi (heading angle)
            }
            
            self.get_logger().debug(
                f"Found reset waypoint {closest_idx}: "
                f"s={reset_position['s']:.2f}, "
                f"yaw={reset_position['yaw']:.2f} rad"
            )
            
            return reset_position
            
        except Exception as e:
            self.get_logger().error(f"Failed to find reset waypoint: {e}")
            return None
    
    def compute_reset_control(self):
        """Compute control commands for alignment during reset"""
        if self.reset_target_waypoint is None or self.odom_data is None:
            return 0.0, 0.0
        
        # Current orientation
        curr_yaw = self.odom_data['yaw']
        
        # Target orientation from waypoint
        target_yaw = self.reset_target_waypoint['yaw']
        
        # Compute orientation error (normalize to [-pi, pi])
        orientation_error = target_yaw - curr_yaw
        orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))
        
        # Control logic for alignment phase
        if self.reset_phase == 'align':
            # Align with track orientation
            if abs(orientation_error) > self.reset_angle_tolerance:
                # Turn in place to align
                steering = np.clip(self.reset_kp_steering * orientation_error, -0.4189, 0.4189)
                speed = 0.3  # Very slow forward motion while aligning
                return steering, speed
            else:
                # Alignment complete
                self.reset_phase = 'complete'
                self.get_logger().debug(f"Reset alignment complete, orientation error: {orientation_error:.3f} rad")
                return 0.0, 0.0
        
        return 0.0, 0.0
    
    def execute_reset(self):
        """Execute the reset sequence: backup, align, complete"""
        if not self.reset_in_progress:
            # Initialize reset
            self.reset_in_progress = True
            self.reset_timer = 0
            self.collision_detected = True
            self.reset_phase = 'backup'
            
            # Clear histories
            self.position_history.clear()
            self.velocity_history.clear()
            self.command_history.clear()
            
            # Reset LSTM states for recurrent policies
            if self.is_recurrent:
                self.reset_lstm_state()
            
            # Find reset waypoint for alignment reference
            self.reset_target_waypoint = self.find_reset_position()
            if self.reset_target_waypoint:
                # Update s_guess for next iterations
                self.s_guess = self.reset_target_waypoint['s']
                self.get_logger().debug(
                    f"Reset target orientation: yaw={self.reset_target_waypoint['yaw']:.2f} rad"
                )
            else:
                self.get_logger().error("Failed to find reset waypoint, using current orientation")
                # Fall back to current orientation
                self.reset_target_waypoint = {'yaw': self.odom_data['yaw']}
        
        # Execute reset phases
        if self.reset_phase == 'backup':
            # Phase 1: Back up to clear immediate obstacle
            if self.reset_timer < self.reset_backup_duration:
                return np.array([0.0, -1.0])  # straight, slow reverse
            else:
                self.reset_phase = 'align'
                self.get_logger().debug("Backup complete, starting alignment")
                return np.array([0.0, 0.0])  # stop briefly
                
        elif self.reset_phase == 'align':
            # Phase 2: Align with track orientation
            steering, speed = self.compute_reset_control()
            
            if self.reset_timer < self.reset_duration:
                return np.array([steering, speed])
            else:
                self.reset_phase = 'complete'
                self.get_logger().debug("Alignment complete, starting complete")
                return np.array([0.0, 0.0])
            
        elif self.reset_phase == 'complete':
            # Reset complete
            self.reset_in_progress = False
            self.collision_detected = False
            self.reset_timer = 0
            self.consecutive_collision_count = 0
            self.reset_target_waypoint = None
            self.get_logger().debug("Reset sequence completed successfully")
            return None
        
        # Default: stop
        return np.array([0.0, 0.0])
    
    def publish_reset_pose(self, reset_pose):
        """Publish reset pose for simulator (if supported)"""
        try:
            reset_msg = Odometry()
            reset_msg.header.stamp = self.get_clock().now().to_msg()
            reset_msg.header.frame_id = 'map'
            reset_msg.pose.pose.position.x = reset_pose['x']
            reset_msg.pose.pose.position.y = reset_pose['y']
            reset_msg.pose.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            reset_msg.pose.pose.orientation.w = np.cos(reset_pose['yaw'] / 2)
            reset_msg.pose.pose.orientation.x = 0.0
            reset_msg.pose.pose.orientation.y = 0.0
            reset_msg.pose.pose.orientation.z = np.sin(reset_pose['yaw'] / 2)
            
            self.reset_pose_pub.publish(reset_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish reset pose: {e}")
    
    def reset_lstm_state(self):
        """Reset LSTM states when needed (e.g., at the beginning of a new episode)"""
        if self.is_recurrent:
            self.lstm_states = None
            self.episode_starts = np.ones((1,), dtype=bool)
            self.get_logger().info("Reset LSTM states")

    def _calculate_and_limit_commanded_values(self, commanded_steering: float, commanded_speed: float, current_timestamp: float) -> Tuple[float, float]:
        """
        Calculate commanded acceleration and steering velocity, then apply acceleration limits.
        
        Args:
            commanded_steering: Raw commanded steering angle (rad)
            commanded_speed: Raw commanded speed (m/s)
            current_timestamp: Current timestamp (seconds)
            
        Returns:
            Tuple of (limited_steering, limited_speed) with acceleration constraints applied
        """
        # Calculate time delta
        if self.previous_timestamp is not None:
            dt = current_timestamp - self.previous_timestamp
            # # Ensure reasonable dt (fallback to nominal if too small/large)
            # if dt <= 0 or dt > 0.1:  # Sanity check: dt should be ~0.02s
            #     dt = self.control_loop_dt
        else:
            dt = self.control_loop_dt
        
        # Get current observed velocity from odometry
        current_velocity = self.odom_data['velocity'] if self.odom_data else 0.0
        
        # Calculate commanded acceleration: (commanded_speed - previous_observed_velocity) / dt
        commanded_acceleration = (commanded_speed - current_velocity) / dt
        
        # Calculate commanded steering velocity: (commanded_steering - previous_commanded_steering) / dt
        commanded_steering_velocity = (commanded_steering - self.previous_commanded_steering) / dt
        
        # Apply acceleration limiting by adjusting commanded speed
        limited_commanded_speed = commanded_speed
        if abs(commanded_acceleration) > self.max_acceleration:
            # Limit the commanded speed to achieve max acceleration
            if commanded_acceleration > 0:
                limited_commanded_speed = current_velocity + self.max_acceleration * dt
                self.get_logger().debug(f"Limiting acceleration: requested={commanded_acceleration:.2f}, limited to {self.max_acceleration:.2f} m/s²")
            else:
                limited_commanded_speed = current_velocity - self.max_acceleration * dt
                self.get_logger().debug(f"Limiting deceleration: requested={commanded_acceleration:.2f}, limited to {-self.max_acceleration:.2f} m/s²")
            
            # Recalculate actual commanded acceleration after limiting
            actual_commanded_acceleration = (limited_commanded_speed - current_velocity) / dt
        else:
            actual_commanded_acceleration = commanded_acceleration
        
        # Apply steering velocity limiting by adjusting commanded steering
        limited_commanded_steering = commanded_steering
        if abs(commanded_steering_velocity) > self.max_steering_velocity:
            # Limit the commanded steering to achieve max steering velocity
            if commanded_steering_velocity > 0:
                limited_commanded_steering = self.previous_commanded_steering + self.max_steering_velocity * dt
                self.get_logger().debug(f"Limiting steering velocity: requested={commanded_steering_velocity:.2f}, limited to {self.max_steering_velocity:.2f} rad/s")
            else:
                limited_commanded_steering = self.previous_commanded_steering - self.max_steering_velocity * dt
                self.get_logger().debug(f"Limiting steering velocity: requested={commanded_steering_velocity:.2f}, limited to {-self.max_steering_velocity:.2f} rad/s")
            
            # Recalculate actual commanded steering velocity after limiting
            actual_commanded_steering_velocity = (limited_commanded_steering - self.previous_commanded_steering) / dt
        else:
            actual_commanded_steering_velocity = commanded_steering_velocity
        
        # Log commanded derivatives occasionally for monitoring
        if self.current_step % 50 == 0:  # Log every 50 steps
            self.get_logger().debug(
                f"Commanded derivatives - Acceleration: {actual_commanded_acceleration:.2f} m/s², "
                f"Steering velocity: {actual_commanded_steering_velocity:.2f} rad/s"
            )
        
        # Update previous commanded values for next iteration
        self.previous_commanded_steering = limited_commanded_steering
        self.previous_commanded_speed = limited_commanded_speed
        
        return limited_commanded_steering, limited_commanded_speed

    def _calculate_drive_derivatives(self, current_steering: float,
                                   current_velocity: float, current_timestamp: float) -> Tuple[float, float, float]:
        """
        Calculate acceleration, steering angle velocity, and jerk for AckermannDrive message.
        
        Args:
            current_steering: Current commanded steering angle (rad)
            current_speed: Current commanded speed (m/s)
            current_velocity: Current actual velocity from odometry (m/s)
            current_timestamp: Current timestamp (seconds)
            
        Returns:
            Tuple of (acceleration, steering_angle_velocity, jerk) in SI units
        """
        # Calculate actual time delta if we have previous timestamp
        if self.previous_timestamp is not None:
            dt = current_timestamp - self.previous_timestamp
            # # Ensure reasonable dt (fallback to nominal if too small/large)
            # if dt <= 0 or dt > 0.1:  # Sanity check: dt should be ~0.02s
            #     dt = self.control_loop_dt
        else:
            dt = self.control_loop_dt
        
        # Calculate acceleration (m/s²) from velocity change
        acceleration = (current_velocity - self.previous_velocity) / dt
        
        # Calculate steering angle velocity (rad/s) from steering angle change
        steering_angle_velocity = (current_steering - self.previous_steering_angle) / dt
        
        # Calculate jerk (m/s³) from acceleration change
        jerk = (acceleration - self.previous_acceleration) / dt
        
        # # Apply reasonable bounds to prevent extreme values due to noise
        # acceleration = np.clip(acceleration, -20.0, 20.0)  # Reasonable acceleration limits
        # steering_angle_velocity = np.clip(steering_angle_velocity, -10.0, 10.0)  # Reasonable steering rate limits
        # jerk = np.clip(jerk, -50.0, 50.0)  # Reasonable jerk limits
        
        # Update previous values for next iteration
        self.previous_velocity = current_velocity
        self.previous_acceleration = acceleration
        self.previous_steering_angle = current_steering
        self.previous_timestamp = current_timestamp
        
        return acceleration, steering_angle_velocity, jerk

def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='F1TENTH RL Controller')
    parser.add_argument('--algorithm', type=str, default='RECURRENT_PPO', choices=['SAC', 'RECURRENT_PPO'],
                        help='RL algorithm to use (SAC or RECURRENT_PPO)')
    parser.add_argument('--model_path', type=str, default='./logs/best_model/best_model.zip',
                        help='Path to the trained model')
    parser.add_argument('--vecnorm_path', type=str, default=None,
                        help='Path to VecNormalize statistics')
    parser.add_argument('--map_index', type=int, default=63,
                        help='Index of the map to use')
    parser.add_argument('--map_dir', type=str, default='./f1tenth_racetracks/',
                        help='Directory containing the track maps')
    parser.add_argument('--lidar_scan_in_obs_mode', type=str, default='FULL', choices=['FULL', 'NONE', 'DOWNSAMPLED'],
                        help='Mode for including lidar scans in observations')
    parser.add_argument('--enable_lidar_plot', action='store_true', default=False,
                        help='Enable lidar visualization plots')
    
    # Collision detection parameters
    parser.add_argument('--enable_collision_reset', action='store_true', default=False,
                        help='Enable automatic collision detection and reset')
    parser.add_argument('--stuck_threshold', type=float, default=0.1,
                        help='Distance threshold (m) for stuck detection')
    parser.add_argument('--min_lidar_distance', type=float, default=0.2,
                        help='Minimum safe distance (m) from obstacles')
    parser.add_argument('--reset_duration', type=int, default=150,
                        help='Number of control loops for reset sequence')
    parser.add_argument('--reset_backup_duration', type=int, default=50,
                        help='Number of control loops for backing up')
    parser.add_argument('--reset_angle_tolerance', type=float, default=0.2,
                        help='Angle tolerance (rad) for aligning with track')
    parser.add_argument('--reset_kp_steering', type=float, default=2.0,
                        help='Proportional gain for reset steering control')
    
    # Acceleration limiting parameters
    parser.add_argument('--max_acceleration', type=float, default=8.0,
                        help='Maximum allowed acceleration (m/s²)')
    parser.add_argument('--max_steering_velocity', type=float, default=3.2,
                        help='Maximum allowed steering velocity (rad/s)')
    
    # Parse known args to avoid conflicts with ROS args
    parsed_args, unknown = parser.parse_known_args(args=args)
    
    rclpy.init(args=args)
    print("RL F1Tenth Controller Initialized")
    
    # Create the controller with parsed arguments
    controller = RLF1TenthController(
        algorithm=parsed_args.algorithm,
        model_path=parsed_args.model_path,
        vecnorm_path=parsed_args.vecnorm_path,
        map_index=parsed_args.map_index,
        map_dir_=parsed_args.map_dir,
        lidar_scan_in_obs_mode=parsed_args.lidar_scan_in_obs_mode,
        enable_lidar_plot=parsed_args.enable_lidar_plot
    )
    
    # Update collision detection parameters if provided
    if hasattr(parsed_args, 'enable_collision_reset') and not parsed_args.enable_collision_reset:
        controller.get_logger().info("Collision detection and auto-reset disabled")
        controller.check_collision = lambda: False  # Disable collision detection
    
    if hasattr(parsed_args, 'stuck_threshold'):
        controller.stuck_threshold = parsed_args.stuck_threshold
    if hasattr(parsed_args, 'min_lidar_distance'):
        controller.min_lidar_distance = parsed_args.min_lidar_distance
    if hasattr(parsed_args, 'reset_duration'):
        controller.reset_duration = parsed_args.reset_duration
    if hasattr(parsed_args, 'reset_backup_duration'):
        controller.reset_backup_duration = parsed_args.reset_backup_duration
    if hasattr(parsed_args, 'reset_angle_tolerance'):
        controller.reset_angle_tolerance = parsed_args.reset_angle_tolerance
    if hasattr(parsed_args, 'reset_kp_steering'):
        controller.reset_kp_steering = parsed_args.reset_kp_steering
    
    if hasattr(parsed_args, 'max_acceleration'):
        controller.max_acceleration = parsed_args.max_acceleration
    
    if hasattr(parsed_args, 'max_steering_velocity'):
        controller.max_steering_velocity = parsed_args.max_steering_velocity
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()