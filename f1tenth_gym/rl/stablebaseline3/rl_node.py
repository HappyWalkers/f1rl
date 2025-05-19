#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
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
                 vecnorm_path=None, map_index=63, map_dir_="./f1tenth_racetracks/"):
        super().__init__('rl_f1tenth_controller')
        
        # Topics
        lidar_topic = '/scan'
        odom_topic = '/ego_racecar/odom'
        # odom_topic = '/pf/pose/odom'
        drive_topic = '/drive'
        
        # Store configuration as attributes
        self.algorithm = algorithm
        model_path = os.path.expanduser(model_path)
        
        # Add last steering angle tracking
        self.last_steering_angle = 0.0
        
        # For recurrent policies (LSTM state tracking)
        self.is_recurrent = self.algorithm == "RECURRENT_PPO"
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)  # Mark the start of the episode
        
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
            # Updated shape to 1084: [s, ey, vel, yaw_angle] + 1080 lidar
            observation_space = spaces.Box(
                low=np.concatenate(([-1000.0, -5.0, -5.0, -np.pi], np.zeros(1080))),
                high=np.concatenate(([1000.0, 5.0, 12.0, np.pi], np.full(1080, 30.0))),
                shape=(1084,), dtype=np.float32
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
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        
        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )
        
        # Create a timer for control loop
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info(f"RL F1Tenth Controller initialized with {self.algorithm} algorithm")

    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        self.lidar_data = msg.ranges
        self.check_state_ready()
    
    def odom_callback(self, msg):
        """Process odometry data"""
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

        # Combine with lidar scans
        # Ensure lidar data has 1080 points, pad if necessary (shouldn't be needed with real lidar)
        lidar_scan_processed = np.array(self.lidar_data[:1080], dtype=np.float32)
        if len(lidar_scan_processed) < 1080:
             lidar_scan_processed = np.pad(lidar_scan_processed, (0, 1080 - len(lidar_scan_processed)), 'constant', constant_values=30.0)

        # Handle NaNs or Infs in lidar data (replace with max range)
        lidar_scan_processed[np.isnan(lidar_scan_processed)] = 30.0
        lidar_scan_processed[np.isinf(lidar_scan_processed)] = 30.0
        lidar_scan_processed[lidar_scan_processed < 0.05] = 30.0 # Treat very close readings as max range

        observation = np.concatenate((state, lidar_scan_processed))

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
        if observation.shape != (1084,):
            self.get_logger().error(f"Observation shape mismatch: expected (1084,), got {observation.shape}")
            return None # Don't return incorrect shape

        return observation
    
    def control_loop(self):
        """Main control loop that gets predictions from the model and sends commands"""
        if not self.state_ready:
            self.get_logger().info("Waiting for sensor data...")
            return
            
        # Increment step counter
        self.current_step += 1
        
        # Prepare observation
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
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        
        # Action is [steering, speed]
        steering = float(action[0])
        speed = float(action[1])
        
        # Update last steering angle
        self.last_steering_angle = steering
        
        # Limit values if needed based on the constraints in the environment
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed
        
        # Publish the drive command
        self.drive_pub.publish(drive_msg)
        
        # Log info
        self.get_logger().info(
            f"State: s={obs[0]:.4f}, ey={obs[1]:.4f}, vel={obs[2]:.4f}, yaw={obs[3]:.4f}, " +
            f"Action: steering={steering:.4f}, speed={speed:.4f}, " +
            f"Inference time: {inference_time*1000:.2f}ms"
        )
    
    def reset_lstm_state(self):
        """Reset LSTM states when needed (e.g., at the beginning of a new episode)"""
        if self.is_recurrent:
            self.lstm_states = None
            self.episode_starts = np.ones((1,), dtype=bool)
            self.get_logger().info("Reset LSTM states")

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
        map_dir_=parsed_args.map_dir
    )
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()