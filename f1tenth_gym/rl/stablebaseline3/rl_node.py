#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from stable_baselines3 import SAC
import os
import time
from gymnasium import spaces
import sys
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.Track import Track
from utils import utils # Added import

class RLF1TenthController(Node):
    """
    ROS Node that uses a trained RL model to control an F1Tenth car
    """
    def __init__(self):
        super().__init__('rl_f1tenth_controller')
        
        # Topics
        lidar_topic = '/scan'
        odom_topic = '/ego_racecar/odom'
        # odom_topic = '/pf/pose/odom'
        drive_topic = '/drive'
        
        # Add last steering angle tracking
        self.last_steering_angle = 0.0
        
        # Load the trained model
        model_path = os.path.expanduser("./logs/best_model/best_model.zip")
        self.get_logger().info(f"Loading model from {model_path}")
        try:
            # Create observation and action spaces matching training environment
            # Updated shape to 1085: [s, ey, vel, yaw_angle, yaw_rate] + 1080 lidar
            observation_space = spaces.Box(
                low=np.concatenate(([-1000.0, -5.0, -5.0, -np.pi, -10.0], np.zeros(1080))),
                high=np.concatenate(([1000.0, 5.0, 20.0, np.pi, 10.0], np.full(1080, 30.0))),
                shape=(1085,), dtype=np.float32
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
            
            # Load model with spaces and custom objects
            self.model = SAC.load(model_path, custom_objects=custom_objects)
            self.get_logger().info("Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
        
        # State variables
        self.lidar_data = None
        self.odom_data = None
        self.state_ready = False
        self.s_guess = 0.0 # Initialize s_guess for Frenet conversion

        # Load Track for Frenet Conversion
        # TODO: Make map index and map directory configurable via ROS params or args
        map_index = 63 # Using default map index from main.py
        map_dir_ = './f1tenth_racetracks/' # Relative to workspace root where node is likely run from

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
        
        self.get_logger().info("RL F1Tenth Controller initialized")

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
        yaw_rate = self.odom_data['yaw_rate']

        # Convert to Frenet frame
        try:
            s, ey, ephi = self.track.cartesian_to_frenet(x, y, yaw, s_guess=self.s_guess)
            self.s_guess = s # Update s_guess for next iteration
        except Exception as e:
            self.get_logger().error(f"Frenet conversion failed: {e}. Using previous guess or zeros.")
            # Handle potential errors, e.g., if track not loaded or conversion fails
            s = self.s_guess
            ey = 0.0 # Defaulting ey might be risky, but needed for shape

        # Format observation: [s, ey, vel, yaw_angle, yaw_rate, lidar_scan]
        state = [
            s,
            ey,
            velocity,
            yaw, # Keep global yaw for consistency with training env observation
            yaw_rate
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

        # Final check for shape
        if observation.shape != (1085,):
            self.get_logger().error(f"Observation shape mismatch: expected (1085,), got {observation.shape}")
            return None # Don't return incorrect shape

        return observation
    
    def control_loop(self):
        """Main control loop that gets predictions from the model and sends commands"""
        if not self.state_ready:
            self.get_logger().info("Waiting for sensor data...")
            return
        
        # Prepare observation
        obs = self.prepare_observation()
        
        if obs is None:
            self.get_logger().warn("Failed to prepare observation, skipping control loop iteration.")
            return
        
        # Get action from model
        start_time = time.time()
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
            f"State: s={obs[0]:.4f}, ey={obs[1]:.4f}, vel={obs[2]:.4f}, yaw={obs[3]:.4f}, yaw_rate={obs[4]:.4f}" +
            f"Action: steering={steering:.4f}, speed={speed:.4f}, " +
            f"Inference time: {inference_time*1000:.2f}ms"
        )

def main(args=None):
    rclpy.init(args=args)
    print("RL F1Tenth Controller Initialized")
    controller = RLF1TenthController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()