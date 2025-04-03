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
            observation_space = spaces.Box(low=0, high=30, shape=(1082,), dtype=np.float32)
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
        
        # Format now matches the reduced observation space (velocity, yaw angle, lidar)
        state = [
            self.odom_data['velocity'],  # velocity
            self.odom_data['yaw']        # yaw angle
        ]
        
        # Combine with lidar scans
        lidar_data = self.lidar_data[:1080]
        # adapt to a special lidar
        for i in range(1080):
            if lidar_data[i] < 0.05:
                lidar_data[i] = 30
        observation = np.concatenate((state, lidar_data))
        
        return observation
    
    def control_loop(self):
        """Main control loop that gets predictions from the model and sends commands"""
        if not self.state_ready:
            self.get_logger().info("Waiting for sensor data...")
            return
        
        # Prepare observation
        obs = self.prepare_observation()
        
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