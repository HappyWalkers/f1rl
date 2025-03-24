import numpy as np
import time
from absl import logging

class WallFollowPolicy:
    """
    Wall following policy for F1Tenth gym environment
    Similar approach to the ROS-based wall follower but adapted for the gym environment
    """
    def __init__(self, desired_distance=1.0):
        # PID controller parameters
        self.kp = 1.5  # Proportional gain
        self.ki = 0.0  # Integral gain (usually not needed for wall following)
        self.kd = 0.15  # Derivative gain
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0
        self.first_reading = True
        
        # Configuration
        self.desired_distance = desired_distance
        self.angle_min = -2.35  # From LiDAR configuration
        self.angle_increment = 0.00435  # From LiDAR configuration
        self.looking_ahead_distance = 1.0  # Look-ahead for smoother control
        
        logging.info("Wall following policy initialized")
        
    def predict(self, observation, deterministic=True):
        """
        Implements the predict interface expected by the evaluate function
        
        Args:
            observation: The environment observation (speed, yaw, lidar scan)
            deterministic: Whether to use deterministic actions (ignored in wall following)
            
        Returns:
            action: [steering, speed] action
            _: None (to match the RL policy interface)
        """
        # Extract lidar data (the last 1080 elements of observation)
        lidar_scan = observation[-1080:]
        
        # Get velocity from observation for adaptive speed control
        current_velocity = observation[0]
        
        # Calculate error to the wall
        error = self.get_error(lidar_scan)
        
        # Update PID controller
        self.integral += error
        if not self.first_reading:
            derivative = error - self.prev_error
            # Calculate steering command (negative because steering is inverted)
            steering = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
            
            # Clamp steering to valid range [-0.4189, 0.4189]
            steering = np.clip(steering, -0.4189, 0.4189)
        else:
            steering = 0.0
            self.first_reading = False
        
        # Store error for next iteration
        self.prev_error = error
        
        # Determine adaptive speed based on error and steering
        speed = self.adaptive_velocity(error, steering)
        
        return np.array([steering, speed]), None
    
    def get_range(self, ranges, angle):
        """
        Get the range measurement at the specified angle
        
        Args:
            ranges: Array of LiDAR distance measurements
            angle: Angle in radians to get measurement for
            
        Returns:
            range: Distance measurement at the specified angle
        """
        # Convert angle to index in the LiDAR array
        index = int((angle - self.angle_min) / self.angle_increment)
        
        # Ensure index is within bounds
        if index < 0 or index >= len(ranges):
            return float('inf')
        
        # Return measurement, handling NaN or inf values
        value = ranges[index]
        if np.isnan(value) or np.isinf(value):
            # Try adjacent values if this one is invalid
            for offset in [1, -1, 2, -2, 3, -3]:
                idx = index + offset
                if 0 <= idx < len(ranges) and not np.isnan(ranges[idx]) and not np.isinf(ranges[idx]):
                    return ranges[idx]
            return 10.0  # Default "far away" value if all nearby readings are invalid
        return value
    
    def angle_to_radian(self, angle):
        """Convert degrees to radians"""
        return angle * np.pi / 180
    
    def get_error(self, ranges):
        """
        Calculate the error to the wall using geometry
        
        Args:
            ranges: LiDAR scan data
            
        Returns:
            error: Error from desired distance to wall
        """
        # Get measurements at 90 degrees (directly to the left) and 45 degrees
        angle_90 = self.angle_to_radian(90)
        angle_45 = self.angle_to_radian(45)
        
        b = self.get_range(ranges, angle_90)
        a = self.get_range(ranges, angle_45)
        
        # Handle invalid measurements
        if b > 10.0 or a > 10.0:
            # If we're too far from a wall, default to small error
            return 0.1
        
        # Calculate angle to the wall using trigonometry
        theta = 45  # The difference between our two measurements
        theta_rad = self.angle_to_radian(theta)
        
        # Avoid division by zero
        denominator = a * np.sin(theta_rad)
        if abs(denominator) < 1e-6:
            return 0.0
            
        # Calculate angle to the wall
        alpha = np.arctan((a * np.cos(theta_rad) - b) / denominator)
        
        # Calculate current distance to the wall
        Dt = b * np.cos(alpha)
        
        # Calculate future distance with look-ahead
        Dt_future = Dt + self.looking_ahead_distance * np.sin(alpha)
        
        # Error is the difference between desired and projected distance
        error = self.desired_distance - Dt_future
        
        return error
    
    def adaptive_velocity(self, error, steering):
        """
        Adjust velocity based on error magnitude and steering angle
        
        Args:
            error: Current error from desired path
            steering: Current steering angle
            
        Returns:
            speed: Appropriate speed based on conditions
        """
        # Calculate absolute values for decision making
        abs_error = abs(error)
        abs_steering = abs(steering)
        
        # Base speed on a combination of error and steering
        if abs_error < self.angle_to_radian(10) and abs_steering < 0.1:
            # Very small error and steering - go fast
            return 6.0
        elif abs_error < self.angle_to_radian(20) and abs_steering < 0.2:
            # Moderate error and steering - medium speed
            return 4.0
        elif abs_error < self.angle_to_radian(30) and abs_steering < 0.3:
            # Larger error - slower
            return 2.5
        else:
            # Significant error or sharp turn - be cautious
            return 1.5
    
    def reset(self):
        """Reset the controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_reading = True
