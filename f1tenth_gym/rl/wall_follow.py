import numpy as np
import time
from absl import logging
from absl import flags

FLAGS = flags.FLAGS

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
        self.looking_ahead_distance = 0.5  # Look-ahead for smoother control
        
        # Get lidar mode from FLAGS if available, otherwise default to FULL
        try:
            self.lidar_mode = FLAGS.lidar_scan_in_obs_mode
        except:
            self.lidar_mode = "FULL"  # Default fallback
            
        # Calculate expected observation dimensions
        self.state_dim = 4  # [s, ey, vel, yaw_angle]
        if self.lidar_mode == "NONE":
            self.lidar_dim = 0
            self.angle_increment = 0.0  # Not used
        elif self.lidar_mode == "FULL":
            self.lidar_dim = 1080
            self.angle_increment = 0.00435  # Full resolution
        elif self.lidar_mode == "DOWNSAMPLED":
            self.lidar_dim = 108
            self.angle_increment = 0.00435 * 10  # Downsampled by factor of 10
        else:
            logging.warning(f"Unknown lidar mode: {self.lidar_mode}, defaulting to FULL")
            self.lidar_mode = "FULL"
            self.lidar_dim = 1080
            self.angle_increment = 0.00435
            
        # Check if parameters are included in observation
        try:
            self.include_params = FLAGS.include_params_in_obs
            self.param_dim = 12 if self.include_params else 0
        except:
            self.include_params = False
            self.param_dim = 0
        
        logging.info(f"Wall following policy initialized with lidar_mode={self.lidar_mode}, "
                    f"lidar_dim={self.lidar_dim}, include_params={self.include_params}")
        
    def predict(self, observation, deterministic=True):
        # Parse observation components
        obs_components = self._parse_observation(observation)
        
        # Extract velocity for adaptive speed control
        current_velocity = obs_components['velocity']
        
        # Get lidar data if available
        lidar_scan = obs_components['lidar_scan']
        
        if lidar_scan is None or len(lidar_scan) == 0:
            logging.warning("No LiDAR data available for wall following. Using default behavior.")
            # Default behavior when no lidar: straight line with moderate speed
            return np.array([0.0, 3.0]), None
        
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
    
    def _parse_observation(self, observation) -> dict:
        obs_dict = {
            's': observation[0],           # Frenet arc length
            'ey': observation[1],          # Lateral deviation
            'velocity': observation[2],    # Velocity
            'yaw_angle': observation[3],   # Yaw angle
            'lidar_scan': None,           # Will be filled if available
            'params': None                # Will be filled if available
        }
        
        # Extract lidar data if present
        lidar_start_idx = self.state_dim
        lidar_end_idx = lidar_start_idx + self.lidar_dim
        
        if self.lidar_dim > 0 and len(observation) >= lidar_end_idx:
            lidar_data = observation[lidar_start_idx:lidar_end_idx]
            obs_dict['lidar_scan'] = lidar_data
        
        # Extract parameters if present
        if self.param_dim > 0 and len(observation) >= lidar_end_idx + self.param_dim:
            param_start_idx = lidar_end_idx
            param_end_idx = param_start_idx + self.param_dim
            obs_dict['params'] = observation[param_start_idx:param_end_idx]
        
        return obs_dict
    
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
        """
        # Get measurements at 90 degrees (directly to the left) and 45 degrees
        angle_90 = self.angle_to_radian(90)
        angle_45 = self.angle_to_radian(45)
        
        b = self.get_range(ranges, angle_90)
        a = self.get_range(ranges, angle_45)

        logging.debug(f"distance to angle 90: {b}, distance to angle 45: {a}")
        
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
        # Calculate absolute values
        abs_error = abs(error)
        abs_steering = abs(steering)
        
        # Convert error from radians to a more convenient scale
        error_normalized = abs_error / self.angle_to_radian(30)
        
        # Normalize steering within its range (max is 0.4189)
        steering_normalized = abs_steering / 0.4189
        
        # Calculate weights for error and steering (can be tuned)
        error_weight = 0.6
        steering_weight = 0.4
        
        # Combined factor (0.0 = perfect, 1.0 = max deviation)
        combined_factor = error_weight * error_normalized + steering_weight * steering_normalized
        
        # Clamp the factor between 0 and 1
        combined_factor = np.clip(combined_factor, 0.0, 1.0)
        
        # Map the factor to speed range: 1.5 (cautious) to 6.0 (fast)
        min_speed = 0.5
        max_speed = 2.0
        speed = max_speed - combined_factor * (max_speed - min_speed)
        
        return speed
    
    def reset(self):
        """Reset the controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_reading = True
