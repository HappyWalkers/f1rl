import numpy as np
from absl import logging
from utils.Track import Track

class PurePursuitPolicy:
    """
    Pure Pursuit policy for F1Tenth gym environment
    Follows waypoints with a lookahead distance approach using Frenet frame.
    """
    def __init__(self, track: Track, lookahead_distance=0.5, wheelbase=0.33):
        # Configuration
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase  # Distance between front and rear axles
        self.track = track
        
        # Speed control
        self.max_speed = 4.0
        self.min_speed = 0.5
        
        logging.info("Pure Pursuit policy initialized with Frenet frame support")
        
    def predict(self, observation, deterministic=True):
        """
        Implements the predict interface expected by the evaluate function
        
        Args:
            observation: The environment observation [s, ey, vel, yaw_angle, yaw_rate, lidar_scan]
            deterministic: Whether to use deterministic actions (ignored in pure pursuit)
            
        Returns:
            action: [steering, speed] action
            _: None (to match the RL policy interface)
        """
        # Extract state from observation
        current_s = observation[0]
        current_ey = observation[1]
        current_vel = observation[2]
        current_yaw_car_global = observation[3] # Car's yaw angle in the global frame
        
        # Get current position and track yaw from track object
        current_x, current_y, current_yaw_track = self.track.frenet_to_cartesian(current_s, 0, 0)
        # Calculate the car's actual x, y using ey
        current_x -= current_ey * np.sin(current_yaw_track)
        current_y += current_ey * np.cos(current_yaw_track)
        
        # Find lookahead point in Frenet frame
        lookahead_s = (current_s + self.lookahead_distance) % self.track.s_frame_max
        lookahead_point_on_centerline = self._get_lookahead_point(lookahead_s)
        
        if lookahead_point_on_centerline is None:
            # Fallback: maintain current speed, zero steering if track data unavailable
            return np.array([0.0, max(self.min_speed, current_vel)]), None
            
        # Calculate steering angle
        # We need the lookahead point's Cartesian coordinates relative to the car
        target_x, target_y = lookahead_point_on_centerline[:2]
        steering, target_speed = self._get_actuation(
            current_yaw_car_global, # Car's actual heading
            lookahead_point_on_centerline, # Target point [x, y, speed, kappa]
            np.array([current_x, current_y]) # Car's actual position
        )
        
        # Clip steering to valid range
        steering = np.clip(steering, -0.4189, 0.4189)
        
        return np.array([steering, target_speed]), None
        
    def _get_lookahead_point(self, lookahead_s):
        """
        Find the point on the track centerline at a specific arc length s.
        
        Args:
            lookahead_s: Arc length along the track centerline.
            
        Returns:
            lookahead_info: [x, y, target_speed, curvature] at lookahead_s
        """
        if self.track is None:
            logging.warning("No track available for Pure Pursuit")
            return None
            
        # Get centerline position, curvature, and raceline speed at lookahead_s
        lookahead_x, lookahead_y = self.track.centerline.calc_position(lookahead_s)
        curvature = self.track.curvature(lookahead_s)
        # Use raceline velocity if available, otherwise adaptive speed
        # try:
        #     # Assuming raceline might have velocity info tied to s
        #     target_speed = self.track.raceline.calc_velocity(lookahead_s)
        #     target_speed = max(self.min_speed, min(target_speed, self.max_speed))
        # except AttributeError:
        #     target_speed = self._adaptive_speed(curvature)
        target_speed = self._adaptive_speed(curvature)
        
        return np.array([lookahead_x, lookahead_y, target_speed, curvature])
            
    def _adaptive_speed(self, curvature):
        """
        Adjust speed based on path curvature
        
        Args:
            curvature: Path curvature at current point
            
        Returns:
            speed: Appropriate speed based on curvature
        """
        # Higher curvature (sharper turn) = slower speed
        curvature = max(0.0001, min(abs(curvature), 1.0))  # Bound curvature
        speed = self.max_speed / (1.0 + 5.0 * curvature)
        return max(self.min_speed, min(speed, self.max_speed))
            
    def _get_actuation(self, pose_theta, lookahead_point, position):
        """
        Calculate steering angle and speed to reach lookahead point
        
        Args:
            pose_theta: Current heading angle
            lookahead_point: Target point to steer towards
            position: Current position [x,y]
            
        Returns:
            steering_angle: Steering angle to reach lookahead point
            speed: Target speed
        """
        # Calculate waypoint coordinates in car reference frame
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), 
                           lookahead_point[0:2] - position)
                           
        speed = lookahead_point[2]
        
        # If waypoint is directly ahead or behind (rare case)
        if np.abs(waypoint_y) < 1e-6:
            return 0.0, speed
            
        # Calculate steering angle using pure pursuit formula
        radius = 1/(2.0 * waypoint_y / self.lookahead_distance**2)
        steering_angle = np.arctan(self.wheelbase / radius)
        
        return steering_angle, speed
        
    def reset(self):
        """Reset any internal state if needed"""
        pass
