import numpy as np
from absl import logging

class PurePursuitPolicy:
    """
    Pure Pursuit policy for F1Tenth gym environment
    Follows waypoints with a lookahead distance approach
    """
    def __init__(self, waypoints=None, lookahead_distance=1.5, wheelbase=0.33):
        # Configuration
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase  # Distance between front and rear axles
        
        # Waypoints
        self.waypoints = waypoints
        self.max_reacquire = 20.0
        
        # Speed control
        self.max_speed = 6.0
        self.min_speed = 1.5
        
        logging.info("Pure Pursuit policy initialized")
        
    def predict(self, observation, deterministic=True):
        """
        Implements the predict interface expected by the evaluate function
        
        Args:
            observation: The environment observation (speed, yaw, lidar scan)
            deterministic: Whether to use deterministic actions (ignored in pure pursuit)
            
        Returns:
            action: [steering, speed] action
            _: None (to match the RL policy interface)
        """
        # Extract state from observation
        current_speed = observation[0]
        current_yaw = observation[1]
        
        # Extract position from observation context (this would need to be provided)
        # Here we assume the first two elements are speed and yaw, followed by position
        if len(observation) > 1082:  # If position is included in observation
            pose_x, pose_y = observation[2], observation[3]
        else:
            # If position is not in observation, we'd need another way to get it
            # For now, we'll assume a default position
            pose_x, pose_y = 0.0, 0.0
            
        # Find lookahead point
        lookahead_point = self._get_lookahead_point(pose_x, pose_y, current_yaw)
        logging.info(f"lookahead_point: {lookahead_point}")
        
        if lookahead_point is None:
            return np.array([0.0, self.min_speed]), None
            
        # Calculate steering angle
        steering, target_speed = self._get_actuation(current_yaw, lookahead_point, 
                                                  np.array([pose_x, pose_y]))
        
        # Clip steering to valid range
        steering = np.clip(steering, -0.4189, 0.4189)
        
        return np.array([steering, target_speed]), None
        
    def _get_lookahead_point(self, pose_x, pose_y, pose_theta):
        """
        Find the point on the waypoint path that is lookahead_distance away
        
        Args:
            pose_x: Current x position
            pose_y: Current y position
            pose_theta: Current heading angle
            
        Returns:
            lookahead_point: Point on path that is lookahead_distance away
        """
        if self.waypoints is None:
            logging.warning("No waypoints available for Pure Pursuit")
            return None
            
        position = np.array([pose_x, pose_y])
        
        # Extract x,y coordinates from waypoints
        wpts = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T
        
        # Find nearest point on the waypoint path
        nearest_point, nearest_dist, t, i = self._nearest_point_on_trajectory(position, wpts)
        
        if nearest_dist < self.lookahead_distance:
            # Find the first waypoint that is at least lookahead_distance away
            lookahead_point, i2, t2 = self._first_point_on_trajectory_intersecting_circle(
                position, self.lookahead_distance, wpts, i+t, wrap=True)
                
            if i2 is None:
                return None
                
            # Create point with speed
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed - we can use the curvature in waypoints to adjust speed
            if self.waypoints.shape[1] > 4 and len(self.waypoints) > i:
                # If there's curvature information in waypoints
                curvature = abs(self.waypoints[i, 4]) if i < len(self.waypoints) else 0
                current_waypoint[2] = self._adaptive_speed(curvature)
            else:
                current_waypoint[2] = self.max_speed
                
            return current_waypoint
            
        elif nearest_dist < self.max_reacquire:
            # If we're close enough to the path but not at lookahead distance yet
            if self.waypoints.shape[1] > 4 and i < len(self.waypoints):
                curvature = abs(self.waypoints[i, 4])
                speed = self._adaptive_speed(curvature)
            else:
                speed = self.max_speed
                
            return np.append(wpts[i, :], speed)
        else:
            # If we're far from the path
            return None
            
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
        
    def _nearest_point_on_trajectory(self, point, trajectory):
        """
        Find nearest point on the trajectory to the given point
        
        Args:
            point: Query point [x,y]
            trajectory: List of trajectory points [[x1,y1], [x2,y2], ...]
            
        Returns:
            nearest_point: Nearest point on trajectory
            nearest_distance: Distance to nearest point
            t: Interpolation parameter
            i: Index of nearest segment
        """
        dists = np.linalg.norm(trajectory - point, axis=1)
        min_dist_idx = np.argmin(dists)
        nearest_point = trajectory[min_dist_idx]
        nearest_distance = dists[min_dist_idx]
        
        return nearest_point, nearest_distance, 0, min_dist_idx
        
    def _first_point_on_trajectory_intersecting_circle(self, point, radius, trajectory, start_idx=0, wrap=False):
        """
        Find the first point on the trajectory that intersects with a circle
        
        Args:
            point: Center of circle [x,y]
            radius: Radius of circle
            trajectory: List of trajectory points
            start_idx: Index to start search from
            wrap: Whether to wrap around to the beginning of trajectory
            
        Returns:
            intersection_point: First point on trajectory that intersects circle
            i: Index of the segment containing the intersection
            t: Interpolation parameter
        """
        n = trajectory.shape[0]
        
        # Check if start_idx is valid
        if start_idx < 0 or start_idx >= n:
            start_idx = start_idx % n if wrap else 0
            
        # Search for intersection
        for i in range(n):
            idx = (start_idx + i) % n if wrap else min(start_idx + i, n-1)
            next_idx = (idx + 1) % n if wrap else min(idx + 1, n-1)
            
            # Skip if we're at the last point and not wrapping
            if idx == next_idx and not wrap:
                continue
                
            pt = trajectory[idx]
            dist = np.linalg.norm(pt - point)
            
            # If this point is beyond the lookahead distance
            if dist > radius:
                return pt, idx, 0
                
        # If no intersection found
        return None, None, None
        
    def reset(self):
        """Reset any internal state if needed"""
        pass
