import numpy as np
from absl import logging
from utils.Track import Track

class LatticePlannerPolicy:
    """
    Lattice Planner policy for F1Tenth gym environment
    Generates multiple potential trajectories (lattice) and selects the optimal one.
    Designed to handle overtaking scenarios in racing.
    """
    def __init__(self, track: Track, num_trajectories=10, planning_horizon=10.0):
        # Configuration
        self.track = track
        self.num_trajectories = num_trajectories  # Number of lateral trajectories to consider
        self.planning_horizon = planning_horizon  # How far ahead to plan in s coordinate
        self.planning_resolution = 30
        
        # Trajectory parameters
        self.lateral_offsets = np.linspace(-2, 2, num_trajectories)  # Lateral offset options
        
        # Speed control
        self.max_speed = 6.0
        self.min_speed = 0.5
        
        # Obstacle parameters
        self.obstacle_view_angle = np.pi / 4
        self.obstacle_detection_range = 1.0  # Increased range to detect obstacles in lidar
        self.min_obstacle_separation = 0.3  # Minimum separation between obstacle points to be considered different obstacles
        self.max_obstacles = 3  # Maximum number of obstacles to track
        
        # Cost weights
        self.w_deviation = 0.0  # Weight for centerline deviation
        self.w_velocity = 1.0   # Weight for velocity (higher is better)
        self.w_obstacle = 5.0   # Weight for obstacle avoidance (higher is safer)
        self.w_smoothness = 1.0 # Weight for trajectory smoothness
        
        logging.info(f"Lattice Planner initialized with {num_trajectories} trajectory options")
    
    def predict(self, observation, deterministic=True):
        """
        Implements the predict interface expected by the evaluate function
        
        Args:
            observation: The environment observation [s, ey, vel, yaw_angle, lidar_scan, env_params(optional)]
            deterministic: Whether to use deterministic actions (ignored for lattice planner)
            
        Returns:
            action: [steering, speed] action
            _: None (to match the RL policy interface)
        """
        # Extract state from observation
        current_s = observation[0]
        current_ey = observation[1]
        current_vel = observation[2]
        current_yaw = observation[3]
        
        # Extract lidar scan (starts at index 4, length 1080)
        lidar_scan = observation[4:1084]
        
        # Detect obstacles using lidar
        obstacles = self._detect_opponent(lidar_scan)
        
        # Generate candidate trajectories
        trajectories = self._generate_trajectories(current_s, current_ey, current_vel, current_yaw)
        
        # Evaluate trajectories and select the best one
        best_trajectory, best_cost = self._evaluate_trajectories(
            trajectories, 
            current_s, 
            current_ey, 
            current_vel,
            obstacles
        )
        
        # Extract control inputs from best trajectory
        steering, target_speed = self._get_control_from_trajectory(
            best_trajectory, 
            current_s, 
            current_ey, 
            current_vel, 
            current_yaw
        )
        
        # Clip steering to valid range
        steering = np.clip(steering, -0.4189, 0.4189)
        
        return np.array([steering, target_speed]), None
    
    def _detect_opponent(self, lidar_scan):
        """
        Detect multiple obstacles using lidar scan in all directions
        
        Args:
            lidar_scan: 1080 length lidar scan
            
        Returns:
            obstacles: List of detected obstacles, each with [distance, angle, estimated_s, estimated_ey]
        """
        angle_inc = 1.5 * np.pi / len(lidar_scan)
        angles = np.arange(len(lidar_scan)) * angle_inc - 0.75 * np.pi
        
        # Filter points based on wider view angle (180 degrees)
        view_indices = np.where(np.abs(angles) < self.obstacle_view_angle)[0]
        view_scan = lidar_scan[view_indices]
        view_angles = angles[view_indices]
        
        # Find all points within detection range
        close_points_indices = np.where((view_scan < self.obstacle_detection_range) & 
                                       (view_scan > 0.1))[0]  # Exclude very close points (likely noise)
        
        obstacles = []
        
        if len(close_points_indices) > 0:
            # Sort by distance to process closest obstacles first
            sorted_indices = close_points_indices[np.argsort(view_scan[close_points_indices])]
            
            # Iterate through detected points and cluster into obstacles
            for idx in sorted_indices:
                if len(obstacles) >= self.max_obstacles:
                    break
                
                distance = view_scan[idx]
                angle = view_angles[idx]
                
                # Skip if this point is too close to an existing obstacle (i.e., part of the same obstacle)
                is_new_obstacle = True
                for obs in obstacles:
                    # Calculate distance between this point and existing obstacle in polar coordinates
                    dx = distance * np.cos(angle) - obs[0] * np.cos(obs[1])
                    dy = distance * np.sin(angle) - obs[0] * np.sin(obs[1])
                    if np.sqrt(dx**2 + dy**2) < self.min_obstacle_separation:
                        is_new_obstacle = False
                        break
                
                # If this is a new obstacle, add it to the list
                if is_new_obstacle:
                    # Store [distance, angle, estimated_s, estimated_ey]
                    # Note: s and ey will be filled in _evaluate_trajectories based on current position
                    obstacles.append([distance, angle, None, None])
        
        return obstacles
    
    def _generate_trajectories(self, current_s, current_ey, current_vel, current_yaw):
        """
        Generate a set of candidate trajectories with different lateral offsets
        
        Args:
            current_s: Current position along track centerline
            current_ey: Current lateral offset from centerline
            current_vel: Current velocity
            current_yaw: Current heading
            
        Returns:
            trajectories: List of trajectory options, each with [s_points, ey_points, v_points]
        """
        trajectories = []
        
        # Create sample points along s
        s_points = np.linspace(current_s, 
                              (current_s + self.planning_horizon) % self.track.s_frame_max, 
                              self.planning_resolution)
        
        for target_offset in self.lateral_offsets:
            # Generate a trajectory that smoothly transitions from current offset to target
            # Using a simple exponential decay model for lateral offset
            decay_factor = np.exp(-np.linspace(0, 3, len(s_points)))
            ey_points = current_ey * decay_factor + target_offset * (1 - decay_factor)
            
            # Set velocities based on curvature and obstacles
            v_points = np.ones_like(s_points) * self.max_speed
            
            # Adjust velocity based on curvature
            for i, s in enumerate(s_points):
                curvature = self.track.curvature(s)
                v_points[i] = self._adaptive_speed(curvature, abs(ey_points[i]))
            
            trajectories.append({
                's_points': s_points,
                'ey_points': ey_points,
                'v_points': v_points
            })
            
        return trajectories
    
    def _evaluate_trajectories(self, trajectories, current_s, current_ey, current_vel, obstacles):
        """
        Evaluate all trajectories and select the best one based on cost function
        
        Args:
            trajectories: List of candidate trajectories
            current_s, current_ey, current_vel: Current state
            obstacles: List of detected obstacles [distance, angle, estimated_s, estimated_ey]
            
        Returns:
            best_trajectory: Selected trajectory
            best_cost: Cost of the selected trajectory
        """
        best_cost = float('inf')
        best_trajectory = None
        
        # First, calculate the estimated s and ey for each obstacle (if not already calculated)
        for obstacle in obstacles:
            if obstacle[2] is None or obstacle[3] is None:  # If s and ey not yet calculated
                # Convert from polar to frenet coordinates
                distance, angle = obstacle[0], obstacle[1]
                # Estimate obstacle's position in frenet coordinates
                obstacle_s_est = current_s + distance * np.cos(angle)
                obstacle_ey_est = current_ey + distance * np.sin(angle)
                # Update the obstacle with estimated coordinates
                obstacle[2] = obstacle_s_est
                obstacle[3] = obstacle_ey_est
        
        for traj in trajectories:
            # Calculate various costs
            # 1. Deviation cost - penalty for deviating from centerline
            deviation_cost = np.mean(np.abs(traj['ey_points'])) * self.w_deviation
            
            # 2. Velocity cost - reward for higher speeds
            velocity_cost = -np.mean(traj['v_points']) * self.w_velocity
            
            # 3. Obstacle cost - high penalty for trajectories close to any obstacle
            obstacle_cost = 0
            
            if obstacles:  # If there are any obstacles
                # For each point in the trajectory
                for i, s in enumerate(traj['s_points']):
                    ey = traj['ey_points'][i]
                    
                    # Calculate minimum distance to any obstacle from this trajectory point
                    min_dist_to_obstacle = float('inf')
                    
                    for obstacle in obstacles:
                        # Get obstacle coordinates
                        obstacle_s, obstacle_ey = obstacle[2], obstacle[3]
                        
                        # Use s-coordinate distance (considering track loop)
                        s_diff = min(
                            abs(s - obstacle_s),
                            abs(s - obstacle_s + self.track.s_frame_max),
                            abs(s - obstacle_s - self.track.s_frame_max)
                        )
                        
                        # Calculate Euclidean distance in frenet space
                        dist_to_obstacle = np.sqrt(s_diff**2 + (ey - obstacle_ey)**2)
                        
                        # Remember minimum distance
                        min_dist_to_obstacle = min(min_dist_to_obstacle, dist_to_obstacle)
                    
                    # Add cost inversely proportional to minimum distance
                    # Using a safety threshold to heavily penalize very close approaches
                    safety_threshold = 0.5
                    if min_dist_to_obstacle < safety_threshold:
                        # Exponential cost increase as distance approaches zero
                        obstacle_cost += (1.0 / max(0.1, min_dist_to_obstacle)) * self.w_obstacle
                    else:
                        # Linear cost for distances beyond safety threshold
                        obstacle_cost += (1.0 / min_dist_to_obstacle) * self.w_obstacle * 0.5
            
            # 4. Smoothness cost - penalize rapid changes in lateral position
            ey_diffs = np.diff(traj['ey_points'])
            smoothness_cost = np.sum(ey_diffs**2) * self.w_smoothness
            
            # Total cost
            total_cost = deviation_cost + velocity_cost + obstacle_cost + smoothness_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_trajectory = traj
        
        return best_trajectory, best_cost
    
    def _get_control_from_trajectory(self, trajectory, current_s, current_ey, current_vel, current_yaw):
        """
        Extract control inputs from the selected trajectory
        
        Args:
            trajectory: Selected trajectory
            current_s, current_ey, current_vel, current_yaw: Current state
            
        Returns:
            steering: Steering angle
            speed: Target speed
        """
        # Get next point on trajectory
        lookahead_distance = min(1.0, current_vel * 0.5)  # Dynamic lookahead based on speed
        
        # Find point approximately lookahead_distance away on s
        lookahead_s = (current_s + lookahead_distance) % self.track.s_frame_max
        
        # Find closest s value in the trajectory
        s_diffs = np.abs(trajectory['s_points'] - lookahead_s)
        closest_idx = np.argmin(s_diffs)
        
        # Get target lateral offset and speed
        target_ey = trajectory['ey_points'][closest_idx]
        target_speed = trajectory['v_points'][closest_idx]
        
        # Get cartesian coordinates for steering calculation
        current_x, current_y, track_yaw = self.track.frenet_to_cartesian(current_s, current_ey, 0)
        target_x, target_y, _ = self.track.frenet_to_cartesian(lookahead_s, target_ey, 0)
        
        # Calculate desired heading
        dx = target_x - current_x
        dy = target_y - current_y
        desired_yaw = np.arctan2(dy, dx)
        
        # Calculate cross-track error
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)
        
        # Calculate steering (simple proportional control)
        steering = yaw_error * 0.5
        
        return steering, target_speed
    
    def _adaptive_speed(self, curvature, lateral_offset):
        """
        Adjust speed based on path curvature and lateral offset
        
        Args:
            curvature: Path curvature
            lateral_offset: Distance from centerline
            
        Returns:
            speed: Appropriate speed
        """
        # Higher curvature (sharper turn) = slower speed
        curvature = max(0.0001, min(abs(curvature), 1.0))  # Bound curvature
        
        # Reduce speed when far from centerline
        offset_factor = 1.0 - 0.5 * min(1.0, abs(lateral_offset))
        
        speed = self.max_speed * offset_factor / (1.0 + 7.0 * curvature)
        return max(self.min_speed, min(speed, self.max_speed))
    
    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """Reset any internal state if needed"""
        pass
