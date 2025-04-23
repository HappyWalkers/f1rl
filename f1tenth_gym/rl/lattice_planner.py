import numpy as np
from absl import logging
from utils.Track import Track

class LatticePlannerPolicy:
    """
    Lattice Planner policy for F1Tenth gym environment
    Generates multiple potential trajectories (lattice) and selects the optimal one.
    Designed to handle overtaking scenarios in racing.
    """
    def __init__(self, track: Track, num_trajectories=7, planning_horizon=2.0, wheelbase=0.33):
        # Configuration
        self.track = track
        self.num_trajectories = num_trajectories  # Number of lateral trajectories to consider
        self.planning_horizon = planning_horizon  # How far ahead to plan in s coordinate
        self.wheelbase = wheelbase  # Distance between front and rear axles
        
        # Trajectory parameters
        self.lateral_offsets = np.linspace(-0.8, 0.8, num_trajectories)  # Lateral offset options
        
        # Speed control
        self.max_speed = 5.0
        self.min_speed = 0.5
        self.max_accel = 3.0
        self.max_decel = 5.0
        
        # Obstacle parameters
        self.min_obstacle_distance = 0.5  # Minimum distance to maintain from obstacles
        self.obstacle_detection_range = 10.0  # Range to detect obstacles in lidar
        
        # Cost weights
        self.w_deviation = 1.0  # Weight for centerline deviation
        self.w_velocity = 2.0   # Weight for velocity (higher is better)
        self.w_obstacle = 5.0   # Weight for obstacle avoidance (higher is safer)
        self.w_smoothness = 1.5 # Weight for trajectory smoothness
        
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
        opponent_detected, opponent_distance, opponent_angle = self._detect_opponent(lidar_scan)
        
        # Generate candidate trajectories
        trajectories = self._generate_trajectories(current_s, current_ey, current_vel, current_yaw)
        
        # Evaluate trajectories and select the best one
        best_trajectory, best_cost = self._evaluate_trajectories(
            trajectories, 
            current_s, 
            current_ey, 
            current_vel,
            opponent_detected, 
            opponent_distance, 
            opponent_angle
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
        Detect opponent car using lidar scan
        
        Args:
            lidar_scan: 1080 length lidar scan
            
        Returns:
            opponent_detected: Boolean indicating if opponent is detected
            opponent_distance: Distance to opponent if detected
            opponent_angle: Angle to opponent if detected
        """
        # For simplicity, we'll assume any close points in front are the opponent
        # Real implementation would need better clustering/tracking
        angle_inc = 2 * np.pi / len(lidar_scan)
        angles = np.arange(len(lidar_scan)) * angle_inc - np.pi
        
        # Filter points in front of the car (within ±45 degrees)
        front_indices = np.where(np.abs(angles) < np.pi/4)[0]
        front_scan = lidar_scan[front_indices]
        front_angles = angles[front_indices]
        
        # Find closest point within a reasonable range
        close_points = np.where((front_scan < self.obstacle_detection_range) & 
                               (front_scan > 0.1))[0]  # Exclude very close points (likely noise)
        
        if len(close_points) > 0:
            # Find closest point
            min_idx = np.argmin(front_scan[close_points])
            min_distance = front_scan[close_points][min_idx]
            min_angle = front_angles[close_points][min_idx]
            
            # If point is close enough, consider it an opponent
            if min_distance < self.obstacle_detection_range:
                return True, min_distance, min_angle
        
        return False, None, None
    
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
                              20)
        
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
    
    def _evaluate_trajectories(self, trajectories, current_s, current_ey, current_vel, 
                              opponent_detected, opponent_distance, opponent_angle):
        """
        Evaluate all trajectories and select the best one based on cost function
        
        Args:
            trajectories: List of candidate trajectories
            current_s, current_ey, current_vel: Current state
            opponent_detected, opponent_distance, opponent_angle: Opponent info
            
        Returns:
            best_trajectory: Selected trajectory
            best_cost: Cost of the selected trajectory
        """
        best_cost = float('inf')
        best_trajectory = None
        
        for traj in trajectories:
            # Calculate various costs
            # 1. Deviation cost - penalty for deviating from centerline
            deviation_cost = np.mean(np.abs(traj['ey_points'])) * self.w_deviation
            
            # 2. Velocity cost - reward for higher speeds
            velocity_cost = -np.mean(traj['v_points']) * self.w_velocity
            
            # 3. Obstacle cost - high penalty for trajectories close to opponent
            obstacle_cost = 0
            if opponent_detected:
                # Estimate opponent's position in frenet coordinates
                opponent_s_est = current_s + opponent_distance * np.cos(opponent_angle)
                opponent_ey_est = current_ey + opponent_distance * np.sin(opponent_angle)
                
                # Check if any point in trajectory is too close to estimated opponent position
                for i, s in enumerate(traj['s_points']):
                    dist_to_opponent = np.sqrt((s - opponent_s_est)**2 + 
                                              (traj['ey_points'][i] - opponent_ey_est)**2)
                    if dist_to_opponent < self.min_obstacle_distance:
                        obstacle_cost += (self.min_obstacle_distance - dist_to_opponent) * self.w_obstacle
            
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
