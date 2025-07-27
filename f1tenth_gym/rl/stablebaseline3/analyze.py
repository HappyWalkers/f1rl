import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import numpy as np
import datetime
import os
import logging

def compute_statistics(env_episode_rewards, env_episode_lengths, env_lap_times, env_velocities, num_envs):
    """Computes statistics from evaluation results including velocity and acceleration."""
    env_stats = []
    for env_idx in range(num_envs):
        env_mean_reward = np.mean(env_episode_rewards[env_idx])
        env_std_reward = np.std(env_episode_rewards[env_idx])
        env_mean_episode_length = np.mean(env_episode_lengths[env_idx])
        env_mean_lap_time = np.mean(env_lap_times[env_idx])
        
        # Compute velocity and acceleration statistics
        velocities = np.array(env_velocities[env_idx]) if len(env_velocities[env_idx]) > 0 else np.array([])
        if len(velocities) > 0:
            env_mean_velocity = np.mean(velocities)
            env_std_velocity = np.std(velocities)
            env_max_velocity = np.max(velocities)
            env_min_velocity = np.min(velocities)
            
            # Calculate acceleration (assuming dt = 0.02 for 50Hz update rate)
            if len(velocities) > 1:
                dt = 0.02
                accelerations = np.diff(velocities) / dt
                accelerations = np.clip(accelerations, -10, 10)  # Clip extreme values
                
                env_mean_acceleration = np.mean(accelerations)
                env_std_acceleration = np.std(accelerations)
                env_max_acceleration = np.max(accelerations)
                env_min_acceleration = np.min(accelerations)
                
                # Separate positive (acceleration) and negative (braking) accelerations
                positive_accel = accelerations[accelerations > 0]
                negative_accel = accelerations[accelerations < 0]
                
                env_mean_positive_accel = np.mean(positive_accel) if len(positive_accel) > 0 else 0.0
                env_mean_negative_accel = np.mean(negative_accel) if len(negative_accel) > 0 else 0.0
            else:
                env_mean_acceleration = 0.0
                env_std_acceleration = 0.0
                env_max_acceleration = 0.0
                env_min_acceleration = 0.0
                env_mean_positive_accel = 0.0
                env_mean_negative_accel = 0.0
        else:
            env_mean_velocity = 0.0
            env_std_velocity = 0.0
            env_max_velocity = 0.0
            env_min_velocity = 0.0
            env_mean_acceleration = 0.0
            env_std_acceleration = 0.0
            env_max_acceleration = 0.0
            env_min_acceleration = 0.0
            env_mean_positive_accel = 0.0
            env_mean_negative_accel = 0.0
        
        env_stats.append({
            "env_idx": env_idx,
            "mean_reward": env_mean_reward,
            "std_reward": env_std_reward,
            "mean_episode_length": env_mean_episode_length,
            "mean_lap_time": env_mean_lap_time,
            "mean_velocity": env_mean_velocity,
            "std_velocity": env_std_velocity,
            "max_velocity": env_max_velocity,
            "min_velocity": env_min_velocity,
            "mean_acceleration": env_mean_acceleration,
            "std_acceleration": env_std_acceleration,
            "max_acceleration": env_max_acceleration,
            "min_acceleration": env_min_acceleration,
            "mean_positive_acceleration": env_mean_positive_accel,
            "mean_braking": env_mean_negative_accel,
            "episode_rewards": env_episode_rewards[env_idx],
            "episode_lengths": env_episode_lengths[env_idx],
            "lap_times": env_lap_times[env_idx],
            "velocities": env_velocities[env_idx]
        })
        
        logging.info(f"Environment {env_idx+1} statistics:")
        logging.info(f"  Mean reward: {env_mean_reward:.2f} ± {env_std_reward:.2f}")
        logging.info(f"  Mean episode length: {env_mean_episode_length:.2f} steps")
        logging.info(f"  Mean lap time: {env_mean_lap_time:.2f} seconds")
        logging.info(f"  Mean velocity: {env_mean_velocity:.2f} ± {env_std_velocity:.2f} m/s")
        logging.info(f"  Velocity range: {env_min_velocity:.2f} - {env_max_velocity:.2f} m/s")
        logging.info(f"  Mean acceleration: {env_mean_acceleration:.2f} ± {env_std_acceleration:.2f} m/s²")
        logging.info(f"  Acceleration range: {env_min_acceleration:.2f} - {env_max_acceleration:.2f} m/s²")
        logging.info(f"  Mean positive acceleration: {env_mean_positive_accel:.2f} m/s²")
        logging.info(f"  Mean braking: {env_mean_negative_accel:.2f} m/s²")
    
    # Compute overall statistics
    all_rewards = [reward for env_rewards in env_episode_rewards for reward in env_rewards]
    all_lengths = [length for env_lengths in env_episode_lengths for length in env_lengths]
    all_lap_times = [time for env_times in env_lap_times for time in env_times]
    all_velocities = [vel for env_vels in env_velocities for vel in env_vels]
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_episode_length = np.mean(all_lengths)
    mean_lap_time = np.mean(all_lap_times)
    
    # Overall velocity and acceleration statistics
    if len(all_velocities) > 0:
        all_velocities = np.array(all_velocities)
        mean_velocity = np.mean(all_velocities)
        std_velocity = np.std(all_velocities)
        max_velocity = np.max(all_velocities)
        min_velocity = np.min(all_velocities)
        
        # Calculate overall acceleration
        if len(all_velocities) > 1:
            dt = 0.02
            all_accelerations = []
            # Calculate accelerations for each environment separately to maintain temporal continuity
            for env_vels in env_velocities:
                if len(env_vels) > 1:
                    env_accels = np.diff(np.array(env_vels)) / dt
                    env_accels = np.clip(env_accels, -10, 10)
                    all_accelerations.extend(env_accels)
            
            if len(all_accelerations) > 0:
                all_accelerations = np.array(all_accelerations)
                mean_acceleration = np.mean(all_accelerations)
                std_acceleration = np.std(all_accelerations)
                max_acceleration = np.max(all_accelerations)
                min_acceleration = np.min(all_accelerations)
                
                positive_accel = all_accelerations[all_accelerations > 0]
                negative_accel = all_accelerations[all_accelerations < 0]
                
                mean_positive_accel = np.mean(positive_accel) if len(positive_accel) > 0 else 0.0
                mean_negative_accel = np.mean(negative_accel) if len(negative_accel) > 0 else 0.0
            else:
                mean_acceleration = std_acceleration = max_acceleration = min_acceleration = 0.0
                mean_positive_accel = mean_negative_accel = 0.0
        else:
            mean_acceleration = std_acceleration = max_acceleration = min_acceleration = 0.0
            mean_positive_accel = mean_negative_accel = 0.0
    else:
        mean_velocity = std_velocity = max_velocity = min_velocity = 0.0
        mean_acceleration = std_acceleration = max_acceleration = min_acceleration = 0.0
        mean_positive_accel = mean_negative_accel = 0.0
    
    logging.info(f"Overall evaluation completed:")
    logging.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logging.info(f"  Mean episode length: {mean_episode_length:.2f} steps")
    logging.info(f"  Mean lap time: {mean_lap_time:.2f} seconds")
    logging.info(f"  Mean velocity: {mean_velocity:.2f} ± {std_velocity:.2f} m/s")
    logging.info(f"  Velocity range: {min_velocity:.2f} - {max_velocity:.2f} m/s")
    logging.info(f"  Mean acceleration: {mean_acceleration:.2f} ± {std_acceleration:.2f} m/s²")
    logging.info(f"  Acceleration range: {min_acceleration:.2f} - {max_acceleration:.2f} m/s²")
    logging.info(f"  Mean positive acceleration: {mean_positive_accel:.2f} m/s²")
    logging.info(f"  Mean braking: {mean_negative_accel:.2f} m/s²")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_episode_length,
        "mean_lap_time": mean_lap_time,
        "mean_velocity": mean_velocity,
        "std_velocity": std_velocity,
        "max_velocity": max_velocity,
        "min_velocity": min_velocity,
        "mean_acceleration": mean_acceleration,
        "std_acceleration": std_acceleration,
        "max_acceleration": max_acceleration,
        "min_acceleration": min_acceleration,
        "mean_positive_acceleration": mean_positive_accel,
        "mean_braking": mean_negative_accel,
        "episode_rewards": all_rewards,
        "episode_lengths": all_lengths,
        "lap_times": all_lap_times,
        "velocities": all_velocities,
        "env_stats": env_stats
    }

def plot_velocity_profiles(env_positions, env_velocities, env_params, num_envs, track=None, model_path=None, algorithm="SAC"):
    """Creates and saves velocity profile plots."""
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = f"./velocity_profiles_{timestamp}"
    
    if model_path is not None:
        output_dir = os.path.dirname(model_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, f"velocity_profiles_{timestamp}")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # Overview plot
    plt.figure(figsize=(15, 10))
    plt.title(f"Velocity Profiles Overview - {algorithm} Algorithm")
    
    env_colors = plt.cm.tab10(np.linspace(0, 1, num_envs))
    legend_patches = []
    using_frenet = False
    
    for env_idx in range(num_envs):
        if len(env_positions[env_idx]) == 0:
            continue
        
        positions = np.array(env_positions[env_idx])
        velocities = np.array(env_velocities[env_idx])
        
        if len(positions) == 0:
            continue
        
        # Determine coordinate system
        is_frenet = abs(positions[0][0]) > 10 and abs(positions[0][1]) < 5
        if is_frenet:
            using_frenet = True
        
        # Individual env plot
        plt.figure(figsize=(12, 8))
        
        # Format parameter string
        param_string = ""
        if env_params[env_idx] is not None:
            param_info = []
            for key, value in env_params[env_idx].items():
                if key in ["mu", "C_Sf", "C_Sr", "m", "I", "lidar_noise_stddev"]:
                    param_info.append(f"{key}={value:.3f}")
            if param_info:
                param_string = ", ".join(param_info)
        
        plot_title = f"Velocity Profile - Env {env_idx+1}/{num_envs}"
        if param_string:
            plot_title += f" ({param_string})"
        
        scatter = plt.scatter(positions[:, 0], positions[:, 1], c=velocities, 
                              cmap='viridis', s=10, alpha=0.7)
        
        if is_frenet:
            plt.xlabel('s - Distance along track (m)')
            plt.ylabel('ey - Lateral deviation (m)')
            s_min, s_max = positions[:, 0].min(), positions[:, 0].max()
            plt.plot([s_min, s_max], [0, 0], 'k-', linewidth=1, alpha=0.5)
            plt.ylim(-2, 2)
        else:
            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
            if track is not None and hasattr(track, 'centerline'):
                plt.plot(track.centerline.xs, track.centerline.ys, 'k-', linewidth=1, alpha=0.5)
            plt.axis('equal')
        
        plt.title(plot_title)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Velocity (m/s)')
        
        env_plot_filename = os.path.join(plot_dir, f"velocity_profile_env_{env_idx+1}.png")
        plt.savefig(env_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to overview plot
        plt.figure(0)
        plt.scatter(positions[:, 0], positions[:, 1], c=env_colors[env_idx], 
                   s=5, alpha=0.5, edgecolors='none')
        
        legend_label = f"Env {env_idx+1}"
        if param_string:
            legend_label += f" ({param_string})"
        legend_patches.append(mpatches.Patch(color=env_colors[env_idx], label=legend_label))
    
    # Finalize overview plot
    if legend_patches:
        plt.figure(0)
        
        if using_frenet:
            plt.xlabel('s - Distance along track (m)')
            plt.ylabel('ey - Lateral deviation (m)')
            s_min = min([np.min(np.array(env_positions[i])[:, 0]) for i in range(num_envs) if len(env_positions[i]) > 0])
            s_max = max([np.max(np.array(env_positions[i])[:, 0]) for i in range(num_envs) if len(env_positions[i]) > 0])
            plt.plot([s_min, s_max], [0, 0], 'k-', linewidth=2)
            plt.ylim(-2, 2)
        else:
            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
            if track is not None and hasattr(track, 'centerline'):
                plt.plot(track.centerline.xs, track.centerline.ys, 'k-', linewidth=2)
            plt.axis('equal')
        
        plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        overview_filename = os.path.join(plot_dir, "velocity_profile_overview.png")
        plt.savefig(overview_filename, dpi=300, bbox_inches='tight')
    
    plt.close('all')

def plot_acceleration_profiles(env_positions, env_velocities, env_params, num_envs, track=None, model_path=None, algorithm="SAC"):
    """Creates and saves 2D acceleration profile plots with color-coded acceleration values."""
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = f"./acceleration_profiles_{timestamp}"
    
    if model_path is not None:
        output_dir = os.path.dirname(model_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, f"acceleration_profiles_{timestamp}")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate accelerations for each environment
    env_accelerations = []
    for env_idx in range(num_envs):
        if len(env_velocities[env_idx]) < 2:
            env_accelerations.append([])
            continue
        
        velocities = np.array(env_velocities[env_idx])
        # Calculate acceleration as change in velocity between consecutive steps
        # Assuming constant time step (dt = 0.02 for 50Hz update rate)
        dt = 0.02
        accelerations = np.diff(velocities) / dt
        
        # Handle potential numerical issues
        accelerations = np.clip(accelerations, -5, 5)  # Reasonable acceleration limits for F1/10
        env_accelerations.append(accelerations.tolist())
    
    # Overview plot
    plt.figure(figsize=(15, 10))
    plt.title(f"Acceleration Profiles Overview - {algorithm} Algorithm")
    
    env_colors = plt.cm.tab10(np.linspace(0, 1, num_envs))
    legend_patches = []
    using_frenet = False
    
    for env_idx in range(num_envs):
        if len(env_positions[env_idx]) < 2 or len(env_accelerations[env_idx]) == 0:
            continue
        
        positions = np.array(env_positions[env_idx])
        accelerations = np.array(env_accelerations[env_idx])
        
        # Skip first position since we lose one point when calculating acceleration
        positions = positions[1:]
        
        if len(positions) == 0 or len(accelerations) == 0:
            continue
        
        # Determine coordinate system
        is_frenet = abs(positions[0][0]) > 10 and abs(positions[0][1]) < 5
        if is_frenet:
            using_frenet = True
        
        # Individual env plot
        plt.figure(figsize=(12, 8))
        
        # Format parameter string
        param_string = ""
        if env_params[env_idx] is not None:
            param_info = []
            for key, value in env_params[env_idx].items():
                if key in ["mu", "C_Sf", "C_Sr", "m", "I", "lidar_noise_stddev"]:
                    param_info.append(f"{key}={value:.3f}")
            if param_info:
                param_string = ", ".join(param_info)
        
        plot_title = f"Acceleration Profile - Env {env_idx+1}/{num_envs}"
        if param_string:
            plot_title += f" ({param_string})"
        
        # Create scatter plot with acceleration as color
        # Center the colormap around zero for proper diverging visualization
        vmax = max(abs(accelerations.min()), abs(accelerations.max()))
        scatter = plt.scatter(positions[:, 0], positions[:, 1], c=accelerations, 
                              cmap='RdBu_r', s=15, alpha=0.7, edgecolors='none',
                              vmin=-vmax, vmax=vmax)
        
        if is_frenet:
            plt.xlabel('s - Distance along track (m)')
            plt.ylabel('ey - Lateral deviation (m)')
            s_min, s_max = positions[:, 0].min(), positions[:, 0].max()
            plt.plot([s_min, s_max], [0, 0], 'k-', linewidth=1, alpha=0.5)
            plt.ylim(-2, 2)
        else:
            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
            if track is not None and hasattr(track, 'centerline'):
                plt.plot(track.centerline.xs, track.centerline.ys, 'k-', linewidth=1, alpha=0.5)
            plt.axis('equal')
        
        plt.title(plot_title)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Acceleration (m/s²)')
        
        # Add text annotation to clarify colormap
        if len(accelerations) > 0:
            ax = plt.gca()
            ax.text(0.02, 0.98, 'Red: Accelerating\nBlue: Braking', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
        
        env_plot_filename = os.path.join(plot_dir, f"acceleration_profile_env_{env_idx+1}.png")
        plt.savefig(env_plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to overview plot
        plt.figure(0)
        plt.scatter(positions[:, 0], positions[:, 1], c=env_colors[env_idx], 
                   s=8, alpha=0.6, edgecolors='none')
        
        legend_label = f"Env {env_idx+1}"
        if param_string:
            legend_label += f" ({param_string})"
        legend_patches.append(mpatches.Patch(color=env_colors[env_idx], label=legend_label))
    
    # Finalize overview plot
    if legend_patches:
        plt.figure(0)
        
        if using_frenet:
            plt.xlabel('s - Distance along track (m)')
            plt.ylabel('ey - Lateral deviation (m)')
            s_min = min([np.min(np.array(env_positions[i])[1:, 0]) for i in range(num_envs) if len(env_positions[i]) > 1])
            s_max = max([np.max(np.array(env_positions[i])[1:, 0]) for i in range(num_envs) if len(env_positions[i]) > 1])
            plt.plot([s_min, s_max], [0, 0], 'k-', linewidth=2)
            plt.ylim(-2, 2)
        else:
            plt.xlabel('X position (m)')
            plt.ylabel('Y position (m)')
            if track is not None and hasattr(track, 'centerline'):
                plt.plot(track.centerline.xs, track.centerline.ys, 'k-', linewidth=2)
            plt.axis('equal')
        
        plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        overview_filename = os.path.join(plot_dir, "acceleration_profile_overview.png")
        plt.savefig(overview_filename, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    logging.info(f"2D acceleration profile plots saved to {plot_dir}")

def plot_velocity_time_profiles(env_velocities, env_desired_velocities, env_episode_lengths, env_params, num_envs, num_episodes, model_path=None, algorithm="SAC"):
    """Creates and saves velocity vs time profile plots with both observed and desired velocities, plus acceleration on right axis."""
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = f"./velocity_time_profiles_{timestamp}"
    
    if model_path is not None:
        output_dir = os.path.dirname(model_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, f"velocity_time_profiles_{timestamp}")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # Simulation time step (assuming 50Hz update rate)
    dt = 0.02
    
    # Overview plot
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = ax1.twinx()  # Create right axis for acceleration
    
    plt.title(f"Observed/Desired Velocity and Acceleration vs Time Profiles Overview - {algorithm} Algorithm")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)', color='blue')
    ax2.set_ylabel('Acceleration (m/s²) [log scale]', color='red')
    
    # Set log scale for acceleration axis in overview plot
    ax2.set_yscale('symlog', linthresh=10)
    
    env_colors = plt.cm.tab10(np.linspace(0, 1, num_envs))
    legend_patches = []
    
    for env_idx in range(num_envs):
        if len(env_velocities[env_idx]) == 0:
            continue
        
        observed_velocities = np.array(env_velocities[env_idx])
        desired_velocities = np.array(env_desired_velocities[env_idx])
        time_points = np.arange(len(observed_velocities)) * dt
        
        # Calculate acceleration from observed velocities
        accelerations = np.zeros_like(observed_velocities)
        if len(observed_velocities) > 1:
            accelerations[1:] = np.diff(observed_velocities) / dt
        
        # Individual environment plot
        fig_ind, ax1_ind = plt.subplots(figsize=(12, 8))
        ax2_ind = ax1_ind.twinx()
        
        # Format parameter string
        param_string = ""
        if env_params[env_idx] is not None:
            param_info = []
            for key, value in env_params[env_idx].items():
                if key in ["mu", "C_Sf", "C_Sr", "m", "I", "lidar_noise_stddev"]:
                    param_info.append(f"{key}={value:.3f}")
            if param_info:
                param_string = ", ".join(param_info)
        
        plot_title = f"Observed/Desired Velocity and Acceleration vs Time - Env {env_idx+1}/{num_envs}"
        if param_string:
            plot_title += f" ({param_string})"
        
        # Plot observed velocity, desired velocity, and acceleration
        line1 = ax1_ind.plot(time_points, observed_velocities, 'b-', linewidth=1.5, alpha=0.8, label='Observed Velocity')
        line2 = ax1_ind.plot(time_points, desired_velocities, 'g-', linewidth=1.5, alpha=0.8, label='Desired Velocity')
        line3 = ax2_ind.plot(time_points, accelerations, 'r-', linewidth=1.5, alpha=0.5, label='Acceleration')
        
        # Set log scale for acceleration axis (symmetric log to handle negative values)
        ax2_ind.set_yscale('symlog', linthresh=10)
        
        # Add episode boundaries if we have episode length data
        if env_idx < len(env_episode_lengths) and len(env_episode_lengths[env_idx]) > 0:
            episode_lengths = env_episode_lengths[env_idx]
            current_time = 0
            for episode_num, episode_length in enumerate(episode_lengths):
                episode_end_time = current_time + episode_length * dt
                if episode_num < len(episode_lengths) - 1:  # Don't draw line after last episode
                    ax1_ind.axvline(x=episode_end_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
                current_time = episode_end_time
        
        ax1_ind.set_xlabel('Time (s)')
        ax1_ind.set_ylabel('Velocity (m/s)', color='blue')
        ax2_ind.set_ylabel('Acceleration (m/s²) [log scale]', color='red')
        ax1_ind.set_title(plot_title)
        ax1_ind.grid(True, alpha=0.3)
        
        # Color the y-axis labels
        ax1_ind.tick_params(axis='y', labelcolor='blue')
        ax2_ind.tick_params(axis='y', labelcolor='red')
        
        # Add combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1_ind.legend(lines, labels, loc='upper right')
        
        # Add statistics text
        if len(observed_velocities) > 0 and len(desired_velocities) > 0 and len(accelerations) > 0:
            mean_obs_vel = np.mean(observed_velocities)
            max_obs_vel = np.max(observed_velocities)
            min_obs_vel = np.min(observed_velocities)
            std_obs_vel = np.std(observed_velocities)
            
            mean_des_vel = np.mean(desired_velocities)
            max_des_vel = np.max(desired_velocities)
            min_des_vel = np.min(desired_velocities)
            std_des_vel = np.std(desired_velocities)
            
            mean_acc = np.mean(accelerations)
            max_acc = np.max(accelerations)
            min_acc = np.min(accelerations)
            std_acc = np.std(accelerations)
            
            # Calculate velocity tracking error
            vel_error = np.mean(np.abs(observed_velocities - desired_velocities))
            
            stats_text = (f'Observed Velocity:\n'
                         f'  Mean: {mean_obs_vel:.2f} m/s\n'
                         f'  Max: {max_obs_vel:.2f} m/s\n'
                         f'  Min: {min_obs_vel:.2f} m/s\n'
                         f'  Std: {std_obs_vel:.2f} m/s\n\n'
                         f'Desired Velocity:\n'
                         f'  Mean: {mean_des_vel:.2f} m/s\n'
                         f'  Max: {max_des_vel:.2f} m/s\n'
                         f'  Min: {min_des_vel:.2f} m/s\n'
                         f'  Std: {std_des_vel:.2f} m/s\n\n'
                         f'Velocity Tracking:\n'
                         f'  MAE: {vel_error:.2f} m/s\n\n'
                         f'Acceleration:\n'
                         f'  Mean: {mean_acc:.2f} m/s²\n'
                         f'  Max: {max_acc:.2f} m/s²\n'
                         f'  Min: {min_acc:.2f} m/s²\n'
                         f'  Std: {std_acc:.2f} m/s²')
            ax1_ind.text(0.02, 0.98, stats_text, transform=ax1_ind.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)
        
        env_plot_filename = os.path.join(plot_dir, f"velocity_acceleration_time_env_{env_idx+1}.png")
        plt.savefig(env_plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)
        
        # Add to overview plot
        ax1.plot(time_points, observed_velocities, color=env_colors[env_idx], linewidth=1.5, alpha=0.7, linestyle='-')
        ax1.plot(time_points, desired_velocities, color=env_colors[env_idx], linewidth=1.5, alpha=0.7, linestyle=':')
        ax2.plot(time_points, accelerations, color=env_colors[env_idx], linewidth=1.5, alpha=0.7, linestyle='--')
        
        legend_label = f"Env {env_idx+1}"
        if param_string:
            # Truncate long parameter strings for legend
            if len(param_string) > 50:
                param_string = param_string[:47] + "..."
            legend_label += f" ({param_string})"
        legend_patches.append(mpatches.Patch(color=env_colors[env_idx], label=legend_label))
    
    # Finalize overview plot
    if legend_patches:
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add custom legend explaining the line styles
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Observed Velocity (solid line)'),
            Line2D([0], [0], color='green', lw=2, linestyle=':', label='Desired Velocity (dotted line)'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Acceleration (dashed line)')
        ]
        legend_elements.extend(legend_patches)
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        overview_filename = os.path.join(plot_dir, "velocity_acceleration_time_overview.png")
        plt.savefig(overview_filename, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    logging.info(f"Velocity and acceleration vs time profile plots saved to {plot_dir}")

def plot_steering_time_profiles(env_steering_angles, env_episode_lengths, env_params, num_envs, num_episodes, model_path=None, algorithm="SAC"):
    """Creates and saves steering angle and steering angle velocity vs time profile plots."""
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = f"./steering_time_profiles_{timestamp}"
    
    if model_path is not None:
        output_dir = os.path.dirname(model_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, f"steering_time_profiles_{timestamp}")
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # Simulation time step (assuming 50Hz update rate)
    dt = 0.02
    
    # Overview plot
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = ax1.twinx()  # Create right axis for steering angle velocity
    
    plt.title(f"Steering Angle and Steering Velocity vs Time Profiles Overview - {algorithm} Algorithm")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Steering Angle (rad)', color='blue')
    ax2.set_ylabel('Steering Velocity (rad/s)', color='red')
    
    env_colors = plt.cm.tab10(np.linspace(0, 1, num_envs))
    legend_patches = []
    
    for env_idx in range(num_envs):
        if len(env_steering_angles[env_idx]) == 0:
            continue
        
        steering_angles = np.array(env_steering_angles[env_idx])
        time_points = np.arange(len(steering_angles)) * dt
        
        # Calculate steering angle velocity (derivative of steering angle)
        steering_velocities = np.zeros_like(steering_angles)
        if len(steering_angles) > 1:
            steering_velocities[1:] = np.diff(steering_angles) / dt
            # # Apply reasonable limits to avoid numerical issues
            # steering_velocities = np.clip(steering_velocities, -50, 50)  # rad/s limits
        
        # Individual environment plot
        fig_ind, ax1_ind = plt.subplots(figsize=(12, 8))
        ax2_ind = ax1_ind.twinx()
        
        # Format parameter string
        param_string = ""
        if env_params[env_idx] is not None:
            param_info = []
            for key, value in env_params[env_idx].items():
                if key in ["mu", "C_Sf", "C_Sr", "m", "I", "lidar_noise_stddev"]:
                    param_info.append(f"{key}={value:.3f}")
            if param_info:
                param_string = ", ".join(param_info)
        
        plot_title = f"Steering Angle and Steering Velocity vs Time - Env {env_idx+1}/{num_envs}"
        if param_string:
            plot_title += f" ({param_string})"
        
        # Plot steering angle and steering velocity
        line1 = ax1_ind.plot(time_points, steering_angles, 'b-', linewidth=1.5, alpha=0.8, label='Steering Angle')
        line2 = ax2_ind.plot(time_points, steering_velocities, 'r-', linewidth=1.5, alpha=0.7, label='Steering Velocity')
        
        # Add episode boundaries if we have episode length data
        if env_idx < len(env_episode_lengths) and len(env_episode_lengths[env_idx]) > 0:
            episode_lengths = env_episode_lengths[env_idx]
            current_time = 0
            for episode_num, episode_length in enumerate(episode_lengths):
                episode_end_time = current_time + episode_length * dt
                if episode_num < len(episode_lengths) - 1:  # Don't draw line after last episode
                    ax1_ind.axvline(x=episode_end_time, color='gray', linestyle='--', alpha=0.6, linewidth=1)
                current_time = episode_end_time
        
        ax1_ind.set_xlabel('Time (s)')
        ax1_ind.set_ylabel('Steering Angle (rad)', color='blue')
        ax2_ind.set_ylabel('Steering Velocity (rad/s)', color='red')
        ax1_ind.set_title(plot_title)
        ax1_ind.grid(True, alpha=0.3)
        
        # Color the y-axis labels
        ax1_ind.tick_params(axis='y', labelcolor='blue')
        ax2_ind.tick_params(axis='y', labelcolor='red')
        
        # Add combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1_ind.legend(lines, labels, loc='upper right')
        
        # Add statistics text
        if len(steering_angles) > 0 and len(steering_velocities) > 0:
            mean_steering = np.mean(steering_angles)
            max_steering = np.max(steering_angles)
            min_steering = np.min(steering_angles)
            std_steering = np.std(steering_angles)
            
            mean_steering_vel = np.mean(steering_velocities)
            max_steering_vel = np.max(steering_velocities)
            min_steering_vel = np.min(steering_velocities)
            std_steering_vel = np.std(steering_velocities)
            
            # Convert radians to degrees for more intuitive reading
            mean_steering_deg = np.degrees(mean_steering)
            max_steering_deg = np.degrees(max_steering)
            min_steering_deg = np.degrees(min_steering)
            std_steering_deg = np.degrees(std_steering)
            
            mean_steering_vel_deg = np.degrees(mean_steering_vel)
            max_steering_vel_deg = np.degrees(max_steering_vel)
            min_steering_vel_deg = np.degrees(min_steering_vel)
            std_steering_vel_deg = np.degrees(std_steering_vel)
            
            stats_text = (f'Steering Angle:\n'
                         f'  Mean: {mean_steering:.3f} rad ({mean_steering_deg:.1f}°)\n'
                         f'  Max: {max_steering:.3f} rad ({max_steering_deg:.1f}°)\n'
                         f'  Min: {min_steering:.3f} rad ({min_steering_deg:.1f}°)\n'
                         f'  Std: {std_steering:.3f} rad ({std_steering_deg:.1f}°)\n\n'
                         f'Steering Velocity:\n'
                         f'  Mean: {mean_steering_vel:.3f} rad/s ({mean_steering_vel_deg:.1f}°/s)\n'
                         f'  Max: {max_steering_vel:.3f} rad/s ({max_steering_vel_deg:.1f}°/s)\n'
                         f'  Min: {min_steering_vel:.3f} rad/s ({min_steering_vel_deg:.1f}°/s)\n'
                         f'  Std: {std_steering_vel:.3f} rad/s ({std_steering_vel_deg:.1f}°/s)')
            ax1_ind.text(0.02, 0.98, stats_text, transform=ax1_ind.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)
        
        env_plot_filename = os.path.join(plot_dir, f"steering_time_env_{env_idx+1}.png")
        plt.savefig(env_plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)
        
        # Add to overview plot
        ax1.plot(time_points, steering_angles, color=env_colors[env_idx], linewidth=1.5, alpha=0.7, linestyle='-')
        ax2.plot(time_points, steering_velocities, color=env_colors[env_idx], linewidth=1.5, alpha=0.7, linestyle='--')
        
        legend_label = f"Env {env_idx+1}"
        if param_string:
            # Truncate long parameter strings for legend
            if len(param_string) > 50:
                param_string = param_string[:47] + "..."
            legend_label += f" ({param_string})"
        legend_patches.append(mpatches.Patch(color=env_colors[env_idx], label=legend_label))
    
    # Finalize overview plot
    if legend_patches:
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add custom legend explaining the line styles
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Steering Angle (solid line)'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Steering Velocity (dashed line)')
        ]
        legend_elements.extend(legend_patches)
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        overview_filename = os.path.join(plot_dir, "steering_time_overview.png")
        plt.savefig(overview_filename, dpi=300, bbox_inches='tight')
    
    plt.close('all')
    logging.info(f"Steering angle vs time profile plots saved to {plot_dir}")
