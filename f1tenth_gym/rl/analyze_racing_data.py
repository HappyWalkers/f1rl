#!/usr/bin/env python3
"""
Script to analyze the collected racing data from different policies.
This loads JSON files containing episode data and provides simple analysis.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data_file(filepath):
    """Load data from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_policy_name(filename):
    """Extract policy name from the filename"""
    # Assuming filename format: eval_data_POLICY_racing1_TIMESTAMP.json
    parts = os.path.basename(filename).split('_')
    if len(parts) > 2:
        return parts[2].upper()  # Convert to uppercase for consistency
    return "UNKNOWN"

def calculate_statistics(data):
    """Calculate basic statistics for the data"""
    if not data:
        return None
        
    stats = {
        "num_episodes": len(data),
        "total_steps": sum(episode["episode_length"] for episode in data),
        "avg_episode_length": np.mean([episode["episode_length"] for episode in data]),
        "avg_reward": np.mean([episode["total_reward"] for episode in data]),
        "avg_time": np.mean([episode["episode_time"] for episode in data]),
    }
    
    # Calculate average speed across all episodes
    all_speeds = []
    for episode in data:
        for step in episode["steps"]:
            # Speed is in the structured observation
            all_speeds.append(step["observation"]["state"]["vel"])
    
    if all_speeds:
        stats["avg_speed"] = np.mean(all_speeds)
        stats["max_speed"] = np.max(all_speeds)
    
    # Calculate average lateral offset (distance from centerline)
    all_ey = []
    for episode in data:
        for step in episode["steps"]:
            # Lateral offset is in the structured observation
            all_ey.append(abs(step["observation"]["state"]["ey"]))
    
    if all_ey:
        stats["avg_abs_lateral_offset"] = np.mean(all_ey)
    
    return stats

def plot_trajectories(data, policy_name):
    """Plot the trajectories followed by the policy"""
    plt.figure(figsize=(12, 10))
    
    # Plot each episode as a separate color
    for i, episode in enumerate(data):
        # Extract s and ey (arc length and lateral offset) from each step
        s_values = []
        ey_values = []
        
        for step in episode["steps"]:
            s_values.append(step["observation"]["state"]["s"])
            ey_values.append(step["observation"]["state"]["ey"])
        
        plt.plot(s_values, ey_values, label=f"Episode {i+1}")
    
    plt.title(f"Trajectories for {policy_name} Policy")
    plt.xlabel("Arc Length (s)")
    plt.ylabel("Lateral Offset (ey)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_speed_profiles(data, policy_name):
    """Plot the speed profiles for each episode"""
    plt.figure(figsize=(12, 6))
    
    # Plot each episode as a separate color
    for i, episode in enumerate(data):
        # Extract s and velocity from each step
        s_values = []
        vel_values = []
        
        for step in episode["steps"]:
            s_values.append(step["observation"]["state"]["s"])
            vel_values.append(step["observation"]["state"]["vel"])
        
        plt.plot(s_values, vel_values, label=f"Episode {i+1}")
    
    plt.title(f"Speed Profiles for {policy_name} Policy")
    plt.xlabel("Arc Length (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_steering_profiles(data, policy_name):
    """Plot the steering profiles for each episode"""
    plt.figure(figsize=(12, 6))
    
    # Plot each episode as a separate color
    for i, episode in enumerate(data):
        # Extract s and steering from each step
        s_values = []
        steering_values = []
        
        for step in episode["steps"]:
            s_values.append(step["observation"]["state"]["s"])
            # Steering is in the structured action
            steering_values.append(step["action"]["steering"])
        
        plt.plot(s_values, steering_values, label=f"Episode {i+1}")
    
    plt.title(f"Steering Profiles for {policy_name} Policy")
    plt.xlabel("Arc Length (s)")
    plt.ylabel("Steering Angle")
    plt.grid(True)
    plt.legend()
    plt.show()

def compare_policies(policy_stats):
    """Compare statistics across different policies"""
    if not policy_stats:
        print("No policy statistics available for comparison")
        return
        
    policies = list(policy_stats.keys())
    
    # Common metrics to compare
    metrics = [
        "avg_episode_length", 
        "avg_reward",
        "avg_time",
        "avg_speed", 
        "max_speed",
        "avg_abs_lateral_offset"
    ]
    
    # Filter to metrics that exist in all policies
    available_metrics = []
    for metric in metrics:
        if all(metric in policy_stats[p] for p in policies):
            available_metrics.append(metric)
    
    # Create comparison bar charts
    for metric in available_metrics:
        plt.figure(figsize=(10, 6))
        values = [policy_stats[p][metric] for p in policies]
        bars = plt.bar(policies, values)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(f"Comparison of {metric}")
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze collected racing data")
    parser.add_argument("data_dir", type=str, help="Directory containing JSON data files")
    parser.add_argument("--plot_trajectories", action="store_true", help="Plot trajectories for each policy")
    parser.add_argument("--plot_speed", action="store_true", help="Plot speed profiles for each policy")
    parser.add_argument("--plot_steering", action="store_true", help="Plot steering profiles for each policy")
    parser.add_argument("--compare", action="store_true", help="Compare policies against each other")
    args = parser.parse_args()
    
    # Discover all JSON files in the directory
    data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                 if f.endswith('.json') and os.path.isfile(os.path.join(args.data_dir, f))]
    
    if not data_files:
        print(f"No JSON files found in {args.data_dir}")
        return
    
    print(f"Found {len(data_files)} data files")
    
    # Load and process each file
    policy_data = {}
    policy_stats = {}
    
    for file_path in data_files:
        policy_name = extract_policy_name(file_path)
        print(f"Processing {os.path.basename(file_path)} (Policy: {policy_name})")
        
        data = load_data_file(file_path)
        if not data:
            continue
            
        policy_data[policy_name] = data
        policy_stats[policy_name] = calculate_statistics(data)
        
        print(f"  Episodes: {policy_stats[policy_name]['num_episodes']}")
        print(f"  Total steps: {policy_stats[policy_name]['total_steps']}")
        print(f"  Avg episode length: {policy_stats[policy_name]['avg_episode_length']:.2f} steps")
        print(f"  Avg reward: {policy_stats[policy_name]['avg_reward']:.2f}")
        print(f"  Avg episode time: {policy_stats[policy_name]['avg_time']:.2f} seconds")
        
        if "avg_speed" in policy_stats[policy_name]:
            print(f"  Avg speed: {policy_stats[policy_name]['avg_speed']:.2f} m/s")
            print(f"  Max speed: {policy_stats[policy_name]['max_speed']:.2f} m/s")
            
        if "avg_abs_lateral_offset" in policy_stats[policy_name]:
            print(f"  Avg abs lateral offset: {policy_stats[policy_name]['avg_abs_lateral_offset']:.2f} m")
        
        # Generate plots if requested
        if args.plot_trajectories:
            plot_trajectories(data, policy_name)
            
        if args.plot_speed:
            plot_speed_profiles(data, policy_name)
            
        if args.plot_steering:
            plot_steering_profiles(data, policy_name)
    
    # Compare policies if requested
    if args.compare and len(policy_stats) > 1:
        compare_policies(policy_stats)

if __name__ == "__main__":
    main() 