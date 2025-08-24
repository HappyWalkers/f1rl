from stable_baselines3 import PPO, DDPG, DQN, TD3, SAC
from sb3_contrib import RecurrentPPO
from absl import logging
from absl import flags
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, unwrap_vec_normalize
from functools import partial
import os
import time
import numpy as np
import datetime
from tqdm import tqdm
from typing import Tuple, List, Optional, Any
from ..rl_env import F110GymWrapper # Import the wrapper
from .feature_extractor import F1TenthFeaturesExtractor, MLPFeaturesExtractor, ResNetFeaturesExtractor, TransformerFeaturesExtractor, MoEFeaturesExtractor
from sortedcontainers import SortedList
from .analyze import *
from imitation.algorithms import bc
from imitation.data import types as data_types
from imitation.data.rollout import flatten_trajectories
from ..wall_follow import WallFollowPolicy
from ..pure_pursuit import PurePursuitPolicy
from ..lattice_planner import LatticePlannerPolicy
from ..utils.Track import Track

import torch

FLAGS = flags.FLAGS

# Algorithm constants
ALGO_SAC = "sac"
ALGO_PPO = "ppo"
ALGO_RECURRENT_PPO = "recurrent_ppo"
ALGO_DDPG = "ddpg"
ALGO_TD3 = "td3"
ALGO_DQN = "dqn"
ALGO_A2C = "a2c"
ALGO_WALL_FOLLOW = "wall_follow"
ALGO_PURE_PURSUIT = "pure_pursuit"
ALGO_LATTICE = "lattice"

def is_rl_policy(algorithm: str) -> bool:
    if algorithm in [ALGO_PPO, ALGO_RECURRENT_PPO, ALGO_SAC, ALGO_TD3, ALGO_DDPG, ALGO_DQN, ALGO_A2C]:
        return True
    return False

# Function to select feature extractor based on name
def get_feature_extractor_class(feature_extractor_name):
    """Returns the appropriate feature extractor class based on the name."""
    if feature_extractor_name == "FILM":
        return F1TenthFeaturesExtractor
    elif feature_extractor_name == "MLP":
        return MLPFeaturesExtractor
    elif feature_extractor_name == "RESNET":
        return ResNetFeaturesExtractor
    elif feature_extractor_name == "TRANSFORMER":
        return TransformerFeaturesExtractor
    elif feature_extractor_name == "MOE":
        return MoEFeaturesExtractor
    else:
        logging.warning(f"Unknown feature extractor: {feature_extractor_name}, defaulting to FILM")
        return F1TenthFeaturesExtractor

def create_ppo(env, seed):
    """Create a PPO model with custom neural network architecture"""
    # Read parameters from FLAGS
    include_params_in_obs = FLAGS.include_params_in_obs
    feature_extractor_name = FLAGS.feature_extractor
    lidar_scan_in_obs_mode = FLAGS.lidar_scan_in_obs_mode
    
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    
    # Calculate lidar_dim based on mode
    if lidar_scan_in_obs_mode == "NONE":
        lidar_dim = 0
    elif lidar_scan_in_obs_mode == "FULL":
        lidar_dim = 1080
    elif lidar_scan_in_obs_mode == "DOWNSAMPLED":
        lidar_dim = 108
    else:
        raise ValueError(f"Unknown lidar_scan_in_obs_mode: {lidar_scan_in_obs_mode}")
        
    include_lidar = lidar_scan_in_obs_mode != "NONE"
    state_dim = 4
    param_dim = 12
    
    # Get the appropriate feature extractor class
    features_extractor_class = get_feature_extractor_class(feature_extractor_name)
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs,
            "include_lidar": include_lidar
        },
        "net_arch": [
            dict(pi=[256, 128, 64], vf=[256, 128, 64])
        ]
    }
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,          # how many steps to collect before an update
        batch_size=512,
        n_epochs=10,            # how many gradient steps per update
        gamma=0.99,            # discount factor
        gae_lambda=0.95,       # advantage discount
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=1,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model

def create_recurrent_ppo(env, seed):
    """Create a RecurrentPPO model with LSTM policy architecture"""
    # Read parameters from FLAGS
    include_params_in_obs = FLAGS.include_params_in_obs
    feature_extractor_name = FLAGS.feature_extractor
    lidar_scan_in_obs_mode = FLAGS.lidar_scan_in_obs_mode
    
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    
    # Calculate lidar_dim based on mode
    if lidar_scan_in_obs_mode == "NONE":
        lidar_dim = 0
    elif lidar_scan_in_obs_mode == "FULL":
        lidar_dim = 1080
    elif lidar_scan_in_obs_mode == "DOWNSAMPLED":
        lidar_dim = 108
    else:
        raise ValueError(f"Unknown lidar_scan_in_obs_mode: {lidar_scan_in_obs_mode}")
        
    include_lidar = lidar_scan_in_obs_mode != "NONE"
    state_dim = 4
    param_dim = 12
    
    # Get the appropriate feature extractor class
    features_extractor_class = get_feature_extractor_class(feature_extractor_name)
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs,
            "include_lidar": include_lidar
        },
        # LSTM settings
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True
    }
    
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,          # how many steps to collect before an update
        batch_size=512,
        n_epochs=10,           # how many gradient steps per update
        gamma=0.99,            # discount factor
        gae_lambda=0.95,       # advantage discount
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=1,
        max_grad_norm=0.5,
        target_kl=None,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model

def create_ddpg(env, seed):
    """Create a DDPG model with custom neural network architecture"""
    # Read parameters from FLAGS
    include_params_in_obs = FLAGS.include_params_in_obs
    feature_extractor_name = FLAGS.feature_extractor
    lidar_scan_in_obs_mode = FLAGS.lidar_scan_in_obs_mode
    
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    
    # Calculate lidar_dim based on mode
    if lidar_scan_in_obs_mode == "NONE":
        lidar_dim = 0
    elif lidar_scan_in_obs_mode == "FULL":
        lidar_dim = 1080
    elif lidar_scan_in_obs_mode == "DOWNSAMPLED":
        lidar_dim = 108
    else:
        raise ValueError(f"Unknown lidar_scan_in_obs_mode: {lidar_scan_in_obs_mode}")
        
    include_lidar = lidar_scan_in_obs_mode != "NONE"
    state_dim = 4
    param_dim = 12
    
    # Get the appropriate feature extractor class
    features_extractor_class = get_feature_extractor_class(feature_extractor_name)
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs,
            "include_lidar": include_lidar
        },
        "net_arch": [400, 300]
    }
    
    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=128,
        tau=0.001,
        gamma=0.99,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )
    return model

def create_td3(env, seed):
    """Create a TD3 model with custom neural network architecture"""
    # Read parameters from FLAGS
    include_params_in_obs = FLAGS.include_params_in_obs
    feature_extractor_name = FLAGS.feature_extractor
    lidar_scan_in_obs_mode = FLAGS.lidar_scan_in_obs_mode
    
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    
    # Calculate lidar_dim based on mode
    if lidar_scan_in_obs_mode == "NONE":
        lidar_dim = 0
    elif lidar_scan_in_obs_mode == "FULL":
        lidar_dim = 1080
    elif lidar_scan_in_obs_mode == "DOWNSAMPLED":
        lidar_dim = 108
    else:
        raise ValueError(f"Unknown lidar_scan_in_obs_mode: {lidar_scan_in_obs_mode}")
        
    include_lidar = lidar_scan_in_obs_mode != "NONE"
    state_dim = 4
    param_dim = 12
    
    # Get the appropriate feature extractor class
    features_extractor_class = get_feature_extractor_class(feature_extractor_name)
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs,
            "include_lidar": include_lidar
        },
        "net_arch": [400, 300]
    }
    
    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=128,
        tau=0.001,
        gamma=0.99,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )
    return model

def create_sac(env, seed):
    """Create a SAC model with custom neural network architecture"""
    # Read parameters from FLAGS
    include_params_in_obs = FLAGS.include_params_in_obs
    feature_extractor_name = FLAGS.feature_extractor
    lidar_scan_in_obs_mode = FLAGS.lidar_scan_in_obs_mode
    
    # Get the appropriate feature extractor class
    features_extractor_class = get_feature_extractor_class(feature_extractor_name)
    
    # Calculate lidar_dim based on mode
    if lidar_scan_in_obs_mode == "NONE":
        lidar_dim = 0
    elif lidar_scan_in_obs_mode == "FULL":
        lidar_dim = 1080
    elif lidar_scan_in_obs_mode == "DOWNSAMPLED":
        lidar_dim = 108
    else:
        raise ValueError(f"Unknown lidar_scan_in_obs_mode: {lidar_scan_in_obs_mode}")
        
    include_lidar = lidar_scan_in_obs_mode != "NONE"
    state_dim = 4
    param_dim = 12
    
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {
            "features_dim": 1024,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs,
            "include_lidar": include_lidar
        },
        "net_arch": [1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64]
    }
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        train_freq=(10, "step"),
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        tensorboard_log="./sac_tensorboard/",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model

def load_model_for_evaluation(algorithm: str, model_path: str = None, track: Track = None):
    """Loads or returns the model for evaluation based on algorithm type."""
    if algorithm == ALGO_WALL_FOLLOW:
        return WallFollowPolicy()
    elif algorithm == ALGO_PURE_PURSUIT:
        return PurePursuitPolicy(track=track)
    elif algorithm == ALGO_LATTICE:
        return LatticePlannerPolicy(track=track, lidar_scan_in_obs_mode=FLAGS.lidar_scan_in_obs_mode)
    elif algorithm == ALGO_PPO:
        return PPO.load(model_path)
    elif algorithm == ALGO_RECURRENT_PPO:
        return RecurrentPPO.load(model_path)
    elif algorithm == ALGO_DDPG:
        return DDPG.load(model_path)
    elif algorithm == ALGO_TD3:
        return TD3.load(model_path)
    elif algorithm == ALGO_SAC:
        return SAC.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

 

def run_evaluation_episode(eval_env: DummyVecEnv | SubprocVecEnv | VecNormalize, model, is_recurrent: bool, env_idx: int = 0) -> Tuple[float, int, float, List[tuple], List[float], List[float], List[float]]:
    """Runs a single evaluation episode on a specific vectorized env index and returns metrics.
    """
    # Reset only the specified environment (returns raw obs from underlying env)
    obs_raw = eval_env.env_method('reset', indices=[env_idx])[0]
    # If reset returns (obs, info) from gymnasium-compatible envs, take obs
    if isinstance(obs_raw, tuple) and len(obs_raw) == 2:
        obs_raw = obs_raw[0]

    def maybe_normalize_obs(observation: np.ndarray) -> np.ndarray:
        """Normalize observation using VecNormalize stats if available."""
        try:
            if isinstance(eval_env, VecNormalize) and hasattr(eval_env, 'normalize_obs'):
                obs_arr = np.array(observation, dtype=np.float32)
                if obs_arr.ndim == 1:
                    obs_arr = obs_arr[None, :]
                obs_norm = eval_env.normalize_obs(obs_arr)
                return obs_norm[0] if obs_norm.ndim > 1 else obs_norm
        except Exception:
            # Fallback to raw observation if anything goes wrong
            pass
        return observation

    # Prepare first policy input
    obs_input = maybe_normalize_obs(obs_raw)

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0
    episode_start_time = time.time()

    episode_positions = []
    episode_velocities = []
    episode_desired_velocities = []  # New: collect desired velocities from model actions
    episode_steering_angles = []  # New: collect steering angles from model actions

    while not (terminated or truncated):
        if FLAGS.render_in_eval:
            eval_env.env_method('render', indices=[env_idx])

        single_obs_for_policy = obs_input

        if is_recurrent:
            action, lstm_states = model.predict(
                single_obs_for_policy,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
        else:
            action, _states = model.predict(single_obs_for_policy, deterministic=True)

        # Extract desired velocity from model action (action[1] is desired speed)
        # Handle different action formats from different policies
        if hasattr(action, '__len__') and len(action) > 1:
            desired_velocity = float(action[1])
        elif hasattr(action, '__len__') and len(action) == 1:
            # Some policies might only output steering, use observed (RAW) velocity
            desired_velocity = float(obs_raw[2]) if len(obs_raw) > 2 else 0.0
        elif FLAGS.algorithm in [ALGO_WALL_FOLLOW, ALGO_PURE_PURSUIT, ALGO_LATTICE]:
            desired_velocity = float(action[1])
        else:
            # Fallback for other action formats
            desired_velocity = 0.0

        # Extract steering angle from model action (action[0] is steering angle)
        if hasattr(action, '__len__') and len(action) > 0:
            steering_angle = float(action[0])
        else:
            # Fallback for scalar actions
            steering_angle = float(action) if np.isscalar(action) else 0.0

        # Step only the specified env (returns RAW next obs)
        step_result = eval_env.env_method('step', np.array(action), indices=[env_idx])[0]
        # step_result expected: (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
        if len(step_result) == 5:
            next_obs_raw, reward, terminated, truncated, info = step_result
        else:
            next_obs_raw, reward, done, info = step_result
            terminated = bool(done)
            truncated = bool(info.get('TimeLimit.truncated', False) or info.get('truncated', False))
        reward = float(reward)

        # Metrics and trajectory use RAW observations
        position, velocity = extract_position_velocity(next_obs_raw, eval_env, env_idx, info)
        if position is not None and velocity is not None:
            episode_positions.append(position)
            episode_velocities.append(velocity)
            episode_desired_velocities.append(desired_velocity)
            episode_steering_angles.append(steering_angle)

        # Prepare next loop
        obs_raw = next_obs_raw
        obs_input = maybe_normalize_obs(obs_raw)

        if is_recurrent:
            episode_starts = np.zeros((1,), dtype=bool)

        total_reward += reward
        step_count += 1

    episode_time = time.time() - episode_start_time
    return total_reward, step_count, episode_time, episode_positions, episode_velocities, episode_desired_velocities, episode_steering_angles

def extract_position_velocity(obs, env, agent_idx=0, info=None):
    """Extracts position and velocity data from observation and info."""
    s = float(obs[0])  # Frenet arc length
    ey = float(obs[1])  # Lateral offset
    velocity = float(obs[2])  # Velocity
    
    position = (s, ey)  # Default to frenet coordinates
    
    # Try to get track to convert to cartesian
    track = None
    if hasattr(env, 'get_attr'):
        try:
            track = env.get_attr("track", indices=[agent_idx])[0]
        except Exception:
            track = None
    elif hasattr(env, "track"):
        track = env.track
        
    if track is not None:
        x_pos, y_pos, _ = track.frenet_to_cartesian(s, ey, 0)
        position = (x_pos, y_pos)
    
    # Try to get position from info if available
    if info and "poses_x" in info and "poses_y" in info:
        position = (info["poses_x"][agent_idx], info["poses_y"][agent_idx])
        
    return position, velocity

def evaluate(eval_env):
    # Read parameters from FLAGS
    algorithm = FLAGS.algorithm
    num_episodes = FLAGS.num_eval_episodes
    racing_mode = FLAGS.racing_mode
    model_path = FLAGS.model_path
    num_envs = FLAGS.num_envs

    if not is_rl_policy(algorithm):
        track = eval_env.get_attr("track", indices=0)[0]
        model = load_model_for_evaluation(algorithm, track=track)
    else:
        model = load_model_for_evaluation(algorithm, model_path=model_path)
    
    # Initialize metrics and trajectory data storage
    env_episode_rewards = [[] for _ in range(num_envs)]
    env_episode_lengths = [[] for _ in range(num_envs)]
    env_lap_times = [[] for _ in range(num_envs)]
    env_positions = [[] for _ in range(num_envs)]
    env_velocities = [[] for _ in range(num_envs)]
    env_desired_velocities = [[] for _ in range(num_envs)]  # New: store desired velocities
    env_steering_angles = [[] for _ in range(num_envs)]  # New: store steering angles
    env_params = [None for _ in range(num_envs)]
    
    is_recurrent = algorithm == ALGO_RECURRENT_PPO or (hasattr(model, 'policy') and hasattr(model.policy, '_initial_state'))
    
    # Evaluate on each environment
    for env_idx in range(num_envs):
        logging.info(f"Evaluating on environment {env_idx+1}/{num_envs}")
        
        for episode in range(num_episodes):
            logging.info(f"Starting evaluation episode {episode+1}/{num_episodes} on env {env_idx+1}")
            
            # Run a single evaluation episode
            total_reward, step_count, episode_time, episode_positions, episode_velocities, episode_desired_velocities, episode_steering_angles = run_evaluation_episode(
                eval_env, model, is_recurrent, env_idx=env_idx
            )
            
            # Record metrics for this environment
            env_episode_rewards[env_idx].append(total_reward)
            env_episode_lengths[env_idx].append(step_count)
            env_lap_times[env_idx].append(episode_time)
            
            # Store trajectory data for plotting
            if episode_positions and episode_velocities:
                env_positions[env_idx].extend(episode_positions)
                env_velocities[env_idx].extend(episode_velocities)
                env_desired_velocities[env_idx].extend(episode_desired_velocities)
                env_steering_angles[env_idx].extend(episode_steering_angles)
            
            logging.info(f"Episode {episode+1} finished:")
            logging.info(f"  Reward: {total_reward:.2f}")
            logging.info(f"  Length: {step_count} steps")
            logging.info(f"  Time: {episode_time:.2f} seconds")
    
    # Plot velocity profiles if we have collected data
    any_data = any(len(pos) > 0 for pos in env_positions)
    if any_data and FLAGS.plot_in_eval:
        plot_velocity_profiles(env_positions, env_velocities, env_params, num_envs, track, model_path, algorithm)
        plot_acceleration_profiles(env_positions, env_velocities, env_params, num_envs, track, model_path, algorithm)
        plot_velocity_time_profiles(env_velocities, env_desired_velocities, env_episode_lengths, env_params, num_envs, num_episodes, model_path, algorithm)
        plot_steering_time_profiles(env_steering_angles, env_episode_lengths, env_params, num_envs, num_episodes, model_path, algorithm)
    
    # Compute statistics from evaluation results
    return compute_statistics(env_episode_rewards, env_episode_lengths, env_lap_times, env_velocities, num_envs)

def initialize_with_imitation_learning(model, env, imitation_policy_type="PURE_PURSUIT", total_transitions=1000_000, racing_mode=False, algorithm="SAC"):
    """
    Initialize a reinforcement learning model using imitation learning from a specified policy.
    """
    # Check if env is a VecEnv and raise an error if it's not
    if not isinstance(env, (DummyVecEnv, SubprocVecEnv, VecNormalize)):
        raise TypeError("env must be a VecEnv instance")
    
    # Check if environment is wrapped with VecNormalize
    vec_normalize = unwrap_vec_normalize(env)
    is_normalized = vec_normalize is not None
    
    if is_normalized:
        logging.info("Environment is wrapped with VecNormalize. Expert policies will use raw observations.")
        # Get the underlying VecEnv
        raw_vec_env = vec_normalize.venv
    else:
        logging.info("Environment is not normalized.")
        raw_vec_env = env
    
    # Initialize expert policies
    expert_policies = initialize_expert_policies(raw_vec_env, imitation_policy_type, racing_mode)
    
    # Collect demonstrations from the expert policies
    logging.info(f"Collecting ~{total_transitions} transitions across {raw_vec_env.num_envs} environments")
    demonstrations = collect_expert_rollouts(model, env, raw_vec_env, expert_policies, 
                                           total_transitions, is_normalized)
    
    # Pretrain the model using the demonstrations
    logging.info("Pretraining model with demonstrations")
    model = pretrain_with_demonstrations(model, demonstrations, algorithm, env, raw_vec_env.num_envs)
    
    logging.info("Imitation learning completed")
    return model

def initialize_expert_policies(vec_env, imitation_policy_type, racing_mode):
    """
    Initialize expert policies for each environment.
    """
    # Import policies for imitation learning
    from wall_follow import WallFollowPolicy
    from pure_pursuit import PurePursuitPolicy
    from lattice_planner import LatticePlannerPolicy

    # Initialize the expert policies for each environment
    logging.info(f"Initializing {imitation_policy_type} expert policies (racing_mode={racing_mode})")
    expert_policies = []
    for i in range(vec_env.num_envs):
        # Extract the track for this environment
        track = vec_env.get_attr("track", indices=i)[0]
        # Ensure the environment has the track object needed by the policies
        if not track:
            raise ValueError("Environment does not have a 'track' attribute required for policy imitation.")
            
        if imitation_policy_type == "WALL_FOLLOW":
            expert_policies.append(WallFollowPolicy())
        elif imitation_policy_type == "PURE_PURSUIT":
            expert_policies.append(PurePursuitPolicy(track=track))
        elif imitation_policy_type == "LATTICE":
            if not racing_mode:
                logging.warning("LATTICE planner is designed for racing mode. Using it in non-racing mode may not be optimal.")
            expert_policies.append(LatticePlannerPolicy(track=track, lidar_scan_in_obs_mode=FLAGS.lidar_scan_in_obs_mode))
        else:
            raise ValueError(f"Unsupported imitation_policy_type: {imitation_policy_type}")
    
    return expert_policies

def collect_expert_rollouts(model, env, raw_vec_env, expert_policies, total_transitions, is_normalized):
    """
    Collect rollouts from expert policies and filter by reward.
    """
    # Calculate per-environment transition quota to ensure balanced data collection
    num_envs = raw_vec_env.num_envs
    transitions_per_env = total_transitions // num_envs
    
    # Track collected transitions for each environment
    env_transitions_collected = [0] * num_envs
    
    # Use sorted lists to store the best rollouts for each environment
    # Each rollout is a tuple of (total_reward, transitions)
    env_rollouts = [SortedList(key=lambda x: x[0]) for _ in range(num_envs)]

    # Create progress bar for collecting transitions
    with tqdm(total=total_transitions, desc="Collecting Expert Demonstration Rollouts") as pbar:
        while min(env_transitions_collected) < transitions_per_env * 0.9:
            # Buffer for current rollouts, one per environment
            current_rollouts = [[] for _ in range(num_envs)]
            current_rollout_rewards = [0.0 for _ in range(num_envs)]
            
            # Start a new episode by resetting the normalized environment
            normalized_observations = env.reset()
            
            # Get original (unnormalized) observations
            if is_normalized:
                raw_observations = env.get_original_obs()
            else:
                raw_observations = normalized_observations
                
            # Reset all expert policies
            for policy in expert_policies:
                if hasattr(policy, 'reset'):
                    policy.reset()
            
            # Active environment tracker
            active_envs = [True] * num_envs
            
            # Run episodes until completion in all environments
            while any(active_envs):
                # Generate expert actions for active environments
                actions = np.zeros((num_envs,) + env.action_space.shape)
                
                for i in range(num_envs):
                    if active_envs[i]:
                        # Get action from the expert policy using RAW observations
                        action, _ = expert_policies[i].predict(raw_observations[i], deterministic=True)
                        actions[i] = action
                
                # Step only the normalized environment
                next_normalized_observations, normalized_rewards, dones, infos = env.step(actions)
                
                # Get the original (unnormalized) observations
                if is_normalized:
                    next_raw_observations = env.get_original_obs()
                    raw_rewards = env.get_original_reward()
                else:
                    next_raw_observations = next_normalized_observations
                    raw_rewards = normalized_rewards
                
                # Store transitions for active environments
                for i in range(num_envs):
                    if active_envs[i]:
                        # Add the transition to the current rollout
                        current_rollouts[i].append((
                            normalized_observations[i], 
                            actions[i], 
                            normalized_rewards[i], 
                            next_normalized_observations[i], 
                            dones[i]
                        ))
                        current_rollout_rewards[i] += raw_rewards[i]
                        
                        # Check if episode finished
                        if dones[i]:
                            logging.debug(f"Episode {i} finished with reward {current_rollout_rewards[i]}")
                            active_envs[i] = False
                            
                            # Only keep rollouts with positive rewards
                            if current_rollout_rewards[i] > -1000:
                                # Add to the sorted list
                                env_rollouts[i].add((current_rollout_rewards[i], current_rollouts[i]))
                                
                                # If we have too many rollouts, remove the worst one
                                while (sum(len(rollout) for _, rollout in env_rollouts[i]) > transitions_per_env):
                                    _, removed_rollout = env_rollouts[i].pop(0)  # Remove rollout with lowest reward
                                    
                                # Update collected transitions for this environment
                                old_transitions = env_transitions_collected[i]
                                env_transitions_collected[i] = sum(len(rollout) for _, rollout in env_rollouts[i])
                                
                                # Update the progress bar with the actual number of new transitions kept
                                new_transitions = env_transitions_collected[i] - old_transitions
                                pbar.update(new_transitions)
                                if int(pbar.n / pbar.total * 100) % 10 == 0:
                                    log_message = ", ".join(f"env{i}: {env_transitions_collected[i]}/{transitions_per_env}" for i in range(num_envs))
                                    logging.debug(f"Collection Progress: {log_message}")
                                
                
                # Update observations for next step
                normalized_observations = next_normalized_observations
                raw_observations = next_raw_observations
    
    # Extract transitions grouped by environment
    transitions_by_env = [[] for _ in range(num_envs)]
    for env_idx, rollout_list_for_env in enumerate(env_rollouts):
        for _reward_val, rollout_data in rollout_list_for_env: 
            for transition in rollout_data: 
                transitions_by_env[env_idx].append(transition)

    total_collected_transitions = sum(len(transitions) for transitions in transitions_by_env)
    logging.info(f"Collected {total_collected_transitions} demonstration transitions from successful rollouts, grouped by environment.")
    
    return transitions_by_env

def pretrain_off_policy(model, demonstrations, num_envs):
    """
    Pretrain off-policy algorithms (SAC, TD3, DDPG) using demonstrations in the replay buffer.
    """
    if not hasattr(model, 'replay_buffer'):
        logging.warning("Model does not have a replay buffer. Skipping off-policy pretraining.")
        return model
    
    if not demonstrations or all(len(env_demos) == 0 for env_demos in demonstrations):
        logging.info("No demonstration transitions to pretrain on. Skipping replay buffer population.")
        return model
    
    # Check that we have the expected number of environments
    if len(demonstrations) != num_envs:
        logging.warning(f"Expected {num_envs} environments but got {len(demonstrations)}. Adjusting.")
        num_envs = len(demonstrations)
    
    # Determine how many complete batches we can form
    min_transitions = min(len(env_demos) for env_demos in demonstrations if len(env_demos) > 0)
    
    if min_transitions == 0:
        logging.warning("Not enough transitions to form batches. Skipping replay buffer population.")
        return model
    
    # Form batches and add to replay buffer
    logging.info(f"Adding {min_transitions} batches of demonstrations to the replay buffer.")
    for i in range(min_transitions):
        batch_obs_list = []
        batch_action_list = []
        batch_reward_list = []
        batch_next_obs_list = []
        batch_done_list = []

        for env_j in range(num_envs):
            if i < len(demonstrations[env_j]):
                obs, action, reward, next_obs, done = demonstrations[env_j][i]
                batch_obs_list.append(obs)
                batch_action_list.append(action)
                batch_reward_list.append(reward)
                batch_next_obs_list.append(next_obs)
                batch_done_list.append(done)
        
        # Only add batch if we have data from all environments with transitions
        if len(batch_obs_list) == num_envs:
            batch_obs = np.array(batch_obs_list)
            batch_action = np.array(batch_action_list)
            batch_reward = np.array(batch_reward_list)
            batch_next_obs = np.array(batch_next_obs_list)
            batch_done = np.array(batch_done_list)
            
            infos = [{} for _ in range(num_envs)]

            # TODO: check the interface
            model.replay_buffer.add(batch_obs, batch_next_obs, batch_action, batch_reward, batch_done, infos)
    
    # Perform gradient steps on the demonstrations
    if hasattr(model, 'train'):
        logging.info("Training on demonstrations using replay buffer")
        # Initialize the model by calling learn with 1 step
        model.learn(total_timesteps=1, log_interval=1) 
        
        # Now perform gradient steps
        gradient_steps = 10_000
        with tqdm(total=gradient_steps, desc="Off-Policy Imitation Learning Progress") as pbar:
            for _ in range(gradient_steps): 
                model.train(gradient_steps=1, batch_size=model.batch_size)
                pbar.update(1)
    else:
        logging.warning("Model does not have a 'train' method. Skipping gradient steps on demonstrations.")

    return model

def pretrain_on_policy(model, demonstrations, env):
    """
    Pretrain on-policy algorithms (PPO, RecurrentPPO) using behavior cloning.
    """
    if not demonstrations or all(len(env_demos) == 0 for env_demos in demonstrations):
        logging.info("No demonstration transitions to pretrain on. Skipping behavior cloning.")
        return model
    
    # Flatten all demonstrations from all environments
    all_transitions = []
    for env_demos in demonstrations:
        all_transitions.extend(env_demos)
    
    logging.info(f"Starting behavior cloning with {len(all_transitions)} transitions")
    
    # Convert our transitions to the format expected by imitation library
    observations = []
    actions = []
    next_observations = []
    dones = []
    
    for obs, action, reward, next_obs, done in all_transitions:
        observations.append(obs)
        actions.append(action)
        next_observations.append(next_obs)
        dones.append(done)
    
    observations = np.array(observations)
    actions = np.array(actions)
    next_observations = np.array(next_observations)
    dones = np.array(dones)
    
    # Create Transitions object for imitation library with complete data
    transitions = data_types.Transitions(
        obs=observations,
        acts=actions,
        next_obs=next_observations,
        dones=dones,
        infos=[{} for _ in range(len(observations))]
    )
    
    # Check if this is a recurrent policy and handle accordingly
    policy_for_bc = model.policy
    
    # For recurrent policies, we need to create a non-recurrent wrapper
    if hasattr(model.policy, 'lstm_actor'):
        logging.info("Detected recurrent policy. Creating non-recurrent wrapper for behavior cloning.")
        
        # Create a simple wrapper that provides LSTM states when needed
        class NonRecurrentWrapper(torch.nn.Module):
            def __init__(self, original_policy):
                super().__init__()
                # Store as a module so it's properly registered
                self.original_policy = original_policy
                self.observation_space = original_policy.observation_space
                self.action_space = original_policy.action_space
                
            def evaluate_actions(self, obs, actions):
                # For BC training, provide dummy LSTM states
                batch_size = obs.shape[0]
                device = obs.device if hasattr(obs, 'device') else 'cpu'
                episode_starts = torch.ones((batch_size,), dtype=torch.float32, device=device)
                
                # Create dummy LSTM states for recurrent policy
                from sb3_contrib.common.recurrent.type_aliases import RNNStates
                hidden_state = torch.zeros((self.original_policy.lstm_actor.num_layers, batch_size, self.original_policy.lstm_actor.hidden_size), device=device)
                cell_state = torch.zeros((self.original_policy.lstm_actor.num_layers, batch_size, self.original_policy.lstm_actor.hidden_size), device=device)
                lstm_states = RNNStates((hidden_state, cell_state), (hidden_state, cell_state))
                
                return self.original_policy.evaluate_actions(obs, actions, lstm_states, episode_starts)
                
            def forward(self, obs, deterministic=False):
                # For BC, we just need the action distribution
                batch_size = obs.shape[0]
                device = obs.device if hasattr(obs, 'device') else 'cpu'
                episode_starts = torch.ones((batch_size,), dtype=torch.float32, device=device)
                
                # Create dummy LSTM states for recurrent policy
                from sb3_contrib.common.recurrent.type_aliases import RNNStates
                hidden_state = torch.zeros((self.original_policy.lstm_actor.num_layers, batch_size, self.original_policy.lstm_actor.hidden_size), device=device)
                cell_state = torch.zeros((self.original_policy.lstm_actor.num_layers, batch_size, self.original_policy.lstm_actor.hidden_size), device=device)
                lstm_states = RNNStates((hidden_state, cell_state), (hidden_state, cell_state))
                
                return self.original_policy.forward(obs, lstm_states, episode_starts, deterministic)
                
            def predict(self, observation, state=None, episode_start=None, deterministic=False):
                return self.original_policy.predict(observation, state, episode_start, deterministic)
                
            @property
            def device(self):
                # Return the device of the original policy
                try:
                    return next(self.original_policy.parameters()).device
                except StopIteration:
                    return torch.device('cpu')
                
            def to(self, device):
                # Move the original policy to the device more carefully
                # Use the parent class to() method to move this module
                super().to(device)
                # Move the original policy
                self.original_policy = self.original_policy.to(device)
                return self
                
            def parameters(self):
                # Delegate parameters to the original policy
                return self.original_policy.parameters()
                
            def state_dict(self):
                # Delegate state_dict to the original policy
                return self.original_policy.state_dict()
                
            def load_state_dict(self, state_dict):
                # Delegate load_state_dict to the original policy
                return self.original_policy.load_state_dict(state_dict)
                
            def set_training_mode(self, mode: bool):
                # Delegate training mode to the original policy
                return self.original_policy.set_training_mode(mode)
        
        policy_for_bc = NonRecurrentWrapper(model.policy)
    
    # Create behavior cloning trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy_for_bc,  # Use the wrapped policy for recurrent models
        rng=np.random.default_rng(42),
        batch_size=min(512, len(all_transitions) // 4),  # Adaptive batch size
        optimizer_kwargs={"lr": 3e-4},
        ent_weight=0.01,
        l2_weight=1e-5,
    )
    
    # Train using behavior cloning
    n_epochs = max(1, 1000_000 // len(all_transitions))  # More epochs for smaller datasets
    logging.info(f"Training behavior cloning for {n_epochs} epochs")
    
    bc_trainer.train(
        n_epochs=n_epochs,
        log_interval=max(1, n_epochs // 10),
        progress_bar=True
    )
    
    # Update the model's policy with the trained policy
    logging.info(f"Model policy type: {type(model.policy)}")
    logging.info(f"BC trainer policy type: {type(bc_trainer.policy)}")
    logging.info(f"Model policy has lstm_actor: {hasattr(model.policy, 'lstm_actor')}")
    
    if hasattr(model.policy, 'lstm_actor'):
        # For recurrent policies, we need to update the underlying parameters
        logging.info("Updating recurrent policy parameters from behavior cloning.")
        # Copy the trained parameters back to the original policy
        model.policy.load_state_dict(bc_trainer.policy.original_policy.state_dict())
        logging.info("Successfully updated original policy parameters.")
    else:
        logging.info("Replacing model policy with BC trained policy.")
        model.policy = bc_trainer.policy
    
    logging.info("Behavior cloning completed")
    return model

def pretrain_with_demonstrations(model, demonstrations, algorithm, env, num_envs):
    """
    Pretrain the model using demonstrations, choosing the appropriate method based on model type.
    
    Args:
        model: The RL model to train
        demonstrations: List of lists - demonstrations[env_idx] = [(obs, action, reward, next_obs, done), ...]
        algorithm: The algorithm type (for logging purposes)
        env: The environment
        num_envs: Number of environments used to collect demonstrations
        
    Returns:
        Trained model
    """
    if not demonstrations or all(len(env_demos) == 0 for env_demos in demonstrations):
        logging.info("No demonstrations available for pretraining.")
        return model
    
    # Determine if this is an off-policy or on-policy algorithm by checking model attributes
    # Off-policy algorithms typically have a replay buffer
    if hasattr(model, 'replay_buffer'):
        logging.info(f"Detected off-policy model ({algorithm}) - using replay buffer pretraining")
        return pretrain_off_policy(model, demonstrations, num_envs)
    else:
        logging.info(f"Detected on-policy model ({algorithm}) - using behavior cloning")
        return pretrain_on_policy(model, demonstrations, env)

def make_env(env_id, rank, seed=0, env_kwargs=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID (unused here, but common pattern)
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param env_kwargs: (dict) arguments to pass to the env constructor
    """
    def _init():
        nonlocal env_kwargs
        if env_kwargs is None:
            env_kwargs = {}
        # Ensure each subprocess gets a unique seed
        current_seed = seed + rank 
        env = F110GymWrapper(**env_kwargs, seed=current_seed)
        # Optional: Wrap with Monitor for logging episode stats
        env = Monitor(env)
        return env
    # set_global_seeds(seed) # Deprecated in SB3
    return _init

def generate_param_combinations(seed: int, num_param_cmbs: int) -> List[dict]:
    """
    Generate domain randomization parameter combinations deterministically.
    Returns a list of parameter dicts of length num_param_cmbs.
    """
    param_combinations: List[dict] = []
    for i in range(num_param_cmbs):
        param_seed = seed + i
        rng = np.random.default_rng(param_seed)
        param_set = {
            'mu': float(rng.uniform(0.8, 1.1)),
            'C_Sf': float(rng.uniform(4.0, 5.5)),
            'C_Sr': float(rng.uniform(4.0, 5.5)),
            'm': float(rng.uniform(3.0, 4.5)),
            'I': float(rng.uniform(0.03, 0.06)),
            'lidar_noise_stddev': float(rng.uniform(0.0, 0.1)),
            's_noise_stddev': float(rng.uniform(0.0, 1)),
            'ey_noise_stddev': float(rng.uniform(0.0, 0.5)),
            'vel_noise_stddev': float(rng.uniform(0.0, 0.5)),
            'yaw_noise_stddev': float(rng.uniform(0, 0.5)),
        }
        param_set['push_0_prob'] = float(rng.uniform(0.0, 0.1))
        param_set['push_2_prob'] = float(rng.uniform(0.0, 0.05))

        logging.info(
            f"Param Set {i} Parameters: mu: {param_set['mu']}, C_Sf: {param_set['C_Sf']}, C_Sr: {param_set['C_Sr']}, m: {param_set['m']}, I: {param_set['I']}; "
            f"Observation noise: lidar_noise_stddev: {param_set['lidar_noise_stddev']}, s: {param_set['s_noise_stddev']}, ey: {param_set['ey_noise_stddev']}, vel: {param_set['vel_noise_stddev']}, yaw: {param_set['yaw_noise_stddev']}; "
            f"Push probabilities: push_0_prob: {param_set['push_0_prob']}, push_2_prob: {param_set['push_2_prob']}"
        )
        param_combinations.append(param_set)

    return param_combinations

def expand_env_kwargs_for_envs(
    base_env_kwargs: dict,
    param_combinations: List[dict],
    num_envs: int,
    include_params_in_obs: bool,
    lidar_scan_in_obs_mode: str,
    racing_mode: bool
) -> List[dict]:
    """
    Create per-env kwargs by combining base kwargs with DR parameter combinations and flags.
    If param_combinations is empty, only flags are applied.
    """
    per_env_kwargs: List[dict] = []
    for i in range(num_envs):
        current_env_kwargs = base_env_kwargs.copy()
        current_env_kwargs['include_params_in_obs'] = include_params_in_obs
        current_env_kwargs['lidar_scan_in_obs_mode'] = lidar_scan_in_obs_mode
        if 'racing_mode' not in current_env_kwargs:
            current_env_kwargs['racing_mode'] = racing_mode

        if param_combinations:
            param_idx = i % len(param_combinations)
            for key, value in param_combinations[param_idx].items():
                current_env_kwargs[key] = value
            logging.info(f"Env {i} using parameter set {param_idx}")

        per_env_kwargs.append(current_env_kwargs)

    return per_env_kwargs

def create_vec_env(env_kwargs, seed):
    # Read parameters from FLAGS
    num_envs = FLAGS.num_envs
    num_param_cmbs = FLAGS.num_param_cmbs
    use_domain_randomization = FLAGS.use_dr
    include_params_in_obs = FLAGS.include_params_in_obs
    racing_mode = FLAGS.racing_mode
    lidar_scan_in_obs_mode = FLAGS.lidar_scan_in_obs_mode
    
    # If num_param_cmbs is not specified but DR is used, default to num_envs
    if num_param_cmbs is None and use_domain_randomization:
        num_param_cmbs = num_envs
    # If using DR, ensure num_param_cmbs is at least 1
    elif use_domain_randomization and num_param_cmbs < 1:
        num_param_cmbs = 1
        logging.warning(f"num_param_cmbs must be at least 1 for domain randomization. Setting to 1.")

    # Generate parameter combinations first if using domain randomization
    param_combinations: List[dict] = []
    if use_domain_randomization and num_param_cmbs > 0:
        param_combinations = generate_param_combinations(seed=seed, num_param_cmbs=num_param_cmbs)
    
    # --- Create Environment(s) ---
    # Create per-env kwargs list
    per_env_kwargs = expand_env_kwargs_for_envs(
        base_env_kwargs=env_kwargs,
        param_combinations=param_combinations,
        num_envs=num_envs,
        include_params_in_obs=include_params_in_obs,
        lidar_scan_in_obs_mode=lidar_scan_in_obs_mode,
        racing_mode=racing_mode
    )

    # Create the vectorized environment thunks
    env_fns = []
    for i, kwargs_i in enumerate(per_env_kwargs):
        env_fn = partial(make_env(env_id=f"f110-rank{i}", rank=i, seed=seed, env_kwargs=kwargs_i))
        env_fns.append(env_fn)
        
    # Create the vectorized environment
    vec_env_cls = DummyVecEnv if num_envs == 1 else SubprocVecEnv
    vec_env = vec_env_cls(env_fns)
    
    return vec_env

def setup_vecnormalize_env_train(vec_env) -> VecNormalize | Any:
    """
    Wrap the environment with VecNormalize for training when using an RL algorithm.

    Returns the possibly wrapped environment with training flags enabled.
    """
    algorithm = FLAGS.algorithm
    if not is_rl_policy(algorithm):
        return vec_env

    # Always create a training VecNormalize wrapper (do not pre-check existing wrappers)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    )
    vec_env.training = True
    vec_env.norm_reward = True
    logging.info("VecNormalize wrapper initialized for training")
    return vec_env

def setup_vecnormalize_env_eval(vec_env, model_path: Optional[str], vecnorm_path: Optional[str]) -> VecNormalize | Any:
    """
    Load VecNormalize statistics for evaluation when using an RL algorithm.

    If statistics are found, returns a VecNormalize wrapper with training disabled.
    Otherwise, returns the environment unchanged.
    """
    algorithm = FLAGS.algorithm
    if not is_rl_policy(algorithm):
        return vec_env

    # Resolve stats path
    resolved_path = vecnorm_path
    if (resolved_path is None or not os.path.exists(resolved_path)) and model_path is not None:
        model_dir = os.path.dirname(model_path)
        potential_vecnorm_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_vecnorm_path):
            resolved_path = potential_vecnorm_path

    if resolved_path is not None and os.path.exists(resolved_path):
        vec_env = VecNormalize.load(resolved_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        logging.info(f"Loaded VecNormalize statistics from {resolved_path} for evaluation")
    else:
        logging.warning("No VecNormalize statistics found for evaluation; proceeding without normalization")
    return vec_env

# Updated train function to handle VecEnv and Domain Randomization
def train(env, seed):
    """
    Trains the RL model.

    Args:
        env: Vectorized environment for training.
        seed (int): Random seed.
        
    Note:
        All other parameters are read from FLAGS singleton.
    """
    # Read parameters from FLAGS
    num_envs = FLAGS.num_envs
    num_param_cmbs = FLAGS.num_param_cmbs
    use_domain_randomization = FLAGS.use_dr
    use_imitation_learning = FLAGS.use_il
    imitation_policy_type = FLAGS.il_policy
    algorithm = FLAGS.algorithm
    include_params_in_obs = FLAGS.include_params_in_obs
    racing_mode = FLAGS.racing_mode
    feature_extractor_name = FLAGS.feature_extractor
    lidar_scan_in_obs_mode = FLAGS.lidar_scan_in_obs_mode
    
    # --- Create Model ---
    logging.info(f"Creating {algorithm} model with {feature_extractor_name} feature extractor")
    if algorithm == ALGO_PPO:
        model = create_ppo(env, seed)
    elif algorithm == ALGO_RECURRENT_PPO:
        model = create_recurrent_ppo(env, seed)
    elif algorithm == ALGO_DDPG:
        model = create_ddpg(env, seed)
    elif algorithm == ALGO_TD3:
        model = create_td3(env, seed)
    elif algorithm == ALGO_SAC:
        model = create_sac(env, seed)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # --- Imitation Learning (Optional, might need adaptation for VecEnv) ---
    if use_imitation_learning:
        logging.info("Using imitation learning to bootstrap the model.")
        model = initialize_with_imitation_learning(model, env, imitation_policy_type=imitation_policy_type, racing_mode=racing_mode, algorithm=algorithm)
    else:
        logging.info("Skipping imitation learning.")

    logging.info(f"Starting RL training with {env.num_envs} environments.")

    # --- RL Training ---
    # Create formatted path based on training parameters and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = f"{algorithm}_{feature_extractor_name}_envs{num_envs}_params{num_param_cmbs if num_param_cmbs is not None else num_envs}_dr{int(use_domain_randomization)}_il{int(use_imitation_learning)}_crl{int(include_params_in_obs)}_racing{int(racing_mode)}"
    if use_imitation_learning:
        model_dir_name += f"_{imitation_policy_type}"
    model_dir_name += f"_seed{seed}_{timestamp}"
    
    # Create the full paths
    best_model_path = os.path.join("./logs", model_dir_name, "best_model")
    os.makedirs(best_model_path, exist_ok=True)

    # Import callbacks for saving best model based on training rewards
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    # Custom callback to save model with highest training reward
    class SaveOnBestTrainingRewardCallback(BaseCallback):
        def __init__(self, check_freq, save_path, verbose=1):
            super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.model_save_path = save_path
            self.best_mean_reward = -float('inf')
        
        def _init_callback(self):
            # Create folder if needed
            if self.model_save_path is not None:
                os.makedirs(self.model_save_path, exist_ok=True)
        
        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                # Get the mean episode reward from recent episodes
                if len(self.model.ep_info_buffer) > 0:
                    mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                    
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Mean training reward: {mean_reward:.2f}")
                    
                    # New best model, save it
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose > 0:
                            print(f"Saving new best model with mean reward: {mean_reward:.2f}")
                        self.model.save(os.path.join(self.model_save_path, "best_model"))
                        
                        # Save normalization statistics if the environment is wrapped with VecNormalize
                        if isinstance(self.training_env, VecNormalize) or hasattr(self.training_env, 'venv') and isinstance(self.training_env.venv, VecNormalize):
                            # Get the VecNormalize wrapper
                            vec_normalize = self.training_env if isinstance(self.training_env, VecNormalize) else self.training_env.venv
                            # Save the normalization statistics
                            vec_normalize.save(os.path.join(self.model_save_path, "vec_normalize.pkl"))
                            if self.verbose > 0:
                                print(f"Saved VecNormalize statistics to {os.path.join(self.model_save_path, 'vec_normalize.pkl')}")
            
            return True

    # Import VecNormalize for type checking
    from stable_baselines3.common.vec_env import VecNormalize

    # Create the callback
    # Check frequency should be frequent enough to capture improvements but not too frequent
    save_callback = SaveOnBestTrainingRewardCallback(
        check_freq=max(1000 // num_envs, 1),
        save_path=best_model_path,
        verbose=1
    )

    model.learn(
        total_timesteps=10_000_000,
        log_interval=10, # Log less frequently for VecEnv
        reset_num_timesteps=True, # Start timesteps from 0
        callback=save_callback
    )
    
    # Save final model and VecNormalize statistics
    final_model_path = os.path.join("./logs", model_dir_name, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save(os.path.join(final_model_path, "final_model"))
    
    # Save VecNormalize statistics if the environment is wrapped
    if isinstance(env, VecNormalize):
        env.save(os.path.join(final_model_path, "vec_normalize.pkl"))
        logging.info(f"Saved final VecNormalize statistics to {os.path.join(final_model_path, 'vec_normalize.pkl')}")
    
    logging.info(f"Training completed. Final model saved to {os.path.join(final_model_path, 'final_model.zip')}")
    logging.info(f"Best model saved to {os.path.join(best_model_path, 'best_model.zip')}")
    
    return model