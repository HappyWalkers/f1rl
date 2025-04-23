from stable_baselines3 import PPO, DDPG, DQN, TD3, SAC
from absl import logging
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from functools import partial
import os
import time
import numpy as np
import datetime
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Type, Union, Callable, Optional, Any
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm

from rl_env import F110GymWrapper # Import the wrapper


class F1TenthFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the F1Tenth environment.
    
    This network separately processes state features, lidar scans, and environment parameters,
    then combines them into a single feature vector.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1024,
        state_dim: int = 4,
        lidar_dim: int = 1080,
        param_dim: int = 15, 
        include_params: bool = True
    ):
        # Determine whether the observation includes parameters based on its dimension
        self.state_dim = state_dim  # 4 state components: [s, ey, vel, yaw_angle]
        self.lidar_dim = lidar_dim  # LiDAR points
        self.param_dim = param_dim  # Environment parameters
        self.include_params = include_params
        
        # The expected observation dimension with parameters included
        expected_dim_with_params = state_dim + lidar_dim + param_dim
        expected_dim_without_params = state_dim + lidar_dim
        
        # Check if observation space has the expected shape
        obs_shape = observation_space.shape[0]
        if include_params:
            assert obs_shape == expected_dim_with_params, f"Expected observation dimension {expected_dim_with_params}, got {obs_shape}"
        else:
            assert obs_shape == expected_dim_without_params, f"Expected observation dimension {expected_dim_without_params}, got {obs_shape}"
            
        super().__init__(observation_space, features_dim)
        
        # Network for processing state variables (s, ey, vel, yaw_angle)
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Network for processing LiDAR scan
        # # Use dilated convolutions to capture patterns at multiple scales
        # self.lidar_net = nn.Sequential(
        #     # Reshape lidar to [batch, 1, lidar_dim]
        #     Reshape((-1, 1, lidar_dim)),
            
        #     nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, dilation=1, padding=1),
        #     nn.ReLU(),
            
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, dilation=2, padding=2),
        #     nn.ReLU(),
            
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=4, padding=4),
        #     nn.ReLU(),
            
        #     nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, dilation=8, padding=8),
        #     nn.ReLU(),
            
        #     nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, dilation=16, padding=16),
        #     nn.ReLU(),
            
        #     # Dimensionality reduction
        #     nn.Conv1d(in_channels=16, out_channels=8, kernel_size=9, stride=8, padding=4),
        #     nn.ReLU(),
            
        #     # Flatten the output
        #     nn.Flatten(),
            
        #     # Fully connected layers
        #     nn.Linear(8 * 135, 128),  # 1080/8 = 135
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU()
        # )
        # Pure MLP architecture for processing LiDAR data
        self.lidar_net = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(lidar_dim, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Network for processing environment parameters (if included)
        if include_params:
            self.param_net = nn.Sequential(
                nn.Linear(param_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            # Combined dimension from all branches
            combined_dim = 128 + 128 + 128  # state + lidar + param
        else:
            self.param_net = None
            combined_dim = 128 + 128  # state + lidar
        
        # Final layers to combine all features
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract different components from observation
        state_features = observations[:, :self.state_dim]
        lidar_features = observations[:, self.state_dim:self.state_dim + self.lidar_dim]
        
        # Process state features
        state_output = self.state_net(state_features)
        
        # Process lidar features
        lidar_output = self.lidar_net(lidar_features)
        
        if self.include_params:
            # Extract and process environment parameters
            param_features = observations[:, self.state_dim + self.lidar_dim:]
            param_output = self.param_net(param_features)
            
            # Combine all features
            combined_features = torch.cat([state_output, lidar_output, param_output], dim=1)
        else:
            # Combine only state and lidar features
            combined_features = torch.cat([state_output, lidar_output], dim=1)
        
        # Final processing
        output = self.combined_net(combined_features)
        return output


class Reshape(nn.Module):
    """
    Helper module to reshape tensors.
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


def create_ppo(env, seed, include_params_in_obs=True):
    """Create a PPO model with custom neural network architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
    state_dim = 4
    param_dim = 15
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": F1TenthFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs
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

def create_ddpg(env, seed, include_params_in_obs=True):
    """Create a DDPG model with custom neural network architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
    state_dim = 4
    param_dim = 15
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": F1TenthFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs
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

def create_td3(env, seed, include_params_in_obs=True):
    """Create a TD3 model with custom neural network architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
    state_dim = 4
    param_dim = 15
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": F1TenthFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs
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

def create_sac(env, seed, include_params_in_obs=True):
    """Create a SAC model with custom neural network architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
    state_dim = 4
    param_dim = 15
    
    # Define policy kwargs with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": F1TenthFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 1024,
            "state_dim": state_dim,
            "lidar_dim": lidar_dim,
            "param_dim": param_dim,
            "include_params": include_params_in_obs
        },
        # Using a moderately sized network for actor and critic
        "net_arch": [1024, 512, 256, 128, 64]
    }
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=256,
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

def evaluate(eval_env, model_path="./logs/best_model/best_model.zip", algorithm="SAC", num_episodes=5, model=None, racing_mode=False):
    """
    Evaluates a trained model or wall-following policy on the environment.
    
    Args:
        eval_env: The environment (single instance or VecEnv) to evaluate in
        model_path: Path to the saved model (ignored when algorithm is WALL_FOLLOW, PURE_PURSUIT, LATTICE, or if model is provided)
        algorithm: Algorithm type (SAC, PPO, DDPG, TD3, WALL_FOLLOW, PURE_PURSUIT, LATTICE)
        num_episodes: Number of episodes to evaluate
        model: Optional pre-loaded model object (takes precedence over model_path)
        racing_mode: Whether to evaluate in racing mode with two cars
    """
    # Set racing mode in environment if needed and not already set
    if racing_mode:
        logging.info("Evaluating in racing mode with two cars")
        eval_env.racing_mode = racing_mode
            
    # Check if eval_env is a VecEnv
    is_vec_env = isinstance(eval_env, (DummyVecEnv, SubprocVecEnv))
    num_envs = eval_env.num_envs if is_vec_env else 1
    
    if algorithm == "WALL_FOLLOW":
        from wall_follow import WallFollowPolicy
        logging.info("Using wall-following policy for evaluation")
        model = WallFollowPolicy()
    elif algorithm == "PURE_PURSUIT":
        from pure_pursuit import PurePursuitPolicy
        logging.info("Using pure pursuit policy for evaluation")
        # Get track from the environment if available
        if is_vec_env:
            track = eval_env.get_attr("track", indices=0)[0]  # Get track from first env
        else:
            track = getattr(eval_env, 'track', None)  # Access unwrapped env for track
        model = PurePursuitPolicy(track=track)
    elif algorithm == "LATTICE":
        from lattice_planner import LatticePlannerPolicy
        logging.info("Using lattice planner policy for evaluation")
        # Get track from the environment if available
        if is_vec_env:
            track = eval_env.get_attr("track", indices=0)[0]  # Get track from first env
        else:
            track = getattr(eval_env, 'track', None)  # Access unwrapped env for track
        model = LatticePlannerPolicy(track=track)
    elif model is None:
        # Only load model from path if not provided directly
        logging.info(f"Loading {algorithm} model from {model_path}")
        
        # Load the appropriate model based on algorithm type
        if algorithm == "PPO":
            model = PPO.load(model_path)
        elif algorithm == "DDPG":
            model = DDPG.load(model_path)
        elif algorithm == "TD3":
            model = TD3.load(model_path)
        elif algorithm == "SAC":
            model = SAC.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
        logging.info("Model loaded successfully")
    else:
        logging.info("Using provided model for evaluation")
    
    # Initialize metrics
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    
    # Run evaluation episodes for each environment
    for env_idx in range(num_envs):
        logging.info(f"Evaluating on environment {env_idx+1}/{num_envs}")
        
        for episode in range(num_episodes):
            logging.info(f"Starting evaluation episode {episode+1}/{num_episodes} on env {env_idx+1}")
            
            # Reset the environment
            if is_vec_env:
                obs = eval_env.env_method('reset', indices=[env_idx])[0][0]  # Reset specific env
                # Wrap single obs in list to match expected format
                obs = np.array([obs])
            else:
                obs = eval_env.reset()
            
            # Reset the policy if needed
            if hasattr(model, 'reset'):
                model.reset()
            
            terminated = False
            truncated = False
            total_reward = 0
            step_count = 0
            episode_start_time = time.time()
            
            while not (terminated or truncated):
                # Render environment - only render the current environment
                if is_vec_env:
                    eval_env.env_method('render', indices=[env_idx])
                else:
                    eval_env.render()
                
                # Get action from model
                # Extract the observation for a single environment from the batch
                single_obs = obs[0] if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs
                action, _states = model.predict(single_obs, deterministic=True)
                
                # Take step in environment
                if is_vec_env:
                    # Step only the current environment
                    obs, reward, terminated, truncated, info = eval_env.env_method(
                        'step', action, indices=[env_idx]
                    )[0]
                    # Wrap outputs to match expected format
                    obs = np.array([obs])
                    terminated = bool(terminated)
                    truncated = bool(truncated)
                    reward = float(reward)
                else:
                    # Gymnasium envs return 5 values
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                
                total_reward += reward
                step_count += 1
            
            # Episode completed
            episode_time = time.time() - episode_start_time
            
            # Record metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            lap_times.append(episode_time)
            
            logging.info(f"Episode {episode+1} finished:")
            logging.info(f"  Reward: {total_reward:.2f}")
            logging.info(f"  Length: {step_count} steps")
            logging.info(f"  Time: {episode_time:.2f} seconds")
            
            # Reset for next episode
            print(f"Episode {episode+1} finished. Resetting...")
    
    # Compute summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_episode_length = np.mean(episode_lengths)
    mean_lap_time = np.mean(lap_times)
    
    logging.info(f"Evaluation completed over {num_episodes * num_envs} episodes across {num_envs} environments:")
    logging.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logging.info(f"  Mean episode length: {mean_episode_length:.2f} steps")
    logging.info(f"  Mean lap time: {mean_lap_time:.2f} seconds")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_episode_length,
        "mean_lap_time": mean_lap_time,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "lap_times": lap_times
    }

def initialize_with_imitation_learning(model, env, imitation_policy_type="PURE_PURSUIT", total_transitions=1000_000, racing_mode=False):
    """
    Initialize a reinforcement learning model using imitation learning from a specified policy.
    
    Args:
        model: The RL model to initialize
        env: The environment (must be a VecEnv) to collect demonstrations from.
        imitation_policy_type: Type of imitation policy (WALL_FOLLOW, PURE_PURSUIT, or LATTICE)
        total_transitions: Total number of transitions to collect across all environments
        racing_mode: Whether to use racing mode with two cars for overtaking scenarios
    
    Returns:
        model: The initialized model
        
    Raises:
        TypeError: If env is not a VecEnv instance
    """
    # Check if env is a VecEnv and raise an error if it's not
    if not isinstance(env, (DummyVecEnv,SubprocVecEnv)):
        raise TypeError("env must be a VecEnv instance")

    # Import policies for imitation learning
    from wall_follow import WallFollowPolicy
    from pure_pursuit import PurePursuitPolicy
    from lattice_planner import LatticePlannerPolicy

    # Initialize the expert policies for each environment
    logging.info(f"Starting imitation learning from {imitation_policy_type} policy (racing_mode={racing_mode})")
    expert_policies = []
    for i in range(env.num_envs):
        # Extract the track for this environment
        track = env.get_attr("track", indices=i)[0]
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
            expert_policies.append(LatticePlannerPolicy(track=track))
        else:
            raise ValueError(f"Unsupported imitation_policy_type: {imitation_policy_type}")

    # Collect demonstrations from the expert policies
    demonstrations = []
    collected_transitions = 0
    
    # Reset all environments
    observations = env.reset()
    # Reset all expert policies
    for policy in expert_policies:
        if hasattr(policy, 'reset'):
            policy.reset()
    
    dones = [False] * env.num_envs
    
    logging.info(f"Collecting {total_transitions} transitions across {env.num_envs} environments")
    
    while collected_transitions < total_transitions:
        # Generate expert actions for each environment
        actions = []
        for i in range(env.num_envs):
            # Get action from the appropriate expert policy
            action, _ = expert_policies[i].predict(observations[i], deterministic=True)
            actions.append(action)
        
        # Convert to numpy array for VecEnv
        actions = np.array(actions)
        
        # Step all environments together
        next_observations, rewards, dones, infos = env.step(actions)
        
        # Store transitions 
        demonstrations.append((observations, actions, rewards, next_observations, dones))
        collected_transitions += env.num_envs
        
        # reset the finished environments
        for i in range(env.num_envs):
            if dones[i]:
                next_observations[i] = env.env_method('reset', indices=i)[0][0]
                if hasattr(expert_policies[i], 'reset'):
                    expert_policies[i].reset()
        observations = next_observations
        
        # Print progress every progress_interval transitions
        if collected_transitions % (total_transitions // 10) < env.num_envs:
            logging.info(f"Progress: {collected_transitions}/{total_transitions} transitions collected")
            
    logging.info(f"Collected {collected_transitions} demonstration transitions")

    # Pretrain the model using the demonstrations
    logging.info("Pretraining model with demonstrations")

    # For models with replay buffers, add the demonstrations
    if hasattr(model, 'replay_buffer'):
        for obs, action, reward, next_obs, done in demonstrations:
            model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])

        # Perform gradient steps on the demonstrations
        if hasattr(model, 'train'):
            logging.info("Training on demonstrations")
            # Initialize the model by calling learn with 1 step before training
            # This ensures the logger and other components are properly set up
            model.learn(total_timesteps=1, log_interval=1)
            # Now we can safely call train
            gradient_steps = 10_000
            # Create a progress bar for training
            with tqdm(total=gradient_steps, desc="Imitation Learning Progress") as pbar:
                for i in range(0, gradient_steps, 1):
                    # Perform training step
                    model.train(gradient_steps=1, batch_size=model.batch_size)
                    # Update progress bar
                    pbar.update(1)

    logging.info("Imitation learning completed")
    return model

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

def create_vec_env(env_kwargs, seed, num_envs=1, use_domain_randomization=False, include_params_in_obs=True, racing_mode=False):
    """
    Creates vectorized environments for training and evaluation.

    Args:
        env_kwargs (dict): Base arguments for the F110GymWrapper environment.
        seed (int): Random seed.
        num_envs (int): Number of parallel environments to use.
        use_domain_randomization (bool): Whether to randomize environment parameters.
        include_params_in_obs (bool): Whether to include environment parameters in observations.
        racing_mode (bool): Whether to use racing mode with two cars.

    Returns:
        VecEnv: The vectorized environment.
    """
    # --- Create Environment(s) ---
    env_fns = []
    for i in range(num_envs):
        rank_seed = seed + i
        current_env_kwargs = env_kwargs.copy()
        current_env_kwargs['include_params_in_obs'] = include_params_in_obs
        
        # Ensure racing_mode is passed to environment
        if 'racing_mode' not in current_env_kwargs:
            current_env_kwargs['racing_mode'] = racing_mode
        
        if use_domain_randomization:
            # Sample parameters randomly for this environment instance
            rng = np.random.default_rng(rank_seed)
            current_env_kwargs['mu'] = rng.uniform(0.6, 1.1)
            current_env_kwargs['C_Sf'] = rng.uniform(4.0, 5.5)
            current_env_kwargs['C_Sr'] = rng.uniform(4.0, 5.5)
            current_env_kwargs['m'] = rng.uniform(3.0, 4.5)
            current_env_kwargs['I'] = rng.uniform(0.03, 0.06)
            current_env_kwargs['lidar_noise_stddev'] = rng.uniform(0.0, 0.01)
            current_env_kwargs['s_noise_stddev'] = rng.uniform(0.0, 0.01)
            current_env_kwargs['ey_noise_stddev'] = rng.uniform(0.0, 0.01)
            current_env_kwargs['vel_noise_stddev'] = rng.uniform(0.0, 0.01)
            current_env_kwargs['yaw_noise_stddev'] = rng.uniform(0.0, 0.01)
            sampled_push_0_prob = rng.uniform(0.0, 0.01)
            current_env_kwargs['push_0_prob'] = sampled_push_0_prob
            current_env_kwargs['push_2_prob'] = sampled_push_0_prob
            
            logging.info(
                f"Env {i} Parameters: mu: {current_env_kwargs['mu']}, C_Sf: {current_env_kwargs['C_Sf']}, C_Sr: {current_env_kwargs['C_Sr']}, m: {current_env_kwargs['m']}, I: {current_env_kwargs['I']}; "
                f"Observation noise: lidar_noise_stddev: {current_env_kwargs['lidar_noise_stddev']}, s: {current_env_kwargs['s_noise_stddev']}, ey: {current_env_kwargs['ey_noise_stddev']}, vel: {current_env_kwargs['vel_noise_stddev']}, yaw: {current_env_kwargs['yaw_noise_stddev']}; "
                f"Push probabilities: push_0_prob: {current_env_kwargs['push_0_prob']}, push_2_prob: {current_env_kwargs['push_2_prob']}"
            )
        
        # Create the thunk (function) for this env instance
        # Use partial to pass the potentially modified kwargs
        env_fn = partial(make_env(env_id=f"f110-rank{i}", rank=i, seed=seed, env_kwargs=current_env_kwargs))
        env_fns.append(env_fn)
        
    # Create the vectorized environment
    vec_env_cls = DummyVecEnv if num_envs == 1 else SubprocVecEnv
    env = vec_env_cls(env_fns)
    
    return env

# Updated train function to handle VecEnv and Domain Randomization
def train(env, seed, num_envs=1, use_domain_randomization=False, use_imitation_learning=True, imitation_policy_type="PURE_PURSUIT", algorithm="SAC", include_params_in_obs=True, racing_mode=False):
    """
    Trains the RL model.

    Args:
        env: Vectorized environment for training.
        seed (int): Random seed.
        num_envs (int): Number of parallel environments to use.
        use_domain_randomization (bool): Whether to randomize environment parameters.
        use_imitation_learning (bool): Whether to use imitation learning before RL training.
        imitation_policy_type (str): Policy for imitation learning.
        algorithm (str): RL algorithm to use (e.g., SAC, PPO).
        include_params_in_obs (bool): Whether to include environment parameters in observations.
        racing_mode (bool): Whether to train in racing mode with two cars.
    """
    # --- Create Model ---
    if algorithm == "PPO":
        model = create_ppo(env, seed, include_params_in_obs)
    elif algorithm == "DDPG":
        model = create_ddpg(env, seed, include_params_in_obs)
    elif algorithm == "TD3":
        model = create_td3(env, seed, include_params_in_obs)
    elif algorithm == "SAC":
        model = create_sac(env, seed, include_params_in_obs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # --- Imitation Learning (Optional, might need adaptation for VecEnv) ---
    if use_imitation_learning:
        logging.info("Using imitation learning to bootstrap the model.")
        model = initialize_with_imitation_learning(model, env, imitation_policy_type=imitation_policy_type, racing_mode=racing_mode)
    else:
        logging.info("Skipping imitation learning.")

    logging.info(f"Starting RL training with {env.num_envs} environments.")

    # --- RL Training ---
    # Create formatted path based on training parameters and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = f"{algorithm}_envs{num_envs}_dr{int(use_domain_randomization)}_il{int(use_imitation_learning)}_crl{int(include_params_in_obs)}_racing{int(racing_mode)}"
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
            self.model_save_path = os.path.join(save_path, "best_model")
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
                        self.model.save(self.model_save_path)
            
            return True

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