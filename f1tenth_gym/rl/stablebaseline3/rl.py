from stable_baselines3 import PPO, DDPG, DQN, TD3, SAC
from sb3_contrib import RecurrentPPO
from absl import logging
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
from rl_env import F110GymWrapper # Import the wrapper
from stablebaseline3.feature_extractor import F1TenthFeaturesExtractor, MLPFeaturesExtractor, ResNetFeaturesExtractor, TransformerFeaturesExtractor, MoEFeaturesExtractor
from sortedcontainers import SortedList

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

def create_ppo(env, seed, include_params_in_obs=True, feature_extractor_name="FILM"):
    """Create a PPO model with custom neural network architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
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

def create_recurrent_ppo(env, seed, include_params_in_obs=False, feature_extractor_name="FILM"):
    """Create a RecurrentPPO model with LSTM policy architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
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
            "include_params": include_params_in_obs
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

def create_ddpg(env, seed, include_params_in_obs=True, feature_extractor_name="FILM"):
    """Create a DDPG model with custom neural network architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
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

def create_td3(env, seed, include_params_in_obs=True, feature_extractor_name="FILM"):
    """Create a TD3 model with custom neural network architecture"""
    # Determine if the environment observation includes parameters
    obs_dim = env.observation_space.shape[0]
    lidar_dim = 1080
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

def create_sac(env, seed, include_params_in_obs=True, feature_extractor_name="FILM"):
    """Create a SAC model with custom neural network architecture"""
    # Get the appropriate feature extractor class
    features_extractor_class = get_feature_extractor_class(feature_extractor_name)
    
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": {
            "features_dim": 1024,
            "state_dim": 4,
            "lidar_dim": 1080,
            "param_dim": 12,
            "include_params": include_params_in_obs
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

def evaluate(eval_env, model_path="./logs/best_model/best_model.zip", algorithm="SAC", num_episodes=5, model=None, racing_mode=False, vecnorm_path=None):
    """
    Evaluates a trained model or wall-following policy on the environment.
    
    Args:
        eval_env: The environment (single instance or VecEnv) to evaluate in
        model_path: Path to the saved model (ignored when algorithm is WALL_FOLLOW, PURE_PURSUIT, LATTICE, or if model is provided)
        algorithm: Algorithm type (SAC, PPO, DDPG, TD3, WALL_FOLLOW, PURE_PURSUIT, LATTICE)
        num_episodes: Number of episodes to evaluate
        model: Optional pre-loaded model object (takes precedence over model_path)
        racing_mode: Whether to evaluate in racing mode with two cars
        vecnorm_path: Path to the saved VecNormalize statistics file. If None, will look for a file
                     in the same directory as model_path with "_vecnormalize.pkl" suffix.
    """
    # Set racing mode in environment if needed and not already set
    if racing_mode:
        logging.info("Evaluating in racing mode with two cars")
        eval_env.racing_mode = racing_mode
    
    # Load VecNormalize statistics if provided or can be inferred
    from stable_baselines3.common.vec_env import VecNormalize

    # Infer vecnorm_path from model_path if not explicitly provided
    if vecnorm_path is None and model_path is not None:
        model_dir = os.path.dirname(model_path)
        potential_vecnorm_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_vecnorm_path):
            vecnorm_path = potential_vecnorm_path
            logging.info(f"Found VecNormalize statistics at {vecnorm_path}")

    # Always load VecNormalize wrapper with loaded statistics for evaluation
    if vecnorm_path is not None:
        if os.path.exists(vecnorm_path):
            logging.info(f"Loading VecNormalize statistics from {vecnorm_path}")
            # Avoid double-wrapping: use base environment if already wrapped
            try:
                base_env = eval_env.venv
            except AttributeError:
                base_env = eval_env
            # Load normalization wrapper with saved stats
            eval_env = VecNormalize.load(vecnorm_path, base_env)
            # Disable further updates and reward normalization
            eval_env.training = False
            eval_env.norm_reward = False
            logging.info("Environment wrapped with VecNormalize (training=False, norm_reward=False)")
        else:
            logging.warning(f"VecNormalize statistics file not found at {vecnorm_path}")
    
    # Check if eval_env is a VecEnv
    is_vec_env = isinstance(eval_env, (DummyVecEnv, SubprocVecEnv, VecNormalize))
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
        elif algorithm == "RECURRENT_PPO":
            model = RecurrentPPO.load(model_path)
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
    
    # Initialize metrics per environment
    env_episode_rewards = [[] for _ in range(num_envs)]
    env_episode_lengths = [[] for _ in range(num_envs)]
    env_lap_times = [[] for _ in range(num_envs)]
    
    # For RecurrentPPO, we need to track LSTM states
    is_recurrent = algorithm == "RECURRENT_PPO" or (hasattr(model, 'policy') and hasattr(model.policy, '_initial_state'))
    
    # Run evaluation episodes for each environment
    for env_idx in range(num_envs):
        logging.info(f"Evaluating on environment {env_idx+1}/{num_envs}")
        
        for episode in range(num_episodes):
            logging.info(f"Starting evaluation episode {episode+1}/{num_episodes} on env {env_idx+1}")
            
            # Reset the environment
            if is_vec_env:
                obs = eval_env.env_method('reset', indices=[env_idx])[0][0]  # Reset specific env
                obs = np.array([obs])
                # Manually normalize if the environment is VecNormalize
                if isinstance(eval_env, VecNormalize):
                    obs = eval_env.normalize_obs(obs)
            else:
                obs = eval_env.reset()
            
            # For recurrent policies, initialize the LSTM states
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)  # Mark the start of the episode
            
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
                
                # For recurrent policies, we need to pass lstm_states and episode_starts
                if is_recurrent:
                    action, lstm_states = model.predict(
                        single_obs, 
                        state=lstm_states, 
                        episode_start=episode_starts, 
                        deterministic=True
                    )
                else:
                    action, _states = model.predict(single_obs, deterministic=True)
                
                # Take step in environment
                if is_vec_env:
                    # Step only the current environment
                    obs, reward, terminated, truncated, info = eval_env.env_method(
                        'step', action, indices=[env_idx]
                    )[0]
                    obs = np.array([obs])
                    # Manually normalize observation if needed
                    if isinstance(eval_env, VecNormalize):
                        obs = eval_env.normalize_obs(obs)
                    terminated = bool(terminated)
                    truncated = bool(truncated)
                    reward = float(reward)
                else:
                    # Gymnasium envs return 5 values
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                
                # Update episode_starts for recurrent policies
                if is_recurrent:
                    episode_starts = np.zeros((1,), dtype=bool)  # After first step, no longer episode start
                
                total_reward += reward
                step_count += 1
            
            # Episode completed
            episode_time = time.time() - episode_start_time
            
            # Record metrics for this environment
            env_episode_rewards[env_idx].append(total_reward)
            env_episode_lengths[env_idx].append(step_count)
            env_lap_times[env_idx].append(episode_time)
            
            logging.info(f"Episode {episode+1} finished:")
            logging.info(f"  Reward: {total_reward:.2f}")
            logging.info(f"  Length: {step_count} steps")
            logging.info(f"  Time: {episode_time:.2f} seconds")
            
            # Reset for next episode
            print(f"Episode {episode+1} finished. Resetting...")
    
    # Compute per-environment statistics
    env_stats = []
    for env_idx in range(num_envs):
        env_mean_reward = np.mean(env_episode_rewards[env_idx])
        env_std_reward = np.std(env_episode_rewards[env_idx])
        env_mean_episode_length = np.mean(env_episode_lengths[env_idx])
        env_mean_lap_time = np.mean(env_lap_times[env_idx])
        
        env_stats.append({
            "env_idx": env_idx,
            "mean_reward": env_mean_reward,
            "std_reward": env_std_reward,
            "mean_episode_length": env_mean_episode_length,
            "mean_lap_time": env_mean_lap_time,
            "episode_rewards": env_episode_rewards[env_idx],
            "episode_lengths": env_episode_lengths[env_idx],
            "lap_times": env_lap_times[env_idx]
        })
        
        logging.info(f"Environment {env_idx+1} statistics:")
        logging.info(f"  Mean reward: {env_mean_reward:.2f} ± {env_std_reward:.2f}")
        logging.info(f"  Mean episode length: {env_mean_episode_length:.2f} steps")
        logging.info(f"  Mean lap time: {env_mean_lap_time:.2f} seconds")
    
    # Compute overall statistics by flattening all environment data
    all_rewards = [reward for env_rewards in env_episode_rewards for reward in env_rewards]
    all_lengths = [length for env_lengths in env_episode_lengths for length in env_lengths]
    all_lap_times = [time for env_times in env_lap_times for time in env_times]
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_episode_length = np.mean(all_lengths)
    mean_lap_time = np.mean(all_lap_times)
    
    logging.info(f"Overall evaluation completed over {num_episodes * num_envs} episodes across {num_envs} environments:")
    logging.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logging.info(f"  Mean episode length: {mean_episode_length:.2f} steps")
    logging.info(f"  Mean lap time: {mean_lap_time:.2f} seconds")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": mean_episode_length,
        "mean_lap_time": mean_lap_time,
        "episode_rewards": all_rewards,
        "episode_lengths": all_lengths,
        "lap_times": all_lap_times,
        "env_stats": env_stats  # Added per-environment statistics
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
    model = pretrain_with_demonstrations(model, demonstrations)
    
    logging.info("Imitation learning completed")
    return model

def initialize_expert_policies(vec_env, imitation_policy_type, racing_mode):
    """
    Initialize expert policies for each environment.
    
    Args:
        vec_env: The vector environment
        imitation_policy_type: Type of imitation policy
        racing_mode: Whether racing mode is enabled
        
    Returns:
        List of expert policies
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
            expert_policies.append(LatticePlannerPolicy(track=track))
        else:
            raise ValueError(f"Unsupported imitation_policy_type: {imitation_policy_type}")
    
    return expert_policies

def collect_expert_rollouts(model, env, raw_vec_env, expert_policies, total_transitions, is_normalized):
    """
    Collect rollouts from expert policies and filter by reward.
    
    Args:
        model: The RL model
        env: The normalized environment (if normalization is used)
        raw_vec_env: The raw vector environment
        expert_policies: List of expert policies
        total_transitions: Target total number of transitions to collect
        is_normalized: Whether environment is normalized
        
    Returns:
        List of demonstrations (transitions from successful rollouts)
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
        while min(env_transitions_collected) < transitions_per_env - 1000:
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
                            if current_rollout_rewards[i] > -100:
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
    
    # --- Start of new section for batching demonstrations ---
    num_envs_for_batching = raw_vec_env.num_envs # Ensure we use the correct num_envs

    # 1. Extract all transitions per environment from env_rollouts
    # Each element in env_rollouts is a SortedList of (reward, rollout_data)
    # rollout_data is a list of transitions: (obs, action, reward, next_obs, done)
    transitions_by_env = [[] for _ in range(num_envs_for_batching)]
    for env_idx, rollout_list_for_env in enumerate(env_rollouts):
        for _reward_val, rollout_data in rollout_list_for_env: 
            for transition in rollout_data: 
                transitions_by_env[env_idx].append(transition)

    # 2. Determine the number of full batches we can form
    # Check if any environment has no transitions collected or if num_envs_for_batching is not positive
    if not num_envs_for_batching > 0 or not all(transitions_by_env) or any(len(t_list) == 0 for t_list in transitions_by_env) :
        logging.warning("One or more environments have no successful rollouts, or no environments processed. "
                        "Cannot form batches for pretraining. Pretraining steps requiring the replay buffer will be skipped.")
        return [] # Return empty list; pretraining loop won't run

    min_transitions_per_env = min(len(trans_list) for trans_list in transitions_by_env)

    if min_transitions_per_env == 0:
        # This case should ideally be covered by the comprehensive check above, but acts as a safeguard.
        logging.warning("Not enough transitions across all environments to form batches (min is 0 after filtering). "
                        "Pretraining steps requiring the replay buffer will be skipped.")
        return []

    # 3. Form batches
    batched_demonstrations = []
    for i in range(min_transitions_per_env):
        batch_obs_list = []
        batch_action_list = []
        batch_reward_list = []
        batch_next_obs_list = []
        batch_done_list = []

        for env_j in range(num_envs_for_batching):
            # Each transition is (obs, action, reward, next_obs, done)
            obs, action, reward, next_obs, done = transitions_by_env[env_j][i]
            batch_obs_list.append(obs)
            batch_action_list.append(action)
            batch_reward_list.append(reward)
            batch_next_obs_list.append(next_obs)
            batch_done_list.append(done)
        
        # Stack to create numpy arrays for the batch
        batched_demonstrations.append((
            np.array(batch_obs_list),      # Shape: (num_envs, obs_dim)
            np.array(batch_action_list),   # Shape: (num_envs, action_dim)
            np.array(batch_reward_list),   # Shape: (num_envs,)
            np.array(batch_next_obs_list), # Shape: (num_envs, obs_dim)
            np.array(batch_done_list)      # Shape: (num_envs,)
        ))

    total_batched_transitions = min_transitions_per_env * num_envs_for_batching
    logging.info(f"Collected and batched {total_batched_transitions} demonstration transitions, "
                 f"forming {min_transitions_per_env} batches of size {num_envs_for_batching}.")
    
    return batched_demonstrations
    # --- End of new section ---

def pretrain_with_demonstrations(model, demonstrations):
    """
    Pretrain the model using filtered demonstrations.
    
    Args:
        model: The RL model to train
        demonstrations: List of transitions from successful rollouts
        
    Returns:
        Trained model
    """
    if hasattr(model, 'replay_buffer'):
        if not demonstrations: # Check if demonstrations list is empty (it's a list of batches)
            logging.info("No demonstration batches to pretrain on. Skipping replay buffer population and subsequent training on demonstrations.")
        else:
            # demonstrations is now a list of (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
            logging.info(f"Adding {len(demonstrations)} batches of demonstrations to the replay buffer.")
            for batch_obs, batch_action, batch_reward, batch_next_obs, batch_done in demonstrations:
                # batch_obs, batch_action, etc. are already numpy arrays with shape (num_envs, ...)
                num_envs_in_batch = batch_obs.shape[0]
                infos = [{} for _ in range(num_envs_in_batch)] # Create a list of info dicts for the batch
                model.replay_buffer.add(batch_obs, batch_next_obs, batch_action, batch_reward, batch_done, infos)
            
            # Perform gradient steps on the demonstrations, only if model has a train method
            if hasattr(model, 'train'):
                logging.info("Training on demonstrations")
                # Initialize the model by calling learn with 1 step before training
                # This ensures the logger and other components are properly set up.
                # reset_num_timesteps defaults to True for the first call to model.learn(), which is appropriate here.
                model.learn(total_timesteps=1, log_interval=1) 
                
                # Now we can safely call train
                gradient_steps = 10_000 # This can be made configurable if needed
                with tqdm(total=gradient_steps, desc="Imitation Learning Progress") as pbar:
                    for _ in range(gradient_steps): 
                        model.train(gradient_steps=1, batch_size=model.batch_size)
                        pbar.update(1)
            else:
                logging.info("Model does not have a 'train' method. Skipping gradient steps on demonstrations.")
    else:
        logging.info("Model does not have a replay buffer. Skipping pretraining with demonstrations.")

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

def create_vec_env(env_kwargs, seed, num_envs=1, num_param_cmbs=None, use_domain_randomization=False, include_params_in_obs=True, racing_mode=False, normalize_obs=True, normalize_reward=True):
    """
    Creates vectorized environments for training and evaluation.

    Args:
        env_kwargs (dict): Base arguments for the F110GymWrapper environment.
        seed (int): Random seed.
        num_envs (int): Number of parallel environments to use.
        num_param_cmbs (int): Number of parameter combinations to use for domain randomization.
                             If None and use_domain_randomization is True, defaults to num_envs.
        use_domain_randomization (bool): Whether to randomize environment parameters.
        include_params_in_obs (bool): Whether to include environment parameters in observations.
        racing_mode (bool): Whether to use racing mode with two cars.
        normalize_obs (bool): Whether to normalize observations.
        normalize_reward (bool): Whether to normalize rewards.

    Returns:
        VecEnv: The vectorized environment, optionally wrapped with VecNormalize.
    """
    # If num_param_cmbs is not specified but DR is used, default to num_envs
    if num_param_cmbs is None and use_domain_randomization:
        num_param_cmbs = num_envs
    # If using DR, ensure num_param_cmbs is at least 1
    elif use_domain_randomization and num_param_cmbs < 1:
        num_param_cmbs = 1
        logging.warning(f"num_param_cmbs must be at least 1 for domain randomization. Setting to 1.")
    
    # Generate parameter combinations first if using domain randomization
    param_combinations = []
    if use_domain_randomization and num_param_cmbs > 0:
        # Create the specified number of parameter combinations
        for i in range(num_param_cmbs):
            param_seed = seed + i
            rng = np.random.default_rng(param_seed)
            param_set = {
                'mu': rng.uniform(0.8, 1.1),
                'C_Sf': rng.uniform(4.0, 5.5),
                'C_Sr': rng.uniform(4.0, 5.5),
                'm': rng.uniform(3.0, 4.5),
                'I': rng.uniform(0.03, 0.06),
                'lidar_noise_stddev': rng.uniform(0.0, 0.1),
                's_noise_stddev': rng.uniform(0.0, 1),
                'ey_noise_stddev': rng.uniform(0.0, 0.5),
                'vel_noise_stddev': rng.uniform(0.0, 0.5),
                'yaw_noise_stddev': rng.uniform(0, 0.5)
            }
            param_set['push_0_prob'] = rng.uniform(0.0, 0.1)
            param_set['push_2_prob'] = rng.uniform(0.0, 0.05)
            
            logging.info(
                f"Param Set {i} Parameters: mu: {param_set['mu']}, C_Sf: {param_set['C_Sf']}, C_Sr: {param_set['C_Sr']}, m: {param_set['m']}, I: {param_set['I']}; "
                f"Observation noise: lidar_noise_stddev: {param_set['lidar_noise_stddev']}, s: {param_set['s_noise_stddev']}, ey: {param_set['ey_noise_stddev']}, vel: {param_set['vel_noise_stddev']}, yaw: {param_set['yaw_noise_stddev']}; "
                f"Push probabilities: push_0_prob: {param_set['push_0_prob']}, push_2_prob: {param_set['push_2_prob']}"
            )
            param_combinations.append(param_set)
    
    # --- Create Environment(s) ---
    env_fns = []
    for i in range(num_envs):
        rank_seed = seed + i
        current_env_kwargs = env_kwargs.copy()
        current_env_kwargs['include_params_in_obs'] = include_params_in_obs
        
        # Ensure racing_mode is passed to environment
        if 'racing_mode' not in current_env_kwargs:
            current_env_kwargs['racing_mode'] = racing_mode
        
        if use_domain_randomization and param_combinations:
            # Select parameter set from the combinations we created
            # Each parameter set is used for (num_envs / num_param_cmbs) environments
            param_idx = i % num_param_cmbs
            param_set = param_combinations[param_idx]
            
            # Assign parameters to this environment
            for key, value in param_set.items():
                current_env_kwargs[key] = value
            
            logging.info(f"Env {i} using parameter set {param_idx}")
        
        # Create the thunk (function) for this env instance
        # Use partial to pass the potentially modified kwargs
        env_fn = partial(make_env(env_id=f"f110-rank{i}", rank=i, seed=seed, env_kwargs=current_env_kwargs))
        env_fns.append(env_fn)
        
    # Create the vectorized environment
    vec_env_cls = DummyVecEnv if num_envs == 1 else SubprocVecEnv
    vec_env = vec_env_cls(env_fns)
    
    # Optionally wrap with VecNormalize
    if normalize_obs or normalize_reward:
        from stable_baselines3.common.vec_env import VecNormalize
        vec_env = VecNormalize(
            vec_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )
        logging.info(f"Environment wrapped with VecNormalize (norm_obs={normalize_obs}, norm_reward={normalize_reward})")
    
    return vec_env

# Updated train function to handle VecEnv and Domain Randomization
def train(env, seed, num_envs=1, num_param_cmbs=None, use_domain_randomization=False, use_imitation_learning=True, imitation_policy_type="PURE_PURSUIT", algorithm="SAC", include_params_in_obs=True, racing_mode=False, normalize_obs=True, normalize_reward=True, feature_extractor_name="FILM"):
    """
    Trains the RL model.

    Args:
        env: Vectorized environment for training.
        seed (int): Random seed.
        num_envs (int): Number of parallel environments to use.
        num_param_cmbs (int): Number of parameter combinations to use for domain randomization.
                             If None and use_domain_randomization is True, defaults to num_envs.
        use_domain_randomization (bool): Whether to randomize environment parameters.
        use_imitation_learning (bool): Whether to use imitation learning before RL training.
        imitation_policy_type (str): Policy for imitation learning.
        algorithm (str): RL algorithm to use (e.g., SAC, PPO, RECURRENT_PPO).
        include_params_in_obs (bool): Whether to include environment parameters in observations.
        racing_mode (bool): Whether to train in racing mode with two cars.
        normalize_obs (bool): Whether to normalize observations.
        normalize_reward (bool): Whether to normalize rewards.
        feature_extractor_name (str): Name of feature extractor architecture to use.
    """
    # --- Create Model ---
    logging.info(f"Creating {algorithm} model with {feature_extractor_name} feature extractor")
    if algorithm == "PPO":
        model = create_ppo(env, seed, include_params_in_obs, feature_extractor_name)
    elif algorithm == "RECURRENT_PPO":
        model = create_recurrent_ppo(env, seed, include_params_in_obs, feature_extractor_name)
    elif algorithm == "DDPG":
        model = create_ddpg(env, seed, include_params_in_obs, feature_extractor_name)
    elif algorithm == "TD3":
        model = create_td3(env, seed, include_params_in_obs, feature_extractor_name)
    elif algorithm == "SAC":
        model = create_sac(env, seed, include_params_in_obs, feature_extractor_name)
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