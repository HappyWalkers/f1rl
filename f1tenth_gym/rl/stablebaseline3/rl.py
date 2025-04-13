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

from rl_env import F110GymWrapper # Import the wrapper


def create_ppo(env, seed):
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
        policy_kwargs={
            "net_arch": [
                dict(pi=[256, 256, 256], vf=[256, 256, 256])
            ]
        },
        verbose=1,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model

def create_ddpg(env, seed):
    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=64,
        tau=0.001,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )
    return model

def create_td3(env, seed):
    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=64,
        tau=0.001,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )
    return model

def create_sac(env, seed):
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
        policy_kwargs={
            "net_arch": [1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16]
        },
        verbose=1,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model

def evaluate(eval_env, model_path="./logs/best_model/best_model.zip", algorithm="SAC", num_episodes=5, model=None):
    """
    Evaluates a trained model or wall-following policy on the environment.
    
    Args:
        eval_env: The environment (single instance or VecEnv) to evaluate in
        model_path: Path to the saved model (ignored when algorithm is WALL_FOLLOW, PURE_PURSUIT, or if model is provided)
        algorithm: Algorithm type (SAC, PPO, DDPG, TD3, WALL_FOLLOW, PURE_PURSUIT)
        num_episodes: Number of episodes to evaluate
        model: Optional pre-loaded model object (takes precedence over model_path)
    """
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
            track = getattr(eval_env.unwrapped, 'track', None)  # Access unwrapped env for track
        model = PurePursuitPolicy(track=track)
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
                
                # Optional: Add a small delay for better visualization
                time.sleep(0.01)
            
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

def initialize_with_imitation_learning(model, env, imitation_policy_type="PURE_PURSUIT", total_transitions=1000_000):
    """
    Initialize a reinforcement learning model using imitation learning from a specified policy.
    
    Args:
        model: The RL model to initialize
        env: The environment (must be a VecEnv) to collect demonstrations from.
        imitation_policy_type: Type of imitation policy (WALL_FOLLOW or PURE_PURSUIT)
        total_transitions: Total number of transitions to collect across all environments
    
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

    # Initialize the expert policies for each environment
    logging.info(f"Starting imitation learning from {imitation_policy_type} policy")
    expert_policies = []
    for i in range(env.num_envs):
        if imitation_policy_type == "WALL_FOLLOW":
            expert_policies.append(WallFollowPolicy())
        elif imitation_policy_type == "PURE_PURSUIT":
            # Extract the track for this environment
            track = env.get_attr("track", indices=i)[0]
            # Ensure the environment has the track object needed by PurePursuit
            if not track:
                raise ValueError("Environment does not have a 'track' attribute required for PURE_PURSUIT imitation.")
            expert_policies.append(PurePursuitPolicy(track=track))
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
            model.train(gradient_steps=10_000, batch_size=model.batch_size)

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

def create_vec_env(env_kwargs, seed, num_envs=1, use_domain_randomization=False):
    """
    Creates vectorized environments for training and evaluation.

    Args:
        env_kwargs (dict): Base arguments for the F110GymWrapper environment.
        seed (int): Random seed.
        num_envs (int): Number of parallel environments to use.
        use_domain_randomization (bool): Whether to randomize environment parameters.

    Returns:
        VecEnv: The vectorized environment.
    """
    # --- Create Environment(s) ---
    env_fns = []
    for i in range(num_envs):
        rank_seed = seed + i
        current_env_kwargs = env_kwargs.copy()
        
        if use_domain_randomization:
            # Sample parameters randomly for this environment instance
            rng = np.random.default_rng(rank_seed)
            current_env_kwargs['mu'] = rng.uniform(0.7, 1.3) # Example range
            current_env_kwargs['C_Sf'] = rng.uniform(3.0, 6.0)
            current_env_kwargs['C_Sr'] = rng.uniform(3.0, 6.0)
            # Add more parameter randomization here (mass, inertia, lengths, etc.)
            current_env_kwargs['m'] = rng.uniform(3.0, 4.5)
            current_env_kwargs['I'] = rng.uniform(0.03, 0.06)
            current_env_kwargs['lidar_noise_stddev'] = rng.uniform(0.0, 0.02) # Example range
            # Randomize push probabilities, keeping average push = 1
            sampled_push_0_prob = rng.uniform(0.1, 0.4) # Sample p0
            current_env_kwargs['push_0_prob'] = sampled_push_0_prob
            current_env_kwargs['push_2_prob'] = sampled_push_0_prob # Set p2 = p0

        # Create the thunk (function) for this env instance
        # Use partial to pass the potentially modified kwargs
        env_fn = partial(make_env(env_id=f"f110-rank{i}", rank=i, seed=seed, env_kwargs=current_env_kwargs))
        env_fns.append(env_fn)
        
    # Create the vectorized environment
    vec_env_cls = DummyVecEnv if num_envs == 1 else SubprocVecEnv
    env = vec_env_cls(env_fns)
    
    return env

# Updated train function to handle VecEnv and Domain Randomization
def train(env, seed, num_envs=1, use_domain_randomization=False, use_imitation_learning=True, imitation_policy_type="PURE_PURSUIT", algorithm="SAC"):
    """
    Trains the RL model.

    Args:
        env: Vectorized environment for training.
        seed (int): Random seed.
        num_envs (int): Number of parallel environments to use.
        use_domain_randomization (bool): Whether to randomize environment parameters.
        use_imitation_learning (bool): Whether to use imitation learning before RL.
        imitation_policy_type (str): Policy for imitation learning.
        algorithm (str): RL algorithm to use (e.g., SAC, PPO).
    """
    # --- Create Model ---
    if algorithm == "PPO":
        model = create_ppo(env, seed)
    elif algorithm == "DDPG":
        model = create_ddpg(env, seed)
    elif algorithm == "TD3":
        model = create_td3(env, seed)
    elif algorithm == "SAC":
        model = create_sac(env, seed)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # --- Imitation Learning (Optional, might need adaptation for VecEnv) ---
    if use_imitation_learning:
        logging.info("Using imitation learning to bootstrap the model.")
        model = initialize_with_imitation_learning(model, env, imitation_policy_type=imitation_policy_type)
    else:
        logging.info("Skipping imitation learning.")

    logging.info(f"Starting RL training with {env.num_envs} environments.")

    # --- RL Training ---
    # Create formatted path based on training parameters and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = f"{algorithm}_envs{num_envs}_dr{int(use_domain_randomization)}_il{int(use_imitation_learning)}"
    if use_imitation_learning:
        model_dir_name += f"_{imitation_policy_type}"
    model_dir_name += f"_seed{seed}_{timestamp}"
    
    # Create the full paths
    best_model_path = os.path.join("./logs", model_dir_name, "best_model")
    log_path = os.path.join("./logs", model_dir_name, "results")
    
    # Ensure the directories exist
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Import callbacks for saving best model based on training rewards
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    # Custom callback to save model with highest training reward
    class SaveOnBestTrainingRewardCallback(BaseCallback):
        def __init__(self, check_freq, save_path, verbose=1):
            super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path
            self.best_mean_reward = -float('inf')
        
        def _init_callback(self):
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
        
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
                        self.model.save(self.save_path)
            
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