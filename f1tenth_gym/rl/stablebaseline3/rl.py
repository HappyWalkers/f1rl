from stable_baselines3 import PPO, DDPG, DQN, TD3, SAC
from absl import logging
from stable_baselines3.common.callbacks import EvalCallback
import os
import time
import numpy as np


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
        learning_starts=1000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
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
            "net_arch": [256, 256, 256, 256]
        },
        verbose=1,
        seed=seed,
        device="auto",
        _init_setup_model=True,
    )
    return model

def evaluate(env, model_path="./logs/best_model/best_model.zip", algorithm="SAC", num_episodes=5):
    """
    Evaluates a trained model on the environment.
    
    Args:
        env: The environment to evaluate in
        model_path: Path to the saved model
        algorithm: Algorithm type (SAC, PPO, DDPG, TD3)
        num_episodes: Number of episodes to evaluate
    """
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
    
    # Initialize metrics
    episode_rewards = []
    episode_lengths = []
    lap_times = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        logging.info(f"Starting evaluation episode {episode+1}/{num_episodes}")
        obs, info = env.reset()
        
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        episode_start_time = time.time()
        
        while not (terminated or truncated):
            # Render environment
            env.render()
            
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
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
        obs, info = env.reset()
        terminated = False
        truncated = False
        print(f"Episode {episode+1} finished. Resetting...")
    
    # Compute summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_episode_length = np.mean(episode_lengths)
    mean_lap_time = np.mean(lap_times)
    
    logging.info(f"Evaluation completed over {num_episodes} episodes:")
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

def train(env, seed):
    # model = create_ppo(env, seed)
    # model = create_ddpg(env, seed)
    # model = create_td3(env, seed)
    model = create_sac(env, seed)

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=10_000_000,
        log_interval=1,
        tb_log_name="PPO",
        reset_num_timesteps=True,
        progress_bar=False,
        callback=eval_callback
    )

    # After training, evaluate the best model
    evaluate(env, model_path="./logs/best_model/best_model.zip", algorithm="SAC", num_episodes=5)