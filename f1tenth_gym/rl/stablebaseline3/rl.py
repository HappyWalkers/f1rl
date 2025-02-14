from stable_baselines3 import PPO, DDPG, DQN, TD3, SAC
from absl import logging


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

def train(env, seed):
    # Create PPO model
    model = create_ppo(env, seed)
    # model = create_ddpg(env, seed)
    # model = create_td3(env, seed)
    # model = create_sac(env, seed)

    # Train
    model.learn(
        total_timesteps=30_000_000,
        log_interval=1,
        tb_log_name="PPO",
        reset_num_timesteps=True,
        progress_bar=True
    )
    
    # Evaluate or run an infinite loop to visualize the learned policy
    obs = env.reset()
    terminated = False
    while True:
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        logging.info(f"action: {action}")
        obs, reward, terminated, info = env.step(action)
        if terminated:
            obs = env.reset()
            terminated = False
            print("Episode finished. Resetting...")