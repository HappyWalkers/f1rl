import gym
import torch
import numpy as np
import random
import torch.distributions as D
from absl import logging
from absl import app
from absl import flags
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from sk.ppo import create_ppo_agent
from sk.ddpg import create_ddpg_agent
from sk.sac import create_sac_agent
from sk.ddqn import create_ddqn_agent
from sk.dqn import create_dqn_agent
from sk.rpo import create_rpo_agent
from sk.trpo import create_trpo_agent
from sk.td3 import create_td3_agent



def train(env):
    # torch.autograd.set_detect_anomaly(True)
    
    # SKRL requires an environment wrapper
    env = wrap_env(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # agent
    agent = create_ppo_agent(env, device)
    # agent = create_ddpg_agent(env, device)
    # agent = create_dqn_agent(env, device)
    # agent = create_ddqn_agent(env, device)
    # agent = create_rpo_agent(env, device)
    # agent = create_trpo_agent(env, device)
    # agent = create_td3_agent(env, device)
    # agent = create_sac_agent(env, device)

    # Training
    trainer_cfg = {
        "timesteps": 1024 * 20,
        "headless": True  # set to False if you want to see some rendering
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()

    # # Evaluation
    # trainer.cfg["timesteps"] = 2000
    # trainer.cfg["headless"] = False
    # trainer.eval()
    
    # render
    while True:
        obs, _ = env.reset()
        done = False
        truncated = False
        i = 0
        while not (done or truncated):
            i += 1
            env.render()
            action, _, _ = agent.models['policy'].act({"states": torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)}, role="policy")
            obs, step_reward, done, truncated, info = env.step(action.detach())
        print('finish one episode')