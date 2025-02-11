import gym
import torch
import torch.nn as nn

from absl import logging

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer

class CustomActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        """
        TD3 actor that outputs a deterministic action for each state.
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=True)  # clip to valid range

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.act_dim)
        )

    def compute(self, inputs, role=""):
        """
        The DeterministicMixin expects (action, extras).
        inputs["states"] has shape (batch_size, obs_dim).
        """
        states = inputs["states"]
        action = self.network(states)
        
        logging.info(f"(Actor) action: {action}")
        return action, {}


class CustomCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        """
        TD3 critic takes (state, action) as input, outputs a Q-value (scalar).
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        # (state_dim + action_dim) => hidden => Q-value
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def compute(self, inputs, role=""):
        """
        Inputs dict will have:
          - "states" of shape (batch_size, obs_dim)
          - "taken_actions" of shape (batch_size, act_dim)
        """
        states = inputs["states"]
        actions = inputs["taken_actions"]
        x = torch.cat([states, actions], dim=-1)  # concatenate
        
        q_value = self.network(x)
        logging.info(f"(Critic) Q-value: {q_value}")
        return q_value, {}

def create_td3_agent(env, device):
    """
    Create a TD3 agent with:
      - RandomMemory (replay buffer)
      - 1 actor, 2 critics (and their target networks)
      - Standard TD3 configs
    """
    # Replay memory
    memory_size = 100000
    memory = RandomMemory(memory_size=memory_size, num_envs=1, device=device)

    # TD3 configuration
    cfg = TD3_DEFAULT_CONFIG.copy()
    cfg["random_timesteps"] = 1000           # collect experience randomly before learning
    cfg["learning_starts"] = 1000
    cfg["batch_size"] = 64
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005                    # soft-update coefficient
    cfg["actor_learning_rate"] = 1e-4
    cfg["critic_learning_rate"] = 1e-3

    # TD3-specific hyperparameters
    cfg["smooth_regression"] = True          # smooth the target using clipped noise
    cfg["smooth_noise_std"] = 0.2            # std for target noise
    cfg["smooth_noise_clip"] = 0.5           # clip range for target noise
    cfg["policy_delay"] = 2                  # delayed policy update (update critics every step, actor every 2 steps)

    # Optional: normalizers
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}

    # Create models: Actor, Critic1, Critic2, and their targets
    models = {
        "policy": CustomActor(env.observation_space, env.action_space, device),
        "critic_1": CustomCritic(env.observation_space, env.action_space, device),
        "critic_2": CustomCritic(env.observation_space, env.action_space, device),
        "target_policy": CustomActor(env.observation_space, env.action_space, device),
        "target_critic_1": CustomCritic(env.observation_space, env.action_space, device),
        "target_critic_2": CustomCritic(env.observation_space, env.action_space, device),
    }

    # Create TD3 agent
    agent = TD3(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    return agent
