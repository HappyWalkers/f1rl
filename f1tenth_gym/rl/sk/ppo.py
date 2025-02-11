import gym
import torch
import numpy as np
import random
import torch.distributions as D
from absl import logging
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler


class CustomPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        # Initialize the Gaussian Mixin (clip_actions controls whether actions 
        # are clipped to the action space; you can also use clip_log_std, etc.)
        GaussianMixin.__init__(self, clip_actions=True)

        hidden_size = 128
        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_space.shape[0])
        )
        # log-std parameter for continuous actions
        self.log_std = torch.nn.Parameter(torch.zeros(action_space.shape[0]))

    def compute(self, inputs, role=""):
        """
        This method is used internally by the mixin to build a diagonal Gaussian distribution.
        Return a dict with "mean" and "log_std" (or "std") keys.
        """
        states = inputs["states"]
        logging.info(f"(Policy) states: {states}")

        mu = self.network(states)                  # shape: [batch_size, action_dim]
        log_std = self.log_std.expand_as(mu)       # shape: [batch_size, action_dim]

        # Optionally log:
        logging.info(f"(Policy) mu: {mu}")

        # Return the distribution parameters to GaussianMixin
        return mu, log_std, {}


class CustomValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        # Initialize the base Model class
        Model.__init__(self, observation_space, action_space, device)
        # Initialize the DeterministicMixin
        # (clip_actions = False, since this is a value function, not an action)
        DeterministicMixin.__init__(self, clip_actions=False)

        hidden_size = 128
        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)  # single scalar for value
        )

    def compute(self, inputs, role=""):
        """
        This method is used by DeterministicMixin to get output.
        
        Return:
          - tensor of shape [batch_size, 1] for the value
          - dictionary (optional) with extra outputs
        """
        states = inputs["states"]
        v = self.network(states)  # shape: [batch_size, 1]

        logging.info(f"(Value) v: {v}")

        # Return the value and an (optional) empty dictionary
        return v, {}
    

def create_ppo_agent(env, device):
    # Set up memory
    rollouts = 1024
    memory = RandomMemory(memory_size=rollouts, num_envs=1, device=device)
    
    # PPO configuration
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts
    cfg["learning_epochs"] = 10
    cfg["mini_batches"] = 64
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["mixed_precision"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Create agent
    models = {
        "policy": CustomPolicy(env.observation_space, env.action_space, device),
        "value": CustomValue(env.observation_space, env.action_space, device)
    }
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    return agent
