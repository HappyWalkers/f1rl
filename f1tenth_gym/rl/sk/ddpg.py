import gym
import torch
import numpy as np
from absl import logging

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler

# --------------------------------------------
# Custom Actor (Deterministic Policy)
# --------------------------------------------
class CustomActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        """
        :param observation_space: environment's observation (state) space
        :param action_space: environment's action space
        :param device: torch device
        :param clip_actions: whether to clip the actions to the valid range
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        hidden_size = 128
        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_space.shape[0])
        )

    def compute(self, inputs, role=""):
        """
        The DeterministicMixin expects a tuple (outputs, additional_info) as return.
        - inputs["states"] has shape (batch_size, observation_dim)
        - We want to return an action of shape (batch_size, action_dim)
        """
        states = inputs["states"]
        action = self.network(states)
        
        logging.info(f"(Actor) action: {action}")
        return action, {}


# --------------------------------------------
# Custom Critic (Deterministic Q-function)
# --------------------------------------------
class CustomCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        """
        The critic in DDPG takes both the state and the action as input
        and outputs a single Q-value.
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        hidden_size = 128
        # input dimension = state_dim + action_dim
        input_dim = observation_space.shape[0] + action_space.shape[0]

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)  # Q-value
        )

    def compute(self, inputs, role=""):
        """
        The DeterministicMixin expects (q_value, {}) as return.
        - inputs["states"] has shape (batch_size, observation_dim)
        - inputs["taken_actions"] has shape (batch_size, action_dim)
        """
        states = inputs["states"]
        actions = inputs["taken_actions"]
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=-1)
        q_value = self.network(x)
        
        logging.info(f"(Critic) Q-value: {q_value}")
        return q_value, {}


# --------------------------------------------
# Helper function to create a DDPG agent
# --------------------------------------------
def create_ddpg_agent(env, device):
    """
    Create and return a DDPG agent with:
      - A replay buffer (RandomMemory)
      - An actor, critic, and their target networks
      - DDPG configuration
    """
    # Replay memory
    memory_size = 100000
    memory = RandomMemory(memory_size=memory_size, num_envs=1, device=device)

    # Default DDPG configuration
    cfg = DDPG_DEFAULT_CONFIG.copy()
    cfg["random_timesteps"] = 1000         # Collect experiences before learning
    cfg["learning_starts"] = 1000
    cfg["batch_size"] = 64
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005                  # Soft-update parameter
    cfg["actor_learning_rate"] = 3e-4
    cfg["critic_learning_rate"] = 3e-4
    # Optionally, use state normalizers
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}

    # Create models (actor, critic, and their targets)
    models = {
        "policy": CustomActor(env.observation_space, env.action_space, device, clip_actions=True),
        "critic": CustomCritic(env.observation_space, env.action_space, device),
        "target_policy": CustomActor(env.observation_space, env.action_space, device, clip_actions=True),
        "target_critic": CustomCritic(env.observation_space, env.action_space, device),
    }

    # Create DDPG agent
    agent = DDPG(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    return agent
