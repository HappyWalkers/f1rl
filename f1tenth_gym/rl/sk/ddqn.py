import gym
import torch
import numpy as np
from absl import logging

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler

# --------------------------------------------
# Custom Q-Network
# --------------------------------------------
class CustomQNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        """
        For DQN/Double DQN in a discrete action space, the network outputs Q-values for all actions.
        :param observation_space: environment's observation space
        :param action_space: environment's action space (discrete)
        :param device: torch device
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # The number of possible discrete actions
        self.num_actions = action_space.shape[0]

        hidden_size = 128
        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.num_actions)
        )

    def compute(self, inputs, role=""):
        """
        The DeterministicMixin expects (outputs, {}) as return.
        :param inputs: dictionary with "states" among other possible entries
        :param role: string identifying the model's role
        :return: A tuple (Q-values, dict)
        """
        states = inputs["states"]
        # Q-values for each action
        q_values = self.network(states)  # shape: [batch_size, num_actions]

        logging.info(f"(Q-Network) Q-values: {q_values}")
        return q_values, {}


# --------------------------------------------
# Helper function to create a Double DQN agent
# --------------------------------------------
def create_ddqn_agent(env, device):
    """
    Create and return a Double DQN agent with:
      - Replay buffer (RandomMemory)
      - Q-network, target Q-network
      - Double DQN configuration
    """
    # Replay memory
    memory_size = 100000
    memory = RandomMemory(memory_size=memory_size, num_envs=1, device=device)

    # Default DQN configuration
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["random_timesteps"] = 1000       # Steps using random actions before learning
    cfg["learning_starts"] = 1000
    cfg["batch_size"] = 64
    cfg["discount_factor"] = 0.99
    cfg["double_q_learning"] = True      # Enable Double Q-learning
    cfg["learning_rate"] = 1e-4
    # Optional: normalize states
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}

    # Create models (Q-network and its target)
    models = {
        "q_network": CustomQNetwork(env.observation_space, env.action_space, device),
        "target_q_network": CustomQNetwork(env.observation_space, env.action_space, device),
    }

    # Create DDQN agent
    agent = DQN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    return agent