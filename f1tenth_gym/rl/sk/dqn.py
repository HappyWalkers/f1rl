import gym
import torch
import numpy as np
from absl import logging

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.models.torch import Model, DeterministicMixin

# --------------------------------------------
# Custom Q-Network
# --------------------------------------------
class CustomQNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        """
        DQN for a discrete action space outputs Q-values for all actions.
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
        The DeterministicMixin expects (outputs, additional_info).
        Q-values for each action: shape [batch_size, num_actions]
        """
        states = inputs["states"]  # shape: (batch_size, observation_dim)
        q_values = self.network(states)
        
        logging.info(f"(Q-Network) Q-values: {q_values}")
        return q_values, {}


# --------------------------------------------
# Create DQN agent
# --------------------------------------------
def create_dqn_agent(env, device):
    """
    Creates and returns a standard DQN agent with:
      - Replay buffer (RandomMemory)
      - Q-network and target Q-network
      - DQN configuration (double Q-learning disabled)
    """
    # Replay memory
    memory_size = 100_000
    memory = RandomMemory(memory_size=memory_size, num_envs=1, device=device)

    # DQN configuration
    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["random_timesteps"] = 1000       # random actions before learning
    cfg["learning_starts"] = 1000
    cfg["batch_size"] = 64
    cfg["discount_factor"] = 0.99
    cfg["learning_rate"] = 1e-3
    cfg["double_q_learning"] = False     # Ensure Double Q-learning is off (default is False)
    
    # Create networks (Q-network and its target)
    models = {
        "q_network": CustomQNetwork(env.observation_space, env.action_space, device),
        "target_q_network": CustomQNetwork(env.observation_space, env.action_space, device),
    }

    # Create the DQN agent
    agent = DQN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    return agent