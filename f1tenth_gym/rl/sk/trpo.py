import gym
import torch
from absl import logging

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer

class CustomPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        """
        A Gaussian policy outputs mean (mu) and log_std for continuous actions.
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)  # clip actions within valid range

        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_space.shape[0])
        )

        # Log-std parameter for continuous actions
        self.log_std = torch.nn.Parameter(torch.zeros(action_space.shape[0]))

    def compute(self, inputs, role=""):
        """
        The GaussianMixin expects: (mu, log_std, extra_info).
         - mu: mean of the Gaussian (shape: [batch_size, action_dim])
         - log_std: log standard deviation (shape: [batch_size, action_dim])
         - extra_info: dictionary with optional extra outputs
        """
        states = inputs["states"]
        mu = self.network(states)
        log_std = self.log_std.expand_as(mu)

        logging.info(f"(Policy) mu: {mu}")

        # Return a dict with 'rnn_states' if you're using RNNs. For MLP, we can return {}.
        return mu, log_std, {}

class CustomValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        """
        A value function outputs a single scalar value for each state.
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

    def compute(self, inputs, role=""):
        """
        The DeterministicMixin expects: (value, extra_info).
         - value: shape [batch_size, 1]
        """
        states = inputs["states"]
        value = self.network(states)

        logging.info(f"(Value) value: {value}")

        return value, {}

def create_trpo_agent(env, device):
    """
    Create a TRPO agent with:
      - RandomMemory for on-policy rollouts
      - A custom policy and value function
      - TRPO-specific configuration
    """
    # Number of timesteps (and thus experiences) per rollout
    rollouts = 1024
    memory = RandomMemory(memory_size=rollouts, num_envs=1, device=device)

    # Default TRPO configuration
    cfg = TRPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4  # If needed, though TRPO often uses full-batch
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["damping_coeff"] = 0.1
    cfg["max_kl_divergence"] = 0.01
    cfg["cg_max_steps"] = 10
    cfg["cg_residual_tol"] = 1e-10
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    # Optionally, normalizers
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Create models
    models = {
        "policy": CustomPolicy(env.observation_space, env.action_space, device),
        "value": CustomValue(env.observation_space, env.action_space, device)
    }

    agent = TRPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    return agent
