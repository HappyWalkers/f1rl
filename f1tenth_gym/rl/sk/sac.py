import torch
import torch.nn as nn
from absl import logging
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler


class CustomSACPolicy(GaussianMixin, Model):
    """
    Policy network for SAC. 
    This outputs the mean and log_std of a Gaussian distribution.
    """
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)  # clip_actions typically True for SAC

        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space.shape[0])  # mean
        )
        # log-std parameter for each action dimension
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))

    def compute(self, inputs, role=""):
        """
        Returns:
          - mu: mean of Gaussian (shape: [batch_size, action_dim])
          - log_std: log of std-dev (shape: [batch_size, action_dim])
          - dictionary: optional extra outputs (empty here)
        """
        states = inputs["states"]  # shape: [batch_size, obs_dim]
        mu = self.network(states)
        log_std = self.log_std.expand_as(mu)

        return mu, log_std, {}


class CustomSACCritic(DeterministicMixin, Model):
    """
    Critic (Q-function) network for SAC.
    We will build two of these: critic1 and critic2.
    Each returns a single scalar Q-value.
    """
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # For Q-network, input is [states, actions]
        input_dim = observation_space.shape[0] + action_space.shape[0]

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # single scalar output
        )

    def compute(self, inputs, role=""):
        """
        Returns:
          - Q(s, a): shape [batch_size, 1]
          - dictionary: optional extra outputs (empty here)
        """
        states = inputs["states"]        # shape: [batch_size, obs_dim]
        actions = inputs["taken_actions"]  # shape: [batch_size, act_dim]
        x = torch.cat([states, actions], dim=-1)
        q_value = self.network(x)
        return q_value, {}


def create_sac_agent(env, device):
    """
    Create and return a Soft Actor-Critic agent using custom networks.
    """
    # Replay memory (experience buffer)
    memory_size = 100000
    memory = RandomMemory(memory_size=memory_size, num_envs=1, device=device)

    # Default SAC config (can be customized)
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["batch_size"] = 128
    cfg["buffer_size"] = memory_size
    cfg["discount_factor"] = 0.99
    cfg["tau"] = 0.005  # soft update coefficient
    cfg["initial_random_steps"] = 1024  # number of initial timesteps to take random actions
    cfg["learning_starts"] = 1024
    cfg["actor_learning_rate"] = 1e-3
    cfg["critic_learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = None  # disable LR scheduler for simplicity
    cfg["learn_entropy"] = True  # SAC alpha auto-tuning
    cfg["entropy_learning_rate"] = 1e-3
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Build the actor (policy)
    actor = CustomSACPolicy(env.observation_space, env.action_space, device)

    # Build the two online critics
    critic1 = CustomSACCritic(env.observation_space, env.action_space, device)
    critic2 = CustomSACCritic(env.observation_space, env.action_space, device)

    # Build the two target critics
    # (Initialize them with the same weights as the online critics)
    target_critic1 = CustomSACCritic(env.observation_space, env.action_space, device)
    target_critic2 = CustomSACCritic(env.observation_space, env.action_space, device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    # Organize models for SKRL
    models = {
        "policy": actor,
        "critic_1": critic1,
        "critic_2": critic2,
        "target_critic_1": target_critic1,
        "target_critic_2": target_critic2
    }

    # Create the SAC agent
    sac_agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    return sac_agent
