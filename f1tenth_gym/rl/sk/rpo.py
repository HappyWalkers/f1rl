import gym
import torch
import torch.nn as nn

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer

# For optional logging
from absl import logging


class RecurrentPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)

        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.hidden_size = hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.obs_dim, hidden_size=self.hidden_size, batch_first=True)

        # Fully connected layer after LSTM
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.act_dim)
        )

        # Log std parameter (initialized to 0)
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

    def compute(self, inputs, role=""):
        """
        Inputs is a dict that contains:
          - "states": shape [batch_size, sequence_length, obs_dim]
          - "rnn_initial_states": (h0, c0) for LSTM, each shape [num_layers, batch_size, hidden_size]
          - "rnn_sequence_lengths": the sequence lengths for each item in the batch
        We must return: (mu, log_std, { ... }) for GaussianMixin
        """
        states = inputs["states"]  # (batch, seq, obs_dim)
        seq_lengths = inputs["rnn_sequence_lengths"]  # (batch,)

        # Pack the sequences for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            states, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Retrieve initial states (h0, c0)
        rnn_initial_states = inputs.get("rnn_initial_states", None)
        if rnn_initial_states is None:
            # If no hidden states are provided, create zeros
            batch_size = states.shape[0]
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
            rnn_initial_states = (h0, c0)

        # Forward LSTM
        packed_output, (h_n, c_n) = self.lstm(packed, rnn_initial_states)

        # Unpack
        unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Because each sequence might have different length, we want the last output in each sequence
        # There's a handy way to do that with the sequence lengths:
        last_outputs = []
        for i, length in enumerate(seq_lengths):
            # length - 1 because of zero-based index
            last_outputs.append(unpacked_output[i, length - 1, :])
        # [batch_size, hidden_size]
        last_outputs = torch.stack(last_outputs, dim=0)

        # Fully connected
        mu = self.fc(last_outputs)  # shape: [batch_size, act_dim]
        log_std = self.log_std.expand_as(mu)  # shape: [batch_size, act_dim]

        # Logging (optional)
        logging.info(f"(RecurrentPolicy) mu: {mu}")

        # Return new hidden states so skrl can continue the sequence in the next step
        extra_info = {
            "rnn_states": (h_n, c_n)  # the updated hidden states after LSTM
        }

        return mu, log_std, extra_info

class RecurrentValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=128):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.obs_dim = observation_space.shape[0]
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=self.obs_dim, hidden_size=self.hidden_size, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def compute(self, inputs, role=""):
        """
        Similarly, we get a dict with "states", "rnn_initial_states", "rnn_sequence_lengths".
        We must return (value, extra_info).
        """
        states = inputs["states"]  # [batch_size, sequence_length, obs_dim]
        seq_lengths = inputs["rnn_sequence_lengths"]  # [batch_size]

        # Pack sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            states, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Retrieve initial states
        rnn_initial_states = inputs.get("rnn_initial_states", None)
        if rnn_initial_states is None:
            batch_size = states.shape[0]
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
            rnn_initial_states = (h0, c0)

        # Forward LSTM
        packed_output, (h_n, c_n) = self.lstm(packed, rnn_initial_states)
        unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Get last output in each sequence
        last_outputs = []
        for i, length in enumerate(seq_lengths):
            last_outputs.append(unpacked_output[i, length - 1, :])
        last_outputs = torch.stack(last_outputs, dim=0)

        # Final value
        value = self.fc(last_outputs)  # shape: [batch_size, 1]

        logging.info(f"(RecurrentValue) value: {value}")

        extra_info = {
            "rnn_states": (h_n, c_n)
        }
        return value, extra_info



def create_rpo_agent(env, device):
    # Number of timesteps (and thus sequence length) per rollout
    rollouts = 1024

    # We store entire sequences in memory with the "sequence_length" key
    memory = RandomMemory(memory_size=rollouts, num_envs=1, device=device, replacement=False)

    # Default PPO config
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts
    cfg["learning_epochs"] = 10
    cfg["mini_batches"] = 32
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3

    # *** Recurrent-specific configuration ***
    cfg["rewards_shaper"] = None  # Optional function to reshape rewards
    cfg["recurrent"] = True       # Enable recurrent support
    cfg["sequence_length"] = 16    # LSTM sequence length for training
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0

    # Optional normalizers
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0], "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Create models
    models = {
        "policy": RecurrentPolicy(env.observation_space, env.action_space, device),
        "value": RecurrentValue(env.observation_space, env.action_space, device)
    }

    # Create PPO agent (Recurrent PPO)
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )

    return agent
