import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
import pickle
import h5py
import argparse
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math

# Assuming stablebaseline3 and rl_env are in the path or installed
# Add f1tenth_gym/rl to sys.path if necessary
import sys
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (independent_study)
parent_dir = os.path.dirname(script_dir)
# Add the rl directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from stablebaseline3.rl import create_vec_env # Use the function from rl.py
    from pure_pursuit import PurePursuitPolicy # Import PurePursuitPolicy
    from rl_env import F110GymWrapper # Import the wrapper
    from utils import utils
    from utils.Track import Track
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure stablebaseline3, rl_env, pure_pursuit, and utils are accessible.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Parameters to predict (subset of those randomized in create_vec_env)
# Make sure these keys match the ones returned by get_env_params and randomized in create_vec_env
PARAMS_TO_PREDICT = ['mu', 'C_Sf', 'C_Sr', 'm', 'I', 'lidar_noise_stddev', 's_noise_stddev', 'ey_noise_stddev', 'vel_noise_stddev', 'yaw_noise_stddev', 'push_0_prob', 'push_2_prob']
PARAM_DIM = len(PARAMS_TO_PREDICT)

# --- Trajectory Collection ---

def collect_trajectories(num_trajectories, max_traj_len, feature_dim, save_path, num_envs=4, seed=42, map_index=63, batch_save_size=1000):
    """
    Collects trajectories from domain-randomized environments and saves them incrementally to an HDF5 file.

    Args:
        num_trajectories (int): Total number of trajectories to collect.
        max_traj_len (int): Maximum length allowed for a single trajectory.
        save_path (str): Path to save the HDF5 file.
        num_envs (int): Number of parallel environments.
        seed (int): Random seed.
        map_index (int): Index of the map to use.
        batch_save_size (int): Number of trajectories to accumulate in memory before writing to disk.
    """
    print(f"Collecting {num_trajectories} trajectories...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- HDF5 File Initialization ---
    try:
        with h5py.File(save_path, 'w') as f:
            # Create datasets with initial size 0 and unlimited max size
            f.create_dataset('params', shape=(0, PARAM_DIM), maxshape=(None, PARAM_DIM), dtype='float32', chunks=(batch_save_size, PARAM_DIM))
            # Store trajectory data flattened, along with lengths and start indices
            f.create_dataset('trajectory_data', shape=(0, feature_dim), maxshape=(None, feature_dim), dtype='float32', chunks=(batch_save_size * max_traj_len, feature_dim))
            f.create_dataset('trajectory_lengths', shape=(0,), maxshape=(None,), dtype='int64', chunks=(batch_save_size,))
            f.create_dataset('trajectory_indices', shape=(0,), maxshape=(None,), dtype='int64', chunks=(batch_save_size,))
            # Store metadata
            f.attrs['num_trajectories'] = 0
            f.attrs['feature_dim'] = feature_dim
            f.attrs['param_dim'] = PARAM_DIM
            f.attrs['param_names'] = PARAMS_TO_PREDICT

        print(f"Initialized HDF5 file at {save_path}")
    except Exception as e:
        print(f"Error initializing HDF5 file: {e}")
        return


    # --- Environment Setup ---
    # Load track once
    class Config(utils.ConfigYAML):
        sim_time_step = 0.1
        map_dir = '../f1tenth_racetracks/' # Relative to rl directory
        use_blank_map = True
        map_ext = '.png'
        map_scale = 1
        map_ind = map_index
    config = Config()
    map_info_path = os.path.join(os.path.dirname(parent_dir), 'f1tenth_racetracks', 'map_info.txt')
    map_info = np.genfromtxt(map_info_path, delimiter='|', dtype='str')
    track_dir = os.path.join(os.path.dirname(parent_dir), 'f1tenth_racetracks/')
    track, config = Track.load_map(track_dir, map_info, config.map_ind, config, scale=config.map_scale, downsample_step=1)

    base_env_kwargs = {
        'waypoints': track.waypoints,
        'map_path': config.map_dir + map_info[config.map_ind][1].split('.')[0],
        'num_agents': 1, # Collect data for single agent for simplicity
        'track': track,
    }

    vec_env = create_vec_env(
        env_kwargs=base_env_kwargs,
        seed=seed,
        num_envs=num_envs,
        use_domain_randomization=True # Crucial for collecting diverse data
    )

    # Get initial ground truth parameters for each environment
    env_params = [vec_env.env_method('get_env_params', indices=[i])[0] for i in range(num_envs)]
    current_env_params = [ {k: p[k] for k in PARAMS_TO_PREDICT} for p in env_params]

    # --- Use Pure Pursuit to collect data ---
    policies = [PurePursuitPolicy(track=vec_env.get_attr('track', indices=i)[0]) for i in range(num_envs)]

    obs = vec_env.reset()
    current_trajectories_data = [[] for _ in range(num_envs)] # Stores (s, a, ns, r) tuples
    dones = [False] * num_envs

    # In-memory buffers before writing to HDF5
    trajectory_batch = []
    params_batch = []
    total_collected = 0
    total_steps_collected = 0 # Keep track of total steps for trajectory_data index

    pbar = tqdm(total=num_trajectories)
    try: # Wrap collection loop in try...finally to ensure file closure and final write
        while total_collected < num_trajectories:
            actions = []
            for i in range(num_envs):
                single_obs = obs[i][0] if isinstance(obs[i], list) and len(obs[i]) == 1 else obs[i]
                action, _ = policies[i].predict(single_obs, deterministic=True)
                actions.append(action)
            actions = np.array(actions)

            next_obs, rewards, current_dones, infos = vec_env.step(actions) # Renamed dones to current_dones

            for i in range(num_envs):
                state = np.array(obs[i][0] if isinstance(obs[i], list) else obs[i], dtype=np.float32)
                action = np.array(actions[i], dtype=np.float32)
                next_state = np.array(next_obs[i], dtype=np.float32)
                reward = float(rewards[i])
                done = bool(current_dones[i])

                if len(current_trajectories_data[i]) < max_traj_len:
                    current_trajectories_data[i].append((state, action, next_state, reward))
                else: # Mark as done if max length reached before env done
                    done = True

                if done:
                    if len(current_trajectories_data[i]) > 1:
                        # Process finished trajectory
                        traj_np = np.array([np.concatenate((s, a, ns, np.array([r], dtype=np.float32)))
                                            for s, a, ns, r in current_trajectories_data[i]], dtype=np.float32)

                        trajectory_batch.append(traj_np)
                        params_batch.append(np.array(list(current_env_params[i].values()), dtype=np.float32))

                        total_collected += 1
                        pbar.update(1)

                        # Check if batch is full and write to HDF5
                        if len(trajectory_batch) >= batch_save_size:
                            print(f"Saving batch of {len(trajectory_batch)} trajectories...")
                            try:
                                with h5py.File(save_path, 'a') as f:
                                    current_num_traj = f.attrs['num_trajectories']
                                    new_total_traj = current_num_traj + len(trajectory_batch)

                                    # Prepare data for appending
                                    lengths_to_append = np.array([len(t) for t in trajectory_batch], dtype=np.int64)
                                    indices_to_append = np.cumsum(np.concatenate(([total_steps_collected], lengths_to_append[:-1]))).astype(np.int64)
                                    data_to_append = np.concatenate(trajectory_batch, axis=0)
                                    params_to_append = np.array(params_batch, dtype=np.float32)

                                    # Resize datasets
                                    f['params'].resize((new_total_traj, PARAM_DIM))
                                    f['trajectory_data'].resize((total_steps_collected + data_to_append.shape[0], feature_dim))
                                    f['trajectory_lengths'].resize((new_total_traj,))
                                    f['trajectory_indices'].resize((new_total_traj,))

                                    # Append data
                                    f['params'][current_num_traj:new_total_traj, :] = params_to_append
                                    f['trajectory_data'][total_steps_collected : total_steps_collected + data_to_append.shape[0], :] = data_to_append
                                    f['trajectory_lengths'][current_num_traj:new_total_traj] = lengths_to_append
                                    f['trajectory_indices'][current_num_traj:new_total_traj] = indices_to_append

                                    # Update counts
                                    total_steps_collected += data_to_append.shape[0]
                                    f.attrs['num_trajectories'] = new_total_traj

                                # Clear buffers
                                trajectory_batch = []
                                params_batch = []
                                print(f"Batch saved. Total trajectories: {new_total_traj}")
                            except Exception as e:
                                print(f"Error writing batch to HDF5: {e}")
                                # Consider how to handle write errors (e.g., retry, stop)


                        if total_collected >= num_trajectories:
                            break # Stop outer loop if enough trajectories collected

                    # Reset the specific environment and get its new parameters
                    current_trajectories_data[i] = []
                    policies[i].reset() # Reset corresponding policy
                    env_params[i] = vec_env.env_method('get_env_params', indices=[i])[0]
                    current_env_params[i] = {k: env_params[i][k] for k in PARAMS_TO_PREDICT}
                    # next_obs[i] already contains the reset observation

            obs = next_obs # Move to next state
            if total_collected >= num_trajectories:
                break

    finally: # Ensure final write and cleanup
        pbar.close()
        # Write any remaining trajectories in the buffer
        if trajectory_batch:
            print(f"Saving final batch of {len(trajectory_batch)} trajectories...")
            try:
                 with h5py.File(save_path, 'a') as f:
                    current_num_traj = f.attrs['num_trajectories']
                    new_total_traj = current_num_traj + len(trajectory_batch)

                    lengths_to_append = np.array([len(t) for t in trajectory_batch], dtype=np.int64)
                    indices_to_append = np.cumsum(np.concatenate(([total_steps_collected], lengths_to_append[:-1]))).astype(np.int64)
                    data_to_append = np.concatenate(trajectory_batch, axis=0)
                    params_to_append = np.array(params_batch, dtype=np.float32)

                    f['params'].resize((new_total_traj, PARAM_DIM))
                    f['trajectory_data'].resize((total_steps_collected + data_to_append.shape[0], feature_dim))
                    f['trajectory_lengths'].resize((new_total_traj,))
                    f['trajectory_indices'].resize((new_total_traj,))

                    f['params'][current_num_traj:new_total_traj, :] = params_to_append
                    f['trajectory_data'][total_steps_collected : total_steps_collected + data_to_append.shape[0], :] = data_to_append
                    f['trajectory_lengths'][current_num_traj:new_total_traj] = lengths_to_append
                    f['trajectory_indices'][current_num_traj:new_total_traj] = indices_to_append

                    total_steps_collected += data_to_append.shape[0]
                    f.attrs['num_trajectories'] = new_total_traj
                 print(f"Final batch saved. Total trajectories: {new_total_traj}")
            except Exception as e:
                 print(f"Error writing final batch to HDF5: {e}")

        print(f"Finished collecting trajectories. Saved to {save_path}")
        vec_env.close()
        # No return value needed as data is saved to disk


# --- Dataset ---

class TrajectoryDataset(Dataset):
    def __init__(self, h5_path, input_mean=None, input_std=None, output_mean=None, output_std=None):
        self.h5_path = h5_path
        self.h5_file = None # File handle, opened in __getitem__ or worker_init_fn if using DataLoader workers

        # Read metadata without keeping file open
        with h5py.File(self.h5_path, 'r') as f:
            self.num_trajectories = f.attrs['num_trajectories']
            self.feature_dim = f.attrs['feature_dim']
            self.param_dim = f.attrs['param_dim']
            # self.param_names = f.attrs['param_names'] # Optional: load if needed

        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

        # Check if stats are provided
        self.normalize_input = self.input_mean is not None and self.input_std is not None
        self.normalize_output = self.output_mean is not None and self.output_std is not None

        if self.normalize_input:
            print("Input normalization enabled.")
            # Move stats to CPU if they aren't already, as normalization happens before moving to device
            if isinstance(self.input_mean, torch.Tensor): self.input_mean = self.input_mean.cpu()
            if isinstance(self.input_std, torch.Tensor): self.input_std = self.input_std.cpu()
        if self.normalize_output:
            print("Output normalization enabled.")
            if isinstance(self.output_mean, torch.Tensor): self.output_mean = self.output_mean.cpu()
            if isinstance(self.output_std, torch.Tensor): self.output_std = self.output_std.cpu()


        # sequences and params are NOT loaded here anymore

    def _open_file(self):
        """Opens the HDF5 file if not already open."""
        if self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.h5_path, 'r')
            except Exception as e:
                print(f"Error opening HDF5 file in getitem: {e}")
                raise # Re-raise the exception

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        self._open_file() # Ensure file is open

        try:
            # Read data for the specific index
            target_params_np = self.h5_file['params'][idx]
            length = self.h5_file['trajectory_lengths'][idx]
            start_idx = self.h5_file['trajectory_indices'][idx]
            sequence_np = self.h5_file['trajectory_data'][start_idx : start_idx + length]

            # Convert to tensors
            sequence = torch.tensor(sequence_np, dtype=torch.float32)
            target_params = torch.tensor(target_params_np, dtype=torch.float32)

            # Normalize input sequence if stats are available
            if self.normalize_input and sequence.numel() > 0:
                sequence = (sequence - self.input_mean) / (self.input_std + 1e-8) # Add epsilon for stability

            # Normalize output target if stats are available
            if self.normalize_output:
                target_params = (target_params - self.output_mean) / (self.output_std + 1e-8)

            return sequence, target_params

        except Exception as e:
            print(f"Error reading data for index {idx} from {self.h5_path}: {e}")
            # Return dummy data or raise error? Raising error might be better.
            raise IndexError(f"Failed to retrieve data for index {idx}") from e

    def close(self):
        """Closes the HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        """Ensures the file is closed when the dataset object is deleted."""
        self.close()


def collate_fn(batch):
    """Pad sequences for batching."""
    sequences, params = zip(*batch)
    # Get sequence lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    # Stack parameters
    target_params = torch.stack(params)
    return padded_sequences, lengths, target_params


# --- Models ---

class LSTMParamPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, param_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True) # Use bidirectional LSTM
        lstm_output_dim = hidden_dim * 2 # Bidirectional
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, lstm_output_dim // 8),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 8, lstm_output_dim // 16),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 16, lstm_output_dim // 32),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 32, lstm_output_dim // 64),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 64, param_dim)
        )

    def forward(self, x, lengths):
        # Pack sequence
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # We use the final hidden state of the last layer
        # hidden is (num_layers * num_directions, batch, hidden_dim)
        # Concatenate the final forward and backward hidden states
        hidden_forward = hidden[-2, :, :] # Last layer, forward
        hidden_backward = hidden[-1, :, :] # Last layer, backward
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Pass through fully connected layer
        output = self.fc(hidden_cat)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embedding_dim) for transformer
        x = x.permute(1, 0, 2) # Change to (seq_len, batch, feature_dim)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2) # Change back to (batch, seq_len, feature_dim)

class TransformerParamPredictor(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, dim_feedforward, param_dim, dropout=0.1, max_len=5000): # Increased max_len
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(model_dim, param_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_proj.weight.data.uniform_(-initrange, initrange)
        self.input_proj.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, src, src_key_padding_mask):
        # src shape: (batch_size, seq_len, input_dim)
        # src_key_padding_mask shape: (batch_size, seq_len) - True where padded

        src = self.input_proj(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src) # Needs (batch, seq_len, model_dim)

        # TransformerEncoder expects mask where True indicates masking
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # output shape: (batch_size, seq_len, model_dim)

        # Use the output corresponding to the first token (or mean pooling)
        # Here we use mean pooling over non-padded elements
        # Invert mask for selecting non-padded elements
        mask_inv = ~src_key_padding_mask # (batch, seq_len), True where NOT padded
        # Add feature dimension to mask
        mask_inv = mask_inv.unsqueeze(-1).float() # (batch, seq_len, 1)
        # Apply mask and sum
        masked_output = output * mask_inv
        summed_output = masked_output.sum(dim=1) # (batch, model_dim)
        # Count non-padded elements per sequence
        non_padded_count = mask_inv.sum(dim=1) # (batch, 1)
        non_padded_count = torch.clamp(non_padded_count, min=1.0) # Avoid division by zero
        # Compute mean
        mean_output = summed_output / non_padded_count # (batch, model_dim)

        # Final prediction layer
        prediction = self.fc(mean_output) # (batch, param_dim)
        return prediction


# --- Training ---

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_type='lstm', model_save_path='best_model.pth'):
    print(f"Training {model_type} model...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        # Use tqdm with postfix to display loss
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in tqdm_bar:
            sequences, lengths, targets = batch
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            if model_type == 'lstm':
                outputs = model(sequences, lengths)
            elif model_type == 'transformer':
                 # Create mask: True for padded elements
                mask = (sequences.sum(dim=2) == 0) # Assuming 0 padding was used
                outputs = model(sequences, mask)
            else:
                raise ValueError("Invalid model_type")

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()

            batch_loss = loss.item()
            epoch_train_loss += batch_loss

            # Update tqdm progress bar with current batch loss
            tqdm_bar.set_postfix(loss=f"{batch_loss:.4f}")
            
            torch.cuda.empty_cache()
        tqdm_bar.close() # Close the inner tqdm bar
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                sequences, lengths, targets = batch
                sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)

                if model_type == 'lstm':
                    outputs = model(sequences, lengths)
                elif model_type == 'transformer':
                    mask = (sequences.sum(dim=2) == 0)
                    outputs = model(sequences, mask)
                else:
                    raise ValueError("Invalid model_type")

                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path} with Val Loss: {best_val_loss:.4f}")

    print("Training finished.")
    return train_losses, val_losses

# --- Evaluation ---

def evaluate_model(model, data_loader, criterion, output_mean, output_std, model_type='lstm'):
    print(f"Evaluating {model_type} model...")
    model.eval()
    total_loss = 0.0
    all_preds_normalized = []
    all_targets_normalized = []
    all_targets_original = [] # Store original scale targets for MAE calculation

    output_mean_cpu = output_mean.cpu()
    output_std_cpu = output_std.cpu()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            sequences, lengths, targets_normalized = batch # targets are already normalized by Dataset
            sequences, targets_normalized = sequences.to(DEVICE), targets_normalized.to(DEVICE)

            # Store original targets (denormalize the normalized targets from batch)
            targets_orig = targets_normalized.cpu() * output_std_cpu + output_mean_cpu
            all_targets_original.append(targets_orig.numpy())
            all_targets_normalized.append(targets_normalized.cpu().numpy())

            if model_type == 'lstm':
                outputs_normalized = model(sequences, lengths)
            elif model_type == 'transformer':
                mask = (sequences.sum(dim=2) == 0)
                outputs_normalized = model(sequences, mask)
            else:
                raise ValueError("Invalid model_type")

            # Loss is calculated on normalized values
            loss = criterion(outputs_normalized, targets_normalized)
            total_loss += loss.item()
            all_preds_normalized.append(outputs_normalized.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation Loss (on normalized data): {avg_loss:.4f}")

    all_preds_normalized = np.concatenate(all_preds_normalized, axis=0)
    all_targets_original = np.concatenate(all_targets_original, axis=0)

    # Denormalize predictions for reporting/plotting
    all_preds_denormalized = all_preds_normalized * output_std_cpu.numpy() + output_mean_cpu.numpy()

    return avg_loss, all_preds_denormalized, all_targets_original

# --- Visualization ---

def plot_loss_curves(train_losses, val_losses, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def plot_predictions(predictions, targets, output_mean, output_std, save_prefix='pred_vs_actual'):
    """Plots predicted vs actual values.

    Args:
        predictions: Denormalized predictions from the model.
        targets: Original, unnormalized target values.
        output_mean: Mean used for normalization (NumPy array).
        output_std: Std dev used for normalization (NumPy array).
        save_prefix: Prefix for saving plot files.
    """
    num_params = predictions.shape[1]
    param_names = PARAMS_TO_PREDICT

    # Ensure predictions and targets are NumPy arrays
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    for i in range(num_params):
        plt.figure(figsize=(8, 8))
        # Use denormalized predictions and original targets for plotting range
        min_val = min(predictions[:, i].min(), targets[:, i].min()) * 0.9 # Add margin
        max_val = max(predictions[:, i].max(), targets[:, i].max()) * 1.1 # Add margin
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal') # y=x line
        plt.xlabel(f"Actual {param_names[i]}")
        plt.ylabel(f"Predicted {param_names[i]}")
        plt.title(f"Predicted vs Actual for {param_names[i]}")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.grid(True)
        # plt.axis('equal') # Might make plot too small if ranges differ a lot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{save_prefix}_{param_names[i]}.png")
        plt.close()
    print(f"Prediction plots saved with prefix: {save_prefix}")

# --- Helper for normalization statistics ---
def load_or_calc_normalization_stats(data_file, feature_dim, param_dim, train_indices, seed):
    """Loads normalization stats from HDF5 if present, otherwise calculates them from training data and stores them."""
    # Attempt to load existing stats
    with h5py.File(data_file, 'a') as f:
        if all(name in f for name in ['input_mean', 'input_std', 'output_mean', 'output_std']):
            input_mean = torch.tensor(f['input_mean'][:], dtype=torch.float32)
            input_std = torch.tensor(f['input_std'][:], dtype=torch.float32)
            output_mean = torch.tensor(f['output_mean'][:], dtype=torch.float32)
            output_std = torch.tensor(f['output_std'][:], dtype=torch.float32)
            print("Loaded normalization stats from HDF5.")
            return input_mean, input_std, output_mean, output_std

    # Stats not found; calculate
    print("Normalization stats not found in HDF5, calculating from training data...")
    input_sum = torch.zeros(feature_dim, dtype=torch.float64)
    input_sq = torch.zeros(feature_dim, dtype=torch.float64)
    count_in = 0
    output_sum = torch.zeros(param_dim, dtype=torch.float64)
    output_sq = torch.zeros(param_dim, dtype=torch.float64)
    count_out = 0

    with h5py.File(data_file, 'r') as f:
        d_p = f['params']
        d_data = f['trajectory_data']
        d_len = f['trajectory_lengths']
        d_idx = f['trajectory_indices']
        for idx in tqdm(train_indices, desc="Calculating Stats"):
            length = d_len[idx]
            start = d_idx[idx]
            seq_np = d_data[start:start+length]
            if length > 0:
                seq_t = torch.tensor(seq_np, dtype=torch.float64)
                p_t = torch.tensor(d_p[idx], dtype=torch.float64)
                input_sum += seq_t.sum(dim=0)
                input_sq += (seq_t**2).sum(dim=0)
                count_in += length
                output_sum += p_t
                output_sq += p_t**2
                count_out += 1

    if count_in == 0 or count_out == 0:
        raise ValueError("No valid data for normalization statistics.")

    input_mean = (input_sum / count_in).float()
    input_std = torch.sqrt(torch.clamp((input_sq / count_in) - input_mean.double()**2, min=0.0)).float()
    input_std[input_std < 1e-8] = 1.0
    output_mean = (output_sum / count_out).float()
    output_std = torch.sqrt(torch.clamp((output_sq / count_out) - output_mean.double()**2, min=0.0)).float()
    output_std[output_std < 1e-8] = 1.0

    # Store stats in HDF5
    with h5py.File(data_file, 'a') as f:
        f.create_dataset('input_mean', data=input_mean.numpy(), dtype='float32')
        f.create_dataset('input_std', data=input_std.numpy(), dtype='float32')
        f.create_dataset('output_mean', data=output_mean.numpy(), dtype='float32')
        f.create_dataset('output_std', data=output_std.numpy(), dtype='float32')
        print("Saved normalization stats to HDF5.")

    return input_mean, input_std, output_mean, output_std

# --- Main Execution ---

def main(args):
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Step 1: Collect Trajectories (or load if exists) ---
    # Define data directory relative to the script location
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True) # Ensure data directory exists
    filename = f"trajectories_dr_{args.num_trajectories}_{args.max_traj_len}_seed{args.seed}.h5"
    data_file = os.path.join(data_dir, filename)

    print(f"Data file path: {data_file}")

    if not os.path.exists(data_file) or args.force_collect:
        collect_trajectories(
            args.num_trajectories,
            args.max_traj_len,
            args.feature_dim,
            data_file,
            num_envs=args.num_envs,
            seed=args.seed,
            map_index=args.map_index,
            batch_save_size=args.batch_save_size
        )

    # --- Step 2a: Load or calculate normalization statistics ---
    # Determine split sizes and training indices
    try:
        with h5py.File(data_file, 'r') as f:
            total_size = f.attrs['num_trajectories']
    except Exception as e:
        print(f"Error reading total size from HDF5 file {data_file}: {e}")
        return

    if total_size < 3:
        print("Error: Need at least 3 trajectories for train/val/test split.")
        return

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    if train_size + val_size + test_size != total_size:
        train_size = total_size - val_size - test_size

    if train_size <= 0:
        print(f"Error: Not enough data for training split ({train_size}/{total_size}). Cannot calculate normalization stats.")
        return

    indices = list(range(total_size))
    random.Random(args.seed).shuffle(indices)
    train_indices = indices[:train_size]

    # Load or calculate stats from HDF5
    input_mean, input_std, output_mean, output_std = load_or_calc_normalization_stats(
        data_file, args.feature_dim, PARAM_DIM, train_indices, args.seed
    )
    print("Input Mean shape:", input_mean.shape)
    print("Output Mean shape:", output_mean.shape)

    # --- Step 2b: Create Dataset and Dataloaders ---
    dataset = TrajectoryDataset(data_file, input_mean, input_std, output_mean, output_std)

    # Input dimension determined during dataset creation or from metadata
    input_dim = dataset.feature_dim
    print(f"Input dimension (state + action + next_state + reward): {input_dim}")
    print(f"Parameter dimension: {PARAM_DIM}")

    # Split dataset using indices determined earlier
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    print(f"Dataset size: {total_size}")
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}")

    if not train_indices or not val_indices or not test_indices:
         print("Error: One or more splits are empty.")
         # Handle minimal split case if necessary, similar to before
         if total_size >= 3:
             print("Creating minimal split: Train=1, Val=1, Test=1")
             train_indices = indices[:1]
             val_indices = indices[1:2]
             test_indices = indices[2:3]
             # Ensure dataset is closed if re-instantiating subsets, or pass indices directly
             train_dataset = Subset(dataset, train_indices)
             val_dataset = Subset(dataset, val_indices)
             test_dataset = Subset(dataset, test_indices)
         else:
             print("Need at least 3 trajectories for a minimal split.")
             dataset.close() # Close the dataset file handle
             return # Exit if dataset is too small
    else:
        # Use Subset with the indices
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

    # Define worker_init_fn to handle HDF5 file opening in each worker process
    def worker_init_fn(worker_id):
        # Ensure the dataset object in the worker process opens its own file handle
        # This relies on Subset passing the underlying dataset object
        worker_info = torch.utils.data.get_worker_info()
        dataset_worker = worker_info.dataset.dataset # Access the original TrajectoryDataset from Subset
        dataset_worker._open_file() # Open the HDF5 file for this worker

    # Close the main process's file handle before starting DataLoader if using workers
    # dataset.close() # Let workers manage their own handles

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False, worker_init_fn=worker_init_fn if args.num_workers > 0 else None, persistent_workers=True if args.num_workers > 0 else False) # Add worker_init_fn and persistent_workers
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False, worker_init_fn=worker_init_fn if args.num_workers > 0 else None, persistent_workers=True if args.num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False, worker_init_fn=worker_init_fn if args.num_workers > 0 else None, persistent_workers=True if args.num_workers > 0 else False)


    # --- Step 3: Train and Evaluate Models ---
    criterion = nn.MSELoss()
    output_mean_np = output_mean.cpu().numpy() # Get NumPy versions for plotting/MAE
    output_std_np = output_std.cpu().numpy()
    if 'lstm' in args.models_to_train:
        lstm_model = LSTMParamPredictor(input_dim, args.hidden_dim, PARAM_DIM, args.num_layers, args.dropout).to(DEVICE)
        lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=args.lr, weight_decay=1e-4)
        lstm_train_losses, lstm_val_losses = train_model(
            lstm_model, train_loader, val_loader, criterion, lstm_optimizer, args.epochs, 'lstm', 'best_lstm_model.pth'
        )
        plot_loss_curves(lstm_train_losses, lstm_val_losses, 'lstm_loss_curve.png')
        
        print("--- LSTM Evaluation ---")
        lstm_model.load_state_dict(torch.load('best_lstm_model.pth', map_location=DEVICE))
        # Pass normalization stats for denormalization during evaluation
        lstm_test_loss, lstm_preds_denorm, lstm_targets_orig = evaluate_model(lstm_model, test_loader, criterion, output_mean, output_std, 'lstm')
        # Pass original targets and denormalized predictions for plotting
        plot_predictions(lstm_preds_denorm, lstm_targets_orig, output_mean_np, output_std_np, 'lstm_pred_vs_actual')
        # Calculate MAE using denormalized predictions and original targets
        lstm_mae = np.mean(np.abs(lstm_preds_denorm - lstm_targets_orig), axis=0)
        print("LSTM Mean Absolute Error per parameter (original scale):")
        for name, mae in zip(PARAMS_TO_PREDICT, lstm_mae):
            print(f"  {name}: {mae:.4f}")

    if 'transformer' in args.models_to_train:
        transformer_model = TransformerParamPredictor(input_dim, args.tf_model_dim, args.tf_nhead, args.tf_encoder_layers, args.tf_dim_feedforward, PARAM_DIM, args.dropout, args.max_traj_len).to(DEVICE) # Make sure max_len matches data potential
        transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=args.lr, weight_decay=1e-4)
        tf_train_losses, tf_val_losses = train_model(
            transformer_model, train_loader, val_loader, criterion, transformer_optimizer, args.epochs, 'transformer', 'best_transformer_model.pth'
        )
        plot_loss_curves(tf_train_losses, tf_val_losses, 'transformer_loss_curve.png')
        
        print("--- Transformer Evaluation ---")
        transformer_model.load_state_dict(torch.load('best_transformer_model.pth', map_location=DEVICE))
        # Pass normalization stats for denormalization during evaluation
        tf_test_loss, tf_preds_denorm, tf_targets_orig = evaluate_model(transformer_model, test_loader, criterion, output_mean, output_std, 'transformer')
        # Pass original targets and denormalized predictions for plotting
        plot_predictions(tf_preds_denorm, tf_targets_orig, output_mean_np, output_std_np, 'transformer_pred_vs_actual')
        # Calculate MAE using denormalized predictions and original targets
        tf_mae = np.mean(np.abs(tf_preds_denorm - tf_targets_orig), axis=0)
        print("Transformer Mean Absolute Error per parameter (original scale):")
        for name, mae in zip(PARAMS_TO_PREDICT, tf_mae):
            print(f"  {name}: {mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models to predict environment parameters from trajectories.")
    parser.add_argument('--num_trajectories', type=int, default=128_000, help='Number of trajectories to collect/load.')
    parser.add_argument('--max_traj_len', type=int, default=256, help='Maximum length of each trajectory.')
    parser.add_argument('--num_envs', type=int, default=24, help='Number of parallel environments for data collection.')
    parser.add_argument('--map_index', type=int, default=63, help='Index of the map to use (determines track).')
    parser.add_argument('--force_collect', action='store_true', help='Force data collection even if file exists.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--batch_save_size', type=int, default=1024, help='Number of trajectories to save to disk at a time during collection.') # Added
    parser.add_argument('--num_workers', type=int, default=24, help='Number of DataLoader workers.') # Added
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='LSTM hidden dimension.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of LSTM layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--tf_model_dim', type=int, default=1024, help='Transformer model dimension (embedding size).')
    parser.add_argument('--tf_nhead', type=int, default=8, help='Number of attention heads in Transformer.')
    parser.add_argument('--tf_encoder_layers', type=int, default=8, help='Number of Transformer encoder layers.')
    parser.add_argument('--tf_dim_feedforward', type=int, default=1024, help='Dimension of feedforward network in Transformer.')
    parser.add_argument('--models_to_train', nargs='+', default=['lstm'], choices=['lstm', 'transformer'], help='Which models to train.')
    parser.add_argument('--feature_dim', type=int, default=1084+2+1084+1, help='Feature dimension for normalization calculation if needed.') # Add feature dim arg

    args = parser.parse_args()
    main(args)
