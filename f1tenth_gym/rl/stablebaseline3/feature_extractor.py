import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Type, Union, Callable, Optional, Any
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from absl import logging


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine transformation to input features based on conditioning signals.
    FiLM(x) = gamma * x + beta, where gamma and beta are derived from the conditioning signal.
    """
    def __init__(self, feature_dim, condition_dim, hidden_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        hidden_dim = hidden_dim or condition_dim * 2
        
        # Network to generate gamma and beta from conditioning signal
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2)  # Outputs gamma and beta
        )
        
    def forward(self, x, condition):
        # Generate gamma and beta from conditioning signal
        film_params = self.condition_net(condition)
        
        # Split into gamma and beta
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # Apply feature-wise affine transformation
        # Adding 1 to gamma for better gradient flow (now it modulates around 1 instead of 0)
        return (1 + gamma) * x + beta


class ResidualBlock(nn.Module):
    """
    A residual block with skip connections.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, use_layer_norm=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        
        self.use_layer_norm = use_layer_norm
        self.same_dim = (in_features == out_features)
        
        # Main branch
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_features))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_features, out_features))
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_features))
        
        self.main_branch = nn.Sequential(*layers)
        
        # Skip connection with projection if needed
        self.projection = None
        if not self.same_dim:
            self.projection = nn.Linear(in_features, out_features)
            
        # Final activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.main_branch(x)
        
        # Skip connection with optional projection
        if not self.same_dim:
            identity = self.projection(identity)
            
        # Add skip connection
        out += identity
        
        # Apply activation
        out = self.relu(out)
        
        return out


class F1TenthFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the F1Tenth environment.
    
    This network separately processes state features and lidar scans, and conditions them
    with environment parameters using FiLM (Feature-wise Linear Modulation).
    """
    @staticmethod
    def _init_weights(m):
        # Orthogonal initialization with ReLU gain and zero bias, matching SB3 defaults
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # Uncomment for debugging initialization:
            # print(f"Initialized Linear layer with shape: {m.weight.shape}")

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1024,
        state_dim: int = 4,
        lidar_dim: int = 1080,
        param_dim: int = 12, 
        include_params: bool = True,
        include_lidar: bool = True
    ):
        # Determine whether the observation includes parameters based on its dimension
        self.state_dim = state_dim  # 4 state components: [s, ey, vel, yaw_angle]
        self.lidar_dim = lidar_dim if include_lidar else 0  # LiDAR points
        self.param_dim = param_dim  # Environment parameters
        self.include_params = include_params
        self.include_lidar = include_lidar
        
        # The expected observation dimension with parameters included
        expected_dim_with_params = state_dim + self.lidar_dim + param_dim
        expected_dim_without_params = state_dim + self.lidar_dim
        
        # Check if observation space has the expected shape
        obs_shape = observation_space.shape[0]
        if include_params:
            assert obs_shape == expected_dim_with_params, f"Expected observation dimension {expected_dim_with_params}, got {obs_shape}"
        else:
            assert obs_shape == expected_dim_without_params, f"Expected observation dimension {expected_dim_without_params}, got {obs_shape}"
            
        super().__init__(observation_space, features_dim)
        
        # Define initial feature dimensions
        state_initial_dim = 128
        lidar_initial_dim = 1024
        
        # Initial processing of state features
        self.state_initial = nn.Sequential(
            nn.Linear(state_dim, state_initial_dim),
            nn.LayerNorm(state_initial_dim),
            nn.ReLU(),
        )
        
        # Initial processing of LiDAR scan (only if lidar is included)
        if self.include_lidar:
            self.lidar_initial = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.lidar_dim, lidar_initial_dim),
                nn.LayerNorm(lidar_initial_dim),
                nn.ReLU(),
            )
        else:
            # No lidar processing needed
            self.lidar_initial = None
            lidar_initial_dim = 0  # Set to 0 for dimension calculations
        
        # Parameter conditioning network (if parameters are included)
        if include_params:
            # Initial parameter processing
            self.param_initial = nn.Sequential(
                nn.Linear(param_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                ResidualBlock(128, 256, 256),
            )
            
            # FiLM layers for conditioning state and lidar features
            self.state_film = FiLMLayer(state_initial_dim, 256)
            if self.include_lidar:
                self.lidar_film = FiLMLayer(lidar_initial_dim, 256)
        
        # Residual blocks for state processing after conditioning
        self.state_residual = nn.Sequential(
            ResidualBlock(state_initial_dim, 256, 512),
                ResidualBlock(512, 1024, 1024),
                ResidualBlock(1024, 1024, 1024),
            )
        
        # Residual blocks for lidar processing after conditioning
        if self.include_lidar:
            self.lidar_residual = nn.Sequential(
                ResidualBlock(lidar_initial_dim, 1024, 1024),
                ResidualBlock(1024, 1024, 1024),
                ResidualBlock(1024, 1024, 1024),
            )
        else:
            self.lidar_residual = None
        
        # Combined dimension from all branches
        combined_dim = 1024  # state
        if self.include_lidar:
            combined_dim += 1024  # + lidar
        
        # Final layers to combine all features with residual connections
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
            nn.Linear(1024, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

        # Initialize feature extractor weights using the orthogonal initializer
        logging.info("Initializing feature extractor weights")
        linear_layer_count = 0
        
        def count_linear_layers(module):
            nonlocal linear_layer_count
            if isinstance(module, nn.Linear):
                linear_layer_count += 1
                
        # Count linear layers before initialization
        self.state_initial.apply(count_linear_layers)
        if self.lidar_initial:
            self.lidar_initial.apply(count_linear_layers)
        self.state_residual.apply(count_linear_layers)
        if self.include_lidar:
            self.lidar_residual.apply(count_linear_layers)
        if include_params:
            self.param_initial.apply(count_linear_layers)
            self.state_film.apply(count_linear_layers)
            if self.include_lidar:
                self.lidar_film.apply(count_linear_layers)
        self.combined_net.apply(count_linear_layers)
        
        logging.info(f"Total Linear layers to initialize: {linear_layer_count}")
        
        # Apply initialization
        self.state_initial.apply(self._init_weights)
        if self.lidar_initial:
            self.lidar_initial.apply(self._init_weights)
        self.state_residual.apply(self._init_weights)
        if self.include_lidar:
            self.lidar_residual.apply(self._init_weights)
        if include_params:
            self.param_initial.apply(self._init_weights)
            self.state_film.apply(self._init_weights)
            if self.include_lidar:
                self.lidar_film.apply(self._init_weights)
        self.combined_net.apply(self._init_weights)
        
        logging.info("Feature extractor weights initialized")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract different components from observation
        state_features = observations[:, :self.state_dim]
        
        # Initial processing
        state_output = self.state_initial(state_features)
        
        if self.include_lidar:
            lidar_features = observations[:, self.state_dim:self.state_dim + self.lidar_dim]
            lidar_output = self.lidar_initial(lidar_features)
        
        if self.include_params:
            # Extract and process environment parameters
            param_features = observations[:, self.state_dim + self.lidar_dim:]
            param_output = self.param_initial(param_features)
            
            # Apply FiLM conditioning - parameters modulate state and lidar features
            state_output = self.state_film(state_output, param_output)
            if self.include_lidar:
                lidar_output = self.lidar_film(lidar_output, param_output)
        
        # Apply residual blocks after conditional modulation
        state_output = self.state_residual(state_output)
        
        if self.include_lidar:
            lidar_output = self.lidar_residual(lidar_output)
            # Combine modulated features
            combined_features = torch.cat([state_output, lidar_output], dim=1)
        else:
            # Only state features
            combined_features = state_output
        
        # Final processing
        output = self.combined_net(combined_features)
        return output


class MoEFeaturesExtractor(BaseFeaturesExtractor):
    """
    Mixture of Experts (MoE) feature extractor for the F1Tenth environment.

    This network processes state and lidar features, and uses a gating network
    (conditioned on environment parameters if available) to weigh multiple expert networks.
    """
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1024, # Default features_dim
        state_dim: int = 4,
        lidar_dim: int = 1080,
        param_dim: int = 12,
        include_params: bool = True,
        include_lidar: bool = True,
        num_experts: int = 24,
        state_proc_dim: int = 64,
        lidar_proc_dim: int = 256,
        expert_hidden_dim: int = 1024,
        gate_hidden_dim: int = 64
    ):
        super().__init__(observation_space, features_dim)

        self.state_dim = state_dim
        self.lidar_dim = lidar_dim if include_lidar else 0
        self.param_dim = param_dim
        self.include_params = include_params
        self.include_lidar = include_lidar
        self.num_experts = num_experts

        # Verify observation space shape
        expected_dim = state_dim + self.lidar_dim
        if include_params:
            expected_dim += param_dim
        assert observation_space.shape[0] == expected_dim, \
            f"Expected observation dimension {expected_dim}, got {observation_space.shape[0]}"

        # Initial processing for state features
        self.state_initial = nn.Sequential(
            nn.Linear(state_dim, state_proc_dim),
            nn.LayerNorm(state_proc_dim),
            nn.ReLU(),
        )

        # Initial processing for LiDAR scan (only if lidar is included)
        if self.include_lidar:
            self.lidar_initial = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.lidar_dim, lidar_proc_dim),
                nn.LayerNorm(lidar_proc_dim),
                nn.ReLU(),
            )
            expert_input_dim = state_proc_dim + lidar_proc_dim
        else:
            self.lidar_initial = None
            expert_input_dim = state_proc_dim

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert_net = nn.Sequential(
                nn.Linear(expert_input_dim, expert_hidden_dim),
                nn.LayerNorm(expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, expert_hidden_dim),
                nn.LayerNorm(expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, features_dim) # Each expert outputs features_dim
            )
            self.experts.append(expert_net)

        if self.include_params:
            self.gating_network = nn.Sequential(
                nn.Linear(param_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, gate_hidden_dim),
                nn.LayerNorm(gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, num_experts) # Raw scores for softmax
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        if self.include_params:
            self.gating_network.apply(self._init_weights) # Gating network also needs init
        
        logging.info(f"MoE Feature extractor initialized with {num_experts} experts.")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract different components from observation
        state_features = observations[:, :self.state_dim]

        # Initial processing
        state_output = self.state_initial(state_features)
        
        if self.include_lidar:
            lidar_features = observations[:, self.state_dim:self.state_dim + self.lidar_dim]
            lidar_output = self.lidar_initial(lidar_features)
            expert_input = torch.cat([state_output, lidar_output], dim=1)
        else:
            expert_input = state_output

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(expert_input))
        
        # Stack expert outputs: (batch_size, num_experts, features_dim)
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)

        if self.include_params:
            param_features = observations[:, self.state_dim + self.lidar_dim:]
            gating_scores = self.gating_network(param_features)
            gating_weights = torch.softmax(gating_scores, dim=-1) # (batch_size, num_experts)
        else:
            # Uniform weights if no params for gating
            gating_weights = torch.ones(observations.shape[0], self.num_experts, device=observations.device) / self.num_experts
            
        # Expand gating_weights for broadcasting: (batch_size, num_experts, 1)
        gating_weights_expanded = gating_weights.unsqueeze(-1)
        
        # Weighted sum of expert outputs
        # (batch_size, num_experts, 1) * (batch_size, num_experts, features_dim) -> sum over dim 1
        output = torch.sum(gating_weights_expanded * expert_outputs_stacked, dim=1)
        
        return output


class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simple MLP feature extractor for the F1Tenth environment.
    
    A straightforward multi-layer perceptron with no specialized processing
    for different input components.
    """
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1024,
        state_dim: int = 4,
        lidar_dim: int = 1080,
        param_dim: int = 12, 
        include_params: bool = True,
        include_lidar: bool = True
    ):
        self.state_dim = state_dim
        self.lidar_dim = lidar_dim if include_lidar else 0
        self.param_dim = param_dim
        self.include_params = include_params
        self.include_lidar = include_lidar
        
        # Determine the input dimension based on whether params are included
        input_dim = state_dim + self.lidar_dim
        if include_params:
            input_dim += param_dim
            
        # Verify observation space
        assert observation_space.shape[0] == input_dim, f"Expected observation dimension {input_dim}, got {observation_space.shape[0]}"
            
        super().__init__(observation_space, features_dim)
        
        # Simple MLP with gradually decreasing layer sizes
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self.network.apply(self._init_weights)
        logging.info("MLP Feature extractor initialized")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # The entire observation is processed at once without splitting
        return self.network(observations)


class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    ResNet-style feature extractor for the F1Tenth environment.
    
    Uses residual connections throughout the network but processes
    the entire observation as a single vector.
    """
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1024,
        state_dim: int = 4,
        lidar_dim: int = 1080,
        param_dim: int = 12, 
        include_params: bool = True,
        include_lidar: bool = True
    ):
        self.state_dim = state_dim
        self.lidar_dim = lidar_dim if include_lidar else 0
        self.param_dim = param_dim
        self.include_params = include_params
        self.include_lidar = include_lidar
        
        # Determine the input dimension based on whether params are included
        input_dim = state_dim + self.lidar_dim
        if include_params:
            input_dim += param_dim
            
        # Verify observation space
        assert observation_space.shape[0] == input_dim, f"Expected observation dimension {input_dim}, got {observation_space.shape[0]}"
            
        super().__init__(observation_space, features_dim)
        
        # Initial processing
        self.initial = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
        )
        
        # Final layer to match features_dim
        self.final = nn.Sequential(
            nn.Linear(1024, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self.initial.apply(self._init_weights)
        self.residual_blocks.apply(self._init_weights)
        self.final.apply(self._init_weights)
        logging.info("ResNet Feature extractor initialized")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Process the entire observation through the network
        x = self.initial(observations)
        x = self.residual_blocks(x)
        return self.final(x)


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for the F1Tenth environment.
    
    Uses self-attention to process the input features.
    """
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1024,
        state_dim: int = 4,
        lidar_dim: int = 1080,
        param_dim: int = 12, 
        include_params: bool = True,
        include_lidar: bool = True,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        embedding_dim: int = 128
    ):
        self.state_dim = state_dim
        self.lidar_dim = lidar_dim if include_lidar else 0
        self.param_dim = param_dim
        self.include_params = include_params
        self.include_lidar = include_lidar
        self.embedding_dim = embedding_dim
        
        # Determine the input dimension and sequence length
        self.total_dim = state_dim + self.lidar_dim
        if include_params:
            self.total_dim += param_dim
            
        # Verify observation space
        assert observation_space.shape[0] == self.total_dim, f"Expected observation dimension {self.total_dim}, got {observation_space.shape[0]}"
            
        super().__init__(observation_space, features_dim)
        
        # Determine how to split the input into tokens
        # We'll use different strategies for different parts of the input
        
        # For lidar, segment into multiple tokens to better capture spatial patterns
        # Break the 1080 lidar points into e.g. 9 tokens of 120 points each
        if self.include_lidar:
            self.lidar_tokens = 9
            self.lidar_points_per_token = self.lidar_dim // self.lidar_tokens
        else:
            self.lidar_tokens = 0
            self.lidar_points_per_token = 0
        
        # For state and params, use one token each
        self.num_tokens = self.lidar_tokens + 1  # lidar tokens + state token
        if include_params:
            self.num_tokens += 1  # Add one more token for params
        
        # Initial embeddings for each part
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        if self.include_lidar:
            self.lidar_embedding = nn.Sequential(
                nn.Linear(self.lidar_points_per_token, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            self.lidar_embedding = None
        
        if include_params:
            self.param_embedding = nn.Sequential(
                nn.Linear(param_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        
        # Positional encoding - learnable
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embedding_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final processing to get to features_dim
        self.final = nn.Sequential(
            nn.Linear(embedding_dim * self.num_tokens, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        self.state_embedding.apply(self._init_weights)
        if self.lidar_embedding:
            self.lidar_embedding.apply(self._init_weights)
        if include_params:
            self.param_embedding.apply(self._init_weights)
        self.final.apply(self._init_weights)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
        
        logging.info("Transformer Feature extractor initialized")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Extract different components
        state = observations[:, :self.state_dim]
        
        if self.include_lidar:
            lidar = observations[:, self.state_dim:self.state_dim + self.lidar_dim]
        
        if self.include_params:
            params = observations[:, self.state_dim + self.lidar_dim:]
        
        # Get state embedding (one token)
        state_emb = self.state_embedding(state)  # shape: [batch_size, embedding_dim]
        state_emb = state_emb.unsqueeze(1)  # shape: [batch_size, 1, embedding_dim]
        
        # Process lidar in chunks to create multiple tokens (only if lidar is included)
        lidar_tokens = []
        if self.include_lidar:
            for i in range(self.lidar_tokens):
                start_idx = i * self.lidar_points_per_token
                end_idx = start_idx + self.lidar_points_per_token
                lidar_chunk = lidar[:, start_idx:end_idx]
                lidar_emb = self.lidar_embedding(lidar_chunk)  # shape: [batch_size, embedding_dim]
                lidar_tokens.append(lidar_emb.unsqueeze(1))  # shape: [batch_size, 1, embedding_dim]
        
        # Concatenate state and lidar tokens
        sequence = [state_emb] + lidar_tokens  # start with state token
        
        # Add parameter token if included
        if self.include_params:
            param_emb = self.param_embedding(params)  # shape: [batch_size, embedding_dim]
            param_emb = param_emb.unsqueeze(1)  # shape: [batch_size, 1, embedding_dim]
            sequence.append(param_emb)
            
        # Combine into sequence
        x = torch.cat(sequence, dim=1)  # shape: [batch_size, num_tokens, embedding_dim]
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Flatten and pass through final layers
        x = x.reshape(batch_size, -1)  # shape: [batch_size, num_tokens*embedding_dim]
        return self.final(x)
