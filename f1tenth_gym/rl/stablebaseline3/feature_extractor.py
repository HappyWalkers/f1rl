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
        include_params: bool = True
    ):
        # Determine whether the observation includes parameters based on its dimension
        self.state_dim = state_dim  # 4 state components: [s, ey, vel, yaw_angle]
        self.lidar_dim = lidar_dim  # LiDAR points
        self.param_dim = param_dim  # Environment parameters
        self.include_params = include_params
        
        # The expected observation dimension with parameters included
        expected_dim_with_params = state_dim + lidar_dim + param_dim
        expected_dim_without_params = state_dim + lidar_dim
        
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
        
        # Initial processing of LiDAR scan
        self.lidar_initial = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lidar_dim, lidar_initial_dim),
            nn.LayerNorm(lidar_initial_dim),
            nn.ReLU(),
        )
        
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
            self.lidar_film = FiLMLayer(lidar_initial_dim, 256)
        
        # Residual blocks for state processing after conditioning
        self.state_residual = nn.Sequential(
            ResidualBlock(state_initial_dim, 256, 512),
                ResidualBlock(512, 1024, 1024),
                ResidualBlock(1024, 1024, 1024),
            )
        
        # Residual blocks for lidar processing after conditioning
        self.lidar_residual = nn.Sequential(
            ResidualBlock(lidar_initial_dim, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
        )
        
        # Combined dimension from all branches
        combined_dim = 1024 * 2  # state + lidar
        
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
        self.lidar_initial.apply(count_linear_layers)
        self.state_residual.apply(count_linear_layers)
        self.lidar_residual.apply(count_linear_layers)
        if include_params:
            self.param_initial.apply(count_linear_layers)
            self.state_film.apply(count_linear_layers)
            self.lidar_film.apply(count_linear_layers)
        self.combined_net.apply(count_linear_layers)
        
        logging.info(f"Total Linear layers to initialize: {linear_layer_count}")
        
        # Apply initialization
        self.state_initial.apply(self._init_weights)
        self.lidar_initial.apply(self._init_weights)
        self.state_residual.apply(self._init_weights)
        self.lidar_residual.apply(self._init_weights)
        if include_params:
            self.param_initial.apply(self._init_weights)
            self.state_film.apply(self._init_weights)
            self.lidar_film.apply(self._init_weights)
        self.combined_net.apply(self._init_weights)
        
        logging.info("Feature extractor weights initialized")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract different components from observation
        state_features = observations[:, :self.state_dim]
        lidar_features = observations[:, self.state_dim:self.state_dim + self.lidar_dim]
        
        # Initial processing
        state_output = self.state_initial(state_features)
        lidar_output = self.lidar_initial(lidar_features)
        
        if self.include_params:
            # Extract and process environment parameters
            param_features = observations[:, self.state_dim + self.lidar_dim:]
            param_output = self.param_initial(param_features)
            
            # Apply FiLM conditioning - parameters modulate state and lidar features
            state_output = self.state_film(state_output, param_output)
            lidar_output = self.lidar_film(lidar_output, param_output)
        
        # Apply residual blocks after conditional modulation
        state_output = self.state_residual(state_output)
        lidar_output = self.lidar_residual(lidar_output)
        
        # Combine modulated features
        combined_features = torch.cat([state_output, lidar_output], dim=1)
        
        # Final processing
        output = self.combined_net(combined_features)
        return output


class Reshape(nn.Module):
    """
    Helper module to reshape tensors.
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)
