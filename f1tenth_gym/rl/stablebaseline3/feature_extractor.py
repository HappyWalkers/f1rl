import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Type, Union, Callable, Optional, Any
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from absl import logging


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
    
    This network separately processes state features, lidar scans, and environment parameters,
    then combines them into a single feature vector.
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
        
        # Network for processing state variables (s, ey, vel, yaw_angle) with residual connections
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            ResidualBlock(128, 256, 256),
            ResidualBlock(256, 512, 512),
            ResidualBlock(512, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
        )
        
        # Network for processing LiDAR scan with residual connections
        self.lidar_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lidar_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
            ResidualBlock(1024, 1024, 1024),
        )
        
        # Network for processing environment parameters (if included)
        if include_params:
            self.param_net = nn.Sequential(
                nn.Linear(param_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                ResidualBlock(128, 256, 256),
                ResidualBlock(256, 512, 512),
                ResidualBlock(512, 1024, 1024),
                ResidualBlock(1024, 1024, 1024),
            )
            # Combined dimension from all branches
            combined_dim = 1024 * 3  # state + lidar + param
        else:
            self.param_net = None
            combined_dim = 1024 * 2  # state + lidar
        
        # Final layers to combine all features with residual connections
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            ResidualBlock(1024, 1024, 1024),
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
        self.state_net.apply(count_linear_layers)
        self.lidar_net.apply(count_linear_layers)
        if self.param_net is not None:
            self.param_net.apply(count_linear_layers)
        self.combined_net.apply(count_linear_layers)
        
        logging.info(f"Total Linear layers to initialize: {linear_layer_count}")
        
        # Apply initialization
        self.state_net.apply(self._init_weights)
        self.lidar_net.apply(self._init_weights)
        if self.param_net is not None:
            self.param_net.apply(self._init_weights)
        self.combined_net.apply(self._init_weights)
        
        logging.info("Feature extractor weights initialized")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract different components from observation
        state_features = observations[:, :self.state_dim]
        lidar_features = observations[:, self.state_dim:self.state_dim + self.lidar_dim]
        
        # Process state features
        state_output = self.state_net(state_features)
        
        # Process lidar features
        lidar_output = self.lidar_net(lidar_features)
        
        if self.include_params:
            # Extract and process environment parameters
            param_features = observations[:, self.state_dim + self.lidar_dim:]
            param_output = self.param_net(param_features)
            
            # Combine all features
            combined_features = torch.cat([state_output, lidar_output, param_output], dim=1)
        else:
            # Combine only state and lidar features
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
