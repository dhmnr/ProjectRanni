"""Hybrid State model - CNN + game state for behavior cloning."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block with Conv -> BatchNorm -> ReLU."""
    
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME',
        )(x)
        
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = nn.relu(x)
        return x


class StateEncoder(nn.Module):
    """MLP encoder for game state features.
    
    Processes raw state (HP, stamina, positions, etc.) into a learned representation.
    """
    
    hidden_features: Tuple[int, ...] = (64, 64)
    output_features: int = 64
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, state, training: bool = True):
        """Encode state features.
        
        Args:
            state: [B, num_state_features] raw state values
            training: Whether in training mode
            
        Returns:
            [B, output_features] encoded state
        """
        x = state
        
        for features in self.hidden_features:
            x = nn.Dense(features=features)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(features=self.output_features)(x)
        x = nn.relu(x)
        
        return x


class HybridState(nn.Module):
    """Hybrid model combining CNN vision encoder with game state.
    
    Architecture:
        Frame [B, C, H, W] -> CNN -> frame_features [B, F1]
        State [B, S] -> MLP -> state_features [B, F2]
        Concat [frame_features, state_features] -> Dense -> actions [B, A]
    
    This allows the model to use both visual information and structured
    game state (HP, stamina, positions, etc.) for action prediction.
    """
    
    num_actions: int
    num_state_features: int
    conv_features: Tuple[int, ...] = (32, 64, 128, 256)
    dense_features: Tuple[int, ...] = (512, 256)
    state_encoder_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, frames, state, training: bool = True):
        """Forward pass.
        
        Args:
            frames: Input frames [B, C, H, W]
            state: Game state [B, num_state_features]
            training: Whether in training mode
            
        Returns:
            Action logits [B, num_actions] (pre-sigmoid)
        """
        # === Vision encoder (CNN) ===
        # Input shape: [B, C, H, W] -> [B, H, W, C] for Flax
        x_vision = jnp.transpose(frames, (0, 2, 3, 1))
        
        # Conv blocks with stride 2 for downsampling
        for i, features in enumerate(self.conv_features):
            stride = (2, 2) if i < len(self.conv_features) - 1 else (1, 1)
            x_vision = ConvBlock(
                features=features,
                kernel_size=(3, 3),
                strides=stride,
                use_batch_norm=self.use_batch_norm,
            )(x_vision, training=training)
        
        # Global average pooling
        x_vision = jnp.mean(x_vision, axis=(1, 2))  # [B, C]
        
        # === State encoder (MLP) ===
        x_state = StateEncoder(
            hidden_features=self.state_encoder_features,
            output_features=self.state_output_features,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        )(state, training=training)
        
        # === Fusion ===
        # Concatenate vision and state features
        x = jnp.concatenate([x_vision, x_state], axis=-1)
        
        # Dense layers for action prediction
        for features in self.dense_features:
            x = nn.Dense(features=features)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer (logits, sigmoid applied in loss)
        x = nn.Dense(features=self.num_actions)(x)
        
        return x


def create_model(
    num_actions: int,
    num_state_features: int,
    conv_features: tuple = (32, 64, 128, 256),
    dense_features: tuple = (512, 256),
    state_encoder_features: tuple = (64, 64),
    state_output_features: int = 64,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
) -> HybridState:
    """Factory function to create HybridState model.
    
    Args:
        num_actions: Number of action outputs
        num_state_features: Number of input state features (e.g., 19)
        conv_features: Tuple of conv layer feature dimensions
        dense_features: Tuple of dense layer feature dimensions
        state_encoder_features: Tuple of state encoder hidden dims
        state_output_features: State encoder output dimension
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        HybridState model instance
    """
    model = HybridState(
        num_actions=num_actions,
        num_state_features=num_state_features,
        conv_features=conv_features,
        dense_features=dense_features,
        state_encoder_features=state_encoder_features,
        state_output_features=state_output_features,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )
    
    logger.info(f"Created HybridState model:")
    logger.info(f"  Conv features: {conv_features}")
    logger.info(f"  Dense features: {dense_features}")
    logger.info(f"  State encoder: {state_encoder_features} -> {state_output_features}")
    logger.info(f"  Dropout rate: {dropout_rate}")
    logger.info(f"  Batch norm: {use_batch_norm}")
    logger.info(f"  Num actions: {num_actions}")
    logger.info(f"  Num state features: {num_state_features}")
    
    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model.
    
    Args:
        params: Model parameters (PyTree)
        
    Returns:
        Total parameter count
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

