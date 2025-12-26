"""Pure CNN model for behavior cloning from vision only."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional
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


class PureCNN(nn.Module):
    """Pure CNN model for behavior cloning.
    
    Architecture:
        Input: [B, C, H, W] frames (RGB)
        -> Conv blocks (feature extraction)
        -> Flatten
        -> Dense layers
        -> Output: [B, num_actions] (sigmoid activation)
    
    Attributes:
        num_actions: Number of action outputs
        conv_features: List of feature dimensions for conv layers
        dense_features: List of feature dimensions for dense layers
        dropout_rate: Dropout rate (0 = no dropout)
    """
    
    num_actions: int
    conv_features: tuple = (32, 64, 128, 256)
    dense_features: tuple = (512, 256)
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """Forward pass.
        
        Args:
            x: Input frames [B, C, H, W]
            training: Whether in training mode
            
        Returns:
            Action logits [B, num_actions] (pre-sigmoid)
        """
        # Input shape: [B, C, H, W] -> [B, H, W, C] for Flax
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Conv blocks with stride 2 for downsampling
        for i, features in enumerate(self.conv_features):
            stride = (2, 2) if i < len(self.conv_features) - 1 else (1, 1)
            x = ConvBlock(
                features=features,
                kernel_size=(3, 3),
                strides=stride,
                use_batch_norm=self.use_batch_norm,
            )(x, training=training)
        
        # Global average pooling (alternative to flatten - reduces params)
        x = jnp.mean(x, axis=(1, 2))  # [B, H, W, C] -> [B, C]
        
        # Dense layers
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
    conv_features: tuple = (32, 64, 128, 256),
    dense_features: tuple = (512, 256),
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
) -> PureCNN:
    """Factory function to create PureCNN model.
    
    Args:
        num_actions: Number of action outputs
        conv_features: Tuple of conv layer feature dimensions
        dense_features: Tuple of dense layer feature dimensions
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        PureCNN model instance
    """
    model = PureCNN(
        num_actions=num_actions,
        conv_features=conv_features,
        dense_features=dense_features,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )
    
    logger.info(f"Created PureCNN model:")
    logger.info(f"  Conv features: {conv_features}")
    logger.info(f"  Dense features: {dense_features}")
    logger.info(f"  Dropout rate: {dropout_rate}")
    logger.info(f"  Batch norm: {use_batch_norm}")
    logger.info(f"  Num actions: {num_actions}")
    
    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model.
    
    Args:
        params: Model parameters (PyTree)
        
    Returns:
        Total parameter count
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

