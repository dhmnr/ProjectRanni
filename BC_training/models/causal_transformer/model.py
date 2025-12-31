"""Causal Transformer model for temporal behavior cloning.

Uses self-attention with causal masking to process frame sequences.
Each position can only attend to itself and previous positions,
enabling autoregressive-style temporal reasoning.

Key advantages over GRU:
- Parallel processing during training
- Better at capturing long-range dependencies
- Explicit attention weights (interpretable)
- Scales better with sequence length
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
import logging
import numpy as np

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


class FrameEncoder(nn.Module):
    """CNN encoder for a single frame. Shared across time steps."""
    
    conv_features: Tuple[int, ...] = (32, 64, 128, 256)
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """Encode a single frame.
        
        Args:
            x: [B, H, W, C] frame (NHWC format)
            training: Whether in training mode
            
        Returns:
            [B, feat_dim] encoded frame
        """
        for i, features in enumerate(self.conv_features):
            stride = (2, 2) if i < len(self.conv_features) - 1 else (1, 1)
            x = ConvBlock(
                features=features,
                kernel_size=(3, 3),
                strides=stride,
                use_batch_norm=self.use_batch_norm,
            )(x, training=training)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # [B, C]
        return x


class StateEncoder(nn.Module):
    """MLP encoder for continuous state features."""
    
    hidden_features: Tuple[int, ...] = (64, 64)
    output_features: int = 64
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, state, training: bool = True):
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


class TransformerBlock(nn.Module):
    """Single Transformer block with causal self-attention.
    
    Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    (Pre-LN variant for better training stability)
    """
    
    d_model: int
    num_heads: int
    d_ff: int  # Feed-forward hidden dimension
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask: Optional[jnp.ndarray] = None, training: bool = True):
        """Forward pass.
        
        Args:
            x: [B, T, D] input sequence
            mask: Optional [T, T] attention mask (causal)
            training: Whether in training mode
            
        Returns:
            [B, T, D] output sequence
        """
        # Pre-LN Self-Attention
        residual = x
        x = nn.LayerNorm()(x)
        
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, x, mask=mask)
        
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = x + residual
        
        # Pre-LN Feed-Forward
        residual = x
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(features=self.d_ff)(x)
        x = nn.gelu(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(features=self.d_model)(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = x + residual
        
        return x


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create causal attention mask.
    
    Returns a mask where position i can only attend to positions j <= i.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        [seq_len, seq_len] mask with True for allowed positions
    """
    # Create lower triangular matrix (including diagonal)
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    return mask


class CausalTransformer(nn.Module):
    """Causal Transformer for behavior cloning.
    
    Architecture:
        frames[t-k:t] -> SharedCNN per frame -> [B, T, F]
        + Positional Encoding
        -> Transformer Blocks with causal masking
        -> Take last position [B, D]
        (Optional) State [B, S] -> MLP -> [B, F2]
        (Optional) Anim embeds [B] -> Embed -> [B, E]
        (Optional) Action history [B, K, A] -> MLP -> [B, F3]
        Concat all -> Dense -> actions [B, A]
    """
    
    num_actions: int
    num_history_frames: int  # Number of past frames (not including current)
    num_action_history: int  # Number of past actions
    
    # Transformer configuration
    d_model: int = 256        # Model dimension
    num_heads: int = 4        # Number of attention heads
    num_layers: int = 2       # Number of transformer blocks
    d_ff: int = 512           # Feed-forward hidden dimension
    
    # Optional state features
    use_state: bool = False
    num_state_features: int = 10
    hero_anim_vocab_size: int = 67
    npc_anim_vocab_size: int = 54
    anim_embed_dim: int = 16
    
    # Architecture
    conv_features: Tuple[int, ...] = (32, 64, 128, 256)
    dense_features: Tuple[int, ...] = (256, 128)
    state_encoder_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    action_history_features: int = 64
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    
    # Position encoding
    max_seq_len: int = 32     # Maximum sequence length (for learned positional embeddings)
    
    @nn.compact
    def __call__(
        self,
        frames,
        action_history,
        state=None,
        hero_anim_idx=None,
        npc_anim_idx=None,
        training: bool = True,
    ):
        """Forward pass.
        
        Args:
            frames: [B, T, C, H, W] stacked frames (T = num_history_frames + 1)
            action_history: [B, K, num_actions] past actions
            state: Optional [B, num_state_features] current state
            hero_anim_idx: Optional [B] hero animation indices
            npc_anim_idx: Optional [B] NPC animation indices
            training: Whether in training mode
            
        Returns:
            Action logits [B, num_actions]
        """
        batch_size = frames.shape[0]
        num_frames = frames.shape[1]
        
        # === Encode each frame with shared CNN ===
        frame_encoder = FrameEncoder(
            conv_features=self.conv_features,
            use_batch_norm=self.use_batch_norm,
        )
        
        # Process each timestep: [B, T, C, H, W] -> [B, T, feat_dim]
        frame_features = []
        for t in range(num_frames):
            # [B, C, H, W] -> [B, H, W, C]
            frame_t = jnp.transpose(frames[:, t], (0, 2, 3, 1))
            feat_t = frame_encoder(frame_t, training=training)
            frame_features.append(feat_t)
        
        # Stack to [B, T, feat_dim]
        frame_features = jnp.stack(frame_features, axis=1)
        
        # === Project to d_model if needed ===
        cnn_feat_dim = self.conv_features[-1]  # 256 typically
        if cnn_feat_dim != self.d_model:
            frame_features = nn.Dense(features=self.d_model)(frame_features)
        
        # === Add positional encoding ===
        # Learned positional embeddings
        pos_embed = nn.Embed(
            num_embeddings=self.max_seq_len,
            features=self.d_model,
            name='position_embed',
        )
        positions = jnp.arange(num_frames)
        pos_embeddings = pos_embed(positions)  # [T, D]
        
        # Add to frame features
        x = frame_features + pos_embeddings[None, :, :]  # [B, T, D]
        
        # Apply dropout
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # === Create causal mask ===
        causal_mask = create_causal_mask(num_frames)  # [T, T]
        
        # === Apply Transformer blocks ===
        for layer_idx in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{layer_idx}',
            )(x, mask=causal_mask, training=training)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # === Take last position (current frame representation) ===
        x = x[:, -1, :]  # [B, D]
        
        # === Process action history ===
        # Flatten and encode: [B, K, A] -> [B, K*A] -> [B, action_feat]
        action_history_flat = action_history.reshape(batch_size, -1)
        x_actions = nn.Dense(features=self.action_history_features)(action_history_flat)
        if self.use_batch_norm:
            x_actions = nn.BatchNorm(use_running_average=not training)(x_actions)
        x_actions = nn.relu(x_actions)
        
        # === Combine features ===
        features_to_concat = [x, x_actions]
        
        # === Optional: State features ===
        if self.use_state and state is not None:
            x_state = StateEncoder(
                hidden_features=self.state_encoder_features,
                output_features=self.state_output_features,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
            )(state, training=training)
            features_to_concat.append(x_state)
            
            # Animation embeddings
            if hero_anim_idx is not None and npc_anim_idx is not None:
                hero_embed = nn.Embed(
                    num_embeddings=self.hero_anim_vocab_size,
                    features=self.anim_embed_dim,
                    name='hero_anim_embed',
                )(hero_anim_idx)
                
                npc_embed = nn.Embed(
                    num_embeddings=self.npc_anim_vocab_size,
                    features=self.anim_embed_dim,
                    name='npc_anim_embed',
                )(npc_anim_idx)
                
                features_to_concat.extend([hero_embed, npc_embed])
        
        # === Fusion and prediction ===
        x = jnp.concatenate(features_to_concat, axis=-1)
        
        for features in self.dense_features:
            x = nn.Dense(features=features)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output logits
        logits = nn.Dense(features=self.num_actions)(x)
        
        return logits


def create_model(
    num_actions: int,
    num_history_frames: int = 4,
    num_action_history: int = 4,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 512,
    use_state: bool = False,
    num_state_features: int = 10,
    hero_anim_vocab_size: int = 67,
    npc_anim_vocab_size: int = 54,
    anim_embed_dim: int = 16,
    conv_features: tuple = (32, 64, 128, 256),
    dense_features: tuple = (256, 128),
    state_encoder_features: tuple = (64, 64),
    state_output_features: int = 64,
    action_history_features: int = 64,
    dropout_rate: float = 0.1,
    use_batch_norm: bool = True,
    max_seq_len: int = 32,
) -> CausalTransformer:
    """Factory function to create Causal Transformer model.
    
    Args:
        num_actions: Number of action outputs
        num_history_frames: Number of past frames
        num_action_history: Number of past actions
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension
        use_state: Whether to use state features
        num_state_features: Number of continuous state features
        hero_anim_vocab_size: Size of hero animation embedding
        npc_anim_vocab_size: Size of NPC animation embedding
        anim_embed_dim: Animation embedding dimension
        conv_features: CNN feature dimensions
        dense_features: Dense layer dimensions
        state_encoder_features: State encoder hidden dims
        state_output_features: State encoder output dim
        action_history_features: Action history encoder dim
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch norm
        max_seq_len: Maximum sequence length for positional embeddings
        
    Returns:
        CausalTransformer instance
    """
    model = CausalTransformer(
        num_actions=num_actions,
        num_history_frames=num_history_frames,
        num_action_history=num_action_history,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        use_state=use_state,
        num_state_features=num_state_features,
        hero_anim_vocab_size=hero_anim_vocab_size,
        npc_anim_vocab_size=npc_anim_vocab_size,
        anim_embed_dim=anim_embed_dim,
        conv_features=conv_features,
        dense_features=dense_features,
        state_encoder_features=state_encoder_features,
        state_output_features=state_output_features,
        action_history_features=action_history_features,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        max_seq_len=max_seq_len,
    )
    
    total_frames = num_history_frames + 1
    logger.info(f"Created CausalTransformer:")
    logger.info(f"  Model dim (d_model): {d_model}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  Transformer layers: {num_layers}")
    logger.info(f"  FF hidden dim: {d_ff}")
    logger.info(f"  Total frames: {total_frames} (history={num_history_frames} + current)")
    logger.info(f"  Action history: {num_action_history}")
    logger.info(f"  Conv features: {conv_features}")
    logger.info(f"  Dense features: {dense_features}")
    logger.info(f"  Use state: {use_state}")
    if use_state:
        logger.info(f"  State features: {num_state_features}")
        logger.info(f"  Anim embeds: hero={hero_anim_vocab_size}, npc={npc_anim_vocab_size}, dim={anim_embed_dim}")
    logger.info(f"  Action history features: {action_history_features}")
    logger.info(f"  Dropout: {dropout_rate}")
    logger.info(f"  Batch norm: {use_batch_norm}")
    
    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))



