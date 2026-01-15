"""Onset/Offset Transformer for behavior cloning.

Predicts onset (0→1) and offset (1→0) transitions for action sequences.
Uses encoder-decoder transformer architecture similar to ActionChunkingTransformer,
but outputs separate onset/offset predictions instead of raw action states.

Output shape: [B, chunk_size, num_actions, 2] where last dim is (onset, offset)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """CNN encoder for a single frame.

    Architecture:
        Conv2d(3 → 32, kernel=8, stride=4, padding=2) → ReLU
        Conv2d(32 → 64, kernel=4, stride=2, padding=1) → ReLU
        Conv2d(64 → 128, kernel=3, stride=2, padding=1) → ReLU
        Flatten → Linear(→ d_model) → ReLU → Dropout
    """

    d_model: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Encode a single frame.

        Args:
            x: [B, H, W, C] frame (NHWC format) - expects (B, 144, 256, 3)
            training: Whether in training mode

        Returns:
            [B, d_model] encoded frame
        """
        # Conv layer 1: (B, 144, 256, 3) -> (B, 36, 64, 32)
        x = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding=((2, 2), (2, 2)),
        )(x)
        x = nn.relu(x)

        # Conv layer 2: (B, 36, 64, 32) -> (B, 18, 32, 64)
        x = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
        )(x)
        x = nn.relu(x)

        # Conv layer 3: (B, 18, 32, 64) -> (B, 9, 16, 128)
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
        )(x)
        x = nn.relu(x)

        # Flatten and project
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=self.d_model)(x)
        x = nn.relu(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention (Pre-LN)."""

    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Pre-LN Self-Attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, x)
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


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention (Pre-LN)."""

    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x,
        memory,
        self_attn_mask: Optional[jnp.ndarray] = None,
        training: bool = True
    ):
        # Pre-LN Self-Attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, x, mask=self_attn_mask)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = x + residual

        # Pre-LN Cross-Attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, memory)
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


class OnsetOffsetTransformer(nn.Module):
    """Onset/Offset Transformer for behavior cloning.

    Predicts onset and offset transitions for future actions.

    Input:
        - frames: [B, T, C, H, W] observation history

    Output:
        - onset_offset_logits: [B, chunk_size, num_actions, 2]
          where [..., 0] is onset (0→1) and [..., 1] is offset (1→0)
    """

    num_actions: int = 13           # Number of action keys
    num_frames: int = 8             # Number of input frames
    chunk_size: int = 8             # Number of future predictions

    # Transformer configuration
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 2048

    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        frames,
        training: bool = True,
    ):
        """Forward pass.

        Args:
            frames: [B, T, C, H, W] stacked frames
            training: Whether in training mode

        Returns:
            onset_offset_logits: [B, chunk_size, num_actions, 2]
        """
        batch_size = frames.shape[0]
        num_frames = frames.shape[1]

        # === BLOCK 1: Vision Encoder (per-frame CNN) ===
        vision_encoder = VisionEncoder(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
        )

        frame_features = []
        for t in range(num_frames):
            # [B, C, H, W] -> [B, H, W, C]
            frame_t = jnp.transpose(frames[:, t], (0, 2, 3, 1))
            feat_t = vision_encoder(frame_t, training=training)
            frame_features.append(feat_t)

        obs_features = jnp.stack(frame_features, axis=1)  # [B, T, d_model]

        # === BLOCK 2: Temporal Positional Encoding ===
        obs_pos_embed = self.param(
            'obs_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, num_frames, self.d_model)
        )
        obs_features = obs_features + obs_pos_embed

        if self.dropout_rate > 0:
            obs_features = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=not training
            )(obs_features)

        # === BLOCK 3: Transformer Encoder ===
        x = obs_features
        for layer_idx in range(self.num_encoder_layers):
            x = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f'encoder_block_{layer_idx}',
            )(x, training=training)

        memory = nn.LayerNorm(name='encoder_norm')(x)

        # === BLOCK 4: Action Query Embeddings ===
        action_queries = self.param(
            'action_queries',
            nn.initializers.normal(stddev=0.02),
            (1, self.chunk_size, self.d_model)
        )
        action_pos_embed = self.param(
            'action_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, self.chunk_size, self.d_model)
        )

        queries = jnp.broadcast_to(action_queries, (batch_size, self.chunk_size, self.d_model))
        queries = queries + action_pos_embed

        # === BLOCK 5: Transformer Decoder ===
        x = queries
        for layer_idx in range(self.num_decoder_layers):
            x = TransformerDecoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f'decoder_block_{layer_idx}',
            )(x, memory, training=training)

        action_features = nn.LayerNorm(name='decoder_norm')(x)
        # action_features: [B, chunk_size, d_model]

        # === BLOCK 6: Onset/Offset Prediction Head ===
        # Output [B, chunk_size, num_actions, 2] for onset and offset
        # We use a single dense layer to produce num_actions * 2 outputs, then reshape
        onset_offset_logits = nn.Dense(
            features=self.num_actions * 2,
            name='onset_offset_head'
        )(action_features)

        # Reshape: [B, chunk_size, num_actions * 2] -> [B, chunk_size, num_actions, 2]
        onset_offset_logits = onset_offset_logits.reshape(
            batch_size, self.chunk_size, self.num_actions, 2
        )

        return onset_offset_logits


def create_model(
    num_actions: int = 13,
    num_frames: int = 8,
    chunk_size: int = 8,
    d_model: int = 512,
    num_heads: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    d_ff: int = 2048,
    dropout_rate: float = 0.1,
) -> OnsetOffsetTransformer:
    """Factory function to create Onset/Offset Transformer model.

    Args:
        num_actions: Number of action outputs
        num_frames: Number of input frames
        chunk_size: Number of future predictions
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder transformer blocks
        num_decoder_layers: Number of decoder transformer blocks
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate

    Returns:
        OnsetOffsetTransformer instance
    """
    model = OnsetOffsetTransformer(
        num_actions=num_actions,
        num_frames=num_frames,
        chunk_size=chunk_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
    )

    logger.info(f"Created OnsetOffsetTransformer:")
    logger.info(f"  Input: {num_frames} frames -> Output: {chunk_size} onset/offset predictions")
    logger.info(f"  Model dim (d_model): {d_model}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  Encoder layers: {num_encoder_layers}")
    logger.info(f"  Decoder layers: {num_decoder_layers}")
    logger.info(f"  FF hidden dim: {d_ff}")
    logger.info(f"  Num actions: {num_actions}")
    logger.info(f"  Output shape: (B, {chunk_size}, {num_actions}, 2)")

    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
