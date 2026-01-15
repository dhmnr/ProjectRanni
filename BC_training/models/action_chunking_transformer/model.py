"""Action Chunking Transformer for behavior cloning.

Uses encoder-decoder transformer architecture to predict sequences of future actions.
The encoder processes observation history, and the decoder uses learnable action queries
to predict the next chunk_size actions autoregressively.

Architecture:
    - Vision Encoder: Per-frame CNN processing
    - Temporal Positional Encoding: Learnable position embeddings for frames
    - Transformer Encoder: Encodes observation sequence
    - Action Query Embeddings: Learnable queries for future actions
    - Transformer Decoder: Cross-attends to encoder memory to predict actions
    - Multi-hot Action Head: Outputs sigmoid probabilities for each action
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """CNN encoder for a single frame following the architecture spec.

    Architecture:
        Conv2d(3 → 32, kernel=8, stride=4, padding=2) → ReLU
        Conv2d(32 → 64, kernel=4, stride=2, padding=1) → ReLU
        Conv2d(64 → 128, kernel=3, stride=2, padding=1) → ReLU
        Flatten → Linear(18432 → 512) → ReLU → Dropout
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
        # kernel=8, stride=4, padding=2 (SAME-like for this stride)
        x = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding=((2, 2), (2, 2)),
        )(x)
        x = nn.relu(x)

        # Conv layer 2: (B, 36, 64, 32) -> (B, 18, 32, 64)
        # kernel=4, stride=2, padding=1
        x = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
        )(x)
        x = nn.relu(x)

        # Conv layer 3: (B, 18, 32, 64) -> (B, 9, 16, 128)
        # kernel=3, stride=2, padding=1
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
        )(x)
        x = nn.relu(x)

        # Flatten: (B, 9, 16, 128) -> (B, 18432)
        x = x.reshape(x.shape[0], -1)

        # Linear projection to d_model
        x = nn.Dense(features=self.d_model)(x)
        x = nn.relu(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x


class StateEncoder(nn.Module):
    """MLP encoder for continuous state features + animation embeddings.

    Processes:
    - Continuous state (HP, stamina, distances, etc.)
    - Hero animation embedding
    - NPC animation embedding

    Output is projected to match d_model for concatenation with vision features.
    """

    d_model: int = 512
    state_hidden_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    hero_anim_vocab_size: int = 67  # +1 for UNK
    npc_anim_vocab_size: int = 54   # +1 for UNK
    anim_embed_dim: int = 16
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, state, hero_anim_idx, npc_anim_idx, training: bool = True):
        """Encode state features.

        Args:
            state: [B, num_continuous_features] continuous state values
            hero_anim_idx: [B] hero animation indices (int32)
            npc_anim_idx: [B] NPC animation indices (int32)
            training: Whether in training mode

        Returns:
            [B, d_model] encoded state
        """
        # === Continuous state MLP ===
        x_state = state
        for features in self.state_hidden_features:
            x_state = nn.Dense(features=features)(x_state)
            x_state = nn.LayerNorm()(x_state)
            x_state = nn.relu(x_state)
            if self.dropout_rate > 0:
                x_state = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_state)

        x_state = nn.Dense(features=self.state_output_features)(x_state)
        x_state = nn.relu(x_state)

        # === Animation embeddings ===
        hero_embed = nn.Embed(
            num_embeddings=self.hero_anim_vocab_size,
            features=self.anim_embed_dim,
            name='hero_anim_embed',
        )(hero_anim_idx)  # [B, anim_embed_dim]

        npc_embed = nn.Embed(
            num_embeddings=self.npc_anim_vocab_size,
            features=self.anim_embed_dim,
            name='npc_anim_embed',
        )(npc_anim_idx)  # [B, anim_embed_dim]

        # === Concatenate and project to d_model ===
        # Total features: state_output_features + 2 * anim_embed_dim
        x = jnp.concatenate([x_state, hero_embed, npc_embed], axis=-1)

        # Project to d_model
        x = nn.Dense(features=self.d_model)(x)
        x = nn.relu(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention.

    Pre-LN architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    """

    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Forward pass.

        Args:
            x: [B, T, D] input sequence
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
    """Transformer decoder block with self-attention and cross-attention.

    Pre-LN architecture:
        LayerNorm -> Self-Attention -> Residual
        LayerNorm -> Cross-Attention -> Residual
        LayerNorm -> FFN -> Residual
    """

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
        """Forward pass.

        Args:
            x: [B, Q, D] query sequence (action queries)
            memory: [B, K, D] encoder memory (observation encoding)
            self_attn_mask: Optional [Q, Q] mask for self-attention
            training: Whether in training mode

        Returns:
            [B, Q, D] output sequence
        """
        # Pre-LN Self-Attention (on queries)
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

        # Pre-LN Cross-Attention (queries attend to memory)
        residual = x
        x = nn.LayerNorm()(x)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, memory)  # x is query, memory is key/value

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


class ActionChunkingTransformer(nn.Module):
    """Action Chunking Transformer for behavior cloning.

    Predicts a chunk of future actions given observation history and optional state.

    Input:
        - frames: [B, T, C, H, W] observation history
        - Optional state: [B, num_state_features] continuous state
        - Optional hero_anim_idx: [B] hero animation indices
        - Optional npc_anim_idx: [B] NPC animation indices

    Output:
        - action_logits: [B, chunk_size, num_actions] action predictions
    """

    num_actions: int = 12           # Number of action keys
    num_frames: int = 4             # Number of input frames (temporal context)
    chunk_size: int = 16            # Number of future actions to predict

    # Transformer configuration
    d_model: int = 512              # Model dimension
    num_heads: int = 8              # Number of attention heads
    num_encoder_layers: int = 4     # Number of encoder layers
    num_decoder_layers: int = 4     # Number of decoder layers
    d_ff: int = 2048                # Feed-forward hidden dimension

    dropout_rate: float = 0.1

    # State encoder configuration (only used if use_state=True)
    use_state: bool = False
    num_state_features: int = 10
    state_hidden_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    hero_anim_vocab_size: int = 67
    npc_anim_vocab_size: int = 54
    anim_embed_dim: int = 16

    @nn.compact
    def __call__(
        self,
        frames,
        state: Optional[jnp.ndarray] = None,
        hero_anim_idx: Optional[jnp.ndarray] = None,
        npc_anim_idx: Optional[jnp.ndarray] = None,
        training: bool = True,
    ):
        """Forward pass.

        Args:
            frames: [B, T, C, H, W] stacked frames (T = num_frames)
            state: Optional [B, num_state_features] continuous state
            hero_anim_idx: Optional [B] hero animation indices
            npc_anim_idx: Optional [B] NPC animation indices
            training: Whether in training mode

        Returns:
            action_logits: [B, chunk_size, num_actions]
        """
        batch_size = frames.shape[0]
        num_frames = frames.shape[1]

        # === BLOCK 1: Vision Encoder (per-frame CNN) ===
        vision_encoder = VisionEncoder(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
        )

        # Process each frame independently through shared CNN
        # [B, T, C, H, W] -> [B, T, d_model]
        frame_features = []
        for t in range(num_frames):
            # [B, C, H, W] -> [B, H, W, C] (NCHW -> NHWC)
            frame_t = jnp.transpose(frames[:, t], (0, 2, 3, 1))
            feat_t = vision_encoder(frame_t, training=training)
            frame_features.append(feat_t)

        # Stack to [B, T, d_model]
        obs_features = jnp.stack(frame_features, axis=1)

        # === BLOCK 1.5: State Encoder (optional) ===
        if self.use_state and state is not None:
            state_encoder = StateEncoder(
                d_model=self.d_model,
                state_hidden_features=self.state_hidden_features,
                state_output_features=self.state_output_features,
                hero_anim_vocab_size=self.hero_anim_vocab_size,
                npc_anim_vocab_size=self.npc_anim_vocab_size,
                anim_embed_dim=self.anim_embed_dim,
                dropout_rate=self.dropout_rate,
            )

            # Encode state -> [B, d_model]
            state_features = state_encoder(
                state, hero_anim_idx, npc_anim_idx, training=training
            )

            # Add state as an additional "token" in the observation sequence
            # [B, d_model] -> [B, 1, d_model]
            state_features = state_features[:, None, :]

            # Concatenate: [B, T, d_model] + [B, 1, d_model] -> [B, T+1, d_model]
            obs_features = jnp.concatenate([obs_features, state_features], axis=1)
            seq_len = num_frames + 1
        else:
            seq_len = num_frames

        # === BLOCK 2: Temporal Positional Encoding ===
        # Learnable positional encoding for observation sequence
        obs_pos_embed = self.param(
            'obs_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, seq_len, self.d_model)
        )
        obs_features = obs_features + obs_pos_embed[:, :seq_len, :]

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

        # Final layer norm for encoder
        memory = nn.LayerNorm(name='encoder_norm')(x)
        # memory: [B, seq_len, d_model] - encoded observation sequence (frames + optional state)

        # === BLOCK 4: Action Query Embeddings ===
        # Learnable queries for chunk_size future actions
        action_queries = self.param(
            'action_queries',
            nn.initializers.normal(stddev=0.02),
            (1, self.chunk_size, self.d_model)
        )

        # Learnable positional encoding for action sequence
        action_pos_embed = self.param(
            'action_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, self.chunk_size, self.d_model)
        )

        # Expand for batch and add positional encoding
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

        # Final layer norm for decoder
        action_features = nn.LayerNorm(name='decoder_norm')(x)
        # action_features: [B, chunk_size, d_model]

        # === BLOCK 6: Multi-Hot Action Head ===
        # Linear projection to action logits
        action_logits = nn.Dense(
            features=self.num_actions,
            name='action_head'
        )(action_features)
        # action_logits: [B, chunk_size, num_actions]

        return action_logits


def create_model(
    num_actions: int = 12,
    num_frames: int = 4,
    chunk_size: int = 16,
    d_model: int = 512,
    num_heads: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    d_ff: int = 2048,
    dropout_rate: float = 0.1,
    # State encoder parameters
    use_state: bool = False,
    num_state_features: int = 10,
    state_hidden_features: Tuple[int, ...] = (64, 64),
    state_output_features: int = 64,
    hero_anim_vocab_size: int = 67,
    npc_anim_vocab_size: int = 54,
    anim_embed_dim: int = 16,
) -> ActionChunkingTransformer:
    """Factory function to create Action Chunking Transformer model.

    Args:
        num_actions: Number of action outputs (keys)
        num_frames: Number of input frames (temporal context)
        chunk_size: Number of future actions to predict
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder transformer blocks
        num_decoder_layers: Number of decoder transformer blocks
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate
        use_state: Whether to include state encoding
        num_state_features: Number of continuous state features
        state_hidden_features: State encoder hidden layer dimensions
        state_output_features: State encoder output dimension
        hero_anim_vocab_size: Hero animation vocabulary size
        npc_anim_vocab_size: NPC animation vocabulary size
        anim_embed_dim: Animation embedding dimension

    Returns:
        ActionChunkingTransformer instance
    """
    model = ActionChunkingTransformer(
        num_actions=num_actions,
        num_frames=num_frames,
        chunk_size=chunk_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        use_state=use_state,
        num_state_features=num_state_features,
        state_hidden_features=state_hidden_features,
        state_output_features=state_output_features,
        hero_anim_vocab_size=hero_anim_vocab_size,
        npc_anim_vocab_size=npc_anim_vocab_size,
        anim_embed_dim=anim_embed_dim,
    )

    logger.info(f"Created ActionChunkingTransformer:")
    logger.info(f"  Input: {num_frames} frames -> Output: {chunk_size} action chunks")
    logger.info(f"  Model dim (d_model): {d_model}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  Encoder layers: {num_encoder_layers}")
    logger.info(f"  Decoder layers: {num_decoder_layers}")
    logger.info(f"  FF hidden dim: {d_ff}")
    logger.info(f"  Num actions: {num_actions}")
    logger.info(f"  Dropout: {dropout_rate}")
    if use_state:
        logger.info(f"  State encoder: enabled")
        logger.info(f"    Continuous features: {num_state_features}")
        logger.info(f"    Hidden layers: {state_hidden_features}")
        logger.info(f"    Hero anim vocab: {hero_anim_vocab_size}, embed: {anim_embed_dim}")
        logger.info(f"    NPC anim vocab: {npc_anim_vocab_size}, embed: {anim_embed_dim}")
    else:
        logger.info(f"  State encoder: disabled (vision-only)")

    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
