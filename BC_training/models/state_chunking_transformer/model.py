"""State Chunking Transformer for behavior cloning.

State-only model with temporal stacking that predicts binary action chunks.
No vision encoder - uses only state features (HP, stamina, distances, animations).

Takes a sequence of states as context and predicts future action sequences.

Architecture:
    - Temporal State Encoder: Encodes each state in the sequence independently
    - Transformer Encoder: Self-attention over temporal state sequence
    - Action Query Embeddings: Learnable queries for future actions
    - Transformer Decoder: Cross-attends to encoder memory to predict actions
    - Binary Action Head: Sigmoid output for independent binary actions
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalStateEncoder(nn.Module):
    """Encodes temporal state sequence with shared MLP + animation embeddings."""

    d_model: int = 512
    state_hidden_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    hero_anim_vocab_size: int = 67
    npc_anim_vocab_size: int = 54
    anim_embed_dim: int = 16
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,
        hero_anim_ids: jnp.ndarray,
        npc_anim_ids: jnp.ndarray,
        training: bool = True,
    ):
        """Encode temporal state sequence.

        Args:
            states: [B, T, num_continuous_features] temporal state sequence
            hero_anim_ids: [B, T] hero animation indices per timestep
            npc_anim_ids: [B, T] NPC animation indices per timestep
            training: Whether in training mode

        Returns:
            [B, T, d_model] encoded temporal state features
        """
        batch_size, num_states, num_features = states.shape

        # Reshape to process all states with shared MLP: [B*T, num_features]
        states_flat = states.reshape(-1, num_features)
        hero_flat = hero_anim_ids.reshape(-1)
        npc_flat = npc_anim_ids.reshape(-1)

        # Continuous state MLP (shared across all timesteps)
        x_state = states_flat
        for features in self.state_hidden_features:
            x_state = nn.Dense(features=features)(x_state)
            x_state = nn.LayerNorm()(x_state)
            x_state = nn.relu(x_state)
            if self.dropout_rate > 0:
                x_state = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_state)

        x_state = nn.Dense(features=self.state_output_features)(x_state)
        x_state = nn.relu(x_state)

        # Animation embeddings (shared across all timesteps)
        hero_embed = nn.Embed(
            num_embeddings=self.hero_anim_vocab_size,
            features=self.anim_embed_dim,
            name='hero_anim_embed',
        )(hero_flat)

        npc_embed = nn.Embed(
            num_embeddings=self.npc_anim_vocab_size,
            features=self.anim_embed_dim,
            name='npc_anim_embed',
        )(npc_flat)

        # Concatenate and project to d_model
        x = jnp.concatenate([x_state, hero_embed, npc_embed], axis=-1)
        x = nn.Dense(features=self.d_model)(x)
        x = nn.relu(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Reshape back to temporal sequence: [B, T, d_model]
        x = x.reshape(batch_size, num_states, self.d_model)

        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention."""

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
    """Transformer decoder block with self-attention and cross-attention."""

    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, memory, training: bool = True):
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


class StateChunkingTransformer(nn.Module):
    """State Chunking Transformer for behavior cloning (STATE-ONLY).

    Predicts a chunk of future binary actions given temporal state sequence.
    Uses sigmoid output for independent binary action predictions.

    Input:
        - states: [B, num_states, num_features] temporal state sequence
        - hero_anim_ids: [B, num_states] hero animation indices
        - npc_anim_ids: [B, num_states] NPC animation indices

    Output:
        - action_logits: [B, chunk_size, num_actions] binary action logits
    """

    num_actions: int = 13           # Number of binary action outputs
    num_states: int = 8             # Number of temporal state observations
    chunk_size: int = 8             # Number of future actions to predict

    # Transformer configuration
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 2048

    dropout_rate: float = 0.1

    # State encoder configuration
    num_state_features: int = 10
    state_hidden_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    hero_anim_vocab_size: int = 67
    npc_anim_vocab_size: int = 54
    anim_embed_dim: int = 16

    @nn.compact
    def __call__(
        self,
        states: jnp.ndarray,
        hero_anim_ids: jnp.ndarray,
        npc_anim_ids: jnp.ndarray,
        training: bool = True,
    ):
        """Forward pass.

        Args:
            states: [B, num_states, num_features] temporal state sequence
            hero_anim_ids: [B, num_states] hero animation indices
            npc_anim_ids: [B, num_states] NPC animation indices
            training: Whether in training mode

        Returns:
            action_logits: [B, chunk_size, num_actions] - apply sigmoid for probabilities
        """
        batch_size = states.shape[0]
        num_states = states.shape[1]

        # === Temporal State Encoder ===
        state_encoder = TemporalStateEncoder(
            d_model=self.d_model,
            state_hidden_features=self.state_hidden_features,
            state_output_features=self.state_output_features,
            hero_anim_vocab_size=self.hero_anim_vocab_size,
            npc_anim_vocab_size=self.npc_anim_vocab_size,
            anim_embed_dim=self.anim_embed_dim,
            dropout_rate=self.dropout_rate,
        )

        # Encode temporal states -> [B, num_states, d_model]
        obs_features = state_encoder(
            states, hero_anim_ids, npc_anim_ids, training=training
        )

        # === Temporal Positional Encoding ===
        temporal_pos_embed = self.param(
            'temporal_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, num_states, self.d_model)
        )
        obs_features = obs_features + temporal_pos_embed

        if self.dropout_rate > 0:
            obs_features = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=not training
            )(obs_features)

        # === Transformer Encoder (self-attention over temporal states) ===
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

        # === Action Query Embeddings ===
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

        # === Transformer Decoder ===
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

        # === Binary Action Head (sigmoid output) ===
        action_logits = nn.Dense(
            features=self.num_actions,
            name='action_head'
        )(action_features)
        # action_logits: [B, chunk_size, num_actions]

        return action_logits


def create_model(
    num_actions: int = 13,
    num_states: int = 8,
    chunk_size: int = 8,
    d_model: int = 512,
    num_heads: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    d_ff: int = 2048,
    dropout_rate: float = 0.1,
    # State encoder parameters
    num_state_features: int = 10,
    state_hidden_features: Tuple[int, ...] = (64, 64),
    state_output_features: int = 64,
    hero_anim_vocab_size: int = 67,
    npc_anim_vocab_size: int = 54,
    anim_embed_dim: int = 16,
) -> StateChunkingTransformer:
    """Factory function to create State Chunking Transformer model.

    Args:
        num_actions: Number of binary action outputs
        num_states: Number of temporal state observations (context)
        chunk_size: Number of future actions to predict
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder transformer blocks
        num_decoder_layers: Number of decoder transformer blocks
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout rate
        num_state_features: Number of continuous state features
        state_hidden_features: State encoder hidden layer dimensions
        state_output_features: State encoder output dimension
        hero_anim_vocab_size: Hero animation vocabulary size
        npc_anim_vocab_size: NPC animation vocabulary size
        anim_embed_dim: Animation embedding dimension

    Returns:
        StateChunkingTransformer instance
    """
    model = StateChunkingTransformer(
        num_actions=num_actions,
        num_states=num_states,
        chunk_size=chunk_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        num_state_features=num_state_features,
        state_hidden_features=state_hidden_features,
        state_output_features=state_output_features,
        hero_anim_vocab_size=hero_anim_vocab_size,
        npc_anim_vocab_size=npc_anim_vocab_size,
        anim_embed_dim=anim_embed_dim,
    )

    logger.info(f"Created StateChunkingTransformer (STATE-ONLY, temporal stacking):")
    logger.info(f"  Input: {num_states} temporal states")
    logger.info(f"  Output: {chunk_size} action predictions x {num_actions} actions")
    logger.info(f"  Model dim (d_model): {d_model}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  Encoder layers: {num_encoder_layers}")
    logger.info(f"  Decoder layers: {num_decoder_layers}")
    logger.info(f"  FF hidden dim: {d_ff}")
    logger.info(f"  Dropout: {dropout_rate}")
    logger.info(f"  State encoder:")
    logger.info(f"    Continuous features: {num_state_features}")
    logger.info(f"    Hidden layers: {state_hidden_features}")
    logger.info(f"    Hero anim vocab: {hero_anim_vocab_size}, embed: {anim_embed_dim}")
    logger.info(f"    NPC anim vocab: {npc_anim_vocab_size}, embed: {anim_embed_dim}")

    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
