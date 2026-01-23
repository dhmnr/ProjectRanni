"""Simplified agent for dodge-only policy - just learns when to dodge."""

from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


class DodgeOnlyAgent(nn.Module):
    """Agent that only decides when to dodge based on boss animation timing.

    Inputs:
        - boss_anim_id: Embedded boss animation (vocab_size -> embed_dim)
        - elapsed_frames: Sinusoidal positional encoding

    Output:
        - Discrete(2) logits: [no_dodge, dodge]
        - Value estimate
    """
    hidden_dim: int = 64
    anim_embed_dim: int = 32
    anim_vocab_size: int = 43
    sinusoidal_dim: int = 16  # 8 scales * 2 (sin + cos)

    @nn.compact
    def __call__(self, anim_idx: jnp.ndarray, elapsed_sinusoidal: jnp.ndarray):
        """Forward pass.

        Args:
            anim_idx: Boss animation vocab index [batch]
            elapsed_sinusoidal: Sinusoidal encoding of elapsed frames [batch, 16]

        Returns:
            logits: Action logits [batch, 2]
            value: Value estimate [batch]
        """
        # Embed boss animation
        anim_embed = nn.Embed(
            num_embeddings=self.anim_vocab_size,
            features=self.anim_embed_dim,
            name='anim_embedding'
        )(anim_idx)  # [batch, embed_dim]

        # Concatenate: anim_embed + elapsed_sinusoidal
        x = jnp.concatenate([anim_embed, elapsed_sinusoidal], axis=-1)

        # Shared trunk
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name='fc2')(x)
        x = nn.relu(x)

        # Policy head: Discrete(2) - no_dodge or dodge
        logits = nn.Dense(2, name='policy_head')(x)

        # Value head
        value = nn.Dense(1, name='value_head')(x)
        value = value.squeeze(-1)

        return logits, value

    def get_action_and_value(
        self,
        anim_idx: jnp.ndarray,
        elapsed_sinusoidal: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample action and compute log prob + value.

        Returns:
            action: Sampled action [batch]
            log_prob: Log probability of action [batch]
            entropy: Policy entropy [batch]
            value: Value estimate [batch]
        """
        logits, value = self(anim_idx, elapsed_sinusoidal)

        # Sample from categorical distribution
        action = jax.random.categorical(key, logits)

        # Log probability
        log_probs = jax.nn.log_softmax(logits)
        log_prob = jnp.take_along_axis(
            log_probs, action[:, None], axis=-1
        ).squeeze(-1)

        # Entropy
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        return action, log_prob, entropy, value

    def get_value(
        self,
        anim_idx: jnp.ndarray,
        elapsed_sinusoidal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get value estimate only."""
        _, value = self(anim_idx, elapsed_sinusoidal)
        return value


def sinusoidal_encoding(elapsed_frames: jnp.ndarray, num_scales: int = 8) -> jnp.ndarray:
    """Compute sinusoidal positional encoding for elapsed frames.

    Args:
        elapsed_frames: Elapsed frames [batch] (raw, not normalized)
        num_scales: Number of frequency scales

    Returns:
        Encoding [batch, num_scales * 2]
    """
    # Normalize to [0, 1] range assuming max ~120 frames
    normalized = elapsed_frames / 120.0

    # Frequencies: 2^0, 2^1, ..., 2^(num_scales-1)
    freqs = 2.0 ** jnp.arange(num_scales)  # [num_scales]

    # Compute angles
    angles = normalized[:, None] * freqs[None, :] * jnp.pi  # [batch, num_scales]

    # Sin and cos
    encoding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

    return encoding


def create_dodge_only_agent(
    key: jax.random.PRNGKey,
    hidden_dim: int = 64,
    anim_embed_dim: int = 32,
    anim_vocab_size: int = 43,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 0.5,
) -> Tuple[DodgeOnlyAgent, TrainState]:
    """Create dodge-only agent and train state.

    Args:
        key: Random key for initialization
        hidden_dim: Hidden layer dimension
        anim_embed_dim: Animation embedding dimension
        anim_vocab_size: Size of animation vocabulary
        learning_rate: Learning rate
        max_grad_norm: Max gradient norm for clipping

    Returns:
        agent: Agent module
        train_state: Flax TrainState
    """
    agent = DodgeOnlyAgent(
        hidden_dim=hidden_dim,
        anim_embed_dim=anim_embed_dim,
        anim_vocab_size=anim_vocab_size,
    )

    # Initialize with dummy inputs
    dummy_anim_idx = jnp.zeros((1,), dtype=jnp.int32)
    dummy_elapsed = jnp.zeros((1, 16), dtype=jnp.float32)

    params = agent.init(key, dummy_anim_idx, dummy_elapsed)

    # Optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )

    train_state = TrainState.create(
        apply_fn=agent.apply,
        params=params,
        tx=tx,
    )

    return agent, train_state
