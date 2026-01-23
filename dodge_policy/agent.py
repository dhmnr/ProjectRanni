"""PPO Agent network implemented in Flax."""

from typing import Tuple, NamedTuple, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np


class AgentState(NamedTuple):
    """Container for agent parameters and optimizer state."""
    params: dict
    opt_state: optax.OptState


def orthogonal_init(scale: float = np.sqrt(2)):
    """Orthogonal initialization matching PyTorch's implementation."""
    return nn.initializers.orthogonal(scale)


def sinusoidal_encoding(x: jnp.ndarray, num_scales: int = 16) -> jnp.ndarray:
    """Sinusoidal positional encoding for temporal values.

    Args:
        x: Input values [batch] or [batch, 1], assumed normalized to ~[0, 1]
        num_scales: Number of frequency scales (output dim = 2 * num_scales)

    Returns:
        Encoded values [batch, 2 * num_scales]
    """
    x = jnp.atleast_1d(x)
    if x.ndim == 1:
        x = x[:, None]

    # Frequencies: 2^0, 2^1, ..., 2^(num_scales-1)
    freqs = 2.0 ** jnp.arange(num_scales)
    # Scale to reasonable range
    angles = x * freqs * jnp.pi

    # Concatenate sin and cos
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class Actor(nn.Module):
    """Actor network for MultiBinary action space.

    Outputs logits for independent Bernoulli distributions.
    Handles continuous obs, anim_idx embedding, and elapsed_frames sinusoidal encoding.
    """
    num_actions: int
    hidden_dims: Sequence[int] = (128, 128)
    anim_embed_dim: int = 32
    anim_vocab_size: int = 43  # Default for Margit phase 1
    sinusoidal_scales: int = 8

    @nn.compact
    def __call__(self, continuous_obs: jnp.ndarray, anim_idx: jnp.ndarray, elapsed_frames: jnp.ndarray) -> jnp.ndarray:
        batch_size = continuous_obs.shape[0]

        # 1. Continuous observations (already normalized)
        x_cont = continuous_obs

        # 2. Animation embedding (anim_idx already converted via vocab lookup)
        anim_embed = nn.Embed(num_embeddings=self.anim_vocab_size, features=self.anim_embed_dim)(anim_idx)
        anim_embed = anim_embed.reshape(batch_size, -1)

        # 3. Elapsed frames sinusoidal encoding
        elapsed_enc = sinusoidal_encoding(elapsed_frames, self.sinusoidal_scales)

        # Concatenate all features
        x = jnp.concatenate([x_cont, anim_embed, elapsed_enc], axis=-1)

        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=orthogonal_init())(x)
            x = nn.tanh(x)

        # Output logits for independent Bernoulli distributions
        logits = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal_init(0.01)
        )(x)

        return logits


class Critic(nn.Module):
    """Critic network for value estimation."""
    hidden_dims: Sequence[int] = (128, 128)
    anim_embed_dim: int = 32
    anim_vocab_size: int = 43  # Default for Margit phase 1
    sinusoidal_scales: int = 8

    @nn.compact
    def __call__(self, continuous_obs: jnp.ndarray, anim_idx: jnp.ndarray, elapsed_frames: jnp.ndarray) -> jnp.ndarray:
        batch_size = continuous_obs.shape[0]

        # 1. Continuous observations (already normalized)
        x_cont = continuous_obs

        # 2. Animation embedding (anim_idx already converted via vocab lookup)
        anim_embed = nn.Embed(num_embeddings=self.anim_vocab_size, features=self.anim_embed_dim)(anim_idx)
        anim_embed = anim_embed.reshape(batch_size, -1)

        # 3. Elapsed frames sinusoidal encoding
        elapsed_enc = sinusoidal_encoding(elapsed_frames, self.sinusoidal_scales)

        # Concatenate all features
        x = jnp.concatenate([x_cont, anim_embed, elapsed_enc], axis=-1)

        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=orthogonal_init())(x)
            x = nn.tanh(x)

        value = nn.Dense(1, kernel_init=orthogonal_init(1.0))(x)
        return value.squeeze(-1)


class Agent(nn.Module):
    """Combined Actor-Critic agent for PPO with MultiBinary actions.

    Expects structured input:
        - continuous_obs: normalized continuous features [batch, num_continuous]
        - anim_idx: animation vocab index [batch] (will be embedded)
        - elapsed_frames: normalized elapsed frames [batch] (sinusoidal encoding)

    Actions (MultiBinary 5):
        [forward, backward, left, right, dodge]
    """
    num_actions: int
    hidden_dims: Sequence[int] = (128, 128)
    anim_embed_dim: int = 32
    anim_vocab_size: int = 43  # Default for Margit phase 1
    sinusoidal_scales: int = 8

    def setup(self):
        self.actor = Actor(
            self.num_actions,
            self.hidden_dims,
            self.anim_embed_dim,
            self.anim_vocab_size,
            self.sinusoidal_scales,
        )
        self.critic = Critic(
            self.hidden_dims,
            self.anim_embed_dim,
            self.anim_vocab_size,
            self.sinusoidal_scales,
        )

    def __call__(
        self,
        continuous_obs: jnp.ndarray,
        anim_id: jnp.ndarray,
        elapsed_frames: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning logits and value."""
        logits = self.actor(continuous_obs, anim_id, elapsed_frames)
        value = self.critic(continuous_obs, anim_id, elapsed_frames)
        return logits, value

    def get_value(
        self,
        continuous_obs: jnp.ndarray,
        anim_id: jnp.ndarray,
        elapsed_frames: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get value estimate only."""
        return self.critic(continuous_obs, anim_id, elapsed_frames)

    def get_action_and_value(
        self,
        continuous_obs: jnp.ndarray,
        anim_id: jnp.ndarray,
        elapsed_frames: jnp.ndarray,
        key: jax.random.PRNGKey,
        action: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get action, log probability, entropy, and value.

        Uses independent Bernoulli distributions for MultiBinary actions.

        Args:
            continuous_obs: Normalized continuous observations [batch, num_continuous]
            anim_id: Boss animation ID [batch]
            elapsed_frames: Normalized elapsed frames [batch]
            key: PRNG key for sampling
            action: Optional action [batch, num_actions] (for computing log prob)

        Returns:
            action: Sampled or provided action [batch, num_actions]
            log_prob: Log probability of action [batch]
            entropy: Entropy of distribution [batch]
            value: Value estimate [batch]
        """
        logits, value = self(continuous_obs, anim_id, elapsed_frames)

        # Clamp logits to prevent extreme probabilities
        logits = jnp.clip(logits, -20.0, 20.0)

        # Independent Bernoulli for each action
        probs = jax.nn.sigmoid(logits)

        if action is None:
            # Sample from Bernoulli
            action = jax.random.bernoulli(key, probs).astype(jnp.float32)

        # Numerically stable log probability using log_sigmoid
        log_prob_per_action = (
            action * jax.nn.log_sigmoid(logits)
            + (1 - action) * jax.nn.log_sigmoid(-logits)
        )
        log_prob = log_prob_per_action.sum(axis=-1)
        log_prob = jnp.clip(log_prob, -20.0, 0.0)

        # Entropy (clamp probs away from 0/1)
        probs_clamped = jnp.clip(probs, 1e-7, 1 - 1e-7)
        entropy_per_action = -probs_clamped * jnp.log(probs_clamped) - (1 - probs_clamped) * jnp.log(1 - probs_clamped)
        entropy = entropy_per_action.sum(axis=-1)

        return action, log_prob, entropy, value


def create_agent(
    key: jax.random.PRNGKey,
    num_continuous: int,
    num_actions: int,
    learning_rate: float = 2.5e-4,
    hidden_dims: Sequence[int] = (64, 64),
    max_grad_norm: float = 0.5,
    anim_embed_dim: int = 32,
    anim_vocab_size: int = 64,
    sinusoidal_scales: int = 16,
) -> Tuple[Agent, TrainState]:
    """Create agent and training state.

    Args:
        key: PRNG key for initialization
        num_continuous: Number of continuous observation features
        num_actions: Number of binary actions
        learning_rate: Learning rate for optimizer
        hidden_dims: Hidden layer dimensions
        max_grad_norm: Maximum gradient norm for clipping
        anim_embed_dim: Dimension of animation embedding
        anim_vocab_size: Vocabulary size for animation embedding
        sinusoidal_scales: Number of scales for sinusoidal encoding

    Returns:
        agent: Agent module
        train_state: Flax TrainState with params and optimizer
    """
    agent = Agent(
        num_actions=num_actions,
        hidden_dims=hidden_dims,
        anim_embed_dim=anim_embed_dim,
        anim_vocab_size=anim_vocab_size,
        sinusoidal_scales=sinusoidal_scales,
    )

    # Initialize with dummy inputs (3 separate inputs)
    dummy_continuous = jnp.zeros((1, num_continuous))
    dummy_anim_id = jnp.zeros((1,), dtype=jnp.int32)
    dummy_elapsed = jnp.zeros((1,))
    params = agent.init(key, dummy_continuous, dummy_anim_id, dummy_elapsed)

    # Create optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    train_state = TrainState.create(
        apply_fn=agent.apply,
        params=params,
        tx=tx,
    )

    return agent, train_state
