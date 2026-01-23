"""PPO Training implementation in JAX/Flax."""

from typing import Tuple, NamedTuple, Callable, Optional
from dataclasses import dataclass
from pathlib import Path
import functools
import json
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import numpy as np

from .dodge_window_model import DodgeWindowModel


# Load animation vocab
_VOCAB_PATH = Path(__file__).parent / "anim_vocab.json"
with open(_VOCAB_PATH) as f:
    _ANIM_VOCAB_DATA = json.load(f)
# Skip special tokens like <UNK>
ANIM_VOCAB = {int(k): v for k, v in _ANIM_VOCAB_DATA["vocab"].items() if k.isdigit()}
ANIM_VOCAB_SIZE = _ANIM_VOCAB_DATA["vocab_size"]  # 43 (42 known + 1 UNK at index 0)

# Normalization constants (with buffer from margit_arena_metrics.json)
NORM_ARENA_DIST = 65.0   # MAX_ARENA_DIST ~59.55 + buffer
NORM_Z_DIFF = 7.0        # MAX_Z_DIFF = 6.0 + buffer
NORM_SDF = 15.0          # MAX_SDF ~13.56 + buffer
NORM_ELAPSED = 120.0     # Normalize elapsed frames (typical attack ~2 sec @ 60fps)

# Observation indices based on OBS_KEYS in env_factory.py:
# ["dist_to_boss", "boss_z_relative", "boss_anim_id", "elapsed_frames",
#  "sdf_value", "sdf_normal_x", "sdf_normal_y"]
IDX_DIST_TO_BOSS = 0
IDX_BOSS_Z_REL = 1
IDX_ANIM_ID = 2
IDX_ELAPSED = 3
IDX_SDF_VALUE = 4
IDX_SDF_NORMAL_X = 5
IDX_SDF_NORMAL_Y = 6

# Number of continuous features (excluding anim_id and elapsed_frames)
NUM_CONTINUOUS = 5  # 7 total - 2 (anim_id, elapsed_frames)


def anim_id_to_index(anim_id: int) -> int:
    """Convert raw animation ID to vocab index (0 = unknown)."""
    return ANIM_VOCAB.get(int(anim_id), 0)


def preprocess_obs(obs: jnp.ndarray, debug: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Preprocess flat observation into model inputs.

    Args:
        obs: Flat observation array [batch, 7] from OBS_KEYS
             Note: anim_id should already be converted to vocab index
        debug: Print debug info

    Returns:
        continuous_obs: Normalized continuous features [batch, 5]
        anim_idx: Animation vocab index [batch]
        elapsed_frames: Normalized elapsed frames [batch]
    """
    # Extract special features (anim_id already converted to index in _flatten_obs)
    anim_idx = obs[:, IDX_ANIM_ID].astype(jnp.int32)
    # Clamp to valid range to prevent NaN from out-of-bounds embedding
    anim_idx = jnp.clip(anim_idx, 0, ANIM_VOCAB_SIZE - 1)
    elapsed_frames = obs[:, IDX_ELAPSED] / NORM_ELAPSED

    # Build normalized continuous features
    dist_to_boss = obs[:, IDX_DIST_TO_BOSS:IDX_DIST_TO_BOSS+1] / NORM_ARENA_DIST
    boss_z_rel = obs[:, IDX_BOSS_Z_REL:IDX_BOSS_Z_REL+1] / NORM_Z_DIFF

    # SDF can be inf when outside grid bounds - clamp to reasonable range
    sdf_raw = obs[:, IDX_SDF_VALUE:IDX_SDF_VALUE+1]
    sdf_raw = jnp.clip(sdf_raw, -NORM_SDF, NORM_SDF)  # Clamp before normalizing
    sdf_value = sdf_raw / NORM_SDF

    sdf_normals = obs[:, IDX_SDF_NORMAL_X:IDX_SDF_NORMAL_Y+1]  # already in [-1, 1]

    # Concatenate all continuous features
    continuous_obs = jnp.concatenate([
        dist_to_boss,   # 1
        boss_z_rel,     # 1
        sdf_value,      # 1
        sdf_normals,    # 2
    ], axis=-1)

    if debug:
        print(f"[PREPROCESS] raw obs: {obs[0]}")
        print(f"[PREPROCESS] continuous: {continuous_obs[0]}")
        print(f"[PREPROCESS] anim_idx: {anim_idx[0]}, elapsed: {elapsed_frames[0]}")

    return continuous_obs, anim_idx, elapsed_frames


class RolloutBatch(NamedTuple):
    """Batch of rollout data for PPO update."""
    obs: jnp.ndarray          # [batch, obs_dim]
    actions: jnp.ndarray      # [batch, num_actions] - MultiBinary
    log_probs: jnp.ndarray    # [batch]
    advantages: jnp.ndarray   # [batch]
    returns: jnp.ndarray      # [batch]
    values: jnp.ndarray       # [batch]


class RolloutBuffer:
    """Buffer for storing rollout data during environment interaction."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        num_actions: int,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.ptr = 0

        # Pre-allocate buffers
        self.obs = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, num_actions), dtype=np.float32)  # MultiBinary
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
    ):
        """Add a transition to the buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def reset(self):
        """Reset buffer pointer."""
        self.ptr = 0

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        last_done: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for final state
            last_done: Done flag for final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            returns: Discounted returns [num_steps, num_envs]
            advantages: GAE advantages [num_steps, num_envs]
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # dones[t] = whether transition from step t ended the episode
            # If so, don't bootstrap value from next step
            non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * non_terminal
                - self.values[t]
            )
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * non_terminal * last_gae
            )

        returns = advantages + self.values
        return returns, advantages

    def get_batch(
        self,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> RolloutBatch:
        """Get flattened batch for training.

        Args:
            returns: Computed returns
            advantages: Computed advantages

        Returns:
            RolloutBatch with flattened data
        """
        batch_size = self.num_steps * self.num_envs

        return RolloutBatch(
            obs=jnp.array(self.obs.reshape(batch_size, -1)),
            actions=jnp.array(self.actions.reshape(batch_size, self.num_actions)),  # MultiBinary
            log_probs=jnp.array(self.log_probs.reshape(batch_size)),
            advantages=jnp.array(advantages.reshape(batch_size)),
            returns=jnp.array(returns.reshape(batch_size)),
            values=jnp.array(self.values.reshape(batch_size)),
        )


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Environment
    num_envs: int = 1
    num_steps: int = 2048
    total_timesteps: int = 1_000_000

    # PPO hyperparameters
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    update_epochs: int = 4
    num_minibatches: int = 4
    normalize_advantages: bool = True

    # Network
    hidden_dims: Tuple[int, ...] = (64, 64)

    # Logging
    log_interval: int = 1
    save_interval: int = 10
    eval_interval: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size


def compute_ppo_loss(
    params: dict,
    apply_fn: Callable,
    batch: RolloutBatch,
    clip_eps: float,
    clip_vloss: bool,
    ent_coef: float,
    vf_coef: float,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, dict]:
    """Compute PPO loss.

    Args:
        params: Network parameters
        apply_fn: Network apply function
        batch: Batch of rollout data
        clip_eps: PPO clip epsilon
        clip_vloss: Whether to clip value loss
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        key: PRNG key

    Returns:
        loss: Total loss
        info: Dictionary with loss components
    """
    # Preprocess observations
    continuous_obs, anim_idx, elapsed_frames = preprocess_obs(batch.obs)

    # Get logits and values from network
    logits, new_values = apply_fn(params, continuous_obs, anim_idx, elapsed_frames)

    # Clamp logits to prevent extreme values
    logits = jnp.clip(logits, -20.0, 20.0)

    # Independent Bernoulli: log_sigmoid for numerical stability
    log_prob_per_action = (
        batch.actions * jax.nn.log_sigmoid(logits)
        + (1 - batch.actions) * jax.nn.log_sigmoid(-logits)
    )
    new_log_probs = log_prob_per_action.sum(axis=-1)

    # Bernoulli entropy (clamp probs away from 0/1)
    probs = jnp.clip(jax.nn.sigmoid(logits), 1e-7, 1 - 1e-7)
    entropy_per_action = -probs * jnp.log(probs) - (1 - probs) * jnp.log(1 - probs)
    entropy = entropy_per_action.sum(axis=-1).mean()

    # Policy loss with clipping
    # Clamp old log_probs to prevent -inf causing issues
    old_log_probs = jnp.clip(batch.log_probs, -20.0, 0.0)
    log_ratio = new_log_probs - old_log_probs
    # Clip log_ratio to prevent exp explosion
    log_ratio = jnp.clip(log_ratio, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)

    # Approximate KL for logging
    approx_kl = ((ratio - 1) - log_ratio).mean()

    # Clipped surrogate objective
    pg_loss1 = -batch.advantages * ratio
    pg_loss2 = -batch.advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    if clip_vloss:
        # Clipped value loss
        v_loss_unclipped = (new_values - batch.returns) ** 2
        v_clipped = batch.values + jnp.clip(
            new_values - batch.values, -clip_eps, clip_eps
        )
        v_loss_clipped = (v_clipped - batch.returns) ** 2
        v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
    else:
        v_loss = 0.5 * ((new_values - batch.returns) ** 2).mean()

    # Total loss
    loss = pg_loss - ent_coef * entropy + vf_coef * v_loss

    info = {
        "loss/total": loss,
        "loss/policy": pg_loss,
        "loss/value": v_loss,
        "loss/entropy": entropy,
        "loss/approx_kl": approx_kl,
        "loss/clip_frac": jnp.mean(jnp.abs(ratio - 1) > clip_eps),
    }

    return loss, info


@functools.partial(jax.jit, static_argnums=(3,))
def ppo_update_step(
    train_state: TrainState,
    batch: RolloutBatch,
    clip_eps: float,
    clip_vloss: bool,
    ent_coef: float,
    vf_coef: float,
    key: jax.random.PRNGKey,
) -> Tuple[TrainState, dict]:
    """Single PPO update step.

    Args:
        train_state: Flax TrainState
        batch: Batch of rollout data
        clip_eps: PPO clip epsilon
        clip_vloss: Whether to clip value loss
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        key: PRNG key

    Returns:
        new_train_state: Updated TrainState
        info: Loss information
    """
    def loss_fn(params):
        return compute_ppo_loss(
            params, train_state.apply_fn, batch,
            clip_eps, clip_vloss, ent_coef, vf_coef, key
        )

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, info


class PPOTrainer:
    """PPO Trainer class."""

    def __init__(
        self,
        config: PPOConfig,
        agent,
        train_state: TrainState,
        env,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        key: jax.random.PRNGKey,
        live_obs_plot: bool = False,
        env_config=None,
        dodge_window_model: Optional[DodgeWindowModel] = None,
        dodge_window_reward: float = 1.0,
    ):
        """Initialize PPO trainer.

        Args:
            config: PPO configuration
            agent: Agent module
            train_state: Flax TrainState
            env: Gymnasium environment
            obs_shape: Flattened observation shape
            num_actions: Number of actions
            key: PRNG key
            live_obs_plot: Enable live observation plotting
            env_config: Environment config (for engagement rewards)
            dodge_window_model: Optional learned dodge timing model
            dodge_window_reward: Reward scale for dodging in window (default 1.0)
        """
        self.config = config
        self.agent = agent
        self.train_state = train_state
        self.env = env
        self.key = key
        self.live_obs_plot = live_obs_plot
        self.env_config = env_config
        self.dodge_window_model = dodge_window_model
        self.dodge_window_reward = dodge_window_reward

        # Create rollout buffer
        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_shape=obs_shape,
            num_actions=num_actions,
        )

        # Tracking
        self.global_step = 0
        self.update_count = 0

        # Episode stats (for display)
        self._ep_rewards = []
        self._ep_lengths = []
        self._hits = 0
        self._dodges = 0
        self._good_dodges = 0  # Dodges in window
        self._episodes = 0
        self._rollout_step = 0
        self._start_time = None

        # Live obs plot
        self._obs_fig = None
        self._obs_axes = None
        self._obs_history = {
            'dist_to_boss': [], 'boss_z_rel': [], 'anim_idx': [],
            'elapsed': [], 'sdf_value': [], 'reward': [], 'value': []
        }
        self._max_history = 500  # Keep last N steps
        if self.live_obs_plot:
            self._setup_obs_plot()

    def _setup_obs_plot(self):
        """Setup live observation plot."""
        import matplotlib.pyplot as plt
        plt.ion()
        self._obs_fig, self._obs_axes = plt.subplots(3, 2, figsize=(12, 8))
        self._obs_fig.suptitle('Live Observations')

        # Labels for each subplot
        titles = [
            ('dist_to_boss', 'Distance to Boss'),
            ('sdf_value', 'SDF Value'),
            ('anim_idx', 'Animation Index'),
            ('elapsed', 'Elapsed Frames'),
            ('reward', 'Reward'),
            ('value', 'Value Estimate'),
        ]
        for ax, (key, title) in zip(self._obs_axes.flat, titles):
            ax.set_title(title)
            ax.set_xlabel('Step')

        self._obs_fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def _update_obs_plot(self, obs: np.ndarray, reward: float, value: float):
        """Update live observation plot."""
        if not self.live_obs_plot or self._obs_fig is None:
            return

        import matplotlib.pyplot as plt

        # Add to history
        self._obs_history['dist_to_boss'].append(float(obs[0]))
        self._obs_history['boss_z_rel'].append(float(obs[1]))
        self._obs_history['anim_idx'].append(float(obs[2]))
        self._obs_history['elapsed'].append(float(obs[3]))
        self._obs_history['sdf_value'].append(float(obs[4]))
        self._obs_history['reward'].append(float(reward))
        self._obs_history['value'].append(float(value))

        # Trim to max history
        for k in self._obs_history:
            if len(self._obs_history[k]) > self._max_history:
                self._obs_history[k] = self._obs_history[k][-self._max_history:]

        # Update plots every 10 steps
        if len(self._obs_history['dist_to_boss']) % 10 != 0:
            return

        plot_keys = ['dist_to_boss', 'sdf_value', 'anim_idx', 'elapsed', 'reward', 'value']
        for ax, key in zip(self._obs_axes.flat, plot_keys):
            ax.clear()
            data = self._obs_history[key]
            ax.plot(data, 'b-', linewidth=0.5)
            ax.set_title(f'{key}: {data[-1]:.2f}' if data else key)

            # Mark NaN/inf
            for i, v in enumerate(data):
                if np.isnan(v) or np.isinf(v):
                    ax.axvline(i, color='r', alpha=0.5)

        self._obs_fig.canvas.draw()
        self._obs_fig.canvas.flush_events()
        plt.pause(0.001)

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        """Flatten dictionary observation to array using OBS_KEYS.

        Converts boss_anim_id to vocab index and clamps values.
        """
        from .env_factory import OBS_KEYS

        if isinstance(obs, dict):
            values = []
            for k in OBS_KEYS:
                v = obs[k]
                if k == "boss_anim_id":
                    v = anim_id_to_index(v)  # Convert to vocab index
                elif k == "sdf_value":
                    v = np.clip(v, -NORM_SDF, NORM_SDF)  # Clamp inf
                values.append(v)
            return np.array(values, dtype=np.float32)
        return np.array(obs, dtype=np.float32)

    def _format_action(self, action: np.ndarray) -> str:
        """Format MultiBinary action as readable string."""
        parts = []
        if action[0]:
            parts.append("F")
        if action[1]:
            parts.append("B")
        if action[2]:
            parts.append("L")
        if action[3]:
            parts.append("R")
        if action[4]:
            parts.append("DODGE")
        return "+".join(parts) if parts else "NONE"

    def _get_plot_figure(self):
        """Get the live plot figure from SDFObsWrapper if available."""
        env = self.env
        while env is not None:
            if hasattr(env, 'get_plot_figure'):
                return env.get_plot_figure()
            env = getattr(env, 'env', None)
        return None

    def collect_rollout(self, update_display=None) -> Tuple[np.ndarray, np.ndarray]:
        """Collect rollout data from environment.

        Args:
            update_display: Optional callback to update display

        Returns:
            last_value: Value estimate for final state
            last_done: Done flag for final state
        """
        self.buffer.reset()
        self._rollout_step = 0
        obs_dict, _ = self.env.reset()
        obs = self._flatten_obs(obs_dict)
        last_done = False  # Track if last step was terminal

        # Rollout stats (per-episode)
        ep_reward = 0.0
        ep_len = 0
        ep_hits = 0
        ep_dodges = 0
        ep_good_dodges = 0  # Dodges in window

        for step in range(self.config.num_steps):
            self._rollout_step = step + 1
            self.global_step += self.config.num_envs

            # Get action from policy
            self.key, action_key = jax.random.split(self.key)

            # Add batch dimension and preprocess
            obs_batch = jnp.array(obs).reshape(1, -1)
            continuous_obs, anim_idx, elapsed_frames = preprocess_obs(obs_batch)

            action, log_prob, _, value = self.agent.apply(
                self.train_state.params,
                continuous_obs,
                anim_idx,
                elapsed_frames,
                action_key,
                method=self.agent.get_action_and_value,
            )



            # Remove batch dimension
            action = np.array(action[0])
            log_prob = float(log_prob[0])
            value = float(value[0])

            # Step environment
            next_obs_dict, reward, terminated, truncated, info = self.env.step(action)
            next_obs = self._flatten_obs(next_obs_dict)

            # Add engagement reward based on distance to boss
            if self.env_config is not None:
                dist_to_boss = obs[IDX_DIST_TO_BOSS]
                if dist_to_boss < self.env_config.engage_distance:
                    reward += self.env_config.engage_reward
                else:
                    reward += self.env_config.disengage_penalty

            # Episode ends if: terminated (player died), truncated, or OOB (sdf=inf)
            oob = np.isinf(next_obs[IDX_SDF_VALUE])
            if oob:
                next_obs[IDX_SDF_VALUE] = NORM_SDF  # Clamp for this step's storage
            next_done = terminated or truncated or oob

            # Track stats
            ep_len += 1
            if info.get("player_damage_taken", 0) > 0:
                ep_hits += 1

            # Dodge window reward shaping
            if action[4] > 0:  # Dodge action (index 4)
                ep_dodges += 1
                # Check if dodge is in learned window
                if self.dodge_window_model is not None:
                    anim_idx = int(obs[IDX_ANIM_ID])
                    elapsed_frames = float(obs[IDX_ELAPSED])
                    in_window, window_reward = self.dodge_window_model.should_dodge(
                        anim_idx, elapsed_frames, sigma=2.0
                    )
                    if in_window:
                        reward += self.dodge_window_reward * window_reward
                        ep_good_dodges += 1

            ep_reward += reward

            # Update live obs plot
            self._update_obs_plot(obs, reward, float(value))

            # Update display periodically
            if update_display is not None and step % 20 == 0:
                update_display()

            # Store transition (use next_done for this transition!)
            self.buffer.add(
                obs=obs.reshape(1, -1),
                action=action.reshape(1, -1),  # MultiBinary action
                log_prob=np.array([log_prob]),
                reward=np.array([reward]),
                done=np.array([float(next_done)]),
                value=np.array([value]),
            )

            # Update for next iteration
            obs = next_obs
            obs_dict = next_obs_dict
            last_done = next_done  # Track for return

            if next_done:
                # Natural episode end (rare with HP refund)
                obs_dict, _ = self.env.reset()
                obs = self._flatten_obs(obs_dict)

        # Count each rollout as an episode (since env doesn't naturally terminate)
        self._episodes += 1
        self._ep_rewards.append(ep_reward)
        self._ep_lengths.append(ep_len)
        self._hits = ep_hits              # Per-episode hits
        self._dodges = ep_dodges          # Per-episode dodges
        self._good_dodges = ep_good_dodges  # Per-episode dodges in window

        # Get value estimate for final state
        obs_batch = jnp.array(obs).reshape(1, -1)
        continuous_obs, anim_id, elapsed_frames = preprocess_obs(obs_batch)
        last_value = self.agent.apply(
            self.train_state.params,
            continuous_obs,
            anim_id,
            elapsed_frames,
            method=self.agent.get_value,
        )
        last_value = np.array(last_value)

        return last_value, np.array([float(last_done)])

    def update(self, returns: np.ndarray, advantages: np.ndarray) -> dict:
        """Perform PPO update.

        Args:
            returns: Computed returns
            advantages: Computed advantages

        Returns:
            info: Training metrics
        """
        # Normalize advantages
        if self.config.normalize_advantages:
            adv_std = advantages.std()
            if adv_std < 1e-8:
                print(f"[WARN] Advantage std near zero: {adv_std}, skipping normalization")
            else:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # Check for NaN in buffer before creating batch
        if np.any(np.isnan(self.buffer.obs)):
            print(f"[NAN] buffer.obs has NaN!")
        if np.any(np.isnan(self.buffer.rewards)):
            print(f"[NAN] buffer.rewards has NaN! range=[{self.buffer.rewards.min()}, {self.buffer.rewards.max()}]")
        if np.any(np.isnan(self.buffer.values)):
            print(f"[NAN] buffer.values has NaN!")
        if np.any(np.isnan(returns)):
            print(f"[NAN] returns has NaN!")
        if np.any(np.isnan(advantages)):
            print(f"[NAN] advantages has NaN!")

        # Get batch
        batch = self.buffer.get_batch(returns, advantages)
        batch_size = self.config.batch_size

        # Accumulate metrics
        all_info = {}

        for epoch in range(self.config.update_epochs):
            # Shuffle batch
            self.key, shuffle_key = jax.random.split(self.key)
            perm = jax.random.permutation(shuffle_key, batch_size)

            # Update in minibatches
            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = perm[start:end]

                mb_batch = RolloutBatch(
                    obs=batch.obs[mb_indices],
                    actions=batch.actions[mb_indices],
                    log_probs=batch.log_probs[mb_indices],
                    advantages=batch.advantages[mb_indices],
                    returns=batch.returns[mb_indices],
                    values=batch.values[mb_indices],
                )

                self.key, update_key = jax.random.split(self.key)
                self.train_state, info = ppo_update_step(
                    self.train_state,
                    mb_batch,
                    self.config.clip_eps,
                    self.config.clip_vloss,
                    self.config.ent_coef,
                    self.config.vf_coef,
                    update_key,
                )

                # Accumulate metrics
                for k, v in info.items():
                    if k not in all_info:
                        all_info[k] = []
                    all_info[k].append(float(v))

        # Average metrics
        avg_info = {k: np.mean(v) for k, v in all_info.items()}
        self.update_count += 1

        return avg_info

    def train_step(self) -> dict:
        """Perform one training iteration (collect + update).

        Returns:
            info: Training metrics
        """
        # Collect rollout
        last_value, last_done = self.collect_rollout()

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value.flatten(),
            last_done.flatten(),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Update policy
        info = self.update(returns, advantages)
        info["global_step"] = self.global_step
        info["update"] = self.update_count

        return info

    def train(
        self,
        callback: Optional[Callable[[dict], None]] = None,
        wandb_run=None,
    ) -> TrainState:
        """Run full training loop with rich progress display."""
        from rich.live import Live
        from rich.table import Table
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
        from rich.panel import Panel
        from rich.console import Group
        import time

        num_updates = self.config.num_updates

        # Stats tracking
        self._ep_rewards = []
        self._ep_lengths = []
        self._hits = 0
        self._dodges = 0
        self._good_dodges = 0
        self._episodes = 0
        self._start_time = time.time()
        self._rollout_step = 0

        # Create progress bars
        progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        train_task = progress.add_task("[cyan]Training", total=num_updates)
        rollout_task = progress.add_task("[green]Rollout", total=self.config.num_steps)

        # Stats storage for display
        current_stats = {
            "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "reward": 0.0, "hits": 0, "dodges": 0, "good_dodges": 0, "episodes": 0,
            "return_mean": 0.0, "adv_std": 0.0, "kl": 0.0,
        }

        def make_stats_table():
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="cyan", width=12)
            table.add_column(style="yellow", width=10)
            table.add_column(style="cyan", width=12)
            table.add_column(style="yellow", width=10)

            table.add_row(
                "Loss:", f"{current_stats['loss']:.4f}",
                "Policy:", f"{current_stats['policy_loss']:.4f}",
            )
            table.add_row(
                "Value:", f"{current_stats['value_loss']:.4f}",
                "Entropy:", f"{current_stats['entropy']:.4f}",
            )
            table.add_row(
                "Return:", f"{current_stats['return_mean']:+.2f}",
                "Adv Std:", f"{current_stats['adv_std']:.4f}",
            )
            table.add_row(
                "Episodes:", f"{current_stats['episodes']}",
                "KL:", f"{current_stats['kl']:.4f}",
            )
            table.add_row(
                "Hits/ep:", f"[red]{current_stats['hits']}[/red]",
                "Dodges/ep:", f"{current_stats['dodges']}",
            )
            table.add_row(
                "Good dodges:", f"[green]{current_stats['good_dodges']}[/green]",
                "", "",
            )
            return table

        def make_display():
            return Panel(
                Group(progress, make_stats_table()),
                title="[bold blue]Dodge Policy Training[/bold blue]",
                border_style="blue",
            )

        with Live(make_display(), refresh_per_second=4) as live:
            for update in range(1, num_updates + 1):
                # Reset rollout progress
                progress.reset(rollout_task)

                # Collect rollout with progress updates
                self._rollout_step = 0

                def update_progress():
                    progress.update(rollout_task, completed=self._rollout_step)
                    live.update(make_display())

                last_value, last_done = self.collect_rollout(update_display=update_progress)

                # Compute returns and advantages
                returns, advantages = self.buffer.compute_returns_and_advantages(
                    last_value.flatten(),
                    last_done.flatten(),
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )

                # Update policy
                info = self.update(returns, advantages)
                info["global_step"] = self.global_step
                info["update"] = self.update_count

                # Update training progress
                progress.update(train_task, completed=update)

                # Update stats for display
                current_stats.update({
                    "loss": float(info['loss/total']),
                    "policy_loss": float(info['loss/policy']),
                    "value_loss": float(info['loss/value']),
                    "entropy": float(info['loss/entropy']),
                    "reward": float(np.mean(self._ep_rewards[-10:])) if self._ep_rewards else 0.0,
                    "return_mean": float(returns.mean()),
                    "adv_std": float(advantages.std()),
                    "kl": float(info['loss/approx_kl']),
                    "hits": self._hits,
                    "dodges": self._dodges,
                    "good_dodges": self._good_dodges,
                    "episodes": self._episodes,
                })
                live.update(make_display())

                # Wandb logging (per episode = per update)
                if wandb_run is not None:
                    ep_reward = self._ep_rewards[-1] if self._ep_rewards else 0.0
                    wandb_run.log({
                        # Episode metrics (primary)
                        "episode/reward": ep_reward,
                        "episode/hits": self._hits,
                        "episode/dodges": self._dodges,
                        "episode/good_dodges": self._good_dodges,
                        # Training diagnostics
                        "train/gae_return_mean": float(returns.mean()),
                        # Losses
                        "loss/total": float(info["loss/total"]),
                        "loss/policy": float(info["loss/policy"]),
                        "loss/value": float(info["loss/value"]),
                        "loss/entropy": float(info["loss/entropy"]),
                        "loss/kl": float(info["loss/approx_kl"]),
                        # Value estimates
                        "value/mean": float(self.buffer.values.mean()),
                        "value/std": float(self.buffer.values.std()),
                    }, step=self._episodes)

                if callback is not None:
                    callback(info)

        return self.train_state
