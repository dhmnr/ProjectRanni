"""World Model Reward Shaper for dodge-only training.

Uses the learned world model to provide reward shaping based on P(hit).
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


class WorldModel(nn.Module):
    """World model that predicts P(hit) given state and action."""
    n_boss_anims: int = 64
    n_actions: int = 4
    hidden_dim: int = 128
    embed_dim: int = 32

    @nn.compact
    def __call__(self, boss_anim_idx, elapsed_norm, dist_norm, action):
        boss_emb = nn.Embed(self.n_boss_anims, self.embed_dim)(boss_anim_idx)
        action_emb = nn.Embed(self.n_actions, 8)(action)
        if elapsed_norm.ndim == 1:
            elapsed_norm = elapsed_norm[:, None]
        if dist_norm.ndim == 1:
            dist_norm = dist_norm[:, None]
        x = jnp.concatenate([boss_emb, elapsed_norm, dist_norm, action_emb], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return {
            'next_anim_logits': nn.Dense(self.n_boss_anims)(x),
            'next_elapsed': nn.sigmoid(nn.Dense(1)(x).squeeze(-1)),
            'next_dist': nn.sigmoid(nn.Dense(1)(x).squeeze(-1)),
            'hit_logit': nn.Dense(1)(x).squeeze(-1),
            'damage_dealt': nn.relu(nn.Dense(1)(x).squeeze(-1)),
        }


@dataclass
class WorldModelRewardConfig:
    """Configuration for world model reward shaping."""
    model_path: str = "world_model_v2.npz"

    # Reward shaping coefficients
    danger_penalty: float = -1.0      # Penalty for being in high P(hit) state
    dodge_bonus: float = 1.0          # Bonus for dodging when it reduces P(hit)

    # Thresholds
    danger_threshold: float = 0.3     # P(hit) above this is "dangerous"
    dodge_benefit_threshold: float = 0.05  # Minimum P(hit) reduction to reward dodge


class WorldModelRewardShaper:
    """Provides reward shaping based on world model predictions."""

    def __init__(self, config: WorldModelRewardConfig):
        self.config = config
        self.model, self.params, self.n_anims = self._load_model(config.model_path)

        # JIT compile the prediction function
        self._predict_hit = jax.jit(self._predict_hit_impl)

    def _load_model(self, path: str) -> Tuple[WorldModel, any, int]:
        """Load world model from checkpoint."""
        data = np.load(path, allow_pickle=True)
        n_anims = int(data['n_anims'][0])

        model = WorldModel(n_boss_anims=n_anims, n_actions=4)

        # Initialize to get tree structure
        rng = jax.random.PRNGKey(0)
        dummy_i = jnp.zeros((1,), dtype=jnp.int32)
        dummy_f = jnp.zeros((1,), dtype=jnp.float32)
        init_params = model.init(rng, dummy_i, dummy_f, dummy_f, dummy_i)
        _, tree_def = jax.tree_util.tree_flatten(init_params)

        # Load params
        param_arrays = []
        i = 0
        while f'param_{i}' in data:
            param_arrays.append(data[f'param_{i}'])
            i += 1
        params = jax.tree_util.tree_unflatten(tree_def, param_arrays)

        print(f"Loaded world model from {path}")
        print(f"  Boss animations: {n_anims}")

        return model, params, n_anims

    def _predict_hit_impl(self, params, anim_idx, elapsed_norm, dist_norm, action):
        """Predict P(hit) for given state-action."""
        preds = self.model.apply(params, anim_idx, elapsed_norm, dist_norm, action)
        return jax.nn.sigmoid(preds['hit_logit'])

    def get_hit_prob(
        self,
        anim_idx: int,
        elapsed_frames: float,
        dist_to_boss: float,
        action: int,
    ) -> float:
        """Get P(hit) for a single state-action pair.

        Args:
            anim_idx: Boss animation vocab index
            elapsed_frames: Frames since animation started
            dist_to_boss: Distance to boss
            action: 0=nothing, 1=dodge

        Returns:
            Probability of getting hit
        """
        # Normalize inputs
        elapsed_norm = elapsed_frames / 120.0
        dist_norm = min(dist_to_boss / 10.0, 1.0)

        # Predict
        p_hit = self._predict_hit(
            self.params,
            jnp.array([anim_idx]),
            jnp.array([elapsed_norm]),
            jnp.array([dist_norm]),
            jnp.array([action]),
        )
        return float(p_hit[0])

    def compute_reward_shaping(
        self,
        anim_idx: int,
        elapsed_frames: float,
        dist_to_boss: float,
        action: int,
    ) -> Tuple[float, dict]:
        """Compute reward shaping for a single step.

        Args:
            anim_idx: Boss animation vocab index
            elapsed_frames: Frames since animation started
            dist_to_boss: Distance to boss
            action: 0=nothing, 1=dodge

        Returns:
            Tuple of (reward_shaping, info_dict)
        """
        # Get P(hit) for both actions
        p_hit_no_dodge = self.get_hit_prob(anim_idx, elapsed_frames, dist_to_boss, 0)
        p_hit_dodge = self.get_hit_prob(anim_idx, elapsed_frames, dist_to_boss, 1)

        reward = 0.0
        info = {
            'wm_p_hit_no_dodge': p_hit_no_dodge,
            'wm_p_hit_dodge': p_hit_dodge,
            'wm_dodge_benefit': p_hit_no_dodge - p_hit_dodge,
        }

        if action == 0:  # No dodge
            # Penalize being in danger without dodging
            if p_hit_no_dodge > self.config.danger_threshold:
                reward += self.config.danger_penalty * p_hit_no_dodge
                info['wm_danger_penalty'] = reward
        else:  # Dodge
            # Reward dodging when it reduces P(hit)
            dodge_benefit = p_hit_no_dodge - p_hit_dodge
            if dodge_benefit > self.config.dodge_benefit_threshold:
                reward += self.config.dodge_bonus * dodge_benefit
                info['wm_dodge_bonus'] = reward

        info['wm_reward'] = reward
        return reward, info

    def compute_batch_rewards(
        self,
        anim_idx: np.ndarray,
        elapsed_frames: np.ndarray,
        dist_to_boss: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute reward shaping for a batch of transitions.

        Args:
            anim_idx: [T] Boss animation vocab indices
            elapsed_frames: [T] Frames since animation started
            dist_to_boss: [T] Distance to boss
            actions: [T] Actions taken (0=nothing, 1=dodge)

        Returns:
            [T] Reward shaping values
        """
        T = len(anim_idx)

        # Normalize
        elapsed_norm = elapsed_frames / 120.0
        dist_norm = np.clip(dist_to_boss / 10.0, 0, 1)

        # Get P(hit) for both actions
        p_hit_no_dodge = np.array(self._predict_hit(
            self.params,
            jnp.array(anim_idx),
            jnp.array(elapsed_norm),
            jnp.array(dist_norm),
            jnp.zeros(T, dtype=jnp.int32),
        ))

        p_hit_dodge = np.array(self._predict_hit(
            self.params,
            jnp.array(anim_idx),
            jnp.array(elapsed_norm),
            jnp.array(dist_norm),
            jnp.ones(T, dtype=jnp.int32),
        ))

        # Compute rewards
        rewards = np.zeros(T, dtype=np.float32)

        # Danger penalty for not dodging when dangerous
        no_dodge_mask = actions == 0
        danger_mask = p_hit_no_dodge > self.config.danger_threshold
        rewards[no_dodge_mask & danger_mask] += (
            self.config.danger_penalty * p_hit_no_dodge[no_dodge_mask & danger_mask]
        )

        # Dodge bonus when it helps
        dodge_mask = actions == 1
        dodge_benefit = p_hit_no_dodge - p_hit_dodge
        benefit_mask = dodge_benefit > self.config.dodge_benefit_threshold
        rewards[dodge_mask & benefit_mask] += (
            self.config.dodge_bonus * dodge_benefit[dodge_mask & benefit_mask]
        )

        return rewards


def load_world_model_shaper(
    model_path: str,
    danger_penalty: float = -1.0,
    dodge_bonus: float = 1.0,
    danger_threshold: float = 0.3,
) -> WorldModelRewardShaper:
    """Convenience function to load world model reward shaper."""
    config = WorldModelRewardConfig(
        model_path=model_path,
        danger_penalty=danger_penalty,
        dodge_bonus=dodge_bonus,
        danger_threshold=danger_threshold,
    )
    return WorldModelRewardShaper(config)
