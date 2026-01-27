"""Online RUDDER reward shaping for PPO.

Computes credit assignment from RUDDER model and uses it to shape rewards.
Updates RUDDER model continuously during training.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, Optional, Any
from dataclasses import dataclass
import optax
from flax.training.train_state import TrainState

from .rudder_model import RudderLSTM, RudderConfig, load_model


@dataclass
class RudderRewardConfig:
    """Configuration for RUDDER reward shaping."""
    model_path: str = "rudder_model"  # Path to pre-trained model
    credit_scale: float = 1.0  # Scale factor for RUDDER credit
    update_freq: int = 1  # Update RUDDER every N rollouts
    update_epochs: int = 5  # Epochs per RUDDER update
    learning_rate: float = 1e-4  # Learning rate for online updates
    min_segment_length: int = 5  # Minimum attack segment length
    max_segment_length: int = 64  # Maximum attack segment length


class RudderRewardShaper:
    """Online RUDDER reward shaping.

    Computes credit assignment during PPO rollouts and optionally
    updates the RUDDER model with new experience.
    """

    def __init__(self, config: RudderRewardConfig):
        """Initialize RUDDER reward shaper.

        Args:
            config: RUDDER reward configuration
        """
        self.config = config

        # Load pre-trained model
        print(f"Loading RUDDER model from {config.model_path}...")
        self.model, self.params, self.model_config, self.stats = load_model(config.model_path)

        # Check mode
        self.mode = self.stats.get('mode', 'full_episode')
        self.max_len = self.stats.get('max_len', 64)
        print(f"  Mode: {self.mode}")
        print(f"  Max sequence length: {self.max_len}")

        # Create optimizer for online updates
        self.tx = optax.adam(config.learning_rate)
        self.opt_state = self.tx.init(self.params)

        # Tracking
        self.update_count = 0
        self.rollout_count = 0

        # Buffer for collecting training data
        self.trajectory_buffer = []

    def compute_credit(
        self,
        boss_anim_ids: np.ndarray,
        hero_anim_ids: np.ndarray,
        dist_to_boss: np.ndarray,
        hero_hp: np.ndarray,
        actions: np.ndarray,
        damage_taken: np.ndarray,
    ) -> np.ndarray:
        """Compute RUDDER credit for a rollout.

        Segments rollout into attacks and computes credit per-attack.

        Args:
            boss_anim_ids: (T,) boss animation IDs
            hero_anim_ids: (T,) hero animation IDs
            dist_to_boss: (T,) distance to boss
            hero_hp: (T,) hero HP
            actions: (T,) dodge actions (0 or 1 for dodge-only, or index 4 for full action)
            damage_taken: (T,) damage taken per step

        Returns:
            credit: (T,) credit at each timestep (already scaled)
        """
        T = len(boss_anim_ids)

        # Normalize continuous features
        dist = (dist_to_boss - self.stats['dist_mean']) / self.stats['dist_std']
        hp = (hero_hp - self.stats['hp_mean']) / self.stats['hp_std']
        act = actions.astype(np.float32)

        # Stack continuous features
        cont_stack = np.stack([dist, hp, act], axis=-1).astype(np.float32)

        # Collect all attack segments
        segments = []
        start_idx = 0

        for t in range(1, T + 1):
            is_boundary = (t == T) or (boss_anim_ids[t] != boss_anim_ids[t-1])

            if is_boundary:
                end_idx = t
                length = end_idx - start_idx

                if length >= self.config.min_segment_length:
                    seg_boss = int(boss_anim_ids[start_idx])
                    seg_hero = hero_anim_ids[start_idx:end_idx].copy()
                    seg_cont = cont_stack[start_idx:end_idx].copy()

                    # Pad or truncate
                    if length > self.max_len:
                        seg_hero = seg_hero[:self.max_len]
                        seg_cont = seg_cont[:self.max_len]
                    elif length < self.max_len:
                        pad_len = self.max_len - length
                        seg_hero = np.pad(seg_hero, (0, pad_len), mode='edge')
                        seg_cont = np.pad(seg_cont, ((0, pad_len), (0, 0)), mode='edge')

                    segments.append((start_idx, min(length, self.max_len), seg_boss, seg_hero, seg_cont))

                start_idx = t

        if not segments:
            return np.zeros(T, dtype=np.float32)

        # Batch inference
        n_segs = len(segments)
        batch_boss = np.array([s[2] for s in segments], dtype=np.int32)
        batch_hero = np.stack([s[3] for s in segments], axis=0)
        batch_cont = np.stack([s[4] for s in segments], axis=0)

        # Run model
        all_preds = np.array(self.model.apply(
            {'params': self.params},
            jnp.array(batch_boss),
            jnp.array(batch_hero),
            jnp.array(batch_cont),
        ))

        # Compute credit and scatter back
        all_credit = np.zeros(T, dtype=np.float32)

        for i, (start_idx, store_len, _, _, _) in enumerate(segments):
            preds = all_preds[i]

            # Credit = diff of predictions
            credit = np.zeros(self.max_len)
            credit[0] = preds[0]
            credit[1:] = preds[1:] - preds[:-1]

            # Denormalize and scale
            credit = credit * self.stats['return_std'] * self.config.credit_scale

            all_credit[start_idx:start_idx + store_len] = credit[:store_len]

        return all_credit

    def add_to_buffer(
        self,
        boss_anim_ids: np.ndarray,
        hero_anim_ids: np.ndarray,
        dist_to_boss: np.ndarray,
        hero_hp: np.ndarray,
        actions: np.ndarray,
        damage_taken: np.ndarray,
    ):
        """Add rollout data to buffer for RUDDER training.

        Segments into attacks and stores for later training.
        """
        T = len(boss_anim_ids)

        # Segment into attacks
        start_idx = 0
        for t in range(1, T + 1):
            is_boundary = (t == T) or (boss_anim_ids[t] != boss_anim_ids[t-1])

            if is_boundary:
                end_idx = t
                length = end_idx - start_idx

                if length >= self.config.min_segment_length:
                    seg_damage = damage_taken[start_idx:end_idx]
                    seg_actions = actions[start_idx:end_idx]

                    got_hit = seg_damage.sum() > 0
                    has_dodge = seg_actions.sum() > 0

                    # Only keep trajectories with learning signal
                    if got_hit or has_dodge:
                        self.trajectory_buffer.append({
                            'boss_anim_id': int(boss_anim_ids[start_idx]),
                            'hero_anim_id': hero_anim_ids[start_idx:end_idx].copy(),
                            'dist_to_boss': dist_to_boss[start_idx:end_idx].copy(),
                            'hero_hp': hero_hp[start_idx:end_idx].copy(),
                            'actions': actions[start_idx:end_idx].copy(),
                            'target': -1.0 if got_hit else 0.0,
                            'length': length,
                        })

                start_idx = t

    def update_model(self, max_trajectories: int = 256, min_trajectories: int = 8) -> dict:
        """Update RUDDER model with buffered trajectories.

        Args:
            max_trajectories: Maximum trajectories to use per update
            min_trajectories: Minimum trajectories needed to update

        Returns:
            info: Training metrics
        """
        if len(self.trajectory_buffer) < min_trajectories:
            return {
                'rudder_loss': 0.0,
                'rudder_trajectories': 0,
                'rudder_buffer_size': len(self.trajectory_buffer),
            }

        # Sample trajectories
        n_traj = min(len(self.trajectory_buffer), max_trajectories)
        indices = np.random.choice(len(self.trajectory_buffer), n_traj, replace=False)
        trajectories = [self.trajectory_buffer[i] for i in indices]

        # Convert to arrays
        batch_boss = np.array([t['boss_anim_id'] for t in trajectories], dtype=np.int32)
        batch_hero = np.zeros((n_traj, self.max_len), dtype=np.int32)
        batch_cont = np.zeros((n_traj, self.max_len, 3), dtype=np.float32)
        batch_targets = np.zeros(n_traj, dtype=np.float32)

        for i, traj in enumerate(trajectories):
            length = min(traj['length'], self.max_len)

            # Pad or truncate
            hero = traj['hero_anim_id'][:length]
            if length < self.max_len:
                hero = np.pad(hero, (0, self.max_len - length), mode='edge')
            batch_hero[i] = hero

            # Normalize and stack continuous
            dist = (traj['dist_to_boss'][:length] - self.stats['dist_mean']) / self.stats['dist_std']
            hp = (traj['hero_hp'][:length] - self.stats['hp_mean']) / self.stats['hp_std']
            act = traj['actions'][:length].astype(np.float32)

            cont = np.stack([dist, hp, act], axis=-1)
            if length < self.max_len:
                cont = np.pad(cont, ((0, self.max_len - length), (0, 0)), mode='edge')
            batch_cont[i] = cont

            # Normalize target
            batch_targets[i] = (traj['target'] - self.stats['return_mean']) / self.stats['return_std']

        # Convert to JAX arrays
        batch_boss = jnp.array(batch_boss)
        batch_hero = jnp.array(batch_hero)
        batch_cont = jnp.array(batch_cont)
        batch_targets = jnp.array(batch_targets)

        # Training loop
        total_loss = 0.0
        for epoch in range(self.config.update_epochs):
            loss, grads = self._compute_loss_and_grad(
                self.params, batch_boss, batch_hero, batch_cont, batch_targets
            )
            updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)
            total_loss += float(loss)

        avg_loss = total_loss / self.config.update_epochs
        self.update_count += 1

        # Clear old trajectories (keep some for diversity)
        if len(self.trajectory_buffer) > max_trajectories * 2:
            self.trajectory_buffer = self.trajectory_buffer[-max_trajectories:]

        return {
            'rudder_loss': avg_loss,
            'rudder_trajectories': n_traj,
            'rudder_buffer_size': len(self.trajectory_buffer),
        }

    @staticmethod
    @jax.jit
    def _compute_loss_and_grad_static(model, params, boss, hero, cont, targets):
        """JIT-compiled loss computation."""
        def loss_fn(params):
            preds = model.apply({'params': params}, boss, hero, cont)
            final_pred = preds[:, -1]
            loss = jnp.mean((final_pred - targets) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    def _compute_loss_and_grad(self, params, boss, hero, cont, targets):
        """Compute loss and gradients."""
        def loss_fn(params):
            preds = self.model.apply({'params': params}, boss, hero, cont)
            final_pred = preds[:, -1]
            loss = jnp.mean((final_pred - targets) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    def save(self, path: str):
        """Save updated RUDDER model."""
        import json

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save params
        params_dict = {}
        for p, value in jax.tree_util.tree_leaves_with_path(self.params):
            key = '/'.join(str(x.key) if hasattr(x, 'key') else str(x.idx) for x in p)
            params_dict[key] = np.array(value)
        np.savez(save_dir / "params.npz", **params_dict)

        # Save config and stats
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self.model_config.to_dict(), f, indent=2)

        with open(save_dir / "stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"Saved RUDDER model to {save_dir}")

    def generate_credit_visualization(
        self,
        data_dir: str = "rudder_data",
        gt_model_path: str = "dodge_windows.json",
        max_episodes: int = 10,
    ):
        """Generate credit vs ground truth visualization.

        Args:
            data_dir: Directory with episode data
            gt_model_path: Path to ground truth DodgeWindowModel
            max_episodes: Max episodes to process

        Returns:
            matplotlib figure for wandb logging
        """
        import matplotlib.pyplot as plt
        from .dodge_window_model import DodgeWindowModel
        from .ppo import anim_id_to_index

        # Colorblind-friendly palette
        COLOR_POSITIVE = '#0072B2'
        COLOR_NEGATIVE = '#D55E00'
        COLOR_GT_WINDOW = '#CC79A7'
        COLOR_GT_LINE = '#882255'

        # Load ground truth
        try:
            gt_model = DodgeWindowModel.load(gt_model_path)
        except FileNotFoundError:
            print(f"Ground truth model not found: {gt_model_path}")
            return None

        data_path = Path(data_dir)
        episode_files = sorted(data_path.glob("episode_*.npz"))[:max_episodes]

        if not episode_files:
            return None

        # Collect credits by (anim_idx, elapsed_frame)
        anim_credits = {}

        for ep_file in episode_files:
            data = np.load(ep_file)
            T = len(data['actions'])

            credit = self.compute_credit(
                boss_anim_ids=data['boss_anim_id'],
                hero_anim_ids=data['hero_anim_id'],
                dist_to_boss=data['dist_to_boss'],
                hero_hp=data['hero_hp'].astype(np.float32),
                actions=data['actions'].astype(np.float32),
                damage_taken=data['damage_taken'],
            )

            for t in range(T):
                anim_idx = anim_id_to_index(int(data['boss_anim_id'][t]))
                elapsed = int(data['elapsed_frames'][t])

                if anim_idx not in anim_credits:
                    anim_credits[anim_idx] = {}
                if elapsed not in anim_credits[anim_idx]:
                    anim_credits[anim_idx][elapsed] = []
                anim_credits[anim_idx][elapsed].append(credit[t])

        # Find animations with ground truth
        anims_with_windows = [idx for idx in gt_model.windows.keys() if idx in anim_credits]
        n_anims = min(len(anims_with_windows), 12)  # Limit for readability

        if n_anims == 0:
            return None

        cols = 4
        rows = (n_anims + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = np.array(axes).flatten()

        for i, anim_idx in enumerate(anims_with_windows[:n_anims]):
            ax = axes[i]
            elapsed_credits = anim_credits[anim_idx]
            frames = sorted(elapsed_credits.keys())
            max_frame = max(frames) if frames else 100

            # Compute mean credit at each frame
            mean_credits = np.array([
                np.mean(elapsed_credits.get(f, [0])) for f in range(max_frame + 1)
            ])

            # Normalize to [-1, 1]
            max_abs = np.abs(mean_credits).max()
            if max_abs > 0:
                mean_credits = mean_credits / max_abs

            colors = [COLOR_POSITIVE if c > 0 else COLOR_NEGATIVE for c in mean_credits]
            ax.bar(range(max_frame + 1), mean_credits, color=colors, alpha=0.6, width=1.0)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_ylim(-1.1, 1.1)

            # Ground truth windows
            windows = gt_model.windows.get(anim_idx, [])
            for w in windows:
                ax.axvspan(w.dodge_mean_frames - 2*w.dodge_std_frames,
                          w.dodge_mean_frames + 2*w.dodge_std_frames,
                          alpha=0.3, color=COLOR_GT_WINDOW)
                ax.axvline(w.dodge_mean_frames, color=COLOR_GT_LINE, linestyle='--', linewidth=2)

            anim_name = windows[0].anim_name if windows else f"idx_{anim_idx}"
            ax.set_title(f"{anim_name}", fontsize=10)
            ax.set_xlabel("Elapsed Frames")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle("RUDDER Credit vs Ground Truth\n(Pink=GT window, Orange=negative credit)", fontsize=12)
        plt.tight_layout()

        return fig

    def compute_gt_alignment(
        self,
        data_dir: str = "rudder_data",
        gt_model_path: str = "dodge_windows.json",
        max_episodes: int = 50,
    ) -> dict:
        """Compute alignment metrics between RUDDER credit and ground truth.

        Returns:
            dict with alignment metrics for wandb logging
        """
        from .dodge_window_model import DodgeWindowModel
        from .ppo import anim_id_to_index

        try:
            gt_model = DodgeWindowModel.load(gt_model_path)
        except FileNotFoundError:
            return {}

        data_path = Path(data_dir)
        episode_files = sorted(data_path.glob("episode_*.npz"))[:max_episodes]

        if not episode_files:
            return {}

        # Collect credits
        in_window_credits = []
        out_window_credits = []

        for ep_file in episode_files:
            data = np.load(ep_file)
            T = len(data['actions'])

            credit = self.compute_credit(
                boss_anim_ids=data['boss_anim_id'],
                hero_anim_ids=data['hero_anim_id'],
                dist_to_boss=data['dist_to_boss'],
                hero_hp=data['hero_hp'].astype(np.float32),
                actions=data['actions'].astype(np.float32),
                damage_taken=data['damage_taken'],
            )

            for t in range(T):
                anim_idx = anim_id_to_index(int(data['boss_anim_id'][t]))
                elapsed = int(data['elapsed_frames'][t])

                windows = gt_model.windows.get(anim_idx, [])
                in_any_window = any(
                    abs(elapsed - w.dodge_mean_frames) <= 2 * w.dodge_std_frames
                    for w in windows
                )

                if in_any_window:
                    in_window_credits.append(credit[t])
                elif windows:  # Only count out-window if animation has windows
                    out_window_credits.append(credit[t])

        if not in_window_credits or not out_window_credits:
            return {}

        avg_in = float(np.mean(in_window_credits))
        avg_out = float(np.mean(out_window_credits))
        correct = avg_in < avg_out  # Negative credit in window = correct

        return {
            'rudder/credit_in_window': avg_in,
            'rudder/credit_out_window': avg_out,
            'rudder/credit_diff': avg_in - avg_out,
            'rudder/gt_alignment': 1.0 if correct else 0.0,
        }


def create_rudder_shaper(
    model_path: str,
    credit_scale: float = 1.0,
    update_freq: int = 1,
) -> RudderRewardShaper:
    """Create a RUDDER reward shaper.

    Args:
        model_path: Path to pre-trained RUDDER model
        credit_scale: Scale factor for credit (default 1.0)
        update_freq: Update RUDDER every N rollouts

    Returns:
        RudderRewardShaper instance
    """
    config = RudderRewardConfig(
        model_path=model_path,
        credit_scale=credit_scale,
        update_freq=update_freq,
    )
    return RudderRewardShaper(config)
