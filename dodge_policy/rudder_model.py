"""RUDDER-style credit assignment model using LSTM in JAX/Flax.

Learns to predict episode return from (state, action) sequences.
Credit assignment comes from the difference in predictions:
  credit[t] = pred[t] - pred[t-1]
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from pathlib import Path
from typing import Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class RudderConfig:
    """Configuration for RUDDER model."""
    boss_anim_vocab: int = 50
    hero_anim_vocab: int = 200
    anim_embed_dim: int = 16
    continuous_dim: int = 3  # dist, hp, action (no elapsed_frames - that's cheating)
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100

    def to_dict(self) -> dict:
        return {
            'boss_anim_vocab': self.boss_anim_vocab,
            'hero_anim_vocab': self.hero_anim_vocab,
            'anim_embed_dim': self.anim_embed_dim,
            'continuous_dim': self.continuous_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RudderConfig':
        return cls(**d)


class RudderLSTM(nn.Module):
    """LSTM model for RUDDER credit assignment."""
    config: RudderConfig

    @nn.compact
    def __call__(
        self,
        boss_anim: jnp.ndarray,
        hero_anim: jnp.ndarray,
        continuous: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            boss_anim: (B,) or (B, T) boss animation IDs
                       If (B,), broadcast across time (per-attack mode)
            hero_anim: (B, T) hero animation IDs
            continuous: (B, T, 3) [dist, hp, action] - no elapsed_frames

        Returns:
            predictions: (B, T) predicted return at each timestep
        """
        cfg = self.config

        # Handle both (B,) and (B, T) boss_anim
        if boss_anim.ndim == 1:
            # Per-attack mode: boss_anim is constant, broadcast to (B, T)
            B = boss_anim.shape[0]
            T = hero_anim.shape[1]
            boss_anim = jnp.broadcast_to(boss_anim[:, None], (B, T))
        else:
            B, T = boss_anim.shape

        # Clamp to vocab size
        boss_anim = jnp.clip(boss_anim, 0, cfg.boss_anim_vocab - 1)
        hero_anim = jnp.clip(hero_anim, 0, cfg.hero_anim_vocab - 1)

        # Embeddings
        boss_emb = nn.Embed(num_embeddings=cfg.boss_anim_vocab,
                            features=cfg.anim_embed_dim)(boss_anim)
        hero_emb = nn.Embed(num_embeddings=cfg.hero_anim_vocab,
                            features=cfg.anim_embed_dim)(hero_anim)

        # Concatenate features: (B, T, input_dim)
        x = jnp.concatenate([boss_emb, hero_emb, continuous], axis=-1)

        # Use nn.RNN with LSTMCell for proper Flax handling
        lstm = nn.RNN(nn.LSTMCell(features=cfg.hidden_dim))

        # Run LSTM: (B, T, hidden_dim)
        hidden_states = lstm(x)

        # Output head
        x = nn.Dense(cfg.hidden_dim // 2)(hidden_states)
        x = nn.relu(x)
        predictions = nn.Dense(1)(x).squeeze(-1)  # (B, T)

        return predictions


def segment_into_attacks(data_dir: str, max_len: int = 64) -> Tuple[list, dict]:
    """Segment episodes into per-attack trajectories.

    Each attack = contiguous frames with same boss_anim_id.
    Target = 1 if damage taken during attack, 0 otherwise.

    Returns:
        trajectories: List of dicts with keys:
            - boss_anim_id: int (constant for trajectory)
            - elapsed_frames: array (0 to N)
            - hero_anim_id: array
            - actions: array
            - dist_to_boss: array
            - hero_hp: array
            - target: 0 or 1 (got hit?)
        stats: Normalization statistics
    """
    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("episode_*.npz"))

    if len(episode_files) == 0:
        raise ValueError(f"No episodes in {data_dir}")

    trajectories = []

    # First pass: collect all data for stats (no elapsed_frames - that's cheating)
    all_dist = []
    all_hp = []

    for f in episode_files:
        data = np.load(f)
        all_dist.extend(data['dist_to_boss'])
        all_hp.extend(data['hero_hp'].astype(np.float32))

    stats = {
        'dist_mean': float(np.mean(all_dist)),
        'dist_std': float(np.std(all_dist) + 1e-8),
        'hp_mean': float(np.mean(all_hp)),
        'hp_std': float(np.std(all_hp) + 1e-8),
        'return_mean': -0.5,  # Binary target: -1 (hit) or 0 (no hit), center at -0.5
        'return_std': 0.5,    # Scale to [-1, 1]
    }

    # Second pass: segment into attacks
    for f in episode_files:
        data = np.load(f)
        T = len(data['actions'])

        # Find animation boundaries
        boss_anims = data['boss_anim_id']

        start_idx = 0
        for t in range(1, T + 1):
            # Boundary: animation changed or end of episode
            is_boundary = (t == T) or (boss_anims[t] != boss_anims[t-1])

            if is_boundary:
                end_idx = t
                length = end_idx - start_idx

                # Skip very short segments
                if length >= 5:
                    # Extract segment
                    seg_boss = int(boss_anims[start_idx])
                    seg_hero = data['hero_anim_id'][start_idx:end_idx]
                    seg_actions = data['actions'][start_idx:end_idx].astype(np.float32)
                    seg_dist = data['dist_to_boss'][start_idx:end_idx]
                    seg_hp = data['hero_hp'][start_idx:end_idx].astype(np.float32)

                    # Target: -1 if got hit (bad), 0 if survived (neutral)
                    seg_damage = data['damage_taken'][start_idx:end_idx]
                    got_hit = seg_damage.sum() > 0
                    has_dodge = seg_actions.sum() > 0

                    # Filter: skip no-hit + no-dodge trajectories (no learning signal)
                    if not got_hit and not has_dodge:
                        start_idx = t
                        continue

                    target = -1.0 if got_hit else 0.0

                    # Pad or truncate to max_len
                    if length > max_len:
                        # Truncate
                        seg_hero = seg_hero[:max_len]
                        seg_actions = seg_actions[:max_len]
                        seg_dist = seg_dist[:max_len]
                        seg_hp = seg_hp[:max_len]
                        length = max_len
                    elif length < max_len:
                        # Pad with last value
                        pad_len = max_len - length
                        seg_hero = np.pad(seg_hero, (0, pad_len), mode='edge')
                        seg_actions = np.pad(seg_actions, (0, pad_len), mode='constant', constant_values=0)
                        seg_dist = np.pad(seg_dist, (0, pad_len), mode='edge')
                        seg_hp = np.pad(seg_hp, (0, pad_len), mode='edge')

                    trajectories.append({
                        'boss_anim_id': seg_boss,
                        'hero_anim_id': seg_hero,
                        'actions': seg_actions,
                        'dist_to_boss': seg_dist,
                        'hero_hp': seg_hp,
                        'target': target,
                        'original_length': length,
                    })

                start_idx = t

    # Count hits vs no hits (target=-1 means hit)
    n_hits = sum(1 for t in trajectories if t['target'] == -1.0)
    print(f"Segmented into {len(trajectories)} attack trajectories")
    print(f"  Hits: {n_hits} ({100*n_hits/len(trajectories):.1f}%)")
    print(f"  No hits: {len(trajectories) - n_hits} ({100*(len(trajectories)-n_hits)/len(trajectories):.1f}%)")

    return trajectories, stats


def trajectories_to_arrays(trajectories: list, stats: dict, max_len: int = 64):
    """Convert trajectory list to JAX arrays for training."""
    n = len(trajectories)

    all_boss = np.zeros((n,), dtype=np.int32)
    all_hero = np.zeros((n, max_len), dtype=np.int32)
    all_cont = np.zeros((n, max_len, 3), dtype=np.float32)  # dist, hp, action (no elapsed)
    all_targets = np.zeros((n,), dtype=np.float32)

    for i, traj in enumerate(trajectories):
        all_boss[i] = traj['boss_anim_id']
        all_hero[i] = traj['hero_anim_id']

        # Normalize continuous features (no elapsed_frames - that's cheating)
        dist = (traj['dist_to_boss'] - stats['dist_mean']) / stats['dist_std']
        hp = (traj['hero_hp'] - stats['hp_mean']) / stats['hp_std']
        actions = traj['actions']

        all_cont[i] = np.stack([dist, hp, actions], axis=-1)

        # Normalize target to [-1, 1]
        all_targets[i] = (traj['target'] - stats['return_mean']) / stats['return_std']

    return (
        jnp.array(all_boss),
        jnp.array(all_hero),
        jnp.array(all_cont),
        jnp.array(all_targets),
    )


def load_and_preprocess(data_dir: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Load episodes, normalize, and convert to JAX arrays upfront."""
    data_path = Path(data_dir)
    episode_files = sorted(data_path.glob("episode_*.npz"))

    if len(episode_files) == 0:
        raise ValueError(f"No episodes in {data_dir}")

    # Load raw data
    all_boss = []
    all_hero = []
    all_dist = []
    all_hp = []
    all_actions = []
    all_returns = []

    for f in episode_files:
        data = np.load(f)
        all_boss.append(data['boss_anim_id'])
        all_hero.append(data['hero_anim_id'])
        all_dist.append(data['dist_to_boss'])
        all_hp.append(data['hero_hp'])
        all_actions.append(data['actions'])
        all_returns.append(float(data['total_reward']))

    # Stack into arrays: (N, T)
    all_boss = np.stack(all_boss)
    all_hero = np.stack(all_hero)
    all_dist = np.stack(all_dist)
    all_hp = np.stack(all_hp).astype(np.float32)
    all_actions = np.stack(all_actions).astype(np.float32)
    all_returns = np.array(all_returns, dtype=np.float32)

    # Compute stats (no elapsed_frames - that's cheating)
    stats = {
        'dist_mean': float(all_dist.mean()),
        'dist_std': float(all_dist.std() + 1e-8),
        'hp_mean': float(all_hp.mean()),
        'hp_std': float(all_hp.std() + 1e-8),
        'return_mean': float(all_returns.mean()),
        'return_std': float(all_returns.std() + 1e-8),
    }

    # Normalize
    all_dist = (all_dist - stats['dist_mean']) / stats['dist_std']
    all_hp = (all_hp - stats['hp_mean']) / stats['hp_std']
    all_returns = (all_returns - stats['return_mean']) / stats['return_std']

    # Stack continuous: (N, T, 3) - no elapsed_frames
    all_cont = np.stack([all_dist, all_hp, all_actions], axis=-1)

    print(f"Loaded {len(episode_files)} episodes")
    print(f"  Sequence length: {all_boss.shape[1]}")
    print(f"  Avg return: {stats['return_mean']:.2f}")

    # Convert to JAX arrays once
    return (
        jnp.array(all_boss, dtype=jnp.int32),
        jnp.array(all_hero, dtype=jnp.int32),
        jnp.array(all_cont, dtype=jnp.float32),
        jnp.array(all_returns, dtype=jnp.float32),
        stats,
    )


@jax.jit
def train_step(state: TrainState, boss_anim, hero_anim, continuous, targets):
    """Single training step."""
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, boss_anim, hero_anim, continuous)
        # Loss on final prediction
        final_pred = predictions[:, -1]
        loss = jnp.mean((final_pred - targets) ** 2)
        return loss, predictions

    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def train_step_per_attack(state: TrainState, boss_anim, hero_anim, continuous, targets):
    """Single training step for per-attack mode.

    Args:
        boss_anim: (B,) constant boss animation for each trajectory
        hero_anim: (B, T) hero animations
        continuous: (B, T, 4) continuous features
        targets: (B,) binary target (got hit?)
    """
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, boss_anim, hero_anim, continuous)
        # Loss on final prediction
        final_pred = predictions[:, -1]
        loss = jnp.mean((final_pred - targets) ** 2)
        return loss, predictions

    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_rudder(
    data_dir: str,
    config: RudderConfig,
    save_path: str = "rudder_model",
    per_attack: bool = False,
    max_len: int = 64,
):
    """Train RUDDER model.

    Args:
        data_dir: Directory with episode data
        config: Model configuration
        save_path: Where to save model
        per_attack: If True, train on per-attack trajectories (binary hit/no-hit)
                   If False, train on full episodes (total return)
        max_len: Max sequence length for per-attack mode
    """
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.live import Live
    from rich.console import Console

    console = Console()

    if per_attack:
        # Per-attack trajectory training
        console.print("[bold]Loading per-attack trajectories...[/bold]")
        trajectories, stats = segment_into_attacks(data_dir, max_len=max_len)
        all_boss, all_hero, all_cont, all_targets = trajectories_to_arrays(trajectories, stats, max_len)
        n_samples = all_boss.shape[0]
        T = max_len
        stats['mode'] = 'per_attack'
        stats['max_len'] = max_len
    else:
        # Full episode training
        console.print("[bold]Loading full episodes...[/bold]")
        all_boss, all_hero, all_cont, all_targets, stats = load_and_preprocess(data_dir)
        n_samples = all_boss.shape[0]
        T = all_boss.shape[1]
        stats['mode'] = 'full_episode'

    # Create model
    console.print("[bold]Creating model...[/bold]")
    key = jax.random.PRNGKey(42)
    model = RudderLSTM(config=config)

    # Dummy input for init - use (B, T) for boss_anim in full mode, (B,) in per-attack
    if per_attack:
        dummy_boss = jnp.zeros((1,), dtype=jnp.int32)
    else:
        dummy_boss = jnp.zeros((1, T), dtype=jnp.int32)
    dummy_hero = jnp.zeros((1, T), dtype=jnp.int32)
    dummy_cont = jnp.zeros((1, T, 3), dtype=jnp.float32)  # dist, hp, action (no elapsed)

    params = model.init(key, dummy_boss, dummy_hero, dummy_cont)['params']

    # Optimizer
    tx = optax.adam(config.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    console.print(f"  Parameters: {num_params:,}")
    console.print(f"  Mode: {'per-attack' if per_attack else 'full-episode'}")
    console.print(f"  Samples: {n_samples}")
    console.print(f"  Sequence length: {T}")

    # Compile train_step
    console.print("  Compiling...")
    if per_attack:
        _ = train_step_per_attack(state, all_boss[:1], all_hero[:1], all_cont[:1], all_targets[:1])
        step_fn = train_step_per_attack
    else:
        _ = train_step(state, all_boss[:1], all_hero[:1], all_cont[:1], all_targets[:1])
        step_fn = train_step
    console.print("  Ready!")

    # Training
    best_loss = float('inf')

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss: {task.fields[loss]:.4f}"),
    )
    task = progress.add_task("[cyan]Training", total=config.epochs, loss=0.0)

    key = jax.random.PRNGKey(42)

    with Live(progress, refresh_per_second=4):
        for epoch in range(config.epochs):
            # Shuffle
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n_samples)

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, config.batch_size):
                batch_idx = perm[i:i + config.batch_size]
                if batch_idx.shape[0] < config.batch_size:
                    continue

                # Slice from pre-loaded JAX arrays
                boss = all_boss[batch_idx]
                hero = all_hero[batch_idx]
                cont = all_cont[batch_idx]
                targets = all_targets[batch_idx]

                state, loss = step_fn(state, boss, hero, cont, targets)

                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            progress.update(task, advance=1, loss=avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

    # Save
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save params as numpy
    params_dict = {}
    for path, value in jax.tree_util.tree_leaves_with_path(state.params):
        key = '/'.join(str(p.key) if hasattr(p, 'key') else str(p.idx) for p in path)
        params_dict[key] = np.array(value)

    np.savez(save_dir / "params.npz", **params_dict)

    # Save config and stats as JSON
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    with open(save_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Save tree structure for reconstruction
    tree_struct = jax.tree_util.tree_structure(state.params)
    with open(save_dir / "tree_structure.txt", 'w') as f:
        f.write(str(tree_struct))

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Best loss: {best_loss:.4f}")
    console.print(f"  Saved to: {save_dir}")

    return model, state, stats


def load_model(save_path: str) -> Tuple[RudderLSTM, Any, dict, dict]:
    """Load trained model."""
    save_dir = Path(save_path)

    # Load config
    with open(save_dir / "config.json") as f:
        config_dict = json.load(f)

    # Load stats
    with open(save_dir / "stats.json") as f:
        stats = json.load(f)

    # Detect if old model (with elapsed_frames) or new model (without)
    # Old models have elapsed_mean in stats and continuous_dim=4
    use_elapsed = 'elapsed_mean' in stats
    if use_elapsed:
        # Override config to match old model
        config_dict['continuous_dim'] = 4
    else:
        config_dict['continuous_dim'] = 3

    config = RudderConfig.from_dict(config_dict)

    # Create model
    model = RudderLSTM(config=config)

    # Load params
    params_data = np.load(save_dir / "params.npz")

    # Reconstruct params dict
    # We need to rebuild the nested structure
    # For simplicity, reinit and copy values
    key = jax.random.PRNGKey(0)

    # Determine sequence length based on mode
    mode = stats.get('mode', 'full_episode')
    if mode == 'per_attack':
        T = stats.get('max_len', 64)
        dummy_boss = jnp.zeros((1,), dtype=jnp.int32)  # Per-attack: constant boss anim
    else:
        T = 1024  # Full episode default
        dummy_boss = jnp.zeros((1, T), dtype=jnp.int32)

    dummy_hero = jnp.zeros((1, T), dtype=jnp.int32)
    cont_dim = 4 if use_elapsed else 3
    dummy_cont = jnp.zeros((1, T, cont_dim), dtype=jnp.float32)

    init_params = model.init(key, dummy_boss, dummy_hero, dummy_cont)['params']

    # Map loaded values back
    def load_leaf(path, _):
        key = '/'.join(str(p.key) if hasattr(p, 'key') else str(p.idx) for p in path)
        return jnp.array(params_data[key])

    params = jax.tree_util.tree_map_with_path(load_leaf, init_params)

    return model, params, config, stats


def get_credit(
    model: RudderLSTM,
    params: Any,
    boss_anim: jnp.ndarray,
    hero_anim: jnp.ndarray,
    continuous: jnp.ndarray,
) -> jnp.ndarray:
    """Compute credit assignment.

    Args:
        model: RUDDER model
        params: Model parameters
        boss_anim: (T,) or (B, T) boss animation IDs
        hero_anim: (T,) or (B, T) hero animation IDs
        continuous: (T, C) or (B, T, C) continuous features

    Returns:
        credit: (T,) credit at each timestep
    """
    # Add batch dim if needed
    if boss_anim.ndim == 1:
        boss_anim = boss_anim[None, :]
        hero_anim = hero_anim[None, :]
        continuous = continuous[None, :, :]

    predictions = model.apply({'params': params}, boss_anim, hero_anim, continuous)
    predictions = predictions[0]  # Remove batch dim

    # Credit = difference in predictions
    credit = jnp.zeros_like(predictions)
    credit = credit.at[0].set(predictions[0])
    credit = credit.at[1:].set(predictions[1:] - predictions[:-1])

    return credit


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train RUDDER model (JAX)")
    parser.add_argument("--data-dir", type=str, default="rudder_data")
    parser.add_argument("--save-path", type=str, default="rudder_model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--per-attack", action="store_true",
                        help="Train on per-attack trajectories (binary hit/no-hit target)")
    parser.add_argument("--max-len", type=int, default=64,
                        help="Max sequence length for per-attack mode")

    args = parser.parse_args()

    config = RudderConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    train_rudder(
        args.data_dir,
        config,
        args.save_path,
        per_attack=args.per_attack,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
