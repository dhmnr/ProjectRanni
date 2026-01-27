"""Simplified Dreamer-style World Model for Elden Ring combat.

Uses structured observations (not pixels), so no heavy encoder needed.
Learns:
  - Boss animation transitions
  - Hit detection (when player gets hit)
  - Damage prediction (when player attack lands)
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from pathlib import Path
from typing import Dict, Tuple, Any
from functools import partial


class WorldModel(nn.Module):
    """Predicts next state and outcomes given current state and action.

    State: (boss_anim_idx, elapsed_norm, dist_norm, player_hp_norm)
    Action: discrete (0=nothing, 1=dodge, 2=light_attack, 3=heavy_attack)

    Predicts:
    - next_boss_anim (categorical)
    - next_elapsed (continuous, or reset if anim changed)
    - next_dist (continuous)
    - hit_taken (binary)
    - damage_dealt (continuous)
    """
    n_boss_anims: int = 64
    n_actions: int = 4
    hidden_dim: int = 128
    embed_dim: int = 32

    @nn.compact
    def __call__(self, boss_anim_idx, elapsed_norm, dist_norm, action):
        # Embeddings
        boss_emb = nn.Embed(self.n_boss_anims, self.embed_dim)(boss_anim_idx)
        action_emb = nn.Embed(self.n_actions, 8)(action)

        # Continuous features
        if elapsed_norm.ndim == 1:
            elapsed_norm = elapsed_norm[:, None]
        if dist_norm.ndim == 1:
            dist_norm = dist_norm[:, None]

        # Encode state
        x = jnp.concatenate([boss_emb, elapsed_norm, dist_norm, action_emb], axis=-1)

        # Shared trunk
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Prediction heads
        # 1. Next boss animation (categorical)
        next_anim_logits = nn.Dense(self.n_boss_anims)(x)

        # 2. Next elapsed (continuous, normalized)
        next_elapsed = nn.Dense(1)(x).squeeze(-1)
        next_elapsed = nn.sigmoid(next_elapsed)  # 0-1

        # 3. Next distance (continuous, normalized)
        next_dist = nn.Dense(1)(x).squeeze(-1)
        next_dist = nn.sigmoid(next_dist)

        # 4. Hit taken (binary)
        hit_logit = nn.Dense(1)(x).squeeze(-1)

        # 5. Damage dealt (continuous, for attacks)
        damage_dealt = nn.Dense(1)(x).squeeze(-1)
        damage_dealt = nn.relu(damage_dealt)  # Non-negative

        return {
            'next_anim_logits': next_anim_logits,
            'next_elapsed': next_elapsed,
            'next_dist': next_dist,
            'hit_logit': hit_logit,
            'damage_dealt': damage_dealt,
        }


def load_transitions(data_dir: str = "rudder_data") -> Dict[str, np.ndarray]:
    """Load transition data from episodes.

    Returns dict with arrays for:
    - boss_anim_idx, elapsed_norm, dist_norm, action
    - next_boss_anim_idx, next_elapsed_norm, next_dist_norm
    - hit_taken, damage_dealt
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("episode_*.npz"))

    # Load vocab
    anim_vocab_path = Path("dodge_policy/anim_vocab.json")
    if anim_vocab_path.exists():
        with open(anim_vocab_path) as f:
            boss_vocab = json.load(f).get('vocab', {})
    else:
        boss_vocab = {}
        all_anims = set()
        for f in files:
            d = np.load(f)
            all_anims.update(d['boss_anim_id'].astype(int).tolist())
        boss_vocab = {str(a): i for i, a in enumerate(sorted(all_anims))}

    transitions = {
        'boss_anim_idx': [],
        'elapsed_norm': [],
        'dist_norm': [],
        'action': [],
        'next_boss_anim_idx': [],
        'next_elapsed_norm': [],
        'next_dist_norm': [],
        'hit_taken': [],
        'damage_dealt': [],
    }

    for f in files:
        d = np.load(f)
        T = len(d['boss_anim_id'])

        boss_anim = d['boss_anim_id'].astype(int)
        elapsed = d['elapsed_frames'].astype(np.float32) / 120.0
        dist = np.clip(d['dist_to_boss'].astype(np.float32) / 10.0, 0, 1)
        actions = d['actions'].astype(int)
        damage_taken = d['damage_taken'].astype(np.float32)

        # Damage dealt (if available, otherwise zeros)
        if 'damage_dealt' in d:
            damage_dealt = d['damage_dealt'].astype(np.float32)
        else:
            damage_dealt = np.zeros(T, dtype=np.float32)

        boss_idx = np.array([boss_vocab.get(str(a), 0) for a in boss_anim])

        # Create transitions (t → t+1)
        for t in range(T - 1):
            transitions['boss_anim_idx'].append(boss_idx[t])
            transitions['elapsed_norm'].append(elapsed[t])
            transitions['dist_norm'].append(dist[t])
            transitions['action'].append(actions[t])

            transitions['next_boss_anim_idx'].append(boss_idx[t + 1])
            transitions['next_elapsed_norm'].append(elapsed[t + 1])
            transitions['next_dist_norm'].append(dist[t + 1])

            transitions['hit_taken'].append(1.0 if damage_taken[t + 1] > 0 else 0.0)
            transitions['damage_dealt'].append(damage_dealt[t])

    for k in transitions:
        transitions[k] = np.array(transitions[k], dtype=np.float32 if 'norm' in k or k in ['hit_taken', 'damage_dealt'] else np.int32)

    return transitions, len(boss_vocab) + 1


@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, weights, n_anims):
    """Single training step."""

    def loss_fn(params):
        preds = state.apply_fn(
            params,
            batch['boss_anim_idx'],
            batch['elapsed_norm'],
            batch['dist_norm'],
            batch['action'],
        )

        # Animation prediction loss (cross-entropy)
        anim_loss = optax.softmax_cross_entropy_with_integer_labels(
            preds['next_anim_logits'],
            batch['next_boss_anim_idx']
        ).mean()

        # Elapsed prediction loss (MSE)
        elapsed_loss = ((preds['next_elapsed'] - batch['next_elapsed_norm']) ** 2).mean()

        # Distance prediction loss (MSE)
        dist_loss = ((preds['next_dist'] - batch['next_dist_norm']) ** 2).mean()

        # Hit prediction loss (weighted BCE)
        hit_bce = optax.sigmoid_binary_cross_entropy(
            preds['hit_logit'],
            batch['hit_taken']
        )
        hit_weights = jnp.where(batch['hit_taken'] == 1, weights['hit_pos'], 1.0)
        hit_loss = (hit_bce * hit_weights).mean()

        # Total loss
        total_loss = anim_loss + elapsed_loss + dist_loss + hit_loss * 2.0

        return total_loss, {
            'anim_loss': anim_loss,
            'elapsed_loss': elapsed_loss,
            'dist_loss': dist_loss,
            'hit_loss': hit_loss,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


def train_world_model(
    data_dir: str = "rudder_data",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    save_path: str = "world_model.npz",
):
    """Train the world model."""

    print(f"Loading transitions from {data_dir}...")
    transitions, n_anims = load_transitions(data_dir)

    n_transitions = len(transitions['boss_anim_idx'])
    n_hits = transitions['hit_taken'].sum()

    print(f"Transitions: {n_transitions:,}")
    print(f"Unique boss anims: {n_anims}")
    print(f"Hit rate: {n_hits}/{n_transitions} = {n_hits/n_transitions:.4f}")

    # Class weights for hit prediction
    hit_pos_weight = (n_transitions - n_hits) / max(n_hits, 1)
    weights = {'hit_pos': hit_pos_weight}
    print(f"Hit positive weight: {hit_pos_weight:.2f}")

    # Split
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_transitions)
    n_test = int(n_transitions * 0.1)
    train_idx, test_idx = idx[n_test:], idx[:n_test]

    train_data = {k: transitions[k][train_idx] for k in transitions}
    test_data = {k: transitions[k][test_idx] for k in transitions}

    print(f"Train: {len(train_idx):,}, Test: {len(test_idx):,}")

    # Model
    model = WorldModel(n_boss_anims=n_anims, n_actions=4)

    rng = jax.random.PRNGKey(42)
    dummy_i = jnp.zeros((1,), dtype=jnp.int32)
    dummy_f = jnp.zeros((1,), dtype=jnp.float32)
    params = model.init(rng, dummy_i, dummy_f, dummy_f, dummy_i)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params:,}")

    tx = optax.adam(lr)
    opt_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Training
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(train_idx))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_idx), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = {k: jnp.array(train_data[k][batch_idx]) for k in train_data}

            opt_state, loss, metrics = train_step(opt_state, batch, weights, n_anims)
            epoch_loss += float(loss)
            n_batches += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            # Eval
            test_batch = {k: jnp.array(test_data[k]) for k in test_data}
            preds = model.apply(opt_state.params,
                               test_batch['boss_anim_idx'],
                               test_batch['elapsed_norm'],
                               test_batch['dist_norm'],
                               test_batch['action'])

            # Anim accuracy
            pred_anim = preds['next_anim_logits'].argmax(-1)
            anim_acc = (pred_anim == test_batch['next_boss_anim_idx']).mean()

            # Hit accuracy
            pred_hit = (jax.nn.sigmoid(preds['hit_logit']) > 0.5).astype(jnp.float32)
            hit_acc = (pred_hit == test_batch['hit_taken']).mean()

            print(f"Epoch {epoch+1:3d}: Loss={epoch_loss/n_batches:.4f} | "
                  f"Anim Acc={anim_acc:.3f}, Hit Acc={hit_acc:.3f}")

    # Save
    flat_params, tree_def = jax.tree_util.tree_flatten(opt_state.params)
    save_dict = {f'param_{i}': np.array(p) for i, p in enumerate(flat_params)}
    save_dict['n_anims'] = np.array([n_anims])
    np.savez(save_path, **save_dict)
    print(f"\nSaved to {save_path}")

    return opt_state, model


def estimate_data_needs():
    """Estimate how much data is needed for the world model."""

    print("="*60)
    print("DATA REQUIREMENTS ESTIMATE")
    print("="*60)

    # Boss animation coverage
    n_attack_anims = 15  # Rough estimate of attack animations
    n_other_anims = 25   # Idle, recovery, movement, etc.
    frames_per_attack = 60  # Average attack duration

    print(f"\nBoss animations:")
    print(f"  Attack anims: ~{n_attack_anims}")
    print(f"  Other anims: ~{n_other_anims}")
    print(f"  Frames per attack: ~{frames_per_attack}")

    # What we need to learn for attacks
    print(f"\nPer attack animation, need to observe:")
    print(f"  - Hit timing: ~10 samples with varied dodge timing")
    print(f"  - No-hit cases: ~10 samples (dodge in window)")
    print(f"  - Attack during recovery: ~5 samples")

    samples_per_attack = 25
    total_attack_samples = n_attack_anims * samples_per_attack

    print(f"\nMinimum attack samples: {total_attack_samples}")

    # Episode math
    avg_episode_length = 1000  # frames
    attacks_per_episode = 15   # rough

    episodes_needed = total_attack_samples / attacks_per_episode

    print(f"\nWith ~{attacks_per_episode} attacks per episode:")
    print(f"  Minimum episodes: ~{int(episodes_needed)}")
    print(f"  Recommended (3x): ~{int(episodes_needed * 3)}")

    # Current data
    print(f"\n" + "-"*60)
    print("YOUR CURRENT DATA (rudder_data):")

    data_path = Path("rudder_data")
    files = list(data_path.glob("episode_*.npz"))
    total_frames = sum(len(np.load(f)['boss_anim_id']) for f in files)

    print(f"  Episodes: {len(files)}")
    print(f"  Total frames: {total_frames:,}")

    # Check action diversity
    all_actions = []
    for f in files:
        all_actions.extend(np.load(f)['actions'].tolist())

    from collections import Counter
    action_counts = Counter(all_actions)
    print(f"  Action distribution: {dict(action_counts)}")

    # Verdict
    print(f"\n" + "="*60)
    print("VERDICT:")
    if len(files) >= episodes_needed:
        print(f"  ✓ Enough episodes ({len(files)} >= {int(episodes_needed)})")
    else:
        print(f"  ✗ Need more episodes ({len(files)} < {int(episodes_needed)})")

    if 2 in action_counts or 3 in action_counts:
        print(f"  ✓ Has attack actions")
    else:
        print(f"  ✗ No attack actions (dodge-only data)")
        print(f"    → Need to collect with attack actions for full world model")

    print(f"\nFor DODGE-ONLY world model:")
    print(f"  Current data should be sufficient!")
    print(f"  Can learn: boss patterns, hit timing, dodge effectiveness")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true", help="Estimate data needs")
    parser.add_argument("--train", action="store_true", help="Train world model")
    parser.add_argument("--data-dir", default="rudder_data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="world_model.npz")

    args = parser.parse_args()

    if args.estimate:
        estimate_data_needs()
    elif args.train:
        train_world_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            save_path=args.output,
        )
    else:
        estimate_data_needs()
