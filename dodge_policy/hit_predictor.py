"""Hit Predictor Model - Partial World Model (JAX/Flax).

Predicts p(hit) within a horizon given current state and action.
Uses boss_anim_id, hero_anim_id, elapsed_frames, dist_to_boss, and action.
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from functools import partial


class HitPredictor(nn.Module):
    """MLP that predicts probability of getting hit."""

    n_boss_anims: int = 64
    n_hero_anims: int = 64
    embed_dim: int = 16
    hidden_dim: int = 64
    n_actions: int = 2

    @nn.compact
    def __call__(
        self,
        boss_anim_idx: jnp.ndarray,
        hero_anim_idx: jnp.ndarray,
        elapsed_norm: jnp.ndarray,
        dist_norm: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            boss_anim_idx: (B,) boss animation indices
            hero_anim_idx: (B,) hero animation indices
            elapsed_norm: (B,) normalized elapsed frames
            dist_norm: (B,) normalized distance to boss
            action: (B,) action indices (0=no dodge, 1=dodge)

        Returns:
            (B,) logits for p(hit)
        """
        boss_emb = nn.Embed(self.n_boss_anims, self.embed_dim)(boss_anim_idx)
        hero_emb = nn.Embed(self.n_hero_anims, self.embed_dim)(hero_anim_idx)
        action_emb = nn.Embed(self.n_actions, 4)(action)

        # Ensure shape (B, 1)
        if elapsed_norm.ndim == 1:
            elapsed_norm = elapsed_norm[:, None]
        if dist_norm.ndim == 1:
            dist_norm = dist_norm[:, None]

        x = jnp.concatenate([boss_emb, hero_emb, elapsed_norm, dist_norm, action_emb], axis=-1)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        return x.squeeze(-1)


def load_data(
    data_dir: str = "rudder_data",
    horizon: int = 30,
    anim_vocab_path: Optional[str] = None,
    within_anim: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load and process episode data.

    Args:
        data_dir: Directory with episode .npz files
        horizon: Number of frames to look ahead (if within_anim=False)
        anim_vocab_path: Path to animation vocabulary
        within_anim: If True, predict hit within current boss animation instead of fixed horizon

    Returns:
        samples: Dict with arrays for each feature
        metadata: Dict with vocab info
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("episode_*.npz"))

    # Load animation vocab if provided
    boss_vocab = {}
    if anim_vocab_path and Path(anim_vocab_path).exists():
        with open(anim_vocab_path) as f:
            vocab_data = json.load(f)
        boss_vocab = vocab_data.get('vocab', {})

    # First pass: collect all unique animation IDs
    all_boss_anims = set()
    all_hero_anims = set()

    for f in files:
        d = np.load(f)
        all_boss_anims.update(d['boss_anim_id'].astype(int).tolist())
        all_hero_anims.update(d['hero_anim_id'].astype(int).tolist())

    # Build vocab if not provided
    if not boss_vocab:
        boss_vocab = {str(a): i for i, a in enumerate(sorted(all_boss_anims))}

    hero_vocab = {str(a): i for i, a in enumerate(sorted(all_hero_anims))}

    # Second pass: create samples
    all_boss_idx = []
    all_hero_idx = []
    all_elapsed = []
    all_dist = []
    all_actions = []
    all_labels = []
    all_boss_anim_raw = []  # Keep raw anim IDs for analysis

    for f in files:
        d = np.load(f)
        T = len(d['boss_anim_id'])

        boss_anim = d['boss_anim_id'].astype(int)
        hero_anim = d['hero_anim_id'].astype(int)
        elapsed = d['elapsed_frames'].astype(np.float32)
        dist = d['dist_to_boss'].astype(np.float32)
        actions = d['actions'].astype(int)
        damage = d['damage_taken'].astype(np.float32)

        # Create hit labels
        hit_labels = np.zeros(T, dtype=np.float32)

        if within_anim:
            # Label: will there be a hit during the rest of this animation?
            # Find animation boundaries
            anim_starts = [0]
            for t in range(1, T):
                if boss_anim[t] != boss_anim[t-1]:
                    anim_starts.append(t)
            anim_starts.append(T)

            for i in range(len(anim_starts) - 1):
                start, end = anim_starts[i], anim_starts[i+1]
                anim_had_hit = damage[start:end].sum() > 0
                if anim_had_hit:
                    # Find when hit occurred
                    hit_times = np.where(damage[start:end] > 0)[0]
                    first_hit = start + hit_times[0]
                    # Label all frames before hit as positive
                    hit_labels[start:first_hit] = 1.0
        else:
            # Fixed horizon
            for t in range(T):
                end = min(t + horizon, T)
                hit_labels[t] = 1.0 if damage[t:end].sum() > 0 else 0.0

        # Convert anim IDs to vocab indices
        boss_idx = np.array([boss_vocab.get(str(a), 0) for a in boss_anim])
        hero_idx = np.array([hero_vocab.get(str(a), 0) for a in hero_anim])

        # Normalize continuous features
        elapsed_norm = elapsed / 120.0
        dist_norm = np.clip(dist / 10.0, 0, 1)

        # Use all samples for within_anim mode, otherwise exclude last horizon frames
        if within_anim:
            valid = T
        else:
            valid = T - horizon

        all_boss_idx.append(boss_idx[:valid])
        all_hero_idx.append(hero_idx[:valid])
        all_elapsed.append(elapsed_norm[:valid])
        all_dist.append(dist_norm[:valid])
        all_actions.append(actions[:valid])
        all_labels.append(hit_labels[:valid])
        all_boss_anim_raw.append(boss_anim[:valid])

    samples = {
        'boss_anim_idx': np.concatenate(all_boss_idx).astype(np.int32),
        'hero_anim_idx': np.concatenate(all_hero_idx).astype(np.int32),
        'elapsed_norm': np.concatenate(all_elapsed).astype(np.float32),
        'dist_norm': np.concatenate(all_dist).astype(np.float32),
        'action': np.concatenate(all_actions).astype(np.int32),
        'hit_label': np.concatenate(all_labels).astype(np.float32),
        'boss_anim_raw': np.concatenate(all_boss_anim_raw).astype(np.int32),
    }

    metadata = {
        'boss_vocab': boss_vocab,
        'hero_vocab': hero_vocab,
        'n_boss_anims': len(boss_vocab) + 1,
        'n_hero_anims': len(hero_vocab) + 1,
        'horizon': horizon,
        'within_anim': within_anim,
    }

    return samples, metadata


def create_train_state(
    rng: jax.random.PRNGKey,
    n_boss_anims: int,
    n_hero_anims: int,
    lr: float = 1e-3,
) -> train_state.TrainState:
    """Create initial training state."""
    model = HitPredictor(n_boss_anims=n_boss_anims, n_hero_anims=n_hero_anims)

    # Initialize with dummy input
    dummy_boss = jnp.zeros((1,), dtype=jnp.int32)
    dummy_hero = jnp.zeros((1,), dtype=jnp.int32)
    dummy_elapsed = jnp.zeros((1,), dtype=jnp.float32)
    dummy_dist = jnp.zeros((1,), dtype=jnp.float32)
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)

    params = model.init(rng, dummy_boss, dummy_hero, dummy_elapsed, dummy_dist, dummy_action)
    tx = optax.adam(lr)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    pos_weight: float,
    n_boss_anims: int,
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """Single training step."""

    def loss_fn(params):
        logits = state.apply_fn(
            params,
            batch['boss_anim_idx'],
            batch['hero_anim_idx'],
            batch['elapsed_norm'],
            batch['dist_norm'],
            batch['action'],
        )
        labels = batch['hit_label']

        # Weighted BCE loss
        bce = optax.sigmoid_binary_cross_entropy(logits, labels)
        weights = jnp.where(labels == 1, pos_weight, 1.0)
        loss = (bce * weights).mean()

        # Metrics
        preds = jax.nn.sigmoid(logits) > 0.5
        acc = (preds == labels).mean()

        return loss, {'loss': loss, 'acc': acc}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, metrics


@jax.jit
def eval_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    pos_weight: float,
) -> Dict[str, float]:
    """Evaluation step."""
    logits = state.apply_fn(
        state.params,
        batch['boss_anim_idx'],
        batch['hero_anim_idx'],
        batch['elapsed_norm'],
        batch['dist_norm'],
        batch['action'],
    )
    labels = batch['hit_label']

    # Loss
    bce = optax.sigmoid_binary_cross_entropy(logits, labels)
    weights = jnp.where(labels == 1, pos_weight, 1.0)
    loss = (bce * weights).mean()

    # Metrics
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    acc = (preds == labels).mean()

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    return {
        'loss': loss,
        'acc': acc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def train_hit_predictor(
    data_dir: str = "rudder_data",
    horizon: int = 30,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    test_split: float = 0.2,
    save_path: str = "hit_predictor.npz",
    seed: int = 42,
    within_anim: bool = False,
    oversample_safe: float = 0.0,
):
    """Train the hit predictor model.

    Args:
        oversample_safe: If > 0, oversample "safe" samples (no hit, action=1) by this factor.
                        E.g., 2.0 means duplicate safe dodge samples 2x.
    """

    mode = "within animation" if within_anim else f"horizon={horizon}"
    print(f"Loading data from {data_dir} (mode: {mode})...")

    anim_vocab_path = Path(__file__).parent / "anim_vocab.json"
    samples, metadata = load_data(
        data_dir=data_dir,
        horizon=horizon,
        anim_vocab_path=str(anim_vocab_path) if anim_vocab_path.exists() else None,
        within_anim=within_anim,
    )

    n_samples = len(samples['hit_label'])
    n_positive = (samples['hit_label'] > 0).sum()

    print(f"Total samples: {n_samples}")
    print(f"Boss anims: {metadata['n_boss_anims']}, Hero anims: {metadata['n_hero_anims']}")
    print(f"Positive rate: {n_positive}/{n_samples} = {n_positive/n_samples:.4f}")

    # Oversample safe dodges (action=1, no hit)
    if oversample_safe > 0:
        # Find indices where: action=1 (dodge) AND hit_label=0 (safe)
        safe_dodge_mask = (samples['action'] == 1) & (samples['hit_label'] == 0)
        safe_indices = np.where(safe_dodge_mask)[0]
        n_safe = len(safe_indices)

        if n_safe > 0:
            # How many copies to add
            n_copies = int(n_safe * oversample_safe)
            print(f"Oversampling {n_safe} safe dodges by {oversample_safe}x ({n_copies} copies)")

            # Sample with replacement
            rng_os = np.random.default_rng(seed)
            oversample_indices = rng_os.choice(safe_indices, size=n_copies, replace=True)

            # Append to samples
            for key in samples:
                samples[key] = np.concatenate([samples[key], samples[key][oversample_indices]])

            n_samples = len(samples['hit_label'])
            n_positive = (samples['hit_label'] > 0).sum()
            print(f"After oversampling: {n_samples} samples, positive rate: {n_positive/n_samples:.4f}")

    # Train/test split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = int(n_samples * test_split)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_data = {k: v[train_idx] for k, v in samples.items()}
    test_data = {k: v[test_idx] for k, v in samples.items()}

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Compute class weight
    pos_weight = float((n_samples - n_positive) / max(n_positive, 1))
    print(f"Positive class weight: {pos_weight:.2f}")

    # Initialize model
    rng = jax.random.PRNGKey(seed)
    state = create_train_state(
        rng,
        n_boss_anims=metadata['n_boss_anims'],
        n_hero_anims=metadata['n_hero_anims'],
        lr=lr,
    )

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"\nModel parameters: {n_params:,}")

    best_test_loss = float('inf')
    best_params = None

    for epoch in range(epochs):
        # Shuffle training data
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, len(train_idx))

        # Training
        train_metrics = {'loss': 0, 'acc': 0}
        n_batches = 0

        for i in range(0, len(train_idx), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = {k: jnp.array(v[batch_idx]) for k, v in train_data.items()}

            state, metrics = train_step(state, batch, pos_weight, metadata['n_boss_anims'])
            train_metrics['loss'] += float(metrics['loss'])
            train_metrics['acc'] += float(metrics['acc'])
            n_batches += 1

        train_metrics = {k: v / n_batches for k, v in train_metrics.items()}

        # Evaluation
        test_metrics = {'loss': 0, 'acc': 0, 'tp': 0, 'fp': 0, 'fn': 0}
        n_batches = 0

        for i in range(0, len(test_idx), batch_size):
            batch = {k: jnp.array(v[i:i+batch_size]) for k, v in test_data.items()}
            metrics = eval_step(state, batch, pos_weight)

            test_metrics['loss'] += float(metrics['loss'])
            test_metrics['acc'] += float(metrics['acc'])
            test_metrics['tp'] += float(metrics['tp'])
            test_metrics['fp'] += float(metrics['fp'])
            test_metrics['fn'] += float(metrics['fn'])
            n_batches += 1

        test_metrics['loss'] /= n_batches
        test_metrics['acc'] /= n_batches

        precision = test_metrics['tp'] / max(test_metrics['tp'] + test_metrics['fp'], 1)
        recall = test_metrics['tp'] / max(test_metrics['tp'] + test_metrics['fn'], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.4f} | "
                  f"Test Loss={test_metrics['loss']:.4f} Acc={test_metrics['acc']:.4f} "
                  f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        # Save best
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            best_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)
            best_metrics = {
                'test_loss': test_metrics['loss'],
                'test_acc': test_metrics['acc'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

    # Save model
    flat_params, tree_def = jax.tree_util.tree_flatten(best_params)

    # Save as npz with metadata
    save_dict = {
        f'param_{i}': p for i, p in enumerate(flat_params)
    }
    save_dict['tree_def'] = np.array([str(tree_def)], dtype=object)
    save_dict['n_boss_anims'] = np.array([metadata['n_boss_anims']])
    save_dict['n_hero_anims'] = np.array([metadata['n_hero_anims']])
    save_dict['horizon'] = np.array([horizon])
    save_dict['within_anim'] = np.array([within_anim])
    save_dict['boss_vocab'] = np.array([json.dumps(metadata['boss_vocab'])], dtype=object)
    save_dict['hero_vocab'] = np.array([json.dumps(metadata['hero_vocab'])], dtype=object)

    np.savez(save_path, **save_dict)

    print(f"\nBest model saved to {save_path}")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Metrics: P={best_metrics['precision']:.3f} R={best_metrics['recall']:.3f} F1={best_metrics['f1']:.3f}")

    return state


def load_hit_predictor(path: str) -> Tuple[HitPredictor, Dict[str, Any], Any]:
    """Load a trained hit predictor.

    Returns:
        model: HitPredictor module
        metadata: Dict with vocab and config
        params: Model parameters
    """
    data = np.load(path, allow_pickle=True)

    n_boss_anims = int(data['n_boss_anims'][0])
    n_hero_anims = int(data['n_hero_anims'][0])
    horizon = int(data['horizon'][0])
    boss_vocab = json.loads(str(data['boss_vocab'][0]))
    hero_vocab = json.loads(str(data['hero_vocab'][0]))

    model = HitPredictor(n_boss_anims=n_boss_anims, n_hero_anims=n_hero_anims)

    # Reconstruct params
    param_arrays = []
    i = 0
    while f'param_{i}' in data:
        param_arrays.append(data[f'param_{i}'])
        i += 1

    # Initialize to get tree structure
    rng = jax.random.PRNGKey(0)
    dummy_boss = jnp.zeros((1,), dtype=jnp.int32)
    dummy_hero = jnp.zeros((1,), dtype=jnp.int32)
    dummy_elapsed = jnp.zeros((1,), dtype=jnp.float32)
    dummy_dist = jnp.zeros((1,), dtype=jnp.float32)
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)
    init_params = model.init(rng, dummy_boss, dummy_hero, dummy_elapsed, dummy_dist, dummy_action)

    # Get tree structure and unflatten
    _, tree_def = jax.tree_util.tree_flatten(init_params)
    params = jax.tree_util.tree_unflatten(tree_def, param_arrays)

    metadata = {
        'n_boss_anims': n_boss_anims,
        'n_hero_anims': n_hero_anims,
        'horizon': horizon,
        'boss_vocab': boss_vocab,
        'hero_vocab': hero_vocab,
    }

    return model, metadata, params


def compare_with_dodge_windows(
    model_path: str = "hit_predictor.npz",
    windows_path: str = "dodge_windows_expert.json",
    data_dir: str = "rudder_data",
):
    """Compare hit predictor with manual dodge windows."""
    from collections import defaultdict

    # Load model
    model, metadata, params = load_hit_predictor(model_path)
    boss_vocab = metadata['boss_vocab']
    hero_vocab = metadata['hero_vocab']

    # Load dodge windows
    with open(windows_path) as f:
        windows_data = json.load(f)
    windows = windows_data.get('windows', {})
    anim_vocab = windows_data.get('anim_vocab', {})

    # Reverse vocab: idx -> anim_id
    idx_to_anim = {}
    for k, v in boss_vocab.items():
        try:
            idx_to_anim[v] = int(k)
        except ValueError:
            pass  # Skip non-numeric keys like <UNK>

    print("="*70)
    print("COMPARISON: Hit Predictor vs Dodge Windows")
    print("="*70)

    # Load test data
    anim_vocab_path = Path(__file__).parent / "anim_vocab.json"
    samples, _ = load_data(
        data_dir=data_dir,
        within_anim=True,
        anim_vocab_path=str(anim_vocab_path) if anim_vocab_path.exists() else None,
    )

    # Group by boss animation
    anim_stats = defaultdict(lambda: {
        'total': 0,
        'hits': 0,
        'model_prob_sum': 0,
        'elapsed_at_hit': [],
        'model_probs': [],
        'elapsed_frames': [],
    })

    # Run inference
    batch_size = 1024
    n_samples = len(samples['boss_anim_idx'])

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = {
            'boss_anim_idx': jnp.array(samples['boss_anim_idx'][i:end]),
            'hero_anim_idx': jnp.array(samples['hero_anim_idx'][i:end]),
            'elapsed_norm': jnp.array(samples['elapsed_norm'][i:end]),
            'dist_norm': jnp.array(samples['dist_norm'][i:end]),
            'action': jnp.array(samples['action'][i:end]),
        }

        logits = model.apply(
            params,
            batch['boss_anim_idx'],
            batch['hero_anim_idx'],
            batch['elapsed_norm'],
            batch['dist_norm'],
            batch['action'],
        )
        probs = jax.nn.sigmoid(logits)

        # Accumulate stats
        for j in range(end - i):
            idx = i + j
            anim_idx = int(samples['boss_anim_idx'][idx])
            elapsed = float(samples['elapsed_norm'][idx]) * 120  # denormalize
            hit = int(samples['hit_label'][idx])
            prob = float(probs[j])

            stats = anim_stats[anim_idx]
            stats['total'] += 1
            stats['hits'] += hit
            stats['model_prob_sum'] += prob
            stats['model_probs'].append(prob)
            stats['elapsed_frames'].append(elapsed)
            if hit:
                stats['elapsed_at_hit'].append(elapsed)

    # Compare with windows
    print(f"\n{'Anim ID':<12} {'Samples':>8} {'Hit%':>6} {'Model P':>8} | "
          f"{'Window':>12} {'Diff':>8}")
    print("-" * 70)

    for anim_idx in sorted(anim_stats.keys()):
        stats = anim_stats[anim_idx]
        if stats['total'] < 50:  # Skip rare animations
            continue

        anim_id = idx_to_anim.get(anim_idx, 0)
        hit_rate = stats['hits'] / stats['total']
        avg_prob = stats['model_prob_sum'] / stats['total']

        # Get window info if available
        window_info = "-"
        diff = "-"
        if str(anim_idx) in windows:
            w = windows[str(anim_idx)][0]
            window_mean = w['dodge_mean_frames']
            window_info = f"{window_mean:.1f}f"

            # Compare: when does model predict highest danger?
            probs = np.array(stats['model_probs'])
            elapsed = np.array(stats['elapsed_frames'])

            # Find peak danger zone from model
            if len(probs) > 10:
                # Bin by elapsed time and find peak
                bins = np.linspace(0, 120, 13)
                bin_probs = []
                for b in range(len(bins)-1):
                    mask = (elapsed >= bins[b]) & (elapsed < bins[b+1])
                    if mask.sum() > 0:
                        bin_probs.append(((bins[b] + bins[b+1])/2, probs[mask].mean()))

                if bin_probs:
                    peak_time = max(bin_probs, key=lambda x: x[1])[0]
                    diff = f"{peak_time - window_mean:+.1f}f"

        print(f"anim_{anim_id:<6} {stats['total']:>8} {hit_rate*100:>5.1f}% {avg_prob:>8.3f} | "
              f"{window_info:>12} {diff:>8}")

    # Summary stats
    print("\n" + "="*70)
    total_samples = sum(s['total'] for s in anim_stats.values())
    total_hits = sum(s['hits'] for s in anim_stats.values())
    print(f"Total: {total_samples} samples, {total_hits} hits ({total_hits/total_samples*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train hit predictor model")
    parser.add_argument("--data-dir", default="rudder_data")
    parser.add_argument("--horizon", type=int, default=30, help="Frames to look ahead")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="hit_predictor.npz")
    parser.add_argument("--within-anim", action="store_true", help="Predict hit within current animation")
    parser.add_argument("--oversample-safe", type=float, default=0.0, help="Oversample safe dodges by this factor")
    parser.add_argument("--compare", action="store_true", help="Compare with dodge windows instead of training")
    parser.add_argument("--windows", default="dodge_windows_expert.json", help="Dodge windows file for comparison")

    args = parser.parse_args()

    if args.compare:
        compare_with_dodge_windows(
            model_path=args.output,
            windows_path=args.windows,
            data_dir=args.data_dir,
        )
    else:
        train_hit_predictor(
            data_dir=args.data_dir,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.output,
            within_anim=args.within_anim,
            oversample_safe=args.oversample_safe,
        )
