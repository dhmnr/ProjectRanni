"""Training script for dodge-only policy variant.

The agent always walks forward. The only decision is when to dodge.
Input: boss_anim_id + elapsed_frames
Output: Discrete(2) - dodge or no dodge
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple, NamedTuple, Optional, Callable
from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from .dodge_only_agent import (
    DodgeOnlyAgent,
    create_dodge_only_agent,
    sinusoidal_encoding,
)
from .dodge_only_wrapper import DodgeOnlyWrapper
from .dodge_window_model import DodgeWindowModel
from .rudder_reward import RudderRewardShaper, RudderRewardConfig
from .world_model_reward import WorldModelRewardShaper, WorldModelRewardConfig
from .config import load_config
from .env_factory import make_env
from .ppo import ANIM_VOCAB, anim_id_to_index


@dataclass
class DodgeOnlyConfig:
    """Configuration for dodge-only training."""
    # Training
    num_steps: int = 2048
    total_timesteps: int = 1_000_000
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4

    # Hindsight reward labeling
    hindsight_hit_reward: float = -2.0    # Reward for dodges in segments where we got hit
    hindsight_no_hit_reward: float = 2.0  # Reward for dodges in segments where we didn't get hit

    # Network
    hidden_dim: int = 128
    anim_embed_dim: int = 32

    @property
    def batch_size(self) -> int:
        return self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.num_steps


def apply_hindsight_rewards(
    actions: np.ndarray,
    rewards: np.ndarray,
    boss_anim_ids: np.ndarray,
    damage_taken: np.ndarray,
    hit_reward: float,
    no_hit_reward: float,
) -> Tuple[np.ndarray, dict]:
    """Apply hindsight reward labeling to dodges based on segment outcomes.

    For each boss animation segment:
    - If player got hit: all dodges in segment get hit_reward (negative)
    - If player didn't get hit: all dodges get no_hit_reward (positive)

    Args:
        actions: Action array (1 = dodge)
        rewards: Reward array to modify in place
        boss_anim_ids: Boss animation ID at each step
        damage_taken: Damage taken at each step
        hit_reward: Reward for dodges when hit occurred
        no_hit_reward: Reward for dodges when no hit occurred

    Returns:
        Modified rewards and stats dict
    """
    T = len(actions)
    if T == 0:
        return rewards, {}

    # Find segment boundaries (where boss_anim_id changes)
    segments = []
    seg_start = 0
    current_anim = boss_anim_ids[0]

    for t in range(1, T):
        if boss_anim_ids[t] != current_anim:
            segments.append((seg_start, t, current_anim))
            seg_start = t
            current_anim = boss_anim_ids[t]
    # Add final segment
    segments.append((seg_start, T, current_anim))

    # Stats
    dodges_rewarded = 0
    dodges_punished = 0
    segments_hit = 0
    segments_no_hit = 0

    # Apply hindsight rewards to each segment
    for start, end, anim_id in segments:
        # Check if any damage in this segment
        segment_damage = damage_taken[start:end].sum()
        got_hit = segment_damage > 0

        # Find dodges in this segment
        dodge_mask = actions[start:end] == 1
        num_dodges = dodge_mask.sum()

        if num_dodges > 0:
            if got_hit:
                # Punish dodges - they didn't help
                rewards[start:end][dodge_mask] += hit_reward
                dodges_punished += num_dodges
                segments_hit += 1
            else:
                # Reward dodges - they may have helped avoid damage
                rewards[start:end][dodge_mask] += no_hit_reward
                dodges_rewarded += num_dodges
                segments_no_hit += 1

    stats = {
        'hindsight_dodges_rewarded': dodges_rewarded,
        'hindsight_dodges_punished': dodges_punished,
        'hindsight_segments_hit': segments_hit,
        'hindsight_segments_no_hit': segments_no_hit,
    }

    return rewards, stats


class RolloutBatch(NamedTuple):
    """Batch of rollout data."""
    anim_idx: jnp.ndarray       # [batch]
    elapsed_sin: jnp.ndarray    # [batch, 16]
    actions: jnp.ndarray        # [batch]
    log_probs: jnp.ndarray      # [batch]
    advantages: jnp.ndarray     # [batch]
    returns: jnp.ndarray        # [batch]
    values: jnp.ndarray         # [batch]


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self, num_steps: int, store_rudder_data: bool = False):
        self.num_steps = num_steps
        self.ptr = 0
        self.store_rudder_data = store_rudder_data

        self.anim_idx = np.zeros(num_steps, dtype=np.int32)
        self.elapsed_sin = np.zeros((num_steps, 16), dtype=np.float32)
        self.actions = np.zeros(num_steps, dtype=np.int32)
        self.log_probs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)

        # RUDDER data storage
        if store_rudder_data:
            self.boss_anim_ids = np.zeros(num_steps, dtype=np.int32)
            self.hero_anim_ids = np.zeros(num_steps, dtype=np.int32)
            self.dist_to_boss = np.zeros(num_steps, dtype=np.float32)
            self.hero_hp = np.zeros(num_steps, dtype=np.float32)
            self.damage_taken = np.zeros(num_steps, dtype=np.float32)

    def add(self, anim_idx, elapsed_sin, action, log_prob, reward, done, value,
            boss_anim_id=None, hero_anim_id=None, dist_to_boss=None, hero_hp=None, damage_taken=None):
        self.anim_idx[self.ptr] = anim_idx
        self.elapsed_sin[self.ptr] = elapsed_sin
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value

        if self.store_rudder_data:
            self.boss_anim_ids[self.ptr] = boss_anim_id if boss_anim_id is not None else 0
            self.hero_anim_ids[self.ptr] = hero_anim_id if hero_anim_id is not None else 0
            self.dist_to_boss[self.ptr] = dist_to_boss if dist_to_boss is not None else 0
            self.hero_hp[self.ptr] = hero_hp if hero_hp is not None else 0
            self.damage_taken[self.ptr] = damage_taken if damage_taken is not None else 0

        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_value: float, last_done: float, gamma: float, gae_lambda: float):
        advantages = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - last_done
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + self.values
        return returns, advantages

    def get_batch(self, returns, advantages) -> RolloutBatch:
        return RolloutBatch(
            anim_idx=jnp.array(self.anim_idx),
            elapsed_sin=jnp.array(self.elapsed_sin),
            actions=jnp.array(self.actions),
            log_probs=jnp.array(self.log_probs),
            advantages=jnp.array(advantages),
            returns=jnp.array(returns),
            values=jnp.array(self.values),
        )


def compute_loss(
    params,
    apply_fn,
    batch: RolloutBatch,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
):
    """Compute PPO loss for discrete action space."""
    logits, values = apply_fn(params, batch.anim_idx, batch.elapsed_sin)

    # Log probs for taken actions
    log_probs_all = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(
        log_probs_all, batch.actions[:, None], axis=-1
    ).squeeze(-1)

    # Entropy
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs_all, axis=-1).mean()

    # Policy loss
    ratio = jnp.exp(new_log_probs - batch.log_probs)
    pg_loss1 = -batch.advantages * ratio
    pg_loss2 = -batch.advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss = 0.5 * ((values - batch.returns) ** 2).mean()

    # Total loss
    loss = pg_loss - ent_coef * entropy + vf_coef * v_loss

    info = {
        'loss/total': loss,
        'loss/policy': pg_loss,
        'loss/value': v_loss,
        'loss/entropy': entropy,
        'loss/approx_kl': ((ratio - 1) - jnp.log(ratio)).mean(),
    }

    return loss, info


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def update_step(
    train_state: TrainState,
    batch: RolloutBatch,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
):
    def loss_fn(params):
        return compute_loss(params, train_state.apply_fn, batch, clip_eps, ent_coef, vf_coef)

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, info


def save_checkpoint(train_state, global_step, update, checkpoint_dir: Path, name: str = "checkpoint"):
    """Save checkpoint."""
    checkpoint_path = checkpoint_dir / f"{name}_{update}.npz"
    np.savez(
        checkpoint_path,
        params=jax.device_get(train_state.params),
        step=global_step,
        update=update,
    )
    print(f"\nSaved checkpoint: {checkpoint_path}")
    return checkpoint_path


def train_dodge_only(
    config,
    ppo_config: DodgeOnlyConfig,
    dodge_window_model: Optional[DodgeWindowModel] = None,
    dodge_window_reward: float = 2.0,
    rudder_shaper: Optional[RudderRewardShaper] = None,
    rudder_update_freq: int = 5,
    rudder_save_freq: int = 50,
    world_model_shaper: Optional[WorldModelRewardShaper] = None,
    wandb_run=None,
    checkpoint_dir: str = "checkpoints/dodge_only",
    save_freq: int = 10,
):
    """Train dodge-only policy."""
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {checkpoint_path}")

    # Create environment with dodge-only wrapper
    print("Creating environment...")
    base_env = make_env(config.env)
    env = DodgeOnlyWrapper(base_env)
    print(f"  Action space: {env.action_space}")

    # Create agent
    print("Creating agent...")
    key = jax.random.PRNGKey(config.seed)
    key, agent_key = jax.random.split(key)

    agent, train_state = create_dodge_only_agent(
        key=agent_key,
        hidden_dim=ppo_config.hidden_dim,
        anim_embed_dim=ppo_config.anim_embed_dim,
        anim_vocab_size=len(ANIM_VOCAB) + 1,  # +1 for UNK
        learning_rate=ppo_config.learning_rate,
        max_grad_norm=ppo_config.max_grad_norm,
    )

    # Create buffer
    # Always store extra data for hindsight labeling (boss_anim_ids, damage_taken)
    buffer = RolloutBuffer(ppo_config.num_steps, store_rudder_data=True)

    # Training loop
    global_step = 0
    num_updates = ppo_config.num_updates

    # Stats
    ep_rewards = []
    ep_dodges = 0
    ep_good_dodges = 0
    ep_hits = 0

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    train_task = progress.add_task("[cyan]Training", total=num_updates)
    rollout_task = progress.add_task("[green]Rollout", total=ppo_config.num_steps)

    stats = {"reward": 0, "dodges": 0, "good_dodges": 0, "hits": 0, "loss": 0, "wm_reward": 0}

    def make_stats_table():
        table = Table(show_header=False, box=None)
        table.add_column(width=15)
        table.add_column(width=10)
        table.add_row("Reward:", f"{stats['reward']:.2f}")
        table.add_row("Dodges:", f"{stats['dodges']}")
        table.add_row("Good dodges:", f"[green]{stats['good_dodges']}[/green]")
        table.add_row("Hits:", f"[red]{stats['hits']}[/red]")
        table.add_row("WM Reward:", f"[cyan]{stats['wm_reward']:.2f}[/cyan]")
        table.add_row("Loss:", f"{stats['loss']:.4f}")
        return table

    def make_display():
        from rich.console import Group
        return Panel(
            Group(progress, make_stats_table()),
            title="[blue]Dodge-Only Training[/blue]"
        )

    print(f"\nStarting training ({num_updates} updates)...\n")
    print(f"Checkpoints saved every {save_freq} updates. Press Ctrl+C to save and exit.\n")

    update = 0  # Track current update for interrupt saving
    try:
        with Live(make_display(), refresh_per_second=4) as live:
            for update in range(1, num_updates + 1):
                buffer.reset()
                progress.reset(rollout_task)
                ep_reward = 0
                ep_dodges = 0
                ep_good_dodges = 0
                ep_hits = 0
                ep_wm_reward = 0

                # Reset env
                obs_dict, _ = env.reset()

                for step in range(ppo_config.num_steps):
                    progress.update(rollout_task, completed=step + 1)
                    if step % 50 == 0:
                        live.update(make_display())
                    global_step += 1

                    # Get anim_idx and elapsed_frames
                    anim_idx = anim_id_to_index(obs_dict['boss_anim_id'])
                    elapsed = float(obs_dict['elapsed_frames'])

                    # Sinusoidal encoding
                    elapsed_sin = np.array(sinusoidal_encoding(
                        jnp.array([elapsed]), num_scales=8
                    )[0])

                    # Get action
                    key, action_key = jax.random.split(key)
                    action, log_prob, _, value = agent.apply(
                        train_state.params,
                        jnp.array([anim_idx]),
                        jnp.array([elapsed_sin]),
                        action_key,
                        method=agent.get_action_and_value,
                    )
                    action = int(action[0])
                    log_prob = float(log_prob[0])
                    value = float(value[0])

                    # Step env
                    next_obs_dict, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Track hits
                    if info.get("player_damage_taken", 0) > 0:
                        ep_hits += 1

                    # Detect dodge from action (key press)
                    if action == 1:  # Dodge action
                        ep_dodges += 1
                        # Dodge window reward shaping
                        if dodge_window_model is not None:
                            in_window, window_reward = dodge_window_model.should_dodge(
                                anim_idx, elapsed, sigma=2.0
                            )
                            if in_window:
                                reward += dodge_window_reward * window_reward
                                ep_good_dodges += 1

                    # World model reward shaping
                    if world_model_shaper is not None:
                        dist = float(obs_dict.get('dist_to_boss', 5.0))
                        wm_reward, wm_info = world_model_shaper.compute_reward_shaping(
                            anim_idx=anim_idx,
                            elapsed_frames=elapsed,
                            dist_to_boss=dist,
                            action=action,
                        )
                        reward += wm_reward
                        ep_wm_reward += wm_reward

                    ep_reward += reward

                    # Get RUDDER data from obs
                    boss_anim_id = int(obs_dict.get('boss_anim_id', 0))
                    hero_anim_id = int(obs_dict.get('HeroAnimId', 0))
                    dist = float(obs_dict.get('dist_to_boss', 0))
                    hp = float(obs_dict.get('hero_hp', 0))
                    dmg = float(info.get('player_damage_taken', 0))

                    # Store
                    buffer.add(
                        anim_idx, elapsed_sin, action, log_prob, reward, float(done), value,
                        boss_anim_id=boss_anim_id, hero_anim_id=hero_anim_id,
                        dist_to_boss=dist, hero_hp=hp, damage_taken=dmg
                    )

                    obs_dict = next_obs_dict
                    if done:
                        obs_dict, _ = env.reset()

                # Apply RUDDER credit (if enabled)
                rudder_credit_mean = 0.0
                rudder_credit_std = 0.0
                if rudder_shaper is not None:
                    # Compute credit for the rollout
                    credit = rudder_shaper.compute_credit(
                        boss_anim_ids=buffer.boss_anim_ids[:buffer.ptr],
                        hero_anim_ids=buffer.hero_anim_ids[:buffer.ptr],
                        dist_to_boss=buffer.dist_to_boss[:buffer.ptr],
                        hero_hp=buffer.hero_hp[:buffer.ptr],
                        actions=buffer.actions[:buffer.ptr].astype(np.float32),
                        damage_taken=buffer.damage_taken[:buffer.ptr],
                    )
                    # Track credit stats
                    rudder_credit_mean = float(credit.mean())
                    rudder_credit_std = float(credit.std())

                    # Add credit to rewards
                    buffer.rewards[:buffer.ptr] += credit

                    # Add to RUDDER buffer for training
                    rudder_shaper.add_to_buffer(
                        boss_anim_ids=buffer.boss_anim_ids[:buffer.ptr],
                        hero_anim_ids=buffer.hero_anim_ids[:buffer.ptr],
                        dist_to_boss=buffer.dist_to_boss[:buffer.ptr],
                        hero_hp=buffer.hero_hp[:buffer.ptr],
                        actions=buffer.actions[:buffer.ptr].astype(np.float32),
                        damage_taken=buffer.damage_taken[:buffer.ptr],
                    )

                # Compute last value
                anim_idx = anim_id_to_index(obs_dict['boss_anim_id'])
                elapsed = float(obs_dict['elapsed_frames'])
                elapsed_sin = np.array(sinusoidal_encoding(jnp.array([elapsed]), num_scales=8)[0])
                last_value = float(agent.apply(
                    train_state.params,
                    jnp.array([anim_idx]),
                    jnp.array([elapsed_sin]),
                    method=agent.get_value,
                )[0])

                # GAE
                returns, advantages = buffer.compute_gae(
                    last_value, 0.0, ppo_config.gamma, ppo_config.gae_lambda
                )

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get batch
                batch = buffer.get_batch(returns, advantages)

                # Update
                all_losses = {'total': [], 'policy': [], 'value': [], 'entropy': [], 'kl': []}
                for _ in range(ppo_config.update_epochs):
                    key, perm_key = jax.random.split(key)
                    perm = jax.random.permutation(perm_key, ppo_config.batch_size)

                    for start in range(0, ppo_config.batch_size, ppo_config.minibatch_size):
                        end = start + ppo_config.minibatch_size
                        mb_idx = perm[start:end]

                        mb_batch = RolloutBatch(
                            anim_idx=batch.anim_idx[mb_idx],
                            elapsed_sin=batch.elapsed_sin[mb_idx],
                            actions=batch.actions[mb_idx],
                            log_probs=batch.log_probs[mb_idx],
                            advantages=batch.advantages[mb_idx],
                            returns=batch.returns[mb_idx],
                            values=batch.values[mb_idx],
                        )

                        train_state, loss_info = update_step(
                            train_state, mb_batch,
                            ppo_config.clip_eps, ppo_config.ent_coef, ppo_config.vf_coef
                        )
                        all_losses['total'].append(float(loss_info['loss/total']))
                        all_losses['policy'].append(float(loss_info['loss/policy']))
                        all_losses['value'].append(float(loss_info['loss/value']))
                        all_losses['entropy'].append(float(loss_info['loss/entropy']))
                        all_losses['kl'].append(float(loss_info['loss/approx_kl']))

                # Update stats
                ep_rewards.append(ep_reward)
                stats.update({
                    "reward": ep_reward,
                    "dodges": ep_dodges,
                    "good_dodges": ep_good_dodges,
                    "hits": ep_hits,
                    "wm_reward": ep_wm_reward,
                    "loss": np.mean(all_losses['total']),
                })

                progress.update(train_task, completed=update)
                live.update(make_display())

                # Update RUDDER model periodically
                rudder_info = {}
                if rudder_shaper is not None and update % rudder_update_freq == 0:
                    rudder_info = rudder_shaper.update_model()
                    buffer_size = rudder_info.get('rudder_buffer_size', 0)

                    if rudder_info.get('rudder_loss', 0) > 0:
                        print(f"  RUDDER update: loss={rudder_info['rudder_loss']:.4f}, "
                              f"trajectories={rudder_info['rudder_trajectories']}, "
                              f"buffer={buffer_size}")

                # Wandb logging
                if wandb_run is not None:
                    good_dodge_ratio = ep_good_dodges / ep_dodges if ep_dodges > 0 else 0.0
                    log_dict = {
                        # Episode metrics
                        "episode/reward": ep_reward,
                        "episode/dodges": ep_dodges,
                        "episode/good_dodges": ep_good_dodges,
                        "episode/good_dodge_ratio": good_dodge_ratio,
                        "episode/hits": ep_hits,
                        # Training diagnostics
                        "train/gae_return_mean": float(returns.mean()),
                        # Losses
                        "loss/total": np.mean(all_losses['total']),
                        "loss/policy": np.mean(all_losses['policy']),
                        "loss/value": np.mean(all_losses['value']),
                        "loss/entropy": np.mean(all_losses['entropy']),
                        "loss/kl": np.mean(all_losses['kl']),
                        # Value estimates
                        "value/mean": float(buffer.values.mean()),
                        "value/std": float(buffer.values.std()),
                    }

                    # RUDDER metrics
                    if rudder_shaper is not None:
                        log_dict["rudder/buffer_size"] = len(rudder_shaper.trajectory_buffer)
                        log_dict["rudder/credit_mean"] = rudder_credit_mean
                        log_dict["rudder/credit_std"] = rudder_credit_std
                        if rudder_info.get('rudder_loss', 0) > 0:
                            log_dict["rudder/loss"] = rudder_info['rudder_loss']
                            log_dict["rudder/trajectories"] = rudder_info['rudder_trajectories']

                    # World model metrics
                    if world_model_shaper is not None:
                        log_dict["world_model/reward"] = ep_wm_reward

                    wandb_run.log(log_dict, step=update)

                # Periodic checkpoint saving
                if update % save_freq == 0:
                    save_checkpoint(train_state, global_step, update, checkpoint_path, "checkpoint")

                # Save RUDDER model periodically (visualization is logged above on every update)
                if rudder_shaper is not None and update % rudder_save_freq == 0:
                    rudder_save_path = checkpoint_path / f"rudder_model_{update}"
                    rudder_shaper.save(str(rudder_save_path))

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        if update > 0:
            save_checkpoint(train_state, global_step, update, checkpoint_path, "interrupt")
            if rudder_shaper is not None:
                rudder_save_path = checkpoint_path / "rudder_model_interrupt"
                rudder_shaper.save(str(rudder_save_path))
            print("Checkpoint saved. Cleaning up...")
        else:
            print("No training completed, skipping checkpoint.")
    else:
        # Only save final if training completed without interrupt
        save_checkpoint(train_state, global_step, num_updates, checkpoint_path, "final")
        if rudder_shaper is not None:
            rudder_save_path = checkpoint_path / "rudder_model_final"
            rudder_shaper.save(str(rudder_save_path))
    finally:
        env.close()

    return train_state


def load_dodge_only_config(path: str):
    """Load dodge-only config from YAML."""
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw


def main():
    parser = argparse.ArgumentParser(description="Train dodge-only policy")
    parser.add_argument("--config", default="dodge_policy/configs/dodge_only.yaml")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    # Load config
    raw_config = load_dodge_only_config(args.config)

    # Also load env config using existing loader
    config = load_config(args.config)

    seed = args.seed or raw_config.get('seed', 42)
    config.seed = seed

    # PPO config from YAML
    ppo_config = DodgeOnlyConfig(
        num_steps=raw_config.get('num_steps', 2048),
        total_timesteps=raw_config.get('total_timesteps', 1_000_000),
        learning_rate=raw_config.get('learning_rate', 3e-4),
        gamma=raw_config.get('gamma', 0.99),
        gae_lambda=raw_config.get('gae_lambda', 0.95),
        clip_eps=raw_config.get('clip_eps', 0.2),
        ent_coef=raw_config.get('ent_coef', 0.01),
        vf_coef=raw_config.get('vf_coef', 0.5),
        max_grad_norm=raw_config.get('max_grad_norm', 0.5),
        update_epochs=raw_config.get('update_epochs', 4),
        num_minibatches=raw_config.get('num_minibatches', 4),
        hindsight_hit_reward=raw_config.get('hindsight_hit_reward', -2.0),
        hindsight_no_hit_reward=raw_config.get('hindsight_no_hit_reward', 2.0),
        hidden_dim=raw_config.get('hidden_dim', 128),
        anim_embed_dim=raw_config.get('anim_embed_dim', 32),
    )

    # Load dodge window model for reward shaping
    dodge_window_model = None
    dodge_window_path = raw_config.get('dodge_window_model')
    dodge_window_reward = raw_config.get('dodge_window_reward', 2.0)
    if dodge_window_path:
        print(f"Loading dodge window model: {dodge_window_path}")
        dodge_window_model = DodgeWindowModel.load(dodge_window_path)

    # Load RUDDER reward shaper
    rudder_shaper = None
    rudder_model_path = raw_config.get('rudder_model')
    rudder_update_freq = raw_config.get('rudder_update_freq', 5)
    rudder_save_freq = raw_config.get('rudder_save_freq', 50)
    if rudder_model_path:
        print(f"Loading RUDDER model: {rudder_model_path}")
        rudder_config = RudderRewardConfig(
            model_path=rudder_model_path,
            credit_scale=raw_config.get('rudder_credit_scale', 4.0),
            update_freq=rudder_update_freq,
        )
        rudder_shaper = RudderRewardShaper(rudder_config)
        print(f"  Credit scale: {rudder_config.credit_scale}")
        print(f"  Update frequency: every {rudder_update_freq} rollouts")

    # Load world model reward shaper
    world_model_shaper = None
    world_model_path = raw_config.get('world_model')
    if world_model_path:
        print(f"Loading world model: {world_model_path}")
        world_model_config = WorldModelRewardConfig(
            model_path=world_model_path,
            danger_penalty=raw_config.get('world_model_danger_penalty', -1.0),
            dodge_bonus=raw_config.get('world_model_dodge_bonus', 1.0),
            danger_threshold=raw_config.get('world_model_danger_threshold', 0.3),
            dodge_benefit_threshold=raw_config.get('world_model_dodge_benefit_threshold', 0.05),
        )
        world_model_shaper = WorldModelRewardShaper(world_model_config)
        print(f"  Danger penalty: {world_model_config.danger_penalty}")
        print(f"  Dodge bonus: {world_model_config.dodge_bonus}")
        print(f"  Danger threshold: {world_model_config.danger_threshold}")

    # Wandb
    wandb_run = None
    track = args.track or raw_config.get('track', False)
    if track:
        import wandb
        wandb_run = wandb.init(
            project=raw_config.get('wandb_project', 'dodge-policy'),
            entity=raw_config.get('wandb_entity'),
            name=f"{raw_config.get('exp_name', 'dodge_only')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "variant": "dodge_only",
                "num_steps": ppo_config.num_steps,
                "total_timesteps": ppo_config.total_timesteps,
                "learning_rate": ppo_config.learning_rate,
                "hidden_dim": ppo_config.hidden_dim,
                "anim_embed_dim": ppo_config.anim_embed_dim,
            },
        )

    # Checkpoint settings
    checkpoint_dir = raw_config.get('checkpoint_dir', 'checkpoints/dodge_only')
    save_freq = raw_config.get('save_freq', 10)

    train_dodge_only(
        config=config,
        ppo_config=ppo_config,
        dodge_window_model=dodge_window_model,
        dodge_window_reward=dodge_window_reward,
        rudder_shaper=rudder_shaper,
        rudder_update_freq=rudder_update_freq,
        rudder_save_freq=rudder_save_freq,
        world_model_shaper=world_model_shaper,
        wandb_run=wandb_run,
        checkpoint_dir=checkpoint_dir,
        save_freq=save_freq,
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
