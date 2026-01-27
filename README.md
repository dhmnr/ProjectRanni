# Project Ranni

Training an RL agent to defeat Elden Ring bosses, starting with Margit the Fell Omen.

## Goal

Build an end-to-end system that can:
1. Collect gameplay data from human demonstrations and random exploration
2. Learn world models to understand boss attack patterns and timing
3. Train RL policies with shaped rewards to master combat

## Current Approach

**Dodge-only policy**: The agent always walks toward the boss and only decides *when* to dodge. This simplifies the action space from complex movement + attack combos to a binary decision (dodge or not), making the credit assignment problem tractable.

**Reward shaping**: Raw RL rewards (survive = good, die = bad) are too sparse. We use multiple reward shaping techniques:
- **Dodge window model**: Rewards dodging during the correct timing windows for each boss attack
- **World model**: Predicts P(hit | state, action) to reward dodging when it actually reduces hit probability
- **RUDDER**: Learns to redistribute episode returns to individual timesteps based on state-action patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Collection                              │
├─────────────────────────────────────────────────────────────────────┤
│  gameplay_pipeline/     Live recording via Siphon (memory reading)   │
│  record_episode.py      Record human demonstrations                  │
│  collect_rudder_data.py Collect random policy rollouts               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         World Models                                 │
├─────────────────────────────────────────────────────────────────────┤
│  world_model.py         Predicts next state + P(hit) given action   │
│  hit_predictor.py       Specialized hit probability predictor        │
│  dodge_window_model.py  Gaussian timing windows per boss animation   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Reward Shaping                                │
├─────────────────────────────────────────────────────────────────────┤
│  world_model_reward.py  Shapes reward based on P(hit) predictions   │
│  rudder_reward.py       LSTM-based credit assignment                 │
│  dodge_window_model.py  Rewards correct dodge timing                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RL Training                                  │
├─────────────────────────────────────────────────────────────────────┤
│  train_dodge_only.py    PPO training for dodge-only policy          │
│  train.py               Full action space PPO training               │
│  ppo.py                 Core PPO implementation (JAX/Flax)          │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ProjectRanni/
├── dodge_policy/                 # Main RL training module
│   ├── configs/                  # Training configurations
│   │   ├── dodge_only.yaml      # Dodge-only policy config
│   │   └── default.yaml         # Full policy config
│   │
│   ├── # World Models
│   ├── world_model.py           # Dreamer-style world model (JAX)
│   ├── world_model_reward.py    # P(hit)-based reward shaping
│   ├── hit_predictor.py         # Hit probability predictor
│   ├── dodge_window_model.py    # Timing window model
│   │
│   ├── # Reward Shaping
│   ├── rudder_model.py          # RUDDER LSTM for credit assignment
│   ├── rudder_reward.py         # RUDDER reward shaper integration
│   │
│   ├── # Training
│   ├── train_dodge_only.py      # Dodge-only PPO training
│   ├── train.py                 # Full action space training
│   ├── ppo.py                   # PPO implementation
│   ├── agent.py                 # Full action space agent
│   ├── dodge_only_agent.py      # Dodge-only agent
│   │
│   ├── # Data Collection
│   ├── collect_rudder_data.py   # Collect random policy data
│   ├── record_episode.py        # Record human demonstrations
│   ├── build_windows_from_expert.py  # Build dodge windows from demos
│   │
│   ├── # Environment
│   ├── env_factory.py           # Environment creation
│   ├── dodge_only_wrapper.py    # Simplifies to binary action
│   ├── action_wrapper.py        # Action space utilities
│   └── anim_vocab.json          # Boss animation vocabulary
│
├── gameplay_pipeline/            # Live gameplay recording
│   ├── record_gameplay.py       # Record via Siphon
│   └── hf_upload.py             # Upload to HuggingFace
│
├── BC_training/                  # Behavior cloning (JAX/Flax)
│   ├── models/                  # CNN, transformer models
│   └── train.py                 # BC training script
│
├── yt_data_pipeline/            # YouTube video processing
│   └── ...                      # Download, extract, process
│
├── rudder_data_v2/              # Training data (225 episodes)
├── expert_data/                 # Human demonstration data
├── paths/                       # Arena boundary definitions
└── checkpoints/                 # Saved model weights
```

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# For GPU training (CUDA)
uv sync --group dodge_policy
```

## Quick Start

### 1. Collect Data

```bash
# Collect random policy episodes (for world model training)
uv run python -m dodge_policy.collect_rudder_data \
    --episodes 100 \
    --dodge-prob 0.2 \
    --output-dir rudder_data_v2

# Record human demonstrations
uv run python -m dodge_policy.record_episode \
    --output-dir expert_data
```

### 2. Train World Model

```bash
# Train world model on collected data
uv run python -m dodge_policy.world_model \
    --train \
    --data-dir rudder_data_v2 \
    --epochs 100 \
    --output world_model_v2.npz
```

### 3. Build Dodge Windows

```bash
# From expert demonstrations
uv run python -m dodge_policy.build_windows_from_expert \
    --data-dir expert_data \
    --output dodge_windows_expert.json
```

### 4. Train Dodge Policy

```bash
# Train with all reward shapers
uv run python -m dodge_policy.train_dodge_only \
    --config dodge_policy/configs/dodge_only.yaml \
    --track  # Enable wandb logging
```

## Configuration

Edit `dodge_policy/configs/dodge_only.yaml`:

```yaml
# Reward shaping
dodge_window_model: "dodge_windows_expert.json"
dodge_window_reward: 2.0

world_model: "world_model_v2.npz"
world_model_danger_penalty: -1.0
world_model_dodge_bonus: 1.0

# Training
total_timesteps: 500_000
learning_rate: 0.001
num_steps: 2048

# Environment
env:
  host: "192.168.48.1:50051"  # Siphon server
  hit_penalty: -2.0
  dodge_penalty: -0.5
```

## World Model

The world model learns from random policy data to predict:
- **Next boss animation** (94.6% accuracy)
- **P(hit)** given state and action (86.7% accuracy)
- **Distance changes** from player actions

This enables reward shaping without human-labeled dodge windows:

```python
from dodge_policy.world_model_reward import load_world_model_shaper

shaper = load_world_model_shaper("world_model_v2.npz")

# Get reward shaping for a state-action
reward, info = shaper.compute_reward_shaping(
    anim_idx=15,        # Boss animation
    elapsed_frames=30,   # Frames into animation
    dist_to_boss=3.0,   # Distance
    action=1,           # 1 = dodge
)
# reward > 0 if dodging reduces P(hit)
```

## RUDDER Credit Assignment

RUDDER learns to redistribute sparse episode returns to individual timesteps:

```python
from dodge_policy.rudder_reward import RudderRewardShaper, RudderRewardConfig

config = RudderRewardConfig(credit_scale=4.0)
shaper = RudderRewardShaper(config)

# Compute per-step credit from episode data
credit = shaper.compute_credit(
    boss_anim_ids, hero_anim_ids, dist_to_boss,
    hero_hp, actions, damage_taken
)
# credit[t] = contribution of timestep t to episode return
```

## Environment

Requires [Siphon](https://github.com/dhmnr/pysiphon) server running on Windows with Elden Ring:

```bash
# On Windows (with Elden Ring running)
siphon-server --config elden_ring.toml

# On Linux (training)
uv run python -m dodge_policy.train_dodge_only
```

The environment provides:
- **Observations**: boss_anim_id, elapsed_frames, dist_to_boss, hero_hp, hero_anim_id
- **Actions**: MultiBinary(5) for full, Discrete(2) for dodge-only
- **Rewards**: Configurable penalties for hits, dodges, out-of-bounds

## Results

Current world model (trained on 225 episodes, 230k transitions):
- Boss animation prediction: **94.6%** accuracy
- Hit prediction: **86.7%** accuracy
- Learns attack timing patterns from random exploration data

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run ruff format .

# Type check
uv run pyright
```

## References

- [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857)
- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) (Dreamer)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO)

## License

See LICENSE file for details.
