# Behavior Cloning Training

This directory contains training code for behavior cloning models to solve Elden Ring boss fights.

## Structure

```
BC_training/
├── common/              # Shared utilities
│   ├── dataset.py       # Zarr data loader
│   └── metrics.py       # Evaluation metrics
├── models/              # Model implementations
│   └── pure_cnn/        # Pure CNN baseline (vision-only)
│       ├── __init__.py
│       └── model.py
├── configs/             # Training configurations
│   └── pure_cnn.yaml    # Pure CNN config
├── train.py             # Main training script
└── requirements.txt     # Python dependencies
```

## Quick Setup (Cloud/Pod)

One-liner to set up a fresh training environment:

```bash
curl -sSL https://raw.githubusercontent.com/dhmnr/ProjectRanni/main/BC_training/setup_pod.sh | bash
```

This clones the repo, installs uv and dependencies (including JAX with CUDA), downloads the dataset, and configures WandB.

## Installation

```bash
# From project root
uv sync --group bc_training

# Or sync all project dependencies
uv sync
```

## Usage

### Training

```bash
# Train with default config (from project root)
uv run -m BC_training --config configs/pure_cnn.yaml

# Or from BC_training directory
cd BC_training
uv run python train.py --config configs/pure_cnn.yaml

# analyze dataset
uv run -m BC_training.analyze_dataset --dataset your_path

# Specify a different config
uv run -m BC_training --config configs/your_custom_config.yaml
```

### Configuration

Edit `configs/pure_cnn.yaml` to customize:
- Dataset path and split ratios
- Model architecture (conv/dense layers, dropout, etc.)
- Training hyperparameters (batch size, learning rate, epochs)
- Logging (WandB project name, run name)

### Monitoring

Training metrics are logged to:
- Console output (real-time)
- WandB dashboard (if enabled in config)

Key metrics:
- **Loss**: Binary cross-entropy loss
- **Accuracy**: Exact match accuracy (all actions correct)
- **Per-action metrics**: Precision, recall, F1 per action
- **Distribution distance**: How well predicted action frequencies match ground truth

### Checkpoints

Models are saved to `checkpoints/{model_name}/`:
- Best model (highest validation accuracy)
- Periodic checkpoints every N epochs

## Models

### Pure CNN (Baseline)

Vision-only model that predicts actions directly from RGB frames.

**Architecture:**
```
Input [3, 288, 512] RGB frames
  ↓
Conv blocks (32 → 64 → 128 → 256)
  ↓
Global Average Pooling
  ↓
Dense layers (512 → 256)
  ↓
Output [7] actions (multi-label)
```

**Key features:**
- Simple CNN baseline for pipeline validation
- Batch normalization and dropout for regularization
- Class-weighted BCE loss for imbalanced actions
- ~2-3M parameters

### Future Models

- **State + Vision**: Fuse game state (HP, positions) with vision
- **Temporal**: Add LSTM/Transformer for temporal modeling
- **Larger CNNs**: ResNet, EfficientNet backbones
- **VLA-style**: Vision-language-action transformers

## Adding New Models

1. Create model directory: `models/your_model/`
2. Implement model in `models/your_model/model.py` with `create_model()` function
3. Create config: `configs/your_model.yaml`
4. Add model loading logic to `train.py` (if needed)
5. Run training with new config

## Data Format

Expected zarr dataset structure:
```
dataset.zarr/
├── episode_0/
│   ├── frames: [N, 3, H, W] uint8
│   ├── actions: [N, 7] bool
│   └── state: [N, 19] float32 (optional)
├── episode_1/
...
```

See `gameplay_pipeline/explore_zarr_dataset.ipynb` for data exploration.

## Notes

- Start with Pure CNN to validate the full pipeline
- Once baseline works, iterate on architecture improvements
- Use WandB to compare different model variants
- Checkpoint management keeps only the best model + last N checkpoints

