# Using BC Training with UV

## Quick Reference

### Installation

```bash
# From project root
uv sync --group bc_training

# Or sync everything
uv sync
```

### Run Training

```bash
# Method 1: Run as module (recommended, from project root)
uv run -m BC_training --config configs/pure_cnn.yaml

# Method 2: Run script directly (from BC_training directory)
cd BC_training
uv run python train.py --config configs/pure_cnn.yaml
```

### Test Setup

```bash
# From project root
uv run python BC_training/scripts/test_setup.py

# From BC_training directory
cd BC_training
uv run python scripts/test_setup.py
```

### Install GPU Support (Optional)

```bash
# For CUDA 12
uv pip install --upgrade "jax[cuda12]"

# For CUDA 11
uv pip install --upgrade "jax[cuda11]"
```

## Complete Workflow

```bash
# 1. Install dependencies
cd /home/ubuntu/ProjectRanni
uv sync

# 2. Test that everything works
uv run python BC_training/scripts/test_setup.py

# 3. Configure your dataset path
nano BC_training/configs/pure_cnn.yaml
# Edit dataset.path to point to your zarr file

# 4. Login to WandB
wandb login

# 5. Train!
uv run -m BC_training --config configs/pure_cnn.yaml
```

## Why UV?

- **Faster**: UV is much faster than pip
- **Reproducible**: Lock file ensures exact dependency versions
- **Organized**: Dependency groups keep pipelines separate
- **Modern**: Better dependency resolution than pip

## Common Commands

```bash
# Add a new dependency to bc_training group
# Edit pyproject.toml manually, then:
uv sync

# Update dependencies
uv sync --upgrade

# Lock dependencies (for reproducibility)
uv lock

# Show installed packages
uv pip list

# Run Python with project dependencies
uv run python -c "import jax; print(jax.__version__)"
```

## Troubleshooting

### "Module not found" errors

```bash
# Re-sync dependencies
uv sync --group bc_training

# Verify installation
uv pip list | grep jax
```

### GPU not detected

```bash
# Install CUDA-compatible JAX
uv pip install --upgrade "jax[cuda12]"

# Test GPU
uv run python -c "import jax; print(jax.devices())"
```

### Import errors when running scripts

Make sure you're using `uv run`:
```bash
# ✓ Correct
uv run python BC_training/train.py

# ✗ Wrong (won't have dependencies)
python BC_training/train.py
```

## Dependency Groups Explained

Your project has these groups:
- `yt_data_pipeline` - YouTube video processing dependencies
- `gameplay_pipeline` - Live gameplay recording dependencies
- `bc_training` - Machine learning training dependencies
- `dev` - Development tools (Jupyter, etc.)

By default, `uv sync` installs all groups (see `default-groups` in `pyproject.toml`).

To install only specific groups:
```bash
uv sync --only bc_training
```

