# BC Training - Quick Start Guide

## 1. Install Dependencies

```bash
# From project root
uv sync --group bc_training

# Or sync all dependencies
uv sync
```

## 2. Verify Setup

```bash
# From project root
uv run python BC_training/scripts/test_setup.py

# Or from BC_training directory
cd BC_training
uv run python scripts/test_setup.py
```

This will check:
- ✓ All dependencies are installed
- ✓ JAX detects GPU (if available)
- ✓ Config loads correctly
- ✓ Model can be created and initialized

## 3. Configure WandB (Optional but Recommended)

```bash
# Login to WandB
wandb login

# Or set API key
export WANDB_API_KEY=your_api_key_here
```

Edit `configs/pure_cnn.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_project: "ProjectRanni-BC"
  wandb_entity: "your_username"  # Set this!
  wandb_run_name: "pure_cnn_baseline"
```

## 4. Configure Dataset Path

Edit `configs/pure_cnn.yaml`:
```yaml
dataset:
  path: "../dataset/margit_100_512x288.zarr"  # Update this path!
```

## 5. Run Training

```bash
# From project root (recommended)
uv run -m BC_training --config configs/pure_cnn.yaml

# Or from BC_training directory
cd BC_training
uv run python train.py --config configs/pure_cnn.yaml
```

**Expected output:**
```
Loaded config from configs/pure_cnn.yaml
Split 100 episodes: 80 train, 20 val
Loaded dataset from ../dataset/margit_100_512x288.zarr
Using 80 episodes
Model initialized with 2,XXX,XXX parameters
Starting training...

Epoch 1/50
Train Loss: 0.XXXX, Train Acc: 0.XXXX
...
```

## 6. Monitor Training

**Console:** Real-time logs showing loss and accuracy

**WandB Dashboard:** 
- Visit wandb.ai and check your project
- View loss curves, per-action metrics, learning rate schedule

**Checkpoints:**
- Saved to `checkpoints/pure_cnn/`
- Best model: `best_checkpoint_*`
- Periodic: `checkpoint_*`

## 7. Training Tips

### First Time Setup
1. Start with a **small test run** to verify everything works:
   - Edit config: `num_epochs: 2`
   - Run training to completion
   - Check that checkpoints are saved

2. Then do a **full training run**:
   - Reset `num_epochs: 50`
   - Monitor WandB for ~1-2 hours

### Expected Performance (Pure CNN Baseline)
- **Training time**: ~1-2 hours on GPU (RTX 3090/4090)
- **Target validation accuracy**: 70-85% (depending on data quality)
- **Per-action F1**: 0.6-0.8 for common actions, lower for rare actions

### If Training Fails

**Import errors:**
```bash
uv sync --group bc_training
uv run python BC_training/scripts/test_setup.py
```

**Dataset not found:**
- Check `dataset.path` in config matches your zarr location
- Verify zarr structure: `python -c "import zarr; print(list(zarr.open('path/to/dataset.zarr').keys()))"`

**Out of memory:**
- Reduce `batch_size` in config (try 16 or 8)
- Reduce model size: smaller `conv_features` and `dense_features`

**Loss not decreasing:**
- Check action distribution (are actions too imbalanced?)
- Try increasing learning rate
- Enable/disable class weights
- Check data quality in `explore_zarr_dataset.ipynb`

## 8. Next Steps

### Experiment with Hyperparameters

Create a new config file (e.g., `configs/pure_cnn_v2.yaml`):
```yaml
# Try larger model
model:
  conv_features: [64, 128, 256, 512]
  dense_features: [1024, 512]
  dropout_rate: 0.2
```

Then train:
```bash
uv run -m BC_training --config configs/pure_cnn_v2.yaml
```

Compare runs in WandB!

### Add New Model Variants

1. Copy `models/pure_cnn/` to `models/your_new_model/`
2. Modify architecture in `model.py`
3. Create config `configs/your_new_model.yaml`
4. Train and compare

### Future Work

Once Pure CNN baseline is working:
- Add state features (HP, positions, etc.)
- Add temporal modeling (LSTM/Transformer)
- Try larger backbones (ResNet, EfficientNet)
- Move to RL fine-tuning phase

## Troubleshooting

### Common Issues

**"No module named 'jax'"**
→ `uv sync --group bc_training`

**"CUDA not found" / "GPU not available"**
→ Install CUDA-compatible jaxlib: `uv pip install jax[cuda12]`

**WandB not logging**
→ `wandb login` and check `wandb_entity` in config

**Dataset loading slow**
→ Normal for first epoch (zarr is lazy-loaded), speeds up after

**Training stuck at low accuracy**
→ Check data quality, try different learning rates, enable class weights

Need help? Check the main README.md or open an issue!

