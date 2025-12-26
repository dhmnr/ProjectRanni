"""Quick test script to verify setup is correct."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
    except ImportError as e:
        print(f"✗ JAX import failed: {e}")
        return False
    
    try:
        import flax
        print(f"✓ Flax version: {flax.__version__}")
    except ImportError as e:
        print(f"✗ Flax import failed: {e}")
        return False
    
    try:
        import optax
        print(f"✓ Optax version: {optax.__version__}")
    except ImportError as e:
        print(f"✗ Optax import failed: {e}")
        return False
    
    try:
        import zarr
        print(f"✓ Zarr version: {zarr.__version__}")
    except ImportError as e:
        print(f"✗ Zarr import failed: {e}")
        return False
    
    try:
        from omegaconf import OmegaConf
        print(f"✓ OmegaConf imported")
    except ImportError as e:
        print(f"✗ OmegaConf import failed: {e}")
        return False
    
    try:
        from common.dataset import ZarrGameplayDataset
        print("✓ Dataset module imported")
    except ImportError as e:
        print(f"✗ Dataset import failed: {e}")
        return False
    
    try:
        from common.metrics import compute_accuracy
        print("✓ Metrics module imported")
    except ImportError as e:
        print(f"✗ Metrics import failed: {e}")
        return False
    
    try:
        from models.pure_cnn import create_model
        print("✓ PureCNN model imported")
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_jax_devices():
    """Check available JAX devices."""
    import jax
    
    print("\nChecking JAX devices...")
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    for device in devices:
        print(f"  - {device.device_kind}: {device}")
    
    if any('gpu' in str(d).lower() for d in devices):
        print("✓ GPU available!")
    else:
        print("⚠ No GPU detected (CPU only)")


def test_config():
    """Test config loading."""
    from omegaconf import OmegaConf
    
    print("\nTesting config loading...")
    config_path = Path(__file__).parent.parent / "configs" / "pure_cnn.yaml"
    
    try:
        config = OmegaConf.load(config_path)
        print(f"✓ Config loaded from {config_path}")
        print(f"  Model: {config.model.name}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Epochs: {config.training.num_epochs}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation."""
    import jax
    import jax.numpy as jnp
    from models.pure_cnn import create_model
    
    print("\nTesting model creation...")
    
    try:
        # Create model
        model = create_model(
            num_actions=7,
            conv_features=(32, 64, 128, 256),
            dense_features=(512, 256),
            dropout_rate=0.1,
        )
        print("✓ Model created")
        
        # Initialize with dummy input
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((2, 3, 288, 512))  # Batch of 2
        
        variables = model.init(
            {'params': rng, 'dropout': rng},
            dummy_input,
            training=False,
        )
        
        params = variables['params']
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"✓ Model initialized with {param_count:,} parameters")
        
        # Forward pass
        logits = model.apply(variables, dummy_input, training=False)
        print(f"✓ Forward pass successful, output shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*60)
    print("BC Training Setup Test")
    print("="*60)
    
    success = True
    
    success = test_imports() and success
    test_jax_devices()
    success = test_config() and success
    success = test_model_creation() and success
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed! Setup is ready.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*60)

