"""Entry point for running BC training as a module."""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train import train

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train behavior cloning model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pure_cnn.yaml',
        help='Path to config file (relative to BC_training dir)',
    )
    
    args = parser.parse_args()
    
    # Resolve config path relative to BC_training directory
    bc_training_dir = Path(__file__).parent
    config_path = bc_training_dir / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    train(str(config_path))

