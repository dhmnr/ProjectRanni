"""Analyze dataset to get statistics needed for model configuration.

Scans zarr dataset to find:
- Unique HeroAnimId values
- Unique NpcAnimId values
- Other useful statistics

Usage:
    uv run python -m BC_training.analyze_dataset --dataset ./dataset/margit_100_256x144.zarr
"""

import argparse
import zarr
import numpy as np
from pathlib import Path
from collections import Counter
import json


def analyze_animation_ids(dataset_path: str) -> dict:
    """Analyze animation IDs in the dataset.
    
    Args:
        dataset_path: Path to zarr dataset
        
    Returns:
        Dict with analysis results
    """
    root = zarr.open(dataset_path, mode='r')
    
    # Get state attribute names
    state_attrs = root.attrs.get('attributes', [])
    print(f"State attributes: {state_attrs}")
    
    # Find indices for animation IDs
    hero_anim_idx = state_attrs.index('HeroAnimId') if 'HeroAnimId' in state_attrs else None
    npc_anim_idx = state_attrs.index('NpcAnimId') if 'NpcAnimId' in state_attrs else None
    
    if hero_anim_idx is None or npc_anim_idx is None:
        raise ValueError(f"Animation ID attributes not found. Available: {state_attrs}")
    
    print(f"HeroAnimId index: {hero_anim_idx}")
    print(f"NpcAnimId index: {npc_anim_idx}")
    
    # Collect all animation IDs
    hero_anim_ids = []
    npc_anim_ids = []
    
    episodes = sorted([k for k in root.keys() if k.startswith('episode_')])
    print(f"\nScanning {len(episodes)} episodes...")
    
    for ep_name in episodes:
        ep = root[ep_name]
        state = ep['state'][:]
        
        hero_anim_ids.extend(state[:, hero_anim_idx].astype(int).tolist())
        npc_anim_ids.extend(state[:, npc_anim_idx].astype(int).tolist())
    
    # Get unique IDs and counts
    hero_unique = sorted(set(hero_anim_ids))
    npc_unique = sorted(set(npc_anim_ids))
    
    hero_counts = Counter(hero_anim_ids)
    npc_counts = Counter(npc_anim_ids)
    
    print(f"\n=== Hero Animation IDs ===")
    print(f"Unique count: {len(hero_unique)}")
    print(f"Min ID: {min(hero_unique)}, Max ID: {max(hero_unique)}")
    print(f"Top 10 most common:")
    for anim_id, count in hero_counts.most_common(10):
        print(f"  {anim_id}: {count:,} frames ({100*count/len(hero_anim_ids):.1f}%)")
    
    print(f"\n=== NPC Animation IDs ===")
    print(f"Unique count: {len(npc_unique)}")
    print(f"Min ID: {min(npc_unique)}, Max ID: {max(npc_unique)}")
    print(f"Top 10 most common:")
    for anim_id, count in npc_counts.most_common(10):
        print(f"  {anim_id}: {count:,} frames ({100*count/len(npc_anim_ids):.1f}%)")
    
    # Create mapping from ID to index (for embedding lookup)
    hero_id_to_idx = {id_: idx for idx, id_ in enumerate(hero_unique)}
    npc_id_to_idx = {id_: idx for idx, id_ in enumerate(npc_unique)}
    
    results = {
        'hero_anim': {
            'vocab_size': len(hero_unique),
            'unique_ids': hero_unique,
            'id_to_idx': hero_id_to_idx,
            'min_id': min(hero_unique),
            'max_id': max(hero_unique),
        },
        'npc_anim': {
            'vocab_size': len(npc_unique),
            'unique_ids': npc_unique,
            'id_to_idx': npc_id_to_idx,
            'min_id': min(npc_unique),
            'max_id': max(npc_unique),
        },
        'total_frames': len(hero_anim_ids),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset for model configuration')
    parser.add_argument('--dataset', type=str, required=True, help='Path to zarr dataset')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    args = parser.parse_args()
    
    results = analyze_animation_ids(args.dataset)
    
    print(f"\n=== Summary ===")
    print(f"Hero animation vocab size: {results['hero_anim']['vocab_size']}")
    print(f"NPC animation vocab size: {results['npc_anim']['vocab_size']}")
    print(f"Total frames analyzed: {results['total_frames']:,}")
    
    print(f"\n=== Recommended config ===")
    print(f"state_preprocessing:")
    print(f"  hero_anim_vocab_size: {results['hero_anim']['vocab_size']}")
    print(f"  npc_anim_vocab_size: {results['npc_anim']['vocab_size']}")
    print(f"  anim_embed_dim: 16  # or 32 for more capacity")
    
    # Always save mappings for use by state_preprocessing
    output_path = args.output or Path(args.dataset).parent / 'anim_id_mappings.json'
    output = {
        'hero_anim_vocab_size': results['hero_anim']['vocab_size'],
        'npc_anim_vocab_size': results['npc_anim']['vocab_size'],
        # Store as string keys for JSON compatibility
        'hero_id_to_idx': {str(k): v for k, v in results['hero_anim']['id_to_idx'].items()},
        'npc_id_to_idx': {str(k): v for k, v in results['npc_anim']['id_to_idx'].items()},
        'total_frames': results['total_frames'],
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nMappings saved to {output_path}")


if __name__ == '__main__':
    main()

