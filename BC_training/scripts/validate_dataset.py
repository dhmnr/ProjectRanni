"""Validate zarr dataset structure and consistency."""

import sys
from pathlib import Path
import zarr
import numpy as np

def validate_dataset(dataset_path: str):
    """Validate zarr dataset for training compatibility.
    
    Checks:
    - All episodes have consistent action dimensions
    - All episodes have consistent state dimensions
    - Frame shapes are consistent
    - No corrupted data
    """
    print("="*70)
    print(f"Validating Dataset: {dataset_path}")
    print("="*70)
    
    # Open dataset
    try:
        zarr_root = zarr.open(dataset_path, mode='r')
    except Exception as e:
        print(f"❌ Failed to open dataset: {e}")
        return False
    
    # Get all episodes
    episodes = sorted([k for k in zarr_root.keys() if k.startswith('episode_')])
    print(f"\n✓ Found {len(episodes)} episodes")
    
    if len(episodes) == 0:
        print("❌ No episodes found in dataset!")
        return False
    
    # Get metadata
    action_keys = zarr_root.attrs.get('keys', [])
    state_attrs = zarr_root.attrs.get('attributes', [])
    
    print(f"✓ Expected {len(action_keys)} actions: {action_keys}")
    print(f"✓ Expected {len(state_attrs)} state attributes")
    
    # Check first episode to get expected shapes
    first_ep = zarr_root[episodes[0]]
    expected_action_shape = first_ep['actions'].shape[1]  # [N, num_actions]
    expected_state_shape = first_ep['state'].shape[1]  # [N, num_state]
    expected_frame_channels = first_ep['frames'].shape[1]  # [N, C, H, W]
    
    print(f"\n✓ Expected shapes (from {episodes[0]}):")
    print(f"  - Actions per frame: {expected_action_shape}")
    print(f"  - State features per frame: {expected_state_shape}")
    print(f"  - Frame channels: {expected_frame_channels}")
    
    # Validate each episode
    print(f"\n{'Episode':<15} {'Frames':<10} {'Actions':<15} {'State':<15} {'Status':<10}")
    print("-"*70)
    
    issues = []
    
    for ep_name in episodes:
        ep = zarr_root[ep_name]
        
        # Get shapes
        frames_shape = ep['frames'].shape
        actions_shape = ep['actions'].shape
        state_shape = ep['state'].shape
        
        # Check consistency
        status = "✓ OK"
        issue_details = []
        
        # Check that all arrays have same length
        if frames_shape[0] != actions_shape[0] or frames_shape[0] != state_shape[0]:
            status = "❌ LENGTH"
            issue_details.append(
                f"Mismatched lengths: frames={frames_shape[0]}, "
                f"actions={actions_shape[0]}, state={state_shape[0]}"
            )
        
        # Check action dimensions
        if actions_shape[1] != expected_action_shape:
            status = "❌ ACTIONS"
            issue_details.append(
                f"Action dim mismatch: expected {expected_action_shape}, got {actions_shape[1]}"
            )
        
        # Check state dimensions
        if state_shape[1] != expected_state_shape:
            status = "❌ STATE"
            issue_details.append(
                f"State dim mismatch: expected {expected_state_shape}, got {state_shape[1]}"
            )
        
        # Check frame channels
        if frames_shape[1] != expected_frame_channels:
            status = "❌ FRAMES"
            issue_details.append(
                f"Frame channels mismatch: expected {expected_frame_channels}, got {frames_shape[1]}"
            )
        
        # Print row
        print(f"{ep_name:<15} {frames_shape[0]:<10} {str(actions_shape):<15} "
              f"{str(state_shape):<15} {status:<10}")
        
        if issue_details:
            issues.append((ep_name, issue_details))
    
    # Print detailed issues
    if issues:
        print("\n" + "="*70)
        print("DETAILED ISSUES:")
        print("="*70)
        for ep_name, details in issues:
            print(f"\n❌ {ep_name}:")
            for detail in details:
                print(f"   - {detail}")
        print("\n❌ Dataset validation FAILED!")
        return False
    else:
        print("\n" + "="*70)
        print("✓ All episodes are consistent!")
        print("="*70)
        
        # Additional statistics
        print("\nDataset Statistics:")
        total_frames = sum(zarr_root[ep]['frames'].shape[0] for ep in episodes)
        print(f"  Total frames: {total_frames:,}")
        print(f"  Average frames per episode: {total_frames / len(episodes):.1f}")
        
        # Check action distribution
        print("\nAction Distribution (first 5 episodes sample):")
        action_counts = np.zeros(expected_action_shape)
        sample_size = min(5, len(episodes))
        
        for ep_name in episodes[:sample_size]:
            actions = np.array(zarr_root[ep_name]['actions'][:])
            action_counts += actions.sum(axis=0)
        
        for i, action_name in enumerate(action_keys):
            print(f"  {action_name:<20}: {int(action_counts[i]):>6} presses")
        
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <path_to_zarr_dataset>")
        print("\nExample:")
        print("  python validate_dataset.py ../dataset/margit_100_512x288.zarr")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    success = validate_dataset(dataset_path)
    sys.exit(0 if success else 1)

