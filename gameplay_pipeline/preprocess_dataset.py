"""
Preprocess gameplay dataset state attributes.

Simple, direct preprocessing for game state data:
- Normalize HP/SP/FP to 0-1 range
- Compute distance vectors
- Compute relative angles
- etc.

Usage:
    python preprocess_dataset.py input.zarr -o preprocessed.zarr
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()


# =============================================================================
# State Attribute Indices
# =============================================================================
# [ 0] HeroHp           [ 1] HeroMaxHp
# [ 2] HeroSp           [ 3] HeroMaxSp
# [ 4] HeroFp           [ 5] HeroMaxFp
# [ 6] HeroGlobalPosX   [ 7] HeroGlobalPosY   [ 8] HeroGlobalPosZ
# [ 9] HeroAngle        [10] HeroAnimId
# [11] NpcHp            [12] NpcMaxHp         [13] NpcId
# [14] NpcGlobalPosX    [15] NpcGlobalPosY    [16] NpcGlobalPosZ
# [17] NpcGlobalPosAngle [18] NpcAnimId

HERO_HP = 0
HERO_MAX_HP = 1
HERO_SP = 2
HERO_MAX_SP = 3
HERO_FP = 4
HERO_MAX_FP = 5
HERO_POS_X = 6
HERO_POS_Y = 7
HERO_POS_Z = 8
HERO_ANGLE = 9
HERO_ANIM_ID = 10
NPC_HP = 11
NPC_MAX_HP = 12
NPC_ID = 13
NPC_POS_X = 14
NPC_POS_Y = 15
NPC_POS_Z = 16
NPC_ANGLE = 17
NPC_ANIM_ID = 18


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_state(state: np.ndarray, fps: float = 17.0) -> tuple[np.ndarray, list[str]]:
    """
    Preprocess raw state array into normalized features.
    
    Args:
        state: Raw state array [N, 19]
        fps: Frames per second (for velocity computation)
        
    Returns:
        Tuple of (processed_state, feature_names)
    """
    n_frames = state.shape[0]
    features = []
    names = []
    
    # --- Normalized HP/SP/FP (0-1 range) ---
    hero_hp_ratio = state[:, HERO_HP] / (state[:, HERO_MAX_HP] + 1e-6)
    features.append(np.clip(hero_hp_ratio, 0, 1))
    names.append("hero_hp_ratio")
    
    hero_sp_ratio = state[:, HERO_SP] / (state[:, HERO_MAX_SP] + 1e-6)
    features.append(np.clip(hero_sp_ratio, 0, 1))
    names.append("hero_sp_ratio")
    
    hero_fp_ratio = state[:, HERO_FP] / (state[:, HERO_MAX_FP] + 1e-6)
    features.append(np.clip(hero_fp_ratio, 0, 1))
    names.append("hero_fp_ratio")
    
    npc_hp_ratio = state[:, NPC_HP] / (state[:, NPC_MAX_HP] + 1e-6)
    features.append(np.clip(npc_hp_ratio, 0, 1))
    names.append("npc_hp_ratio")
    
    # --- Distance to NPC ---
    hero_pos = state[:, [HERO_POS_X, HERO_POS_Y, HERO_POS_Z]]
    npc_pos = state[:, [NPC_POS_X, NPC_POS_Y, NPC_POS_Z]]
    
    # 3D distance
    diff_3d = npc_pos - hero_pos
    distance_3d = np.linalg.norm(diff_3d, axis=1)
    features.append(distance_3d / 50.0)  # Normalize by ~max arena size
    names.append("distance_3d")
    
    # 2D distance (horizontal plane)
    diff_2d = state[:, [NPC_POS_X, NPC_POS_Y]] - state[:, [HERO_POS_X, HERO_POS_Y]]
    distance_2d = np.linalg.norm(diff_2d, axis=1)
    features.append(distance_2d / 50.0)
    names.append("distance_2d")
    
    # --- Relative position to NPC (normalized) ---
    rel_pos = diff_3d / 50.0
    features.append(rel_pos[:, 0])
    names.append("rel_npc_x")
    features.append(rel_pos[:, 1])
    names.append("rel_npc_y")
    features.append(rel_pos[:, 2])
    names.append("rel_npc_z")
    
    # --- Hero facing direction (sin/cos) ---
    hero_angle = state[:, HERO_ANGLE]
    features.append(np.sin(hero_angle))
    names.append("hero_facing_sin")
    features.append(np.cos(hero_angle))
    names.append("hero_facing_cos")
    
    # --- Relative angle to NPC ---
    # Angle from hero to NPC in world space
    angle_to_npc = np.arctan2(diff_2d[:, 1], diff_2d[:, 0])
    # Relative angle (how much hero needs to turn to face NPC)
    rel_angle = angle_to_npc - hero_angle
    # Normalize to [-pi, pi]
    rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))
    features.append(np.sin(rel_angle))
    names.append("angle_to_npc_sin")
    features.append(np.cos(rel_angle))
    names.append("angle_to_npc_cos")
    
    # --- Hero velocity (position change per frame) ---
    hero_vel = np.zeros_like(hero_pos)
    hero_vel[1:] = (hero_pos[1:] - hero_pos[:-1]) * fps
    hero_vel = hero_vel / 10.0  # Normalize
    features.append(hero_vel[:, 0])
    names.append("hero_vel_x")
    features.append(hero_vel[:, 1])
    names.append("hero_vel_y")
    features.append(hero_vel[:, 2])
    names.append("hero_vel_z")
    
    # Stack all features
    processed = np.stack(features, axis=1).astype(np.float32)
    
    return processed, names


def preprocess_dataset(input_path: str, output_path: str):
    """
    Preprocess entire zarr dataset.
    
    Args:
        input_path: Path to input zarr dataset
        output_path: Path to output zarr dataset
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Open input
    input_root = zarr.open(str(input_path), mode='r')
    
    # Get episodes
    episodes = sorted([
        k for k in input_root.keys() 
        if k.startswith('episode_')
    ], key=lambda x: int(x.split('_')[1]))
    
    console.print(f"[cyan]Input: {input_path}[/cyan]")
    console.print(f"[cyan]Episodes: {len(episodes)}[/cyan]")
    
    # Remove existing output
    if output_path.exists():
        import shutil
        console.print(f"[yellow]Removing existing {output_path}[/yellow]")
        shutil.rmtree(output_path)
    
    # Create output
    output_root = zarr.open_group(str(output_path), mode='w')
    
    # Copy global metadata
    for key, value in input_root.attrs.items():
        output_root.attrs[key] = value
    
    # Process episodes
    all_processed = []
    feature_names = None
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing", total=len(episodes))
        
        for ep_name in episodes:
            input_ep = input_root[ep_name]
            fps = input_ep.attrs.get('fps', 17.0)
            
            # Preprocess state
            raw_state = input_ep['state'][:]
            processed_state, feature_names = preprocess_state(raw_state, fps)
            all_processed.append(processed_state)
            
            # Create output episode
            output_ep = output_root.create_group(ep_name)
            
            # Copy frames and actions
            output_ep.array(
                'frames', input_ep['frames'],
                chunks=input_ep['frames'].chunks,
                dtype=input_ep['frames'].dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            output_ep.array(
                'actions', input_ep['actions'],
                chunks=input_ep['actions'].chunks,
                dtype=input_ep['actions'].dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            # Store processed state
            output_ep.array(
                'state', processed_state,
                chunks=(100, processed_state.shape[1]),
                dtype='float32',
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            # Store raw state too
            output_ep.array(
                'raw_state', raw_state,
                chunks=input_ep['state'].chunks,
                dtype='float32',
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
            
            # Copy episode attrs
            for key, value in input_ep.attrs.items():
                output_ep.attrs[key] = value
            output_ep.attrs['state_features'] = feature_names
            
            progress.update(task, advance=1)
    
    # Store feature info
    output_root.attrs['preprocessed'] = True
    output_root.attrs['state_features'] = feature_names
    output_root.attrs['num_state_features'] = len(feature_names)
    
    # Compute and print statistics
    all_states = np.concatenate(all_processed, axis=0)
    
    console.print(f"\n[bold green]✓ Done![/bold green]")
    console.print(f"  Output: {output_path}")
    console.print(f"  Features: {len(feature_names)}")
    
    # Print stats table
    table = Table(title="Feature Statistics")
    table.add_column("Feature", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    
    for i, name in enumerate(feature_names):
        col = all_states[:, i]
        table.add_row(
            name,
            f"{col.mean():.4f}",
            f"{col.std():.4f}",
            f"{col.min():.4f}",
            f"{col.max():.4f}",
        )
    
    console.print(table)
    
    # Store stats in zarr
    output_root.attrs['state_statistics'] = {
        'mean': all_states.mean(axis=0).tolist(),
        'std': all_states.std(axis=0).tolist(),
        'min': all_states.min(axis=0).tolist(),
        'max': all_states.max(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess gameplay dataset')
    parser.add_argument('input_zarr', type=str, help='Input zarr dataset')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='Output zarr path (default: input_preprocessed.zarr)')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        input_path = Path(args.input_zarr)
        args.output = str(input_path.parent / (input_path.stem + '_preprocessed.zarr'))
    
    preprocess_dataset(args.input_zarr, args.output)


if __name__ == '__main__':
    main()
