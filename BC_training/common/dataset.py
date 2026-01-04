"""Dataset loader for zarr gameplay data."""

import zarr
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from .state_preprocessing import StatePreprocessor, create_preprocessor

logger = logging.getLogger(__name__)


class ZarrGameplayDataset:
    """Dataset for loading gameplay recordings from zarr format.
    
    Loads episodes containing:
    - frames: [N, C, H, W] uint8 arrays
    - state: [N, num_attributes] float32 arrays (not used in pure CNN)
    - actions: [N, num_actions] bool arrays
    """
    
    def __init__(
        self,
        dataset_path: str,
        episode_indices: Optional[List[int]] = None,
        use_state: bool = False,
        normalize_frames: bool = True,
        validate_episodes: bool = True,
        state_preprocessor: Optional[StatePreprocessor] = None,
    ):
        """Initialize dataset.
        
        Args:
            dataset_path: Path to zarr dataset
            episode_indices: List of episode indices to load (None = all)
            use_state: Whether to include state features (not used in pure CNN)
            normalize_frames: Whether to normalize frames to [0, 1]
            validate_episodes: Whether to filter out episodes with mismatched shapes
            state_preprocessor: Optional preprocessor for state features
        """
        self.dataset_path = Path(dataset_path)
        self.use_state = use_state
        self.normalize_frames = normalize_frames
        self.state_preprocessor = state_preprocessor if state_preprocessor else create_preprocessor()
        
        # Open zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')
        
        # Get episode list
        all_episodes = sorted([k for k in self.zarr_root.keys() if k.startswith('episode_')])
        
        if episode_indices is not None:
            self.episodes = [f'episode_{i}' for i in episode_indices if f'episode_{i}' in all_episodes]
        else:
            self.episodes = all_episodes
        
        logger.info(f"Loaded dataset from {dataset_path}")
        logger.info(f"Found {len(self.episodes)} episodes")
        
        # Get metadata
        self.action_keys = self.zarr_root.attrs.get('keys', [])
        self.state_attrs = self.zarr_root.attrs.get('attributes', [])
        self.num_actions = len(self.action_keys)
        
        # Validate and filter episodes if requested
        if validate_episodes:
            self._validate_and_filter_episodes()
        
        logger.info(f"Using {len(self.episodes)} valid episodes")
        
        # Build episode index mapping
        self._build_episode_index()
    
    def _validate_and_filter_episodes(self):
        """Validate episodes and filter out those with mismatched dimensions."""
        valid_episodes = []
        skipped_episodes = []
        
        # Get expected shapes from metadata
        expected_num_actions = self.num_actions
        expected_num_state = len(self.state_attrs)
        
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            
            # Check action dimensions
            actions_shape = ep['actions'].shape
            if actions_shape[1] != expected_num_actions:
                skipped_episodes.append((ep_name, f"actions={actions_shape[1]} (expected {expected_num_actions})"))
                continue
            
            # Check state dimensions
            state_shape = ep['state'].shape
            if state_shape[1] != expected_num_state:
                skipped_episodes.append((ep_name, f"state={state_shape[1]} (expected {expected_num_state})"))
                continue
            
            # Check frame-action-state alignment
            frames_len = ep['frames'].shape[0]
            if actions_shape[0] != frames_len or state_shape[0] != frames_len:
                skipped_episodes.append((ep_name, f"length mismatch: frames={frames_len}, actions={actions_shape[0]}, state={state_shape[0]}"))
                continue
            
            valid_episodes.append(ep_name)
        
        if skipped_episodes:
            logger.warning(f"Skipped {len(skipped_episodes)} episodes with invalid dimensions:")
            for ep_name, reason in skipped_episodes[:10]:  # Show first 10
                logger.warning(f"  - {ep_name}: {reason}")
            if len(skipped_episodes) > 10:
                logger.warning(f"  ... and {len(skipped_episodes) - 10} more")
        
        self.episodes = valid_episodes
        
        if len(valid_episodes) == 0:
            raise ValueError("No valid episodes found in dataset! All episodes have mismatched dimensions.")
        
    def _build_episode_index(self):
        """Build index of (episode_id, frame_idx) tuples for fast access."""
        self.index = []
        self.episode_lengths = []
        
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            num_frames = ep['frames'].shape[0]
            self.episode_lengths.append(num_frames)
            
            for frame_idx in range(num_frames):
                self.index.append((ep_name, frame_idx))
        
        logger.info(f"Total frames: {len(self.index)}")
        logger.info(f"Episode lengths - min: {min(self.episode_lengths)}, "
                   f"max: {max(self.episode_lengths)}, "
                   f"mean: {np.mean(self.episode_lengths):.1f}")
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single frame and its corresponding action.
        
        Args:
            idx: Frame index
            
        Returns:
            Dict with 'frames' and 'actions' (and optionally 'state')
        """
        ep_name, frame_idx = self.index[idx]
        ep = self.zarr_root[ep_name]
        
        # Load frame [C, H, W]
        frame = np.array(ep['frames'][frame_idx])
        
        # Normalize to [0, 1]
        if self.normalize_frames:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.float32)
        
        # Load actions [num_actions]
        actions = np.array(ep['actions'][frame_idx], dtype=np.float32)
        
        result = {
            'frames': frame,
            'actions': actions,
        }
        
        # Optionally load and preprocess state
        if self.use_state:
            state = np.array(ep['state'][frame_idx], dtype=np.float32)
            processed = self.state_preprocessor(state)
            result['state'] = processed['continuous']
            result['hero_anim_idx'] = processed['hero_anim_idx']
            result['npc_anim_idx'] = processed['npc_anim_idx']
        
        return result
    
    def get_episode(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """Get entire episode data.
        
        Args:
            episode_idx: Index in self.episodes list
            
        Returns:
            Dict with full episode data
        """
        ep_name = self.episodes[episode_idx]
        ep = self.zarr_root[ep_name]
        
        frames = np.array(ep['frames'][:])
        if self.normalize_frames:
            frames = frames.astype(np.float32) / 255.0
        else:
            frames = frames.astype(np.float32)
        
        actions = np.array(ep['actions'][:], dtype=np.float32)
        
        result = {
            'frames': frames,
            'actions': actions,
            'episode_name': ep_name,
            'fps': ep.attrs.get('fps', None),
        }
        
        if self.use_state:
            state = np.array(ep['state'][:], dtype=np.float32)
            processed = self.state_preprocessor(state)
            result['state'] = processed['continuous']
            result['hero_anim_idx'] = processed['hero_anim_idx']
            result['npc_anim_idx'] = processed['npc_anim_idx']
        
        return result
    
    def compute_action_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced actions.
        
        Returns:
            Array of shape [num_actions] with weights for each action
        """
        action_counts = np.zeros(self.num_actions)
        
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:])
            action_counts += actions.sum(axis=0)
        
        # Inverse frequency weighting
        total_frames = len(self.index)
        weights = total_frames / (self.num_actions * action_counts + 1e-8)
        
        # Normalize so mean weight is 1
        weights = weights / weights.mean()
        
        logger.info(f"Action counts: {action_counts}")
        logger.info(f"Action weights: {weights}")
        
        return weights


def create_data_loader(
    dataset: ZarrGameplayDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Create a simple numpy-based data loader.
    
    Args:
        dataset: ZarrGameplayDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Yields:
        Batches of data as numpy arrays
    """
    indices = np.arange(len(dataset))
    
    if shuffle:
        np.random.shuffle(indices)
    
    num_batches = len(dataset) // batch_size
    if not drop_last and len(dataset) % batch_size != 0:
        num_batches += 1
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_indices = indices[start_idx:end_idx]
        
        # Collect batch
        batch_data = [dataset[int(idx)] for idx in batch_indices]
        
        # Stack into arrays
        batch = {
            'frames': np.stack([d['frames'] for d in batch_data]),
            'actions': np.stack([d['actions'] for d in batch_data]),
        }
        
        if dataset.use_state:
            batch['state'] = np.stack([d['state'] for d in batch_data])
            batch['hero_anim_idx'] = np.stack([d['hero_anim_idx'] for d in batch_data])
            batch['npc_anim_idx'] = np.stack([d['npc_anim_idx'] for d in batch_data])
        
        yield batch


def split_episodes(
    total_episodes: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split episode indices into train/val sets.
    
    Args:
        total_episodes: Total number of episodes
        train_ratio: Ratio of episodes for training
        val_ratio: Ratio of episodes for validation
        seed: Random seed
        
    Returns:
        (train_indices, val_indices)
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    rng = np.random.RandomState(seed)
    indices = np.arange(total_episodes)
    rng.shuffle(indices)
    
    train_size = int(total_episodes * train_ratio)
    
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    
    logger.info(f"Split {total_episodes} episodes: {len(train_indices)} train, {len(val_indices)} val")
    
    return train_indices, val_indices


class TemporalGameplayDataset:
    """Dataset for loading gameplay recordings with temporal context.
    
    Returns sequences of frames and action history for temporal modeling.
    """
    
    def __init__(
        self,
        dataset_path: str,
        episode_indices: Optional[List[int]] = None,
        use_state: bool = False,
        normalize_frames: bool = True,
        validate_episodes: bool = True,
        state_preprocessor: Optional[StatePreprocessor] = None,
        num_history_frames: int = 4,
        num_action_history: int = 4,
        frame_skip: int = 1,
    ):
        """Initialize temporal dataset.
        
        Args:
            dataset_path: Path to zarr dataset
            episode_indices: List of episode indices to load (None = all)
            use_state: Whether to include state features
            normalize_frames: Whether to normalize frames to [0, 1]
            validate_episodes: Whether to filter out episodes with mismatched shapes
            state_preprocessor: Optional preprocessor for state features
            num_history_frames: Number of past frames to include (total = num_history_frames + 1)
            num_action_history: Number of past actions to include
            frame_skip: Skip interval between history frames (1 = every frame)
        """
        self.dataset_path = Path(dataset_path)
        self.use_state = use_state
        self.normalize_frames = normalize_frames
        self.state_preprocessor = state_preprocessor if state_preprocessor else create_preprocessor()
        self.num_history_frames = num_history_frames
        self.num_action_history = num_action_history
        self.frame_skip = frame_skip
        
        # Total lookback needed
        self.lookback = max(
            num_history_frames * frame_skip,
            num_action_history
        )
        
        # Open zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')
        
        # Get episode list
        all_episodes = sorted([k for k in self.zarr_root.keys() if k.startswith('episode_')])
        
        if episode_indices is not None:
            self.episodes = [f'episode_{i}' for i in episode_indices if f'episode_{i}' in all_episodes]
        else:
            self.episodes = all_episodes
        
        logger.info(f"Loaded temporal dataset from {dataset_path}")
        logger.info(f"Temporal config: {num_history_frames} history frames, {num_action_history} action history, skip={frame_skip}")
        
        # Get metadata
        self.action_keys = self.zarr_root.attrs.get('keys', [])
        self.state_attrs = self.zarr_root.attrs.get('attributes', [])
        self.num_actions = len(self.action_keys)
        
        # Validate and filter episodes if requested
        if validate_episodes:
            self._validate_and_filter_episodes()
        
        logger.info(f"Using {len(self.episodes)} valid episodes")
        
        # Build episode index mapping (skipping early frames that lack history)
        self._build_episode_index()
    
    def _validate_and_filter_episodes(self):
        """Validate episodes and filter out those with mismatched dimensions."""
        valid_episodes = []
        skipped_episodes = []
        
        expected_num_actions = self.num_actions
        expected_num_state = len(self.state_attrs)
        
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            
            # Check action dimensions
            actions_shape = ep['actions'].shape
            if actions_shape[1] != expected_num_actions:
                skipped_episodes.append((ep_name, f"actions={actions_shape[1]} (expected {expected_num_actions})"))
                continue
            
            # Check state dimensions
            state_shape = ep['state'].shape
            if state_shape[1] != expected_num_state:
                skipped_episodes.append((ep_name, f"state={state_shape[1]} (expected {expected_num_state})"))
                continue
            
            # Check minimum length for temporal context
            frames_len = ep['frames'].shape[0]
            if frames_len <= self.lookback:
                skipped_episodes.append((ep_name, f"too short: {frames_len} frames (need >{self.lookback})"))
                continue
            
            valid_episodes.append(ep_name)
        
        if skipped_episodes:
            logger.warning(f"Skipped {len(skipped_episodes)} episodes:")
            for ep_name, reason in skipped_episodes[:10]:
                logger.warning(f"  - {ep_name}: {reason}")
            if len(skipped_episodes) > 10:
                logger.warning(f"  ... and {len(skipped_episodes) - 10} more")
        
        self.episodes = valid_episodes
        
        if len(valid_episodes) == 0:
            raise ValueError("No valid episodes found!")
    
    def _build_episode_index(self):
        """Build index of (episode_id, frame_idx) tuples, skipping early frames."""
        self.index = []
        self.episode_lengths = []
        
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            num_frames = ep['frames'].shape[0]
            self.episode_lengths.append(num_frames)
            
            # Start from lookback to ensure we have enough history
            for frame_idx in range(self.lookback, num_frames):
                self.index.append((ep_name, frame_idx))
        
        logger.info(f"Total valid frames: {len(self.index)} (skipped first {self.lookback} per episode)")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def compute_action_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced actions.
        
        Returns:
            Array of shape [num_actions] with weights for each action
        """
        action_counts = np.zeros(self.num_actions)
        
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:])
            action_counts += actions.sum(axis=0)
        
        # Inverse frequency weighting
        total_frames = len(self.index)
        weights = total_frames / (self.num_actions * action_counts + 1e-8)
        
        # Normalize so mean weight is 1
        weights = weights / weights.mean()
        
        logger.info(f"Action counts: {action_counts}")
        logger.info(f"Action weights: {weights}")
        
        return weights
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a frame sequence with action history.
        
        Returns:
            Dict with:
                - 'frames': [num_history_frames+1, C, H, W] stacked frames
                - 'actions': [num_actions] target action at current time
                - 'action_history': [num_action_history, num_actions] past actions
                - Optional state features if use_state=True
        """
        ep_name, frame_idx = self.index[idx]
        ep = self.zarr_root[ep_name]
        
        # === Frame stack ===
        # Get indices for history frames (from oldest to current)
        frame_indices = [
            frame_idx - i * self.frame_skip 
            for i in range(self.num_history_frames, -1, -1)
        ]  # [t-k*skip, t-(k-1)*skip, ..., t]
        
        frames = []
        for fi in frame_indices:
            frame = np.array(ep['frames'][fi])
            if self.normalize_frames:
                frame = frame.astype(np.float32) / 255.0
            else:
                frame = frame.astype(np.float32)
            frames.append(frame)
        
        # Stack: [num_history_frames+1, C, H, W]
        frames = np.stack(frames, axis=0)
        
        # === Action history (past actions, not including current) ===
        if self.num_action_history > 0:
            action_indices = [frame_idx - i - 1 for i in range(self.num_action_history - 1, -1, -1)]
            # [t-k, t-k+1, ..., t-1]
            
            action_history = []
            for ai in action_indices:
                action = np.array(ep['actions'][ai], dtype=np.float32)
                action_history.append(action)
            
            # Stack: [num_action_history, num_actions]
            action_history = np.stack(action_history, axis=0)
        else:
            # No action history - create empty array with correct shape
            num_actions = len(ep['actions'][frame_idx])
            action_history = np.zeros((0, num_actions), dtype=np.float32)
        
        # === Current action (target) ===
        actions = np.array(ep['actions'][frame_idx], dtype=np.float32)
        
        result = {
            'frames': frames,  # [T, C, H, W]
            'actions': actions,  # [num_actions]
            'action_history': action_history,  # [K, num_actions]
        }
        
        # === Optionally load state ===
        if self.use_state:
            state = np.array(ep['state'][frame_idx], dtype=np.float32)
            processed = self.state_preprocessor(state)
            result['state'] = processed['continuous']
            result['hero_anim_idx'] = processed['hero_anim_idx']
            result['npc_anim_idx'] = processed['npc_anim_idx']
        
        return result


def create_temporal_data_loader(
    dataset: TemporalGameplayDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Create data loader for temporal dataset.
    
    Yields:
        Batches with stacked temporal data
    """
    indices = np.arange(len(dataset))
    
    if shuffle:
        np.random.shuffle(indices)
    
    num_batches = len(dataset) // batch_size
    if not drop_last and len(dataset) % batch_size != 0:
        num_batches += 1
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_indices = indices[start_idx:end_idx]
        
        batch_data = [dataset[int(idx)] for idx in batch_indices]
        
        batch = {
            'frames': np.stack([d['frames'] for d in batch_data]),  # [B, T, C, H, W]
            'actions': np.stack([d['actions'] for d in batch_data]),  # [B, num_actions]
            'action_history': np.stack([d['action_history'] for d in batch_data]),  # [B, K, num_actions]
        }
        
        if dataset.use_state:
            batch['state'] = np.stack([d['state'] for d in batch_data])
            batch['hero_anim_idx'] = np.stack([d['hero_anim_idx'] for d in batch_data])
            batch['npc_anim_idx'] = np.stack([d['npc_anim_idx'] for d in batch_data])
        
        yield batch

