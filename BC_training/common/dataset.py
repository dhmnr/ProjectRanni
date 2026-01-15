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
        oversample_actions: Optional[List[int]] = None,
        oversample_ratio: float = 1.0,
        stack_states: bool = False,
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
            oversample_actions: List of action indices to oversample (e.g., [4, 8] for dodge/attack)
            oversample_ratio: How much to oversample (e.g., 2.0 = double rare samples,
                              or target ratio to match majority class frequency)
            stack_states: Whether to stack states temporally (same as frames) instead of single state
        """
        self.dataset_path = Path(dataset_path)
        self.use_state = use_state
        self.normalize_frames = normalize_frames
        self.state_preprocessor = state_preprocessor if state_preprocessor else create_preprocessor()
        self.num_history_frames = num_history_frames
        self.num_action_history = num_action_history
        self.frame_skip = frame_skip
        self.oversample_actions = oversample_actions
        self.oversample_ratio = oversample_ratio
        self.stack_states = stack_states

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

        # Apply oversampling if requested
        if self.oversample_actions and self.oversample_ratio > 1.0:
            self._apply_oversampling()
    
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

    def _apply_oversampling(self):
        """Oversample sequences containing rare actions to balance class distribution.

        Duplicates entire sequences (frame stack + action history intact) for samples
        where the target action contains any of the specified actions to oversample.
        """
        logger.info(f"Applying oversampling for actions {self.oversample_actions} with ratio {self.oversample_ratio}")

        # Cache action data per episode for efficiency
        episode_actions = {}
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            episode_actions[ep_name] = np.array(ep['actions'][:])

        # Find indices where target action contains any oversample action
        oversample_indices = []
        for idx, (ep_name, frame_idx) in enumerate(self.index):
            target_action = episode_actions[ep_name][frame_idx]
            # Check if any of the oversample actions are active
            if any(target_action[action_idx] > 0.5 for action_idx in self.oversample_actions):
                oversample_indices.append(idx)

        num_oversample = len(oversample_indices)
        num_total = len(self.index)
        logger.info(f"Found {num_oversample} samples ({100*num_oversample/num_total:.1f}%) with target actions")

        if num_oversample == 0:
            logger.warning("No samples found with target actions to oversample!")
            return

        # Calculate how many times to duplicate
        # If oversample_ratio is e.g. 5.0, we want these samples to appear 5x more often
        num_duplicates = int(self.oversample_ratio - 1)  # -1 because original is already there

        if num_duplicates > 0:
            # Duplicate the indices
            duplicated_indices = []
            for _ in range(num_duplicates):
                for idx in oversample_indices:
                    duplicated_indices.append(self.index[idx])

            original_len = len(self.index)
            self.index.extend(duplicated_indices)
            new_len = len(self.index)

            logger.info(f"Oversampling: {original_len} -> {new_len} samples (+{new_len - original_len})")
            logger.info(f"Rare action samples now appear {self.oversample_ratio:.1f}x ({num_oversample} -> {num_oversample * int(self.oversample_ratio)})")

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
            if self.stack_states:
                # Stack states temporally (same indices as frames)
                states = []
                hero_anim_ids = []
                npc_anim_ids = []
                for fi in frame_indices:
                    state = np.array(ep['state'][fi], dtype=np.float32)
                    processed = self.state_preprocessor(state)
                    states.append(processed['continuous'])
                    hero_anim_ids.append(processed['hero_anim_idx'])
                    npc_anim_ids.append(processed['npc_anim_idx'])

                result['state'] = np.stack(states, axis=0)  # [T, num_features]
                result['hero_anim_idx'] = np.array(hero_anim_ids, dtype=np.int32)  # [T]
                result['npc_anim_idx'] = np.array(npc_anim_ids, dtype=np.int32)  # [T]
            else:
                # Single state (current frame only)
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


class ActionChunkingDataset:
    """Dataset for action chunking transformer.

    Returns sequences of frames and future action chunks for training
    action-chunking behavior cloning models.

    Input: frames[t-k:t] -> Output: actions[t:t+chunk_size]
    Optionally includes state features (continuous + animation indices).
    """

    def __init__(
        self,
        dataset_path: str,
        episode_indices: Optional[List[int]] = None,
        normalize_frames: bool = True,
        validate_episodes: bool = True,
        num_frames: int = 4,            # Number of observation frames
        chunk_size: int = 16,           # Number of future actions to predict
        use_state: bool = False,        # Whether to include state features
        state_preprocessor: Optional[StatePreprocessor] = None,
    ):
        """Initialize action chunking dataset.

        Args:
            dataset_path: Path to zarr dataset
            episode_indices: List of episode indices to load (None = all)
            normalize_frames: Whether to normalize frames to [0, 1]
            validate_episodes: Whether to filter out episodes with mismatched shapes
            num_frames: Number of observation frames (temporal context)
            chunk_size: Number of future actions to predict
            use_state: Whether to include state features
            state_preprocessor: Optional preprocessor for state features
        """
        self.dataset_path = Path(dataset_path)
        self.normalize_frames = normalize_frames
        self.num_frames = num_frames
        self.chunk_size = chunk_size
        self.use_state = use_state
        self.state_preprocessor = state_preprocessor if state_preprocessor else create_preprocessor()

        # Total lookback needed for frames (we need num_frames-1 past frames + current)
        self.frame_lookback = num_frames - 1

        # Open zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')

        # Get episode list
        all_episodes = sorted([k for k in self.zarr_root.keys() if k.startswith('episode_')])

        if episode_indices is not None:
            self.episodes = [f'episode_{i}' for i in episode_indices if f'episode_{i}' in all_episodes]
        else:
            self.episodes = all_episodes

        logger.info(f"Loaded action chunking dataset from {dataset_path}")
        logger.info(f"Config: {num_frames} frames -> {chunk_size} action chunk, use_state={use_state}")

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
        """Validate episodes and filter out those with issues."""
        valid_episodes = []
        skipped_episodes = []

        expected_num_actions = self.num_actions
        min_length = self.frame_lookback + self.chunk_size + 1

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]

            # Check action dimensions
            actions_shape = ep['actions'].shape
            if actions_shape[1] != expected_num_actions:
                skipped_episodes.append((ep_name, f"actions={actions_shape[1]} (expected {expected_num_actions})"))
                continue

            # Check minimum length for context + chunk
            frames_len = ep['frames'].shape[0]
            if frames_len < min_length:
                skipped_episodes.append((ep_name, f"too short: {frames_len} frames (need >={min_length})"))
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
        """Build index of (episode_id, frame_idx) tuples.

        We need:
        - frame_lookback frames before current frame (for observation)
        - chunk_size frames after current frame (for action targets)
        """
        self.index = []
        self.episode_lengths = []

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            num_frames = ep['frames'].shape[0]
            self.episode_lengths.append(num_frames)

            # Valid range: [frame_lookback, num_frames - chunk_size)
            # At position t, we use frames[t-lookback:t+1] and actions[t:t+chunk_size]
            for frame_idx in range(self.frame_lookback, num_frames - self.chunk_size):
                self.index.append((ep_name, frame_idx))

        logger.info(f"Total valid samples: {len(self.index)}")
        logger.info(f"(Skipped first {self.frame_lookback} and last {self.chunk_size} frames per episode)")

    def __len__(self) -> int:
        return len(self.index)

    def compute_action_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced actions."""
        action_counts = np.zeros(self.num_actions)

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:])
            action_counts += actions.sum(axis=0)

        total_frames = sum(self.episode_lengths)
        weights = total_frames / (self.num_actions * action_counts + 1e-8)
        weights = weights / weights.mean()

        logger.info(f"Action counts: {action_counts}")
        logger.info(f"Action weights: {weights}")

        return weights

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get observation frames and future action chunk.

        Returns:
            Dict with:
                - 'frames': [num_frames, C, H, W] observation frames
                - 'actions': [chunk_size, num_actions] future actions
                - Optional state features if use_state=True:
                    - 'state': [num_continuous_features] continuous state
                    - 'hero_anim_idx': int32 hero animation index
                    - 'npc_anim_idx': int32 NPC animation index
        """
        ep_name, frame_idx = self.index[idx]
        ep = self.zarr_root[ep_name]

        # === Observation frames ===
        # Get num_frames frames ending at frame_idx (inclusive)
        # [frame_idx - (num_frames-1), ..., frame_idx]
        frame_indices = list(range(frame_idx - self.frame_lookback, frame_idx + 1))

        frames = []
        for fi in frame_indices:
            frame = np.array(ep['frames'][fi])
            if self.normalize_frames:
                frame = frame.astype(np.float32) / 255.0
            else:
                frame = frame.astype(np.float32)
            frames.append(frame)

        # Stack: [num_frames, C, H, W]
        frames = np.stack(frames, axis=0)

        # === Future action chunk ===
        # Get chunk_size actions starting from frame_idx
        # [frame_idx, frame_idx+1, ..., frame_idx+chunk_size-1]
        action_indices = list(range(frame_idx, frame_idx + self.chunk_size))

        actions = []
        for ai in action_indices:
            action = np.array(ep['actions'][ai], dtype=np.float32)
            actions.append(action)

        # Stack: [chunk_size, num_actions]
        actions = np.stack(actions, axis=0)

        result = {
            'frames': frames,   # [num_frames, C, H, W]
            'actions': actions,  # [chunk_size, num_actions]
        }

        # === Optionally load state (from last/current frame) ===
        if self.use_state:
            state = np.array(ep['state'][frame_idx], dtype=np.float32)
            processed = self.state_preprocessor(state)
            result['state'] = processed['continuous']
            result['hero_anim_idx'] = processed['hero_anim_idx']
            result['npc_anim_idx'] = processed['npc_anim_idx']

        return result


def create_action_chunking_data_loader(
    dataset: ActionChunkingDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Create data loader for action chunking dataset.

    Yields:
        Batches with:
            - 'frames': [B, num_frames, C, H, W]
            - 'actions': [B, chunk_size, num_actions]
            - Optional if use_state:
                - 'state': [B, num_continuous_features]
                - 'hero_anim_idx': [B] int32
                - 'npc_anim_idx': [B] int32
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
            'frames': np.stack([d['frames'] for d in batch_data]),   # [B, T, C, H, W]
            'actions': np.stack([d['actions'] for d in batch_data]),  # [B, chunk_size, num_actions]
        }

        # Add state features if available
        if dataset.use_state:
            batch['state'] = np.stack([d['state'] for d in batch_data])
            batch['hero_anim_idx'] = np.stack([d['hero_anim_idx'] for d in batch_data])
            batch['npc_anim_idx'] = np.stack([d['npc_anim_idx'] for d in batch_data])

        yield batch


class OnsetOffsetDataset:
    """Dataset for onset/offset action prediction.

    Instead of predicting raw actions, predicts:
    - onset: when action transitions from 0 → 1 (action starts)
    - offset: when action transitions from 1 → 0 (action ends)

    Input: frames[t-k:t] -> Output: onset_offset[t:t+chunk_size] with shape [chunk_size, num_actions, 2]
    """

    def __init__(
        self,
        dataset_path: str,
        episode_indices: Optional[List[int]] = None,
        normalize_frames: bool = True,
        validate_episodes: bool = True,
        num_frames: int = 8,            # Number of observation frames
        chunk_size: int = 8,            # Number of future actions to predict
    ):
        """Initialize onset/offset dataset.

        Args:
            dataset_path: Path to zarr dataset
            episode_indices: List of episode indices to load (None = all)
            normalize_frames: Whether to normalize frames to [0, 1]
            validate_episodes: Whether to filter out episodes with mismatched shapes
            num_frames: Number of observation frames (temporal context)
            chunk_size: Number of future onset/offset predictions
        """
        self.dataset_path = Path(dataset_path)
        self.normalize_frames = normalize_frames
        self.num_frames = num_frames
        self.chunk_size = chunk_size

        # Total lookback needed for frames
        self.frame_lookback = num_frames - 1

        # Open zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')

        # Get episode list
        all_episodes = sorted([k for k in self.zarr_root.keys() if k.startswith('episode_')])

        if episode_indices is not None:
            self.episodes = [f'episode_{i}' for i in episode_indices if f'episode_{i}' in all_episodes]
        else:
            self.episodes = all_episodes

        logger.info(f"Loaded onset/offset dataset from {dataset_path}")
        logger.info(f"Config: {num_frames} frames -> {chunk_size} onset/offset predictions")

        # Get metadata
        self.action_keys = self.zarr_root.attrs.get('keys', [])
        self.num_actions = len(self.action_keys)

        # Validate and filter episodes if requested
        if validate_episodes:
            self._validate_and_filter_episodes()

        logger.info(f"Using {len(self.episodes)} valid episodes")

        # Build episode index mapping
        self._build_episode_index()

    def _validate_and_filter_episodes(self):
        """Validate episodes and filter out those with issues."""
        valid_episodes = []
        skipped_episodes = []

        expected_num_actions = self.num_actions
        # Need extra frame at the start for previous action detection
        min_length = self.frame_lookback + self.chunk_size + 2

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]

            # Check action dimensions
            actions_shape = ep['actions'].shape
            if actions_shape[1] != expected_num_actions:
                skipped_episodes.append((ep_name, f"actions={actions_shape[1]} (expected {expected_num_actions})"))
                continue

            # Check minimum length for context + chunk + previous action
            frames_len = ep['frames'].shape[0]
            if frames_len < min_length:
                skipped_episodes.append((ep_name, f"too short: {frames_len} frames (need >={min_length})"))
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
        """Build index of (episode_id, frame_idx) tuples."""
        self.index = []
        self.episode_lengths = []

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            num_frames = ep['frames'].shape[0]
            self.episode_lengths.append(num_frames)

            # Valid range: start at frame_lookback+1 to have previous action available
            # At position t, we use:
            #   - frames[t-lookback:t+1] for observation
            #   - actions[t-1] for previous state (onset/offset detection)
            #   - actions[t:t+chunk_size] for targets
            for frame_idx in range(self.frame_lookback + 1, num_frames - self.chunk_size):
                self.index.append((ep_name, frame_idx))

        logger.info(f"Total valid samples: {len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)

    def compute_onset_offset_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute class weights for onset and offset events separately.

        Returns:
            Tuple of (onset_weights, offset_weights), each [num_actions]
        """
        onset_counts = np.zeros(self.num_actions)
        offset_counts = np.zeros(self.num_actions)
        total_transitions = 0

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:], dtype=np.float32)

            # Compute onsets and offsets for entire episode
            prev_actions = np.concatenate([np.zeros((1, self.num_actions)), actions[:-1]], axis=0)
            onsets = (actions > prev_actions).astype(np.float32)  # 0->1
            offsets = (actions < prev_actions).astype(np.float32)  # 1->0

            onset_counts += onsets.sum(axis=0)
            offset_counts += offsets.sum(axis=0)
            total_transitions += len(actions)

        # Inverse frequency weighting
        onset_weights = total_transitions / (self.num_actions * onset_counts + 1e-8)
        offset_weights = total_transitions / (self.num_actions * offset_counts + 1e-8)

        # Normalize
        onset_weights = onset_weights / onset_weights.mean()
        offset_weights = offset_weights / offset_weights.mean()

        logger.info(f"Onset counts: {onset_counts}")
        logger.info(f"Onset weights: {onset_weights}")
        logger.info(f"Offset counts: {offset_counts}")
        logger.info(f"Offset weights: {offset_weights}")

        return onset_weights, offset_weights

    def compute_action_weights(self) -> np.ndarray:
        """Compute combined weights (average of onset + offset weights)."""
        onset_w, offset_w = self.compute_onset_offset_weights()
        return (onset_w + offset_w) / 2

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get observation frames and future onset/offset labels.

        Returns:
            Dict with:
                - 'frames': [num_frames, C, H, W] observation frames
                - 'onset_offset': [chunk_size, num_actions, 2] onset/offset labels
                - 'actions': [chunk_size, num_actions] raw actions (for reference)
        """
        ep_name, frame_idx = self.index[idx]
        ep = self.zarr_root[ep_name]

        # === Observation frames ===
        frame_indices = list(range(frame_idx - self.frame_lookback, frame_idx + 1))

        frames = []
        for fi in frame_indices:
            frame = np.array(ep['frames'][fi])
            if self.normalize_frames:
                frame = frame.astype(np.float32) / 255.0
            else:
                frame = frame.astype(np.float32)
            frames.append(frame)

        frames = np.stack(frames, axis=0)  # [num_frames, C, H, W]

        # === Onset/Offset labels ===
        # Get chunk_size+1 actions: [frame_idx-1, frame_idx, ..., frame_idx+chunk_size-1]
        # We need the previous action to detect onset/offset at the first position
        action_indices = list(range(frame_idx - 1, frame_idx + self.chunk_size))

        all_actions = []
        for ai in action_indices:
            action = np.array(ep['actions'][ai], dtype=np.float32)
            all_actions.append(action)

        all_actions = np.stack(all_actions, axis=0)  # [chunk_size+1, num_actions]

        # Current actions (targets): [chunk_size, num_actions]
        current_actions = all_actions[1:]  # [frame_idx, ..., frame_idx+chunk_size-1]
        # Previous actions: [chunk_size, num_actions]
        prev_actions = all_actions[:-1]  # [frame_idx-1, ..., frame_idx+chunk_size-2]

        # Compute onset: action was 0, now is 1
        onset = ((current_actions > 0) & (prev_actions == 0)).astype(np.float32)
        # Compute offset: action was 1, now is 0
        offset = ((current_actions == 0) & (prev_actions > 0)).astype(np.float32)

        # Stack: [chunk_size, num_actions, 2] where [..., 0] is onset, [..., 1] is offset
        onset_offset = np.stack([onset, offset], axis=-1)

        return {
            'frames': frames,              # [num_frames, C, H, W]
            'onset_offset': onset_offset,  # [chunk_size, num_actions, 2]
            'actions': current_actions,    # [chunk_size, num_actions] for reference
        }


def create_onset_offset_data_loader(
    dataset: OnsetOffsetDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Create data loader for onset/offset dataset.

    Yields:
        Batches with:
            - 'frames': [B, num_frames, C, H, W]
            - 'onset_offset': [B, chunk_size, num_actions, 2]
            - 'actions': [B, chunk_size, num_actions]
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
            'frames': np.stack([d['frames'] for d in batch_data]),
            'onset_offset': np.stack([d['onset_offset'] for d in batch_data]),
            'actions': np.stack([d['actions'] for d in batch_data]),
        }

        yield batch


class ActionComboDataset:
    """Dataset for action combo prediction (STATE-ONLY).

    Instead of predicting 13 independent binary actions, maps action vectors
    to discrete combo IDs. Only ~32 unique combos exist in the dataset out of
    2^13 possible combinations.

    Input: state[t] -> Output: combo_ids[t:t+chunk_size]

    Uses state features only (continuous + animation indices) - no frames/vision.
    """

    def __init__(
        self,
        dataset_path: str,
        episode_indices: Optional[List[int]] = None,
        validate_episodes: bool = True,
        chunk_size: int = 8,            # Number of future combo IDs to predict
        state_preprocessor: Optional[StatePreprocessor] = None,
        combo_mapping: Optional[Dict] = None,  # Pre-computed combo mapping (for val set)
    ):
        """Initialize action combo dataset (state-only).

        Args:
            dataset_path: Path to zarr dataset
            episode_indices: List of episode indices to load (None = all)
            validate_episodes: Whether to filter out episodes with mismatched shapes
            chunk_size: Number of future combo IDs to predict
            state_preprocessor: Optional preprocessor for state features
            combo_mapping: Pre-computed combo mapping from training set (for validation)
        """
        self.dataset_path = Path(dataset_path)
        self.chunk_size = chunk_size
        self.state_preprocessor = state_preprocessor if state_preprocessor else create_preprocessor()

        # Open zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')

        # Get episode list
        all_episodes = sorted([k for k in self.zarr_root.keys() if k.startswith('episode_')])

        if episode_indices is not None:
            self.episodes = [f'episode_{i}' for i in episode_indices if f'episode_{i}' in all_episodes]
        else:
            self.episodes = all_episodes

        logger.info(f"Loaded action combo dataset (state-only) from {dataset_path}")
        logger.info(f"Config: state -> {chunk_size} combo chunk")

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

        # Set up combo mapping
        if combo_mapping is not None:
            # Use pre-computed mapping (for validation set)
            self.combo_to_id = combo_mapping['combo_to_id']
            self.id_to_combo = combo_mapping['id_to_combo']
            self.num_combos = combo_mapping['num_combos']
            logger.info(f"Using pre-computed combo mapping with {self.num_combos} combos")
        else:
            # Discover combos from this dataset (for training set)
            self._discover_action_combos()

    def _validate_and_filter_episodes(self):
        """Validate episodes and filter out those with issues."""
        valid_episodes = []
        skipped_episodes = []

        expected_num_actions = self.num_actions
        min_length = self.chunk_size + 1  # Need at least chunk_size actions

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]

            # Check action dimensions
            actions_shape = ep['actions'].shape
            if actions_shape[1] != expected_num_actions:
                skipped_episodes.append((ep_name, f"actions={actions_shape[1]} (expected {expected_num_actions})"))
                continue

            # Check minimum length for chunk
            state_len = ep['state'].shape[0]
            if state_len < min_length:
                skipped_episodes.append((ep_name, f"too short: {state_len} states (need >={min_length})"))
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
        """Build index of (episode_id, state_idx) tuples."""
        self.index = []
        self.episode_lengths = []

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            num_states = ep['state'].shape[0]
            self.episode_lengths.append(num_states)

            # Valid range: [0, num_states - chunk_size)
            for state_idx in range(0, num_states - self.chunk_size):
                self.index.append((ep_name, state_idx))

        logger.info(f"Total valid samples: {len(self.index)}")

    def _discover_action_combos(self):
        """Scan dataset to discover unique action combinations."""
        logger.info("Discovering unique action combinations...")

        unique_combos = set()

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:])  # [T, num_actions]

            # Convert each action vector to tuple for hashing
            for action in actions:
                combo = tuple(int(a) for a in action)
                unique_combos.add(combo)

        # Sort combos for deterministic ordering
        sorted_combos = sorted(unique_combos)

        # Create mappings
        self.combo_to_id = {combo: idx for idx, combo in enumerate(sorted_combos)}
        self.id_to_combo = sorted_combos
        self.num_combos = len(sorted_combos)

        logger.info(f"Found {self.num_combos} unique action combinations")

        # Log some stats
        combo_counts = np.zeros(self.num_combos)
        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:])
            for action in actions:
                combo = tuple(int(a) for a in action)
                combo_counts[self.combo_to_id[combo]] += 1

        # Show top 10 most common combos
        top_indices = np.argsort(combo_counts)[::-1][:10]
        logger.info("Top 10 most common action combos:")
        for i, idx in enumerate(top_indices):
            combo = self.id_to_combo[idx]
            count = int(combo_counts[idx])
            # Show which actions are active
            active_actions = [self.action_keys[j] for j, v in enumerate(combo) if v == 1]
            active_str = ', '.join(active_actions) if active_actions else '(none)'
            logger.info(f"  {i+1}. ID {idx}: {active_str} ({count} frames)")

    def get_combo_mapping(self) -> Dict:
        """Return combo mapping for use with validation set."""
        return {
            'combo_to_id': self.combo_to_id,
            'id_to_combo': self.id_to_combo,
            'num_combos': self.num_combos,
        }

    def _find_nearest_combo(self, action: np.ndarray) -> int:
        """Find nearest known combo using hamming distance."""
        action_tuple = tuple(int(a) for a in action)

        # Check if exact match exists
        if action_tuple in self.combo_to_id:
            return self.combo_to_id[action_tuple]

        # Find nearest by hamming distance
        min_dist = float('inf')
        nearest_id = 0

        for combo_id, combo in enumerate(self.id_to_combo):
            dist = sum(a != b for a, b in zip(action_tuple, combo))
            if dist < min_dist:
                min_dist = dist
                nearest_id = combo_id

        return nearest_id

    def __len__(self) -> int:
        return len(self.index)

    def compute_combo_weights(self) -> np.ndarray:
        """Compute class weights for imbalanced combos."""
        combo_counts = np.zeros(self.num_combos)

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:])
            for action in actions:
                combo_id = self._find_nearest_combo(action)
                combo_counts[combo_id] += 1

        total_frames = sum(self.episode_lengths)
        weights = total_frames / (self.num_combos * combo_counts + 1e-8)
        weights = weights / weights.mean()

        logger.info(f"Combo weights range: [{weights.min():.3f}, {weights.max():.3f}]")

        return weights.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get state and future combo IDs (state-only, no frames).

        Returns:
            Dict with:
                - 'combo_ids': [chunk_size] int32 combo IDs
                - 'state': [num_continuous_features] continuous state
                - 'hero_anim_idx': int32 hero animation index
                - 'npc_anim_idx': int32 NPC animation index
        """
        ep_name, state_idx = self.index[idx]
        ep = self.zarr_root[ep_name]

        # === Future combo IDs ===
        action_indices = list(range(state_idx, state_idx + self.chunk_size))

        combo_ids = []
        for ai in action_indices:
            action = np.array(ep['actions'][ai])
            combo_id = self._find_nearest_combo(action)
            combo_ids.append(combo_id)

        combo_ids = np.array(combo_ids, dtype=np.int32)  # [chunk_size]

        # === State features (from current state) ===
        state = np.array(ep['state'][state_idx], dtype=np.float32)
        processed = self.state_preprocessor(state)

        return {
            'combo_ids': combo_ids,                    # [chunk_size]
            'state': processed['continuous'],          # [num_continuous_features]
            'hero_anim_idx': processed['hero_anim_idx'],
            'npc_anim_idx': processed['npc_anim_idx'],
        }


def create_action_combo_data_loader(
    dataset: ActionComboDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Create data loader for action combo dataset (state-only).

    Yields:
        Batches with:
            - 'combo_ids': [B, chunk_size] int32
            - 'state': [B, num_continuous_features]
            - 'hero_anim_idx': [B] int32
            - 'npc_anim_idx': [B] int32
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
            'combo_ids': np.stack([d['combo_ids'] for d in batch_data]),
            'state': np.stack([d['state'] for d in batch_data]),
            'hero_anim_idx': np.stack([d['hero_anim_idx'] for d in batch_data]),
            'npc_anim_idx': np.stack([d['npc_anim_idx'] for d in batch_data]),
        }

        yield batch


class StateChunkingDataset:
    """Dataset for state-only action chunking (no frames/vision).

    Stacks multiple states temporally and predicts future action chunks.
    Similar to ActionChunkingDataset but uses only state features.

    Input: states[t-k:t] -> Output: actions[t:t+chunk_size]
    """

    def __init__(
        self,
        dataset_path: str,
        episode_indices: Optional[List[int]] = None,
        validate_episodes: bool = True,
        num_states: int = 8,            # Number of temporal state observations
        chunk_size: int = 8,            # Number of future actions to predict
        state_preprocessor: Optional[StatePreprocessor] = None,
    ):
        """Initialize state chunking dataset.

        Args:
            dataset_path: Path to zarr dataset
            episode_indices: List of episode indices to load (None = all)
            validate_episodes: Whether to filter out episodes with mismatched shapes
            num_states: Number of temporal state observations (context)
            chunk_size: Number of future actions to predict
            state_preprocessor: Preprocessor for state features
        """
        self.dataset_path = Path(dataset_path)
        self.num_states = num_states
        self.chunk_size = chunk_size
        self.state_preprocessor = state_preprocessor if state_preprocessor else create_preprocessor()

        # Lookback for temporal context
        self.state_lookback = num_states - 1

        # Open zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')

        # Get episode list
        all_episodes = sorted([k for k in self.zarr_root.keys() if k.startswith('episode_')])

        if episode_indices is not None:
            self.episodes = [f'episode_{i}' for i in episode_indices if f'episode_{i}' in all_episodes]
        else:
            self.episodes = all_episodes

        logger.info(f"Loaded state chunking dataset from {dataset_path}")
        logger.info(f"Config: {num_states} states -> {chunk_size} action chunk")

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
        """Validate episodes and filter out those with issues."""
        valid_episodes = []
        skipped_episodes = []

        expected_num_actions = self.num_actions
        min_length = self.state_lookback + self.chunk_size + 1

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]

            # Check action dimensions
            actions_shape = ep['actions'].shape
            if actions_shape[1] != expected_num_actions:
                skipped_episodes.append((ep_name, f"actions={actions_shape[1]} (expected {expected_num_actions})"))
                continue

            # Check minimum length
            state_len = ep['state'].shape[0]
            if state_len < min_length:
                skipped_episodes.append((ep_name, f"too short: {state_len} states (need >={min_length})"))
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
        """Build index of (episode_id, state_idx) tuples."""
        self.index = []
        self.episode_lengths = []

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            num_states = ep['state'].shape[0]
            self.episode_lengths.append(num_states)

            # Valid range: need lookback states before and chunk_size actions after
            for state_idx in range(self.state_lookback, num_states - self.chunk_size):
                self.index.append((ep_name, state_idx))

        logger.info(f"Total valid samples: {len(self.index)}")

    def __len__(self) -> int:
        return len(self.index)

    def compute_action_weights(self) -> np.ndarray:
        """Compute per-action class weights for imbalanced actions."""
        action_counts = np.zeros(self.num_actions)
        total_frames = 0

        for ep_name in self.episodes:
            ep = self.zarr_root[ep_name]
            actions = np.array(ep['actions'][:])
            action_counts += actions.sum(axis=0)
            total_frames += actions.shape[0]

        # Inverse frequency weighting
        action_freq = action_counts / total_frames
        weights = 1.0 / (action_freq + 1e-8)
        weights = weights / weights.mean()  # Normalize

        logger.info(f"Action weights range: [{weights.min():.3f}, {weights.max():.3f}]")

        return weights.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get temporal state sequence and future actions.

        Returns:
            Dict with:
                - 'states': [num_states, num_continuous_features] stacked states
                - 'hero_anim_ids': [num_states] hero animation indices
                - 'npc_anim_ids': [num_states] NPC animation indices
                - 'actions': [chunk_size, num_actions] future actions
        """
        ep_name, current_idx = self.index[idx]
        ep = self.zarr_root[ep_name]

        # === Temporal state sequence ===
        state_indices = list(range(current_idx - self.state_lookback, current_idx + 1))

        states = []
        hero_anim_ids = []
        npc_anim_ids = []

        for si in state_indices:
            state = np.array(ep['state'][si], dtype=np.float32)
            processed = self.state_preprocessor(state)
            states.append(processed['continuous'])
            hero_anim_ids.append(processed['hero_anim_idx'])
            npc_anim_ids.append(processed['npc_anim_idx'])

        states = np.stack(states, axis=0)  # [num_states, num_features]
        hero_anim_ids = np.array(hero_anim_ids, dtype=np.int32)  # [num_states]
        npc_anim_ids = np.array(npc_anim_ids, dtype=np.int32)  # [num_states]

        # === Future actions ===
        action_indices = list(range(current_idx, current_idx + self.chunk_size))
        actions = np.array(ep['actions'][action_indices[0]:action_indices[-1]+1], dtype=np.float32)

        return {
            'states': states,              # [num_states, num_features]
            'hero_anim_ids': hero_anim_ids,  # [num_states]
            'npc_anim_ids': npc_anim_ids,    # [num_states]
            'actions': actions,            # [chunk_size, num_actions]
        }


def create_state_chunking_data_loader(
    dataset: StateChunkingDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Create data loader for state chunking dataset.

    Yields:
        Batches with:
            - 'states': [B, num_states, num_features]
            - 'hero_anim_ids': [B, num_states]
            - 'npc_anim_ids': [B, num_states]
            - 'actions': [B, chunk_size, num_actions]
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
            'states': np.stack([d['states'] for d in batch_data]),
            'hero_anim_ids': np.stack([d['hero_anim_ids'] for d in batch_data]),
            'npc_anim_ids': np.stack([d['npc_anim_ids'] for d in batch_data]),
            'actions': np.stack([d['actions'] for d in batch_data]),
        }

        yield batch

