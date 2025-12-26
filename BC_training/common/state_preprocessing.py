"""State preprocessing for behavior cloning models.

Transforms raw game state into normalized/processed features.
Used by all models that consume state data.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# State attribute indices (from zarr dataset)
STATE_INDICES = {
    'HeroHp': 0,
    'HeroMaxHp': 1,
    'HeroSp': 2,
    'HeroMaxSp': 3,
    'HeroFp': 4,
    'HeroMaxFp': 5,
    'HeroGlobalPosX': 6,
    'HeroGlobalPosY': 7,
    'HeroGlobalPosZ': 8,
    'HeroAngle': 9,
    'HeroAnimId': 10,
    'NpcHp': 11,
    'NpcMaxHp': 12,
    'NpcId': 13,
    'NpcGlobalPosX': 14,
    'NpcGlobalPosY': 15,
    'NpcGlobalPosZ': 16,
    'NpcGlobalPosAngle': 17,
    'NpcAnimId': 18,
}


class StatePreprocessor:
    """Preprocesses raw game state into normalized features.
    
    Features:
    - Normalizes HP, SP, FP to [0, 1] using their max values (and drops max columns)
    - Extensible for additional preprocessing (distance, angles, etc.)
    
    Usage:
        preprocessor = StatePreprocessor()
        processed_state = preprocessor(raw_state)
    
    Output dimensions:
        - All options False: 19 features (raw state, pass-through)
        - normalize_resources=True: 15 features (drops 4 max columns)
    """
    
    # Indices of max value columns to drop after normalization
    MAX_VALUE_INDICES = [
        STATE_INDICES['HeroMaxHp'],   # 1
        STATE_INDICES['HeroMaxSp'],   # 3
        STATE_INDICES['HeroMaxFp'],   # 5
        STATE_INDICES['NpcMaxHp'],    # 12
    ]
    
    def __init__(
        self,
        normalize_resources: bool = True,
        compute_distances: bool = False,
        compute_relative_angles: bool = False,
    ):
        """Initialize preprocessor.
        
        Args:
            normalize_resources: Normalize HP/SP/FP by max values and drop max columns
            compute_distances: Add distance features (TODO)
            compute_relative_angles: Add relative angle features (TODO)
        """
        self.normalize_resources = normalize_resources
        self.compute_distances = compute_distances
        self.compute_relative_angles = compute_relative_angles
        
        logger.info(f"StatePreprocessor initialized:")
        logger.info(f"  normalize_resources: {normalize_resources}")
        logger.info(f"  compute_distances: {compute_distances}")
        logger.info(f"  compute_relative_angles: {compute_relative_angles}")
        logger.info(f"  output_dim: {self.output_dim}")
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Process state array.
        
        Args:
            state: Raw state array [B, num_features] or [num_features]
            
        Returns:
            Processed state array (same shape, may have additional features in future)
        """
        return self.process(state)
    
    def process(self, state: np.ndarray) -> np.ndarray:
        """Process state array.
        
        Args:
            state: Raw state array [B, num_features] or [num_features]
            
        Returns:
            Processed state array
        """
        # Handle both batched and single samples
        single_sample = state.ndim == 1
        if single_sample:
            state = state[np.newaxis, :]
        
        processed = state.copy()
        
        if self.normalize_resources:
            processed = self._normalize_resources(processed)
        
        if self.compute_distances:
            processed = self._add_distance_features(processed)
        
        if self.compute_relative_angles:
            processed = self._add_angle_features(processed)
        
        if single_sample:
            processed = processed[0]
        
        return processed
    
    def _normalize_resources(self, state: np.ndarray) -> np.ndarray:
        """Normalize HP, SP, FP by their max values, then drop max columns.
        
        Replaces raw values with ratios in [0, 1]:
        - HeroHp -> HeroHp / HeroMaxHp
        - HeroSp -> HeroSp / HeroMaxSp  
        - HeroFp -> HeroFp / HeroMaxFp
        - NpcHp -> NpcHp / NpcMaxHp
        
        Then drops the max columns (HeroMaxHp, HeroMaxSp, HeroMaxFp, NpcMaxHp)
        reducing features from 19 to 15.
        
        Args:
            state: State array [B, num_features]
            
        Returns:
            State with normalized resource values and max columns dropped
        """
        result = state.copy()
        
        # Hero HP ratio
        hp_idx = STATE_INDICES['HeroHp']
        max_hp_idx = STATE_INDICES['HeroMaxHp']
        max_hp = np.maximum(result[:, max_hp_idx], 1.0)  # Avoid division by zero
        result[:, hp_idx] = result[:, hp_idx] / max_hp
        
        # Hero SP (stamina) ratio
        sp_idx = STATE_INDICES['HeroSp']
        max_sp_idx = STATE_INDICES['HeroMaxSp']
        max_sp = np.maximum(result[:, max_sp_idx], 1.0)
        result[:, sp_idx] = result[:, sp_idx] / max_sp
        
        # Hero FP (mana) ratio
        fp_idx = STATE_INDICES['HeroFp']
        max_fp_idx = STATE_INDICES['HeroMaxFp']
        max_fp = np.maximum(result[:, max_fp_idx], 1.0)
        result[:, fp_idx] = result[:, fp_idx] / max_fp
        
        # NPC HP ratio
        npc_hp_idx = STATE_INDICES['NpcHp']
        npc_max_hp_idx = STATE_INDICES['NpcMaxHp']
        npc_max_hp = np.maximum(result[:, npc_max_hp_idx], 1.0)
        result[:, npc_hp_idx] = result[:, npc_hp_idx] / npc_max_hp
        
        # Drop max value columns (keep all columns except the max indices)
        keep_indices = [i for i in range(result.shape[1]) if i not in self.MAX_VALUE_INDICES]
        result = result[:, keep_indices]
        
        return result
    
    def _add_distance_features(self, state: np.ndarray) -> np.ndarray:
        """Add distance-based features.
        
        TODO: Compute distance between hero and NPC
        
        Args:
            state: State array [B, num_features]
            
        Returns:
            State with distance features appended
        """
        # TODO: Implement when needed
        # hero_pos = state[:, [STATE_INDICES['HeroGlobalPosX'], 
        #                      STATE_INDICES['HeroGlobalPosY'],
        #                      STATE_INDICES['HeroGlobalPosZ']]]
        # npc_pos = state[:, [STATE_INDICES['NpcGlobalPosX'],
        #                     STATE_INDICES['NpcGlobalPosY'], 
        #                     STATE_INDICES['NpcGlobalPosZ']]]
        # distance = np.linalg.norm(hero_pos - npc_pos, axis=1, keepdims=True)
        # return np.concatenate([state, distance], axis=1)
        return state
    
    def _add_angle_features(self, state: np.ndarray) -> np.ndarray:
        """Add relative angle features.
        
        TODO: Compute angle between hero facing and NPC direction
        
        Args:
            state: State array [B, num_features]
            
        Returns:
            State with angle features appended
        """
        # TODO: Implement when needed
        return state
    
    @property
    def output_dim(self) -> int:
        """Get output dimension after preprocessing.
        
        Returns:
            Number of features after preprocessing
        """
        base_dim = len(STATE_INDICES)  # 19
        
        # Drop max columns if normalizing
        if self.normalize_resources:
            base_dim -= len(self.MAX_VALUE_INDICES)  # 19 - 4 = 15
        
        extra_dim = 0
        if self.compute_distances:
            extra_dim += 1  # distance to NPC
        
        if self.compute_relative_angles:
            extra_dim += 1  # relative angle
        
        return base_dim + extra_dim


def create_preprocessor(config: Optional[Dict] = None) -> StatePreprocessor:
    """Factory function to create preprocessor from config.
    
    Args:
        config: Optional config dict with preprocessing settings
        
    Returns:
        StatePreprocessor instance
    """
    if config is None:
        config = {}
    
    preprocess_config = config.get('state_preprocessing', {})
    
    return StatePreprocessor(
        normalize_resources=preprocess_config.get('normalize_resources', True),
        compute_distances=preprocess_config.get('compute_distances', False),
        compute_relative_angles=preprocess_config.get('compute_relative_angles', False),
    )

