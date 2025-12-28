"""State preprocessing for behavior cloning models.

Transforms raw game state into normalized/processed features.
Used by all models that consume state data.
"""

import numpy as np
from typing import Dict, Optional
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

# NPC Y coordinate offset (discovered empirically - 8 bits)
NPC_Y_OFFSET = -8.0


class StatePreprocessor:
    """Preprocesses raw game state into normalized features.
    
    Features:
    - Normalizes HP, SP, FP to [0, 1] using their max values
    - Computes relative position features (distance, direction to NPC)
    - Drops useless columns: HeroAngle (unreliable), NpcAngle (stuck at 1), NpcId (useless)
    
    Usage:
        preprocessor = StatePreprocessor()
        processed_state = preprocessor(raw_state)
    
    Output structure (with both options enabled = 12 features):
        Resources (4): hero_hp_ratio, hero_sp_ratio, hero_fp_ratio, npc_hp_ratio
        Distances (6): distance_xy, distance_3d, rel_x, rel_y, rel_z, angle_to_npc
        Anim IDs (2): hero_anim_id, npc_anim_id
    """
    
    def __init__(
        self,
        normalize_resources: bool = True,
        compute_distances: bool = False,
    ):
        """Initialize preprocessor.
        
        Args:
            normalize_resources: Normalize HP/SP/FP by max values and drop max columns
            compute_distances: Replace raw coordinates with derived features:
                - distance_xy: 2D distance to NPC on ground plane
                - distance_3d: 3D Euclidean distance to NPC
                - rel_x, rel_y, rel_z: Relative position of NPC to hero
                - angle_to_npc: Computed angle from hero to NPC (radians)
                Also drops: HeroAngle (unreliable), NpcGlobalPosAngle (stuck at 1)
        """
        self.normalize_resources = normalize_resources
        self.compute_distances = compute_distances
        
        logger.info(f"StatePreprocessor initialized:")
        logger.info(f"  normalize_resources: {normalize_resources}")
        logger.info(f"  compute_distances: {compute_distances}")
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
        
        # Extract all values from original indices FIRST
        # Resources
        hero_hp = state[:, STATE_INDICES['HeroHp']]
        hero_max_hp = np.maximum(state[:, STATE_INDICES['HeroMaxHp']], 1.0)
        hero_sp = state[:, STATE_INDICES['HeroSp']]
        hero_max_sp = np.maximum(state[:, STATE_INDICES['HeroMaxSp']], 1.0)
        hero_fp = state[:, STATE_INDICES['HeroFp']]
        hero_max_fp = np.maximum(state[:, STATE_INDICES['HeroMaxFp']], 1.0)
        npc_hp = state[:, STATE_INDICES['NpcHp']]
        npc_max_hp = np.maximum(state[:, STATE_INDICES['NpcMaxHp']], 1.0)
        
        # Coordinates
        hero_x = state[:, STATE_INDICES['HeroGlobalPosX']]
        hero_y = state[:, STATE_INDICES['HeroGlobalPosY']]
        hero_z = state[:, STATE_INDICES['HeroGlobalPosZ']]
        npc_x = state[:, STATE_INDICES['NpcGlobalPosX']]
        npc_y = state[:, STATE_INDICES['NpcGlobalPosY']] + NPC_Y_OFFSET  # Apply 8-bit offset
        npc_z = state[:, STATE_INDICES['NpcGlobalPosZ']]
        
        # Animation IDs (keep as-is, but drop NpcId - it leaks identity)
        hero_anim_id = state[:, STATE_INDICES['HeroAnimId']]
        npc_anim_id = state[:, STATE_INDICES['NpcAnimId']]
        
        # Build output features
        features = []
        
        # 1. Resources (normalized or raw)
        if self.normalize_resources:
            features.extend([
                hero_hp / hero_max_hp,
                hero_sp / hero_max_sp,
                hero_fp / hero_max_fp,
                npc_hp / npc_max_hp,
            ])
        else:
            features.extend([
                hero_hp, hero_max_hp,
                hero_sp, hero_max_sp,
                hero_fp, hero_max_fp,
                npc_hp, npc_max_hp,
            ])
        
        # 2. Coordinates (derived or raw)
        if self.compute_distances:
            # Relative position
            rel_x = npc_x - hero_x
            rel_y = npc_y - hero_y
            rel_z = npc_z - hero_z
            
            # Distances
            distance_xy = np.sqrt(rel_x**2 + rel_y**2)
            distance_3d = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
            
            # Computed angle from hero to NPC (in radians, [-pi, pi])
            angle_to_npc = np.arctan2(rel_y, rel_x)
            
            features.extend([
                distance_xy,
                distance_3d,
                rel_x,
                rel_y,
                rel_z,
                angle_to_npc,
            ])
        else:
            # Raw coordinates (no angles - they're useless)
            features.extend([
                hero_x, hero_y, hero_z,
                npc_x, npc_y, npc_z,
            ])
        
        # 3. Animation IDs (NpcId dropped - leaks identity info)
        features.extend([
            hero_anim_id,
            npc_anim_id,
        ])
        
        # Stack into [B, num_features]
        processed = np.stack(features, axis=1)
        
        if single_sample:
            processed = processed[0]
        
        return processed
    
    @property
    def output_dim(self) -> int:
        """Get output dimension after preprocessing.
        
        Output structure:
        - Resources: 4 (normalized) or 8 (raw with max values)
        - Coordinates: 6 (derived: dist_xy, dist_3d, rel_x, rel_y, rel_z, angle) or 6 (raw: 3 hero + 3 npc)
        - IDs: 2 (hero_anim, npc_anim) - NpcId dropped
        
        Returns:
            Number of features after preprocessing
        """
        # Resources
        if self.normalize_resources:
            dim = 4  # hero_hp, hero_sp, hero_fp, npc_hp (all normalized)
        else:
            dim = 8  # hero_hp, hero_max_hp, hero_sp, hero_max_sp, hero_fp, hero_max_fp, npc_hp, npc_max_hp
        
        # Coordinates
        dim += 6  # Either derived (dist_xy, dist_3d, rel_x, rel_y, rel_z, angle) or raw (6 coords)
        
        # Animation IDs (NpcId dropped - leaks identity)
        dim += 2  # hero_anim_id, npc_anim_id
        
        return dim
    
    def get_feature_names(self) -> list:
        """Get names of output features for debugging/logging."""
        names = []
        
        # Resources
        if self.normalize_resources:
            names.extend(['hero_hp_ratio', 'hero_sp_ratio', 'hero_fp_ratio', 'npc_hp_ratio'])
        else:
            names.extend(['hero_hp', 'hero_max_hp', 'hero_sp', 'hero_max_sp', 
                         'hero_fp', 'hero_max_fp', 'npc_hp', 'npc_max_hp'])
        
        # Coordinates
        if self.compute_distances:
            names.extend(['distance_xy', 'distance_3d', 'rel_x', 'rel_y', 'rel_z', 'angle_to_npc'])
        else:
            names.extend(['hero_x', 'hero_y', 'hero_z', 'npc_x', 'npc_y', 'npc_z'])
        
        # Animation IDs (NpcId dropped)
        names.extend(['hero_anim_id', 'npc_anim_id'])
        
        return names


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
    )

