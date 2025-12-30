"""GRU model - Recurrent temporal processing for behavior cloning.

Uses a GRU to process frame sequences, maintaining hidden state across timesteps.
Better for variable-length sequences and natural for real-time inference.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block with Conv -> BatchNorm -> ReLU."""
    
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME',
        )(x)
        
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        
        x = nn.relu(x)
        return x


class FrameEncoder(nn.Module):
    """CNN encoder for a single frame. Shared across time steps."""
    
    conv_features: Tuple[int, ...] = (32, 64, 128, 256)
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """Encode a single frame.
        
        Args:
            x: [B, H, W, C] frame (NHWC format)
            training: Whether in training mode
            
        Returns:
            [B, feat_dim] encoded frame
        """
        for i, features in enumerate(self.conv_features):
            stride = (2, 2) if i < len(self.conv_features) - 1 else (1, 1)
            x = ConvBlock(
                features=features,
                kernel_size=(3, 3),
                strides=stride,
                use_batch_norm=self.use_batch_norm,
            )(x, training=training)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # [B, C]
        return x


class StateEncoder(nn.Module):
    """MLP encoder for continuous state features."""
    
    hidden_features: Tuple[int, ...] = (64, 64)
    output_features: int = 64
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, state, training: bool = True):
        x = state
        
        for features in self.hidden_features:
            x = nn.Dense(features=features)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(features=self.output_features)(x)
        x = nn.relu(x)
        
        return x


class GRUTemporalModel(nn.Module):
    """GRU-based temporal model for behavior cloning.
    
    Architecture:
        frames[t-k:t] -> SharedCNN per frame -> [B, T, F]
        [B, T, F] -> GRU -> hidden states -> final hidden [B, H]
        (Optional) State [B, S] -> MLP -> [B, F2]
        (Optional) Anim embeds [B] -> Embed -> [B, E]
        (Optional) Action history [B, K, A] -> MLP -> [B, F3]
        Concat all -> Dense -> actions [B, A]
    
    Key advantages over TemporalCNN:
        - Learns temporal dependencies explicitly
        - Hidden state carries context naturally
        - Better for variable-length sequences
        - More natural for real-time inference (hidden state persists)
    """
    
    num_actions: int
    num_history_frames: int  # Number of past frames (not including current)
    num_action_history: int  # Number of past actions
    
    # GRU configuration
    gru_hidden_size: int = 256
    gru_num_layers: int = 1
    bidirectional: bool = False  # Must be False for causal inference!
    
    # Optional state features
    use_state: bool = False
    num_state_features: int = 10
    hero_anim_vocab_size: int = 67
    npc_anim_vocab_size: int = 54
    anim_embed_dim: int = 16
    
    # Architecture
    conv_features: Tuple[int, ...] = (32, 64, 128, 256)
    dense_features: Tuple[int, ...] = (256, 128)
    state_encoder_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    action_history_features: int = 64
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(
        self,
        frames,
        action_history,
        state=None,
        hero_anim_idx=None,
        npc_anim_idx=None,
        training: bool = True,
        initial_hidden: Optional[jnp.ndarray] = None,
    ):
        """Forward pass.
        
        Args:
            frames: [B, T, C, H, W] stacked frames (T = num_history_frames + 1)
            action_history: [B, K, num_actions] past actions
            state: Optional [B, num_state_features] current state
            hero_anim_idx: Optional [B] hero animation indices
            npc_anim_idx: Optional [B] NPC animation indices
            training: Whether in training mode
            initial_hidden: Optional [B, gru_hidden_size] initial hidden state
            
        Returns:
            Action logits [B, num_actions]
        """
        batch_size = frames.shape[0]
        num_frames = frames.shape[1]
        
        # === Encode each frame with shared CNN ===
        frame_encoder = FrameEncoder(
            conv_features=self.conv_features,
            use_batch_norm=self.use_batch_norm,
        )
        
        # Process each timestep: [B, T, C, H, W] -> list of [B, feat_dim]
        frame_features = []
        for t in range(num_frames):
            # [B, C, H, W] -> [B, H, W, C]
            frame_t = jnp.transpose(frames[:, t], (0, 2, 3, 1))
            feat_t = frame_encoder(frame_t, training=training)
            frame_features.append(feat_t)
        
        # Stack to [B, T, feat_dim]
        frame_features = jnp.stack(frame_features, axis=1)
        
        # === Process through GRU ===
        # Initialize hidden state
        if initial_hidden is None:
            hidden = jnp.zeros((batch_size, self.gru_hidden_size))
        else:
            hidden = initial_hidden
        
        # Multi-layer GRU
        for layer_idx in range(self.gru_num_layers):
            gru_cell = nn.GRUCell(
                features=self.gru_hidden_size,
                name=f'gru_layer_{layer_idx}'
            )
            
            # Process sequence
            hiddens = []
            h = hidden
            for t in range(num_frames):
                # GRUCell input: (carry, inputs) -> (new_carry, outputs)
                # For GRUCell, carry == output, so new_carry is the new hidden state
                h, _ = gru_cell(h, frame_features[:, t, :])
                hiddens.append(h)
            
            # Use output of last timestep as input to next layer (if multi-layer)
            # Or stack for potential attention later
            frame_features = jnp.stack(hiddens, axis=1)  # [B, T, H]
            hidden = h  # Final hidden state
        
        # Apply dropout to GRU output
        if self.dropout_rate > 0 and training:
            hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(hidden)
        
        # === Process action history ===
        # Flatten and encode: [B, K, A] -> [B, K*A] -> [B, action_feat]
        action_history_flat = action_history.reshape(batch_size, -1)
        x_actions = nn.Dense(features=self.action_history_features)(action_history_flat)
        if self.use_batch_norm:
            x_actions = nn.BatchNorm(use_running_average=not training)(x_actions)
        x_actions = nn.relu(x_actions)
        
        # === Combine features ===
        features_to_concat = [hidden, x_actions]
        
        # === Optional: State features ===
        if self.use_state and state is not None:
            x_state = StateEncoder(
                hidden_features=self.state_encoder_features,
                output_features=self.state_output_features,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
            )(state, training=training)
            features_to_concat.append(x_state)
            
            # Animation embeddings
            if hero_anim_idx is not None and npc_anim_idx is not None:
                hero_embed = nn.Embed(
                    num_embeddings=self.hero_anim_vocab_size,
                    features=self.anim_embed_dim,
                    name='hero_anim_embed',
                )(hero_anim_idx)
                
                npc_embed = nn.Embed(
                    num_embeddings=self.npc_anim_vocab_size,
                    features=self.anim_embed_dim,
                    name='npc_anim_embed',
                )(npc_anim_idx)
                
                features_to_concat.extend([hero_embed, npc_embed])
        
        # === Fusion and prediction ===
        x = jnp.concatenate(features_to_concat, axis=-1)
        
        for features in self.dense_features:
            x = nn.Dense(features=features)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output logits
        logits = nn.Dense(features=self.num_actions)(x)
        
        return logits


def create_model(
    num_actions: int,
    num_history_frames: int = 4,
    num_action_history: int = 4,
    gru_hidden_size: int = 256,
    gru_num_layers: int = 1,
    use_state: bool = False,
    num_state_features: int = 10,
    hero_anim_vocab_size: int = 67,
    npc_anim_vocab_size: int = 54,
    anim_embed_dim: int = 16,
    conv_features: tuple = (32, 64, 128, 256),
    dense_features: tuple = (256, 128),
    state_encoder_features: tuple = (64, 64),
    state_output_features: int = 64,
    action_history_features: int = 64,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
) -> GRUTemporalModel:
    """Factory function to create GRU model.
    
    Args:
        num_actions: Number of action outputs
        num_history_frames: Number of past frames
        num_action_history: Number of past actions
        gru_hidden_size: GRU hidden state dimension
        gru_num_layers: Number of stacked GRU layers
        use_state: Whether to use state features
        num_state_features: Number of continuous state features
        hero_anim_vocab_size: Size of hero animation embedding
        npc_anim_vocab_size: Size of NPC animation embedding
        anim_embed_dim: Animation embedding dimension
        conv_features: CNN feature dimensions
        dense_features: Dense layer dimensions
        state_encoder_features: State encoder hidden dims
        state_output_features: State encoder output dim
        action_history_features: Action history encoder dim
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch norm
        
    Returns:
        GRUTemporalModel instance
    """
    model = GRUTemporalModel(
        num_actions=num_actions,
        num_history_frames=num_history_frames,
        num_action_history=num_action_history,
        gru_hidden_size=gru_hidden_size,
        gru_num_layers=gru_num_layers,
        use_state=use_state,
        num_state_features=num_state_features,
        hero_anim_vocab_size=hero_anim_vocab_size,
        npc_anim_vocab_size=npc_anim_vocab_size,
        anim_embed_dim=anim_embed_dim,
        conv_features=conv_features,
        dense_features=dense_features,
        state_encoder_features=state_encoder_features,
        state_output_features=state_output_features,
        action_history_features=action_history_features,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )
    
    total_frames = num_history_frames + 1
    logger.info(f"Created GRUTemporalModel:")
    logger.info(f"  GRU hidden size: {gru_hidden_size}")
    logger.info(f"  GRU layers: {gru_num_layers}")
    logger.info(f"  Total frames: {total_frames} (history={num_history_frames} + current)")
    logger.info(f"  Action history: {num_action_history}")
    logger.info(f"  Conv features: {conv_features}")
    logger.info(f"  Dense features: {dense_features}")
    logger.info(f"  Use state: {use_state}")
    if use_state:
        logger.info(f"  State features: {num_state_features}")
        logger.info(f"  Anim embeds: hero={hero_anim_vocab_size}, npc={npc_anim_vocab_size}, dim={anim_embed_dim}")
    logger.info(f"  Action history features: {action_history_features}")
    logger.info(f"  Dropout: {dropout_rate}")
    logger.info(f"  Batch norm: {use_batch_norm}")
    
    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


