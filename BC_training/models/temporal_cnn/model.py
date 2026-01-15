"""Temporal CNN model - Frame stacking with action history for behavior cloning.

This model processes a sequence of frames and past actions to predict the next action,
providing temporal context that a single-frame model lacks.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
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
            x: [B, H, W, C] frame (already transposed)
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


class TemporalStateEncoder(nn.Module):
    """Encoder for temporally stacked states.

    Processes each timestep with shared MLP, then aggregates temporal info.
    """

    hidden_features: Tuple[int, ...] = (64, 64)
    output_features: int = 64
    anim_embed_dim: int = 16
    hero_anim_vocab_size: int = 67
    npc_anim_vocab_size: int = 54
    dropout_rate: float = 0.0
    use_batch_norm: bool = True

    @nn.compact
    def __call__(
        self,
        states,
        hero_anim_ids,
        npc_anim_ids,
        training: bool = True,
    ):
        """Encode temporal state sequence.

        Args:
            states: [B, T, num_features] stacked states
            hero_anim_ids: [B, T] hero animation indices
            npc_anim_ids: [B, T] NPC animation indices
            training: Whether in training mode

        Returns:
            [B, output_dim] encoded temporal state
        """
        batch_size, num_timesteps, num_features = states.shape

        # Animation embeddings
        hero_embed = nn.Embed(
            num_embeddings=self.hero_anim_vocab_size,
            features=self.anim_embed_dim,
            name='hero_anim_embed',
        )(hero_anim_ids)  # [B, T, embed_dim]

        npc_embed = nn.Embed(
            num_embeddings=self.npc_anim_vocab_size,
            features=self.anim_embed_dim,
            name='npc_anim_embed',
        )(npc_anim_ids)  # [B, T, embed_dim]

        # Concatenate state + embeddings at each timestep
        x = jnp.concatenate([states, hero_embed, npc_embed], axis=-1)  # [B, T, features+2*embed]

        # Process each timestep with shared MLP
        for features in self.hidden_features:
            x = nn.Dense(features=features)(x)
            if self.use_batch_norm:
                # Reshape for BatchNorm: [B*T, features]
                x_flat = x.reshape(-1, features)
                x_flat = nn.BatchNorm(use_running_average=not training)(x_flat)
                x = x_flat.reshape(batch_size, num_timesteps, features)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Flatten temporal dimension and project to output
        x = x.reshape(batch_size, -1)  # [B, T * hidden_features[-1]]
        x = nn.Dense(features=self.output_features)(x)
        x = nn.relu(x)

        return x


class TemporalCNN(nn.Module):
    """Temporal CNN with frame stacking and action history.

    Architecture:
        Frames [B, T, C, H, W] -> SharedCNN per frame -> [B, T, F]
        Action History [B, K, A] -> Flatten -> [B, K*A]
        (Optional) State [B, S] or [B, T, S] -> MLP -> [B, F2]
        (Optional) Anim embeds [B] or [B, T] -> Embed -> [B, E]
        Concat all temporal features -> Dense -> actions [B, A]

    Options for frame processing:
        1. Channel stacking: Concat frames along channel dim before CNN
        2. Shared encoder: Process each frame independently, concat features
        3. 3D Conv: Use 3D convolutions (not implemented)
    """

    num_actions: int
    num_history_frames: int  # Number of past frames (not including current)
    num_action_history: int  # Number of past actions

    # Optional state features
    use_state: bool = False
    stack_states: bool = False  # Whether states are temporally stacked [B, T, S]
    num_state_features: int = 10
    hero_anim_vocab_size: int = 67
    npc_anim_vocab_size: int = 54
    anim_embed_dim: int = 16

    # Architecture
    frame_mode: str = 'channel_stack'  # 'channel_stack' or 'shared_encoder'
    conv_features: Tuple[int, ...] = (32, 64, 128, 256)
    dense_features: Tuple[int, ...] = (512, 256)
    state_encoder_features: Tuple[int, ...] = (64, 64)
    state_output_features: int = 64
    action_history_features: int = 64  # Hidden dim for action history encoding
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
    ):
        """Forward pass.
        
        Args:
            frames: [B, T, C, H, W] stacked frames (T = num_history_frames + 1)
            action_history: [B, K, num_actions] past actions
            state: Optional [B, num_state_features] current state
            hero_anim_idx: Optional [B] hero animation indices
            npc_anim_idx: Optional [B] NPC animation indices
            training: Whether in training mode
            
        Returns:
            Action logits [B, num_actions]
        """
        batch_size = frames.shape[0]
        num_frames = frames.shape[1]
        
        # === Process frames ===
        if self.frame_mode == 'channel_stack':
            # Stack all frames along channel dimension
            # [B, T, C, H, W] -> [B, H, W, T*C]
            frames_hwc = jnp.transpose(frames, (0, 3, 4, 1, 2))  # [B, H, W, T, C]
            frames_hwc = frames_hwc.reshape(
                batch_size, frames_hwc.shape[1], frames_hwc.shape[2], -1
            )  # [B, H, W, T*C]
            
            # Single CNN on stacked channels
            x_vision = frames_hwc
            for i, features in enumerate(self.conv_features):
                stride = (2, 2) if i < len(self.conv_features) - 1 else (1, 1)
                x_vision = ConvBlock(
                    features=features,
                    kernel_size=(3, 3),
                    strides=stride,
                    use_batch_norm=self.use_batch_norm,
                )(x_vision, training=training)
            
            x_vision = jnp.mean(x_vision, axis=(1, 2))  # [B, feat_dim]
            
        else:  # shared_encoder
            # Process each frame with shared CNN, then concat
            frame_encoder = FrameEncoder(
                conv_features=self.conv_features,
                use_batch_norm=self.use_batch_norm,
            )
            
            # Process each timestep
            frame_features = []
            for t in range(num_frames):
                # [B, C, H, W] -> [B, H, W, C]
                frame_t = jnp.transpose(frames[:, t], (0, 2, 3, 1))
                feat_t = frame_encoder(frame_t, training=training)
                frame_features.append(feat_t)
            
            # Concat all frame features: [B, T * feat_dim]
            x_vision = jnp.concatenate(frame_features, axis=-1)
        
        # === Combine features ===
        features_to_concat = [x_vision]
        
        # === Process action history (if provided) ===
        if self.num_action_history > 0:
            # Flatten action history: [B, K, A] -> [B, K*A]
            action_history_flat = action_history.reshape(batch_size, -1)
            
            # Encode action history
            x_actions = nn.Dense(features=self.action_history_features)(action_history_flat)
            if self.use_batch_norm:
                x_actions = nn.BatchNorm(use_running_average=not training)(x_actions)
            x_actions = nn.relu(x_actions)
            features_to_concat.append(x_actions)
        
        # === Optional: State features ===
        if self.use_state and state is not None:
            if self.stack_states:
                # Temporal state encoding: state [B, T, S], anim_ids [B, T]
                x_state = TemporalStateEncoder(
                    hidden_features=self.state_encoder_features,
                    output_features=self.state_output_features,
                    anim_embed_dim=self.anim_embed_dim,
                    hero_anim_vocab_size=self.hero_anim_vocab_size,
                    npc_anim_vocab_size=self.npc_anim_vocab_size,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.use_batch_norm,
                )(state, hero_anim_idx, npc_anim_idx, training=training)
                features_to_concat.append(x_state)
            else:
                # Single state encoding: state [B, S], anim_ids [B]
                x_state = StateEncoder(
                    hidden_features=self.state_encoder_features,
                    output_features=self.state_output_features,
                    dropout_rate=self.dropout_rate,
                    use_batch_norm=self.use_batch_norm,
                )(state, training=training)
                features_to_concat.append(x_state)

                # Animation embeddings (only for non-stacked mode)
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
        x = nn.Dense(features=self.num_actions)(x)
        
        return x


def create_model(
    num_actions: int,
    num_history_frames: int = 4,
    num_action_history: int = 4,
    use_state: bool = False,
    stack_states: bool = False,
    num_state_features: int = 10,
    hero_anim_vocab_size: int = 67,
    npc_anim_vocab_size: int = 54,
    anim_embed_dim: int = 16,
    frame_mode: str = 'channel_stack',
    conv_features: tuple = (32, 64, 128, 256),
    dense_features: tuple = (512, 256),
    state_encoder_features: tuple = (64, 64),
    state_output_features: int = 64,
    action_history_features: int = 64,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = True,
) -> TemporalCNN:
    """Factory function to create TemporalCNN model.

    Args:
        num_actions: Number of action outputs
        num_history_frames: Number of past frames
        num_action_history: Number of past actions
        use_state: Whether to use state features
        stack_states: Whether to stack states temporally (same as frames)
        num_state_features: Number of continuous state features
        hero_anim_vocab_size: Size of hero animation embedding
        npc_anim_vocab_size: Size of NPC animation embedding
        anim_embed_dim: Animation embedding dimension
        frame_mode: 'channel_stack' or 'shared_encoder'
        conv_features: CNN feature dimensions
        dense_features: Dense layer dimensions
        state_encoder_features: State encoder hidden dims
        state_output_features: State encoder output dim
        action_history_features: Action history encoder dim
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch norm

    Returns:
        TemporalCNN model instance
    """
    model = TemporalCNN(
        num_actions=num_actions,
        num_history_frames=num_history_frames,
        num_action_history=num_action_history,
        use_state=use_state,
        stack_states=stack_states,
        num_state_features=num_state_features,
        hero_anim_vocab_size=hero_anim_vocab_size,
        npc_anim_vocab_size=npc_anim_vocab_size,
        anim_embed_dim=anim_embed_dim,
        frame_mode=frame_mode,
        conv_features=conv_features,
        dense_features=dense_features,
        state_encoder_features=state_encoder_features,
        state_output_features=state_output_features,
        action_history_features=action_history_features,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    )

    total_frames = num_history_frames + 1
    logger.info(f"Created TemporalCNN model:")
    logger.info(f"  Frame mode: {frame_mode}")
    logger.info(f"  Total frames: {total_frames} (history={num_history_frames} + current)")
    logger.info(f"  Action history: {num_action_history}")
    logger.info(f"  Conv features: {conv_features}")
    logger.info(f"  Dense features: {dense_features}")
    logger.info(f"  Use state: {use_state}")
    if use_state:
        logger.info(f"  Stack states: {stack_states}")
        logger.info(f"  State features: {num_state_features}")
        logger.info(f"  Anim embeds: hero={hero_anim_vocab_size}, npc={npc_anim_vocab_size}, dim={anim_embed_dim}")
    logger.info(f"  Action history features: {action_history_features}")
    logger.info(f"  Dropout: {dropout_rate}")
    logger.info(f"  Batch norm: {use_batch_norm}")

    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


