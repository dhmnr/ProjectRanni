"""Temporal CNN model - Frame stacking with action history for behavior cloning.

This model processes a sequence of frames and past actions to predict the next action,
providing temporal context that a single-frame model lacks.

Supports optional attention mechanisms:
- Temporal attention over frames
- Temporal attention over states
- Cross-attention between frames and states
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


class TemporalSelfAttention(nn.Module):
    """Self-attention over temporal dimension.

    Learns to weight different timesteps based on their relevance.
    Includes learned positional embeddings for temporal order awareness.
    """

    num_heads: int = 4
    head_dim: int = 32
    dropout_rate: float = 0.0
    max_seq_len: int = 16  # Maximum sequence length for positional embeddings

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Apply self-attention over temporal dimension.

        Args:
            x: [B, T, D] temporal features
            training: Whether in training mode

        Returns:
            [B, T, D] attended features (same shape as input)
        """
        batch_size, num_timesteps, feat_dim = x.shape

        # Learned positional embeddings
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, feat_dim)
        )
        x = x + pos_embed[:num_timesteps]  # Add positional info

        # Multi-head attention
        x_attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads * self.head_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, x)  # Self-attention: query=key=value=x

        # Residual connection
        x = x + x_attended

        # Layer norm
        x = nn.LayerNorm()(x)

        return x


class TemporalAttentionPooling(nn.Module):
    """Attention-based pooling over temporal dimension.

    Instead of mean/concat, learns which timesteps to focus on.
    Returns weighted combination of temporal features.
    """

    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Pool temporal features using attention.

        Args:
            x: [B, T, D] temporal features
            training: Whether in training mode

        Returns:
            [B, D] pooled features
        """
        batch_size, num_timesteps, feat_dim = x.shape

        # Compute attention scores for each timestep
        # Project to scalar score per timestep
        scores = nn.Dense(features=self.hidden_dim)(x)  # [B, T, hidden]
        scores = nn.tanh(scores)
        scores = nn.Dense(features=1)(scores)  # [B, T, 1]
        scores = scores.squeeze(-1)  # [B, T]

        # Softmax over timesteps
        weights = nn.softmax(scores, axis=-1)  # [B, T]

        # Weighted sum
        weights = weights[:, :, None]  # [B, T, 1]
        pooled = jnp.sum(x * weights, axis=1)  # [B, D]

        return pooled


class CrossAttention(nn.Module):
    """Cross-attention between two feature sequences.

    Used to let frames attend to states or vice versa.
    Includes learned positional embeddings for both sequences.
    """

    num_heads: int = 4
    head_dim: int = 32
    dropout_rate: float = 0.0
    max_seq_len: int = 16  # Maximum sequence length for positional embeddings

    @nn.compact
    def __call__(self, query, key_value, training: bool = True):
        """Apply cross-attention.

        Args:
            query: [B, T1, D1] query features
            key_value: [B, T2, D2] key/value features
            training: Whether in training mode

        Returns:
            [B, T1, D1] attended features (same shape as query)
        """
        batch_size, num_q_timesteps, q_feat_dim = query.shape
        _, num_kv_timesteps, kv_feat_dim = key_value.shape

        # Learned positional embeddings for query and key_value
        query_pos_embed = self.param(
            'query_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, q_feat_dim)
        )
        kv_pos_embed = self.param(
            'kv_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, kv_feat_dim)
        )
        query = query + query_pos_embed[:num_q_timesteps]
        key_value = key_value + kv_pos_embed[:num_kv_timesteps]

        # Cross-attention: query attends to key_value
        x_attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads * self.head_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(query, key_value)

        # Residual connection (project if dimensions differ)
        if query.shape[-1] != x_attended.shape[-1]:
            query = nn.Dense(features=x_attended.shape[-1])(query)
        x = query + x_attended

        # Layer norm
        x = nn.LayerNorm()(x)

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

    Auxiliary tasks:
        - NPC animation prediction: Predict NPC animation ID from visual features only
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

    # Attention options
    use_frame_attention: bool = False  # Self-attention over frame features
    use_state_attention: bool = False  # Self-attention over state features (requires stack_states)
    use_cross_attention: bool = False  # Cross-attention between frames and states
    attention_num_heads: int = 4
    attention_head_dim: int = 32

    # Auxiliary task: predict NPC animation from vision
    use_aux_npc_anim: bool = False  # Enable auxiliary NPC animation prediction
    aux_npc_anim_classes: int = 16  # Number of animation classes (top N + "other")

    # Animation onset timing feature
    use_anim_onset_timing: bool = False  # Include frames_since_npc_anim_onset
    anim_onset_encoding_dim: int = 16  # Dimension of sinusoidal encoding
    
    @nn.compact
    def __call__(
        self,
        frames,
        action_history,
        state=None,
        hero_anim_idx=None,
        npc_anim_idx=None,
        anim_onset_encoding=None,
        training: bool = True,
    ):
        """Forward pass.

        Args:
            frames: [B, T, C, H, W] stacked frames (T = num_history_frames + 1)
            action_history: [B, K, num_actions] past actions
            state: Optional [B, num_state_features] current state or [B, T, num_state_features]
            hero_anim_idx: Optional [B] or [B, T] hero animation indices
            npc_anim_idx: Optional [B] or [B, T] NPC animation indices
            anim_onset_encoding: Optional [B, encoding_dim] or [B, T, encoding_dim] sinusoidal timing
            training: Whether in training mode
            
        Returns:
            Action logits [B, num_actions]
        """
        batch_size = frames.shape[0]
        num_frames = frames.shape[1]
        
        # === Process frames ===
        # Track temporal frame features for potential attention/cross-attention
        frame_features_temporal = None  # [B, T, D] if needed

        if self.frame_mode == 'channel_stack' and not self.use_frame_attention:
            # Stack all frames along channel dimension (no attention possible)
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

        else:  # shared_encoder or frame_attention enabled
            # Process each frame with shared CNN
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

            # Stack as temporal sequence: [B, T, feat_dim]
            frame_features_temporal = jnp.stack(frame_features, axis=1)

            if self.use_frame_attention:
                # Apply self-attention over frame features
                frame_features_temporal = TemporalSelfAttention(
                    num_heads=self.attention_num_heads,
                    head_dim=self.attention_head_dim,
                    dropout_rate=self.dropout_rate,
                )(frame_features_temporal, training=training)

                # Attention pooling over time
                x_vision = TemporalAttentionPooling(
                    hidden_dim=self.conv_features[-1],
                )(frame_features_temporal, training=training)
            else:
                # Concat all frame features: [B, T * feat_dim]
                x_vision = frame_features_temporal.reshape(batch_size, -1)
        
        # === Auxiliary task: predict NPC animation from vision ===
        aux_npc_anim_logits = None
        if self.use_aux_npc_anim:
            # Predict NPC animation from visual features ONLY (before fusion with state)
            # This forces the visual encoder to learn attack recognition
            aux_hidden = nn.Dense(features=128)(x_vision)
            if self.use_batch_norm:
                aux_hidden = nn.BatchNorm(use_running_average=not training)(aux_hidden)
            aux_hidden = nn.relu(aux_hidden)
            aux_hidden = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(aux_hidden)
            aux_npc_anim_logits = nn.Dense(features=self.aux_npc_anim_classes, name='aux_npc_anim_head')(aux_hidden)

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
        state_features_temporal = None  # [B, T, D] if needed for cross-attention

        if self.use_state and state is not None:
            if self.stack_states:
                # Build temporal state features: [B, T, D]
                # First embed animations and concat with state
                hero_embed = nn.Embed(
                    num_embeddings=self.hero_anim_vocab_size,
                    features=self.anim_embed_dim,
                    name='hero_anim_embed',
                )(hero_anim_idx)  # [B, T, embed_dim]

                npc_embed = nn.Embed(
                    num_embeddings=self.npc_anim_vocab_size,
                    features=self.anim_embed_dim,
                    name='npc_anim_embed',
                )(npc_anim_idx)  # [B, T, embed_dim]

                # Concat state + embeddings: [B, T, state_features + 2*embed_dim]
                state_with_embeds = jnp.concatenate([state, hero_embed, npc_embed], axis=-1)

                # Encode each timestep with shared MLP
                state_features_temporal = state_with_embeds
                for features in self.state_encoder_features:
                    state_features_temporal = nn.Dense(features=features)(state_features_temporal)
                    if self.use_batch_norm:
                        # Reshape for BatchNorm
                        orig_shape = state_features_temporal.shape
                        state_features_temporal = state_features_temporal.reshape(-1, features)
                        state_features_temporal = nn.BatchNorm(use_running_average=not training)(state_features_temporal)
                        state_features_temporal = state_features_temporal.reshape(orig_shape)
                    state_features_temporal = nn.relu(state_features_temporal)

                # Apply state attention if enabled
                if self.use_state_attention:
                    state_features_temporal = TemporalSelfAttention(
                        num_heads=self.attention_num_heads,
                        head_dim=self.attention_head_dim,
                        dropout_rate=self.dropout_rate,
                    )(state_features_temporal, training=training)

                    # Attention pooling
                    x_state = TemporalAttentionPooling(
                        hidden_dim=self.state_encoder_features[-1],
                    )(state_features_temporal, training=training)
                else:
                    # Flatten temporal: [B, T * D]
                    x_state = state_features_temporal.reshape(batch_size, -1)
                    x_state = nn.Dense(features=self.state_output_features)(x_state)
                    x_state = nn.relu(x_state)

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

        # === Cross-attention between frames and states ===
        if self.use_cross_attention and frame_features_temporal is not None and state_features_temporal is not None:
            # Frames attend to states
            frame_attended = CrossAttention(
                num_heads=self.attention_num_heads,
                head_dim=self.attention_head_dim,
                dropout_rate=self.dropout_rate,
            )(frame_features_temporal, state_features_temporal, training=training)

            # Pool attended features
            x_cross = TemporalAttentionPooling(
                hidden_dim=self.conv_features[-1],
            )(frame_attended, training=training)

            features_to_concat.append(x_cross)

        # === Animation onset timing feature ===
        if self.use_anim_onset_timing and anim_onset_encoding is not None:
            if anim_onset_encoding.ndim == 3:
                # Stacked: [B, T, encoding_dim] -> use last timestep [B, encoding_dim]
                x_onset = anim_onset_encoding[:, -1, :]
            else:
                # Single: [B, encoding_dim]
                x_onset = anim_onset_encoding
            features_to_concat.append(x_onset)

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
        action_logits = nn.Dense(features=self.num_actions)(x)

        # Return dict if auxiliary task is enabled, else just action logits
        if self.use_aux_npc_anim:
            return {
                'action_logits': action_logits,
                'aux_npc_anim_logits': aux_npc_anim_logits,
            }

        return action_logits


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
    use_frame_attention: bool = False,
    use_state_attention: bool = False,
    use_cross_attention: bool = False,
    attention_num_heads: int = 4,
    attention_head_dim: int = 32,
    use_aux_npc_anim: bool = False,
    aux_npc_anim_classes: int = 16,
    use_anim_onset_timing: bool = False,
    anim_onset_encoding_dim: int = 16,
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
        use_frame_attention: Self-attention over frame features
        use_state_attention: Self-attention over state features
        use_cross_attention: Cross-attention between frames and states
        attention_num_heads: Number of attention heads
        attention_head_dim: Dimension per attention head
        use_aux_npc_anim: Enable auxiliary NPC animation prediction from vision
        aux_npc_anim_classes: Number of NPC animation classes for auxiliary task
        use_anim_onset_timing: Enable animation onset timing feature
        anim_onset_encoding_dim: Dimension of sinusoidal encoding for onset timing

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
        use_frame_attention=use_frame_attention,
        use_state_attention=use_state_attention,
        use_cross_attention=use_cross_attention,
        attention_num_heads=attention_num_heads,
        attention_head_dim=attention_head_dim,
        use_aux_npc_anim=use_aux_npc_anim,
        aux_npc_anim_classes=aux_npc_anim_classes,
        use_anim_onset_timing=use_anim_onset_timing,
        anim_onset_encoding_dim=anim_onset_encoding_dim,
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
    if use_frame_attention or use_state_attention or use_cross_attention:
        logger.info(f"  Attention: frame={use_frame_attention}, state={use_state_attention}, cross={use_cross_attention}")
        logger.info(f"  Attention heads: {attention_num_heads}, head_dim: {attention_head_dim}")
    if use_aux_npc_anim:
        logger.info(f"  Auxiliary task: NPC animation prediction ({aux_npc_anim_classes} classes)")
    if use_anim_onset_timing:
        logger.info(f"  Animation onset timing: encoding_dim={anim_onset_encoding_dim}")

    return model


def count_parameters(params) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


