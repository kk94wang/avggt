# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from vggt.models.aggregator import Aggregator, slice_expand_and_flatten


class AudioAggregator(Aggregator):
    """
    Extended Aggregator that incorporates audio tokens for audio-visual learning.
    
    Adds audio tokens to the token sequence to enable cross-modal attention
    between visual and audio features. The audio tokens capture acoustic
    information from the environment.
    """
    
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        num_audio_tokens=2,
        audio_feature_dim=768,
        block_fn=None,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        """
        Args:
            num_audio_tokens: Number of audio tokens per frame (default: 2)
                - One for direct sound characteristics
                - One for reverberant/ambient characteristics
            audio_feature_dim: Dimension of input audio features
            Other args: Same as base Aggregator
        """
        # Import here to avoid circular dependency
        from vggt.layers.block import Block as DefaultBlock
        if block_fn is None:
            block_fn = DefaultBlock
            
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            block_fn=block_fn,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            patch_embed=patch_embed,
            aa_order=aa_order,
            aa_block_size=aa_block_size,
            qk_norm=qk_norm,
            rope_freq=rope_freq,
            init_values=init_values,
        )
        
        self.num_audio_tokens = num_audio_tokens
        
        # Audio tokens: 2 tokens per frame (direct + reverberant)
        self.audio_token = nn.Parameter(torch.randn(1, 2, num_audio_tokens, embed_dim))
        nn.init.normal_(self.audio_token, std=1e-6)
        
        # Audio feature projection
        if audio_feature_dim != embed_dim:
            self.audio_proj = nn.Linear(audio_feature_dim, embed_dim)
        else:
            self.audio_proj = nn.Identity()
        
        # Update patch start index to account for audio tokens
        self.patch_start_idx = 1 + num_audio_tokens + num_register_tokens
        
    def forward(
        self,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Forward pass with optional audio features.
        
        Args:
            images: Input images [B, S, 3, H, W]
            audio_features: Audio features [B, S, audio_feature_dim] or None
            
        Returns:
            Tuple of:
                - List of aggregated tokens at different depths
                - Starting index of patch tokens
        """
        B, S, C_in, H, W = images.shape
        
        # Normalize images and get patch tokens
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
            
        _, P, C = patch_tokens.shape
        
        # Prepare special tokens
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        audio_token = slice_expand_and_flatten(self.audio_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        
        # Process audio features if provided
        if audio_features is not None:
            # Project audio features to embedding dimension
            audio_features = audio_features.view(B * S, -1)
            audio_features = self.audio_proj(audio_features)
            
            # Modulate audio tokens with audio features
            # This allows the tokens to capture frame-specific audio information
            audio_features = audio_features.unsqueeze(1)  # [B*S, 1, C]
            audio_token = audio_token + audio_features.expand(-1, self.num_audio_tokens, -1)
        
        # Concatenate all tokens
        # Order: camera, audio, register, patch
        tokens = torch.cat([camera_token, audio_token, register_token, patch_tokens], dim=1)
        
        # Position embeddings
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            
        if self.patch_start_idx > 0:
            # Don't use position embedding for special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
            
        # Update P to include all tokens
        _, P, C = tokens.shape
        
        # Process through alternating attention blocks
        frame_idx = 0
        global_idx = 0
        output_list = []
        
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                    
            for i in range(len(frame_intermediates)):
                # Concatenate frame and global intermediates
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
                
        del concat_inter
        del frame_intermediates
        del global_intermediates
        
        return output_list, self.patch_start_idx
    
    def get_audio_tokens(self, aggregated_tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract audio tokens from aggregated tokens.
        
        Args:
            aggregated_tokens: Tokens with shape [B, S, P, C]
            
        Returns:
            Audio tokens with shape [B, S, num_audio_tokens, C]
        """
        # Audio tokens are after camera token (1) and before register tokens
        audio_start = 1
        audio_end = audio_start + self.num_audio_tokens
        return aggregated_tokens[:, :, audio_start:audio_end, :]