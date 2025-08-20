# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional, Dict
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.vggt import VGGT
from vggt.models.audio_aggregator import AudioAggregator
from vggt.heads.acoustic_head import AcousticHead
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGTAudio(nn.Module, PyTorchModelHubMixin):
    """
    VGGT extended with audio understanding capabilities.
    
    This model jointly learns visual geometry and acoustic properties of environments,
    enabling physics-based audio propagation simulation in reconstructed 3D scenes.
    """
    
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        enable_acoustic=True,
        num_audio_tokens=2,
        audio_feature_dim=768,
        freq_bands=8,
    ):
        """
        Args:
            enable_acoustic: Whether to enable acoustic property prediction
            num_audio_tokens: Number of audio tokens in aggregator
            audio_feature_dim: Dimension of input audio features
            freq_bands: Number of frequency bands for acoustic properties
            Other args: Same as base VGGT
        """
        super().__init__()
        
        # Use audio-aware aggregator
        self.aggregator = AudioAggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_audio_tokens=num_audio_tokens,
            audio_feature_dim=audio_feature_dim,
        )
        
        # Initialize prediction heads
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None
        self.acoustic_head = AcousticHead(dim_in=2 * embed_dim, freq_bands=freq_bands) if enable_acoustic else None
        
        self.enable_acoustic = enable_acoustic
        
    def forward(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the audio-enabled VGGT model.
        
        Args:
            images: Input images [B, S, 3, H, W] or [S, 3, H, W]
            query_points: Query points for tracking [B, N, 2] or [N, 2]
            audio_features: Audio features [B, S, audio_feature_dim] or [S, audio_feature_dim]
            
        Returns:
            Dictionary containing:
                - All outputs from base VGGT (pose_enc, depth, world_points, etc.)
                - acoustic_properties: Dict with absorption, reflection, scattering, impedance
                - acoustic_conf: Confidence scores for acoustic predictions
        """
        # Handle batch dimension
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
            
        if audio_features is not None and len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(0)
            
        # Get aggregated tokens with audio features
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, audio_features)
        
        predictions = {}
        
        # Run all prediction heads
        device_type = 'cuda' if images.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
                
            if self.acoustic_head is not None:
                acoustic_props, acoustic_conf = self.acoustic_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["acoustic_properties"] = acoustic_props
                predictions["acoustic_conf"] = acoustic_conf
                
        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf
            
        if not self.training:
            predictions["images"] = images
            if audio_features is not None:
                predictions["audio_features"] = audio_features
                
        return predictions
    
    @classmethod
    def from_pretrained_vggt(
        cls,
        model_name: str,
        enable_acoustic: bool = True,
        freeze_visual: bool = True,
        **kwargs
    ):
        """
        Load from pretrained VGGT and extend with acoustic capabilities.
        
        Args:
            model_name: Name of pretrained VGGT model (e.g., "facebook/VGGT-1B")
            enable_acoustic: Whether to add acoustic head
            freeze_visual: Whether to freeze visual components during acoustic training
            **kwargs: Additional arguments for model initialization
            
        Returns:
            VGGTAudio model with loaded visual weights
        """
        # Load base VGGT
        base_vggt = VGGT.from_pretrained(model_name)
        
        # Create audio model
        audio_model = cls(enable_acoustic=enable_acoustic, **kwargs)
        
        # Copy weights from base model
        # Note: This is a simplified version - in practice you'd need more careful weight copying
        audio_model.camera_head.load_state_dict(base_vggt.camera_head.state_dict())
        audio_model.depth_head.load_state_dict(base_vggt.depth_head.state_dict())
        audio_model.point_head.load_state_dict(base_vggt.point_head.state_dict())
        if hasattr(base_vggt, 'track_head') and base_vggt.track_head is not None:
            audio_model.track_head.load_state_dict(base_vggt.track_head.state_dict())
            
        # Copy aggregator weights (except audio tokens)
        base_state = base_vggt.aggregator.state_dict()
        audio_state = audio_model.aggregator.state_dict()
        for key in base_state:
            if key in audio_state and audio_state[key].shape == base_state[key].shape:
                audio_state[key] = base_state[key]
        audio_model.aggregator.load_state_dict(audio_state)
        
        # Optionally freeze visual components
        if freeze_visual:
            for param in audio_model.aggregator.parameters():
                param.requires_grad = False
            # Only unfreeze audio tokens
            audio_model.aggregator.audio_token.requires_grad = True
            audio_model.aggregator.audio_proj.requires_grad = True
            
            # Freeze visual heads
            for head in [audio_model.camera_head, audio_model.depth_head, 
                        audio_model.point_head, audio_model.track_head]:
                if head is not None:
                    for param in head.parameters():
                        param.requires_grad = False
                        
        return audio_model