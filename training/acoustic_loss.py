# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import torchaudio


class AcousticLoss(nn.Module):
    """
    Multi-task loss for acoustic property learning in VGGT-Audio.
    
    Combines several loss components:
    - Acoustic rendering loss (match recorded audio)
    - Physical consistency loss (enforce physical constraints)
    - Perceptual loss (match perceptual features)
    - RT60 loss (match reverberation time)
    """
    
    def __init__(
        self,
        rendering_weight: float = 1.0,
        physical_weight: float = 0.1,
        perceptual_weight: float = 0.5,
        rt60_weight: float = 0.2,
        freq_bands: int = 8,
        sample_rate: int = 48000,
    ):
        """
        Args:
            rendering_weight: Weight for audio rendering loss
            physical_weight: Weight for physical consistency
            perceptual_weight: Weight for perceptual similarity
            rt60_weight: Weight for reverberation time matching
            freq_bands: Number of frequency bands
            sample_rate: Audio sample rate
        """
        super().__init__()
        
        self.rendering_weight = rendering_weight
        self.physical_weight = physical_weight
        self.perceptual_weight = perceptual_weight
        self.rt60_weight = rt60_weight
        self.freq_bands = freq_bands
        
        # Mel spectrogram for perceptual loss
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        
    def forward(
        self,
        rendered_audio: torch.Tensor,
        target_audio: torch.Tensor,
        acoustic_properties: Dict[str, torch.Tensor],
        rt60_target: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute acoustic losses.
        
        Args:
            rendered_audio: Rendered audio from acoustic propagation [batch, 2, samples]
            target_audio: Ground truth recorded audio [batch, 2, samples]
            acoustic_properties: Predicted acoustic properties from AcousticHead
            rt60_target: Target RT60 values [batch, freq_bands] (optional)
            confidence: Confidence scores from acoustic head [batch, H, W]
            
        Returns:
            Dictionary of individual losses and total loss
        """
        losses = {}
        
        # 1. Audio rendering loss (time domain)
        rendering_loss = self._compute_rendering_loss(
            rendered_audio, target_audio, confidence
        )
        losses["rendering"] = rendering_loss
        
        # 2. Perceptual loss (frequency domain)
        perceptual_loss = self._compute_perceptual_loss(
            rendered_audio, target_audio
        )
        losses["perceptual"] = perceptual_loss
        
        # 3. Physical consistency loss
        physical_loss = self._compute_physical_loss(acoustic_properties)
        losses["physical"] = physical_loss
        
        # 4. RT60 loss (if ground truth available)
        if rt60_target is not None:
            rt60_loss = self._compute_rt60_loss(
                acoustic_properties, rt60_target
            )
            losses["rt60"] = rt60_loss
        else:
            losses["rt60"] = torch.tensor(0.0)
            
        # Combine losses
        total_loss = (
            self.rendering_weight * losses["rendering"] +
            self.perceptual_weight * losses["perceptual"] +
            self.physical_weight * losses["physical"] +
            self.rt60_weight * losses["rt60"]
        )
        
        losses["total"] = total_loss
        return losses
    
    def _compute_rendering_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        confidence: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute L1 loss between rendered and target audio."""
        # Convert to mono for comparison if needed
        if rendered.shape[1] == 2 and target.shape[1] == 2:
            # Binaural - compute loss for each channel
            loss = F.l1_loss(rendered, target, reduction='none')
            loss = loss.mean(dim=(1, 2))  # Average over channels and time
        else:
            loss = F.l1_loss(rendered, target, reduction='none')
            loss = loss.mean(dim=-1)
            
        # Weight by confidence if provided
        if confidence is not None:
            # Average confidence over spatial dimensions
            conf_weight = confidence.mean(dim=(1, 2))
            loss = loss * conf_weight
            
        return loss.mean()
    
    def _compute_perceptual_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss using mel spectrograms."""
        # Convert to mono
        rendered_mono = rendered.mean(dim=1)
        target_mono = target.mean(dim=1)
        
        # Compute mel spectrograms
        rendered_mel = self.mel_transform(rendered_mono)
        target_mel = self.mel_transform(target_mono)
        
        # Log scale
        rendered_mel_db = 20 * torch.log10(rendered_mel + 1e-6)
        target_mel_db = 20 * torch.log10(target_mel + 1e-6)
        
        # L1 loss in mel space
        loss = F.l1_loss(rendered_mel_db, target_mel_db)
        
        return loss
    
    def _compute_physical_loss(
        self,
        acoustic_properties: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Enforce physical constraints on acoustic properties."""
        losses = []
        
        # 1. Absorption should be between 0 and 1
        absorption = acoustic_properties["absorption"]
        absorption_loss = torch.relu(-absorption) + torch.relu(absorption - 1.0)
        losses.append(absorption_loss.mean())
        
        # 2. Reflection coefficient should be between 0 and 1
        reflection = acoustic_properties["reflection"]
        reflection_loss = torch.relu(-reflection) + torch.relu(reflection - 1.0)
        losses.append(reflection_loss.mean())
        
        # 3. Energy conservation: absorption + reflection <= 1
        energy_sum = absorption.mean(dim=-1) + reflection.squeeze(-1)
        conservation_loss = torch.relu(energy_sum - 1.0)
        losses.append(conservation_loss.mean())
        
        # 4. Scattering coefficient between 0 and 1
        scattering = acoustic_properties["scattering"]
        scattering_loss = torch.relu(-scattering) + torch.relu(scattering - 1.0)
        losses.append(scattering_loss.mean())
        
        # 5. Impedance should be positive (real part)
        impedance_real = acoustic_properties["impedance"][..., 0]
        impedance_loss = torch.relu(-impedance_real)
        losses.append(impedance_loss.mean())
        
        return sum(losses) / len(losses)
    
    def _compute_rt60_loss(
        self,
        acoustic_properties: Dict[str, torch.Tensor],
        rt60_target: torch.Tensor,
    ) -> torch.Tensor:
        """Match predicted RT60 to target."""
        # Compute RT60 from acoustic properties
        # This is simplified - full implementation would use room geometry
        avg_absorption = acoustic_properties["absorption"].mean(dim=(1, 2, 3))
        
        # Approximate RT60 using average absorption
        # RT60 ∝ 1 / absorption
        predicted_rt60 = 1.0 / (avg_absorption + 0.1)  # Add small constant
        
        # Scale to match typical RT60 range (0.1 to 2.0 seconds)
        predicted_rt60 = predicted_rt60 * 0.5
        
        # L1 loss on RT60
        loss = F.l1_loss(predicted_rt60, rt60_target)
        
        return loss


class AcousticConsistencyLoss(nn.Module):
    """
    Ensures consistency between visual and acoustic predictions.
    
    For example:
    - Hard surfaces (high depth gradient) should have low absorption
    - Large open spaces should have longer RT60
    - Occluded paths should have attenuated sound
    """
    
    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.consistency_weight = consistency_weight
        
    def forward(
        self,
        acoustic_properties: Dict[str, torch.Tensor],
        depth_map: torch.Tensor,
        world_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency between geometry and acoustics.
        
        Args:
            acoustic_properties: Predicted acoustic properties
            depth_map: Depth predictions [batch, H, W]
            world_points: 3D point predictions [batch, H, W, 3]
            
        Returns:
            Consistency loss scalar
        """
        losses = []
        
        # 1. Surface hardness from depth gradients
        depth_grad_x = torch.abs(depth_map[:, :, 1:] - depth_map[:, :, :-1])
        depth_grad_y = torch.abs(depth_map[:, 1:, :] - depth_map[:, :-1, :])
        
        # High gradient = hard surface = low absorption
        avg_grad_x = F.avg_pool2d(depth_grad_x.unsqueeze(1), kernel_size=2).squeeze(1)
        avg_grad_y = F.avg_pool2d(depth_grad_y.unsqueeze(1), kernel_size=2).squeeze(1)
        
        # Pad to match acoustic property size
        avg_grad_x = F.pad(avg_grad_x, (0, 1, 0, 0))
        avg_grad_y = F.pad(avg_grad_y, (0, 0, 0, 1))
        
        surface_hardness = (avg_grad_x + avg_grad_y) / 2.0
        surface_hardness = torch.clamp(surface_hardness / surface_hardness.max(), 0, 1)
        
        # Expected absorption (inverse of hardness)
        expected_absorption = 1.0 - surface_hardness
        predicted_absorption = acoustic_properties["absorption"].mean(dim=-1).squeeze(-1)
        
        # Resize if needed
        if expected_absorption.shape != predicted_absorption.shape:
            expected_absorption = F.interpolate(
                expected_absorption.unsqueeze(1),
                size=predicted_absorption.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
        absorption_consistency = F.mse_loss(predicted_absorption, expected_absorption)
        losses.append(absorption_consistency)
        
        # 2. Room size consistency
        # Larger rooms should have different acoustic properties
        room_size = world_points.max(dim=1)[0] - world_points.min(dim=1)[0]
        room_volume = room_size.prod(dim=-1)
        
        # Normalize room volume
        room_volume_norm = torch.log(room_volume + 1.0) / 10.0
        
        # Large rooms tend to have lower average absorption
        expected_avg_absorption = torch.clamp(1.0 - room_volume_norm * 0.1, 0.3, 0.7)
        actual_avg_absorption = acoustic_properties["absorption"].mean()
        
        volume_consistency = F.mse_loss(actual_avg_absorption, expected_avg_absorption)
        losses.append(volume_consistency)
        
        return self.consistency_weight * sum(losses) / len(losses)