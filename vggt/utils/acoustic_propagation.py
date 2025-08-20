# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class AcousticPropagation(nn.Module):
    """
    Simulates acoustic propagation through 3D environments using learned acoustic properties.
    
    This module takes:
    - Source audio and position
    - Listener position and orientation
    - 3D geometry (depth/points)
    - Acoustic properties (absorption, reflection, etc.)
    
    And produces:
    - Spatialized audio at the listener position
    - Including effects like reverb, occlusion, and distance attenuation
    """
    
    def __init__(
        self,
        max_order_reflections: int = 2,
        speed_of_sound: float = 343.0,
        air_absorption: bool = True,
        freq_bands: int = 8,
    ):
        """
        Args:
            max_order_reflections: Maximum order of reflections to simulate
            speed_of_sound: Speed of sound in m/s
            air_absorption: Whether to model frequency-dependent air absorption
            freq_bands: Number of frequency bands for filtering
        """
        super().__init__()
        
        self.max_order_reflections = max_order_reflections
        self.speed_of_sound = speed_of_sound
        self.air_absorption = air_absorption
        self.freq_bands = freq_bands
        
        # Learnable parameters for acoustic transfer function
        self.distance_attenuation = nn.Parameter(torch.tensor(1.0))
        self.air_absorption_coeffs = nn.Parameter(torch.zeros(freq_bands))
        
        # Neural acoustic filter
        self.acoustic_filter = nn.Sequential(
            nn.Linear(freq_bands * 4 + 7, 256),  # Input: acoustic features
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, freq_bands * 2),  # Output: filter params per band
        )
        
        # HRTF approximation for spatial audio
        self.hrtf_net = nn.Sequential(
            nn.Linear(6, 128),  # Input: relative position + orientation
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * freq_bands),  # Output: left/right filters
        )
        
    def forward(
        self,
        source_audio: torch.Tensor,
        source_position: torch.Tensor,
        listener_position: torch.Tensor,
        listener_orientation: torch.Tensor,
        world_points: torch.Tensor,
        acoustic_properties: Dict[str, torch.Tensor],
        depth_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate acoustic propagation from source to listener.
        
        Args:
            source_audio: Source audio signal [batch, samples] or [batch, freq_bands, samples]
            source_position: 3D position of source [batch, 3]
            listener_position: 3D position of listener [batch, 3]
            listener_orientation: Listener orientation quaternion [batch, 4]
            world_points: 3D points from VGGT [batch, H, W, 3]
            acoustic_properties: Dict from AcousticHead with absorption, reflection, etc.
            depth_map: Optional depth map for occlusion [batch, H, W]
            
        Returns:
            Dict containing:
                - binaural_audio: Spatialized stereo audio [batch, 2, samples]
                - reverb_audio: Reverberant component [batch, samples]
                - direct_audio: Direct sound component [batch, samples]
                - acoustic_params: Parameters used for rendering
        """
        batch_size = source_position.shape[0]
        
        # 1. Compute direct path
        direct_path_info = self._compute_direct_path(
            source_position, listener_position, world_points, depth_map
        )
        
        # 2. Compute reflections
        reflection_paths = self._compute_reflections(
            source_position, listener_position, world_points, acoustic_properties
        )
        
        # 3. Apply acoustic filtering based on paths and materials
        filtered_audio = self._apply_acoustic_filtering(
            source_audio, direct_path_info, reflection_paths, acoustic_properties
        )
        
        # 4. Spatialize audio (convert to binaural)
        binaural_audio = self._spatialize_audio(
            filtered_audio["direct"],
            filtered_audio["reflections"],
            source_position,
            listener_position,
            listener_orientation,
        )
        
        return {
            "binaural_audio": binaural_audio,
            "reverb_audio": filtered_audio["reflections"].sum(dim=1),  # Sum all reflections
            "direct_audio": filtered_audio["direct"],
            "acoustic_params": {
                "direct_distance": direct_path_info["distance"],
                "occlusion": direct_path_info["occlusion"],
                "rt60": self._estimate_rt60(acoustic_properties, world_points),
            }
        }
    
    def _compute_direct_path(
        self,
        source_pos: torch.Tensor,
        listener_pos: torch.Tensor,
        world_points: torch.Tensor,
        depth_map: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute direct path from source to listener with occlusion check."""
        # Calculate distance
        distance = torch.norm(listener_pos - source_pos, dim=-1)
        
        # Check for occlusion using ray marching through depth/points
        occlusion = torch.zeros(source_pos.shape[0])
        
        if depth_map is not None:
            # Simplified occlusion check
            # In practice, you'd ray march through the 3D scene
            direction = (listener_pos - source_pos) / distance.unsqueeze(-1)
            # ... ray marching code ...
            
        return {
            "distance": distance,
            "direction": direction,
            "occlusion": occlusion,
        }
    
    def _compute_reflections(
        self,
        source_pos: torch.Tensor,
        listener_pos: torch.Tensor,
        world_points: torch.Tensor,
        acoustic_properties: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute reflection paths up to max_order."""
        reflections = []
        
        # For each reflection order
        for order in range(1, self.max_order_reflections + 1):
            # Find reflection points on surfaces
            # This is a simplified version - full implementation would use
            # image source method or ray tracing
            
            # Placeholder: compute some reflection paths
            reflection = {
                "order": order,
                "path_length": torch.zeros(source_pos.shape[0]),
                "reflection_points": torch.zeros(source_pos.shape[0], order, 3),
                "surface_properties": [],  # Acoustic properties at each reflection
            }
            reflections.append(reflection)
            
        return reflections
    
    def _apply_acoustic_filtering(
        self,
        source_audio: torch.Tensor,
        direct_path: Dict[str, torch.Tensor],
        reflections: List[Dict[str, torch.Tensor]],
        acoustic_properties: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply frequency-dependent filtering based on propagation paths."""
        batch_size = source_audio.shape[0]
        
        # Convert audio to frequency domain if needed
        if source_audio.dim() == 2:  # [batch, samples]
            # Apply FFT to get frequency representation
            audio_freq = torch.fft.rfft(source_audio)
            freq_bins = audio_freq.shape[-1]
        else:  # Already in frequency bands
            audio_freq = source_audio
            freq_bins = self.freq_bands
            
        # Filter direct sound
        direct_filtered = self._filter_direct_sound(
            audio_freq, direct_path["distance"], direct_path["occlusion"]
        )
        
        # Filter reflections
        reflection_filtered = []
        for reflection in reflections:
            filtered = self._filter_reflection(
                audio_freq, reflection, acoustic_properties
            )
            reflection_filtered.append(filtered)
            
        # Convert back to time domain if needed
        if source_audio.dim() == 2:
            direct_filtered = torch.fft.irfft(direct_filtered)
            reflection_filtered = [torch.fft.irfft(r) for r in reflection_filtered]
            
        return {
            "direct": direct_filtered,
            "reflections": torch.stack(reflection_filtered) if reflection_filtered else torch.zeros_like(direct_filtered),
        }
    
    def _filter_direct_sound(
        self,
        audio_freq: torch.Tensor,
        distance: torch.Tensor,
        occlusion: torch.Tensor,
    ) -> torch.Tensor:
        """Apply distance attenuation and air absorption to direct sound."""
        # Distance attenuation (inverse square law)
        attenuation = 1.0 / (distance + 1.0) ** self.distance_attenuation
        
        # Air absorption (frequency dependent)
        if self.air_absorption:
            # Higher frequencies absorbed more over distance
            freq_attenuation = torch.exp(-self.air_absorption_coeffs * distance.unsqueeze(-1))
            attenuation = attenuation.unsqueeze(-1) * freq_attenuation
            
        # Occlusion filtering (low-pass effect)
        occlusion_filter = 1.0 - occlusion.unsqueeze(-1) * 0.8  # Reduce high frequencies
        
        return audio_freq * attenuation * occlusion_filter
    
    def _filter_reflection(
        self,
        audio_freq: torch.Tensor,
        reflection: Dict[str, torch.Tensor],
        acoustic_properties: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply surface absorption and scattering to reflected sound."""
        # Get average surface properties along reflection path
        # In practice, this would sample properties at each reflection point
        avg_absorption = acoustic_properties["absorption"].mean(dim=(1, 2, 3))
        avg_reflection = acoustic_properties["reflection"].mean(dim=(1, 2, 3))
        
        # Apply surface absorption (frequency dependent)
        surface_filter = (1.0 - avg_absorption) * avg_reflection
        
        # Apply distance attenuation for total path length
        distance_atten = 1.0 / (reflection["path_length"] + 1.0) ** self.distance_attenuation
        
        return audio_freq * surface_filter * distance_atten.unsqueeze(-1)
    
    def _spatialize_audio(
        self,
        direct_audio: torch.Tensor,
        reflection_audio: torch.Tensor,
        source_pos: torch.Tensor,
        listener_pos: torch.Tensor,
        listener_orient: torch.Tensor,
    ) -> torch.Tensor:
        """Convert mono audio to binaural using learned HRTF."""
        # Calculate relative position in listener's coordinate frame
        relative_pos = source_pos - listener_pos
        
        # Convert listener orientation from quaternion to rotation matrix
        # Simplified - full implementation would properly handle rotations
        
        # Get HRTF filters
        hrtf_input = torch.cat([relative_pos, listener_orient[:, :3]], dim=-1)
        hrtf_filters = self.hrtf_net(hrtf_input)
        
        # Split into left/right ear filters
        left_filter = hrtf_filters[:, :self.freq_bands]
        right_filter = hrtf_filters[:, self.freq_bands:]
        
        # Apply filters to create binaural audio
        # Simplified - full implementation would properly convolve in time domain
        left_audio = direct_audio * left_filter.mean(dim=-1, keepdim=True)
        right_audio = direct_audio * right_filter.mean(dim=-1, keepdim=True)
        
        # Add reflections (less directional)
        if reflection_audio.dim() > 2:
            reflection_sum = reflection_audio.sum(dim=1)
        else:
            reflection_sum = reflection_audio
            
        left_audio = left_audio + 0.5 * reflection_sum
        right_audio = right_audio + 0.5 * reflection_sum
        
        return torch.stack([left_audio, right_audio], dim=1)
    
    def _estimate_rt60(
        self,
        acoustic_properties: Dict[str, torch.Tensor],
        world_points: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate reverberation time from acoustic properties."""
        # Estimate room volume from point cloud bounds
        min_coords = world_points.min(dim=1)[0].min(dim=1)[0]
        max_coords = world_points.max(dim=1)[0].max(dim=1)[0]
        room_dimensions = max_coords - min_coords
        volume = room_dimensions.prod(dim=-1)
        
        # Estimate surface area (simplified as box)
        surface_area = 2 * (
            room_dimensions[:, 0] * room_dimensions[:, 1] +
            room_dimensions[:, 1] * room_dimensions[:, 2] +
            room_dimensions[:, 0] * room_dimensions[:, 2]
        )
        
        # Average absorption
        avg_absorption = acoustic_properties["absorption"].mean(dim=(1, 2, 3, 4))
        
        # Sabine's equation: RT60 = 0.161 * V / (A * S)
        rt60 = 0.161 * volume / (avg_absorption * surface_area + 1e-6)
        
        return rt60