# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from vggt.heads.dpt_head import DPTHead


class AcousticHead(nn.Module):
    """
    Acoustic property prediction head for VGGT.
    
    Predicts frequency-dependent acoustic properties for each 3D point in the scene:
    - Absorption coefficients across frequency bands
    - Reflection coefficients 
    - Scattering coefficients (diffuse vs specular reflection)
    - Acoustic impedance (complex valued)
    
    This enables physics-based audio propagation simulation in the reconstructed 3D environment.
    """
    
    def __init__(
        self,
        dim_in: int = 2048,
        freq_bands: int = 8,
        features: int = 256,
        layers: List[int] = [4, 11, 17, 23],
        activation: str = "sigmoid",
        conf_activation: str = "expp1",
    ):
        """
        Args:
            dim_in: Input dimension from aggregated tokens
            freq_bands: Number of frequency bands for absorption prediction
            features: Feature dimension for DPT backbone
            layers: Which transformer layers to use for multi-scale features
            activation: Activation for acoustic properties
            conf_activation: Activation for confidence scores
        """
        super().__init__()
        
        self.freq_bands = freq_bands
        
        # Total output channels:
        # - absorption: freq_bands channels
        # - reflection: 1 channel
        # - scattering: 1 channel  
        # - impedance: 2 channels (real + imaginary)
        # - confidence: 1 channel
        output_dim = freq_bands + 5
        
        # Use DPT backbone for dense prediction
        self.dpt = DPTHead(
            dim_in=dim_in,
            features=features,
            output_dim=output_dim,
            intermediate_layer_idx=layers,
            activation="linear",  # We'll apply custom activations
            conf_activation="sigmoid"
        )
        
        # Frequency band centers for interpretation (in Hz)
        # Covers typical audio range: 125 Hz to 16 kHz
        freq_centers = torch.tensor([125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        self.register_buffer("freq_centers", freq_centers[:freq_bands])
        
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        return_raw: bool = False
    ) -> Tuple[dict, torch.Tensor]:
        """
        Forward pass to predict acoustic properties.
        
        Args:
            aggregated_tokens_list: Multi-scale features from aggregator
            images: Input images [B, S, 3, H, W]
            patch_start_idx: Starting index of patch tokens
            return_raw: Return raw predictions without activation
            
        Returns:
            Tuple of:
                - Dictionary containing acoustic properties
                - Confidence scores [B, S, H, W]
        """
        B, S = images.shape[:2]
        
        # Get raw predictions from DPT
        raw_output, conf_from_dpt = self.dpt(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx
        )
        
        if return_raw:
            return raw_output, conf_from_dpt
            
        # Split into individual properties
        # Note: DPT returns the last channel as confidence, so we have one less channel for properties
        actual_channels = raw_output.shape[-1]
        expected_channels = self.freq_bands + 4  # absorption + reflection + scattering + impedance
        
        if actual_channels < expected_channels:
            # Pad with zeros if needed
            padding = expected_channels - actual_channels
            raw_output = torch.nn.functional.pad(raw_output, (0, padding))
        
        absorption = raw_output[..., :self.freq_bands]
        reflection = raw_output[..., self.freq_bands:self.freq_bands+1]
        scattering = raw_output[..., self.freq_bands+1:self.freq_bands+2]
        impedance = raw_output[..., self.freq_bands+2:self.freq_bands+4]
        
        # Apply appropriate activations
        # Absorption: 0 to 1 per frequency band
        absorption = torch.sigmoid(absorption)
        
        # Reflection: 0 to 1 (energy reflection coefficient)
        reflection = torch.sigmoid(reflection)
        
        # Scattering: 0 to 1 (0 = purely specular, 1 = purely diffuse)
        scattering = torch.sigmoid(scattering)
        
        # Impedance: Keep as is (can be complex valued)
        # In practice, this represents normalized acoustic impedance
        
        # Use confidence from DPT head
        confidence = conf_from_dpt
        
        acoustic_properties = {
            "absorption": absorption,
            "reflection": reflection,
            "scattering": scattering,
            "impedance": impedance,
            "freq_centers": self.freq_centers
        }
        
        return acoustic_properties, confidence
    
    def compute_rt60(self, acoustic_properties: dict, room_volume: float, surface_area: float) -> torch.Tensor:
        """
        Compute frequency-dependent RT60 (reverberation time) from acoustic properties.
        Uses Sabine's equation: RT60 = 0.161 * V / A
        
        Args:
            acoustic_properties: Dict containing absorption coefficients
            room_volume: Volume of the room in m³
            surface_area: Total surface area in m²
            
        Returns:
            RT60 values for each frequency band [freq_bands]
        """
        # Average absorption coefficient per frequency
        avg_absorption = acoustic_properties["absorption"].mean(dim=(0, 1, 2, 3))
        
        # Sabine's equation
        # A = total absorption = avg_absorption * surface_area
        # RT60 = 0.161 * V / A
        rt60 = 0.161 * room_volume / (avg_absorption * surface_area + 1e-6)
        
        return rt60