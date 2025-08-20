# LoRA Adapter for VGGT-Audio Fine-tuning
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

class LoRALayer(nn.Module):
    """
    LoRA adapter layer for parameter-efficient fine-tuning.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize with Kaiming
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply LoRA: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_out * self.scaling


class VGGTAudioLoRA(nn.Module):
    """
    LoRA wrapper for VGGT-Audio model with parameter-efficient audio fine-tuning.
    """
    def __init__(
        self,
        base_model,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: List[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        
        if target_modules is None:
            # Default: Apply LoRA to audio-related and attention modules
            target_modules = [
                "audio_proj",  # Audio feature projection
                "acoustic_head",  # Acoustic property prediction
                # Attention layers in last 6 blocks (most task-specific)
                "global_blocks.18", "global_blocks.19", "global_blocks.20",
                "global_blocks.21", "global_blocks.22", "global_blocks.23",
            ]
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Add LoRA adapters
        self.lora_layers = nn.ModuleDict()
        self._inject_lora(target_modules, dropout)
        
    def _inject_lora(self, target_modules: List[str], dropout: float):
        """Inject LoRA layers into target modules."""
        for name, module in self.base_model.named_modules():
            # Check if this module should get LoRA
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Add LoRA adapter
                    lora_key = name.replace(".", "_")
                    self.lora_layers[lora_key] = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=dropout
                    )
                    
                    # Store reference to original linear layer
                    module.lora_adapter = self.lora_layers[lora_key]
                    
                    # Monkey-patch forward method
                    original_forward = module.forward
                    def new_forward(self, x):
                        base_out = original_forward(x)
                        if hasattr(self, 'lora_adapter') and self.training:
                            base_out = base_out + self.lora_adapter(x)
                        return base_out
                    module.forward = new_forward.__get__(module, type(module))
    
    def forward(self, images, audio_features=None, **kwargs):
        return self.base_model(images, audio_features=audio_features, **kwargs)
    
    def get_trainable_params(self):
        """Return only LoRA parameters for training."""
        return self.lora_layers.parameters()
    
    def merge_and_unload(self):
        """Merge LoRA weights into base model for inference."""
        for name, module in self.base_model.named_modules():
            if hasattr(module, 'lora_adapter'):
                # Merge weights: W = W + BA * scaling
                with torch.no_grad():
                    lora = module.lora_adapter
                    delta_w = lora.lora_B @ lora.lora_A * lora.scaling
                    module.weight.data += delta_w
                # Remove adapter
                delattr(module, 'lora_adapter')
        return self.base_model