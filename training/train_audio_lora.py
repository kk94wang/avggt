#!/usr/bin/env python3
"""
Training script for VGGT-Audio with LoRA adapters.
Designed for parameter-efficient fine-tuning with limited visual-acoustic data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from vggt.models.vggt_audio import VGGTAudio
from vggt.models.lora_adapter import VGGTAudioLoRA
from vggt.utils.audio_utils import AudioFeatureExtractor
from vggt.utils.colmap_loader import COLMAPAudioDataset
from training.acoustic_loss import AcousticLoss


class UnrealAudioDataset(torch.utils.data.Dataset):
    """Dataset for Unreal Engine exported visual-acoustic data."""
    
    def __init__(
        self,
        data_dir: str,
        scene_list: list,
        max_frames: int = 30,
        audio_extractor: AudioFeatureExtractor = None,
    ):
        self.data_dir = Path(data_dir)
        self.scenes = scene_list
        self.max_frames = max_frames
        self.audio_extractor = audio_extractor or AudioFeatureExtractor()
        
        # Load all scene metadata
        self.scene_data = []
        for scene_id in self.scenes:
            scene_path = self.data_dir / scene_id
            if (scene_path / "transforms.json").exists():
                self.scene_data.append({
                    "scene_id": scene_id,
                    "path": scene_path,
                    "transforms": json.load(open(scene_path / "transforms.json"))
                })
    
    def __len__(self):
        return len(self.scene_data)
    
    def __getitem__(self, idx):
        scene = self.scene_data[idx]
        
        # Load frames
        frames = []
        frame_data = scene["transforms"]["frames"][:self.max_frames]
        for frame_info in frame_data:
            img_path = scene["path"] / frame_info["file_path"]
            img = load_and_preprocess_image(str(img_path))
            frames.append(img)
        images = torch.stack(frames)
        
        # Load audio
        source_audio, _ = torchaudio.load(scene["path"] / "audio/source.wav")
        binaural_audio, _ = torchaudio.load(scene["path"] / "audio/binaural.wav")
        
        # Extract audio features
        audio_features = self.audio_extractor.extract_features(binaural_audio)
        
        # Load acoustic properties if available
        acoustic_props = None
        if (scene["path"] / "acoustic_properties.json").exists():
            acoustic_props = json.load(open(scene["path"] / "acoustic_properties.json"))
        
        return {
            "images": images,
            "audio_features": audio_features,
            "source_audio": source_audio,
            "binaural_audio": binaural_audio,
            "acoustic_properties": acoustic_props,
            "camera_params": scene["transforms"]
        }


def train_lora(args):
    """Main training loop for LoRA fine-tuning."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Load base model
    print(f"Loading base VGGT model from {args.checkpoint}")
    base_model = VGGTAudio.from_pretrained_vggt(
        args.checkpoint,
        enable_acoustic=True,
        audio_feature_dim=args.audio_feature_dim,
        freq_bands=args.freq_bands
    ).to(device)
    
    # Wrap with LoRA
    print(f"Adding LoRA adapters with rank={args.lora_rank}")
    model = VGGTAudioLoRA(
        base_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.dropout,
        target_modules=args.target_modules
    )
    
    # Setup dataset
    print(f"Loading dataset from {args.data_dir}")
    train_scenes = json.load(open(args.data_dir / "train_scenes.json"))
    val_scenes = json.load(open(args.data_dir / "val_scenes.json"))
    
    audio_extractor = AudioFeatureExtractor(
        feature_type=args.audio_feature_type,
        n_mels=128
    )
    
    train_dataset = UnrealAudioDataset(
        args.data_dir,
        train_scenes,
        max_frames=args.max_frames,
        audio_extractor=audio_extractor
    )
    
    val_dataset = UnrealAudioDataset(
        args.data_dir,
        val_scenes,
        max_frames=args.max_frames,
        audio_extractor=audio_extractor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Setup optimizer (only LoRA parameters)
    trainable_params = list(model.get_trainable_params())
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in base_model.parameters())
    print(f"Trainable params: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.2f}%)")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader)
    )
    
    # Loss function
    acoustic_loss_fn = AcousticLoss(
        rendering_weight=args.rendering_weight,
        physical_weight=args.physical_weight,
        perceptual_weight=args.perceptual_weight
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            # Move to device
            images = batch["images"].to(device)
            audio_features = batch["audio_features"].to(device)
            binaural_target = batch["binaural_audio"].to(device)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                predictions = model(images, audio_features=audio_features)
                
                # Compute loss
                if "acoustic_properties" in predictions:
                    loss = acoustic_loss_fn(
                        rendered_audio=predictions.get("rendered_audio", binaural_target),
                        target_audio=binaural_target,
                        acoustic_properties=predictions["acoustic_properties"],
                        confidence=predictions.get("acoustic_conf")
                    )
                    total_loss = sum(loss.values())
                else:
                    total_loss = torch.tensor(0.0).to(device)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            train_losses.append(total_loss.item())
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = batch["images"].to(device)
                audio_features = batch["audio_features"].to(device)
                binaural_target = batch["binaural_audio"].to(device)
                
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    predictions = model(images, audio_features=audio_features)
                    
                    if "acoustic_properties" in predictions:
                        loss = acoustic_loss_fn(
                            rendered_audio=predictions.get("rendered_audio", binaural_target),
                            target_audio=binaural_target,
                            acoustic_properties=predictions["acoustic_properties"]
                        )
                        total_loss = sum(loss.values())
                    else:
                        total_loss = torch.tensor(0.0)
                
                val_losses.append(total_loss.item())
        
        # Print epoch summary
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model to {args.output_dir}/best_lora.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.lora_layers.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
                "args": args
            }, f"{args.output_dir}/best_lora.pt")
    
    # Merge LoRA weights for deployment
    if args.merge_weights:
        print("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        torch.save(merged_model.state_dict(), f"{args.output_dir}/merged_model.pt")
        print(f"Saved merged model to {args.output_dir}/merged_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train VGGT-Audio with LoRA")
    
    # Data arguments
    parser.add_argument("--data_dir", type=Path, required=True,
                       help="Directory containing Unreal Engine exported data")
    parser.add_argument("--checkpoint", type=str, default="facebook/VGGT-1B",
                       help="Base VGGT checkpoint to fine-tune")
    parser.add_argument("--output_dir", type=Path, default="checkpoints/lora",
                       help="Output directory for checkpoints")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank (lower = fewer parameters)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA scaling factor")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, nargs="+",
                       default=None, help="Modules to apply LoRA to")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (scenes per batch)")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate (higher than normal due to LoRA)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping value")
    
    # Model arguments
    parser.add_argument("--audio_feature_dim", type=int, default=768,
                       help="Audio feature dimension")
    parser.add_argument("--freq_bands", type=int, default=8,
                       help="Number of frequency bands")
    parser.add_argument("--audio_feature_type", type=str, default="mel_spectrogram",
                       choices=["mel_spectrogram", "mfcc"],
                       help="Type of audio features")
    parser.add_argument("--max_frames", type=int, default=30,
                       help="Maximum frames per scene")
    
    # Loss weights
    parser.add_argument("--rendering_weight", type=float, default=1.0,
                       help="Weight for audio rendering loss")
    parser.add_argument("--physical_weight", type=float, default=0.1,
                       help="Weight for physical consistency loss")
    parser.add_argument("--perceptual_weight", type=float, default=0.5,
                       help="Weight for perceptual loss")
    
    # Other arguments
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--merge_weights", action="store_true",
                       help="Merge LoRA weights into base model after training")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    train_lora(args)


if __name__ == "__main__":
    main()