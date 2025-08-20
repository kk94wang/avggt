#!/usr/bin/env python3
"""
Simplified test script for VGGT-Audio with COLMAP data.
Focuses on basic functionality without complex audio propagation.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from vggt.models.vggt import VGGT
from vggt.models.vggt_audio import VGGTAudio
from vggt.utils.colmap_loader import COLMAPAudioDataset
from vggt.utils.audio_utils import AudioFeatureExtractor


def visualize_acoustic_properties(predictions, batch, output_dir, frame_idx=0):
    """Simple visualization of acoustic properties."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    img = batch['images'][0, frame_idx].permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')
    
    # Depth
    if 'depth' in predictions:
        depth = predictions['depth'][0, frame_idx, :, :, 0].cpu().numpy()
        im = axes[0, 1].imshow(depth, cmap='viridis')
        axes[0, 1].set_title("Depth Map")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
    
    # Acoustic properties
    if 'acoustic_properties' in predictions:
        props = predictions['acoustic_properties']
        
        # Average absorption
        absorption = props['absorption'][0, frame_idx].mean(dim=-1).cpu().numpy()
        im = axes[0, 2].imshow(absorption, cmap='RdYlBu', vmin=0, vmax=1)
        axes[0, 2].set_title("Absorption (avg)")
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Reflection
        reflection = props['reflection'][0, frame_idx, :, :, 0].cpu().numpy()
        im = axes[1, 0].imshow(reflection, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1, 0].set_title("Reflection")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Scattering
        scattering = props['scattering'][0, frame_idx, :, :, 0].cpu().numpy()
        im = axes[1, 1].imshow(scattering, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title("Scattering")
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
    
    # Confidence
    if 'acoustic_conf' in predictions:
        conf = predictions['acoustic_conf'][0, frame_idx].cpu().numpy()
        im = axes[1, 2].imshow(conf, cmap='hot')
        axes[1, 2].set_title("Confidence")
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_dir / "acoustic_properties.png", dpi=150)
    plt.close()
    print(f"Saved visualization to {output_dir / 'acoustic_properties.png'}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Load dataset
    print(f"\nLoading COLMAP dataset from {args.scene_dir}")
    dataset = COLMAPAudioDataset(
        scene_dir=args.scene_dir,
        source_position=args.source_position,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )
    
    # Prepare batch
    print("Preparing data batch...")
    batch = dataset.prepare_vggt_batch()
    
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    print(f"Loaded {batch['images'].shape[1]} frames")
    print(f"Image shape: {batch['images'].shape}")
    print(f"Audio sample rate: {batch['sample_rate']} Hz")
    
    if args.use_audio:
        print("\nInitializing VGGT-Audio model...")
        model = VGGTAudio(
            enable_acoustic=True,
            audio_feature_dim=64,  # Match the n_mels below
            freq_bands=8,
        ).to(device)
        
        # Extract audio features
        print("Extracting audio features...")
        samples_per_frame = batch['binaural_audio'].shape[-1]
        n_fft = min(512, samples_per_frame // 2)
        hop_length = n_fft // 4
        
        audio_extractor = AudioFeatureExtractor(
            feature_type="mel_spectrogram",
            sample_rate=batch['sample_rate'],
            n_mels=64,  # Reduced for short segments
            n_fft=n_fft,
            hop_length=hop_length,
        ).to(device)
        
        audio_features = []
        for i in range(batch['binaural_audio'].shape[1]):
            frame_audio = batch['binaural_audio'][0, i]
            features = audio_extractor.extract_features(frame_audio)
            audio_features.append(features)
        audio_features = torch.stack(audio_features).to(device).unsqueeze(0)
        
    else:
        print("\nInitializing standard VGGT model...")
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        audio_features = None
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            if args.use_audio:
                predictions = model(batch['images'], audio_features=audio_features)
            else:
                predictions = model(batch['images'])
    
    print("\nPredictions completed!")
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            print(f"  {key}: {predictions[key].shape}")
        elif isinstance(predictions[key], dict):
            print(f"  {key}:")
            for k, v in predictions[key].items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
    
    # Analyze acoustic properties
    if 'acoustic_properties' in predictions:
        props = predictions['acoustic_properties']
        print("\nAcoustic property statistics:")
        print(f"  Average absorption: {props['absorption'].mean().item():.3f}")
        print(f"  Average reflection: {props['reflection'].mean().item():.3f}")
        print(f"  Average scattering: {props['scattering'].mean().item():.3f}")
        
        # Frequency-dependent absorption
        freq_absorption = props['absorption'][0, 0].mean(dim=(0, 1)).cpu().numpy()
        if 'freq_centers' in props:
            freq_centers = props['freq_centers'].cpu().numpy()
            print("\n  Frequency-dependent absorption:")
            for i, (freq, abs_val) in enumerate(zip(freq_centers[:len(freq_absorption)], freq_absorption)):
                print(f"    {int(freq)} Hz: {abs_val:.3f}")
    
    # Visualize
    if args.visualize and args.use_audio:
        print("\nVisualizing acoustic properties...")
        visualize_acoustic_properties(predictions, batch, Path(args.output_dir))
    
    # Save results
    if args.save_results:
        output_path = Path(args.output_dir) / "results.pt"
        torch.save({
            'predictions': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in predictions.items()},
            'source_position': batch['source_position'].cpu(),
            'camera_positions': batch['camera_positions'].cpu(),
        }, output_path)
        print(f"\nSaved results to {output_path}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple VGGT-Audio test")
    parser.add_argument("--scene_dir", type=str, default="data/1")
    parser.add_argument("--source_position", type=float, nargs=3, default=[0.06, 0.13, -0.45])
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--frame_skip", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="output_colmap")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    
    args = parser.parse_args()
    main(args)