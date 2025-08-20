#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Demo script for VGGT-Audio: 3D scene reconstruction with acoustic properties.

This demo shows how to:
1. Load a video with audio from a moving camera
2. Reconstruct 3D geometry and acoustic properties
3. Simulate audio propagation for new source positions
4. Generate spatialized audio for novel viewpoints
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import torchaudio

from vggt.models.vggt_audio import VGGTAudio
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.audio_utils import AudioFeatureExtractor, load_audio_video_pair
from vggt.utils.acoustic_propagation import AcousticPropagation


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Initialize models
    print("Loading VGGT-Audio model...")
    model = VGGTAudio(
        enable_acoustic=True,
        audio_feature_dim=768,
        freq_bands=8,
    ).to(device)
    
    # Load from pretrained VGGT if available
    if args.pretrained_vggt:
        print(f"Loading pretrained weights from {args.pretrained_vggt}")
        model = VGGTAudio.from_pretrained_vggt(
            args.pretrained_vggt,
            enable_acoustic=True,
            freeze_visual=not args.train_visual,
        )
        model = model.to(device)
    
    # Initialize audio feature extractor
    audio_extractor = AudioFeatureExtractor(
        feature_type=args.audio_feature_type,
        sample_rate=args.sample_rate,
        n_mels=128,
    )
    
    # Initialize acoustic propagation module
    propagation = AcousticPropagation(
        max_order_reflections=2,
        freq_bands=8,
    ).to(device)
    
    # Load video and audio
    print(f"Loading video and audio from {args.input_path}")
    if args.input_path.suffix in ['.mp4', '.avi', '.mov']:
        # Load from video file
        video_frames, audio_waveform, sync_info = load_audio_video_pair(
            str(args.input_path),
            str(args.input_path),  # Audio from same file
            target_fps=args.fps,
            target_sample_rate=args.sample_rate,
        )
    else:
        # Load from image folder + separate audio
        image_paths = sorted(Path(args.input_path).glob("*.jpg"))
        video_frames = load_and_preprocess_images([str(p) for p in image_paths])
        audio_waveform, sample_rate = torchaudio.load(args.audio_path)
        if sample_rate != args.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, args.sample_rate)
            audio_waveform = resampler(audio_waveform)
    
    # Extract audio features
    print("Extracting audio features...")
    audio_features = audio_extractor.extract_features(
        audio_waveform,
        frame_timestamps=sync_info.get("frame_timestamps") if 'sync_info' in locals() else None,
    )
    
    # Prepare inputs
    video_frames = video_frames.to(device)
    audio_features = audio_features.to(device)
    
    # Run inference
    print("Running VGGT-Audio inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(
                video_frames,
                audio_features=audio_features,
            )
    
    # Extract predictions
    camera_poses = predictions["pose_enc"]
    depth_maps = predictions["depth"]
    world_points = predictions["world_points"]
    acoustic_properties = predictions["acoustic_properties"]
    acoustic_conf = predictions["acoustic_conf"]
    
    print("\nReconstruction complete!")
    print(f"Camera poses shape: {camera_poses.shape}")
    print(f"Depth maps shape: {depth_maps.shape}")
    print(f"World points shape: {world_points.shape}")
    print(f"Acoustic absorption shape: {acoustic_properties['absorption'].shape}")
    
    # Visualize acoustic properties
    if args.visualize:
        visualize_acoustic_properties(
            video_frames[0],
            depth_maps[0],
            acoustic_properties,
            acoustic_conf[0],
            args.output_dir,
        )
    
    # Simulate audio propagation for new source position
    if args.simulate_audio:
        print("\nSimulating audio propagation...")
        
        # Define new audio source position (example: center of scene)
        scene_center = world_points[0].mean(dim=(0, 1))
        source_position = scene_center + torch.tensor([0, 1.0, 0]).to(device)  # 1m above center
        
        # Define listener trajectory (example: circular path)
        num_positions = 36
        radius = 3.0
        angles = torch.linspace(0, 2 * np.pi, num_positions)
        listener_positions = torch.stack([
            scene_center[0] + radius * torch.cos(angles),
            scene_center[1] * torch.ones_like(angles),
            scene_center[2] + radius * torch.sin(angles),
        ], dim=1).to(device)
        
        # Load source audio
        if args.source_audio:
            source_audio, sr = torchaudio.load(args.source_audio)
            if sr != args.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, args.sample_rate)
                source_audio = resampler(source_audio)
            source_audio = source_audio.mean(dim=0).to(device)  # Convert to mono
        else:
            # Generate test tone
            duration = 2.0
            t = torch.linspace(0, duration, int(duration * args.sample_rate))
            source_audio = torch.sin(2 * np.pi * 440 * t).to(device)  # 440 Hz tone
        
        # Simulate for each listener position
        binaural_audio_list = []
        for i, listener_pos in enumerate(listener_positions):
            result = propagation(
                source_audio.unsqueeze(0),
                source_position.unsqueeze(0),
                listener_pos.unsqueeze(0),
                torch.tensor([[1, 0, 0, 0]]).to(device),  # Identity quaternion
                world_points[0].unsqueeze(0),
                {k: v[0].unsqueeze(0) for k, v in acoustic_properties.items()},
                depth_maps[0, 0].unsqueeze(0),
            )
            binaural_audio_list.append(result["binaural_audio"])
        
        # Save spatialized audio
        output_audio = torch.cat(binaural_audio_list, dim=0)
        output_path = Path(args.output_dir) / "spatialized_audio.wav"
        torchaudio.save(
            str(output_path),
            output_audio[0].cpu(),  # Save first position
            args.sample_rate,
        )
        print(f"Saved spatialized audio to {output_path}")
        
        # Save acoustic analysis
        save_acoustic_analysis(
            acoustic_properties,
            world_points[0],
            args.output_dir,
        )


def visualize_acoustic_properties(images, depth, acoustic_props, confidence, output_dir):
    """Visualize acoustic properties as heatmaps."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get first frame
    if images.dim() == 4:
        image = images[0]
    else:
        image = images
        
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Depth map
    depth_np = depth[0, 0].cpu().numpy()
    im = axes[0, 1].imshow(depth_np, cmap='viridis')
    axes[0, 1].set_title("Depth Map")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Average absorption
    absorption_avg = acoustic_props["absorption"][0].mean(dim=-1).cpu().numpy()
    im = axes[0, 2].imshow(absorption_avg, cmap='RdYlBu', vmin=0, vmax=1)
    axes[0, 2].set_title("Average Absorption")
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Reflection coefficient
    reflection = acoustic_props["reflection"][0, :, :, 0].cpu().numpy()
    im = axes[1, 0].imshow(reflection, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 0].set_title("Reflection Coefficient")
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Scattering coefficient
    scattering = acoustic_props["scattering"][0, :, :, 0].cpu().numpy()
    im = axes[1, 1].imshow(scattering, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title("Scattering (0=specular, 1=diffuse)")
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Confidence
    conf_np = confidence.cpu().numpy()
    im = axes[1, 2].imshow(conf_np, cmap='hot')
    axes[1, 2].set_title("Prediction Confidence")
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_dir / "acoustic_properties.png", dpi=150)
    plt.close()
    print(f"Saved acoustic property visualization to {output_dir / 'acoustic_properties.png'}")


def save_acoustic_analysis(acoustic_props, world_points, output_dir):
    """Save acoustic analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute room statistics
    room_min = world_points.min(dim=0)[0].min(dim=0)[0]
    room_max = world_points.max(dim=0)[0].max(dim=0)[0]
    room_size = room_max - room_min
    room_volume = room_size.prod().item()
    
    # Average acoustic properties
    avg_absorption = acoustic_props["absorption"].mean().item()
    avg_reflection = acoustic_props["reflection"].mean().item()
    avg_scattering = acoustic_props["scattering"].mean().item()
    
    # Frequency-dependent absorption
    freq_absorption = acoustic_props["absorption"].mean(dim=(0, 1, 2)).cpu().numpy()
    freq_centers = acoustic_props["freq_centers"].cpu().numpy()
    
    # Save analysis
    analysis = {
        "room_dimensions": room_size.cpu().numpy().tolist(),
        "room_volume": room_volume,
        "average_absorption": avg_absorption,
        "average_reflection": avg_reflection,
        "average_scattering": avg_scattering,
        "frequency_absorption": {
            f"{int(freq)}Hz": float(abs_val) 
            for freq, abs_val in zip(freq_centers, freq_absorption)
        }
    }
    
    import json
    with open(output_dir / "acoustic_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
        
    print(f"Saved acoustic analysis to {output_dir / 'acoustic_analysis.json'}")
    print(f"Room volume: {room_volume:.2f} m³")
    print(f"Average absorption: {avg_absorption:.3f}")
    print(f"Average reflection: {avg_reflection:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT-Audio Demo")
    parser.add_argument("input_path", type=Path, help="Path to video file or image folder")
    parser.add_argument("--audio_path", type=Path, help="Path to audio file (if separate from video)")
    parser.add_argument("--output_dir", type=Path, default="output_audio", help="Output directory")
    parser.add_argument("--pretrained_vggt", type=str, help="Pretrained VGGT model name")
    parser.add_argument("--audio_feature_type", choices=["mel_spectrogram", "mfcc", "wav2vec2"], 
                        default="mel_spectrogram", help="Type of audio features")
    parser.add_argument("--sample_rate", type=int, default=48000, help="Audio sample rate")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--visualize", action="store_true", help="Visualize acoustic properties")
    parser.add_argument("--simulate_audio", action="store_true", help="Simulate audio propagation")
    parser.add_argument("--source_audio", type=Path, help="Audio file for simulation source")
    parser.add_argument("--train_visual", action="store_true", help="Also train visual components")
    
    args = parser.parse_args()
    main(args)