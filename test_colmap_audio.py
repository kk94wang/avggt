#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for VGGT-Audio with COLMAP format data.
Specifically designed to work with the data/1 sample dataset.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torchaudio

from vggt.models.vggt import VGGT
from vggt.models.vggt_audio import VGGTAudio
from vggt.utils.colmap_loader import COLMAPAudioDataset
from vggt.utils.audio_utils import AudioFeatureExtractor
from vggt.utils.acoustic_propagation import AcousticPropagation
from vggt.utils.pose_enc import pose_encoding_to_extri_intri as pose_encoding_to_camera


def visualize_results(
    images: torch.Tensor,
    predictions: dict,
    ground_truth: dict,
    output_dir: Path,
    max_frames: int = 5
):
    """Visualize reconstruction results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select frames to visualize
    num_frames = min(images.shape[1], max_frames)
    frame_indices = np.linspace(0, images.shape[1]-1, num_frames, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(num_frames, 4, figsize=(16, 4*num_frames))
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    for idx, frame_idx in enumerate(frame_indices):
        # Original image
        img = images[0, frame_idx].permute(1, 2, 0).cpu().numpy()
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Frame {frame_idx}")
        axes[idx, 0].axis('off')
        
        # Depth map
        if "depth" in predictions:
            depth = predictions["depth"][0, frame_idx, :, :, 0].cpu().numpy()
            im = axes[idx, 1].imshow(depth, cmap='viridis')
            axes[idx, 1].set_title("Predicted Depth")
            axes[idx, 1].axis('off')
            plt.colorbar(im, ax=axes[idx, 1], fraction=0.046)
        
        # Acoustic properties (average absorption)
        if "acoustic_properties" in predictions:
            absorption = predictions["acoustic_properties"]["absorption"][0, frame_idx].mean(dim=-1).cpu().numpy()
            im = axes[idx, 2].imshow(absorption, cmap='RdYlBu', vmin=0, vmax=1)
            axes[idx, 2].set_title("Acoustic Absorption")
            axes[idx, 2].axis('off')
            plt.colorbar(im, ax=axes[idx, 2], fraction=0.046)
            
        # Confidence map
        if "acoustic_conf" in predictions:
            conf = predictions["acoustic_conf"][0, frame_idx].cpu().numpy()
            im = axes[idx, 3].imshow(conf, cmap='hot')
            axes[idx, 3].set_title("Confidence")
            axes[idx, 3].axis('off')
            plt.colorbar(im, ax=axes[idx, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_results.png", dpi=150)
    plt.close()
    
    # Plot camera trajectory
    if "pose_enc" in predictions:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Predicted camera positions
        pred_poses = predictions["pose_enc"][0].cpu()
        pred_positions = []
        for i in range(pred_poses.shape[0]):
            extri, intri = pose_encoding_to_camera(
                pred_poses[i:i+1],
                (images.shape[-2], images.shape[-1])
            )
            # Extract camera position from extrinsic matrix
            # Camera position is -R^T @ t where extri = [R|t]
            R = extri[0, :3, :3]
            t = extri[0, :3, 3]
            cam_pos = -R.T @ t
            pred_positions.append(cam_pos)
        pred_positions = torch.stack(pred_positions)
        
        # Ground truth camera positions
        gt_positions = ground_truth["camera_positions"][0].cpu()
        
        # Plot trajectories
        ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
                'b-', label='Predicted', linewidth=2)
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                'r--', label='Ground Truth', linewidth=2)
        
        # Plot audio source
        source_pos = ground_truth["source_position"][0].cpu()
        ax.scatter(source_pos[0], source_pos[1], source_pos[2], 
                  c='green', s=200, marker='*', label='Audio Source')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('Camera Trajectory and Audio Source')
        
        plt.savefig(output_dir / "camera_trajectory.png", dpi=150)
        plt.close()


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
    
    # Initialize model
    if args.use_audio:
        print("\nInitializing VGGT-Audio model...")
        model = VGGTAudio(
            enable_acoustic=True,
            audio_feature_dim=128,  # Using mel spectrogram features
            freq_bands=8,
        ).to(device)
        
        # Extract audio features
        print("Extracting audio features...")
        # Adjust FFT parameters for short audio segments
        samples_per_frame = batch['binaural_audio'].shape[-1]
        n_fft = min(2048, samples_per_frame // 2)  # Ensure n_fft is smaller than segment length
        hop_length = n_fft // 4
        
        audio_extractor = AudioFeatureExtractor(
            feature_type="mel_spectrogram",
            sample_rate=batch['sample_rate'],
            n_mels=128,
            n_fft=n_fft,
            hop_length=hop_length,
        ).to(device)
        
        # Process audio features for each frame
        audio_features = []
        for i in range(batch['binaural_audio'].shape[1]):
            frame_audio = batch['binaural_audio'][0, i]  # [2, samples]
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
    
    # Evaluate camera pose accuracy
    if "pose_enc" in predictions:
        pose_error = torch.norm(
            predictions["pose_enc"] - batch["pose_encodings"], 
            dim=-1
        ).mean()
        print(f"\nAverage pose encoding error: {pose_error:.4f}")
    
    # Visualize results
    if args.visualize:
        print("\nVisualizing results...")
        visualize_results(
            batch['images'],
            predictions,
            batch,
            Path(args.output_dir),
            max_frames=args.vis_frames
        )
    
    # Simulate audio propagation
    if args.use_audio and args.simulate_propagation:
        print("\nSimulating audio propagation...")
        
        # Initialize propagation module
        propagation = AcousticPropagation(
            max_order_reflections=2,
            freq_bands=8,
        ).to(device)
        
        # Simulate for a few listener positions
        listener_positions = batch['camera_positions'][0, ::10]  # Every 10th camera
        
        results = []
        for i, listener_pos in enumerate(listener_positions):
            result = propagation(
                batch['source_audio'].to(device),
                batch['source_position'],
                listener_pos.unsqueeze(0),
                torch.tensor([[1, 0, 0, 0]]).to(device),  # Identity orientation
                predictions['world_points'][0, i].unsqueeze(0),
                {k: v[0, i].unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 else v 
                 for k, v in predictions['acoustic_properties'].items()},
                predictions['depth'][0, i, :, :, 0].unsqueeze(0) if 'depth' in predictions else None,
            )
            results.append(result['binaural_audio'])
        
        # Save simulated audio
        output_audio = torch.cat(results, dim=0)
        output_path = Path(args.output_dir) / "simulated_binaural.wav"
        torchaudio.save(
            str(output_path),
            output_audio[0].cpu(),  # Save first simulation
            batch['sample_rate'],
        )
        print(f"Saved simulated audio to {output_path}")
        
    # Save predictions
    if args.save_predictions:
        output_path = Path(args.output_dir) / "predictions.pt"
        torch.save({
            'predictions': {k: v.cpu() for k, v in predictions.items() if isinstance(v, torch.Tensor)},
            'ground_truth': {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)},
        }, output_path)
        print(f"\nSaved predictions to {output_path}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VGGT-Audio with COLMAP data")
    
    # For the sample data in data/1
    parser.add_argument("--scene_dir", type=str, default="data/1",
                        help="Path to scene directory")
    parser.add_argument("--source_position", type=float, nargs=3,
                        default=[0.06, 0.13, -0.45],
                        help="3D position of audio source")
    
    # Model options
    parser.add_argument("--use_audio", action="store_true",
                        help="Use VGGT-Audio model instead of standard VGGT")
    parser.add_argument("--pretrained", type=str, default="facebook/VGGT-1B",
                        help="Pretrained model to use")
    
    # Data options
    parser.add_argument("--max_frames", type=int, default=50,
                        help="Maximum number of frames to process")
    parser.add_argument("--frame_skip", type=int, default=10,
                        help="Process every nth frame")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="output_colmap",
                        help="Output directory for results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--vis_frames", type=int, default=5,
                        help="Number of frames to visualize")
    parser.add_argument("--simulate_propagation", action="store_true",
                        help="Simulate audio propagation")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predictions to file")
    
    args = parser.parse_args()
    main(args)