#!/usr/bin/env python3
"""
Script to prepare Unreal Engine exported data for VGGT-Audio training.
Converts Unreal exports to the required format and validates data quality.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import shutil
from typing import Dict, List
import wave
import cv2
from tqdm import tqdm


def validate_transform_matrix(matrix: List[List[float]]) -> bool:
    """Validate a 4x4 transformation matrix."""
    matrix = np.array(matrix)
    if matrix.shape != (4, 4):
        return False
    # Check if last row is [0, 0, 0, 1]
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        return False
    # Check if rotation part is orthogonal
    rotation = matrix[:3, :3]
    should_be_identity = rotation @ rotation.T
    return np.allclose(should_be_identity, np.eye(3), atol=1e-5)


def unreal_to_vggt_transform(unreal_matrix: np.ndarray) -> np.ndarray:
    """
    Convert Unreal Engine coordinate system to VGGT format.
    Unreal: Z-up, left-handed
    VGGT: Y-up, right-handed (OpenCV convention)
    """
    # Coordinate system conversion matrix
    conversion = np.array([
        [1, 0, 0, 0],   # X stays X
        [0, 0, -1, 0],  # Z becomes -Y
        [0, 1, 0, 0],   # Y becomes Z
        [0, 0, 0, 1]
    ])
    
    # Convert from Unreal to VGGT coordinates
    vggt_matrix = conversion @ unreal_matrix @ np.linalg.inv(conversion)
    
    return vggt_matrix


def process_scene(scene_dir: Path, output_dir: Path) -> Dict:
    """Process a single scene from Unreal Engine export."""
    
    scene_name = scene_dir.name
    output_scene = output_dir / scene_name
    output_scene.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing scene: {scene_name}")
    
    # 1. Process images
    frames_dir = scene_dir / "frames"
    output_frames = output_scene / "frames"
    output_frames.mkdir(exist_ok=True)
    
    image_files = sorted(frames_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No images found in {frames_dir}")
    
    print(f"  Found {len(image_files)} frames")
    
    # Copy and validate images
    for img_path in tqdm(image_files, desc="  Copying frames"):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load {img_path}")
        
        # Optionally resize to standard resolution
        if img.shape[:2] != (270, 480):
            img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_LANCZOS4)
        
        output_path = output_frames / img_path.name
        cv2.imwrite(str(output_path), img)
    
    # 2. Process camera parameters
    camera_file = scene_dir / "cameras.json"
    if not camera_file.exists():
        raise ValueError(f"Missing camera parameters: {camera_file}")
    
    with open(camera_file) as f:
        camera_data = json.load(f)
    
    # Convert to VGGT format
    transforms = {
        "camera_angle_x": camera_data.get("fov_x", 1.4968),
        "fl_x": camera_data.get("focal_x", 258.43),
        "fl_y": camera_data.get("focal_y", 258.43),
        "cx": camera_data.get("principal_x", 240.0),
        "cy": camera_data.get("principal_y", 135.0),
        "w": camera_data.get("width", 480),
        "h": camera_data.get("height", 270),
        "frames": []
    }
    
    # Process each frame's camera
    for i, frame_data in enumerate(camera_data["frames"]):
        # Convert Unreal transform to VGGT format
        unreal_matrix = np.array(frame_data["transform_matrix"])
        vggt_matrix = unreal_to_vggt_transform(unreal_matrix)
        
        if not validate_transform_matrix(vggt_matrix):
            print(f"  Warning: Invalid transform matrix for frame {i}")
        
        transforms["frames"].append({
            "file_path": f"frames/{i+1:05d}.png",
            "transform_matrix": vggt_matrix.tolist()
        })
    
    # Save transforms
    with open(output_scene / "transforms_train.json", "w") as f:
        json.dump(transforms, f, indent=2)
    
    # 3. Process audio
    audio_dir = scene_dir / "audio"
    if audio_dir.exists():
        print("  Processing audio...")
        
        # Copy source audio
        source_audio = audio_dir / "source.wav"
        if source_audio.exists():
            shutil.copy(source_audio, output_scene / "source_syn_re.wav")
        
        # Process binaural audio
        binaural_audio = audio_dir / "binaural.wav"
        if binaural_audio.exists():
            # Validate audio length matches video
            with wave.open(str(binaural_audio), 'rb') as wav:
                n_frames_audio = wav.getnframes()
                sample_rate = wav.getframerate()
                duration = n_frames_audio / sample_rate
                expected_duration = len(image_files) / 30.0  # Assuming 30 FPS
                
                if abs(duration - expected_duration) > 0.1:
                    print(f"  Warning: Audio duration ({duration:.2f}s) doesn't match "
                          f"video duration ({expected_duration:.2f}s)")
            
            shutil.copy(binaural_audio, output_scene / "binaural_syn_re.wav")
    
    # 4. Process acoustic properties
    materials_file = scene_dir / "materials.json"
    if materials_file.exists():
        with open(materials_file) as f:
            materials = json.load(f)
        
        # Convert to VGGT format
        acoustic_props = {}
        for mat_name, mat_data in materials.items():
            # Ensure we have 8 frequency bands
            absorption = mat_data.get("absorption", [0.1] * 8)
            if len(absorption) < 8:
                absorption.extend([absorption[-1]] * (8 - len(absorption)))
            elif len(absorption) > 8:
                absorption = absorption[:8]
            
            acoustic_props[mat_name] = {
                "absorption": absorption,
                "scattering": mat_data.get("scattering", 0.1),
                "reflection": mat_data.get("reflection", 0.9),
                "impedance": mat_data.get("impedance", [1.0, 0.0])
            }
        
        with open(output_scene / "acoustic_properties.json", "w") as f:
            json.dump(acoustic_props, f, indent=2)
    
    # 5. Create metadata
    metadata = {
        "scene_id": scene_name,
        "num_frames": len(image_files),
        "has_audio": (audio_dir.exists() and binaural_audio.exists()),
        "has_acoustic_properties": materials_file.exists(),
        "source_position": camera_data.get("audio_source_position", [0, 0, 0])
    }
    
    with open(output_scene / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def create_dataset_splits(
    processed_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> None:
    """Create train/val/test splits."""
    
    scenes = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    np.random.shuffle(scenes)
    
    n_total = len(scenes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train:n_train + n_val]
    test_scenes = scenes[n_train + n_val:]
    
    # Save splits
    with open(processed_dir / "train_scenes.json", "w") as f:
        json.dump(train_scenes, f, indent=2)
    
    with open(processed_dir / "val_scenes.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
    
    with open(processed_dir / "test_scenes.json", "w") as f:
        json.dump(test_scenes, f, indent=2)
    
    print(f"\nDataset splits created:")
    print(f"  Train: {len(train_scenes)} scenes")
    print(f"  Val: {len(val_scenes)} scenes")
    print(f"  Test: {len(test_scenes)} scenes")


def validate_dataset(processed_dir: Path) -> None:
    """Validate the processed dataset."""
    
    print("\nValidating dataset...")
    
    issues = []
    scene_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    for scene_dir in tqdm(scene_dirs, desc="Validating scenes"):
        # Check required files
        required_files = [
            "transforms_train.json",
            "metadata.json"
        ]
        
        for req_file in required_files:
            if not (scene_dir / req_file).exists():
                issues.append(f"{scene_dir.name}: Missing {req_file}")
        
        # Check frames
        frames_dir = scene_dir / "frames"
        if not frames_dir.exists():
            issues.append(f"{scene_dir.name}: Missing frames directory")
        else:
            n_frames = len(list(frames_dir.glob("*.png")))
            if n_frames < 10:
                issues.append(f"{scene_dir.name}: Only {n_frames} frames (minimum 10 recommended)")
        
        # Check audio
        if (scene_dir / "binaural_syn_re.wav").exists():
            # Validate audio file
            try:
                with wave.open(str(scene_dir / "binaural_syn_re.wav"), 'rb') as wav:
                    if wav.getnchannels() != 2:
                        issues.append(f"{scene_dir.name}: Binaural audio is not stereo")
                    if wav.getframerate() != 48000:
                        issues.append(f"{scene_dir.name}: Audio sample rate is {wav.getframerate()}, expected 48000")
            except Exception as e:
                issues.append(f"{scene_dir.name}: Error reading audio: {e}")
    
    if issues:
        print("\nValidation issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✓ Dataset validation passed!")


def main():
    parser = argparse.ArgumentParser(description="Prepare Unreal Engine data for VGGT-Audio")
    parser.add_argument("--input_dir", type=Path, required=True,
                       help="Directory containing Unreal Engine exports")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Ratio of data for validation")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate existing processed data")
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_dataset(args.output_dir)
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all scenes
    scene_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(scene_dirs)} scenes to process")
    
    metadata_list = []
    for scene_dir in scene_dirs:
        try:
            metadata = process_scene(scene_dir, args.output_dir)
            metadata_list.append(metadata)
        except Exception as e:
            print(f"Error processing {scene_dir.name}: {e}")
            continue
    
    # Create dataset splits
    create_dataset_splits(
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio
    )
    
    # Validate the processed dataset
    validate_dataset(args.output_dir)
    
    # Save overall statistics
    stats = {
        "total_scenes": len(metadata_list),
        "scenes_with_audio": sum(1 for m in metadata_list if m["has_audio"]),
        "scenes_with_properties": sum(1 for m in metadata_list if m["has_acoustic_properties"]),
        "total_frames": sum(m["num_frames"] for m in metadata_list)
    }
    
    with open(args.output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Processing complete!")
    print(f"  Total scenes: {stats['total_scenes']}")
    print(f"  Scenes with audio: {stats['scenes_with_audio']}")
    print(f"  Total frames: {stats['total_frames']}")


if __name__ == "__main__":
    main()