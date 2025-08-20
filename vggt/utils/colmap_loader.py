# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torchaudio
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import extri_intri_to_pose_encoding as camera_to_pose_encoding
from vggt.utils.pose_enc import pose_encoding_to_extri_intri as pose_encoding_to_camera


def load_colmap_transforms(json_path: str) -> Dict:
    """Load COLMAP/NeRF format transforms file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def parse_transform_matrix(matrix: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse 4x4 transform matrix into rotation and translation.
    
    Args:
        matrix: 4x4 camera-to-world transformation matrix
        
    Returns:
        Tuple of:
            - extrinsic: 3x4 camera extrinsic matrix (world-to-camera)
            - translation: 3D translation vector
    """
    matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Extract rotation and translation (camera-to-world)
    R_c2w = matrix[:3, :3]
    t_c2w = matrix[:3, 3]
    
    # Convert to world-to-camera (extrinsic)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    
    # Create 3x4 extrinsic matrix
    extrinsic = torch.zeros(3, 4)
    extrinsic[:3, :3] = R_w2c
    extrinsic[:3, 3] = t_w2c
    
    return extrinsic, t_c2w


def create_intrinsic_matrix(
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    w: float,
    h: float
) -> torch.Tensor:
    """Create 3x3 intrinsic matrix from camera parameters."""
    K = torch.tensor([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    return K


class COLMAPAudioDataset:
    """
    Load COLMAP-format data with synchronized audio for VGGT-Audio.
    
    Expected directory structure:
    scene_dir/
        frames/
            00001.png
            00002.png
            ...
        transforms_train.json
        source_syn_re.wav      # Source audio (mono)
        binaural_syn_re.wav    # Binaural recordings at camera positions
    """
    
    def __init__(
        self,
        scene_dir: str,
        transforms_file: str = "transforms_train.json",
        source_audio_file: str = "source_syn_re.wav",
        binaural_audio_file: str = "binaural_syn_re.wav",
        source_position: Optional[List[float]] = None,
        img_size: int = 518,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
    ):
        """
        Args:
            scene_dir: Path to scene directory
            transforms_file: Name of transforms JSON file
            source_audio_file: Name of source audio file
            binaural_audio_file: Name of binaural audio file
            source_position: 3D position of audio source [x, y, z]
            img_size: Target image size for VGGT
            max_frames: Maximum number of frames to load
            frame_skip: Load every nth frame
        """
        self.scene_dir = Path(scene_dir)
        self.img_size = img_size
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        
        # Load transforms
        transforms_path = self.scene_dir / transforms_file
        self.transforms = load_colmap_transforms(str(transforms_path))
        
        # Extract camera parameters
        self.camera_angle_x = self.transforms.get("camera_angle_x", 0)
        self.fl_x = self.transforms["fl_x"]
        self.fl_y = self.transforms["fl_y"]
        self.cx = self.transforms["cx"]
        self.cy = self.transforms["cy"]
        self.w = self.transforms["w"]
        self.h = self.transforms["h"]
        
        # Create intrinsic matrix
        self.intrinsic = create_intrinsic_matrix(
            self.fl_x, self.fl_y, self.cx, self.cy, self.w, self.h
        )
        
        # Load audio files
        self.source_audio_path = self.scene_dir / source_audio_file
        self.binaural_audio_path = self.scene_dir / binaural_audio_file
        
        # Source position
        self.source_position = torch.tensor(
            source_position if source_position else [0.0, 0.0, 0.0],
            dtype=torch.float32
        )
        
        # Filter frames
        self.frames = self.transforms["frames"][::frame_skip]
        if max_frames:
            self.frames = self.frames[:max_frames]
            
    def load_images(self) -> torch.Tensor:
        """
        Load and preprocess all images.
        
        Returns:
            Images tensor [N, 3, H, W]
        """
        # Collect image paths
        image_paths = []
        for frame in self.frames:
            img_path = str(self.scene_dir / frame["file_path"])
            image_paths.append(img_path)
        
        # Use VGGT's built-in image loading and preprocessing
        # This function loads images and preprocesses them to 518x518 by default
        # It returns a tuple of (images, positions)
        images, _ = load_and_preprocess_images_square(image_paths, target_size=self.img_size)
        
        return images
    
    def load_camera_poses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load camera extrinsics and intrinsics.
        
        Returns:
            Tuple of:
                - extrinsics: [N, 3, 4] world-to-camera transformations
                - intrinsics: [N, 3, 3] camera intrinsic matrices
        """
        extrinsics = []
        camera_positions = []
        
        for frame in self.frames:
            extrinsic, cam_pos = parse_transform_matrix(frame["transform_matrix"])
            extrinsics.append(extrinsic)
            camera_positions.append(cam_pos)
            
        extrinsics = torch.stack(extrinsics)
        
        # Replicate intrinsics for all frames
        intrinsics = self.intrinsic.unsqueeze(0).repeat(len(self.frames), 1, 1)
        
        # Adjust intrinsics for image resizing
        scale_x = self.img_size / self.w
        scale_y = self.img_size / self.h
        intrinsics[:, 0, 0] *= scale_x  # fl_x
        intrinsics[:, 1, 1] *= scale_y  # fl_y
        intrinsics[:, 0, 2] *= scale_x  # cx
        intrinsics[:, 1, 2] *= scale_y  # cy
        
        self.camera_positions = torch.stack(camera_positions)
        
        return extrinsics, intrinsics
    
    def load_audio(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Load source and binaural audio.
        
        Returns:
            Tuple of:
                - source_audio: Mono source audio [1, samples]
                - binaural_audio: Binaural recordings [2, samples]
                - sample_rate: Audio sample rate
        """
        # Load source audio (mono)
        source_audio, sr_source = torchaudio.load(str(self.source_audio_path))
        if source_audio.shape[0] > 1:
            source_audio = source_audio.mean(dim=0, keepdim=True)
            
        # Load binaural audio
        binaural_audio, sr_binaural = torchaudio.load(str(self.binaural_audio_path))
        
        # Ensure same sample rate
        if sr_source != sr_binaural:
            resampler = torchaudio.transforms.Resample(sr_binaural, sr_source)
            binaural_audio = resampler(binaural_audio)
            
        return source_audio, binaural_audio, sr_source
    
    def get_audio_segments(
        self,
        binaural_audio: torch.Tensor,
        sample_rate: int,
        fps: float = 30.0
    ) -> torch.Tensor:
        """
        Split binaural audio into segments corresponding to each frame.
        
        Args:
            binaural_audio: Full binaural audio [2, total_samples]
            sample_rate: Audio sample rate
            fps: Video frame rate
            
        Returns:
            Audio segments [N, 2, samples_per_frame]
        """
        samples_per_frame = int(sample_rate / fps)
        num_frames = len(self.frames)
        
        segments = []
        for i in range(num_frames):
            start_idx = i * samples_per_frame
            end_idx = (i + 1) * samples_per_frame
            
            if end_idx <= binaural_audio.shape[1]:
                segment = binaural_audio[:, start_idx:end_idx]
            else:
                # Pad last segment if necessary
                segment = binaural_audio[:, start_idx:]
                padding = end_idx - binaural_audio.shape[1]
                segment = torch.nn.functional.pad(segment, (0, padding))
                
            segments.append(segment)
            
        return torch.stack(segments)
    
    def prepare_vggt_batch(self) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch in VGGT format.
        
        Returns:
            Dictionary containing all necessary data for VGGT-Audio
        """
        # Load all data
        images = self.load_images()
        extrinsics, intrinsics = self.load_camera_poses()
        source_audio, binaural_audio, sample_rate = self.load_audio()
        
        # Get audio segments for each frame
        audio_segments = self.get_audio_segments(binaural_audio, sample_rate)
        
        # Convert camera parameters to VGGT pose encoding
        # The function expects 4D tensors: [batch, sequence, 3, 4/3]
        # Add batch dimension to make it [1, N, 3, 4/3]
        pose_encodings = camera_to_pose_encoding(
            extrinsics.unsqueeze(0),  # [1, N, 3, 4]
            intrinsics.unsqueeze(0),  # [1, N, 3, 3]
            (self.img_size, self.img_size),
        )
        pose_encodings = pose_encodings.squeeze(0)  # Remove batch dimension [N, 9]
        
        return {
            "images": images.unsqueeze(0),  # Add batch dimension [1, N, 3, H, W]
            "extrinsics": extrinsics.unsqueeze(0),  # [1, N, 3, 4]
            "intrinsics": intrinsics.unsqueeze(0),  # [1, N, 3, 3]
            "pose_encodings": pose_encodings.unsqueeze(0),  # [1, N, 9]
            "source_audio": source_audio,  # [1, samples]
            "binaural_audio": audio_segments.unsqueeze(0),  # [1, N, 2, samples_per_frame]
            "source_position": self.source_position.unsqueeze(0),  # [1, 3]
            "camera_positions": self.camera_positions.unsqueeze(0),  # [1, N, 3]
            "sample_rate": sample_rate,
        }