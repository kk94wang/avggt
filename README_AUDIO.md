# VGGT-Audio: 3D Acoustic Environment Reconstruction

This extension adds acoustic property prediction and audio propagation simulation to VGGT, enabling the reconstruction of both visual and acoustic properties of 3D environments.

## Overview

VGGT-Audio extends the base VGGT model to:
1. **Predict acoustic material properties** (absorption, reflection, scattering) for each point in 3D space
2. **Simulate realistic audio propagation** including reverb, occlusion, and distance effects
3. **Generate spatialized audio** for novel viewpoints and sound source positions

## Key Features

- **Joint audio-visual learning**: Leverages correlations between visual appearance and acoustic properties
- **Physics-based audio rendering**: Simulates sound propagation using predicted material properties
- **Spatial audio synthesis**: Generates binaural audio for immersive experiences
- **Real-time capable**: Fast inference for interactive applications

## Architecture

### New Components

1. **AudioAggregator**: Extended aggregator with audio tokens for cross-modal attention
2. **AcousticHead**: Predicts frequency-dependent acoustic properties
3. **AcousticPropagation**: Simulates sound propagation through the environment
4. **Audio-visual losses**: Multi-task learning objectives

### Model Structure

```
Video + Audio → VGGT-Audio → {
    Camera poses,
    Depth maps,
    3D points,
    Acoustic properties {
        Absorption (frequency-dependent),
        Reflection coefficients,
        Scattering (diffuse vs specular),
        Acoustic impedance
    }
}
```

## Installation

```bash
# Install additional audio dependencies
pip install torchaudio librosa soundfile

# Optional: for advanced audio processing
pip install pyroomacoustics scipy
```

## Usage

### Basic Inference

```python
from vggt.models.vggt_audio import VGGTAudio
from vggt.utils.audio_utils import AudioFeatureExtractor

# Load model
model = VGGTAudio.from_pretrained_vggt("facebook/VGGT-1B", enable_acoustic=True)

# Extract audio features
audio_extractor = AudioFeatureExtractor(feature_type="mel_spectrogram")
audio_features = audio_extractor.extract_features(waveform)

# Run inference
predictions = model(images, audio_features=audio_features)

# Access acoustic properties
acoustic_props = predictions["acoustic_properties"]
absorption = acoustic_props["absorption"]  # [B, S, H, W, freq_bands]
reflection = acoustic_props["reflection"]  # [B, S, H, W, 1]
```

### Audio Propagation Simulation

```python
from vggt.utils.acoustic_propagation import AcousticPropagation

# Initialize propagation module
propagation = AcousticPropagation(max_order_reflections=2)

# Simulate audio at listener position
result = propagation(
    source_audio=source_waveform,
    source_position=torch.tensor([1.0, 1.5, 2.0]),
    listener_position=torch.tensor([4.0, 1.5, 3.0]),
    listener_orientation=torch.tensor([1, 0, 0, 0]),  # quaternion
    world_points=predictions["world_points"],
    acoustic_properties=predictions["acoustic_properties"],
)

binaural_audio = result["binaural_audio"]  # Spatialized stereo audio
```

### Demo Script

```bash
# Run demo on video with audio
python demo_audio.py path/to/video.mp4 \
    --visualize \
    --simulate_audio \
    --source_audio path/to/source.wav \
    --output_dir results/

# Run on image sequence with separate audio
python demo_audio.py path/to/images/ \
    --audio_path path/to/audio.wav \
    --pretrained_vggt facebook/VGGT-1B \
    --visualize
```

## Training

### Data Format

Training data should include synchronized audio-visual captures:

```python
{
    "images": [frame1, frame2, ..., frameN],
    "audio": {
        "waveform": audio_array,
        "sample_rate": 48000,
        "source_position": [x, y, z],  # 3D position of sound source
    },
    "camera_params": {...},  # Standard VGGT format
    "acoustic_measurements": {  # Optional ground truth
        "rt60": [0.3, 0.35, 0.4, ...],  # Per frequency band
        "absorption_coefficients": {...},
    }
}
```

### Training Script

```python
from training.acoustic_loss import AcousticLoss, AcousticConsistencyLoss

# Add acoustic losses to training
acoustic_loss = AcousticLoss(
    rendering_weight=1.0,
    physical_weight=0.1,
    perceptual_weight=0.5,
)

consistency_loss = AcousticConsistencyLoss(consistency_weight=0.1)

# In training loop
losses = acoustic_loss(
    rendered_audio=propagation_output["binaural_audio"],
    target_audio=batch["recorded_audio"],
    acoustic_properties=predictions["acoustic_properties"],
)
```

## Applications

1. **Virtual Reality**: Realistic spatial audio for immersive experiences
2. **Architectural Acoustics**: Preview acoustic properties before construction
3. **Game Development**: Automatic acoustic material assignment
4. **Robotics**: Audio-based navigation and scene understanding
5. **Film Production**: Virtual location scouting with accurate acoustics

## Limitations

- Requires synchronized audio-visual training data
- Simplified acoustic model (ray-based, not wave-based)
- Limited to static scenes (no moving sound sources during capture)
- Best results with controlled acoustic environments

## Future Work

- Dynamic sound source tracking
- Wave-based acoustic simulation for low frequencies
- Multi-source separation and localization
- Real-time optimization for mobile devices

## Citation

If you use VGGT-Audio in your research, please cite:

```bibtex
@article{vggt-audio2024,
  title={VGGT-Audio: Joint Visual-Acoustic 3D Scene Reconstruction},
  author={...},
  journal={...},
  year={2024}
}
```

## License

Same as VGGT - see LICENSE file in the root directory.