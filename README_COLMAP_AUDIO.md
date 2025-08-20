# Using VGGT-Audio with COLMAP Format Data

This guide explains how to use VGGT-Audio with COLMAP/NeRF format datasets that include synchronized audio.

## Data Format

The expected directory structure for audio-visual COLMAP data:

```
scene_directory/
├── frames/                      # Directory containing images
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── transforms_train.json        # Camera parameters in NeRF format
├── transforms_val.json          # (optional) Validation set
├── source_syn_re.wav           # Source audio (mono)
├── binaural_syn_re.wav         # Binaural recordings at camera positions
└── position.json               # (optional) Audio source positions
```

### Camera Parameters Format (transforms_*.json)

```json
{
    "camera_angle_x": 1.4968,
    "fl_x": 258.433,          // Focal length x
    "fl_y": 258.433,          // Focal length y
    "cx": 240.0,              // Principal point x
    "cy": 135.0,              // Principal point y
    "w": 480.0,               // Image width
    "h": 270.0,               // Image height
    "frames": [
        {
            "file_path": "frames/00001.png",
            "transform_matrix": [  // 4x4 camera-to-world transformation
                [-0.0905, 0.0244, 0.9956, -1.1161],
                [0.9955, 0.0305, 0.0898, -1.5546],
                [-0.0282, 0.9992, -0.0271, -0.2133],
                [0.0, 0.0, 0.0, 1.0]
            ]
        },
        // ... more frames
    ]
}
```

### Audio Source Position Format (position.json)

```json
{
    "1": {"source_position": [0.06, 0.13, -0.45]},
    "2": {"source_position": [-0.85, -0.95, -0.33]},
    // ... more scenes
}
```

## Quick Start

### 1. Test with Sample Data

```bash
# Run basic test without audio
python test_colmap_audio.py \
    --scene_dir data/1 \
    --source_position 0.06 0.13 -0.45 \
    --max_frames 30 \
    --visualize

# Run with audio features
python test_colmap_audio.py \
    --scene_dir data/1 \
    --source_position 0.06 0.13 -0.45 \
    --use_audio \
    --max_frames 30 \
    --visualize \
    --simulate_propagation
```

### 2. Use Convenience Script

```bash
# Run all tests on data/1
./run_data1_test.sh
```

## Python API Usage

```python
from vggt.utils.colmap_loader import COLMAPAudioDataset
from vggt.models.vggt_audio import VGGTAudio
from vggt.utils.audio_utils import AudioFeatureExtractor

# Load dataset
dataset = COLMAPAudioDataset(
    scene_dir="data/1",
    source_position=[0.06, 0.13, -0.45],
    max_frames=50,
    frame_skip=10
)

# Prepare batch
batch = dataset.prepare_vggt_batch()

# Initialize model
model = VGGTAudio(enable_acoustic=True)

# Extract audio features
audio_extractor = AudioFeatureExtractor(feature_type="mel_spectrogram")
audio_features = []
for frame_audio in batch['binaural_audio'][0]:
    features = audio_extractor.extract_features(frame_audio)
    audio_features.append(features)
audio_features = torch.stack(audio_features).unsqueeze(0)

# Run inference
predictions = model(batch['images'], audio_features=audio_features)
```

## Key Features

1. **Automatic Data Loading**: The `COLMAPAudioDataset` class handles all data loading and preprocessing
2. **Audio Synchronization**: Binaural audio is automatically segmented to match video frames
3. **Flexible Processing**: Support for frame skipping and limiting frame count for faster testing
4. **Visualization**: Built-in visualization of results including depth, acoustic properties, and camera trajectories

## Outputs

The test script generates:
- `reconstruction_results.png`: Visual comparison of predictions
- `camera_trajectory.png`: 3D plot of camera paths and audio source
- `simulated_binaural.wav`: Synthesized spatial audio
- `predictions.pt`: Saved model predictions for further analysis

## Troubleshooting

1. **Memory Issues**: Reduce `--max_frames` or increase `--frame_skip`
2. **Missing Audio**: Ensure audio files match expected names or modify in dataset constructor
3. **CUDA Errors**: The model will automatically fall back to CPU if CUDA is unavailable

## Custom Datasets

To use your own COLMAP data:

1. Ensure your data follows the expected directory structure
2. Export camera parameters in NeRF format (transforms.json)
3. Provide synchronized audio recordings
4. Specify the audio source position

## Advanced Usage

### Custom Audio Features

```python
# Use different audio features
audio_extractor = AudioFeatureExtractor(
    feature_type="mfcc",  # or "wav2vec2"
    n_mfcc=40
)
```

### Acoustic Propagation Simulation

```python
from vggt.utils.acoustic_propagation import AcousticPropagation

propagation = AcousticPropagation(max_order_reflections=3)
binaural_audio = propagation(
    source_audio,
    source_position,
    listener_position,
    listener_orientation,
    world_points,
    acoustic_properties
)
```