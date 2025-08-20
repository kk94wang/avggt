# VGGT-Audio: Technical Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Audio Token Mechanism](#audio-token-mechanism)
5. [Acoustic Property Prediction](#acoustic-property-prediction)
6. [Cross-Modal Learning](#cross-modal-learning)
7. [Audio Processing Pipeline](#audio-processing-pipeline)
8. [Training Strategy](#training-strategy)
9. [Implementation Details](#implementation-details)
10. [Usage Guide](#usage-guide)
11. [Limitations and Future Work](#limitations-and-future-work)

## 1. Introduction

VGGT-Audio extends the Visual Geometry Grounded Transformer (VGGT) to jointly learn visual geometry and acoustic properties of 3D environments. This enables physics-based audio propagation simulation in reconstructed scenes, creating an "acoustic digital twin" of real-world spaces.

### Key Innovations
- **Audio Tokens**: Specialized tokens that capture acoustic information and interact with visual tokens through cross-modal attention
- **Acoustic Property Prediction**: Dense prediction of frequency-dependent material properties (absorption, reflection, scattering)
- **Unified Architecture**: Single forward pass for both visual and acoustic understanding
- **Physical Interpretability**: Outputs correspond to real acoustic properties that can be used for physics-based simulation

## 2. Architecture Overview

### High-Level Data Flow
```
Input:
  - Images: [B, S, 3, H, W] (multi-view sequences)
  - Audio Features: [B, S, D] (extracted from recordings)
  
Processing:
  - AudioAggregator: Alternating attention with audio tokens
  - Multiple Prediction Heads: Camera, Depth, Point, Acoustic
  
Output:
  - Visual: Camera poses, depth maps, 3D points
  - Acoustic: Material properties per pixel
    - Absorption: [B, S, H, W, freq_bands]
    - Reflection: [B, S, H, W, 1]
    - Scattering: [B, S, H, W, 1]
    - Impedance: [B, S, H, W, 2]
```

### Model Components
```python
VGGTAudio
├── AudioAggregator (extends Aggregator)
│   ├── Patch Embedding (DINOv2)
│   ├── Special Tokens
│   │   ├── Camera Tokens (2)
│   │   ├── Audio Tokens (2) ← NEW
│   │   └── Register Tokens (4)
│   └── Alternating Attention Blocks
│       ├── Frame Attention
│       └── Global Attention
└── Prediction Heads
    ├── CameraHead → Pose encoding
    ├── DepthHead → Depth maps
    ├── PointHead → 3D coordinates
    ├── TrackHead → Point tracking
    └── AcousticHead → Material properties ← NEW
```

## 3. Core Components

### 3.1 VGGTAudio Class

The main model class that orchestrates all components:

```python
class VGGTAudio(nn.Module):
    def __init__(self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_acoustic=True,
        num_audio_tokens=2,
        audio_feature_dim=768,
        freq_bands=8
    ):
```

Key features:
- Inherits from PyTorchModelHubMixin for easy model sharing
- Supports loading from pretrained VGGT weights
- Modular design - each head can be enabled/disabled
- Handles variable input dimensions gracefully

### 3.2 AudioAggregator

Extends the base Aggregator with audio understanding:

```python
class AudioAggregator(Aggregator):
    def __init__(self, ..., num_audio_tokens=2, audio_feature_dim=768):
        # Audio-specific parameters
        self.audio_token = nn.Parameter(torch.randn(1, 2, num_audio_tokens, embed_dim))
        self.audio_proj = nn.Linear(audio_feature_dim, embed_dim)
        
        # Update token ordering
        self.patch_start_idx = 1 + num_audio_tokens + num_register_tokens
```

Token ordering in sequence:
```
[camera_tokens | audio_tokens | register_tokens | patch_tokens]
     1 token    |   2 tokens   |   4 tokens     | H*W/p² tokens
```

### 3.3 AcousticHead

Predicts dense acoustic properties using DPT architecture:

```python
class AcousticHead(nn.Module):
    def __init__(self,
        dim_in=2048,        # Input from aggregator (2 * embed_dim)
        freq_bands=8,       # Frequency resolution
        features=256,       # DPT feature dimension
        layers=[4,11,17,23] # Multi-scale layers
    ):
```

Output channels:
- Absorption: 8 channels (125Hz to 16kHz)
- Reflection: 1 channel (energy coefficient)
- Scattering: 1 channel (diffuse vs specular)
- Impedance: 2 channels (complex valued)

## 4. Audio Token Mechanism

### 4.1 Token Design

Audio tokens capture two complementary aspects:
1. **Direct Sound Token**: Encodes direct path characteristics
2. **Ambient Sound Token**: Encodes reverberant field properties

```python
# Initialize with small random values
self.audio_token = nn.Parameter(torch.randn(1, 2, num_audio_tokens, embed_dim))
nn.init.normal_(self.audio_token, std=1e-6)
```

### 4.2 Audio Feature Integration

Audio features modulate the base audio tokens:

```python
if audio_features is not None:
    # Project to embedding dimension
    audio_features = self.audio_proj(audio_features)  # [B*S, embed_dim]
    
    # Modulate tokens with frame-specific audio
    audio_features = audio_features.unsqueeze(1)  # [B*S, 1, embed_dim]
    audio_token = audio_token + audio_features.expand(-1, num_audio_tokens, -1)
```

This allows tokens to adapt to the specific acoustic characteristics of each frame while maintaining learned priors.

### 4.3 Cross-Modal Attention

Audio tokens participate in both attention types:

**Frame Attention** (within single frame):
```
camera_token ← → audio_tokens ← → register_tokens ← → patch_tokens
```
Enables audio tokens to gather information about the visual scene structure.

**Global Attention** (across all frames):
```
Frame 1 tokens ← → Frame 2 tokens ← → ... ← → Frame N tokens
```
Enables temporal consistency and multi-view acoustic understanding.

## 5. Acoustic Property Prediction

### 5.1 Physical Properties

The model predicts physically meaningful acoustic properties:

1. **Absorption Coefficient (α)**
   - Range: [0, 1] per frequency band
   - Meaning: Fraction of sound energy absorbed by material
   - Frequency bands: 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz

2. **Reflection Coefficient (R)**
   - Range: [0, 1]
   - Meaning: Fraction of sound energy reflected
   - Constraint: α + R ≤ 1 (energy conservation)

3. **Scattering Coefficient (s)**
   - Range: [0, 1]
   - Meaning: 0 = specular reflection, 1 = diffuse scattering

4. **Acoustic Impedance (Z)**
   - Complex valued: Z = Z_real + i*Z_imag
   - Meaning: Material's resistance to sound propagation

### 5.2 Multi-Scale Processing

The AcousticHead uses DPT's multi-scale architecture:

```python
# Extract features from multiple depths
layers = [4, 11, 17, 23]  # Shallow to deep

# Process at different scales
features_4  → 4x upsampling  → 518x518
features_11 → 2x upsampling  → 518x518
features_17 → 1x (native)    → 518x518
features_23 → 2x downsampling → 518x518

# Fuse via RefineNet blocks
fused_features = RefineNet(features_4, features_11, features_17, features_23)
```

### 5.3 Physical Constraints

Activations ensure physical validity:

```python
# Absorption and reflection: sigmoid ensures [0,1]
absorption = torch.sigmoid(raw_absorption)
reflection = torch.sigmoid(raw_reflection)

# Scattering: sigmoid for [0,1]
scattering = torch.sigmoid(raw_scattering)

# Impedance: no activation (can be any real/complex value)
impedance = raw_impedance
```

## 6. Cross-Modal Learning

### 6.1 Information Flow

The model learns correlations between visual appearance and acoustic behavior:

```
Visual Features → Shared Attention → Acoustic Properties
     ↓                    ↕                    ↓
Hard surfaces      Audio tokens learn      Low absorption
Glass/metal      material associations     High reflection
Soft furnishing                            High absorption
```

### 6.2 Attention Patterns

Audio tokens can attend to:
- **Geometric features**: Depth discontinuities indicate surfaces
- **Texture patterns**: Material recognition from visual appearance
- **Spatial layout**: Room shape affects reverberation

### 6.3 Multi-View Consistency

Global attention enforces consistency across views:
- Same material should have similar acoustic properties from different viewpoints
- Audio source localization benefits from multiple perspectives

## 7. Audio Processing Pipeline

### 7.1 Audio Feature Extraction

```python
class AudioFeatureExtractor:
    def __init__(self, feature_type="mel_spectrogram", n_mels=128):
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        )
```

Supported features:
- **Mel-spectrograms**: Time-frequency representation
- **MFCCs**: Compact spectral features
- **Wav2Vec2**: Pre-trained representations (future)

### 7.2 Synchronization

Audio is segmented to match video frames:

```python
def get_audio_segments(audio, sample_rate, fps=30):
    samples_per_frame = int(sample_rate / fps)
    segments = []
    for i in range(num_frames):
        start = i * samples_per_frame
        end = (i + 1) * samples_per_frame
        segments.append(audio[:, start:end])
    return segments
```

### 7.3 Spatial Audio Encoding

For binaural audio:
```python
# Extract spatial cues
ITD = compute_interaural_time_difference(left, right)
ILD = compute_interaural_level_difference(left, right)
```

## 8. Training Strategy

### 8.1 Loss Functions

The model uses multiple loss components:

```python
class AcousticLoss(nn.Module):
    def forward(self, rendered_audio, target_audio, acoustic_props):
        # Audio rendering loss
        rendering_loss = F.l1_loss(rendered_audio, target_audio)
        
        # Perceptual loss (mel-spectrogram space)
        perceptual_loss = F.l1_loss(mel(rendered), mel(target))
        
        # Physical consistency
        energy_conservation = relu(absorption + reflection - 1.0)
        
        # RT60 matching (if available)
        rt60_loss = F.l1_loss(compute_rt60(acoustic_props), rt60_target)
        
        return rendering_loss + 0.5*perceptual_loss + 0.1*physical_loss
```

### 8.2 Data Requirements

Training data needs:
1. **Multi-view images** with known camera poses
2. **Synchronized audio** recordings
3. **Audio source position** (3D coordinates)
4. **Optional**: Ground truth acoustic measurements

Data format:
```python
{
    "images": [frame1, ..., frameN],        # RGB frames
    "audio": {
        "source": source_waveform,          # Mono source
        "binaural": binaural_recordings,    # At camera positions
        "source_position": [x, y, z]
    },
    "camera_params": {...},                 # Extrinsics/intrinsics
    "acoustic_gt": {                        # Optional
        "rt60": [...],
        "absorption": [...]
    }
}
```

### 8.3 Training Procedure

1. **Stage 1: Pretrain on visual data**
   - Use existing VGGT checkpoints
   - Learn robust 3D geometry

2. **Stage 2: Add acoustic components**
   - Initialize audio tokens randomly
   - Freeze visual components (optional)
   - Train on audio-visual data

3. **Stage 3: Fine-tune end-to-end**
   - Unfreeze all components
   - Joint optimization

## 9. Implementation Details

### 9.1 Memory Optimization

- **Gradient checkpointing**: Enabled during training
- **Mixed precision**: BF16/FP16 autocast
- **Frame chunking**: Process video in segments

### 9.2 Device Handling

```python
# Automatic device detection
device_type = 'cuda' if images.is_cuda else 'cpu'
with torch.amp.autocast(device_type=device_type, dtype=dtype):
    # Forward pass
```

### 9.3 Dimension Management

The model handles various input formats:
```python
# Add batch dimension if needed
if len(images.shape) == 4:  # [S, C, H, W]
    images = images.unsqueeze(0)  # [1, S, C, H, W]

# Audio features
if audio_features.shape == 2:  # [S, D]
    audio_features = audio_features.unsqueeze(0)  # [1, S, D]
```

## 10. Usage Guide

### 10.1 Basic Inference

```python
# Initialize model
model = VGGTAudio(enable_acoustic=True, audio_feature_dim=128)

# Extract audio features
audio_extractor = AudioFeatureExtractor(n_mels=128)
audio_features = audio_extractor.extract_features(audio_waveform)

# Run inference
predictions = model(images, audio_features=audio_features)

# Access results
acoustic_props = predictions['acoustic_properties']
absorption = acoustic_props['absorption']  # [B, S, H, W, 8]
```

### 10.2 Loading from Pretrained

```python
# Start from VGGT checkpoint
model = VGGTAudio.from_pretrained_vggt(
    "facebook/VGGT-1B",
    enable_acoustic=True,
    freeze_visual=True  # Keep visual weights fixed
)
```

### 10.3 Acoustic Analysis

```python
# Compute reverberation time
rt60 = model.acoustic_head.compute_rt60(
    acoustic_props,
    room_volume=100.0,  # m³
    surface_area=120.0  # m²
)

# Analyze frequency response
freq_absorption = acoustic_props['absorption'].mean(dim=(0,1,2,3))
for freq, absorption in zip([125,250,500,1000,2000,4000,8000,16000], freq_absorption):
    print(f"{freq}Hz: {absorption:.3f}")
```

## 11. Limitations and Future Work

### Current Limitations

1. **Simplified Acoustic Model**
   - Ray-based propagation (no wave effects)
   - Limited to static scenes
   - Single source assumption

2. **Data Requirements**
   - Needs synchronized audio-visual data
   - Ground truth acoustic measurements rare
   - Limited to indoor scenes in current training

3. **Computational Cost**
   - Additional tokens increase memory usage
   - Acoustic propagation simulation is expensive

### Future Directions

1. **Advanced Acoustic Modeling**
   - Wave-based simulation for low frequencies
   - Dynamic sound sources
   - Multiple source separation

2. **Applications**
   - Real-time spatial audio for VR/AR
   - Acoustic planning for architecture
   - Robot navigation using audio

3. **Model Improvements**
   - Larger frequency range (sub-bass to ultrasonic)
   - Material classification
   - Outdoor scene support

## Conclusion

VGGT-Audio represents a significant step toward unified audio-visual 3D understanding. By encoding acoustic properties directly into the 3D representation, it enables physically-based audio simulation from visual input alone. The modular design and cross-modal learning approach provide a foundation for future research in spatial audio and acoustic scene understanding.