# Unreal Engine Data Export Requirements for VGGT-Audio

## 1. Visual Data Requirements

### Camera Setup
- **Multi-view captures**: 30-100 frames per scene
- **Camera trajectory**: Smooth, continuous motion (avoid teleporting)
- **Resolution**: 480×270 or higher (will be resized to 518×518)
- **Field of View**: 60-90 degrees recommended
- **Frame rate**: 30 FPS for video, or individual frames

### Per-Frame Exports
```json
{
  "frame_id": "00001",
  "camera_matrix": [4x4 transformation matrix],  // World-to-camera
  "intrinsics": {
    "fx": 258.43,  // Focal length X
    "fy": 258.43,  // Focal length Y  
    "cx": 240.0,   // Principal point X
    "cy": 135.0    // Principal point Y
  },
  "image_path": "frames/00001.png"
}
```

## 2. Acoustic Data Requirements

### Audio Source Configuration
- **Source position**: 3D world coordinates [x, y, z]
- **Source audio**: Clean, dry mono recording (no reverb)
- **Sample rate**: 48000 Hz preferred
- **Duration**: Match video duration

### Binaural Audio Capture
For each camera position, record:
- **Left channel**: Audio at left "ear" position
- **Right channel**: Audio at right "ear" position
- **Format**: Stereo WAV, synchronized with frames

### Acoustic Material Properties (Ground Truth)
For training supervision, export per-material:
```json
{
  "material_id": "concrete_wall",
  "absorption": [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06],  // 8 frequency bands
  "scattering": 0.1,  // 0=specular, 1=diffuse
  "transmission": 0.001
}
```

## 3. Unreal Engine Blueprint Setup

### A. Scene Capture Component
```cpp
// In your capture actor
UCameraComponent* Camera;
USceneCaptureComponent2D* SceneCapture;

// Capture settings
SceneCapture->CaptureSource = SCS_FinalColorLDR;
SceneCapture->bCaptureEveryFrame = false;
SceneCapture->TextureTarget = RenderTarget;
```

### B. Audio Capture System
```cpp
// Audio listener at camera position
UAudioListenerComponent* ListenerLeft;
UAudioListenerComponent* ListenerRight;

// Position listeners for binaural capture
FVector CameraLocation = Camera->GetComponentLocation();
FVector CameraForward = Camera->GetForwardVector();
FVector CameraRight = Camera->GetRightVector();

// Inter-aural distance ~17cm
ListenerLeft->SetWorldLocation(CameraLocation - CameraRight * 8.5f);
ListenerRight->SetWorldLocation(CameraLocation + CameraRight * 8.5f);
```

### C. Material Property Export
```cpp
// For each physical material in scene
UPhysicalMaterial* PhysMat = HitResult.PhysMaterial.Get();
if (PhysMat) {
    FString MaterialName = PhysMat->GetName();
    float Absorption = PhysMat->GetCustomProperty("Absorption");
    float Scattering = PhysMat->GetCustomProperty("Scattering");
    // Export to JSON
}
```

## 4. Data Organization Structure

```
scene_001/
├── frames/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
├── audio/
│   ├── source.wav          # Dry source audio
│   ├── binaural.wav        # Recorded at camera positions
│   └── impulse_responses/  # Optional IR measurements
├── transforms.json         # Camera parameters
├── acoustic_properties.json  # Material properties
└── metadata.json          # Scene configuration
```

## 5. Minimum Dataset Requirements

For effective LoRA fine-tuning:
- **Scenes**: 50-100 diverse environments
- **Frames per scene**: 30-50 views
- **Materials**: 10-20 distinct acoustic materials
- **Audio sources**: Vary position across scenes
- **Environments**: Mix of rooms, halls, outdoor spaces

## 6. Export Script Template

```python
import unreal
import json
import numpy as np

def export_vggt_data(scene_name, output_dir):
    """Export scene data in VGGT-Audio format."""
    
    # Get camera actor
    camera = unreal.EditorLevelLibrary.get_selected_level_actors()[0]
    
    # Setup data
    transforms = {
        "fl_x": camera.camera_component.field_of_view,
        "w": 480, "h": 270,
        "frames": []
    }
    
    # Capture trajectory
    for frame in range(num_frames):
        # Move camera along spline
        camera.set_actor_location(spline.get_location_at_time(frame))
        
        # Capture image
        capture_component.capture_scene()
        
        # Get transform matrix
        transform = camera.get_actor_transform()
        matrix = transform_to_matrix(transform)
        
        transforms["frames"].append({
            "file_path": f"frames/{frame:05d}.png",
            "transform_matrix": matrix.tolist()
        })
        
        # Record binaural audio
        record_binaural_audio(frame)
    
    # Save data
    with open(f"{output_dir}/transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)
```

## 7. Quality Checklist

- [ ] Images are sharp and well-lit
- [ ] Camera motion is smooth (no jumps)
- [ ] Audio is synchronized with video
- [ ] Source audio has no clipping
- [ ] Binaural recordings capture spatial effects
- [ ] Material properties are physically plausible
- [ ] All coordinate systems are consistent
- [ ] File paths are relative and correct

## 8. Data Augmentation Tips

To maximize limited data:
1. **Vary lighting**: Day/night, different light positions
2. **Change materials**: Swap textures while keeping geometry
3. **Audio variations**: Different source sounds, positions
4. **Camera paths**: Multiple trajectories per scene
5. **Acoustic conditions**: Open/close doors, add/remove objects