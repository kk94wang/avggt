# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer) is a feed-forward neural network for 3D scene reconstruction from images. It directly infers camera parameters, depth maps, 3D point maps, and 3D point tracks from one or multiple views, processing them within seconds. This project was awarded Best Paper at CVPR 2025.

## Common Development Commands

### Installation
```bash
# Install basic dependencies
pip install -r requirements.txt

# Install as editable package for development
pip install -e .

# Install demo dependencies (for visualization)
pip install -r requirements_demo.txt
```

### Running Demos
```bash
# Web interface demo
python demo_gradio.py

# 3D visualization
python demo_viser.py --image_folder path/to/images

# Export to COLMAP format
python demo_colmap.py --scene_dir=/path/to/scene

# Export with bundle adjustment
python demo_colmap.py --scene_dir=/path/to/scene --use_ba
```

### Training
```bash
# Navigate to training directory
cd training

# Multi-GPU training (e.g., 4 GPUs)
torchrun --nproc_per_node=4 launch.py

# Single GPU training
python launch.py
```

### Dataset Preparation
```bash
# Download VKitti dataset
cd training/data/preprocess
bash vkitti.sh
```

## Architecture Overview

### Core Model Flow
```
Input Images [B, S, 3, H, W] → Aggregator (Alternating Attention) → Multiple Prediction Heads
```

### Key Components

1. **Aggregator** (`vggt/models/aggregator.py`)
   - Vision Transformer backbone (DINOv2 ViT-L/14)
   - Alternates between frame-wise and global attention
   - Special tokens: camera (2), register (4), patch tokens

2. **Prediction Heads** (in `vggt/heads/`)
   - **Camera Head**: Predicts 9D pose encoding (translation + rotation + FoV)
   - **Depth Head**: Outputs depth maps with confidence scores
   - **Point Head**: Predicts 3D world coordinates per pixel
   - **Track Head**: Tracks query points across frames

3. **Input Processing** (`vggt/utils/load_fn.py`)
   - Default resolution: 518×518 pixels
   - ResNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - Handles both single images and image sequences

### Training Infrastructure

- **Configuration**: Uses Hydra/OmegaConf (`training/config/default.yaml`)
- **Multi-task Loss** (`training/loss.py`): Combines camera, depth, point, and track losses
- **Distributed Training**: PyTorch DDP support
- **Datasets**: Co3D and VKitti support with dynamic dataloaders

### Key Design Patterns

1. **Modular Architecture**: Each prediction head can be enabled/disabled independently
2. **Confidence-Aware**: All dense predictions include confidence estimation
3. **Iterative Refinement**: Camera and track heads use multiple refinement iterations
4. **Multi-Scale Features**: DPT heads fuse features from layers [4, 11, 17, 23]

## Important Configuration

Key configuration parameters in `training/config/default.yaml`:
- `CO3D_DIR`: Path to Co3D dataset
- `resume_checkpoint_path`: Pre-trained checkpoint path
- `max_img_per_gpu`: Batch size per GPU
- `accum_steps`: Gradient accumulation steps

## Development Notes

- **No formal test suite**: Only `test.py` for manual testing
- **No linting tools configured**: No flake8, black, or ruff setup
- **Model Distribution**: Via Hugging Face Model Hub ("facebook/VGGT-*")
- **GPU Memory**: Use bfloat16 for compute capability ≥ 8, otherwise float16