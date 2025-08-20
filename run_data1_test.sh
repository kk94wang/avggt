#!/bin/bash
# Test script for VGGT-Audio with data/1 sample

echo "Testing VGGT-Audio with COLMAP data from data/1..."

# Test 1: Basic VGGT without audio
echo -e "\n=== Test 1: Basic VGGT (no audio) ==="
python test_colmap_audio.py \
    --scene_dir data/1 \
    --source_position 0.06 0.13 -0.45 \
    --max_frames 30 \
    --frame_skip 10 \
    --visualize \
    --vis_frames 5 \
    --output_dir output_colmap/test1_basic \
    --save_predictions

# Test 2: VGGT-Audio with acoustic features
echo -e "\n=== Test 2: VGGT-Audio with acoustic features ==="
python test_colmap_audio.py \
    --scene_dir data/1 \
    --source_position 0.06 0.13 -0.45 \
    --use_audio \
    --max_frames 30 \
    --frame_skip 10 \
    --visualize \
    --vis_frames 5 \
    --simulate_propagation \
    --output_dir output_colmap/test2_audio \
    --save_predictions

echo -e "\nAll tests completed! Check output_colmap/ for results."