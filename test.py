import torch
import numpy as np
from PIL import Image
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda:1" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["/home/kwang385/vggt/examples/single_oil_painting/images/model_was_never_trained_on_single_image_or_oil_painting.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

print("Predictions keys:", predictions.keys())

# Save the image predictions
if 'images' in predictions:
    image_tensor = predictions['images']
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image tensor dtype: {image_tensor.dtype}")
    print(f"Image tensor range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
    
    # Create output directory
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensor to numpy and handle different batch sizes
    image_np = image_tensor.cpu().float().numpy()
    
    # Handle different tensor shapes
    if len(image_np.shape) == 5:  # [B, T, C, H, W] - video format
        batch_size, time_steps, channels, height, width = image_np.shape
        for b in range(batch_size):
            for t in range(time_steps):
                # Extract single frame
                frame = image_np[b, t]  # [C, H, W]
                
                # Convert from [C, H, W] to [H, W, C]
                if channels == 3:
                    frame = np.transpose(frame, (1, 2, 0))
                elif channels == 1:
                    frame = frame.squeeze(0)  # Remove channel dimension for grayscale
                
                # Normalize to [0, 255] range
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # Save image
                if channels == 3:
                    img = Image.fromarray(frame, 'RGB')
                else:
                    img = Image.fromarray(frame, 'L')
                    
                img.save(f"{output_dir}/prediction_batch{b}_frame{t}.png")
                print(f"Saved: {output_dir}/prediction_batch{b}_frame{t}.png")
                
    elif len(image_np.shape) == 4:  # [B, C, H, W] - image format
        batch_size, channels, height, width = image_np.shape
        for b in range(batch_size):
            # Extract single image
            img_data = image_np[b]  # [C, H, W]
            
            # Convert from [C, H, W] to [H, W, C]
            if channels == 3:
                img_data = np.transpose(img_data, (1, 2, 0))
            elif channels == 1:
                img_data = img_data.squeeze(0)  # Remove channel dimension for grayscale
            
            # Normalize to [0, 255] range
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            
            # Save image
            if channels == 3:
                img = Image.fromarray(img_data, 'RGB')
            else:
                img = Image.fromarray(img_data, 'L')
                
            img.save(f"{output_dir}/prediction_batch{b}.png")
            print(f"Saved: {output_dir}/prediction_batch{b}.png")
            
    else:
        print(f"Unsupported image tensor shape: {image_np.shape}")
        
else:
    print("No 'images' key found in predictions")
    print("Available keys:", list(predictions.keys()))  
