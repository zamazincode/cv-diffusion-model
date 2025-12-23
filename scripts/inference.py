#!/usr/bin/env python3
"""
Inference Script
================
Tek görüntü veya klasör için inference.

Kullanım:
    # Tek görüntü
    python scripts/inference.py --input dark_image.jpg --output enhanced.jpg
    
    # Klasör
    python scripts/inference.py --input input_folder --output output_folder
    
    # ONNX model ile
    python scripts/inference.py --input dark.jpg --model exports/model.onnx --format onnx
"""

import argparse
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Low-Light Enhancement Inference")
    
    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input image or folder")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image or folder")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch checkpoint (if not using exported model)")
    parser.add_argument("--model", type=str, default=None,
                        help="Exported model path (ONNX/TFLite)")
    parser.add_argument("--format", type=str, default="pytorch",
                        choices=["pytorch", "onnx", "tflite"],
                        help="Model format")
    
    # Model config (for PyTorch)
    parser.add_argument("--variant", type=str, default="small",
                        help="Model variant")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Processing image size")
    parser.add_argument("--num_steps", type=int, default=4,
                        help="LCM inference steps")
    
    # Device
    import torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device,
                        help="Device (cuda/cpu)")
    
    return parser.parse_args()


def load_pytorch_model(checkpoint_path: str, variant: str, image_size: int, device: str):
    """PyTorch model yükle"""
    from src.models import LowLightDiffusion
    
    model = LowLightDiffusion(
        unet_variant=variant,
        image_size=image_size,
        num_inference_steps=4,
    )
    
    if checkpoint_path:
        # CPU'ya yükle (CUDA yoksa)
        map_location = "cpu" if device == "cpu" or not torch.cuda.is_available() else device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    model.eval()
    
    return model


def load_onnx_model(model_path: str):
    """ONNX model yükle"""
    from src.export.onnx_export import ONNXInference
    return ONNXInference(model_path)


def load_tflite_model(model_path: str):
    """TFLite model yükle"""
    from src.export.tflite_export import TFLiteInference
    return TFLiteInference(model_path)


def preprocess_image(image_path: str, target_size: int) -> tuple:
    """Görüntüyü preprocess et"""
    import cv2
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_size = image.shape[:2]
    
    # Resize
    image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1.0
    
    # HWC -> CHW -> BCHW
    image = image.transpose(2, 0, 1)[np.newaxis, ...]
    
    return image, original_size


def postprocess_image(output: np.ndarray, original_size: tuple) -> np.ndarray:
    """Model çıktısını görüntüye dönüştür"""
    import cv2
    
    # BCHW -> CHW -> HWC
    output = output[0].transpose(1, 2, 0)
    
    # Denormalize
    output = (output + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Resize to original
    output = cv2.resize(output, (original_size[1], original_size[0]))
    
    return output


def enhance_pytorch(model, image: np.ndarray, num_steps: int, device: str) -> np.ndarray:
    """PyTorch ile inference"""
    
    image_tensor = torch.from_numpy(image).to(device)
    
    with torch.no_grad():
        enhanced = model.enhance(image_tensor, num_inference_steps=num_steps)
    
    return enhanced.cpu().numpy()


def enhance_onnx(model, image: np.ndarray, num_steps: int, image_size: int) -> np.ndarray:
    """ONNX ile inference (LCM loop dahil)"""
    from src.export.android_pipeline import LCMDenoisingLoop
    
    loop = LCMDenoisingLoop(num_inference_steps=num_steps)
    
    # Initial noise
    latents = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    # Denoising loop
    for timestep in loop.timesteps:
        model_input = np.concatenate([latents, image], axis=1)
        t = np.array([timestep], dtype=np.int64)
        
        noise_pred = model(
            torch.from_numpy(model_input),
            torch.from_numpy(t)
        ).numpy()
        
        latents = loop.step(noise_pred, timestep, latents)
    
    return np.clip(latents, -1, 1)


def enhance_tflite(model, image: np.ndarray, num_steps: int, image_size: int) -> np.ndarray:
    """TFLite ile inference"""
    from src.export.android_pipeline import LCMDenoisingLoop
    
    loop = LCMDenoisingLoop(num_inference_steps=num_steps)
    
    latents = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    for timestep in loop.timesteps:
        model_input = np.concatenate([latents, image], axis=1)
        t = np.array([timestep], dtype=np.int64)
        
        noise_pred = model(model_input, t)
        latents = loop.step(noise_pred, timestep, latents)
    
    return np.clip(latents, -1, 1)


def process_single_image(args, model, input_path: str, output_path: str):
    """Tek görüntü işle"""
    import cv2
    
    print(f"Processing: {input_path}")
    
    # Preprocess
    image, original_size = preprocess_image(input_path, args.image_size)
    
    # Inference
    start = time.perf_counter()
    
    if args.format == "pytorch":
        enhanced = enhance_pytorch(model, image, args.num_steps, args.device)
    elif args.format == "onnx":
        enhanced = enhance_onnx(model, image, args.num_steps, args.image_size)
    else:
        enhanced = enhance_tflite(model, image, args.num_steps, args.image_size)
    
    elapsed = time.perf_counter() - start
    
    # Postprocess
    enhanced = postprocess_image(enhanced, original_size)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    
    # Save
    cv2.imwrite(output_path, enhanced)
    
    print(f"  Saved to: {output_path}")
    print(f"  Time: {elapsed*1000:.1f} ms")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Low-Light Enhancement Inference")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model ({args.format})...")
    
    if args.format == "pytorch":
        model = load_pytorch_model(
            args.checkpoint, args.variant, args.image_size, args.device
        )
    elif args.format == "onnx":
        model = load_onnx_model(args.model)
    else:
        model = load_tflite_model(args.model)
    
    print("Model loaded!")
    
    # Process
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Single image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        process_single_image(args, model, str(input_path), str(output_path))
    
    elif input_path.is_dir():
        # Folder
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        print(f"\nProcessing {len(images)} images...")
        
        total_time = 0
        for img_path in images:
            out_path = output_path / img_path.name
            
            start = time.perf_counter()
            process_single_image(args, model, str(img_path), str(out_path))
            total_time += time.perf_counter() - start
        
        avg_time = total_time / len(images) * 1000
        print(f"\nAverage time per image: {avg_time:.1f} ms")
        print(f"Throughput: {1000/avg_time:.1f} FPS")
    
    else:
        print(f"Error: {input_path} not found")
        return
    
    print("\nDone!")


if __name__ == "__main__":
    main()

