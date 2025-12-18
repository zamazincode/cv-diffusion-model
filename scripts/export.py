#!/usr/bin/env python3
"""
Export Script
=============
Model export ve quantization.

Kullanım:
    # ONNX export
    python scripts/export.py --checkpoint best_model.pt --format onnx
    
    # TFLite export (FP16)
    python scripts/export.py --checkpoint best_model.pt --format tflite --quantize fp16
    
    # TFLite export (INT8)
    python scripts/export.py --checkpoint best_model.pt --format tflite --quantize int8
    
    # Android package
    python scripts/export.py --checkpoint best_model.pt --android
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models import LowLightDiffusion
from src.export import (
    export_to_onnx,
    export_to_tflite,
    quantize_model,
    QuantizationConfig,
)
from src.export.android_pipeline import create_android_package, AndroidPipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Export Model")
    
    # Input
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    
    # Model config
    parser.add_argument("--variant", type=str, default="small", help="Model variant")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    
    # Export format
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "tflite", "pytorch"],
                        help="Export format")
    
    # Quantization
    parser.add_argument("--quantize", type=str, default="fp16",
                        choices=["fp16", "int8", "int8_dynamic", "none"],
                        help="Quantization type")
    
    # Android package
    parser.add_argument("--android", action="store_true",
                        help="Create Android deployment package")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="exports",
                        help="Output directory")
    
    # Benchmark
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, variant: str, image_size: int) -> LowLightDiffusion:
    """Load model from checkpoint"""
    
    print(f"Loading model from: {checkpoint_path}")
    
    model = LowLightDiffusion(
        unet_variant=variant,
        image_size=image_size,
        num_inference_steps=4,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Low-Light Enhancement Model Export")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, args.variant, args.image_size)
    
    model_size = model.get_model_size()
    print(f"\nOriginal model:")
    print(f"  Parameters: {model_size['num_params']:,}")
    print(f"  FP32 size: {model_size['fp32_mb']:.2f} MB")
    
    # Android package
    if args.android:
        print("\nCreating Android package...")
        config = AndroidPipelineConfig(
            image_size=args.image_size,
            num_inference_steps=4,
            use_fp16=(args.quantize == "fp16"),
        )
        create_android_package(model, str(output_dir / "android"), config)
        return
    
    # Quantization
    if args.quantize != "none":
        print(f"\nApplying {args.quantize} quantization...")
        
        config = QuantizationConfig(dtype=args.quantize)
        model = quantize_model(model, config)
    
    # Export
    if args.format == "onnx":
        print("\nExporting to ONNX...")
        output_path = output_dir / f"model_{args.quantize}.onnx"
        export_to_onnx(
            model,
            str(output_path),
            image_size=args.image_size,
            in_channels=6,
            fp16=(args.quantize == "fp16"),
        )
        
    elif args.format == "tflite":
        print("\nExporting to TFLite...")
        output_path = output_dir / f"model_{args.quantize}.tflite"
        export_to_tflite(
            model,
            str(output_path),
            image_size=args.image_size,
            in_channels=6,
            quantize=(args.quantize != "none"),
        )
        
    elif args.format == "pytorch":
        print("\nSaving PyTorch model...")
        output_path = output_dir / f"model_{args.quantize}.pt"
        torch.save(model.state_dict(), output_path)
        print(f"Saved to: {output_path}")
    
    # Benchmark
    if args.benchmark and args.format == "onnx":
        from src.export.onnx_export import ONNXInference
        
        print("\nRunning benchmark...")
        inference = ONNXInference(str(output_path))
        results = inference.benchmark(input_shape=(1, 6, args.image_size, args.image_size))
        
        print(f"\nBenchmark Results:")
        print(f"  Mean latency: {results['mean_latency_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
    
    print("\nExport complete!")


if __name__ == "__main__":
    main()

