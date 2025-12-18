#!/usr/bin/env python3
"""
Benchmark Script
================
Model performans testleri.

Kullanım:
    python scripts/benchmark.py --model exports/model_fp16.onnx --format onnx
    python scripts/benchmark.py --model exports/model.tflite --format tflite
"""

import argparse
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Model")
    
    # Model
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "tflite", "pytorch"],
                        help="Model format")
    
    # Benchmark config
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--num_steps", type=int, default=4, help="LCM steps")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    
    return parser.parse_args()


def benchmark_pytorch(model_path: str, args):
    """PyTorch model benchmark"""
    from src.models import LowLightDiffusion
    
    model = LowLightDiffusion(
        unet_variant="small",
        image_size=args.image_size,
        num_inference_steps=args.num_steps,
    )
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # Test input
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(args.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.enhance(x, num_inference_steps=args.num_steps)
    
    if args.device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(args.num_runs):
            start = time.perf_counter()
            _ = model.enhance(x, num_inference_steps=args.num_steps)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    return times


def benchmark_onnx(model_path: str, args):
    """ONNX model benchmark"""
    from src.export.onnx_export import ONNXInference
    
    inference = ONNXInference(
        model_path,
        device=args.device,
        num_threads=args.threads,
    )
    
    results = inference.benchmark(
        input_shape=(args.batch_size, 6, args.image_size, args.image_size),
        num_runs=args.num_runs,
        warmup_runs=args.warmup,
    )
    
    return results


def benchmark_tflite(model_path: str, args):
    """TFLite model benchmark"""
    from src.export.tflite_export import TFLiteInference
    
    inference = TFLiteInference(
        model_path,
        num_threads=args.threads,
    )
    
    results = inference.benchmark(
        input_shape=(args.batch_size, 6, args.image_size, args.image_size),
        num_runs=args.num_runs,
        warmup_runs=args.warmup,
    )
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Low-Light Enhancement Model Benchmark")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Format: {args.format}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Threads: {args.threads}")
    print(f"  Runs: {args.num_runs}")
    
    print("\nRunning benchmark...")
    
    if args.format == "pytorch":
        times = benchmark_pytorch(args.model, args)
        
        results = {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "min_latency_ms": np.min(times) * 1000,
            "max_latency_ms": np.max(times) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
        }
        
    elif args.format == "onnx":
        results = benchmark_onnx(args.model, args)
        
    elif args.format == "tflite":
        results = benchmark_tflite(args.model, args)
    
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"Mean latency:  {results['mean_latency_ms']:.2f} ms")
    print(f"Std latency:   {results['std_latency_ms']:.2f} ms")
    print(f"Min latency:   {results['min_latency_ms']:.2f} ms")
    print(f"Max latency:   {results['max_latency_ms']:.2f} ms")
    print(f"Throughput:    {results['throughput_fps']:.1f} FPS")
    
    # Per-step latency (LCM)
    per_step = results['mean_latency_ms'] / args.num_steps
    print(f"\nPer-step latency: {per_step:.2f} ms ({args.num_steps} steps)")
    
    # Real-time analysis
    target_fps = 30
    target_latency = 1000 / target_fps  # ~33ms
    
    print(f"\n{'=' * 40}")
    print("REAL-TIME ANALYSIS")
    print(f"{'=' * 40}")
    print(f"Target: {target_fps} FPS ({target_latency:.1f} ms)")
    
    if results['mean_latency_ms'] < target_latency:
        print(f"✓ Model CAN achieve real-time ({results['throughput_fps']:.1f} FPS)")
    else:
        required_speedup = results['mean_latency_ms'] / target_latency
        print(f"✗ Model CANNOT achieve real-time")
        print(f"  Required speedup: {required_speedup:.1f}x")
        print(f"  Suggestions:")
        print(f"    - Use smaller model variant (tiny)")
        print(f"    - Reduce image size")
        print(f"    - Use fewer LCM steps")
        print(f"    - Use INT8 quantization")
        print(f"    - Use GPU/NPU delegate")


if __name__ == "__main__":
    main()

