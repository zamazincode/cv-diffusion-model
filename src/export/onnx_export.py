"""
ONNX Export Module
==================
PyTorch modelini ONNX formatına export etme.

ONNX Avantajları:
- Platform bağımsız (Android, iOS, Web)
- ONNX Runtime ile hızlı inference
- TensorRT, OpenVINO ile optimize edilebilir
- Quantization desteği
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import copy

import torch
import torch.nn as nn

try:
    import onnx
    from onnxsim import simplify
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class ONNXExportWrapper(nn.Module):
    """
    ONNX export için model wrapper.
    
    Diffusion modeli için timestep'i de alacak şekilde düzenlenir.
    ONNX'in dynamic axes ve control flow kısıtlamalarını handle eder.
    """
    
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
    
    def forward(
        self, 
        sample: torch.Tensor, 
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        ONNX-compatible forward pass.
        
        Args:
            sample: Noisy image + condition [B, 6, H, W]
            timestep: Timestep [B]
            
        Returns:
            Predicted noise [B, 3, H, W]
        """
        return self.unet(sample, timestep)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    image_size: int = 256,
    in_channels: int = 6,
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    fp16: bool = False,
    simplify_model: bool = True,
    verbose: bool = True,
) -> str:
    """
    PyTorch modelini ONNX'e export et.
    
    Args:
        model: UNet modeli
        output_path: Çıktı dosya yolu (.onnx)
        image_size: Giriş görüntü boyutu
        in_channels: Giriş kanal sayısı (6 = image + condition)
        batch_size: Batch boyutu
        opset_version: ONNX opset versiyonu
        dynamic_batch: Dynamic batch size desteği
        fp16: FP16 export
        simplify_model: ONNX simplifier uygula
        
    Returns:
        Kaydedilen dosya yolu
    """
    
    if not HAS_ONNX:
        raise ImportError("ONNX not installed. Run: pip install onnx onnxsim")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Model hazırlığı
    model = copy.deepcopy(model)
    model.eval()
    
    if fp16:
        model = model.half()
    
    # Export wrapper
    if hasattr(model, 'unet'):
        wrapper = ONNXExportWrapper(model.unet)
    else:
        wrapper = ONNXExportWrapper(model)
    
    wrapper.eval()
    
    # Dummy input
    dtype = torch.float16 if fp16 else torch.float32
    dummy_input = torch.randn(batch_size, in_channels, image_size, image_size, dtype=dtype)
    dummy_timestep = torch.zeros(batch_size, dtype=torch.long)
    
    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "sample": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    
    # Export
    if verbose:
        print(f"Exporting to ONNX: {output_path}")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Opset version: {opset_version}")
        print(f"  FP16: {fp16}")
    
    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_timestep),
        str(output_path),
        input_names=["sample", "timestep"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    
    # Simplify
    if simplify_model:
        if verbose:
            print("  Simplifying ONNX model...")
        
        onnx_model = onnx.load(str(output_path))
        simplified_model, check = simplify(onnx_model)
        
        if check:
            onnx.save(simplified_model, str(output_path))
            if verbose:
                print("  Simplification successful!")
        else:
            if verbose:
                print("  Simplification failed, using original model")
    
    # Verify
    if verbose:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"  Model verified successfully!")
        
        # Model size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")
    
    return str(output_path)


def optimize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    optimization_level: str = "all",  # "basic", "extended", "all"
    target: str = "mobile",  # "mobile", "gpu", "cpu"
) -> str:
    """
    ONNX modelini optimize et.
    
    Optimizasyonlar:
    - Constant folding
    - Node fusion (Conv+BN, MatMul+Add)
    - Dead code elimination
    - Memory optimization
    """
    
    if not HAS_ORT:
        raise ImportError("ONNX Runtime not installed. Run: pip install onnxruntime")
    
    from onnxruntime.transformers import optimizer
    
    if output_path is None:
        output_path = input_path.replace(".onnx", "_optimized.onnx")
    
    # Optimization options
    opt_options = optimizer.FusionOptions(target)
    
    if target == "mobile":
        opt_options.enable_gelu = False  # Mobile'da GELU yok
        opt_options.enable_layer_norm = False
    
    # Optimize
    optimized_model = optimizer.optimize_model(
        input_path,
        optimization_options=opt_options,
        opt_level=99 if optimization_level == "all" else 1,
    )
    
    optimized_model.save_model_to_file(output_path)
    
    print(f"Optimized model saved to: {output_path}")
    
    return output_path


def quantize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    quantization_type: str = "dynamic",  # "dynamic", "static"
    calibration_data: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    ONNX modelini quantize et.
    
    Args:
        input_path: Input ONNX model
        output_path: Output path
        quantization_type: "dynamic" veya "static"
        calibration_data: Static quantization için calibration data
    """
    
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        CalibrationDataReader,
        QuantType,
    )
    
    if output_path is None:
        output_path = input_path.replace(".onnx", f"_{quantization_type}_int8.onnx")
    
    if quantization_type == "dynamic":
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
    else:
        # Static quantization
        class DataReader(CalibrationDataReader):
            def __init__(self, data):
                self.data = data
                self.idx = 0
            
            def get_next(self):
                if self.idx >= len(self.data):
                    return None
                result = self.data[self.idx]
                self.idx += 1
                return result
        
        if calibration_data is None:
            raise ValueError("Static quantization requires calibration_data")
        
        reader = DataReader(calibration_data)
        
        quantize_static(
            input_path,
            output_path,
            reader,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
    
    print(f"Quantized model saved to: {output_path}")
    
    return output_path


class ONNXInference:
    """
    ONNX Runtime ile inference.
    
    Test ve benchmark için kullanılabilir.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",  # "cpu", "cuda", "tensorrt"
        num_threads: int = 4,
    ):
        if not HAS_ORT:
            raise ImportError("ONNX Runtime not installed")
        
        self.model_path = model_path
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Providers
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
        
        # Input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference"""
        
        # Convert to numpy
        sample_np = sample.cpu().numpy()
        timestep_np = timestep.cpu().numpy()
        
        # Run
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: sample_np,
                self.input_names[1]: timestep_np,
            },
        )
        
        return torch.from_numpy(outputs[0])
    
    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 6, 256, 256),
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference latency"""
        
        import time
        import numpy as np
        
        sample = np.random.randn(*input_shape).astype(np.float32)
        timestep = np.zeros(input_shape[0], dtype=np.int64)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.session.run(
                self.output_names,
                {self.input_names[0]: sample, self.input_names[1]: timestep},
            )
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.session.run(
                self.output_names,
                {self.input_names[0]: sample, self.input_names[1]: timestep},
            )
            times.append(time.perf_counter() - start)
        
        return {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "min_latency_ms": np.min(times) * 1000,
            "max_latency_ms": np.max(times) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
        }


if __name__ == "__main__":
    print("ONNX Export Module")
    print(f"ONNX available: {HAS_ONNX}")
    print(f"ONNX Runtime available: {HAS_ORT}")

