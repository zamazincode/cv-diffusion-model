"""
TFLite Export Module
====================
Android için TensorFlow Lite formatına export.

TFLite Avantajları:
- Android native desteği
- GPU Delegate (Adreno, Mali)
- NNAPI Delegate (Android 8.1+)
- Hexagon DSP Delegate (Qualcomm)
- INT8 quantization desteği
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import tempfile
import copy

import torch
import torch.nn as nn
import numpy as np

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import ai_edge_torch
    HAS_AI_EDGE = True
except ImportError:
    HAS_AI_EDGE = False


class TFLiteExportWrapper(nn.Module):
    """TFLite export için model wrapper"""
    
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
    
    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.unet(sample, timestep)


def export_via_onnx(
    model: nn.Module,
    output_path: str,
    image_size: int = 256,
    in_channels: int = 6,
    quantize: bool = True,
    representative_dataset: Optional[List[np.ndarray]] = None,
) -> str:
    """
    ONNX üzerinden TFLite'a export.
    
    PyTorch → ONNX → TF SavedModel → TFLite
    """
    
    if not HAS_TF:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
    
    from .onnx_export import export_to_onnx
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. ONNX export
        onnx_path = Path(tmpdir) / "model.onnx"
        export_to_onnx(
            model,
            str(onnx_path),
            image_size=image_size,
            in_channels=in_channels,
            dynamic_batch=False,  # TFLite static batch gerektirir
            verbose=False,
        )
        
        # 2. ONNX → TF SavedModel
        try:
            import onnx
            from onnx_tf.backend import prepare
            
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            
            saved_model_path = Path(tmpdir) / "saved_model"
            tf_rep.export_graph(str(saved_model_path))
        except ImportError:
            raise ImportError("onnx-tf not installed. Run: pip install onnx-tf")
        
        # 3. TF SavedModel → TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        # Optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    return str(output_path)


def export_via_ai_edge_torch(
    model: nn.Module,
    output_path: str,
    image_size: int = 256,
    in_channels: int = 6,
    batch_size: int = 1,
) -> str:
    """
    Google AI Edge Torch ile doğrudan TFLite export.
    
    Bu yöntem daha yeni ve PyTorch modellerini doğrudan
    TFLite'a convert edebilir (ONNX aracı olmadan).
    """
    
    if not HAS_AI_EDGE:
        raise ImportError("ai-edge-torch not installed. Run: pip install ai-edge-torch")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Model hazırlığı
    model = copy.deepcopy(model)
    model.eval()
    
    if hasattr(model, 'unet'):
        wrapper = TFLiteExportWrapper(model.unet)
    else:
        wrapper = TFLiteExportWrapper(model)
    
    # Sample inputs
    sample_inputs = (
        torch.randn(batch_size, in_channels, image_size, image_size),
        torch.zeros(batch_size, dtype=torch.long),
    )
    
    # Convert
    edge_model = ai_edge_torch.convert(wrapper, sample_inputs)
    
    # Export
    edge_model.export(str(output_path))
    
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    return str(output_path)


def export_to_tflite(
    model: nn.Module,
    output_path: str,
    image_size: int = 256,
    in_channels: int = 6,
    method: str = "auto",  # "auto", "onnx", "ai_edge"
    quantize: bool = True,
    quantization_type: str = "fp16",  # "fp16", "int8", "dynamic"
    representative_dataset: Optional[List[np.ndarray]] = None,
) -> str:
    """
    Ana TFLite export fonksiyonu.
    
    Args:
        model: Export edilecek model
        output_path: Çıktı dosya yolu (.tflite)
        image_size: Görüntü boyutu
        in_channels: Giriş kanal sayısı
        method: Export yöntemi
        quantize: Quantization uygula
        quantization_type: Quantization tipi
        representative_dataset: INT8 için calibration data
        
    Returns:
        Kaydedilen dosya yolu
    """
    
    # Yöntem seçimi
    if method == "auto":
        if HAS_AI_EDGE:
            method = "ai_edge"
        elif HAS_TF:
            method = "onnx"
        else:
            raise ImportError("No TFLite export method available. Install tensorflow or ai-edge-torch")
    
    if method == "ai_edge":
        return export_via_ai_edge_torch(model, output_path, image_size, in_channels)
    else:
        return export_via_onnx(
            model, output_path, image_size, in_channels,
            quantize=quantize,
            representative_dataset=representative_dataset,
        )


def add_metadata_to_tflite(
    model_path: str,
    output_path: Optional[str] = None,
    model_name: str = "LowLightEnhancer",
    model_description: str = "Low-light image enhancement model",
    input_description: str = "RGB image concatenated with condition",
    output_description: str = "Enhanced RGB image",
) -> str:
    """
    TFLite modeline metadata ekle.
    
    Android'de model bilgilerini göstermek için kullanılır.
    """
    
    if not HAS_TF:
        raise ImportError("TensorFlow not installed")
    
    from tflite_support import metadata as _metadata
    from tflite_support import metadata_schema_py_generated as _metadata_fb
    
    if output_path is None:
        output_path = model_path.replace(".tflite", "_with_metadata.tflite")
    
    # Model metadata
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = model_name
    model_meta.description = model_description
    
    # Input metadata
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "input"
    input_meta.description = input_description
    
    # Output metadata
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "output"
    output_meta.description = output_description
    
    # Subgraph metadata
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]
    
    # Populate metadata
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
    )
    metadata_buf = b.Output()
    
    # Write to model
    populator = _metadata.MetadataPopulator.with_model_file(model_path)
    populator.load_metadata_buffer(bytes(metadata_buf))
    populator.populate()
    
    # Save
    import shutil
    shutil.copy(model_path, output_path)
    
    return output_path


class TFLiteInference:
    """
    TFLite ile inference (test için).
    """
    
    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        use_gpu: bool = False,
        use_nnapi: bool = False,
    ):
        if not HAS_TF:
            raise ImportError("TensorFlow not installed")
        
        # Interpreter options
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        
        # GPU delegate
        if use_gpu:
            try:
                gpu_delegate = tf.lite.experimental.load_delegate('libdelegate.so')
                self.interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[gpu_delegate],
                )
            except Exception as e:
                print(f"GPU delegate failed: {e}, using CPU")
        
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def __call__(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
    ) -> np.ndarray:
        """Run inference"""
        
        # Set inputs
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            sample.astype(np.float32)
        )
        self.interpreter.set_tensor(
            self.input_details[1]['index'],
            timestep.astype(np.int64)
        )
        
        # Run
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output
    
    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 6, 256, 256),
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference latency"""
        
        import time
        
        sample = np.random.randn(*input_shape).astype(np.float32)
        timestep = np.zeros(input_shape[0], dtype=np.int64)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self(sample, timestep)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self(sample, timestep)
            times.append(time.perf_counter() - start)
        
        return {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "min_latency_ms": np.min(times) * 1000,
            "max_latency_ms": np.max(times) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
        }


if __name__ == "__main__":
    print("TFLite Export Module")
    print(f"TensorFlow available: {HAS_TF}")
    print(f"AI Edge Torch available: {HAS_AI_EDGE}")

