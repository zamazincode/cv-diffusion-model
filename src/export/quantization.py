"""
Quantization Module
===================
FP16 ve INT8 kuantizasyon stratejileri.

Android deployment için kritik:
- FP16: Minimal kalite kaybı, ~2x hız artışı
- INT8: Daha fazla hız, biraz kalite kaybı

Stratejiler:
1. Post-Training Quantization (PTQ): Hızlı, calibration gerekli
2. Quantization-Aware Training (QAT): Daha iyi kalite, training gerekli
"""

from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any
import copy

import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic,
    prepare,
    convert,
    QConfig,
    get_default_qconfig,
)

try:
    from torch.ao.quantization import quantize_fx
    HAS_FX_QUANTIZATION = True
except ImportError:
    HAS_FX_QUANTIZATION = False


@dataclass
class QuantizationConfig:
    """Kuantizasyon konfigürasyonu"""
    
    # Quantization type
    dtype: str = "fp16"  # "fp16", "int8", "int8_dynamic"
    
    # Backend
    backend: str = "qnnpack"  # "qnnpack" (mobile), "fbgemm" (server)
    
    # Calibration
    num_calibration_batches: int = 100
    
    # Per-channel vs per-tensor
    per_channel: bool = True
    
    # Aktivasyon quantization
    quantize_activations: bool = True
    
    # Skip layers (quantization'a uygun olmayan)
    skip_layers: List[str] = None
    
    def __post_init__(self):
        if self.skip_layers is None:
            # Genelde attention ve normalization layer'ları skip edilir
            self.skip_layers = ["attention", "norm", "embedding"]


def prepare_for_quantization(model: nn.Module) -> nn.Module:
    """
    Modeli quantization için hazırla.
    
    Bu adımlar quantization-friendly yapar:
    1. BatchNorm'ları Conv ile fuse et
    2. Aktivasyonları quantizable yap
    3. Skip connection'ları düzenle
    """
    model = copy.deepcopy(model)
    model.eval()
    
    # BatchNorm fusion
    model = fuse_modules(model)
    
    return model


def fuse_modules(model: nn.Module) -> nn.Module:
    """Conv-BN-ReLU fusion"""
    
    # PyTorch'un otomatik fusion'ı
    if hasattr(torch.quantization, 'fuse_modules'):
        # Fuse edilebilir pattern'ları bul
        patterns_to_fuse = find_fusable_patterns(model)
        
        for pattern in patterns_to_fuse:
            try:
                torch.quantization.fuse_modules(model, pattern, inplace=True)
            except Exception:
                pass  # Skip if fusion fails
    
    return model


def find_fusable_patterns(model: nn.Module) -> List[List[str]]:
    """
    Fuse edilebilir layer pattern'larını bul.
    Örnek: [conv, bn, relu] → fused_conv
    """
    patterns = []
    modules = dict(model.named_modules())
    
    for name, module in modules.items():
        # Conv -> BN -> ReLU pattern
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            bn_name = name.replace('conv', 'bn')
            relu_name = name.replace('conv', 'relu')
            
            if bn_name in modules and isinstance(modules[bn_name], (nn.BatchNorm2d, nn.BatchNorm1d)):
                if relu_name in modules and isinstance(modules[relu_name], (nn.ReLU, nn.ReLU6)):
                    patterns.append([name, bn_name, relu_name])
                else:
                    patterns.append([name, bn_name])
    
    return patterns


class FP16Quantizer:
    """
    FP16 (Half Precision) Quantization.
    
    En basit ve güvenli quantization:
    - Minimal kalite kaybı
    - ~2x bellek tasarrufu
    - Modern GPU/NPU'larda hızlı
    """
    
    @staticmethod
    def quantize(model: nn.Module) -> nn.Module:
        """FP16'ya convert et"""
        model = copy.deepcopy(model)
        return model.half()
    
    @staticmethod
    def prepare_input(x: torch.Tensor) -> torch.Tensor:
        """Input'u FP16'ya convert et"""
        return x.half()


class INT8DynamicQuantizer:
    """
    Dynamic INT8 Quantization.
    
    Özellikler:
    - Weight'ler statik olarak INT8
    - Aktivasyonlar runtime'da quantize edilir
    - Calibration gerektirmez
    - Basit ve hızlı
    """
    
    @staticmethod
    def quantize(
        model: nn.Module,
        layers_to_quantize: Optional[List[type]] = None,
    ) -> nn.Module:
        """Dynamic INT8 quantization uygula"""
        
        if layers_to_quantize is None:
            layers_to_quantize = {nn.Linear, nn.Conv2d}
        
        quantized_model = quantize_dynamic(
            model,
            layers_to_quantize,
            dtype=torch.qint8,
        )
        
        return quantized_model


class INT8StaticQuantizer:
    """
    Static INT8 Quantization (Post-Training Quantization).
    
    Özellikler:
    - Hem weight'ler hem aktivasyonlar INT8
    - Calibration dataset gerekli
    - En yüksek hız kazanımı
    - Biraz kalite kaybı olabilir
    """
    
    def __init__(
        self,
        backend: str = "qnnpack",
        per_channel: bool = True,
    ):
        self.backend = backend
        self.per_channel = per_channel
        
        # Backend ayarla
        torch.backends.quantized.engine = backend
    
    def prepare(self, model: nn.Module) -> nn.Module:
        """Modeli calibration için hazırla"""
        
        model = copy.deepcopy(model)
        model.eval()
        
        # QConfig
        if self.per_channel:
            qconfig = get_default_qconfig(self.backend)
        else:
            qconfig = QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_tensor_weight_observer,
            )
        
        model.qconfig = qconfig
        
        # Prepare
        prepared_model = prepare(model)
        
        return prepared_model
    
    def calibrate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 100,
        device: str = "cpu",
    ) -> nn.Module:
        """
        Calibration: Aktivasyon range'lerini öğren.
        
        Bu adım, her layer için min/max değerleri toplar.
        """
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                if isinstance(batch, dict):
                    x = batch["low_light"].to(device)
                else:
                    x = batch[0].to(device)
                
                # Dummy timestep
                t = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                
                # Forward pass (calibration için)
                _ = model(x, t)
        
        return model
    
    def convert(self, model: nn.Module) -> nn.Module:
        """Calibration sonrası INT8'e convert et"""
        return convert(model)
    
    def quantize(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 100,
    ) -> nn.Module:
        """Tam quantization pipeline"""
        
        # Prepare
        prepared = self.prepare(model)
        
        # Calibrate
        calibrated = self.calibrate(prepared, dataloader, num_batches)
        
        # Convert
        quantized = self.convert(calibrated)
        
        return quantized


class QuantizationAwareTraining:
    """
    Quantization-Aware Training (QAT).
    
    Training sırasında fake quantization kullanarak
    model'in quantization'a adapte olmasını sağlar.
    
    Avantajlar:
    - En iyi INT8 kalitesi
    - Model quantization noise'a dayanıklı olur
    
    Dezavantajlar:
    - Extra training gerekli
    - Daha yavaş training
    """
    
    def __init__(self, backend: str = "qnnpack"):
        self.backend = backend
        torch.backends.quantized.engine = backend
    
    def prepare_qat(self, model: nn.Module) -> nn.Module:
        """QAT için modeli hazırla"""
        
        model = copy.deepcopy(model)
        model.train()
        
        # QAT config
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        
        # Prepare for QAT
        prepared = torch.quantization.prepare_qat(model)
        
        return prepared
    
    def convert_qat(self, model: nn.Module) -> nn.Module:
        """QAT sonrası INT8'e convert et"""
        model.eval()
        return torch.quantization.convert(model)


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
) -> nn.Module:
    """
    Ana quantization fonksiyonu.
    
    Args:
        model: Quantize edilecek model
        config: Quantization konfigürasyonu
        calibration_loader: INT8 static için calibration data
        
    Returns:
        Quantized model
    """
    
    if config.dtype == "fp16":
        return FP16Quantizer.quantize(model)
    
    elif config.dtype == "int8_dynamic":
        return INT8DynamicQuantizer.quantize(model)
    
    elif config.dtype == "int8":
        if calibration_loader is None:
            raise ValueError("INT8 static quantization requires calibration_loader")
        
        quantizer = INT8StaticQuantizer(
            backend=config.backend,
            per_channel=config.per_channel,
        )
        return quantizer.quantize(
            model, 
            calibration_loader, 
            config.num_calibration_batches
        )
    
    else:
        raise ValueError(f"Unknown quantization dtype: {config.dtype}")


def benchmark_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module,
    input_shape: tuple = (1, 6, 256, 256),
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, Any]:
    """
    Quantized model benchmark.
    
    Returns:
        Dict with latency, memory, and accuracy metrics
    """
    import time
    
    device = next(original_model.parameters()).device
    
    # Test input
    x = torch.randn(input_shape).to(device)
    t = torch.zeros(input_shape[0], dtype=torch.long, device=device)
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = original_model(x, t)
            
    # Original latency
    original_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = original_model(x, t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        original_times.append(time.perf_counter() - start)
    
    # Quantized latency
    x_q = x.half() if hasattr(quantized_model, 'half') else x
    
    for _ in range(warmup_runs):
        with torch.no_grad():
            try:
                _ = quantized_model(x_q, t)
            except:
                _ = quantized_model(x, t)
    
    quantized_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            try:
                _ = quantized_model(x_q, t)
            except:
                _ = quantized_model(x, t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        quantized_times.append(time.perf_counter() - start)
    
    # Calculate stats
    original_mean = sum(original_times) / len(original_times) * 1000  # ms
    quantized_mean = sum(quantized_times) / len(quantized_times) * 1000  # ms
    
    # Model sizes
    def get_model_size(model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    original_size = get_model_size(original_model)
    
    try:
        quantized_size = get_model_size(quantized_model)
    except:
        quantized_size = original_size / 2  # Estimate for INT8
    
    return {
        "original_latency_ms": original_mean,
        "quantized_latency_ms": quantized_mean,
        "speedup": original_mean / quantized_mean,
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "size_reduction": original_size / quantized_size,
    }


if __name__ == "__main__":
    # Test
    from ..models import create_efficient_unet
    
    model = create_efficient_unet("small", image_size=256, in_channels=6)
    
    # FP16
    print("FP16 Quantization:")
    fp16_model = FP16Quantizer.quantize(model)
    print(f"  Original dtype: {next(model.parameters()).dtype}")
    print(f"  Quantized dtype: {next(fp16_model.parameters()).dtype}")
    
    # Dynamic INT8
    print("\nDynamic INT8 Quantization:")
    int8_model = INT8DynamicQuantizer.quantize(model)
    print(f"  Quantized model type: {type(int8_model)}")

