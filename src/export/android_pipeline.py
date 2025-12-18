"""
Android Inference Pipeline
==========================
Android cihazlarda düşük ışık iyileştirmesi için tam inference pipeline.

Bu modül:
1. Model yükleme ve initialization
2. Pre/post processing
3. LCM denoising loop
4. Memory-efficient inference
5. Benchmark tools
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class AndroidPipelineConfig:
    """Android inference konfigürasyonu"""
    
    # Model
    model_path: str = "model.tflite"
    model_format: str = "tflite"  # "tflite", "onnx"
    
    # Input/Output
    image_size: int = 256
    in_channels: int = 6  # image + condition
    out_channels: int = 3
    
    # LCM
    num_inference_steps: int = 4
    
    # Quantization
    use_fp16: bool = True
    
    # Performance
    num_threads: int = 4
    use_gpu: bool = False
    use_nnapi: bool = False
    
    # Memory
    enable_memory_optimization: bool = True
    
    # Noise schedule (precomputed)
    alphas_cumprod: Optional[List[float]] = None


class PreProcessor:
    """
    Android için görüntü ön işleme.
    
    Operations:
    1. Resize to target size
    2. Normalize to [-1, 1]
    3. Convert to model input format
    """
    
    def __init__(
        self,
        target_size: int = 256,
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.normalize = normalize
    
    def __call__(
        self,
        image: np.ndarray,  # [H, W, 3] uint8
        keep_aspect_ratio: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Görüntüyü model için hazırla.
        
        Args:
            image: Input görüntü [H, W, 3] uint8
            keep_aspect_ratio: Aspect ratio koru
            
        Returns:
            processed: [1, 3, target_size, target_size] float32
            metadata: Orijinal boyut bilgisi (postprocess için)
        """
        import cv2
        
        original_size = image.shape[:2]
        
        if keep_aspect_ratio:
            # Aspect ratio koruyarak resize
            h, w = image.shape[:2]
            scale = self.target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Padding
            pad_h = self.target_size - new_h
            pad_w = self.target_size - new_w
            
            padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            padded[:new_h, :new_w] = resized
            
            image = padded
            
            metadata = {
                "original_size": original_size,
                "resized_size": (new_h, new_w),
                "padding": (pad_h, pad_w),
                "scale": scale,
            }
        else:
            image = cv2.resize(image, (self.target_size, self.target_size))
            metadata = {
                "original_size": original_size,
                "resized_size": (self.target_size, self.target_size),
                "padding": (0, 0),
                "scale": 1.0,
            }
        
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        
        # Normalize to [-1, 1]
        if self.normalize:
            image = image.astype(np.float32) / 127.5 - 1.0
        else:
            image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = image[np.newaxis, ...]  # [1, 3, H, W]
        
        return image, metadata


class PostProcessor:
    """
    Model çıktısını görüntüye dönüştür.
    """
    
    def __call__(
        self,
        output: np.ndarray,  # [1, 3, H, W] float32
        metadata: Dict[str, Any],
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Model çıktısını görüntüye dönüştür.
        
        Args:
            output: Model çıktısı [1, 3, H, W]
            metadata: Preprocessing metadata
            denormalize: [-1, 1] → [0, 255]
            
        Returns:
            image: [H, W, 3] uint8
        """
        import cv2
        
        # Remove batch dimension
        output = output[0]  # [3, H, W]
        
        # CHW -> HWC
        output = output.transpose(1, 2, 0)  # [H, W, 3]
        
        # Denormalize
        if denormalize:
            output = (output + 1.0) * 127.5
        else:
            output = output * 255.0
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Remove padding
        if metadata["padding"] != (0, 0):
            new_h, new_w = metadata["resized_size"]
            output = output[:new_h, :new_w]
        
        # Resize to original
        original_h, original_w = metadata["original_size"]
        output = cv2.resize(output, (original_w, original_h))
        
        return output


class LCMDenoisingLoop:
    """
    LCM denoising loop - 4-8 adımda inference.
    
    Bu sınıf noise schedule ve denoising mantığını içerir.
    Model-agnostic, farklı backend'lerle kullanılabilir.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 4,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # Beta schedule (scaled linear)
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # LCM timesteps
        self.timesteps = self._get_lcm_timesteps()
    
    def _get_lcm_timesteps(self) -> np.ndarray:
        """LCM timestep'lerini hesapla"""
        original_steps = 50
        c = self.num_train_timesteps // original_steps
        
        lcm_timesteps = np.arange(1, original_steps + 1) * c - 1
        
        skip = len(lcm_timesteps) // self.num_inference_steps
        timesteps = lcm_timesteps[::skip][:self.num_inference_steps]
        
        return timesteps[::-1]  # Reverse
    
    def add_noise(
        self,
        original: np.ndarray,
        noise: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        """Görüntüye noise ekle"""
        
        sqrt_alpha = np.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alpha = np.sqrt(1 - self.alphas_cumprod[timestep])
        
        return sqrt_alpha * original + sqrt_one_minus_alpha * noise
    
    def step(
        self,
        noise_pred: np.ndarray,
        timestep: int,
        sample: np.ndarray,
    ) -> np.ndarray:
        """Tek denoising adımı"""
        
        # Current timestep index
        t_idx = np.where(self.timesteps == timestep)[0][0]
        
        # Next timestep
        if t_idx + 1 < len(self.timesteps):
            prev_t = self.timesteps[t_idx + 1]
        else:
            prev_t = 0
        
        # Alpha values
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else self.alphas_cumprod[0]
        
        # x_0 prediction
        x0_pred = (sample - np.sqrt(1 - alpha_t) * noise_pred) / np.sqrt(alpha_t)
        
        # Clamp
        x0_pred = np.clip(x0_pred, -1, 1)
        
        # Final step
        if prev_t == 0:
            return x0_pred
        
        # Next sample
        noise = np.random.randn(*sample.shape).astype(np.float32)
        prev_sample = np.sqrt(alpha_prev) * x0_pred + np.sqrt(1 - alpha_prev) * noise
        
        return prev_sample


class AndroidInferencePipeline:
    """
    Android için tam inference pipeline.
    
    Kullanım:
    ```python
    pipeline = AndroidInferencePipeline(config)
    enhanced_image = pipeline(low_light_image)
    ```
    """
    
    def __init__(self, config: AndroidPipelineConfig):
        self.config = config
        
        # Preprocessor
        self.preprocessor = PreProcessor(config.image_size)
        
        # Postprocessor
        self.postprocessor = PostProcessor()
        
        # Denoising loop
        self.denoising_loop = LCMDenoisingLoop(
            num_inference_steps=config.num_inference_steps
        )
        
        # Model loader (lazy loading)
        self._model = None
    
    @property
    def model(self):
        """Lazy model loading"""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """Model yükle"""
        
        if self.config.model_format == "tflite":
            from .tflite_export import TFLiteInference
            return TFLiteInference(
                self.config.model_path,
                num_threads=self.config.num_threads,
                use_gpu=self.config.use_gpu,
                use_nnapi=self.config.use_nnapi,
            )
        elif self.config.model_format == "onnx":
            from .onnx_export import ONNXInference
            return ONNXInference(
                self.config.model_path,
                device="cpu",
                num_threads=self.config.num_threads,
            )
        else:
            raise ValueError(f"Unknown model format: {self.config.model_format}")
    
    def __call__(
        self,
        low_light_image: np.ndarray,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Düşük ışıklı görüntüyü iyileştir.
        
        Args:
            low_light_image: Input görüntü [H, W, 3] uint8
            num_inference_steps: LCM adım sayısı (default: config)
            seed: Random seed (reproducibility için)
            
        Returns:
            enhanced: Enhanced görüntü [H, W, 3] uint8
        """
        
        if seed is not None:
            np.random.seed(seed)
        
        if num_inference_steps is not None:
            self.denoising_loop.num_inference_steps = num_inference_steps
            self.denoising_loop.timesteps = self.denoising_loop._get_lcm_timesteps()
        
        # Preprocess
        condition, metadata = self.preprocessor(low_light_image)
        
        # Initialize with pure noise
        latents = np.random.randn(
            1, 3, self.config.image_size, self.config.image_size
        ).astype(np.float32)
        
        # Denoising loop
        for timestep in self.denoising_loop.timesteps:
            # Concat condition
            model_input = np.concatenate([latents, condition], axis=1)
            
            # Timestep tensor
            t = np.array([timestep], dtype=np.int64)
            
            # Model inference
            noise_pred = self.model(model_input, t)
            
            # Denoising step
            latents = self.denoising_loop.step(noise_pred, timestep, latents)
        
        # Clamp
        latents = np.clip(latents, -1, 1)
        
        # Postprocess
        enhanced = self.postprocessor(latents, metadata)
        
        return enhanced
    
    def benchmark(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_runs: int = 20,
        warmup_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Pipeline benchmark.
        
        Returns:
            Dict with latency metrics
        """
        
        # Test image
        test_image = np.random.randint(
            0, 255, (*image_size, 3), dtype=np.uint8
        )
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self(test_image)
        
        # Benchmark
        total_times = []
        preprocess_times = []
        inference_times = []
        postprocess_times = []
        
        for _ in range(num_runs):
            # Preprocess timing
            start = time.perf_counter()
            condition, metadata = self.preprocessor(test_image)
            preprocess_times.append(time.perf_counter() - start)
            
            # Inference timing
            start = time.perf_counter()
            
            latents = np.random.randn(
                1, 3, self.config.image_size, self.config.image_size
            ).astype(np.float32)
            
            for timestep in self.denoising_loop.timesteps:
                model_input = np.concatenate([latents, condition], axis=1)
                t = np.array([timestep], dtype=np.int64)
                noise_pred = self.model(model_input, t)
                latents = self.denoising_loop.step(noise_pred, timestep, latents)
            
            inference_times.append(time.perf_counter() - start)
            
            # Postprocess timing
            start = time.perf_counter()
            _ = self.postprocessor(np.clip(latents, -1, 1), metadata)
            postprocess_times.append(time.perf_counter() - start)
            
            total_times.append(
                preprocess_times[-1] + inference_times[-1] + postprocess_times[-1]
            )
        
        return {
            "total_latency_ms": np.mean(total_times) * 1000,
            "preprocess_latency_ms": np.mean(preprocess_times) * 1000,
            "inference_latency_ms": np.mean(inference_times) * 1000,
            "postprocess_latency_ms": np.mean(postprocess_times) * 1000,
            "throughput_fps": 1.0 / np.mean(total_times),
            "per_step_latency_ms": np.mean(inference_times) * 1000 / self.config.num_inference_steps,
            "num_inference_steps": self.config.num_inference_steps,
        }


def create_android_package(
    model: nn.Module,
    output_dir: str,
    config: Optional[AndroidPipelineConfig] = None,
) -> str:
    """
    Android deployment için komple paket oluştur.
    
    Oluşturulan dosyalar:
    - model.tflite: Quantized model
    - model_config.json: Model konfigürasyonu
    - noise_schedule.npy: Precomputed noise schedule
    """
    
    from .tflite_export import export_to_tflite
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = config or AndroidPipelineConfig()
    
    # 1. Export model
    model_path = output_dir / "model.tflite"
    export_to_tflite(
        model,
        str(model_path),
        image_size=config.image_size,
        in_channels=config.in_channels,
        quantize=config.use_fp16,
    )
    
    # 2. Save config
    config_dict = {
        "image_size": config.image_size,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "num_inference_steps": config.num_inference_steps,
        "use_fp16": config.use_fp16,
    }
    
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # 3. Save noise schedule
    denoising_loop = LCMDenoisingLoop(
        num_inference_steps=config.num_inference_steps
    )
    
    np.savez(
        output_dir / "noise_schedule.npz",
        alphas_cumprod=denoising_loop.alphas_cumprod,
        timesteps=denoising_loop.timesteps,
    )
    
    print(f"Android package created at: {output_dir}")
    print(f"Contents:")
    for f in output_dir.iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name}: {size_kb:.1f} KB")
    
    return str(output_dir)


if __name__ == "__main__":
    # Test denoising loop
    loop = LCMDenoisingLoop(num_inference_steps=4)
    print(f"LCM Timesteps: {loop.timesteps}")
    print(f"Alphas cumprod shape: {loop.alphas_cumprod.shape}")

