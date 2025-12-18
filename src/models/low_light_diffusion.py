"""
Low-Light Enhancement Diffusion Model
======================================
Düşük ışık görüntülerini iyileştirmek için tasarlanmış diffusion modeli.

Özellikler:
- Conditional diffusion (low-light image as condition)
- LCM training support for fast inference
- Image-to-image denoising (not text-to-image)
- Optimized for mobile deployment
"""

from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficient_unet import EfficientUNet, EfficientUNetConfig, create_efficient_unet
from .lcm_scheduler import LCMScheduler


@dataclass
class LowLightDiffusionOutput:
    """Model çıktısı"""
    enhanced: torch.Tensor  # Enhanced image
    intermediate: Optional[list] = None  # Intermediate steps (debug için)


class LowLightDiffusion(nn.Module):
    """
    Low-Light Enhancement için Conditional Diffusion Model.
    
    Çalışma Prensibi:
    1. Low-light görüntü condition olarak verilir
    2. Model, normal ışıklı görüntüyü tahmin eder
    3. Diffusion process noise → enhanced image
    
    Training:
    - Input: (low_light_image, normal_light_image) çiftleri
    - Forward process: normal_light → noisy
    - Reverse process: noisy + low_light_condition → normal_light
    
    Inference:
    - Input: low_light_image
    - Output: enhanced_image (4-8 LCM steps)
    """
    
    def __init__(
        self,
        unet: Optional[EfficientUNet] = None,
        scheduler: Optional[LCMScheduler] = None,
        unet_variant: str = "small",
        image_size: int = 256,
        num_inference_steps: int = 4,
        condition_mode: str = "concat",  # "concat" veya "add"
    ):
        """
        Args:
            unet: EfficientUNet modeli (None ise otomatik oluşturulur)
            scheduler: LCM Scheduler (None ise otomatik oluşturulur)
            unet_variant: UNet varyantı ("tiny", "small", "base", "large")
            image_size: Training görüntü boyutu
            num_inference_steps: LCM inference adım sayısı
            condition_mode: Conditioning yöntemi
                - "concat": Low-light görüntüyü input'a concat et (6 kanal)
                - "add": Low-light görüntüyü latent'e ekle
        """
        super().__init__()
        
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.condition_mode = condition_mode
        
        # Conditioning için input kanallarını ayarla
        in_channels = 6 if condition_mode == "concat" else 3
        
        # UNet
        if unet is None:
            config = EfficientUNetConfig(
                in_channels=in_channels,
                out_channels=3,
                image_size=image_size,
            )
            # Variant'a göre ayarla
            self.unet = create_efficient_unet(
                variant=unet_variant,
                image_size=image_size,
                in_channels=in_channels,
            )
        else:
            self.unet = unet
        
        # Scheduler
        if scheduler is None:
            self.scheduler = LCMScheduler(
                num_train_timesteps=1000,
                beta_schedule="scaled_linear",
                prediction_type="epsilon",
                num_inference_steps=num_inference_steps,
                rescale_betas_zero_snr=True,  # Karanlık görüntüler için önemli
            )
        else:
            self.scheduler = scheduler
        
        # Condition encoder (add mode için)
        if condition_mode == "add":
            self.condition_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 3, 3, padding=1),
            )
    
    def forward(
        self,
        low_light: torch.Tensor,
        normal_light: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Training forward pass.
        
        Args:
            low_light: Low-light input görüntü [B, 3, H, W]
            normal_light: Normal-light target görüntü [B, 3, H, W]
            timesteps: Diffusion timesteps [B]
            noise: Pre-generated noise (optional)
            
        Returns:
            Training modunda: predicted noise ve loss
            Inference modunda: enhanced görüntü
        """
        batch_size = low_light.shape[0]
        device = low_light.device
        
        # Training mode
        if normal_light is not None:
            # Random timesteps
            if timesteps is None:
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (batch_size,), device=device
                )
            
            # Random noise
            if noise is None:
                noise = torch.randn_like(normal_light)
            
            # Noisy normal-light image
            noisy_image = self.scheduler.add_noise(normal_light, noise, timesteps)
            
            # Conditioning
            if self.condition_mode == "concat":
                model_input = torch.cat([noisy_image, low_light], dim=1)
            else:
                condition_feat = self.condition_encoder(low_light)
                model_input = noisy_image + condition_feat
            
            # Predict noise
            noise_pred = self.unet(model_input, timesteps)
            
            if return_dict:
                return {
                    "noise_pred": noise_pred,
                    "noise": noise,
                    "timesteps": timesteps,
                }
            return noise_pred
        
        # Inference mode
        else:
            return self.enhance(low_light)
    
    @torch.no_grad()
    def enhance(
        self,
        low_light: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, LowLightDiffusionOutput]:
        """
        Low-light görüntüyü iyileştir (inference).
        
        Args:
            low_light: Low-light input [B, 3, H, W], değerler [-1, 1] arasında
            num_inference_steps: LCM adım sayısı (default: model config)
            generator: Random generator (reproducibility için)
            return_intermediate: Ara adımları döndür
            
        Returns:
            Enhanced görüntü [B, 3, H, W]
        """
        device = low_light.device
        batch_size = low_light.shape[0]
        
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        
        # Scheduler timesteps ayarla
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Pure noise'dan başla
        latents = torch.randn(
            batch_size, 3, self.image_size, self.image_size,
            device=device, generator=generator
        )
        
        intermediate_results = []
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Timestep tensor
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Conditioning
            if self.condition_mode == "concat":
                model_input = torch.cat([latents, low_light], dim=1)
            else:
                condition_feat = self.condition_encoder(low_light)
                model_input = latents + condition_feat
            
            # Predict noise
            noise_pred = self.unet(model_input, t_tensor)
            
            # Scheduler step
            output = self.scheduler.step(
                noise_pred, t, latents, generator=generator
            )
            latents = output.prev_sample
            
            if return_intermediate:
                intermediate_results.append(latents.clone())
        
        # Clamp to valid range
        enhanced = latents.clamp(-1, 1)
        
        if return_intermediate:
            return LowLightDiffusionOutput(
                enhanced=enhanced,
                intermediate=intermediate_results,
            )
        
        return enhanced
    
    def compute_loss(
        self,
        low_light: torch.Tensor,
        normal_light: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """
        Training loss hesapla.
        
        Args:
            low_light: Low-light input
            normal_light: Normal-light target
            loss_type: "mse", "huber", veya "l1"
        """
        output = self.forward(low_light, normal_light)
        noise_pred = output["noise_pred"]
        noise = output["noise"]
        
        if loss_type == "mse":
            loss = F.mse_loss(noise_pred, noise)
        elif loss_type == "huber":
            loss = F.huber_loss(noise_pred, noise)
        elif loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    def get_model_size(self) -> Dict[str, float]:
        """Model boyutunu döndür"""
        return self.unet.get_memory_footprint()


class LowLightLCMDistillation(nn.Module):
    """
    LCM Distillation için wrapper.
    
    Teacher model'den student model'e consistency distillation.
    Daha az adımda daha iyi sonuç için.
    """
    
    def __init__(
        self,
        teacher_model: LowLightDiffusion,
        student_model: LowLightDiffusion,
        num_ddim_timesteps: int = 50,
        guidance_scale_range: Tuple[float, float] = (3.0, 15.0),
    ):
        super().__init__()
        
        self.teacher = teacher_model
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        
        self.student = student_model
        
        self.num_ddim_timesteps = num_ddim_timesteps
        self.guidance_scale_range = guidance_scale_range
        
        # EMA student (target model)
        import copy
        self.ema_student = copy.deepcopy(student_model)
        self.ema_student.eval()
        self.ema_student.requires_grad_(False)
    
    @torch.no_grad()
    def update_ema(self, decay: float = 0.95):
        """EMA model güncelle"""
        for ema_param, student_param in zip(
            self.ema_student.parameters(),
            self.student.parameters(),
        ):
            ema_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)
    
    def consistency_distillation_loss(
        self,
        low_light: torch.Tensor,
        normal_light: torch.Tensor,
        num_inference_steps: int = 4,
    ) -> torch.Tensor:
        """
        Consistency distillation loss.
        
        Student model, teacher'ın bir adımda gittiği yere
        consistency ile ulaşmayı öğrenir.
        """
        batch_size = low_light.shape[0]
        device = low_light.device
        
        # Random noise
        noise = torch.randn_like(normal_light)
        
        # Teacher scheduler için timestep çiftleri
        c = self.teacher.scheduler.config.num_train_timesteps // self.num_ddim_timesteps
        k = self.num_ddim_timesteps // num_inference_steps  # Skip step
        
        # Random timestep seç
        idx = torch.randint(0, self.num_ddim_timesteps - k, (batch_size,), device=device)
        
        t = idx * c + c - 1  # Current timestep
        t_next = (idx + k) * c + c - 1  # Next timestep (after k DDIM steps)
        
        # Noisy samples
        x_t = self.teacher.scheduler.add_noise(normal_light, noise, t)
        
        # Teacher: t'den t_next'e DDIM step (veya çoklu step)
        with torch.no_grad():
            # Teacher prediction at t
            if self.teacher.condition_mode == "concat":
                teacher_input = torch.cat([x_t, low_light], dim=1)
            else:
                condition_feat = self.teacher.condition_encoder(low_light)
                teacher_input = x_t + condition_feat
            
            teacher_noise_pred = self.teacher.unet(teacher_input, t)
            
            # DDIM step to get x_{t_next}
            alpha_t = self.teacher.scheduler.alphas_cumprod[t]
            alpha_t_next = self.teacher.scheduler.alphas_cumprod[t_next]
            
            # Reshape for broadcasting
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            alpha_t_next = alpha_t_next.view(-1, 1, 1, 1)
            
            # x_0 prediction
            x_0_pred = (x_t - (1 - alpha_t).sqrt() * teacher_noise_pred) / alpha_t.sqrt()
            
            # DDIM deterministic step
            x_t_next = alpha_t_next.sqrt() * x_0_pred + (1 - alpha_t_next).sqrt() * teacher_noise_pred
        
        # Student: doğrudan consistency prediction
        # x_t için student prediction
        if self.student.condition_mode == "concat":
            student_input_t = torch.cat([x_t, low_light], dim=1)
            student_input_t_next = torch.cat([x_t_next, low_light], dim=1)
        else:
            condition_feat = self.student.condition_encoder(low_light)
            student_input_t = x_t + condition_feat
            student_input_t_next = x_t_next + condition_feat
        
        # Student predictions (farklı timestep'lerde)
        student_pred_t = self.student.unet(student_input_t, t)
        
        # EMA student prediction (target)
        with torch.no_grad():
            target_pred = self.ema_student.unet(student_input_t_next, t_next)
        
        # Consistency loss: student(x_t, t) should match target(x_{t_next}, t_next)
        # Her iki prediction da aynı x_0'ı predict etmeli
        
        # x_0 predictions
        student_x0 = (x_t - (1 - alpha_t).sqrt() * student_pred_t) / alpha_t.sqrt()
        target_x0 = (x_t_next - (1 - alpha_t_next).sqrt() * target_pred) / alpha_t_next.sqrt()
        
        # Huber loss (LCM paper recommendation)
        loss = F.huber_loss(student_x0, target_x0)
        
        return loss


# Utility functions
def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """[0, 1] → [-1, 1]"""
    return image * 2 - 1


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]"""
    return (image + 1) / 2


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model oluştur
    model = LowLightDiffusion(
        unet_variant="small",
        image_size=256,
        num_inference_steps=4,
    ).to(device)
    
    print(f"Model parameters: {model.get_model_size()}")
    
    # Test data
    low_light = torch.randn(2, 3, 256, 256).to(device)
    normal_light = torch.randn(2, 3, 256, 256).to(device)
    
    # Training forward
    output = model(low_light, normal_light)
    print(f"Training output keys: {output.keys()}")
    print(f"Noise prediction shape: {output['noise_pred'].shape}")
    
    # Loss
    loss = model.compute_loss(low_light, normal_light)
    print(f"Loss: {loss.item():.4f}")
    
    # Inference
    with torch.no_grad():
        enhanced = model.enhance(low_light, num_inference_steps=4)
    print(f"Enhanced image shape: {enhanced.shape}")
    print(f"Enhanced image range: [{enhanced.min():.2f}, {enhanced.max():.2f}]")

