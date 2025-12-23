"""
LCM (Latent Consistency Model) Scheduler
=========================================
4-8 adımda yüksek kaliteli inference için Consistency Training.

Referanslar:
- Latent Consistency Models (LCM): https://arxiv.org/abs/2310.04378
- Consistency Models: https://arxiv.org/abs/2303.01469

Temel Prensipler:
1. Teacher model'den self-consistency loss ile öğrenme
2. Progressive timestep skipping
3. CFG-aware distillation
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SchedulerMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


@dataclass
class LCMSchedulerOutput:
    """LCM Scheduler çıktısı"""
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class LCMScheduler(SchedulerMixin, ConfigMixin):
    """
    LCM Scheduler - Az adımda yüksek kaliteli diffusion.
    
    Bu scheduler, standart DDPM'in 50-1000 adımını 4-8 adıma indirir.
    
    Çalışma prensibi:
    1. Consistency function f(x_t, t) → x_0 öğrenir
    2. Her adımda doğrudan x_0 tahmini yapar
    3. Sonra bir sonraki timestep'e gürültü ekler
    
    Avantajlar:
    - 4-8 adımda inference
    - Teacher model ile aynı kalite (veya çok yakın)
    - CFG (classifier-free guidance) desteği
    """
    
    order = 1
    
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        num_inference_steps: int = 4,
        original_inference_steps: int = 50,
        lcm_origin_steps: int = 50,
    ):
        """
        Args:
            num_train_timesteps: Toplam training timestep sayısı
            beta_start/end: Noise schedule parametreleri
            beta_schedule: "linear", "scaled_linear", "squaredcos_cap_v2"
            prediction_type: "epsilon" (noise) veya "v_prediction"
            num_inference_steps: LCM inference adım sayısı (4-8 önerilen)
            original_inference_steps: Teacher model adım sayısı
        """
        # Beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # Stable Diffusion'ın kullandığı schedule
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps
            ) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Zero SNR rescaling (optional, improves dark image generation)
        if rescale_betas_zero_snr:
            self.alphas_cumprod = self._rescale_zero_terminal_snr(self.alphas_cumprod)
        
        # Sigma hesaplama
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        
        # Final alpha
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        
        # Initialize
        self.num_inference_steps = None
        self.timesteps = None
        self._step_index = None
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine noise schedule - daha smooth geçişler"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def _rescale_zero_terminal_snr(self, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """
        Zero terminal SNR rescaling.
        SNR(t=T) = 0 yaparak tamamen gürültülü başlangıç sağlar.
        Karanlık görüntüler için önemli.
        """
        alphas_bar_sqrt = alphas_cumprod.sqrt()
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
        
        return alphas_bar_sqrt ** 2
    
    def set_timesteps(
        self, 
        num_inference_steps: int = 4,
        device: Union[str, torch.device] = "cpu",
        original_inference_steps: Optional[int] = None,
    ):
        """
        Inference için timestep'leri ayarla.
        
        LCM için timestep'ler teacher model'in timestep'lerinden seçilir.
        Örnek: Teacher 50 adım, LCM 4 adım → [999, 749, 499, 249]
        """
        self.num_inference_steps = num_inference_steps
        
        if original_inference_steps is None:
            original_inference_steps = self.config.original_inference_steps
        
        # LCM timestep selection
        # Teacher timestep'lerinden eşit aralıklı seçim
        c = self.config.num_train_timesteps // original_inference_steps
        
        # Timestep indeksleri (LCM'de kullanılan)
        lcm_origin_timesteps = torch.arange(
            1, original_inference_steps + 1
        ) * c - 1
        
        # LCM için seçilen timestep'ler
        skipping_step = len(lcm_origin_timesteps) // num_inference_steps
        
        timesteps = lcm_origin_timesteps[::skipping_step][:num_inference_steps]
        timesteps = timesteps.flip(0)  # Büyükten küçüğe
        
        self.timesteps = timesteps.to(device)
        self._step_index = 0
        
        # Sigma değerleri
        self.sigmas = self.sigmas.to(device)
        
    def _get_prev_timestep(self, timestep: int) -> int:
        """Bir önceki timestep'i al"""
        index = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if index + 1 < len(self.timesteps):
            return self.timesteps[index + 1].item()
        return 0
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LCMSchedulerOutput, Tuple]:
        """
        Tek bir denoising adımı.
        
        LCM'de her adım:
        1. x_0 tahmini yap (consistency function)
        2. Bir sonraki timestep için gürültü ekle
        
        Args:
            model_output: UNet çıktısı (noise veya v-prediction)
            timestep: Mevcut timestep
            sample: Mevcut noisy sample x_t
            
        Returns:
            Denoised sample (bir sonraki adım için)
        """
        # Timestep indeksi
        if self._step_index is None:
            self._step_index = 0
        
        # Mevcut ve sonraki timestep
        t = timestep
        prev_t = self._get_prev_timestep(t)
        
        # Alpha ve sigma değerleri
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # x_0 tahmini
        if self.config.prediction_type == "epsilon":
            # model_output = predicted noise
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.config.prediction_type == "v_prediction":
            # model_output = v = alpha * noise - sigma * x_0
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # Clamp x_0 (opsiyonel, stabilite için)
        # pred_original_sample = pred_original_sample.clamp(-1, 1)
        
        # Son adım mı?
        if prev_t == 0:
            prev_sample = pred_original_sample
        else:
            # Bir sonraki timestep için gürültü ekle (DDIM-style)
            # x_{t-1} = sqrt(alpha_{t-1}) * x_0 + sqrt(1-alpha_{t-1}) * noise
            
            # Deterministic (DDIM) veya stochastic (DDPM) seçimi
            # LCM genelde deterministic kullanır
            # Not: generator parametresi bazı PyTorch versiyonlarında desteklenmiyor
            noise = torch.randn_like(sample)
            
            prev_sample = (
                alpha_prod_t_prev ** 0.5 * pred_original_sample +
                beta_prod_t_prev ** 0.5 * noise
            )
        
        # Step index güncelle
        self._step_index += 1
        
        if return_dict:
            return LCMSchedulerOutput(
                prev_sample=prev_sample,
                pred_original_sample=pred_original_sample,
            )
        
        return (prev_sample, pred_original_sample)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training için forward process - gürültü ekleme.
        
        x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Broadcast için reshape
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        V-prediction için velocity hesapla.
        v = sqrt(alpha_t) * noise - sqrt(1 - alpha_t) * sample
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )
        timesteps = timesteps.to(sample.device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity


class LCMTrainer:
    """
    LCM Training için yardımcı sınıf.
    
    Consistency Training:
    - Self-consistency loss ile model eğitimi
    - Teacher-free veya teacher-guided training
    - Progressive timestep skipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: LCMScheduler,
        learning_rate: float = 1e-4,
        ema_decay: float = 0.95,
        num_ddim_timesteps: int = 50,
        guidance_scale: float = 7.5,
        w_min: float = 3.0,
        w_max: float = 15.0,
    ):
        """
        Args:
            model: UNet modeli
            scheduler: LCM Scheduler
            ema_decay: EMA target model için decay rate
            num_ddim_timesteps: Teacher DDIM adım sayısı
            guidance_scale: CFG scale (training sırasında random seçilir)
            w_min/w_max: CFG scale aralığı
        """
        self.model = model
        self.scheduler = scheduler
        self.num_ddim_timesteps = num_ddim_timesteps
        self.w_min = w_min
        self.w_max = w_max
        
        # EMA model (target network)
        self.ema_model = self._create_ema_model(model, ema_decay)
        
        # Timestep indices for consistency training
        self.c = scheduler.config.num_train_timesteps // num_ddim_timesteps
        
    def _create_ema_model(self, model: nn.Module, decay: float):
        """EMA target model oluştur"""
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        ema_model.eval()
        return ema_model
    
    @torch.no_grad()
    def update_ema(self, decay: float = 0.95):
        """EMA model güncelle"""
        for ema_param, model_param in zip(
            self.ema_model.parameters(), 
            self.model.parameters()
        ):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def get_timestep_pairs(
        self, 
        batch_size: int, 
        num_inference_steps: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Consistency training için timestep çiftleri.
        
        (t_n, t_{n+k}) çiftleri döndürür, burada k skipping step.
        """
        # Teacher timestep'leri
        all_timesteps = torch.arange(
            1, self.num_ddim_timesteps + 1, device=device
        ) * self.c - 1
        
        # Skipping step
        k = self.num_ddim_timesteps // num_inference_steps
        
        # Random başlangıç noktası seç
        start_indices = torch.randint(
            0, self.num_ddim_timesteps - k, (batch_size,), device=device
        )
        
        t = all_timesteps[start_indices]
        t_next = all_timesteps[start_indices + k]
        
        return t, t_next
    
    def consistency_loss(
        self,
        model_output: torch.Tensor,
        target_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Consistency loss: L2 distance between predictions.
        
        f(x_t, t) ≈ f(x_{t'}, t') for consistency
        """
        return F.mse_loss(model_output, target_output)
    
    def huber_loss(
        self,
        model_output: torch.Tensor,
        target_output: torch.Tensor,
        delta: float = 1.0,
    ) -> torch.Tensor:
        """
        Huber loss - outlier'lara karşı daha robust.
        LCM paper'da önerilen.
        """
        return F.huber_loss(model_output, target_output, delta=delta)


def get_lcm_timesteps(
    num_inference_steps: int = 4,
    num_train_timesteps: int = 1000,
    original_inference_steps: int = 50,
) -> List[int]:
    """
    LCM inference timestep'lerini hesapla.
    
    Örnek:
    - 4 adım: [999, 749, 499, 249]
    - 8 adım: [999, 874, 749, 624, 499, 374, 249, 124]
    """
    c = num_train_timesteps // original_inference_steps
    
    lcm_origin_timesteps = [i * c - 1 for i in range(1, original_inference_steps + 1)]
    
    skipping_step = len(lcm_origin_timesteps) // num_inference_steps
    
    timesteps = lcm_origin_timesteps[::skipping_step][:num_inference_steps]
    timesteps = list(reversed(timesteps))
    
    return timesteps


if __name__ == "__main__":
    # Test
    scheduler = LCMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        num_inference_steps=4,
    )
    
    # Timestep ayarla
    scheduler.set_timesteps(num_inference_steps=4, device="cpu")
    
    print("LCM Timesteps (4 steps):", scheduler.timesteps.tolist())
    
    scheduler.set_timesteps(num_inference_steps=8, device="cpu")
    print("LCM Timesteps (8 steps):", scheduler.timesteps.tolist())
    
    # Utility function test
    print("\nUtility function test:")
    print("4 steps:", get_lcm_timesteps(4))
    print("8 steps:", get_lcm_timesteps(8))

