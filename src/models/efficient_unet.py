"""
Efficient UNet Architecture for Low-Light Enhancement
======================================================
Android cihazlarda gerçek zamanlı inference için optimize edilmiş UNet.

Mimari Özellikler:
- MobileNetV3-style Inverted Residual Blocks (depthwise separable conv)
- Squeeze-and-Excitation (SE) channel attention
- Linear Attention (O(n) complexity vs O(n²) standart attention)
- Progressive channel reduction (encoder'dan decoder'a)
- Quantization-friendly operations (ReLU6, no LayerNorm in critical paths)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class EfficientUNetConfig:
    """UNet konfigürasyonu"""
    # Giriş/Çıkış
    in_channels: int = 3
    out_channels: int = 3
    
    # Temel kanal boyutları (hafif versiyon)
    # Standart: [64, 128, 256, 512] -> Hafif: [32, 64, 128, 256]
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    
    # Attention ayarları
    attention_resolutions: Tuple[int, ...] = (16, 8)  # Hangi çözünürlüklerde attention
    num_attention_heads: int = 4
    use_linear_attention: bool = True  # Linear attention for mobile
    
    # Block ayarları
    num_res_blocks: int = 2  # Her seviyede kaç residual block
    expansion_ratio: int = 4  # Inverted residual expansion
    use_se: bool = True  # Squeeze-and-Excitation
    se_ratio: float = 0.25
    
    # Time embedding
    time_embed_dim: int = 128
    
    # Dropout (inference'da 0)
    dropout: float = 0.0
    
    # Kuantizasyon desteği
    quantization_friendly: bool = True  # ReLU6, avoid LayerNorm etc.
    
    # Çözünürlük (training için)
    image_size: int = 256


class SinusoidalPosEmb(nn.Module):
    """Timestep için sinusoidal positional embedding"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Mobile-friendly: uses ReLU instead of SiLU for better quantization.
    """
    
    def __init__(self, channels: int, ratio: float = 0.25, quantization_friendly: bool = True):
        super().__init__()
        squeezed = max(1, int(channels * ratio))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeezed, 1)
        self.fc2 = nn.Conv2d(squeezed, channels, 1)
        
        # Quantization-friendly activation
        self.act = nn.ReLU6(inplace=True) if quantization_friendly else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.
    MobileNet-style: depthwise + pointwise convolution.
    Reduces parameters from k²·Cin·Cout to k²·Cin + Cin·Cout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Depthwise: her kanal için ayrı convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        # Pointwise: 1x1 conv for channel mixing
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidualBlock(nn.Module):
    """
    MobileNetV3-style Inverted Residual Block.
    
    Yapı:
    1. Expand: 1x1 conv (channel expansion)
    2. Depthwise: 3x3 depthwise separable conv
    3. SE: Squeeze-and-Excitation (optional)
    4. Project: 1x1 conv (channel projection)
    
    Time embedding ile conditioning için temporal modulation eklenir.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        expansion_ratio: int = 4,
        stride: int = 1,
        use_se: bool = True,
        se_ratio: float = 0.25,
        dropout: float = 0.0,
        quantization_friendly: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expansion_ratio)
        self.act = nn.ReLU6(inplace=True) if quantization_friendly else nn.SiLU(inplace=True)
        
        # Normalization - GroupNorm quantization-friendly
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.norm2 = nn.GroupNorm(min(32, hidden_dim), hidden_dim)
        
        # Expand
        self.expand = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        
        # Depthwise
        self.depthwise = nn.Conv2d(
            hidden_dim, hidden_dim, 3, 
            stride=stride, padding=1, groups=hidden_dim, bias=False
        )
        
        # SE
        self.se = SqueezeExcitation(hidden_dim, se_ratio, quantization_friendly) if use_se else nn.Identity()
        
        # Project
        self.project = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim * 2)  # scale and shift
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection için boyut ayarlama
        if not self.use_residual and in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Expand
        h = self.norm1(x)
        h = self.act(h)
        h = self.expand(h)
        
        # Depthwise with time conditioning
        h = self.norm2(h)
        
        # Time modulation (FiLM-style: scale and shift)
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        scale, shift = time_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.act(h)
        h = self.depthwise(h)
        
        # SE attention
        h = self.se(h)
        
        # Project
        h = self.project(h)
        h = self.dropout(h)
        
        # Residual
        if self.skip is not None:
            residual = self.skip(residual)
        
        if self.use_residual or self.skip is not None:
            h = h + residual
            
        return h


class LinearAttention(nn.Module):
    """
    Linear Attention (O(n) complexity).
    
    Standart attention O(n²) yerine kernel feature maps kullanarak O(n) complexity.
    Mobile deployment için kritik - özellikle yüksek çözünürlüklerde.
    
    Formül: Attention(Q, K, V) ≈ φ(Q) @ (φ(K)ᵀ @ V)
    where φ is a feature map (elu + 1 for non-negativity)
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dim_head: int = 32,
        quantization_friendly: bool = True,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        
        self.to_qkv = nn.Conv2d(channels, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels)
        )
        
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        
        # QKV projection
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        k = rearrange(k, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        v = rearrange(v, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        
        # Feature map for linear attention (ELU + 1 for non-negativity)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention: O(n) complexity
        # Instead of: softmax(QK^T)V which is O(n²)
        # We compute: Q @ (K^T @ V) which is O(n)
        k_sum = k.sum(dim=-2, keepdim=True)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # Normalize
        qk_sum = torch.einsum('bhnd,bhkd->bhnk', q, k_sum)
        qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        out = qkv / (qk_sum + 1e-6)
        
        # Reshape back
        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=h, w=w)
        out = self.to_out(out)
        
        return out + residual


class StandardAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention.
    Daha az çözünürlükte (8x8, 16x16) kullanılır.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dim_head: int = 32,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head
        
        self.norm = nn.GroupNorm(min(32, channels), channels)
        
        self.to_qkv = nn.Conv2d(channels, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, channels, 1, bias=False)
        
        self.scale = dim_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        k = rearrange(k, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        v = rearrange(v, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        
        # Scaled dot-product attention
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=h, w=w)
        out = self.to_out(out)
        
        return out + residual


class Downsample(nn.Module):
    """Strided convolution ile downsampling - pooling yerine"""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        if use_conv:
            # 3x3 strided conv daha iyi gradient flow sağlar
            self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    """Bilinear interpolation + conv ile upsampling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


class EfficientUNet(nn.Module):
    """
    Düşük ışık iyileştirmesi için optimize edilmiş UNet.
    
    Özellikler:
    - MobileNetV3-style inverted residual blocks
    - Linear attention for O(n) complexity
    - SE channel attention
    - Time-conditioned (diffusion için)
    - Quantization-friendly operations
    
    Parametre karşılaştırması (256x256 input):
    - Standart UNet: ~30M params
    - Bu mimari: ~3-5M params
    """
    
    def __init__(self, config: EfficientUNetConfig):
        super().__init__()
        self.config = config
        
        # Channel hesaplama
        channels = [config.base_channels * m for m in config.channel_multipliers]
        # Örnek: [32, 64, 128, 256] for base=32, multipliers=(1,2,4,8)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.base_channels),
            nn.Linear(config.base_channels, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        current_res = config.image_size
        in_ch = channels[0]
        
        for level, out_ch in enumerate(channels):
            level_blocks = nn.ModuleList()
            
            for block_idx in range(config.num_res_blocks):
                level_blocks.append(
                    InvertedResidualBlock(
                        in_channels=in_ch if block_idx == 0 else out_ch,
                        out_channels=out_ch,
                        time_embed_dim=config.time_embed_dim,
                        expansion_ratio=config.expansion_ratio,
                        use_se=config.use_se,
                        se_ratio=config.se_ratio,
                        dropout=config.dropout,
                        quantization_friendly=config.quantization_friendly,
                    )
                )
                
                # Attention at specific resolutions
                if current_res in config.attention_resolutions:
                    if config.use_linear_attention:
                        level_blocks.append(
                            LinearAttention(out_ch, config.num_attention_heads)
                        )
                    else:
                        level_blocks.append(
                            StandardAttention(out_ch, config.num_attention_heads)
                        )
            
            self.encoder_blocks.append(level_blocks)
            in_ch = out_ch
            
            # Downsampler (except last level)
            if level < len(channels) - 1:
                self.downsamplers.append(Downsample(out_ch))
                current_res //= 2
        
        # Middle block
        mid_ch = channels[-1]
        self.mid_block1 = InvertedResidualBlock(
            mid_ch, mid_ch, config.time_embed_dim,
            expansion_ratio=config.expansion_ratio,
            use_se=config.use_se,
            quantization_friendly=config.quantization_friendly,
        )
        self.mid_attn = LinearAttention(mid_ch, config.num_attention_heads) \
            if config.use_linear_attention else StandardAttention(mid_ch, config.num_attention_heads)
        self.mid_block2 = InvertedResidualBlock(
            mid_ch, mid_ch, config.time_embed_dim,
            expansion_ratio=config.expansion_ratio,
            use_se=config.use_se,
            quantization_friendly=config.quantization_friendly,
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        
        for level, out_ch in enumerate(reversed_channels):
            level_blocks = nn.ModuleList()
            
            for block_idx in range(config.num_res_blocks + 1):  # +1 for skip connection
                # Skip connection doubles input channels
                block_in_ch = in_ch + out_ch if block_idx == 0 else out_ch
                
                level_blocks.append(
                    InvertedResidualBlock(
                        in_channels=block_in_ch,
                        out_channels=out_ch,
                        time_embed_dim=config.time_embed_dim,
                        expansion_ratio=config.expansion_ratio,
                        use_se=config.use_se,
                        se_ratio=config.se_ratio,
                        dropout=config.dropout,
                        quantization_friendly=config.quantization_friendly,
                    )
                )
                
                # Attention
                if current_res in config.attention_resolutions:
                    if config.use_linear_attention:
                        level_blocks.append(
                            LinearAttention(out_ch, config.num_attention_heads)
                        )
                    else:
                        level_blocks.append(
                            StandardAttention(out_ch, config.num_attention_heads)
                        )
            
            self.decoder_blocks.append(level_blocks)
            in_ch = out_ch
            
            # Upsampler (except last level)
            if level < len(reversed_channels) - 1:
                self.upsamplers.append(Upsample(out_ch))
                current_res *= 2
        
        # Final output
        self.final_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(channels[0], config.out_channels, 3, padding=1)

    def forward(
        self, 
        x: torch.Tensor, 
        timestep: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, C, H, W]
            timestep: Diffusion timestep [B]
            return_features: Return intermediate features for analysis
            
        Returns:
            Predicted noise or denoised image [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(timestep)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        
        for level, (blocks, downsampler) in enumerate(
            zip(self.encoder_blocks, self.downsamplers + [None])
        ):
            for block in blocks:
                if isinstance(block, InvertedResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)  # Attention
            
            skip_connections.append(h)
            
            if downsampler is not None:
                h = downsampler(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        features = []
        
        for level, (blocks, upsampler) in enumerate(
            zip(self.decoder_blocks, [None] + list(self.upsamplers))
        ):
            if upsampler is not None:
                h = upsampler(h)
            
            # Concat skip connection
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, InvertedResidualBlock):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            
            if return_features:
                features.append(h)
        
        # Final
        h = self.final_norm(h)
        h = self.final_act(h)
        h = self.final_conv(h)
        
        if return_features:
            return h, features
        return h

    def get_num_params(self) -> int:
        """Model parametre sayısını döndür"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_footprint(self, input_size: Tuple[int, int] = (256, 256)) -> dict:
        """Tahmini memory kullanımı"""
        num_params = self.get_num_params()
        
        # FP32
        fp32_size = num_params * 4 / (1024**2)  # MB
        # FP16
        fp16_size = num_params * 2 / (1024**2)
        # INT8
        int8_size = num_params * 1 / (1024**2)
        
        return {
            "num_params": num_params,
            "fp32_mb": fp32_size,
            "fp16_mb": fp16_size,
            "int8_mb": int8_size,
        }


def create_efficient_unet(
    variant: str = "small",
    image_size: int = 256,
    **kwargs,
) -> EfficientUNet:
    """
    Farklı boyutlarda EfficientUNet oluştur.
    
    Variants:
    - tiny: ~1M params, çok hafif, düşük kalite
    - small: ~3M params, mobil için ideal (önerilen)
    - base: ~5M params, daha iyi kalite
    - large: ~10M params, maksimum kalite
    """
    
    configs = {
        "tiny": EfficientUNetConfig(
            base_channels=16,
            channel_multipliers=(1, 2, 4, 8),
            num_res_blocks=1,
            expansion_ratio=2,
            time_embed_dim=64,
            num_attention_heads=2,
            image_size=image_size,
            **kwargs,
        ),
        "small": EfficientUNetConfig(
            base_channels=32,
            channel_multipliers=(1, 2, 4, 8),
            num_res_blocks=2,
            expansion_ratio=4,
            time_embed_dim=128,
            num_attention_heads=4,
            image_size=image_size,
            **kwargs,
        ),
        "base": EfficientUNetConfig(
            base_channels=48,
            channel_multipliers=(1, 2, 4, 8),
            num_res_blocks=2,
            expansion_ratio=4,
            time_embed_dim=192,
            num_attention_heads=6,
            image_size=image_size,
            **kwargs,
        ),
        "large": EfficientUNetConfig(
            base_channels=64,
            channel_multipliers=(1, 2, 4, 8),
            num_res_blocks=3,
            expansion_ratio=4,
            time_embed_dim=256,
            num_attention_heads=8,
            image_size=image_size,
            **kwargs,
        ),
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    return EfficientUNet(configs[variant])


if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for variant in ["tiny", "small", "base", "large"]:
        model = create_efficient_unet(variant, image_size=256).to(device)
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256).to(device)
        t = torch.randint(0, 1000, (1,)).to(device)
        
        with torch.no_grad():
            y = model(x, t)
        
        memory = model.get_memory_footprint()
        
        print(f"\n{variant.upper()} variant:")
        print(f"  Parameters: {memory['num_params']:,}")
        print(f"  FP32: {memory['fp32_mb']:.2f} MB")
        print(f"  FP16: {memory['fp16_mb']:.2f} MB")
        print(f"  INT8: {memory['int8_mb']:.2f} MB")
        print(f"  Output shape: {y.shape}")

