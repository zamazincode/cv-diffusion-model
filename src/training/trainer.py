"""
Training Pipeline
=================
Low-Light Enhancement modeli için training loop ve utilities.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

from ..models import LowLightDiffusion


@dataclass
class TrainingConfig:
    """Training konfigürasyonu"""
    
    # Model
    unet_variant: str = "small"
    image_size: int = 256
    num_inference_steps: int = 4
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # LR Scheduler
    scheduler_type: str = "cosine"  # "cosine" veya "onecycle"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Mixed Precision
    use_amp: bool = True
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Loss
    loss_type: str = "mse"  # "mse", "huber", "l1"
    
    # Logging
    log_interval: int = 100
    save_interval: int = 5  # epochs
    sample_interval: int = 1  # epochs
    num_samples: int = 4
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "low-light-diffusion"
    wandb_run_name: Optional[str] = None
    
    # Resume
    resume_from: Optional[str] = None


class EMAModel:
    """Exponential Moving Average model wrapper"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self, model: nn.Module):
        """EMA weights'i model'e uygula"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """Orijinal weights'e geri dön"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


class LowLightTrainer:
    """
    Low-Light Enhancement model trainer.
    
    Features:
    - Mixed precision training (FP16)
    - EMA
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Wandb logging
    """
    
    def __init__(
        self,
        model: LowLightDiffusion,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # LR Scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs
        
        if self.config.scheduler_type == "cosine":
            T_max = max(1, total_steps - warmup_steps)  # En az 1 olmalı
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=self.config.min_lr,
            )
        else:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
            )
        
        # Mixed precision
        if self.config.use_amp:
            try:
                self.scaler = GradScaler('cuda')  # New API
            except TypeError:
                self.scaler = GradScaler()  # Old API fallback
        else:
            self.scaler = None
        
        # EMA
        self.ema = EMAModel(self.model, self.config.ema_decay) if self.config.use_ema else None
        
        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Directories
        self.output_dir = Path(self.config.output_dir)
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb
        if self.config.use_wandb:
            if not HAS_WANDB:
                print("Warning: wandb not installed. Logging disabled.")
                self.config.use_wandb = False
            else:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=self.config.__dict__,
                )
        
        # Resume
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
    
    def train(self):
        """Full training loop"""
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.get_model_size()}")
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = None
            if self.val_loader is not None:
                val_loss = self.validate()
            
            # Logging
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}", end="")
            if val_loss is not None:
                print(f", val_loss={val_loss:.4f}", end="")
            print()
            
            if self.config.use_wandb:
                wandb.log(log_dict)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Best model
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
            
            # Generate samples
            if (epoch + 1) % self.config.sample_interval == 0:
                self.generate_samples(epoch)
        
        # Final save
        self.save_checkpoint("final_model.pt")
        
        if self.config.use_wandb:
            wandb.finish()
    
    def train_epoch(self) -> float:
        """Tek epoch training"""
        
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Data
            low_light = batch["low_light"].to(self.device)
            normal_light = batch["normal_light"].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.device.type == "cuda":
                with autocast():
                    loss = self.model.compute_loss(
                        low_light, normal_light, 
                        loss_type=self.config.loss_type
                    )
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model.compute_loss(
                    low_light, normal_light,
                    loss_type=self.config.loss_type
                )
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                self.optimizer.step()
            
            # LR scheduler
            self.scheduler.step()
            
            # EMA update
            if self.ema is not None:
                self.ema.update(self.model)
            
            # Logging
            total_loss += loss.item()
            self.global_step += 1
            
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"loss": loss.item()})
                
                if self.config.use_wandb:
                    wandb.log({
                        "train_loss_step": loss.item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validation loop"""
        
        self.model.eval()
        
        # EMA weights kullan
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            low_light = batch["low_light"].to(self.device)
            normal_light = batch["normal_light"].to(self.device)
            
            loss = self.model.compute_loss(low_light, normal_light)
            total_loss += loss.item()
        
        # Orijinal weights'e dön
        if self.ema is not None:
            self.ema.restore(self.model)
        
        return total_loss / len(self.val_loader)
    
    @torch.no_grad()
    def generate_samples(self, epoch: int):
        """Sample görüntüler oluştur"""
        
        self.model.eval()
        
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        # İlk batch'den sample al
        batch = next(iter(self.val_loader or self.train_loader))
        low_light = batch["low_light"][:self.config.num_samples].to(self.device)
        normal_light = batch["normal_light"][:self.config.num_samples].to(self.device)
        
        # Enhancement
        enhanced = self.model.enhance(low_light, num_inference_steps=4)
        
        # Save images
        self._save_comparison(
            low_light, normal_light, enhanced,
            self.output_dir / f"samples_epoch_{epoch}.png"
        )
        
        if self.ema is not None:
            self.ema.restore(self.model)
    
    def _save_comparison(
        self,
        low_light: torch.Tensor,
        normal_light: torch.Tensor,
        enhanced: torch.Tensor,
        path: Path,
    ):
        """Karşılaştırma görüntüsü kaydet"""
        from torchvision.utils import make_grid, save_image
        
        # Denormalize [-1, 1] -> [0, 1]
        low_light = (low_light + 1) / 2
        normal_light = (normal_light + 1) / 2
        enhanced = (enhanced + 1) / 2
        
        # Grid oluştur
        comparison = torch.cat([low_light, enhanced, normal_light], dim=0)
        grid = make_grid(comparison, nrow=len(low_light))
        
        save_image(grid, path)
        
        if self.config.use_wandb:
            wandb.log({"samples": wandb.Image(str(path))})
    
    def save_checkpoint(self, filename: str):
        """Checkpoint kaydet"""
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }
        
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, path: str):
        """Checkpoint yükle"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Loaded checkpoint from epoch {self.epoch - 1}")


def train_model(
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
):
    """Training entry point"""
    
    from .dataset import create_dataloaders
    
    config = config or TrainingConfig()
    
    # Data loaders
    train_loader, val_loader = create_dataloaders(
        train_root=train_data_dir,
        val_root=val_data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
    )
    
    # Model
    model = LowLightDiffusion(
        unet_variant=config.unet_variant,
        image_size=config.image_size,
        num_inference_steps=config.num_inference_steps,
    )
    
    # Trainer
    trainer = LowLightTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Train
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Config test
    config = TrainingConfig()
    print(f"Config: {config}")

