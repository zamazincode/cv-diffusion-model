#!/usr/bin/env python3
"""
Training Script
===============
Low-Light Enhancement Diffusion Model Training

Kullanım:
    python scripts/train.py --data_dir /path/to/lol --epochs 100

LOL Dataset yapısı:
    data/
        train/
            low/
                1.png, 2.png, ...
            high/
                1.png, 2.png, ...
        test/
            low/
                ...
            high/
                ...
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LowLightDiffusion
from src.training import LowLightTrainer, TrainingConfig, create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train Low-Light Enhancement Model")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Validation directory")
    
    # Model
    parser.add_argument("--variant", type=str, default="small", 
                        choices=["tiny", "small", "base", "large"],
                        help="Model variant")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_steps", type=int, default=4, help="LCM inference steps")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="mse", 
                        choices=["mse", "huber", "l1"], help="Loss function")
    
    # Optimization
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay")
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--project", type=str, default="low-light-diffusion", help="W&B project")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Low-Light Enhancement Diffusion Training")
    print("=" * 60)
    
    # Config
    config = TrainingConfig(
        # Model
        unet_variant=args.variant,
        image_size=args.image_size,
        num_inference_steps=args.num_steps,
        
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        loss_type=args.loss,
        
        # Optimization
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        
        # Logging
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.project,
        
        # Resume
        resume_from=args.resume,
    )
    
    print(f"\nConfig:")
    print(f"  Model: {args.variant}")
    print(f"  Image size: {args.image_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Loss: {args.loss}")
    print(f"  Mixed precision: {args.use_amp}")
    print(f"  EMA: {args.use_ema}")
    
    # Data
    print(f"\nLoading data from: {args.data_dir}")
    train_loader, val_loader = create_dataloaders(
        train_root=args.data_dir,
        val_root=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")
    
    # Model
    print("\nCreating model...")
    model = LowLightDiffusion(
        unet_variant=args.variant,
        image_size=args.image_size,
        num_inference_steps=args.num_steps,
    )
    
    model_size = model.get_model_size()
    print(f"  Parameters: {model_size['num_params']:,}")
    print(f"  FP32 size: {model_size['fp32_mb']:.2f} MB")
    print(f"  FP16 size: {model_size['fp16_mb']:.2f} MB")
    
    # Trainer
    print("\nStarting training...")
    trainer = LowLightTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"Samples saved to: {config.output_dir}")


if __name__ == "__main__":
    main()



