"""
Main training script for Kintsugi AI
Run this script to train the image restoration model.
"""

import os
import argparse
import torch

from src.dataset import KintsugiDataset, ProgressiveDataLoader, create_dataloaders
from src.model import create_model
from src.trainer import Trainer
from config import Config, DEFAULT_CONFIG, COLAB_CONFIG, LIGHT_CONFIG


def main(config: Config, resume_from: str = None):
    """Main training function."""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_dataset = KintsugiDataset(
        root_dir=config.data.train_dir,
        image_size=config.data.image_size,
        severity=config.training.start_severity if config.training.progressive else 1.0,
        augment=True
    )
    
    if config.data.val_dir and os.path.exists(config.data.val_dir):
        val_dataset = KintsugiDataset(
            root_dir=config.data.val_dir,
            image_size=config.data.image_size,
            severity=1.0,
            augment=False
        )
    else:
        # Split train dataset
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Create loaders
    if config.training.progressive:
        train_loader = ProgressiveDataLoader(
            train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            start_severity=config.training.start_severity,
            end_severity=config.training.end_severity,
            warmup_epochs=config.training.warmup_epochs
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type=config.model.model_type,
        features=config.model.features,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        epochs=config.training.epochs,
        grad_accum_steps=config.training.grad_accum_steps,
        use_amp=config.training.use_amp,
        l1_weight=config.loss.l1_weight,
        ssim_weight=config.loss.ssim_weight,
        perceptual_weight=config.loss.perceptual_weight,
        edge_weight=config.loss.edge_weight,
        save_dir=config.save_dir,
        save_every=config.save_every,
        log_every=config.log_every
    )
    
    # Train
    history = trainer.train(resume_from=resume_from)
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Kintsugi AI')
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'colab', 'light'],
                        help='Configuration preset')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='Path to training images')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Path to validation images')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Select config preset
    if args.config == 'colab':
        config = COLAB_CONFIG
    elif args.config == 'light':
        config = LIGHT_CONFIG
    else:
        config = DEFAULT_CONFIG
    
    # Override with command line args
    if args.train_dir:
        config.data.train_dir = args.train_dir
    if args.val_dir:
        config.data.val_dir = args.val_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    main(config, resume_from=args.resume)
