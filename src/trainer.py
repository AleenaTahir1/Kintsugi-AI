"""
Training Loop for Kintsugi AI
Implements progressive training, logging, and checkpointing.
"""

import os
import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

from .losses import CompositeLoss, compute_psnr, SSIMLoss
from .dataset import ProgressiveDataLoader


class Trainer:
    """
    Trainer class for Kintsugi AI restoration model.
    
    Features:
    - Progressive training (gradually increase degradation severity)
    - Mixed precision training (FP16) for faster training
    - Gradient accumulation for effective larger batch sizes
    - Cosine annealing learning rate schedule
    - Checkpoint saving and resuming
    - Validation with PSNR/SSIM metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: ProgressiveDataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        # Optimizer settings
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        # Training settings
        epochs: int = 100,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        # Loss weights
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        edge_weight: float = 0.1,
        # Checkpointing
        save_dir: str = 'checkpoints',
        save_every: int = 5,
        # Logging
        log_every: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and device == 'cuda'
        self.save_dir = save_dir
        self.save_every = save_every
        self.log_every = log_every
        
        # Loss function
        self.criterion = CompositeLoss(
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            perceptual_weight=perceptual_weight,
            edge_weight=edge_weight
        ).to(device)
        
        # SSIM for metrics
        self.ssim_metric = SSIMLoss().to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.current_epoch = 0
        self.best_psnr = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'lr': []
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {'l1': 0.0, 'ssim': 0.0, 'perceptual': 0.0, 'edge': 0.0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (degraded, clean) in enumerate(pbar):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(degraded)
                    loss, components = self.criterion(output, clean, return_components=True)
                    loss = loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(degraded)
                loss, components = self.criterion(output, clean, return_components=True)
                loss = loss / self.grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Accumulate losses
            total_loss += components['total'].item()
            for key in loss_components:
                if key in components:
                    loss_components[key] += components[key].item()
            
            # Update progress bar
            if batch_idx % self.log_every == 0:
                pbar.set_postfix({
                    'loss': f"{components['total'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'total': avg_loss, **loss_components}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_loader)
        
        for degraded, clean in tqdm(self.val_loader, desc='Validating'):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(degraded)
            else:
                output = self.model(degraded)
            
            # Compute metrics
            loss = self.criterion(output, clean)
            psnr = compute_psnr(output, clean)
            ssim = self.ssim_metric.compute_ssim(output, clean)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()
        
        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'history': self.history,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['best_psnr']
        self.history = checkpoint['history']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
    
    def train(self, resume_from: Optional[str] = None):
        """Full training loop."""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        print(f"\nStarting training from epoch {self.current_epoch}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Epochs: {self.epochs}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Update degradation severity for progressive training
            if hasattr(self.train_loader, 'set_epoch'):
                self.train_loader.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['total'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.epochs-1}")
            print(f"  Train Loss: {train_metrics['total']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
                print(f"  New best PSNR: {self.best_psnr:.2f} dB")
            
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', is_best=is_best)
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Save training history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.2f} hours")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        
        return self.history
