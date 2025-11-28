"""
Custom PyTorch Dataset for Kintsugi AI
Applies dynamic degradation during training for self-supervised learning.
"""

import os
import random
from typing import Tuple, Optional, List, Callable
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .degradation import DegradationPipeline, numpy_to_tensor


class KintsugiDataset(Dataset):
    """
    Dataset that loads clean images and applies synthetic degradation on-the-fly.
    Supports CelebA-HQ, document datasets, or any folder of images.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        severity: float = 1.0,
        augment: bool = True,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    ):
        """
        Args:
            root_dir: Path to directory containing clean images
            image_size: Target size for images (will be center-cropped and resized)
            severity: Degradation severity (0.0-1.0) for progressive training
            augment: Whether to apply data augmentation (flips, rotations)
            extensions: Valid image file extensions
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.severity = severity
        self.augment = augment
        
        # Collect all image paths
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, f))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        # Initialize degradation pipeline
        self.degradation = DegradationPipeline(severity=severity)
        
        # Base transforms (resize and normalize)
        self.base_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
        ])
    
    def set_severity(self, severity: float):
        """Update degradation severity for progressive training."""
        self.severity = severity
        self.degradation.set_severity(severity)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            degraded: Degraded image tensor (C, H, W) normalized to [0, 1]
            clean: Clean image tensor (C, H, W) normalized to [0, 1]
        """
        # Load image
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random other image
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply base transforms
        img = self.base_transform(img)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
            
            # Random rotation (small angles)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, fill=128)
            
            # Random color jitter (subtle)
            if random.random() > 0.7:
                img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
                img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))
        
        # Convert to numpy for degradation
        clean_np = np.array(img)
        
        # Apply synthetic degradation
        degraded_np = self.degradation(clean_np)
        
        # Convert to tensors
        clean_tensor = numpy_to_tensor(clean_np)
        degraded_tensor = numpy_to_tensor(degraded_np)
        
        return degraded_tensor, clean_tensor


class ProgressiveDataLoader:
    """
    Wrapper around DataLoader that supports progressive training.
    Gradually increases degradation severity over epochs.
    """
    
    def __init__(
        self,
        dataset: KintsugiDataset,
        batch_size: int = 16,
        num_workers: int = 4,
        start_severity: float = 0.3,
        end_severity: float = 1.0,
        warmup_epochs: int = 10,
        shuffle: bool = True,
        pin_memory: bool = True
    ):
        """
        Args:
            dataset: KintsugiDataset instance
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            start_severity: Initial degradation severity
            end_severity: Final degradation severity
            warmup_epochs: Epochs to ramp from start to end severity
            shuffle: Whether to shuffle data
            pin_memory: Pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.start_severity = start_severity
        self.end_severity = end_severity
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    
    def set_epoch(self, epoch: int):
        """Update epoch and adjust degradation severity."""
        self.current_epoch = epoch
        
        # Linear ramp from start to end severity
        if epoch < self.warmup_epochs:
            severity = self.start_severity + (self.end_severity - self.start_severity) * (epoch / self.warmup_epochs)
        else:
            severity = self.end_severity
        
        self.dataset.set_severity(severity)
        print(f"Epoch {epoch}: Degradation severity = {severity:.2f}")
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)


def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    progressive: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Path to training images
        val_dir: Path to validation images (optional, will split train if None)
        image_size: Target image size
        batch_size: Batch size
        num_workers: Data loading workers
        progressive: Use progressive training
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = KintsugiDataset(
        root_dir=train_dir,
        image_size=image_size,
        severity=0.3 if progressive else 1.0,
        augment=True
    )
    
    if val_dir is not None:
        val_dataset = KintsugiDataset(
            root_dir=val_dir,
            image_size=image_size,
            severity=1.0,  # Full severity for validation
            augment=False
        )
    else:
        # Split train dataset
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        # Note: val_dataset inherits the same severity as train
    
    if progressive:
        train_loader = ProgressiveDataLoader(
            train_dataset if not isinstance(train_dataset, torch.utils.data.Subset) else train_dataset.dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
