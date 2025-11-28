"""
Configuration for Kintsugi AI Training
Adjust these parameters based on your hardware and dataset.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data configuration."""
    train_dir: str = "data/train"           # Path to training images
    val_dir: Optional[str] = "data/val"     # Path to validation images (optional)
    image_size: int = 256                    # Training image size
    batch_size: int = 16                     # Batch size (reduce if OOM)
    num_workers: int = 4                     # Data loading workers


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "attention_unet"       # 'attention_unet' or 'unet'
    features: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_attention: bool = True
    use_residual: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100                        # Total training epochs
    lr: float = 1e-4                         # Learning rate
    weight_decay: float = 1e-4               # Weight decay for AdamW
    grad_accum_steps: int = 1                # Gradient accumulation steps
    use_amp: bool = True                     # Mixed precision training
    
    # Progressive training
    progressive: bool = True                 # Use progressive degradation
    start_severity: float = 0.3              # Initial degradation severity
    end_severity: float = 1.0                # Final degradation severity
    warmup_epochs: int = 10                  # Epochs to ramp severity


@dataclass
class LossConfig:
    """Loss function weights."""
    l1_weight: float = 1.0                   # L1/Charbonnier loss weight
    ssim_weight: float = 0.5                 # SSIM loss weight
    perceptual_weight: float = 0.1           # VGG perceptual loss weight
    edge_weight: float = 0.1                 # Edge loss weight


@dataclass
class Config:
    """Full configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Paths
    save_dir: str = "checkpoints"            # Checkpoint save directory
    log_dir: str = "logs"                    # Tensorboard log directory
    
    # Logging
    save_every: int = 5                      # Save checkpoint every N epochs
    log_every: int = 10                      # Log every N batches


# Default configuration
DEFAULT_CONFIG = Config()


# Colab-optimized configuration (for T4/V100 GPUs)
COLAB_CONFIG = Config(
    data=DataConfig(
        batch_size=32,                       # Larger batch for GPU
        num_workers=2,                       # Colab has limited workers
        image_size=256
    ),
    training=TrainingConfig(
        epochs=50,                           # Fewer epochs for demo
        lr=2e-4,                             # Slightly higher LR
        use_amp=True,                        # Always use AMP on Colab
        grad_accum_steps=2                   # Effective batch size = 64
    )
)


# Light configuration (for quick testing)
LIGHT_CONFIG = Config(
    data=DataConfig(
        batch_size=8,
        image_size=128
    ),
    model=ModelConfig(
        features=[32, 64, 128, 256]          # Smaller model
    ),
    training=TrainingConfig(
        epochs=10
    )
)
