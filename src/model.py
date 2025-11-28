"""
U-Net Architecture with Attention Gates for Kintsugi AI
Implements skip connections and attention mechanisms for high-quality image restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for focusing on relevant features.
    Helps the model focus on damaged regions during restoration.
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Feature map from decoder (lower resolution)
            skip: Feature map from encoder (skip connection)
        
        Returns:
            Attention-weighted skip connection
        """
        # Upsample gate to match skip resolution
        gate_up = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        g = self.W_gate(gate_up)
        s = self.W_skip(skip)
        
        # Attention coefficients
        attention = self.psi(self.relu(g + s))
        
        return skip * attention


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates for Image Restoration.
    
    Architecture:
    - Encoder: 4 downsampling stages with ConvBlocks
    - Bottleneck: ConvBlock + ResidualBlocks
    - Decoder: 4 upsampling stages with Attention Gates and skip connections
    - Output: 1x1 conv to map to 3 channels (RGB)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: List[int] = [64, 128, 256, 512],
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(ConvBlock(prev_channels, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feat
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(features[-1], features[-1] * 2),
            ResidualBlock(features[-1] * 2) if use_residual else nn.Identity(),
            ResidualBlock(features[-1] * 2) if use_residual else nn.Identity(),
        )
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        
        for i, feat in enumerate(reversed_features):
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(prev_channels, feat, feat // 2)
                )
            self.decoders.append(ConvBlock(feat * 2, feat))
            prev_channels = feat
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 3, H, W) with values in [0, 1]
        
        Returns:
            Restored image (B, 3, H, W) with values in [0, 1]
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[i]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Apply attention gate
            if self.use_attention:
                skip = self.attention_gates[i](x, skip)
            
            # Concatenate and decode
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        
        return self.output(x)


class UNet(nn.Module):
    """
    Standard U-Net without attention gates (simpler/faster alternative).
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(ConvBlock(prev_channels, feat))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feat
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        
        for feat in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(feat * 2, feat))
            prev_channels = feat
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[i]
            
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        
        return self.output(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    model_type: str = 'attention_unet',
    features: List[int] = [64, 128, 256, 512],
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'attention_unet' or 'unet'
        features: Channel sizes for each encoder/decoder stage
        device: Target device
    
    Returns:
        Model instance
    """
    if model_type == 'attention_unet':
        model = AttentionUNet(features=features, use_attention=True, use_residual=True)
    elif model_type == 'unet':
        model = UNet(features=features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"Created {model_type} with {count_parameters(model):,} trainable parameters")
    
    return model
