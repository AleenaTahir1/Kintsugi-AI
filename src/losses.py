"""
Composite Loss Functions for Kintsugi AI
Implements L1, SSIM, and Perceptual (VGG) losses for image restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional
import math


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    SSIM measures perceptual quality by comparing luminance, contrast, and structure.
    """
    
    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channels))
    
    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        """Create a Gaussian window for SSIM computation."""
        sigma = 1.5
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # Create 2D Gaussian window
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).
        
        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
        
        Returns:
            SSIM loss (lower is better)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        window = self.window.to(pred.device)
        
        mu_pred = F.conv2d(pred, window, padding=self.window_size // 2, groups=self.channels)
        mu_target = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channels)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=self.window_size // 2, groups=self.channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=self.channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=self.channels) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM value (higher is better, max 1.0)."""
        return 1 - self.forward(pred, target)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG16 features.
    Compares high-level features rather than pixel values for more realistic results.
    """
    
    def __init__(
        self,
        layers: List[int] = [3, 8, 15, 22],  # relu1_2, relu2_2, relu3_3, relu4_3
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Freeze VGG weights
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract feature layers
        self.layers = layers
        self.weights = weights or [1.0] * len(layers)
        
        # Split VGG into blocks
        self.blocks = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            block = nn.Sequential(*list(vgg.children())[prev_layer:layer + 1])
            self.blocks.append(block)
            prev_layer = layer + 1
        
        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]
        
        Returns:
            Perceptual loss
        """
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        loss = 0.0
        pred_feat = pred
        target_feat = target
        
        for i, block in enumerate(self.blocks):
            pred_feat = block(pred_feat)
            target_feat = block(target_feat)
            loss += self.weights[i] * F.l1_loss(pred_feat, target_feat)
        
        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable L1).
    More robust than L1/L2 for image restoration.
    """
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.epsilon ** 2))


class EdgeLoss(nn.Module):
    """
    Edge-aware loss using Sobel operator.
    Helps preserve sharp edges during restoration.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def get_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edges using Sobel operator."""
        edge_x = F.conv2d(x, self.sobel_x.to(x.device), padding=1, groups=3)
        edge_y = F.conv2d(x, self.sobel_y.to(x.device), padding=1, groups=3)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        return F.l1_loss(pred_edges, target_edges)


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss functions.
    Default: L1 + SSIM + Perceptual (VGG)
    
    Loss = λ1 * L1 + λ2 * SSIM + λ3 * Perceptual + λ4 * Edge
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        edge_weight: float = 0.1,
        use_charbonnier: bool = True
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        
        # Initialize losses
        self.l1_loss = CharbonnierLoss() if use_charbonnier else nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = VGGPerceptualLoss() if perceptual_weight > 0 else None
        self.edge_loss = EdgeLoss() if edge_weight > 0 else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute composite loss.
        
        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]
            return_components: If True, return dict with individual losses
        
        Returns:
            Total loss (and components dict if return_components=True)
        """
        losses = {}
        
        # L1 / Charbonnier loss
        l1 = self.l1_loss(pred, target)
        losses['l1'] = l1
        total = self.l1_weight * l1
        
        # SSIM loss
        if self.ssim_weight > 0:
            ssim = self.ssim_loss(pred, target)
            losses['ssim'] = ssim
            total += self.ssim_weight * ssim
        
        # Perceptual loss
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            perceptual = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual
            total += self.perceptual_weight * perceptual
        
        # Edge loss
        if self.edge_loss is not None and self.edge_weight > 0:
            edge = self.edge_loss(pred, target)
            losses['edge'] = edge
            total += self.edge_weight * edge
        
        losses['total'] = total
        
        if return_components:
            return total, losses
        return total


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        max_val: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR in dB (higher is better)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(max_val ** 2 / mse)
