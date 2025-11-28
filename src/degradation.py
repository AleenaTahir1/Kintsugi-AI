"""
Synthetic Degradation Pipeline for Kintsugi AI
Applies realistic damage to clean images for self-supervised training.
"""

import numpy as np
import cv2
from PIL import Image
import random
from typing import Tuple, Optional
import torch


class DegradationPipeline:
    """
    Applies various synthetic degradations to simulate historical photo damage.
    Supports progressive training by adjusting severity levels.
    """
    
    def __init__(self, severity: float = 1.0):
        """
        Args:
            severity: Float 0.0-1.0 controlling degradation intensity.
                     Use for progressive training (start low, increase over epochs).
        """
        self.severity = np.clip(severity, 0.0, 1.0)
    
    def set_severity(self, severity: float):
        """Update severity for progressive training."""
        self.severity = np.clip(severity, 0.0, 1.0)
    
    def apply_scratches(self, img: np.ndarray, num_scratches: Optional[int] = None) -> np.ndarray:
        """
        Add bezier curve scratches simulating cracks/scratches on old photos.
        """
        h, w = img.shape[:2]
        result = img.copy()
        
        if num_scratches is None:
            num_scratches = int(random.randint(3, 10) * self.severity)
        
        for _ in range(num_scratches):
            # Random bezier curve control points
            p0 = (random.randint(0, w), random.randint(0, h))
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            p3 = (random.randint(0, w), random.randint(0, h))
            
            # Generate bezier curve points
            points = self._bezier_curve(p0, p1, p2, p3, num_points=100)
            
            # Scratch properties
            color = random.choice([
                (255, 255, 255),  # White scratch
                (200, 200, 200),  # Light gray
                (50, 50, 50),     # Dark scratch
                (0, 0, 0),        # Black
            ])
            thickness = random.randint(1, max(1, int(3 * self.severity)))
            
            # Draw the scratch
            for i in range(len(points) - 1):
                cv2.line(result, points[i], points[i+1], color, thickness)
        
        return result
    
    def _bezier_curve(self, p0, p1, p2, p3, num_points=100):
        """Generate points along a cubic bezier curve."""
        points = []
        for t in np.linspace(0, 1, num_points):
            x = int((1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0])
            y = int((1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1])
            points.append((x, y))
        return points
    
    def apply_gaussian_noise(self, img: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        """Add Gaussian noise simulating film grain."""
        if std is None:
            std = random.uniform(5, 30) * self.severity
        
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_salt_pepper_noise(self, img: np.ndarray, amount: Optional[float] = None) -> np.ndarray:
        """Add salt and pepper noise simulating dust/debris."""
        if amount is None:
            amount = random.uniform(0.01, 0.05) * self.severity
        
        result = img.copy()
        h, w = img.shape[:2]
        
        # Salt (white pixels)
        num_salt = int(amount * h * w)
        coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
        result[coords[0], coords[1]] = 255
        
        # Pepper (black pixels)
        num_pepper = int(amount * h * w)
        coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
        result[coords[0], coords[1]] = 0
        
        return result
    
    def apply_random_mask(self, img: np.ndarray, num_masks: Optional[int] = None) -> np.ndarray:
        """
        Add rectangular masks simulating missing paper chunks or tears.
        """
        h, w = img.shape[:2]
        result = img.copy()
        
        if num_masks is None:
            num_masks = int(random.randint(1, 5) * self.severity)
        
        for _ in range(num_masks):
            # Random rectangle size (relative to image)
            mask_h = int(random.uniform(0.05, 0.15) * h * self.severity)
            mask_w = int(random.uniform(0.05, 0.15) * w * self.severity)
            
            # Random position
            x = random.randint(0, max(0, w - mask_w))
            y = random.randint(0, max(0, h - mask_h))
            
            # Fill with noise or solid color
            if random.random() > 0.5:
                result[y:y+mask_h, x:x+mask_w] = np.random.randint(0, 255, (mask_h, mask_w, 3))
            else:
                color = random.choice([(255, 255, 255), (0, 0, 0), (128, 100, 80)])
                result[y:y+mask_h, x:x+mask_w] = color
        
        return result
    
    def apply_color_fading(self, img: np.ndarray, strength: Optional[float] = None) -> np.ndarray:
        """
        Apply sepia/fading effect simulating aged photographs.
        """
        if strength is None:
            strength = random.uniform(0.3, 0.8) * self.severity
        
        # Convert to sepia
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia = cv2.transform(img, sepia_filter)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        # Blend with original based on strength
        result = cv2.addWeighted(img, 1 - strength, sepia, strength, 0)
        
        # Reduce contrast
        contrast_factor = 1 - (0.3 * strength)
        result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=30 * strength)
        
        return result
    
    def apply_blur(self, img: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        """Apply blur simulating water damage or focus issues."""
        if kernel_size is None:
            kernel_size = int(random.choice([3, 5, 7]) * self.severity)
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def apply_folding_lines(self, img: np.ndarray, num_folds: Optional[int] = None) -> np.ndarray:
        """Add folding/crease lines simulating paper folds."""
        h, w = img.shape[:2]
        result = img.copy()
        
        if num_folds is None:
            num_folds = int(random.randint(1, 3) * self.severity)
        
        for _ in range(num_folds):
            # Horizontal or vertical fold
            if random.random() > 0.5:
                # Horizontal fold
                y = random.randint(int(h * 0.2), int(h * 0.8))
                thickness = random.randint(1, max(1, int(3 * self.severity)))
                color = random.choice([(200, 200, 200), (180, 180, 180), (220, 220, 220)])
                cv2.line(result, (0, y), (w, y), color, thickness)
                # Add slight darkening along fold
                result[max(0, y-2):min(h, y+2), :] = (result[max(0, y-2):min(h, y+2), :] * 0.9).astype(np.uint8)
            else:
                # Vertical fold
                x = random.randint(int(w * 0.2), int(w * 0.8))
                thickness = random.randint(1, max(1, int(3 * self.severity)))
                color = random.choice([(200, 200, 200), (180, 180, 180), (220, 220, 220)])
                cv2.line(result, (x, 0), (x, h), color, thickness)
                result[:, max(0, x-2):min(w, x+2)] = (result[:, max(0, x-2):min(w, x+2)] * 0.9).astype(np.uint8)
        
        return result
    
    def apply_stains(self, img: np.ndarray, num_stains: Optional[int] = None) -> np.ndarray:
        """Add ink/water stains simulating liquid damage."""
        h, w = img.shape[:2]
        result = img.copy().astype(np.float32)
        
        if num_stains is None:
            num_stains = int(random.randint(1, 4) * self.severity)
        
        for _ in range(num_stains):
            # Random ellipse for stain shape
            center = (random.randint(0, w), random.randint(0, h))
            axes = (
                int(random.uniform(0.05, 0.15) * w * self.severity),
                int(random.uniform(0.05, 0.15) * h * self.severity)
            )
            angle = random.randint(0, 180)
            
            # Create stain mask
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Stain color (brownish/yellowish for age spots)
            stain_color = np.array([
                random.uniform(60, 120),  # B
                random.uniform(80, 140),  # G
                random.uniform(100, 180)  # R
            ])
            
            # Apply stain
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - mask * 0.5) + stain_color[c] * mask * 0.5
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def __call__(self, img: np.ndarray, apply_all: bool = False) -> np.ndarray:
        """
        Apply random degradations to an image.
        
        Args:
            img: Input image (H, W, 3) in BGR format, uint8
            apply_all: If True, apply all degradations. Otherwise, random subset.
        
        Returns:
            Degraded image
        """
        result = img.copy()
        
        # List of degradation functions
        degradations = [
            (self.apply_scratches, 0.7),
            (self.apply_gaussian_noise, 0.6),
            (self.apply_salt_pepper_noise, 0.4),
            (self.apply_random_mask, 0.5),
            (self.apply_color_fading, 0.5),
            (self.apply_blur, 0.3),
            (self.apply_folding_lines, 0.3),
            (self.apply_stains, 0.4),
        ]
        
        if apply_all:
            for deg_fn, _ in degradations:
                result = deg_fn(result)
        else:
            # Apply random subset based on probability
            for deg_fn, prob in degradations:
                if random.random() < prob * self.severity:
                    result = deg_fn(result)
        
        return result


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image (H, W, C) to tensor (C, H, W) normalized to [0, 1]."""
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (C, H, W) to numpy image (H, W, C) in uint8."""
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
