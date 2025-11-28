"""
Inference utilities for Kintsugi AI
Load trained models and restore images.
"""

import os
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from .model import create_model, AttentionUNet
from .degradation import numpy_to_tensor, tensor_to_numpy


class Restorer:
    """
    Image restoration inference class.
    Loads a trained model and provides methods to restore damaged images.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = 'attention_unet',
        features: list = [64, 128, 256, 512],
        device: str = 'auto'
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            model_type: 'attention_unet' or 'unet'
            features: Model feature dimensions (must match training)
            device: 'cuda', 'cpu', or 'auto'
        """
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Create and load model
        self.model = create_model(model_type, features, self.device)
        self.load_model(checkpoint_path)
        self.model.eval()
        
        print(f"Restorer initialized on {self.device}")
    
    def load_model(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded model from {path}")
    
    def preprocess(self, image: np.ndarray, target_size: Optional[int] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, 3) in BGR or RGB format
            target_size: Optional size to resize to (preserves aspect ratio)
        
        Returns:
            tensor: Preprocessed tensor (1, 3, H, W)
            original_size: Original (H, W) for later resizing
        """
        original_size = image.shape[:2]
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed (keeping aspect ratio)
        if target_size is not None:
            h, w = image.shape[:2]
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to multiple of 32 (for U-Net)
            pad_h = (32 - new_h % 32) % 32
            pad_w = (32 - new_w % 32) % 32
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        # Convert to tensor
        tensor = numpy_to_tensor(image).unsqueeze(0).to(self.device)
        
        return tensor, original_size
    
    def postprocess(self, tensor: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output.
        
        Args:
            tensor: Model output (1, 3, H, W)
            original_size: Original (H, W) to resize back
        
        Returns:
            image: Restored image (H, W, 3) in RGB format, uint8
        """
        image = tensor_to_numpy(tensor.squeeze(0))
        
        # Resize back to original size
        h, w = original_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return image
    
    @torch.no_grad()
    def restore(
        self,
        image: np.ndarray,
        target_size: Optional[int] = 512,
        tile_size: Optional[int] = None,
        tile_overlap: int = 32
    ) -> np.ndarray:
        """
        Restore a damaged image.
        
        Args:
            image: Input damaged image (H, W, 3)
            target_size: Resize to this size for processing (None = keep original)
            tile_size: If set, process in tiles (for large images)
            tile_overlap: Overlap between tiles
        
        Returns:
            Restored image (H, W, 3) in RGB format
        """
        if tile_size is not None:
            return self._restore_tiled(image, tile_size, tile_overlap)
        
        # Standard processing
        tensor, original_size = self.preprocess(image, target_size)
        output = self.model(tensor)
        restored = self.postprocess(output, original_size)
        
        return restored
    
    def _restore_tiled(
        self,
        image: np.ndarray,
        tile_size: int,
        overlap: int
    ) -> np.ndarray:
        """Process large images in overlapping tiles."""
        h, w = image.shape[:2]
        
        # Prepare output
        output = np.zeros_like(image, dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        # Process tiles
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                # Extract tile
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                
                # Process tile
                tensor, _ = self.preprocess(tile, None)
                with torch.no_grad():
                    tile_output = self.model(tensor)
                restored_tile = tensor_to_numpy(tile_output.squeeze(0))
                
                # Blend into output with weight falloff at edges
                tile_h, tile_w = restored_tile.shape[:2]
                weight = self._create_weight_mask(tile_h, tile_w, overlap)
                
                output[y_start:y_end, x_start:x_end] += restored_tile * weight[:, :, np.newaxis]
                weights[y_start:y_end, x_start:x_end] += weight
        
        # Normalize by weights
        weights = np.maximum(weights, 1e-6)
        output = output / weights[:, :, np.newaxis]
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def _create_weight_mask(self, h: int, w: int, overlap: int) -> np.ndarray:
        """Create a weight mask with smooth falloff at edges."""
        weight = np.ones((h, w), dtype=np.float32)
        
        # Create ramp for edges
        ramp = np.linspace(0, 1, overlap)
        
        # Apply ramps to edges
        if overlap > 0:
            weight[:overlap, :] *= ramp[:, np.newaxis]
            weight[-overlap:, :] *= ramp[::-1, np.newaxis]
            weight[:, :overlap] *= ramp[np.newaxis, :]
            weight[:, -overlap:] *= ramp[::-1][np.newaxis, :]
        
        return weight
    
    def restore_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Restore an image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save restored image (optional)
            **kwargs: Additional arguments for restore()
        
        Returns:
            Restored image
        """
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        # Restore
        restored = self.restore(image, **kwargs)
        
        # Save if requested
        if output_path is not None:
            # Convert RGB to BGR for saving
            restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, restored_bgr)
            print(f"Saved restored image to {output_path}")
        
        return restored


def compare_images(
    degraded: np.ndarray,
    restored: np.ndarray,
    ground_truth: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create side-by-side comparison image.
    
    Args:
        degraded: Degraded input image
        restored: Restored output image
        ground_truth: Original clean image (optional)
    
    Returns:
        Comparison image
    """
    images = [degraded, restored]
    titles = ['Degraded', 'Restored']
    
    if ground_truth is not None:
        images.append(ground_truth)
        titles.append('Ground Truth')
    
    # Resize all to same height
    target_h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        scale = target_h / img.shape[0]
        new_w = int(img.shape[1] * scale)
        resized.append(cv2.resize(img, (new_w, target_h)))
    
    # Concatenate horizontally
    comparison = np.hstack(resized)
    
    return comparison
