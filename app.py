"""
Gradio Web Interface for Kintsugi AI
Upload damaged images and get restored versions.
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import torch
import os

from src.inference import Restorer, compare_images
from src.degradation import DegradationPipeline


# Global restorer (loaded once)
restorer = None


def load_model(checkpoint_path: str = "checkpoints/best_model.pth"):
    """Load the restoration model."""
    global restorer
    
    if not os.path.exists(checkpoint_path):
        return "Model checkpoint not found. Please train the model first."
    
    try:
        restorer = Restorer(
            checkpoint_path=checkpoint_path,
            model_type='attention_unet',
            device='auto'
        )
        return f"Model loaded successfully on {restorer.device}"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def restore_image(
    input_image: np.ndarray,
    tile_mode: bool = False,
    tile_size: int = 512
) -> tuple:
    """
    Restore an uploaded image.
    
    Args:
        input_image: Input image from Gradio
        tile_mode: Process in tiles for large images
        tile_size: Tile size if tile_mode is True
    
    Returns:
        Tuple of (restored_image, comparison_image)
    """
    global restorer
    
    if restorer is None:
        # Try to load model
        result = load_model()
        if "Error" in result or "not found" in result:
            return None, result
    
    if input_image is None:
        return None, "Please upload an image"
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        
        # Ensure BGR format for OpenCV
        if len(input_image.shape) == 2:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        elif input_image.shape[2] == 4:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2BGR)
        elif input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        # Restore
        if tile_mode:
            restored = restorer.restore(input_image, target_size=None, tile_size=tile_size)
        else:
            restored = restorer.restore(input_image, target_size=512)
        
        # Create comparison
        comparison = compare_images(
            cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
            restored
        )
        
        return restored, comparison
    
    except Exception as e:
        return None, f"Error during restoration: {str(e)}"


def apply_synthetic_damage(
    input_image: np.ndarray,
    severity: float = 0.7,
    scratches: bool = True,
    noise: bool = True,
    masks: bool = True,
    fading: bool = True,
    stains: bool = True
) -> np.ndarray:
    """
    Apply synthetic damage to a clean image for testing.
    """
    if input_image is None:
        return None
    
    # Convert to numpy
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    
    # Ensure BGR
    if input_image.shape[2] == 4:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2BGR)
    elif input_image.shape[2] == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Apply degradation
    pipeline = DegradationPipeline(severity=severity)
    result = input_image.copy()
    
    if scratches:
        result = pipeline.apply_scratches(result)
    if noise:
        result = pipeline.apply_gaussian_noise(result)
        result = pipeline.apply_salt_pepper_noise(result)
    if masks:
        result = pipeline.apply_random_mask(result)
    if fading:
        result = pipeline.apply_color_fading(result)
    if stains:
        result = pipeline.apply_stains(result)
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def create_demo():
    """Create the Gradio demo interface."""
    
    with gr.Blocks(title="Kintsugi AI - Image Restoration") as demo:
        gr.Markdown("""
        # 🎌 Kintsugi AI - Digital Image Restoration
        
        Upload a damaged historical photo and watch AI restore it to its former glory.
        Like the Japanese art of Kintsugi, we heal the broken with care and precision.
        
        ---
        """)
        
        with gr.Tab("Restore Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Damaged Image", type="numpy")
                    tile_mode = gr.Checkbox(label="Tile Mode (for large images)", value=False)
                    tile_size = gr.Slider(256, 1024, 512, step=128, label="Tile Size", visible=False)
                    restore_btn = gr.Button("Restore Image", variant="primary")
                
                with gr.Column():
                    restored_image = gr.Image(label="Restored Image")
                    comparison_image = gr.Image(label="Side-by-Side Comparison")
            
            tile_mode.change(
                lambda x: gr.update(visible=x),
                inputs=[tile_mode],
                outputs=[tile_size]
            )
            
            restore_btn.click(
                restore_image,
                inputs=[input_image, tile_mode, tile_size],
                outputs=[restored_image, comparison_image]
            )
        
        with gr.Tab("Test with Synthetic Damage"):
            gr.Markdown("""
            ### Generate Test Images
            Upload a clean image and apply synthetic damage to test the restoration model.
            """)
            
            with gr.Row():
                with gr.Column():
                    clean_image = gr.Image(label="Upload Clean Image", type="numpy")
                    severity = gr.Slider(0.1, 1.0, 0.7, label="Damage Severity")
                    
                    with gr.Row():
                        scratches = gr.Checkbox(label="Scratches", value=True)
                        noise = gr.Checkbox(label="Noise", value=True)
                        masks = gr.Checkbox(label="Missing Patches", value=True)
                    
                    with gr.Row():
                        fading = gr.Checkbox(label="Color Fading", value=True)
                        stains = gr.Checkbox(label="Stains", value=True)
                    
                    damage_btn = gr.Button("Apply Damage", variant="secondary")
                
                with gr.Column():
                    damaged_output = gr.Image(label="Damaged Image")
            
            damage_btn.click(
                apply_synthetic_damage,
                inputs=[clean_image, severity, scratches, noise, masks, fading, stains],
                outputs=[damaged_output]
            )
        
        gr.Markdown("""
        ---
        ### About
        
        Kintsugi AI uses a U-Net architecture with attention gates trained on synthetically 
        degraded images. The model learns to remove scratches, noise, stains, and restore 
        missing patches while preserving important details.
        
        **Model:** Attention U-Net with skip connections  
        **Loss:** L1 + SSIM + Perceptual (VGG)  
        **Training:** Self-supervised with synthetic degradation
        """)
    
    return demo


if __name__ == "__main__":
    # Try to load model at startup
    load_model()
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(share=True)
