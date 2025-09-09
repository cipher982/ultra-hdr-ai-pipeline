"""
HDR Preview Generation - Create web-displayable previews that simulate HDR effect.

Since browsers can't tone-map Ultra HDR JPEGs natively, we generate enhanced
SDR images that show the "HDR look" for web preview purposes.
"""

import numpy as np
from PIL import Image, ImageEnhance
import cv2
from pathlib import Path


def create_hdr_preview(input_path: str, output_path: str, strength: float = 1.0):
    """
    Create an HDR-looking preview image for web display.
    
    This applies tone mapping and enhancement to simulate what the HDR version
    would look like when viewed on an HDR display.
    
    Args:
        input_path: Path to original SDR image
        output_path: Path to save enhanced preview  
        strength: Enhancement strength (0.5 = subtle, 2.0 = dramatic)
    """
    
    # Load image
    with Image.open(input_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy for processing
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Apply HDR-style enhancement
        enhanced = apply_hdr_enhancement(img_array, strength)
        
        # Convert back to PIL and save
        enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))
        enhanced_img.save(output_path, "JPEG", quality=95, optimize=True)


def apply_hdr_enhancement(rgb_array: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply HDR-style enhancement to simulate gain map effect.
    
    This creates the visual "pop" that users expect to see in HDR images
    by enhancing contrast, highlights, and local details.
    """
    
    # Ensure strength is in reasonable range
    strength = np.clip(strength, 0.3, 3.0)
    
    # 1. Compute luminance for analysis
    luminance = 0.2126 * rgb_array[..., 0] + 0.7152 * rgb_array[..., 1] + 0.0722 * rgb_array[..., 2]
    
    # 2. Create adaptive gain map based on image content
    # Bright areas get more enhancement (simulates HDR headroom)
    gain_map = create_adaptive_gain_map(luminance, strength)
    
    # 3. Apply gain with proper tone mapping
    enhanced = apply_gain_with_tone_mapping(rgb_array, gain_map)
    
    # 4. Final polish - subtle contrast and saturation boost
    enhanced = apply_final_polish(enhanced, strength * 0.3)
    
    return np.clip(enhanced, 0.0, 1.0)


def create_adaptive_gain_map(luminance: np.ndarray, strength: float) -> np.ndarray:
    """Create a gain map that enhances bright areas more than dark areas."""
    
    # Base gain (slight enhancement everywhere)
    gain = np.ones_like(luminance) * (1.0 + strength * 0.2)
    
    # Progressive enhancement for brighter areas
    # This simulates how HDR images show more detail in highlights
    bright_mask = luminance > 0.3
    gain[bright_mask] *= (1.0 + strength * 0.8 * ((luminance[bright_mask] - 0.3) / 0.7))
    
    # Very bright areas get the most enhancement
    very_bright_mask = luminance > 0.7
    gain[very_bright_mask] *= (1.0 + strength * 0.5)
    
    # Smooth the gain map to avoid artifacts
    gain = cv2.GaussianBlur(gain, (15, 15), 5.0)
    
    return gain


def apply_gain_with_tone_mapping(rgb_array: np.ndarray, gain_map: np.ndarray) -> np.ndarray:
    """Apply gain map with proper tone mapping to prevent clipping."""
    
    # Apply gain to each channel
    enhanced = rgb_array * gain_map[..., np.newaxis]
    
    # Tone mapping to handle values > 1.0
    # This simulates how HDR displays compress high values into visible range
    enhanced = tone_map_reinhard(enhanced)
    
    return enhanced


def tone_map_reinhard(hdr_rgb: np.ndarray, key: float = 0.18) -> np.ndarray:
    """
    Apply Reinhard tone mapping to compress HDR values into SDR range.
    
    This is a simplified version of what HDR displays do automatically.
    """
    
    # Compute luminance of HDR image
    hdr_lum = 0.2126 * hdr_rgb[..., 0] + 0.7152 * hdr_rgb[..., 1] + 0.0722 * hdr_rgb[..., 2]
    
    # Avoid division by zero
    hdr_lum = np.maximum(hdr_lum, 1e-6)
    
    # Compute scale factor
    lum_avg = np.exp(np.mean(np.log(hdr_lum + 1e-6)))
    scaled_lum = (key / lum_avg) * hdr_lum
    
    # Reinhard operator
    tone_mapped_lum = scaled_lum / (1.0 + scaled_lum)
    
    # Apply tone mapping to RGB channels
    scale_factor = tone_mapped_lum / (hdr_lum + 1e-6)
    tone_mapped_rgb = hdr_rgb * scale_factor[..., np.newaxis]
    
    return tone_mapped_rgb


def apply_final_polish(rgb_array: np.ndarray, intensity: float) -> np.ndarray:
    """Apply final contrast and saturation enhancement."""
    
    # Convert to PIL for enhancement operations
    pil_img = Image.fromarray((np.clip(rgb_array, 0, 1) * 255).astype(np.uint8))
    
    # Subtle contrast boost
    contrast_factor = 1.0 + intensity * 0.3
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast_factor)
    
    # Subtle saturation boost  
    saturation_factor = 1.0 + intensity * 0.2
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(saturation_factor)
    
    # Convert back to numpy
    return np.array(pil_img).astype(np.float32) / 255.0