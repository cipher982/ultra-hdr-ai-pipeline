"""
Gain-map-first HDR pipeline for modern AI models.

This pipeline bypasses HDR generation entirely, using AI models that directly
predict gain maps from SDR input. This is the preferred architecture for
GMNet and other modern gain-map prediction models.
"""

import os
import json
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .models import GainMapModel, create_model, GainMapPrediction, ModelError
from .exporters.libultrahdr_wrapper import export_jpegr, LibUltraHDRConfig, UltraHDRExportError, is_available


@dataclass
class GainMapPipelineConfig:
    """Configuration for gain-map-first pipeline"""
    # Input processing
    max_side: int = 4096
    strip_exif: bool = True
    
    # AI model settings  
    model_type: str = "auto"  # "gmnet", "hdrcnn", "synthetic", "auto"
    model_path: Optional[str] = None
    
    # Export settings
    export_format: str = "auto"  # "jpeg_r", "heic", "both", "auto" 
    export_quality: int = 95
    gain_map_quality: int = 85
    
    # Output control  
    save_intermediate: bool = False  # Save gain map PNG, metadata JSON (debug only)
    require_export_success: bool = True
    timeout_s: int = 30
    
    # Fail-fast validation
    strict_mode: bool = True
    validate_outputs: bool = True


@dataclass 
class GainMapPipelineOutputs:
    """Results from gain-map pipeline execution"""
    ultrahdr_jpeg_path: Optional[str]  # Ultra HDR JPEG_R (primary output)
    heic_path: Optional[str]       # HEIC with gain map (Apple platforms)
    
    # Debug outputs (optional)
    gainmap_png_path: Optional[str]  # Raw gain map (debug only)
    meta_path: Optional[str]         # Pipeline metadata (debug only)
    
    # AI model info
    model_name: str
    model_confidence: Optional[float]


class GainMapPipelineError(Exception):
    """Raised when gain-map pipeline fails"""
    pass


def _prepare_input(path: str, cfg: GainMapPipelineConfig) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load and prepare SDR input image"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image not found: {path}")
        
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert('RGB')
        w, h = im.size
        
        # Resize if too large
        if max(w, h) > cfg.max_side:
            scale = cfg.max_side / float(max(w, h))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h
            
        # Convert to float32 [0,1]
        rgb_array = np.asarray(im).astype(np.float32) / 255.0
        return rgb_array, (h, w)
        
    except Exception as e:
        raise GainMapPipelineError(f"Failed to load input image {path}: {e}")


def _save_sdr_preview(rgb01: np.ndarray, path: str, strip_exif: bool = True) -> None:
    """Save SDR preview JPEG"""
    try:
        im = Image.fromarray((np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode='RGB')
        if strip_exif:
            im.info.pop('exif', None)
        im.save(path, format='JPEG', quality=95, optimize=True)
    except Exception as e:
        raise GainMapPipelineError(f"Failed to save SDR preview to {path}: {e}")


def _save_gainmap_png(gainmap: np.ndarray, path: str) -> None:
    """Save gain map as PNG"""
    try:
        if gainmap.dtype != np.uint8:
            raise ValueError(f"Gain map must be uint8, got {gainmap.dtype}")
        if len(gainmap.shape) != 2:
            raise ValueError(f"Gain map must be 2D, got shape {gainmap.shape}")
            
        Image.fromarray(gainmap, mode='L').save(path, format='PNG', optimize=True)
    except Exception as e:
        raise GainMapPipelineError(f"Failed to save gain map PNG to {path}: {e}")


def _validate_gain_map_output(prediction: GainMapPrediction) -> None:
    """Validate AI model gain map output"""
    gm = prediction.gain_map
    
    if gm is None:
        raise GainMapPipelineError("AI model returned None for gain map")
    if not isinstance(gm, np.ndarray):
        raise GainMapPipelineError(f"Expected numpy array for gain map, got {type(gm)}")
    if gm.dtype != np.uint8:
        raise GainMapPipelineError(f"Gain map must be uint8, got {gm.dtype}")
    if len(gm.shape) != 2:
        raise GainMapPipelineError(f"Gain map must be 2D grayscale, got shape {gm.shape}")
    if gm.size == 0:
        raise GainMapPipelineError("Gain map is empty")
        
    # Check for reasonable gain range
    if prediction.gain_max_log2 <= prediction.gain_min_log2:
        raise GainMapPipelineError(
            f"Invalid gain range: min={prediction.gain_min_log2}, max={prediction.gain_max_log2}"
        )
    if prediction.gain_max_log2 > 6.0:  # 64x boost seems excessive
        raise GainMapPipelineError(f"Suspicious gain_max_log2: {prediction.gain_max_log2} (>6 stops)")
        
    # Check gain map utilization (avoid all-zero or all-max maps)
    unique_values = len(np.unique(gm))
    if unique_values < 10:
        raise GainMapPipelineError(
            f"Gain map has only {unique_values} unique values - may be degenerate"
        )


def _export_ultrahdr(
    sdr_path: str, 
    gainmap_png_path: str,
    prediction: GainMapPrediction,
    out_path: str,
    config: GainMapPipelineConfig
) -> bool:
    """Export Ultra HDR JPEG using libultrahdr"""
    try:
        export_jpegr(
            sdr_jpeg_path=sdr_path,
            gainmap_path=gainmap_png_path,
            out_path=out_path,
            gain_min_log2=prediction.gain_min_log2,
            gain_max_log2=prediction.gain_max_log2,
            gamma=1.0,  # Standard gamma
            offset_sdr=0.0,  # Standard offsets  
            offset_hdr=0.0,
            cap_min_log2=max(0.0, prediction.gain_min_log2),
            cap_max_log2=prediction.gain_max_log2,
            config=LibUltraHDRConfig(
                quality=config.export_quality,
                gain_map_quality=config.gain_map_quality
            )
        )
        return True
    except (UltraHDRExportError, FileNotFoundError) as e:
        if config.strict_mode:
            raise GainMapPipelineError(f"Ultra HDR export failed: {e}")
        print(f"⚠️  Ultra HDR export failed: {e}")
        return False


def _export_heic_swift(
    sdr_path: str,
    gainmap_png_path: str, 
    prediction: GainMapPrediction,
    out_path: str,
    config: GainMapPipelineConfig
) -> bool:
    """Export HEIC using Swift exporter"""
    try:
        # Use existing Swift binary
        bin_dir = os.path.join(os.path.dirname(__file__), '..', 'bin')
        swift_bin = os.path.join(bin_dir, 'gainmap_exporter')
        
        if not os.path.exists(swift_bin):
            # Build it
            src = os.path.join(os.path.dirname(__file__), 'swift', 'GainMapExporter.swift')
            os.makedirs(bin_dir, exist_ok=True)
            subprocess.run(['swiftc', '-O', src, '-o', swift_bin], check=True, timeout=30)
        
        # Run Swift exporter for HEIC
        cmd = [
            swift_bin, 
            sdr_path, 
            gainmap_png_path, 
            out_path,
            str(prediction.gain_min_log2),
            str(prediction.gain_max_log2), 
            "1.0",  # gamma
            "0.0",  # offset_sdr
            "0.0",  # offset_hdr  
            str(max(0.0, prediction.gain_min_log2)),  # cap_min
            str(prediction.gain_max_log2),  # cap_max
            ""      # no reference
        ]
        
        subprocess.run(cmd, check=True, timeout=config.timeout_s, 
                      capture_output=True, text=True)
        return True
        
    except subprocess.CalledProcessError as e:
        if config.strict_mode:
            raise GainMapPipelineError(f"HEIC export failed: {e}")
        print(f"⚠️  HEIC export failed: {e}")
        return False
    except Exception as e:
        if config.strict_mode:
            raise GainMapPipelineError(f"HEIC export error: {e}")
        print(f"⚠️  HEIC export error: {e}")
        return False


def run_gainmap_pipeline(
    img_path: str,
    out_path: str, 
    config: Optional[GainMapPipelineConfig] = None
) -> GainMapPipelineOutputs:
    """
    Execute gain-map-first HDR pipeline.
    
    This is the main entry point for modern AI model integration.
    No HDR generation - AI model directly predicts gain maps.
    
    Args:
        img_path: Input SDR image path
        out_path: Output Ultra HDR JPEG path (if supported)
        config: Pipeline configuration
        
    Returns:
        GainMapPipelineOutputs with all generated files
        
    Raises:
        GainMapPipelineError: If pipeline fails
        FileNotFoundError: If input image not found
    """
    config = config or GainMapPipelineConfig()
    
    # Prepare output paths - use the exact user-specified output path
    base_dir = os.path.dirname(out_path)
    os.makedirs(base_dir, exist_ok=True)
    
    # Main output uses exactly what user specified
    main_output_path = out_path
    
    # Debug files only if requested
    if config.save_intermediate:
        input_stem = os.path.splitext(os.path.basename(img_path))[0]
        output_stem = os.path.splitext(os.path.basename(out_path))[0]
        gainmap_png_path = os.path.join(base_dir, f"{input_stem}_gainmap.png")
        meta_path = os.path.join(base_dir, f"{output_stem}_meta.json")
    else:
        gainmap_png_path = None
        meta_path = None
    
    try:
        # Step 1: Load and prepare input
        print("[1/4] Loading SDR input...")
        sdr_rgb, (orig_h, orig_w) = _prepare_input(img_path, config)
        
        # Step 2: AI model inference - GAIN MAP FIRST!
        print("[2/4] AI gain map prediction...")
        model = create_model(
            model_type=config.model_type,
            model_path=config.model_path,
            gmnet_path=config.model_path,
            hdrcnn_path=config.model_path,  # Legacy compatibility
            weights_path=config.model_path,  # Legacy compatibility  
            max_boost_stops=3.0
        )
        
        if not model.is_available():
            if config.strict_mode:
                raise GainMapPipelineError(f"AI model not available: {model.name}")
            print(f"⚠️  AI model unavailable ({model.name}), falling back to synthetic")
            from .models import SyntheticGainMapModel
            model = SyntheticGainMapModel()
            
        print(f"Using model: {model.name}")
        prediction = model.predict(sdr_rgb)
        
        # Step 3: Validate AI model output - FAIL FAST
        if config.validate_outputs:
            _validate_gain_map_output(prediction)
            
        # Step 4: Prepare files for export
        print("[3/4] Preparing export files...")
        
        # Always need temp SDR and gain map for exporters
        import tempfile
        temp_sdr_fd, temp_sdr_path = tempfile.mkstemp(suffix='.jpg', prefix='sdr_')
        os.close(temp_sdr_fd)
        _save_sdr_preview(sdr_rgb, temp_sdr_path, config.strip_exif)
        
        temp_gm_fd, temp_gm_path = tempfile.mkstemp(suffix='.png', prefix='gm_')
        os.close(temp_gm_fd)  
        _save_gainmap_png(prediction.gain_map, temp_gm_path)
        
        # Save debug files only if requested
        if config.save_intermediate:
            _save_gainmap_png(prediction.gain_map, gainmap_png_path)
            
        # Save metadata (debug only)
        if config.save_intermediate and meta_path:
            metadata = {
                'input': os.path.abspath(img_path),
                'output': os.path.abspath(main_output_path),
                'gainmap_png': os.path.abspath(gainmap_png_path) if gainmap_png_path else None,
                'model': {
                    'name': model.name,
                    'type': config.model_type,
                    'confidence': prediction.confidence,
                },
                'gainmap_meta': {
                    'gainMapMinLog2': prediction.gain_min_log2,
                    'gainMapMaxLog2': prediction.gain_max_log2,
                    'gamma': 1.0,
                    'offsetSDR': 0.0,
                    'offsetHDR': 0.0,
                    'hdrCapacityMin': max(0.0, prediction.gain_min_log2),
                    'hdrCapacityMax': prediction.gain_max_log2,
                },
                'dimensions': {'height': orig_h, 'width': orig_w}
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        # Step 5: Export Ultra HDR containers
        print("[4/4] Exporting Ultra HDR containers...")
        
        ultrahdr_jpeg_path = None
        heic_output_path = None
        
        # Export Ultra HDR containers using temp files
        try:
            # Try libultrahdr for JPEG_R (cross-platform) 
            if config.export_format in ["auto", "jpeg_r", "both"]:
                if is_available():
                    try:
                        success = _export_ultrahdr(temp_sdr_path, temp_gm_path, prediction, main_output_path, config)
                        if success:
                            ultrahdr_jpeg_path = main_output_path
                            print(f"✅ Ultra HDR JPEG_R: {main_output_path}")
                        else:
                            print("⚠️  Ultra HDR JPEG_R export failed")
                    except Exception as e:
                        if config.strict_mode:
                            raise
                        print(f"⚠️  Ultra HDR JPEG_R error: {e}")
                else:
                    print("⚠️  libultrahdr not available, skipping JPEG_R export")
                    
            # Try HEIC export (macOS only)
            if config.export_format in ["heic", "both"] and platform.system() == "Darwin":
                try:
                    success = _export_heic_swift(temp_sdr_path, temp_gm_path, prediction, main_output_path, config)
                    if success:
                        heic_output_path = main_output_path
                        print(f"✅ HEIC: {main_output_path}")
                    else:
                        print("⚠️  HEIC export failed")
                except Exception as e:
                    if config.strict_mode:
                        raise
                    print(f"⚠️  HEIC export error: {e}")
        
        finally:
            # Always cleanup temp files
            for temp_file in [temp_sdr_path, temp_gm_path]:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            
        # Validation: At least one export should succeed
        if config.require_export_success and not ultrahdr_jpeg_path and not heic_output_path:
            raise GainMapPipelineError(
                "No Ultra HDR containers were successfully created. "
                f"libultrahdr available: {is_available()}, "
                f"Platform: {platform.system()}"
            )
            
        return GainMapPipelineOutputs(
            ultrahdr_jpeg_path=ultrahdr_jpeg_path,
            heic_path=heic_output_path,
            gainmap_png_path=gainmap_png_path if config.save_intermediate else None,
            meta_path=meta_path if config.save_intermediate else None,
            model_name=model.name,
            model_confidence=prediction.confidence
        )
        
    except ModelError as e:
        raise GainMapPipelineError(f"AI model failed: {e}")
    except Exception as e:
        if config.strict_mode:
            raise GainMapPipelineError(f"Pipeline failed: {e}")
        raise


def _validate_ultrahdr_output(path: str, expected_min_size: int) -> None:
    """Validate Ultra HDR output - FAIL FAST on silent failures"""
    if not os.path.exists(path):
        raise GainMapPipelineError(f"Ultra HDR output not created: {path}")
        
    file_size = os.path.getsize(path)
    if file_size < expected_min_size:
        raise GainMapPipelineError(
            f"Ultra HDR output suspiciously small: {file_size} bytes "
            f"(expected >{expected_min_size}). May be regular JPEG."
        )
        
    # Check JPEG header
    with open(path, 'rb') as f:
        header = f.read(20)
        if not header.startswith(b'\xff\xd8'):
            raise GainMapPipelineError(f"Invalid JPEG header in {path}")
            
    # TODO: Add metadata validation with exiftool
    print(f"✅ Ultra HDR validation passed: {file_size} bytes")


# CLI integration
def main():
    """Main entry point - delegate to CLI"""
    from .cli import main
    main()