"""
Ultra HDR (JPEG_R) exporter using Google libultrahdr.

This module wraps libultrahdr API-4 to create standards-compliant Ultra HDR JPEG files
from SDR base + gain map + metadata. This is the PRIMARY exporter for cross-platform
compatibility (Android 14+, future iOS support).
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


class UltraHDRExportError(Exception):
    """Raised when libultrahdr export fails"""
    pass


@dataclass
class LibUltraHDRConfig:
    """Configuration for libultrahdr encoding"""
    quality: int = 95                    # JPEG quality for base and gain map
    gamma: float = 1.0                   # Gain map gamma
    gain_map_quality: int = 85           # Gain map compression quality  
    use_multi_channel: bool = False      # Use RGB gain map vs grayscale
    min_content_boost: float = 1.0       # Minimum boost multiplier
    max_content_boost: Optional[float] = None  # Maximum boost (auto from gain_max)


def _find_libultrahdr_binary() -> str:
    """Find libultrahdr binary, build if necessary"""
    tools_dir = Path(__file__).parent.parent.parent / "tools" / "libultrahdr"
    
    # Check for pre-built binary
    possible_paths = [
        tools_dir / "build" / "ultrahdr_app",
        tools_dir / "build" / "Release" / "ultrahdr_app",
        tools_dir / "build" / "Debug" / "ultrahdr_app", 
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return str(path)
    
    # Try to build it
    build_dir = tools_dir / "build"
    if tools_dir.exists():
        try:
            os.makedirs(build_dir, exist_ok=True)
            
            # Run CMake configuration
            subprocess.run([
                "cmake", str(tools_dir), 
                "-DCMAKE_BUILD_TYPE=Release",
                "-DUHDR_BUILD_TESTS=OFF"
            ], cwd=build_dir, check=True, capture_output=True)
            
            # Build
            subprocess.run([
                "cmake", "--build", str(build_dir), "--config", "Release"
            ], check=True, capture_output=True)
            
            # Check if binary was created
            for path in possible_paths:
                if path.exists():
                    return str(path)
                    
        except subprocess.CalledProcessError as e:
            raise UltraHDRExportError(f"Failed to build libultrahdr: {e}")
    
    raise UltraHDRExportError("libultrahdr binary not found and could not be built")


def export_jpegr(
    sdr_jpeg_path: str,
    gainmap_path: str, 
    out_path: str,
    gain_min_log2: float,
    gain_max_log2: float,
    gamma: float = 1.0,
    offset_sdr: float = 0.0,
    offset_hdr: float = 0.0,
    cap_min_log2: float = 0.0,
    cap_max_log2: Optional[float] = None,
    config: Optional[LibUltraHDRConfig] = None,
) -> None:
    """
    Export Ultra HDR JPEG (JPEG_R) using libultrahdr API-4.
    
    This creates proper Multi-Picture Format (MPF) JPEG containers with embedded
    gain maps that work on Android 14+ and other Ultra HDR compatible platforms.
    
    Args:
        sdr_jpeg_path: Path to SDR base JPEG
        gainmap_path: Path to gain map (PNG or JPEG)
        out_path: Output Ultra HDR JPEG path
        gain_min_log2: Minimum gain in log2 stops
        gain_max_log2: Maximum gain in log2 stops
        gamma: Gain map gamma (default 1.0)
        offset_sdr: SDR offset (default 0.0)
        offset_hdr: HDR offset (default 0.0) 
        cap_min_log2: HDR capacity minimum
        cap_max_log2: HDR capacity maximum (defaults to gain_max_log2)
        config: LibUltraHDR configuration
        
    Raises:
        UltraHDRExportError: If export fails
        FileNotFoundError: If input files don't exist
    """
    config = config or LibUltraHDRConfig()
    
    if cap_max_log2 is None:
        cap_max_log2 = gain_max_log2
    
    # Validate inputs - FAIL FAST
    if not os.path.exists(sdr_jpeg_path):
        raise FileNotFoundError(f"SDR JPEG not found: {sdr_jpeg_path}")
    if not os.path.exists(gainmap_path):
        raise FileNotFoundError(f"Gain map not found: {gainmap_path}")
        
    # Ensure gain map is JPEG (libultrahdr expects compressed gain map)
    gainmap_jpeg_path = gainmap_path
    if gainmap_path.lower().endswith('.png'):
        # Convert PNG gain map to JPEG
        gainmap_jpeg_path = _convert_gainmap_to_jpeg(gainmap_path, config.gain_map_quality)
    
    # Create metadata config file for libultrahdr
    metadata_config = _create_metadata_config(
        gain_min_log2, gain_max_log2, gamma, offset_sdr, offset_hdr, 
        cap_min_log2, cap_max_log2, config
    )
    
    try:
        binary_path = _find_libultrahdr_binary()
        
        # Build libultrahdr command for API-4 (compressed SDR + compressed gain map + metadata)
        cmd = [
            binary_path,
            "-m", "0",  # Encoding mode  
            "-i", sdr_jpeg_path,           # Compressed SDR input
            "-g", gainmap_jpeg_path,       # Compressed gain map input
            "-f", metadata_config,         # Metadata config file
            "-o", out_path,               # Output Ultra HDR JPEG
            "-q", str(config.quality),    # Quality
        ]
        
        # Debug: Print command for troubleshooting
        print(f"🔍 libultrahdr command: {' '.join(cmd)}")
        
        # Execute with proper error handling
        result = subprocess.run(
            cmd, 
            check=False,  # Don't auto-raise to capture output
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        # Manual error handling with detailed output
        if result.returncode != 0:
            print(f"❌ libultrahdr failed (exit code {result.returncode})")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        # MANDATORY VALIDATION - fail fast if output is wrong
        if not os.path.exists(out_path):
            raise UltraHDRExportError("libultrahdr completed but no output file created")
            
        # Verify it's actually Ultra HDR, not regular JPEG
        file_size = os.path.getsize(out_path)
        sdr_size = os.path.getsize(sdr_jpeg_path)
        gm_size = os.path.getsize(gainmap_jpeg_path)
        
        # Ultra HDR should be roughly SDR + gain map + overhead
        expected_min_size = sdr_size + (gm_size * 0.5)  # Allow compression
        if file_size < expected_min_size:
            raise UltraHDRExportError(
                f"Output suspiciously small ({file_size} bytes vs expected >{expected_min_size}). "
                f"May be regular JPEG, not Ultra HDR container."
            )
            
        print(f"✅ libultrahdr export successful: {file_size} bytes "
              f"(SDR: {sdr_size}, GM: {gm_size})")
        
    except subprocess.CalledProcessError as e:
        raise UltraHDRExportError(
            f"libultrahdr encoding failed: {e}\n"
            f"stdout: {e.stdout}\nstderr: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired:
        raise UltraHDRExportError("libultrahdr encoding timed out")
    finally:
        # Cleanup temporary files
        if gainmap_jpeg_path != gainmap_path and os.path.exists(gainmap_jpeg_path):
            os.unlink(gainmap_jpeg_path)
        if os.path.exists(metadata_config):
            os.unlink(metadata_config)


def _convert_gainmap_to_jpeg(png_path: str, quality: int) -> str:
    """Convert PNG gain map to JPEG for libultrahdr"""
    from PIL import Image
    
    temp_fd, jpeg_path = tempfile.mkstemp(suffix='.jpg')
    os.close(temp_fd)
    
    try:
        with Image.open(png_path) as img:
            # Ensure grayscale
            if img.mode != 'L':
                img = img.convert('L')
            img.save(jpeg_path, format='JPEG', quality=quality, optimize=True)
        return jpeg_path
    except Exception as e:
        if os.path.exists(jpeg_path):
            os.unlink(jpeg_path)
        raise UltraHDRExportError(f"Failed to convert gain map PNG to JPEG: {e}")


def _create_metadata_config(
    gain_min_log2: float, gain_max_log2: float, gamma: float,
    offset_sdr: float, offset_hdr: float, cap_min_log2: float, 
    cap_max_log2: float, config: LibUltraHDRConfig
) -> str:
    """Create libultrahdr metadata configuration file"""
    temp_fd, config_path = tempfile.mkstemp(suffix='.cfg', prefix='ultrahdr_meta_')
    
    try:
        with os.fdopen(temp_fd, 'w') as f:
            # Calculate boost values with minimum threshold
            min_boost = max(1.0, 2 ** gain_min_log2)  # libultrahdr requires > 1.0
            max_boost = 2 ** gain_max_log2
            min_cap = max(1.0, 2 ** cap_min_log2)
            max_cap = 2 ** cap_max_log2
            
            # libultrahdr metadata format (command line arguments style)
            f.write(f"--maxContentBoost {max_boost:.6f} {max_boost:.6f} {max_boost:.6f}\n")
            f.write(f"--minContentBoost {min_boost:.6f} {min_boost:.6f} {min_boost:.6f}\n") 
            f.write(f"--gamma {gamma:.6f} {gamma:.6f} {gamma:.6f}\n")
            f.write(f"--offsetSdr {offset_sdr:.6f} {offset_sdr:.6f} {offset_sdr:.6f}\n")
            f.write(f"--offsetHdr {offset_hdr:.6f} {offset_hdr:.6f} {offset_hdr:.6f}\n")
            f.write(f"--hdrCapacityMin {min_cap:.6f}\n")
            f.write(f"--hdrCapacityMax {max_cap:.6f}\n")
            f.write(f"--useBaseColorSpace 1\n")
        
        return config_path
        
    except Exception as e:
        os.unlink(config_path)
        raise UltraHDRExportError(f"Failed to create metadata config: {e}")


def is_available() -> bool:
    """Check if libultrahdr is available for use"""
    try:
        _find_libultrahdr_binary()
        return True
    except UltraHDRExportError:
        return False

