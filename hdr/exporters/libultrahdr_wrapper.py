"""
Ultra HDR (JPEG_R) exporter wrapper.

This module will wrap Google libultrahdr (API-4) to combine:
  - SDR base JPEG bytes
  - 8-bit grayscale gain-map JPEG/PNG bytes
  - ISO hdrgm parameters (GainMapMin/Max log2, Gamma, OffsetSDR/HDR, HDRCapacityMin/Max)
into a single standards-compliant Ultra HDR JPEG file.

Research/Integration required:
  - Build libultrahdr (CMake) for the host platform.
  - Expose API-4 to Python via ctypes/cffi or a tiny CLI helper.
  - Map parameters exactly to hdrgm XMP fields.

This is a placeholder to make the package structure ready for the next phase.
"""

from typing import Optional


class UltraHDRExportError(RuntimeError):
    pass


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
) -> None:
    """
    Export Ultra HDR JPEG (JPEG_R) using libultrahdr (to be implemented).

    For now, this function raises UltraHDRExportError to indicate the wrapper
    is not yet wired. The pipeline will use the Apple ImageIO path on macOS as
    a secondary exporter until this is implemented.
    """
    raise UltraHDRExportError("libultrahdr wrapper not yet implemented")

