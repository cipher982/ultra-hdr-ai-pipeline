"""HDR metadata extraction and validation utilities

Tools for deep HDR metadata analysis and compliance validation against iPhone reference.
"""

import os
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import exifread
from PIL import Image
from PIL.ExifTags import TAGS


@dataclass
class HDRMetadata:
    """Structured HDR metadata from Ultra HDR files"""
    # Basic file info
    file_path: str
    file_size: int
    format: str
    
    # HDR-specific EXIF/XMP fields
    hdr_headroom: Optional[float] = None
    hdr_gain: Optional[float] = None
    gain_map_version: Optional[str] = None
    gain_map_headroom: Optional[float] = None
    float_max_value: Optional[float] = None
    float_min_value: Optional[float] = None
    
    # MPF (Multi-Picture Format) info
    has_mpf: bool = False
    jpeg_count: int = 0
    mpf_version: Optional[str] = None
    
    # Gain map binary info
    has_gain_map_data: bool = False
    gain_map_size: int = 0
    
    # Color profile info
    color_space: Optional[str] = None
    icc_profile_present: bool = False


def extract_hdr_metadata(file_path: str) -> HDRMetadata:
    """Extract comprehensive HDR metadata from Ultra HDR JPEG file"""
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    metadata = HDRMetadata(
        file_path=str(path),
        file_size=path.stat().st_size,
        format="Unknown"
    )
    
    try:
        # Basic PIL metadata
        with Image.open(file_path) as img:
            metadata.format = img.format
            
            # Extract EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                exif_dict = img._getexif()
                metadata = _extract_exif_hdr_fields(exif_dict, metadata)
            
            # Check for ICC profile
            if 'icc_profile' in img.info:
                metadata.icc_profile_present = True
                metadata.color_space = _identify_color_space(img.info['icc_profile'])
    
        # Extract detailed EXIF with exifread
        with open(file_path, 'rb') as f:
            exif_tags = exifread.process_file(f, details=True)
            metadata = _extract_exifread_hdr_fields(exif_tags, metadata)
        
        # Analyze MPF structure
        metadata = _analyze_mpf_structure(file_path, metadata)
        
        # Look for HDR gain map binary data
        metadata = _detect_gain_map_data(file_path, metadata)
            
    except Exception as e:
        # Don't fail completely if metadata extraction has issues
        pass
    
    return metadata


def _extract_exif_hdr_fields(exif_dict: Dict, metadata: HDRMetadata) -> HDRMetadata:
    """Extract HDR fields from PIL EXIF dictionary"""
    
    # Look for HDR-related EXIF tags
    for tag_id, value in exif_dict.items():
        tag_name = TAGS.get(tag_id, tag_id)
        
        # Check for HDR headroom in various possible locations
        if isinstance(tag_name, str) and 'hdr' in tag_name.lower():
            if 'headroom' in tag_name.lower():
                try:
                    metadata.hdr_headroom = float(value)
                except (ValueError, TypeError):
                    pass
            elif 'gain' in tag_name.lower():
                try:
                    metadata.hdr_gain = float(value)
                except (ValueError, TypeError):
                    pass
    
    return metadata


def _extract_exifread_hdr_fields(exif_tags: Dict, metadata: HDRMetadata) -> HDRMetadata:
    """Extract HDR fields from exifread tags"""
    
    # Look for XMP and other HDR-related tags
    for tag_name, tag_value in exif_tags.items():
        tag_str = str(tag_value).strip()
        
        # HDR Headroom
        if 'HDR Headroom' in tag_name or 'hdr_headroom' in tag_name.lower():
            try:
                metadata.hdr_headroom = float(tag_str)
            except (ValueError, TypeError):
                pass
        
        # HDR Gain
        elif 'HDR Gain' in tag_name or 'hdr_gain' in tag_name.lower():
            try:
                metadata.hdr_gain = float(tag_str)
            except (ValueError, TypeError):
                pass
        
        # Gain Map Version
        elif 'gain map version' in tag_name.lower() or 'gainmapversion' in tag_name.lower():
            metadata.gain_map_version = tag_str
        
        # Gain Map Headroom
        elif 'gain map headroom' in tag_name.lower() or 'gainmapheadroom' in tag_name.lower():
            try:
                metadata.gain_map_headroom = float(tag_str)
            except (ValueError, TypeError):
                pass
        
        # Float values
        elif 'float max value' in tag_name.lower():
            try:
                metadata.float_max_value = float(tag_str)
            except (ValueError, TypeError):
                pass
        
        elif 'float min value' in tag_name.lower():
            try:
                metadata.float_min_value = float(tag_str)
            except (ValueError, TypeError):
                pass
    
    return metadata


def _analyze_mpf_structure(file_path: str, metadata: HDRMetadata) -> HDRMetadata:
    """Analyze Multi-Picture Format structure for Ultra HDR compliance"""
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read(8192)  # Read first 8KB to find MPF markers
        
        # Look for MPF (Multi-Picture Format) markers  
        mpf_marker = b'\\xff\\xe2'  # MPF App2 marker (lowercase)
        mpf_marker_alt = b'\\xFF\\xE2'  # Alternative
        if mpf_marker in data or mpf_marker_alt in data:
            metadata.has_mpf = True
            
            # Try to count JPEG images in the file
            jpeg_markers = data.count(b'\\xFF\\xD8')  # JPEG SOI marker
            metadata.jpeg_count = jpeg_markers
            
            # Look for MPF version info
            mpf_pos = data.find(mpf_marker)
            if mpf_pos > 0 and mpf_pos + 10 < len(data):
                # Try to extract MPF version (basic heuristic)
                version_data = data[mpf_pos+4:mpf_pos+10]
                if b'MPF' in version_data:
                    metadata.mpf_version = "Present"
    
    except Exception:
        pass
    
    return metadata


def _detect_gain_map_data(file_path: str, metadata: HDRMetadata) -> HDRMetadata:
    """Detect embedded gain map binary data in Ultra HDR file"""
    
    try:
        with open(file_path, 'rb') as f:
            # Read larger chunk to find gain map
            data = f.read(1024 * 1024)  # 1MB should be enough
        
        # Look for patterns that suggest gain map data
        # Ultra HDR files typically have multiple JPEG streams
        jpeg_starts = []
        pos = 0
        while True:
            pos = data.find(b'\\xFF\\xD8', pos)
            if pos == -1:
                break
            jpeg_starts.append(pos)
            pos += 1
        
        if len(jpeg_starts) >= 2:
            metadata.has_gain_map_data = True
            
            # Estimate gain map size (rough heuristic)
            if len(jpeg_starts) >= 2:
                # Size between second JPEG start and end
                second_jpeg_start = jpeg_starts[1]
                # Find end of file or next major marker
                end_pos = len(data)
                for i in range(second_jpeg_start + 100, len(data) - 1):
                    if data[i:i+2] == b'\\xFF\\xD9':  # JPEG EOI
                        end_pos = i + 2
                        break
                
                metadata.gain_map_size = end_pos - second_jpeg_start
    
    except Exception:
        pass
    
    return metadata


def _identify_color_space(icc_data: bytes) -> str:
    """Identify color space from ICC profile data"""
    
    try:
        # Look for common color space signatures in ICC profile
        if b'sRGB' in icc_data[:200]:
            return "sRGB"
        elif b'RGB ' in icc_data[:200]:
            return "RGB"
        elif b'GRAY' in icc_data[:200]:
            return "Grayscale"
        else:
            return "Unknown"
    except Exception:
        return "Unknown"


def compare_hdr_metadata(metadata1: HDRMetadata, metadata2: HDRMetadata, tolerance: float = 0.1) -> Dict[str, bool]:
    """Compare HDR metadata between two files with tolerance for floating point values"""
    
    def values_close(v1, v2, tol=tolerance):
        """Check if two values are close within tolerance"""
        if v1 is None or v2 is None:
            return v1 == v2
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(v1 - v2) <= abs(v1) * tol
        return v1 == v2
    
    comparison = {
        'hdr_headroom_match': values_close(metadata1.hdr_headroom, metadata2.hdr_headroom),
        'hdr_gain_match': values_close(metadata1.hdr_gain, metadata2.hdr_gain),
        'gain_map_headroom_match': values_close(metadata1.gain_map_headroom, metadata2.gain_map_headroom),
        'float_values_reasonable': (
            metadata1.float_max_value is not None and 
            metadata1.float_min_value is not None and
            metadata1.float_max_value > metadata1.float_min_value
        ),
        'mpf_structure_match': (
            metadata1.has_mpf == metadata2.has_mpf and
            metadata1.jpeg_count >= 2 and metadata2.jpeg_count >= 2
        ),
        'gain_map_data_match': (
            metadata1.has_gain_map_data == metadata2.has_gain_map_data and
            metadata1.gain_map_size > 1000  # Should be substantial
        ),
        'file_size_reasonable': (
            50000 < metadata1.file_size < 10000000  # 50KB to 10MB range
        )
    }
    
    return comparison


def validate_hdr_compliance_against_reference(test_file: str, reference_file: str) -> Tuple[bool, List[str]]:
    """Validate HDR compliance by comparing against iPhone reference structure"""
    
    try:
        test_metadata = extract_hdr_metadata(test_file)
        ref_metadata = extract_hdr_metadata(reference_file)
        
        comparison = compare_hdr_metadata(test_metadata, ref_metadata)
        
        errors = []
        
        # Check each compliance aspect
        if not comparison['mpf_structure_match']:
            errors.append(f"MPF structure mismatch - Test: MPF={test_metadata.has_mpf}, JPEGs={test_metadata.jpeg_count}")
        
        if not comparison['gain_map_data_match']:
            errors.append(f"Gain map data mismatch - Test: present={test_metadata.has_gain_map_data}, size={test_metadata.gain_map_size}")
        
        if not comparison['file_size_reasonable']:
            errors.append(f"Unreasonable file size: {test_metadata.file_size} bytes")
        
        # HDR metadata presence (more lenient than exact match)
        if test_metadata.hdr_headroom is None and ref_metadata.hdr_headroom is not None:
            errors.append("Missing HDR headroom metadata")
        
        if test_metadata.gain_map_version is None and ref_metadata.gain_map_version is not None:
            errors.append("Missing gain map version")
        
        is_compliant = len(errors) == 0
        
        return is_compliant, errors
        
    except Exception as e:
        return False, [f"Metadata extraction failed: {e}"]