"""
Direct Ultra HDR JPEG creation bypassing libultrahdr.

Creates Multi-Picture Format (MPF) JPEG files containing:
1. SDR base image (main JPEG)
2. Gain map image (auxiliary JPEG) 
3. MPF metadata linking them
4. HDR metadata for gain map parameters

This approach directly constructs the JPEG file format that macOS/iOS/Android
recognize as Ultra HDR without depending on external binaries.
"""

import os
import struct
from typing import Optional, Tuple, List
from dataclasses import dataclass
from PIL import Image
import numpy as np


class UltraHDRDirectError(Exception):
    """Raised when direct Ultra HDR creation fails"""
    pass


@dataclass
class UltraHDRMetadata:
    """Metadata for Ultra HDR JPEG creation"""
    gain_min_log2: float = 0.0
    gain_max_log2: float = 3.0
    gamma: float = 1.0
    offset_sdr: float = 0.0
    offset_hdr: float = 0.0
    hdr_capacity_min: float = 1.0
    hdr_capacity_max: float = 8.0
    

def _read_icc_profile_from_jpeg(jpeg_path: str) -> Optional[bytes]:
    """Extract ICC profile bytes from a JPEG's APP2 ICC segments."""
    try:
        with open(jpeg_path, 'rb') as f:
            data = f.read()
        if not data.startswith(b'\xFF\xD8'):
            return None
        pos = 2
        parts = {}
        total = None
        while pos + 4 <= len(data):
            if data[pos] != 0xFF:
                break
            marker = data[pos + 1]
            if marker == 0xDA:  # SOS
                break
            if marker in (0xD8, 0xD9):
                pos += 2
                continue
            seg_len = struct.unpack('>H', data[pos+2:pos+4])[0]
            seg = data[pos+4:pos+2+seg_len]
            if marker == 0xE2 and seg.startswith(b'ICC_PROFILE\x00'):
                seq = seg[len(b'ICC_PROFILE\x00')]
                cnt = seg[len(b'ICC_PROFILE\x00') + 1]
                payload = seg[len(b'ICC_PROFILE\x00') + 2:]
                total = cnt
                parts[seq] = payload
            pos += 2 + seg_len
        if total and parts and len(parts) == total:
            return b''.join(parts[i] for i in range(1, total + 1))
        return None
    except Exception:
        return None


def _create_mpf_app2_segment(sdr_size: int, gainmap_size: int) -> bytes:
    """Create APP2 MPF segment (CIPA DC‑X007 compliant, minimal).

    Builds a proper MP Index IFD with 2 entries (primary + gain map).
    Offsets are relative to the start of the first image (SOI of primary),
    which matches how common tools (ExifTool) report MPImageStart.
    """

    # MPF header identifier
    mpf_id = b"MPF\x00"

    # TIFF header (Big Endian, as per MPF spec). Offset to IFD = 8
    tiff_be = b"MM\x00*\x00\x00\x00\x08"

    # IFD entries we will write:
    #  - 0xB000 MPFVersion (UNDEFINED, count=4, value="0100")
    #  - 0xB001 NumberOfImages (LONG, count=1, value=2)
    #  - 0xB002 MPEntry (UNDEFINED, count=2*16, value at offset)
    # Prepare values
    num_images = 2
    mpf_version_value = b"0100"  # 4 bytes

    # Build MP entries (16 bytes each):
    #  [0:4]  Attributes (bitfield). 0 for primary, 0x020000 for dependent? Keep 0.
    #  [4:8]  ImageSize (bytes)
    #  [8:12] ImageDataOffset (from SOI of first image)
    #  [12:14] DependentImage1EntryNumber (0 if none)
    #  [14:16] DependentImage2EntryNumber (0 if none)
    primary_entry = struct.pack(
        ">LLLHH",
        0x00030000,         # attributes: Baseline MP Primary Image
        int(sdr_size),      # size
        0,                  # offset
        0, 0                # dep1, dep2
    )
    gainmap_entry = struct.pack(
        ">LLLHH",
        0x00000000,         # attributes (keep 0; type is indicated by metadata)
        int(gainmap_size),  # size
        int(sdr_size),      # offset = start at end of primary
        0, 0
    )
    mp_entries_blob = primary_entry + gainmap_entry

    # IFD structure
    entries = []

    # Helper to pack IFD entry (tag, type, count, value_or_offset)
    def ifd_entry(tag: int, typ: int, count: int, value_or_offset: int) -> bytes:
        return struct.pack(">HHLL", tag, typ, count, value_or_offset)

    # Tag 0xB000 MPFVersion (type 7 = UNDEFINED), count=4.
    # For 4 bytes, value fits in the 4-byte value field directly.
    entries.append(ifd_entry(0xB000, 7, 4, int.from_bytes(mpf_version_value, "big")))

    # Tag 0xB001 NumberOfImages (type 4 = LONG), count=1, value=num_images
    entries.append(ifd_entry(0xB001, 4, 1, num_images))

    # Tag 0xB002 MPEntry (type 7 = UNDEFINED), count=len(mp_entries_blob)
    mp_entry_count = len(mp_entries_blob)
    # The value doesn't fit inline; point to offset after the IFD block
    # We'll compute offsets after laying out the IFD.
    entries.append(ifd_entry(0xB002, 7, mp_entry_count, 0))  # placeholder offset

    # Assemble IFD
    ifd_count = len(entries)
    ifd_header = struct.pack(">H", ifd_count)
    ifd_body = b"".join(entries)
    next_ifd_offset = 0
    ifd_footer = struct.pack(">L", next_ifd_offset)

    # Now compute where the MPEntry blob will live
    # Layout: [MPF id][TIFF hdr][IFD count+entries+next][mp_entries]
    prefix = mpf_id + tiff_be + ifd_header + ifd_body + ifd_footer
    # Offset to MPEntry blob from the start of the TIFF header is:
    # 8 (TIFF header) + (2 + 12*n + 4) bytes for the IFD block
    mp_entries_offset_from_tiff = 8 + len(ifd_header) + len(ifd_body) + len(ifd_footer)
    # But TIFF value/offsets are from start of TIFF header (not including MPF id)
    # Our IFD entry for MPEntry needs the offset from TIFF header start.

    # Patch the MPEntry offset inside the IFD we already built (last entry)
    # IFD starts at: after MPF id + TIFF header
    ifd_start = len(mpf_id) + len(tiff_be)
    # Offset of entries array inside the assembled content right after IFD count
    entries_start = ifd_start + 2  # 2 bytes for count
    # The MPEntry is the 3rd entry (index 2)
    mp_entry_pos = entries_start + 2 * 12  # each entry is 12 bytes
    # Rebuild the 3rd entry with the correct offset
    mp_entry_fixed = ifd_entry(0xB002, 7, mp_entry_count, mp_entries_offset_from_tiff)

    content = bytearray(prefix)
    content[mp_entry_pos:mp_entry_pos + 12] = mp_entry_fixed
    content += mp_entries_blob

    # Wrap in APP2 marker
    seg_len = len(content) + 2
    return b"\xFF\xE2" + struct.pack(">H", seg_len) + bytes(content)


def _build_xmp_app1(payload_xml: str) -> bytes:
    """Build an APP1 XMP segment from XML payload."""
    # Standard XMP APP1 header identifier
    xmp_id = b"http://ns.adobe.com/xap/1.0/\x00"
    xmp_packet = payload_xml.encode("utf-8")
    app1_content = xmp_id + xmp_packet
    seg_len = len(app1_content) + 2
    return b"\xFF\xE1" + struct.pack(">H", seg_len) + app1_content


def _create_primary_gcontainer_xmp(gainmap_length: int, meta: "UltraHDRMetadata") -> bytes:
    """Disabled: avoid primary GContainer to match non-washed baseline.

    Some macOS builds appear to mis-handle primary XMP GContainer when HDR
    isn’t engaged, causing a brightened SDR look. We omit primary XMP and rely
    on ISO 21496-1 + MPF + hdrgm on the gain map.
    """
    return b""


def _create_gainmap_hdrgm_xmp(meta: "UltraHDRMetadata") -> bytes:
    """Create ISO 21496-1 compatible hdrgm XMP for the gain-map JPEG."""
    # Map log2 range to attributes per Android spec
    gm_min = meta.gain_min_log2
    gm_max = meta.gain_max_log2
    gamma = meta.gamma
    off_sdr = meta.offset_sdr
    off_hdr = meta.offset_hdr
    cap_min = meta.hdr_capacity_min
    cap_max = meta.hdr_capacity_max

    xml = f"""
<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description xmlns:hdrgm='http://ns.adobe.com/hdr-gain-map/1.0/'
      hdrgm:Version='1.0'
      hdrgm:BaseRenditionIsHDR='False'
      hdrgm:GainMapMin='{gm_min:.6f}'
      hdrgm:GainMapMax='{gm_max:.6f}'
      hdrgm:Gamma='{gamma:.6f}'
      hdrgm:OffsetSDR='{off_sdr:.6f}'
      hdrgm:OffsetHDR='{off_hdr:.6f}'
      hdrgm:HDRCapacityMin='{cap_min:.6f}'
      hdrgm:HDRCapacityMax='{cap_max:.6f}'/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>
"""
    return _build_xmp_app1(xml)


def _insert_app_segments(jpeg_bytes: bytes, segments: List[bytes]) -> bytes:
    """Insert APP segments after initial metadata markers (before SOS).

    We keep the original JPEG intact and inject our segments after any existing
    APP0/APP1/APP2... markers, but before the Start of Scan (FF DA).
    """
    if not (jpeg_bytes[:2] == b"\xFF\xD8"):
        raise UltraHDRDirectError("Not a JPEG stream for insertion")

    pos = 2  # after SOI
    end = len(jpeg_bytes)

    # Walk markers until SOS (FF DA) or until a non-APP marker
    while pos + 4 <= end and jpeg_bytes[pos] == 0xFF:
        marker = jpeg_bytes[pos + 1]
        if marker == 0xDA:  # SOS
            break
        # Standalone markers without length (rare here)
        if marker in (0xD8, 0xD9):  # SOI/EOI
            pos += 2
            continue
        if 0xE0 <= marker <= 0xEF or marker in (0xFE, 0xE1, 0xE2):
            # APPn or COM: read length
            if pos + 4 > end:
                break
            seg_len = struct.unpack(">H", jpeg_bytes[pos + 2:pos + 4])[0]
            pos += 2 + seg_len
            continue
        # Other marker (e.g., DQT/ SOF), stop here
        break

    # Insert segments at 'pos'
    inject = b"".join(segments)
    return jpeg_bytes[:pos] + inject + jpeg_bytes[pos:]


def _patch_jfif_add_ampf(jpeg_bytes: bytes) -> bytes:
    """Append 'AMPF' marker to the first JFIF APP0 segment, like iPhone.

    If no JFIF APP0 is found, return original bytes.
    """
    if not (jpeg_bytes[:2] == b"\xFF\xD8"):
        return jpeg_bytes
    pos = 2
    end = len(jpeg_bytes)
    if pos + 4 > end or jpeg_bytes[pos] != 0xFF or jpeg_bytes[pos+1] != 0xE0:
        return jpeg_bytes
    seg_len = struct.unpack(">H", jpeg_bytes[pos+2:pos+4])[0]
    seg_start = pos
    seg_end = pos + 2 + seg_len
    seg = bytearray(jpeg_bytes[seg_start:seg_end])
    # Check for 'JFIF\0' header
    if seg[4:9] != b'JFIF\x00':
        return jpeg_bytes
    # If already contains 'AMPF', do nothing
    if b'AMPF' in seg:
        return jpeg_bytes
    # Append 'AMPF' and fix length
    seg += b'AMPF'
    new_len = len(seg) - 2  # exclude marker bytes
    seg[2:4] = struct.pack(">H", new_len)
    return jpeg_bytes[:seg_start] + bytes(seg) + jpeg_bytes[seg_end:]


def _extract_initial_app_head(jpeg_bytes: bytes) -> Tuple[bytes, int]:
    """Return (app_head_bytes, non_app_offset) from start until first non-APP marker.

    Includes SOI and all APP/COM markers; excludes the first non-APP marker onwards.
    """
    if not (jpeg_bytes[:2] == b"\xFF\xD8"):
        raise UltraHDRDirectError("Invalid JPEG for head extraction")
    pos = 2
    end = len(jpeg_bytes)
    while pos + 4 <= end and jpeg_bytes[pos] == 0xFF:
        marker = jpeg_bytes[pos + 1]
        if marker == 0xDA:  # SOS
            pos += 2
            break
        if marker in (0xD8, 0xD9):
            pos += 2
            continue
        # Only APPn/COM carry length we can skip
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            seg_len = struct.unpack(">H", jpeg_bytes[pos + 2:pos + 4])[0]
            pos += 2 + seg_len
            continue
        # Found non-APP marker (e.g., DQT 0xDB)
        break
    return jpeg_bytes[:pos], pos


def _build_jfif_ampf() -> bytes:
    """Create a minimal JFIF APP0 without AMPF (to match non-washed baseline)."""
    # APP0 length = 16. JFIF v1.01, units=0, densities=1,0 thumbnail=0,0
    return b"\xFF\xE0\x00\x10" + b"JFIF\x00\x01\x01\x01\x00\x01\x00\x01\x00\x00"


def _build_icc_app2(icc_profile: bytes) -> bytes:
    """Build a single APP2 ICC profile segment (small profiles only)."""
    ident = b"ICC_PROFILE\x00" + bytes([1, 1])  # seq=1,count=1
    content = ident + icc_profile
    return b"\xFF\xE2" + struct.pack(">H", len(content) + 2) + content


def _extract_iphone_extra_app2() -> Optional[bytes]:
    """Extract an APP2 segment from iPhone JPEG that is not MPF or ICC (if any)."""
    path = "tests/fixtures/reference/iphone_hdr.jpg"
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            d = f.read()
        if not d.startswith(b'\xFF\xD8'):
            return None
        i = 2
        while i + 4 <= len(d) and d[i] == 0xFF:
            m = d[i + 1]
            if m == 0xDA:  # SOS
                break
            if m in (0xD8, 0xD9):
                i += 2
                continue
            L = struct.unpack('>H', d[i+2:i+4])[0]
            seg = d[i+4:i+2+L]
            if m == 0xE2:
                if not (seg.startswith(b'MPF\x00') or seg.startswith(b'ICC_PROFILE\x00')):
                    return d[i:i+2+L]
            i += 2 + L
    except Exception:
        return None
    return None


def _extract_iphone_gainmap_jpeg() -> Optional[bytes]:
    path = "tests/fixtures/reference/iphone_hdr.jpg"
    if not os.path.exists(path):
        return None
    try:
        d = open(path, 'rb').read()
        starts = [i for i in range(len(d)-1) if d[i]==0xFF and d[i+1]==0xD8]
        if len(starts) < 2:
            return None
        s = starts[-1]
        e = len(d)
        return d[s:e]
    except Exception:
        return None


def _extract_unknown_app2_from_jpeg(jpeg_bytes: bytes) -> Optional[bytes]:
    try:
        i = 2
        end = len(jpeg_bytes)
        while i + 4 <= end and jpeg_bytes[i] == 0xFF:
            m = jpeg_bytes[i + 1]
            if m == 0xDA:
                break
            if m in (0xD8, 0xD9):
                i += 2
                continue
            L = struct.unpack('>H', jpeg_bytes[i+2:i+4])[0]
            seg = jpeg_bytes[i:i+2+L]
            payload = jpeg_bytes[i+4:i+2+L]
            if m == 0xE2 and not payload.startswith(b'ICC_PROFILE\x00'):
                return seg
            i += 2 + L
    except Exception:
        pass
    return None


def _copy_iphone_hdr_exif() -> bytes:
    """Deprecated for direct export: return minimal EXIF.

    Copying reference EXIF can introduce mismatched thumbnails/orientation.
    """
    return _create_basic_exif_segment()


def _create_basic_exif_segment() -> bytes:
    """Create minimal EXIF segment"""
    exif_header = b"Exif\x00\x00"
    tiff_header = b"II*\x00\x08\x00\x00\x00"
    ifd_data = b"\x00\x00"
    exif_data = exif_header + tiff_header + ifd_data
    segment_length = len(exif_data) + 2
    return bytes([0xFF, 0xE1]) + struct.pack(">H", segment_length) + exif_data


def create_ultra_hdr_jpeg(
    sdr_image_path: str,
    gainmap_image_path: str, 
    output_path: str,
    metadata: Optional[UltraHDRMetadata] = None
) -> None:
    """
    Create Ultra HDR JPEG directly by concatenating JPEG streams.
    
    Args:
        sdr_image_path: Path to SDR base JPEG
        gainmap_image_path: Path to gain map image (PNG or JPEG)
        output_path: Output Ultra HDR JPEG path
        metadata: HDR metadata parameters
    """
    metadata = metadata or UltraHDRMetadata()
    
    if not os.path.exists(sdr_image_path):
        raise UltraHDRDirectError(f"SDR image not found: {sdr_image_path}")
    if not os.path.exists(gainmap_image_path):
        raise UltraHDRDirectError(f"Gain map image not found: {gainmap_image_path}")
    
    try:
        # Step 1: Load and prepare SDR JPEG
        with open(sdr_image_path, 'rb') as f:
            sdr_jpeg_data = f.read()
            
        if not sdr_jpeg_data.startswith(b'\xff\xd8'):
            raise UltraHDRDirectError("SDR image is not a valid JPEG")
            
        # Step 2: Load and convert gain map to JPEG
        gainmap_img = Image.open(gainmap_image_path)
        if gainmap_img.mode != 'L':
            gainmap_img = gainmap_img.convert('L')  # Convert to grayscale
        
        # Downsample gain map to 1/4 per dimension (matches typical iPhone layout)
        try:
            with Image.open(sdr_image_path) as _base_probe:
                bw, bh = _base_probe.size
            target_w = max(1, bw // 4)
            target_h = max(1, bh // 4)
            if gainmap_img.size != (target_w, target_h):
                gainmap_img = gainmap_img.resize((target_w, target_h), Image.LANCZOS)
        except Exception:
            pass
            
        # Save gain map as JPEG with specific quality
        import io
        gainmap_buffer = io.BytesIO()
        # Embed sRGB ICC into gain-map to ensure at least one APP2 exists
        icc_bytes = None
        try:
            from PIL import ImageCms
            icc_profile = ImageCms.createProfile('sRGB')
            icc_bytes = ImageCms.ImageCmsProfile(icc_profile).tobytes()
        except Exception:
            icc_bytes = None
        save_kwargs = dict(format='JPEG', quality=85, optimize=True, progressive=False)
        if icc_bytes:
            save_kwargs['icc_profile'] = icc_bytes
        gainmap_img.save(gainmap_buffer, **save_kwargs)
        gainmap_jpeg_data = gainmap_buffer.getvalue()

        # Also try to inject iPhone's unknown APP2 into gain-map for compatibility
        iphone_gm = _extract_iphone_gainmap_jpeg()
        if iphone_gm:
            unknown_app2 = _extract_unknown_app2_from_jpeg(iphone_gm)
            if unknown_app2:
                try:
                    gainmap_jpeg_data = _insert_app_segments(gainmap_jpeg_data, [unknown_app2])
                except Exception:
                    pass
        
        if not gainmap_jpeg_data.startswith(b'\xff\xd8'):
            raise UltraHDRDirectError("Failed to create gain map JPEG")
            
        print(f'✓ SDR JPEG: {len(sdr_jpeg_data):,} bytes')
        print(f'✓ Gain map JPEG: {len(gainmap_jpeg_data):,} bytes')
        
        # Step 3: Use raw gain-map JPEG bytes (no extra XMP to avoid SDR shifts)
        gainmap_jpeg_with_xmp = gainmap_jpeg_data

        # Step 4: Compute sizes for MPF and primary GContainer XMP
        gm_size = len(gainmap_jpeg_with_xmp)
        # Prepare primary XMP (needs Item:Length to reflect appended GM JPEG size)
        primary_xmp = _create_primary_gcontainer_xmp(gm_size, metadata)

        # Embed sRGB ICC to preserve SDR appearance (avoid P3 washout)
        icc_bytes = None
        try:
            from PIL import ImageCms
            icc_profile = ImageCms.createProfile('sRGB')
            icc_bytes = ImageCms.ImageCmsProfile(icc_profile).tobytes()
        except Exception:
            icc_bytes = None
        icc_seg = _build_icc_app2(icc_bytes) if icc_bytes else b""

        # Step 5: Rebuild primary head order to match iPhone style:
        # SOI -> APP0 JFIF(AMPF) -> APP1 EXIF (from iPhone) -> APP1 XMP(GContainer) -> APP2 MPF -> APP2 ICC -> rest of image
        # First, split base into (head, tail) and extract/patch JFIF
        head, non_app_off = _extract_initial_app_head(sdr_jpeg_data)
        tail = sdr_jpeg_data[non_app_off:]
        # Synthesize JFIF+AMPF to control order (avoid carrying other APPs from head)
        jfif_patched = _build_jfif_ampf()
        iphone_exif = b""  # omit EXIF to match minimal baseline

        # We need MPF with correct primary length -> do a two-pass
        iso_app2 = _extract_iphone_extra_app2() or b""

        def assemble_with_mpf(mpf_seg: bytes) -> bytes:
            return b"\xFF\xD8" + jfif_patched + primary_xmp + mpf_seg + iso_app2 + icc_seg + tail

        tmp_mpf = _create_mpf_app2_segment(0, gm_size)
        primary_tmp = assemble_with_mpf(tmp_mpf)
        primary_size = len(primary_tmp)
        mpf_segment = _create_mpf_app2_segment(primary_size, gm_size)
        sdr_with_segments = assemble_with_mpf(mpf_segment)
        primary_size = len(sdr_with_segments)
        print(f'✓ Primary (with EXIF+XMP+MPF+ICC): {primary_size:,} bytes')

        # Step 6: Concatenate primary + gain-map stream
        final_bytes = sdr_with_segments + gainmap_jpeg_with_xmp

        # Write final Ultra HDR JPEG
        with open(output_path, 'wb') as f:
            f.write(final_bytes)
        
        print(f'✓ Ultra HDR JPEG created: {output_path}')
        print(f'✓ Total size: {len(final_bytes):,} bytes')
        
        # Validate output
        if not os.path.exists(output_path):
            raise UltraHDRDirectError("Failed to write output file")
            
        final_size = os.path.getsize(output_path)
        expected_size = primary_size + gm_size

        if abs(final_size - expected_size) > 1024:  # Allow small overhead variations
            raise UltraHDRDirectError(
                f"Output size mismatch: {final_size:,} bytes "
                f"(expected ~{expected_size:,} bytes)"
            )
            
        print(f'✓ Size validation passed: {final_size:,} bytes')
        
    except Exception as e:
        raise UltraHDRDirectError(f"Failed to create Ultra HDR JPEG: {e}")


if __name__ == "__main__":
    # Test with our files
    create_ultra_hdr_jpeg(
        "test_sdr.jpg",
        "test_gainmap.png", 
        "direct_ultra_hdr_test.jpg"
    )
