# HDR Pipeline Developer Notes

**Context for AI assistants and developers working on this codebase.**

## 🏗️ Project Architecture

### Core Philosophy
**Gain-Map-First Architecture**: Modern approach that generates gain maps directly from SDR input, bypassing intermediate HDR generation. This matches how iPhone and modern HDR systems actually work.

**Evolution**: Replaced legacy `pipeline.py` (SDR→HDR→gain map) with `gainmap_pipeline.py` (SDR→gain map direct).

### File Structure
```
hdr/
├── models.py           # AI model abstraction + GMNet integration
├── gainmap_pipeline.py # Main pipeline logic 
├── cli.py             # Command-line interface
├── exporters/
│   └── libultrahdr_wrapper.py  # Cross-platform JPEG_R export
└── swift/
    └── GainMapExporter.swift   # macOS HEIC export

GMNet/                 # ICLR 2025 research repo (gitignored but preserved)
├── checkpoints/
│   ├── G_realworld.pth    # 7.4MB - best for photos
│   └── G_synthetic.pth    # 7.4MB - fallback
└── codes/             # PyTorch inference code

bin/gainmap_exporter   # Compiled Swift binary (84KB)
```

## 🧠 GMNet Integration Details

### Model Loading Quirks
- **Dual Input Required**: GMNet needs `[full_image, thumbnail_256x256]` 
- **Output Format**: Returns tuple `(main_gain_map, auxiliary_data)` - use first element
- **Device Handling**: Force CPU mode - config needs `gpu_ids: []` and `dist: False`
- **Path Setup**: Dynamically adds `GMNet/codes/` to sys.path for imports
- **Lazy Loading**: Model only loads on first `.predict()` call (saves startup time)

### Model Factory Pattern
```python
create_model("gmnet")           # Uses default G_realworld.pth
create_model("synthetic")       # Built-in luminance-based
create_model("auto", model_path="custom.onnx")  # Generic plugin
```

## 🛠️ Environment Setup

### Python Environment
- **Package Manager**: `uv` (not pip) - handles venv automatically
- **Python Version**: 3.11+ (specified in pyproject.toml)
- **Key Dependencies**: PyTorch 2.8.0, OpenCV 4.11.0, nvidia-ml-py

### Installation Commands
```bash
make setup              # Installs everything via uv sync
.venv/bin/hdr-process   # Use venv binary directly
# or
hdr-process            # If you have the venv activated
```

### GMNet Dependencies
Auto-installed by `uv add`: torch, torchvision, opencv-python, pyyaml, nvidia-ml-py

## 🐛 Known Issues & Debugging

### HEIC Export Problem (Critical)
**Symptom**: Generated HEIC files are 131KB (vs iPhone reference 1.7MB) and missing HDR metadata
**Root Cause**: Swift `CGImageDestinationAddAuxiliaryDataInfo` not properly embedding gain map data
**Debug**: Files show no `HDR Headroom`, `HDR Gain`, `Float Max/Min Value` in exiftool output
**Status**: Swift exporter runs without error but auxiliary data not persisting

### JPEG_R Export Problem  
**Symptom**: `libultrahdr` command fails with "received bad value for min boost 0.000000"
**Root Cause**: Metadata format issues - libultrahdr expects command-line args format
**Attempted Fix**: Updated to `--maxContentBoost` format but still failing
**Status**: Alternative to HEIC export, lower priority

### Pipeline Path Handling Bug (Fixed)
**Issue**: `os.makedirs('')` failed when output path had no directory component
**Fix**: Added `if base_dir:` check before mkdir
**Commit**: Included in pipeline refactoring

## 📊 Reference Image Analysis

### iPhone HDR Metadata (Ground Truth)
```
# 01_original_iphone_hdr.jpeg (Ultra HDR JPEG - 248KB)
HDR Headroom: 1.00999999
HDR Gain: 0.004547884226

# 5D2AF0A6-9912-4CFA-B4EF-1A05489FE557.heic (1.7MB)  
HDR Gain Map Version: 0.2.0.0
HDR Gain Map Headroom: 6.415851  (≈2.68 log2 stops)
Float Max Value: 3.351562
Float Min Value: 0.311523
HD Gain Map Info: (Binary data 254 bytes) <- The actual gain map!
```

### Target Specifications
- **Dynamic Range**: 2.68 log2 stops (6.4x linear boost)
- **Container**: HEIC with XMP-HDRGainMap metadata + binary auxiliary data
- **Compatibility**: Must render HDR in macOS Preview/iOS Photos

## 🧪 Testing & Validation

### Quick Tests
```bash
# Test GMNet model directly
.venv/bin/python -c "from hdr.models import GMNetModel; m = GMNetModel(); print(f'Available: {m.is_available()}')"

# Test gain map generation only  
.venv/bin/python -c "from hdr.models import GMNetModel; import cv2; import numpy as np; m = GMNetModel(); img = cv2.cvtColor(cv2.imread('images/02_edited_sdr.jpeg'), cv2.COLOR_BGR2RGB).astype(np.float32)/255; p = m.predict(img); print(f'Gain map: {p.gain_map.shape}, range: [{p.gain_min_log2:.3f}, {p.gain_max_log2:.3f}]')"

# Visual validation (opens in Preview)
open generated_file.heic images/5D2AF0A6-9912-4CFA-B4EF-1A05489FE557.heic
```

### Metadata Comparison
```bash
# Compare HDR metadata between files
exiftool image1.heic | grep -E "(HDR|Float|Gain)"
exiftool image2.jpg | grep -E "(HDR|Float|Gain)"
```

### File Size Validation
Working HDR files should be significantly larger due to embedded gain map:
- iPhone HEIC: 1.7MB ✅
- Generated HEIC: 131KB ❌ (missing gain map data)

## 🔧 Development Commands

### GMNet Testing
```bash
# Test in GMNet directory
cd GMNet/codes
PYTHONPATH=. ../../.venv/bin/python -c "import torch; print('GMNet env ready')"

# Test model loading
cd GMNet/codes  
PYTHONPATH=. ../../.venv/bin/python test.py -opt options/config/test_syn.yml
```

### Swift Exporter Testing
```bash
# Test Swift binary directly
bin/gainmap_exporter sdr.jpg gainmap.png out.heic 0.0 3.0 1.0 0.0 0.0 0.0 3.0

# Rebuild Swift exporter if needed
cd hdr/swift
swiftc -O GainMapExporter.swift -o ../../bin/gainmap_exporter
```

### libultrahdr Testing  
```bash
# Manual libultrahdr test (when fixed)
/Users/davidrose/git/hdr/tools/libultrahdr/build/ultrahdr_app -m 0 -i sdr.jpg -g gainmap.jpg -o out.jpg -q 95
```

## 🎯 Next Steps (Technical Priorities)

### Phase 1: Fix Container Export (Critical)
1. **HEIC Export Debug**:
   - Investigate why `CGImageDestinationAddAuxiliaryDataInfo` isn't persisting data
   - Test different auxiliary data types (HDR vs ISO gain map)
   - Compare auxiliary data dictionary structure against working samples
   - Add detailed Swift logging to trace where data is lost

2. **JPEG_R Export Fix**:
   - Debug libultrahdr metadata format requirements  
   - Test with example configurations from libultrahdr repo
   - Fix minimum boost value handling

### Phase 2: Production Ready
1. **Comprehensive Testing**: Unit tests for each component
2. **Error Handling**: Better user messages, graceful fallbacks  
3. **Performance**: Memory optimization, processing speed
4. **Documentation**: API docs, troubleshooting guide

### Phase 3: Advanced Features
1. **Reference Mode**: Extract gain maps from existing HDR photos
2. **Custom Models**: Training pipeline for specialized use cases
3. **Batch Processing**: GUI or advanced CLI for multiple files
4. **Format Support**: Additional container formats, metadata standards

## 🔍 Debugging Tips

### Silent Failure Prevention
- All exceptions should bubble up with clear messages
- File size validation catches empty/wrong outputs
- Metadata validation ensures proper HDR parameters
- `--debug` mode saves intermediate files for inspection

### Environment Issues
- Always use `.venv/bin/python` or `uv run` - never system python
- GMNet imports require `PYTHONPATH=.` when testing directly
- Swift compiler needs Xcode command line tools
- libultrahdr needs cmake for building

### Visual Validation
The ultimate test is visual - does the image show HDR "pop" in macOS Preview?
- Working: Bright highlights, enhanced contrast, HDR effect visible
- Broken: Looks identical to regular SDR photo
- Test side-by-side with iPhone reference images

## 📝 Code Patterns

### Model Implementation
```python
class YourModel(GainMapModel):
    def predict(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        # sdr_rgb: float32 [0,1] RGB array (H,W,3)
        # Return: 8-bit grayscale gain map + metadata
        return GainMapPrediction(
            gain_map=gain_map_uint8,      # (H,W) uint8
            gain_min_log2=min_stops,      # float
            gain_max_log2=max_stops,      # float  
            confidence=confidence         # optional float [0,1]
        )
```

### Error Handling
```python
# Always fail fast - no silent failures!
if not os.path.exists(required_file):
    raise ModelError(f"Required file missing: {required_file}")
    
# Validate outputs immediately  
if output.size == 0:
    raise ModelError("Model returned empty output")
```

## 🎨 User Experience Notes

### Target Workflow
1. User takes HDR photo on iPhone
2. Edits through AI tool (ChatGPT, Midjourney, etc.) 
3. AI strips gain map → photo loses HDR
4. User runs: `hdr-process --img edited.jpg --out restored.heic --model gmnet`
5. Result displays with HDR "pop" in Preview/Photos

### Success Criteria
Generated HDR image should be **visually indistinguishable** from iPhone original when viewed on HDR display.

---

**Key Insight**: The AI model (GMNet) works perfectly. The challenge is purely in HDR container export - getting the gain map data properly embedded so macOS can render it as HDR.

**Current Status**: 90% complete - just need to fix the final export step to achieve the user's goal of restoring HDR to AI-edited photos.