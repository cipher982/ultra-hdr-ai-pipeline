# 🏗️ **Pipeline Architecture**

## **Core Philosophy: Gain-Map-First**

This pipeline is built around **modern AI models that directly predict gain maps** from SDR input. No HDR intermediate generation - straight from SDR to gain map.

## **📁 File Structure**

```
hdr/
├── models.py              # AI model plugins (any framework)
├── gainmap_pipeline.py    # Main pipeline orchestration  
├── cli.py                 # Clean command-line interface
├── exporters/
│   └── libultrahdr_wrapper.py  # Cross-platform JPEG_R export
└── swift/
    ├── GainMapExporter.swift    # Apple HEIC export
    └── ...
```

## **🔄 Pipeline Flow**

```
SDR Image → AI Model → Gain Map → Ultra HDR Container
    ↓           ↓          ↓            ↓
  Load &     Direct     8-bit      JPEG_R/HEIC
 Process   Prediction  Grayscale   with metadata
```

## **🧩 Plugin Architecture**

### **AI Models** (`models.py`)
- **`ModelPlugin`**: Generic loader for any AI framework
- **`SyntheticGainMapModel`**: Testing/fallback without AI
- **Easy extensibility**: Just implement `_infer_*()` methods

### **Exporters** (`exporters/`)
- **`libultrahdr_wrapper.py`**: Cross-platform JPEG_R (Android 14+)
- **Swift exporters**: Apple-native HEIC/JPEG export
- **Auto-selection**: Best format for target platform

## **🎯 Research-Friendly Design**

### **Any Model, Any Framework**
```bash
# Just point to your model file:
hdr-process --img photo.jpg --model your_research.onnx --out result.jpg
hdr-process --img photo.jpg --model experiment.pt --out result.jpg
hdr-process --img photo.jpg --model custom.h5 --out result.jpg
```

### **Fail-Fast Development**
- ✅ **Explicit errors** (no silent failures)
- ✅ **Output validation** (ensures real Ultra HDR)
- ✅ **Dependency checking** (`--check-deps`)
- ✅ **Verbose logging** (`--verbose`)

## **🔧 Key Design Decisions**

1. **Gain maps are first-class citizens** - not derived from HDR
2. **AI model abstraction** - swap models without touching pipeline
3. **Cross-platform export** - libultrahdr for standards compliance  
4. **Apple-native support** - Swift exporters for optimal macOS/iOS
5. **Research-oriented** - easy to experiment with new approaches

This architecture is **optimized for AI research iteration** - try any model, get immediate results, with proper Ultra HDR output that works across platforms.