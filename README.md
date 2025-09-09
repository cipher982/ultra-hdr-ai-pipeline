# Ultra HDR Gain-Map Pipeline

<p align="center">
  <img src="logo4_final.png" alt="HDR Gain-Map Logo" width="512" height="256">
</p>

**Restore HDR "pop" to your AI-edited photos using state-of-the-art gain map generation.**

## The Problem

Ever taken a beautiful HDR photo on your iPhone, edited it through an AI image editor, and noticed it lost that gorgeous HDR "pop" when viewed on your device? That's because AI editors strip the **gain map** - the secret sauce that tells your screen how to display HDR content.

This pipeline puts it back.

## What Are HDR Gain Maps?

HDR gain maps are like "brightness instructions" stored alongside regular photos. When your iPhone/Mac/Android device encounters a photo with a gain map, it knows to brighten certain areas beyond normal display limits, creating that stunning HDR effect you see in Camera app.

**The Magic**: Your photo looks normal on regular screens, but "pops" with extra brightness and detail on HDR displays.

## Use Cases

- 📱 **AI Photo Editing**: Restore HDR after running photos through ChatGPT, Midjourney, or other AI tools
- 📸 **Social Media**: Add HDR enhancement before sharing (where supported)
- 🎨 **Content Creation**: Generate HDR variants of existing photos
- 🔧 **Workflow Integration**: Batch process edited photos to maintain HDR compatibility

## How It Works

```
Your Edited Photo → AI Model (GMNet) → Gain Map → HDR Container → HDR Display ✨
```

The pipeline uses **GMNet** - cutting-edge research from ICLR 2025 - to predict realistic gain maps that match what your camera would have captured.

## Quick Start

### 🌐 Web Interface (Recommended)

1. **Setup & Launch**
   ```bash
   make setup && make web
   ```
   
2. **Open Browser**
   Visit `http://localhost:8001` and drag & drop your photos!

### 💻 Command Line Interface  

```bash
# Use GMNet AI model (recommended)
hdr-process --img your_photo.jpg --out hdr_result.jpg --model gmnet

# Or use synthetic generation (fallback)  
hdr-process --img your_photo.jpg --out hdr_result.jpg --model synthetic
```

3. **View Results**
   Open `hdr_result.jpg` in **macOS Preview** or transfer to **iOS Photos** to see the HDR enhancement!

## AI Models

### GMNet (Recommended) 🤖
- **Source**: ICLR 2025 research "Learning Gain Map for Inverse Tone Mapping"
- **Quality**: Professional-grade results matching iPhone HDR generation
- **Speed**: ~2-3 seconds per image on modern hardware
- **Use Case**: Best for realistic photo enhancement

### Synthetic (Fallback) ⚡
- **Source**: Built-in luminance-based algorithm
- **Quality**: Good for basic enhancement
- **Speed**: Nearly instant
- **Use Case**: Quick processing or when GMNet unavailable

### Custom Models 🔧
- **ONNX, PyTorch, TensorFlow**: Drop in your own trained models
- **Plugin System**: Extensible architecture for research
- **Training**: Fine-tune on your specific image types

## Platform Compatibility

| Platform | Format | Status | Notes |
|----------|--------|---------|-------|
| **macOS Preview** | Ultra HDR JPEG | ✅ Perfect | Native HDR rendering |
| **iOS Photos** | Ultra HDR JPEG | ✅ Perfect | Native HDR rendering |
| **Android 14+** | Ultra HDR JPEG | ✅ Perfect | Cross-platform standard |
| **Social Media** | Ultra HDR JPEG | 🔄 Varies | Platform-dependent support |

## Examples

### Before: Regular SDR Photo
*Standard photo after AI editing - looks flat*

### After: HDR Gain Map Applied  
*Same photo with restored HDR - highlights pop, shadows detailed*

*(Visual examples would be shown here in actual display)*

## Advanced Usage

### Batch Processing
```bash
for img in photos/*.jpg; do
  hdr-process --img "$img" --out "hdr_$(basename "$img" .jpg).jpg" --model gmnet
done
```

### Debug Mode
```bash
hdr-process --img photo.jpg --out result.jpg --model gmnet --debug
# Generates gain_map.png and metadata.json for inspection
```

### Development Mode
```bash
# Web service with auto-reload for development
make dev

# Run validation tests
make test
```

## Requirements

- **Python 3.11+**: Managed automatically via `uv`
- **8GB+ RAM**: For GMNet AI model inference
- **HDR Display**: To see results (MacBook Pro, iPhone, etc.)
- **Any OS**: Ultra HDR JPEG works cross-platform

## Research & Development

This project integrates cutting-edge research:
- **GMNet (2025)**: State-of-the-art gain map prediction
- **Ultra HDR Standard**: Google/Android HDR format
- **ISO 21496**: Cross-platform HDR metadata standard

Built for researchers, photographers, and developers working with HDR imaging pipelines.

## Contributing

Found a bug? Have a feature request? Want to integrate a new AI model?

See `CLAUDE.md` for developer setup and technical details.

## License

MIT License - Use freely for personal and commercial projects.

---

*Restore the HDR magic to your AI-edited photos* ✨