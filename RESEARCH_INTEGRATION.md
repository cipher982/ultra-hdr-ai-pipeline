# 🧪 **AI Model Integration Guide**

## **Quick Start for Researchers**

This pipeline is designed for **maximum flexibility** when experimenting with gain map AI models. No matter what paper, architecture, or framework you use - just drop it in!

### **🚀 Super Simple Integration**

```python
# In hdr/models.py, find the ModelPlugin class and update ONE method:

def _infer_gmnet(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
    """Replace this with your actual model inference"""
    
    # 1. Load your model (any framework)
    if self._model is None:
        # Your model loading code here:
        # self._model = load_your_model(self.model_path)
        pass
    
    # 2. Run inference  
    # gain_map_output = self._model.predict(sdr_rgb)
    
    # 3. Return the results
    return GainMapPrediction(
        gain_map=your_gain_map_array,      # (H, W) uint8 grayscale
        gain_min_log2=your_min_stops,      # float: minimum boost in log2 stops  
        gain_max_log2=your_max_stops,      # float: maximum boost in log2 stops
        confidence=your_confidence_score   # float 0-1 (optional)
    )
```

### **🎯 Usage Patterns**

```bash
# Test any model instantly:
hdr-process --img photo.jpg --model your_model.onnx --out result.jpg

# Try different models:
hdr-process --img photo.jpg --model gmnet_v2.pt --out result.jpg
hdr-process --img photo.jpg --model custom_model.h5 --out result.jpg  
hdr-process --img photo.jpg --model experiment.tflite --out result.jpg

# Development/testing:
hdr-process --img photo.jpg --model synthetic --out result.jpg --verbose
```

### **🔧 Supported Model Types**

The pipeline **auto-detects** model type from filename/extension:

| Extension | Framework | Auto-detected | Method to Override |
|-----------|-----------|---------------|-------------------|
| `.onnx` | ONNX Runtime | ✅ | `_infer_onnx()` |
| `.pt/.pth` | PyTorch | ✅ | `_infer_pytorch()` |  
| `.h5` | TensorFlow/Keras | ✅ | `_infer_tensorflow()` |
| `.tflite` | TF Lite | ✅ | `_infer_tflite()` |
| `*gmnet*` | Custom GMNet | ✅ | `_infer_gmnet()` |
| Other | Custom | ⚠️ | `_infer_generic()` |

### **📊 Model Output Requirements**

Your AI model just needs to return:

```python  
@dataclass
class GainMapPrediction:
    gain_map: np.ndarray     # (H, W) uint8 grayscale, 0=no boost, 255=max boost
    gain_min_log2: float     # Minimum boost in log2 stops (usually 0.0)
    gain_max_log2: float     # Maximum boost in log2 stops (e.g., 3.0 = 8x brighter)
    confidence: float        # Your model's confidence 0-1 (optional)
```

### **🧪 Research Workflow**

1. **Start with synthetic model** to test the pipeline
2. **Drop your model file** anywhere and point to it
3. **Override the appropriate `_infer_*()` method** 
4. **Test immediately** - pipeline handles all the container/metadata complexity

### **💡 Examples for Different Research Scenarios**

#### **Scenario 1: Testing GMNet Paper**
```python
# Just update _infer_gmnet() with the paper's inference code
def _infer_gmnet(self, sdr_rgb):
    # Load GMNet from paper
    # Run their inference 
    # Return GainMapPrediction
```

#### **Scenario 2: Your Own Custom Model**
```python  
# Use the generic plugin
def _infer_generic(self, sdr_rgb):
    # Your custom model loading/inference
    # Return GainMapPrediction
```

#### **Scenario 3: Trying Multiple Papers**
```bash
# Test different approaches instantly:
hdr-process --img test.jpg --model paper1_gmnet.onnx --out result1.jpg
hdr-process --img test.jpg --model paper2_custom.pt --out result2.jpg  
hdr-process --img test.jpg --model your_experiment.h5 --out result3.jpg
```

### **🎯 The Beauty of This Design**

- ✅ **No pipeline changes needed** - just update one method
- ✅ **Any AI framework** - ONNX, PyTorch, TensorFlow, etc.
- ✅ **Instant testing** - synthetic model for development
- ✅ **Proper error handling** - fails fast if model doesn't work
- ✅ **Clean separation** - AI model vs export logic independent

**Perfect for research iterations!** Try any model, any paper, any architecture - the pipeline handles the rest.