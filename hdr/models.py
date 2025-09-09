"""
AI model interface abstraction for gain-map generation.

This module provides a clean interface for different AI models that can generate
gain maps directly from SDR input, supporting the gain-map-first pipeline architecture.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class GainMapPrediction:
    """Result from AI model gain map prediction"""
    gain_map: np.ndarray  # 8-bit grayscale gain map (H, W)
    gain_min_log2: float  # Minimum gain in log2 stops
    gain_max_log2: float  # Maximum gain in log2 stops  
    confidence: Optional[float] = None  # Model confidence (0-1)


class GainMapModel(ABC):
    """Abstract interface for gain map AI models"""
    
    @abstractmethod
    def predict(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """
        Generate gain map directly from SDR input.
        
        Args:
            sdr_rgb: SDR image as float32 array (H, W, 3) in range [0, 1]
            
        Returns:
            GainMapPrediction with gain map and metadata
            
        Raises:
            ModelError: If prediction fails
        """
        pass
    
    @abstractmethod  
    def is_available(self) -> bool:
        """Check if model is loaded and ready for inference"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for logging"""
        pass


class ModelError(Exception):
    """Raised when AI model operations fail"""
    pass


class ModelPlugin(GainMapModel):
    """
    Generic AI model plugin that can load any gain map model.
    
    This provides maximum flexibility for research - just point it at any
    model file and implement the inference function. Perfect for trying
    different papers, architectures, or your own trained models.
    """
    
    def __init__(self, model_path: str, model_name: Optional[str] = None):
        self.model_path = model_path
        self.model_name = model_name or self._detect_model_type(model_path)
        self._model = None
        self._inference_func = None
        
        # Auto-detect and load the appropriate inference function
        self._setup_inference()
    
    def _detect_model_type(self, path: str) -> str:
        """Auto-detect model type from file extension/name"""
        path_lower = path.lower()
        
        if 'gmnet' in path_lower:
            return "GMNet"
        elif path_lower.endswith('.onnx'):
            return "ONNX"
        elif path_lower.endswith('.pt') or path_lower.endswith('.pth'):
            return "PyTorch"
        elif path_lower.endswith('.h5'):
            return "Keras/TensorFlow"
        elif path_lower.endswith('.tflite'):
            return "TensorFlow Lite"
        elif path_lower.endswith('.pkl') or path_lower.endswith('.pickle'):
            return "Scikit-learn/Pickle"
        else:
            return f"Unknown({os.path.basename(path)})"
    
    def _setup_inference(self):
        """Setup the appropriate inference function based on model type"""
        model_type = self.model_name.lower()
        
        if 'gmnet' in model_type:
            self._inference_func = self._infer_gmnet
        elif 'onnx' in model_type:
            self._inference_func = self._infer_onnx
        elif 'pytorch' in model_type or '.pt' in self.model_path.lower():
            self._inference_func = self._infer_pytorch
        elif 'tensorflow' in model_type or '.h5' in self.model_path.lower():
            self._inference_func = self._infer_tensorflow
        elif 'tflite' in model_type:
            self._inference_func = self._infer_tflite
        else:
            self._inference_func = self._infer_generic
    
    def predict(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """Generate gain map using the loaded model"""
        if not self.is_available():
            raise ModelError(f"Model not available: {self.model_path}")
            
        try:
            return self._inference_func(sdr_rgb)
        except Exception as e:
            raise ModelError(f"{self.model_name} inference failed: {e}")
    
    def is_available(self) -> bool:
        """Check if model file exists and can be loaded"""
        if not os.path.exists(self.model_path):
            return False
            
        # Quick availability check based on model type
        model_type = self.model_name.lower()
        try:
            if 'onnx' in model_type:
                import onnxruntime
            elif 'pytorch' in model_type or '.pt' in self.model_path.lower():
                import torch
            elif 'tensorflow' in model_type:
                import tensorflow as tf
            elif 'tflite' in model_type:
                import tensorflow as tf
            return True
        except ImportError:
            return False
    
    @property
    def name(self) -> str:
        return f"{self.model_name}({os.path.basename(self.model_path)})"
    
    # Inference implementations for different model types
    def _infer_gmnet(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """GMNet-specific inference - REPLACE WITH YOUR IMPLEMENTATION"""
        raise ModelError(
            "GMNet inference not yet implemented. "
            "Replace this method with your GMNet model loading and inference code."
        )
    
    def _infer_onnx(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """ONNX model inference"""
        try:
            import onnxruntime as ort
            
            if self._model is None:
                self._model = ort.InferenceSession(self.model_path)
            
            # Prepare input (model-specific preprocessing may be needed)
            input_batch = sdr_rgb[np.newaxis, ...].astype(np.float32)  # Add batch dim
            
            # Run inference  
            outputs = self._model.run(None, {'input': input_batch})
            
            # Parse outputs (model-specific - you'll need to adapt this)
            gain_map = outputs[0].squeeze().astype(np.uint8)
            
            return GainMapPrediction(
                gain_map=gain_map,
                gain_min_log2=0.0,
                gain_max_log2=3.0,  # You'll get these from the model output
                confidence=0.9
            )
        except ImportError:
            raise ModelError("ONNX Runtime not available")
    
    def _infer_pytorch(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """PyTorch model inference"""
        try:
            import torch
            
            if self._model is None:
                self._model = torch.load(self.model_path, map_location='cpu')
                self._model.eval()
            
            # Convert to tensor
            input_tensor = torch.from_numpy(sdr_rgb).permute(2, 0, 1).unsqueeze(0)  # NCHW
            
            with torch.no_grad():
                outputs = self._model(input_tensor)
            
            # Extract gain map (adapt to your model's output format)
            gain_map = outputs.squeeze().cpu().numpy().astype(np.uint8)
            
            return GainMapPrediction(
                gain_map=gain_map,
                gain_min_log2=0.0,
                gain_max_log2=3.0,
                confidence=0.9
            )
        except ImportError:
            raise ModelError("PyTorch not available")
    
    def _infer_tensorflow(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """TensorFlow/Keras model inference"""
        try:
            import tensorflow as tf
            
            if self._model is None:
                self._model = tf.keras.models.load_model(self.model_path)
            
            # Prepare input
            input_batch = sdr_rgb[np.newaxis, ...]  # Add batch dimension
            
            # Inference
            outputs = self._model(input_batch, training=False)
            gain_map = outputs.numpy().squeeze().astype(np.uint8)
            
            return GainMapPrediction(
                gain_map=gain_map,
                gain_min_log2=0.0,
                gain_max_log2=3.0,
                confidence=0.9
            )
        except ImportError:
            raise ModelError("TensorFlow not available")
    
    def _infer_tflite(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """TensorFlow Lite model inference"""
        try:
            import tensorflow as tf
            
            if self._model is None:
                self._model = tf.lite.Interpreter(model_path=self.model_path)
                self._model.allocate_tensors()
            
            # Get input/output details
            input_details = self._model.get_input_details()
            output_details = self._model.get_output_details()
            
            # Set input
            self._model.set_tensor(input_details[0]['index'], sdr_rgb[np.newaxis, ...])
            
            # Run inference
            self._model.invoke()
            
            # Get output
            gain_map = self._model.get_tensor(output_details[0]['index']).squeeze().astype(np.uint8)
            
            return GainMapPrediction(
                gain_map=gain_map,
                gain_min_log2=0.0,
                gain_max_log2=3.0,
                confidence=0.9
            )
        except ImportError:
            raise ModelError("TensorFlow not available")
    
    def _infer_generic(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """Generic inference - override this for custom model types"""
        raise ModelError(
            f"Unknown model type: {self.model_name}. "
            f"Override _infer_generic() method to implement custom inference for {self.model_path}"
        )


# Legacy HDRCNN support removed - gain-map-first architecture only


class SyntheticGainMapModel(GainMapModel):
    """
    Fallback synthetic gain map generation for testing/development.
    
    Creates reasonable gain maps from SDR input without AI models.
    """
    
    def __init__(self, max_boost_stops: float = 3.0):
        self.max_boost_stops = max_boost_stops
        
    def predict(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """Generate synthetic gain map based on luminance"""
        
        # Compute luminance (Rec. 709 weights)
        L = 0.2126 * sdr_rgb[..., 0] + 0.7152 * sdr_rgb[..., 1] + 0.0722 * sdr_rgb[..., 2]
        
        # Progressive boost: brighter areas get more enhancement
        # This creates a more realistic gain map than uniform boost
        boost_linear = np.ones_like(L)
        
        # Mid-tones: 1-3x boost
        mid_mask = (L > 0.2) & (L <= 0.6)
        boost_linear[mid_mask] = 1.0 + 2.0 * ((L[mid_mask] - 0.2) / 0.4)
        
        # Highlights: 3-8x boost  
        bright_mask = L > 0.6
        max_linear_boost = 2 ** self.max_boost_stops
        boost_linear[bright_mask] = 3.0 + (max_linear_boost - 3.0) * ((L[bright_mask] - 0.6) / 0.4)
        
        # Convert to log2 and normalize to 0-255
        boost_log2 = np.log2(np.clip(boost_linear, 1.0, max_linear_boost))
        gain_min = 0.0
        gain_max = self.max_boost_stops
        
        # Normalize and quantize
        normalized = (boost_log2 - gain_min) / max(gain_max - gain_min, 1e-6)
        gain_map = (np.clip(normalized, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        
        return GainMapPrediction(
            gain_map=gain_map,
            gain_min_log2=gain_min,
            gain_max_log2=gain_max,
            confidence=0.0  # Synthetic, no real confidence
        )
    
    def is_available(self) -> bool:
        """Synthetic model is always available"""
        return True
        
    @property
    def name(self) -> str:
        return f"Synthetic(max_boost={self.max_boost_stops}stops)"


class GMNetModel(GainMapModel):
    """
    GMNet (ICLR 2025) gain map generation model.
    
    Implements the state-of-the-art dual-branch architecture for
    direct SDR→gain map prediction without intermediate HDR generation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.gmnet_dir = self._find_gmnet_directory()
        self.model_path = model_path or self._get_default_model_path()
        self._model = None
        self._device = None
        self._setup_gmnet_environment()
        
    def _find_gmnet_directory(self) -> str:
        """Locate GMNet directory relative to this file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        gmnet_path = os.path.join(os.path.dirname(current_dir), "GMNet")
        
        if not os.path.exists(gmnet_path):
            raise ModelError(
                f"GMNet directory not found at {gmnet_path}. "
                "Please ensure GMNet is cloned in the project root."
            )
        return gmnet_path
    
    def _get_default_model_path(self) -> str:
        """Get path to default pretrained model"""
        synthetic_model = os.path.join(self.gmnet_dir, "checkpoints", "G_synthetic.pth")
        realworld_model = os.path.join(self.gmnet_dir, "checkpoints", "G_realworld.pth")
        
        # Prefer real-world model if available
        if os.path.exists(realworld_model):
            return realworld_model
        elif os.path.exists(synthetic_model):
            return synthetic_model
        else:
            raise ModelError(
                f"No GMNet pretrained models found in {self.gmnet_dir}/checkpoints/. "
                "Please ensure GMNet weights are properly downloaded."
            )
    
    def _setup_gmnet_environment(self):
        """Setup GMNet imports and environment"""
        import sys
        gmnet_codes_path = os.path.join(self.gmnet_dir, "codes")
        if gmnet_codes_path not in sys.path:
            sys.path.insert(0, gmnet_codes_path)
        
        try:
            # Import GMNet modules (these will be available after path setup)
            import torch
            self._torch = torch
            self._device = torch.device('cpu')  # Force CPU for compatibility
            
        except ImportError as e:
            raise ModelError(f"GMNet dependencies not available: {e}")
    
    def _load_model(self):
        """Lazy load GMNet model"""
        if self._model is not None:
            return
            
        try:
            import options.options as option
            from models import create_model as gmnet_create_model
            
            # Create minimal config for GMNet
            opt = {
                'gpu_ids': [],  # Force CPU
                'is_train': False,
                'dist': False,  # No distributed training
                'train': None,  # No training config needed
                'network_G': {
                    'which_model_G': 'GMNet',
                    'in_nc': 3,
                    'out_nc': 1,
                    'nf': 64,
                    'nb': 16,
                    'act_type': 'relu'
                },
                'path': {
                    'pretrain_model_G': self.model_path,
                    'strict_load': False
                },
                'model': 'base',
                'scale': 1,
                'peak': 8.0
            }
            
            # Convert to namespace-like object (GMNet expects this format)
            class OptDict:
                def __init__(self, d):
                    self.__dict__.update(d)
                def __getitem__(self, key):
                    return self.__dict__[key]
                def __contains__(self, key):
                    return key in self.__dict__
            
            def dict_to_nonedict(opt):
                if isinstance(opt, dict):
                    new_opt = OptDict({})
                    for key, sub_opt in opt.items():
                        new_opt.__dict__[key] = dict_to_nonedict(sub_opt)
                    return new_opt
                else:
                    return opt
                    
            opt_obj = dict_to_nonedict(opt)
            
            # Create GMNet model
            gmnet_model = gmnet_create_model(opt_obj)
            self._model = gmnet_model.netG
            self._model.eval()
            
            print(f"✓ GMNet loaded: {os.path.basename(self.model_path)}")
            
        except Exception as e:
            raise ModelError(f"Failed to load GMNet model: {e}")
    
    def predict(self, sdr_rgb: np.ndarray) -> GainMapPrediction:
        """Generate gain map using GMNet"""
        self._load_model()  # Lazy loading
        
        try:
            import cv2
            
            # Prepare inputs - GMNet needs [full_image, thumbnail]
            h, w = sdr_rgb.shape[:2]
            
            # Resize if too large (GMNet handles various sizes but be reasonable)
            max_size = 1024
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                sdr_rgb = cv2.resize(sdr_rgb, (new_w, new_h))
            
            # Create thumbnail (256x256 as per GMNet dataset format)
            thumbnail = cv2.resize(sdr_rgb, (256, 256))
            
            # Convert to tensors (C, H, W format)
            img_full = self._torch.from_numpy(sdr_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
            img_thumb = self._torch.from_numpy(thumbnail.transpose(2, 0, 1)).unsqueeze(0).float()
            
            # Run inference
            with self._torch.no_grad():
                outputs = self._model([img_full, img_thumb])
                gain_map_tensor = outputs[0]  # First output is main gain map
            
            # Convert to numpy
            gain_map = gain_map_tensor.squeeze().cpu().numpy()
            
            # GMNet outputs normalized gain maps in [0,1] range
            # Clamp and convert to uint8
            gain_map_clamped = np.clip(gain_map, 0, 1)
            gain_map_uint8 = (gain_map_clamped * 255).astype(np.uint8)
            
            # Compute metadata - GMNet outputs represent log2 gain values
            # Based on training, typical range is [0, log2(peak)] where peak=8
            gain_min_log2 = 0.0
            gain_max_log2 = 3.0  # log2(8) - matches GMNet training
            
            # For more accurate metadata, compute from actual values
            gain_values = gain_map_clamped[gain_map_clamped > 0.01]  # Non-zero values
            if len(gain_values) > 0:
                # Map [0,1] output to actual log2 gain range
                actual_min = float(gain_values.min() * gain_max_log2)
                actual_max = float(gain_values.max() * gain_max_log2)
                gain_min_log2 = max(0.0, actual_min)
                gain_max_log2 = actual_max
            
            return GainMapPrediction(
                gain_map=gain_map_uint8,
                gain_min_log2=gain_min_log2,
                gain_max_log2=gain_max_log2,
                confidence=0.95  # GMNet is a trained model
            )
            
        except Exception as e:
            raise ModelError(f"GMNet inference failed: {e}")
    
    def is_available(self) -> bool:
        """Check if GMNet is available"""
        try:
            return (
                os.path.exists(self.gmnet_dir) and 
                os.path.exists(self.model_path)
            )
        except:
            return False
    
    @property
    def name(self) -> str:
        model_name = os.path.basename(self.model_path).replace('.pth', '')
        return f"GMNet-{model_name}"


def create_model(model_type: str = "auto", **kwargs) -> GainMapModel:
    """
    Factory function for creating gain map models.
    
    Args:
        model_type: "gmnet", "hdrcnn", "synthetic", or "auto"
        **kwargs: Model-specific parameters
        
    Returns:
        Configured GainMapModel instance
        
    Raises:
        ValueError: If model_type is unsupported
        ModelError: If model cannot be created
    """
    if model_type == "auto":
        # Auto-detect based on model path
        model_path = kwargs.get('model_path')
        if model_path and os.path.exists(model_path):
            return ModelPlugin(model_path)
        else:
            return SyntheticGainMapModel(kwargs.get('max_boost_stops', 3.0))
            
    elif model_type == "synthetic":
        return SyntheticGainMapModel(kwargs.get('max_boost_stops', 3.0))
    elif model_type == "gmnet":
        # Use dedicated GMNet implementation
        model_path = kwargs.get('model_path')  # Optional - will use default if not provided
        return GMNetModel(model_path)
    elif model_type in ["onnx", "pytorch", "tensorflow", "tflite"]:
        # Use generic plugin for other AI models
        model_path = kwargs.get('model_path')
        if not model_path:
            raise ValueError(f"{model_type} requires model_path")
        return ModelPlugin(model_path, model_type.upper())
    else:
        raise ValueError(f"Unsupported model type: {model_type}")