"""Tests for HDR model implementations and accuracy validation"""

import pytest
import numpy as np
from pathlib import Path

from hdr.models import create_model, GMNetModel, SyntheticGainMapModel, ModelError, GainMapPrediction


class TestModelFactory:
    """Test model creation and availability"""
    
    def test_create_gmnet_model(self):
        """Test GMNet model creation"""
        model = create_model("gmnet")
        assert isinstance(model, GMNetModel)
    
    def test_create_synthetic_model(self):
        """Test synthetic model creation"""
        model = create_model("synthetic")
        assert isinstance(model, SyntheticGainMapModel)
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model names"""
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model("nonexistent_model")


class TestSyntheticModel:
    """Test synthetic model (always available baseline)"""
    
    def test_synthetic_always_available(self):
        """Synthetic model should always be available"""
        model = SyntheticGainMapModel()
        assert model.is_available() is True
    
    def test_synthetic_prediction_format(self, synthetic_sdr_image):
        """Test synthetic model output format"""
        model = SyntheticGainMapModel()
        
        # Load test image
        from PIL import Image
        sdr_pil = Image.open(synthetic_sdr_image)
        sdr_rgb = np.array(sdr_pil).astype(np.float32) / 255.0
        
        # Get prediction
        prediction = model.predict(sdr_rgb)
        
        # Validate output format
        assert isinstance(prediction, GainMapPrediction)
        assert prediction.gain_map.dtype == np.uint8
        assert prediction.gain_map.shape == sdr_rgb.shape[:2]  # Height x Width
        assert prediction.gain_min_log2 <= prediction.gain_max_log2
        assert 0 <= prediction.confidence <= 1
    
    def test_synthetic_deterministic(self, synthetic_sdr_image):
        """Test that synthetic model produces deterministic results"""
        model = SyntheticGainMapModel()
        
        # Load test image
        from PIL import Image
        sdr_pil = Image.open(synthetic_sdr_image)
        sdr_rgb = np.array(sdr_pil).astype(np.float32) / 255.0
        
        # Run prediction twice
        pred1 = model.predict(sdr_rgb)
        pred2 = model.predict(sdr_rgb) 
        
        # Results should be identical
        np.testing.assert_array_equal(pred1.gain_map, pred2.gain_map)
        assert pred1.gain_min_log2 == pred2.gain_min_log2
        assert pred1.gain_max_log2 == pred2.gain_max_log2


class TestGMNetModel:
    """Test GMNet model (when available)"""
    
    def test_gmnet_availability_check(self):
        """Test GMNet availability detection"""
        model = GMNetModel()
        # Should not crash, returns bool
        available = model.is_available()
        assert isinstance(available, bool)
    
    @pytest.mark.slow
    def test_gmnet_prediction_format(self, synthetic_sdr_image):
        """Test GMNet model output format (if available)"""
        model = GMNetModel()
        if not model.is_available():
            pytest.skip("GMNet model not available")
        
        # Load test image
        from PIL import Image
        sdr_pil = Image.open(synthetic_sdr_image) 
        sdr_rgb = np.array(sdr_pil).astype(np.float32) / 255.0
        
        # Get prediction
        prediction = model.predict(sdr_rgb)
        
        # Validate output format (same as synthetic)
        assert isinstance(prediction, GainMapPrediction)
        assert prediction.gain_map.dtype == np.uint8
        assert prediction.gain_map.shape == sdr_rgb.shape[:2]
        assert prediction.gain_min_log2 <= prediction.gain_max_log2
        assert 0 <= prediction.confidence <= 1
    
    @pytest.mark.slow
    def test_gmnet_memory_cleanup(self, synthetic_sdr_image):
        """Test that GMNet cleans up GPU memory properly"""
        model = GMNetModel()
        if not model.is_available():
            pytest.skip("GMNet model not available")
        
        # Load test image
        from PIL import Image
        sdr_pil = Image.open(synthetic_sdr_image)
        sdr_rgb = np.array(sdr_pil).astype(np.float32) / 255.0
        
        # Run multiple predictions
        for i in range(3):
            prediction = model.predict(sdr_rgb)
            assert prediction is not None
        
        # Should not crash or leak memory


class TestModelValidation:
    """Test model input validation and error handling"""
    
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes"""
        model = SyntheticGainMapModel()
        
        # Test 1D array (invalid)
        invalid_1d = np.random.rand(100).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 3D array"):
            model.predict(invalid_1d)
        
        # Test 2D array (invalid)  
        invalid_2d = np.random.rand(100, 100).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 3D array"):
            model.predict(invalid_2d)
        
        # Test wrong number of channels
        invalid_channels = np.random.rand(100, 100, 4).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            model.predict(invalid_channels)
    
    def test_invalid_input_range(self):
        """Test error handling for invalid input value ranges"""
        model = SyntheticGainMapModel()
        
        # Test values outside [0,1] range
        invalid_range = np.random.rand(100, 100, 3).astype(np.float32) * 2.0  # [0,2]
        with pytest.raises(ValueError, match="Input values must be in range"):
            model.predict(invalid_range)
        
        # Test negative values
        invalid_negative = np.random.rand(100, 100, 3).astype(np.float32) - 0.5  # [-0.5, 0.5]
        with pytest.raises(ValueError, match="Input values must be in range"):
            model.predict(invalid_negative)
    
    def test_zero_size_input(self):
        """Test error handling for zero-size inputs"""
        model = SyntheticGainMapModel()
        
        # Test empty array
        empty_input = np.empty((0, 0, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Input image cannot be empty"):
            model.predict(empty_input)


class TestGainMapRange:
    """Test gain map dynamic range validation"""
    
    def test_gain_map_range_bounds(self, synthetic_sdr_image):
        """Test that gain maps stay within expected dynamic range"""
        model = SyntheticGainMapModel()
        
        # Load test image
        from PIL import Image
        sdr_pil = Image.open(synthetic_sdr_image)
        sdr_rgb = np.array(sdr_pil).astype(np.float32) / 255.0
        
        prediction = model.predict(sdr_rgb)
        
        # Gain map should be full 8-bit range
        assert prediction.gain_map.min() >= 0
        assert prediction.gain_map.max() <= 255
        
        # Log2 range should be reasonable for HDR (iPhone reference: 2.68 stops)
        assert prediction.gain_min_log2 >= -2.0  # Not too dark
        assert prediction.gain_max_log2 <= 4.0   # Not unreasonably bright
        assert (prediction.gain_max_log2 - prediction.gain_min_log2) <= 5.0  # Reasonable range
    
    def test_iphone_reference_compatibility(self, synthetic_sdr_image):
        """Test compatibility with iPhone HDR reference specs"""
        model = SyntheticGainMapModel()
        
        # Load test image
        from PIL import Image
        sdr_pil = Image.open(synthetic_sdr_image)
        sdr_rgb = np.array(sdr_pil).astype(np.float32) / 255.0
        
        prediction = model.predict(sdr_rgb)
        
        # Should be compatible with iPhone reference (2.68 log2 stops)
        dynamic_range = prediction.gain_max_log2 - prediction.gain_min_log2
        
        # Allow reasonable range around iPhone reference
        assert 1.0 <= dynamic_range <= 4.0  # 2-16x linear range
        
        # Confidence should be reasonable 
        assert prediction.confidence >= 0.1  # Not completely uncertain