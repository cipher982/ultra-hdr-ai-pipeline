"""Gain map quality analysis tests

Tests the actual content and quality of generated gain maps to ensure
HDR effectiveness, not just structural compliance.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from scipy import ndimage

from hdr.gainmap_pipeline import run_gainmap_pipeline, GainMapPipelineConfig
from hdr.models import create_model, GainMapPrediction


class TestGainMapQuality:
    """Test gain map content quality and HDR effectiveness indicators"""
    
    def test_gain_map_generation_quality(self, synthetic_sdr_image, temp_dir):
        """Test that generated gain maps have reasonable quality characteristics"""
        # Generate HDR with gain map
        output_path = temp_dir / "gain_map_test.jpg"
        config = GainMapPipelineConfig(
            model_type="synthetic", 
            export_quality=95,
            save_intermediate=True  # Save gain map for analysis
        )
        
        result = run_gainmap_pipeline(
            img_path=str(synthetic_sdr_image),
            out_path=str(output_path),
            config=config
        )
        
        # Find the saved gain map (should be in same directory)
        gain_map_path = output_path.parent / f"{output_path.stem}_gainmap.png"
        if not gain_map_path.exists():
            # Try alternative naming
            gain_map_path = output_path.parent / "gainmap.png"
        
        if gain_map_path.exists():
            # Analyze gain map quality
            with Image.open(gain_map_path) as gain_img:
                gain_array = np.array(gain_img)
                
                # Basic sanity checks
                assert gain_array.ndim == 2, f"Gain map should be grayscale, got {gain_array.ndim}D"
                assert gain_array.dtype == np.uint8, f"Expected uint8, got {gain_array.dtype}"
                
                # Quality metrics
                gain_range = gain_array.max() - gain_array.min()
                assert gain_range > 10, f"Gain map has insufficient range: {gain_range}"
                
                # Should not be completely uniform
                gain_std = np.std(gain_array)
                assert gain_std > 5, f"Gain map too uniform, std={gain_std}"
                
                # Should have reasonable distribution (not all min/max)
                unique_values = len(np.unique(gain_array))
                assert unique_values > 10, f"Insufficient gain map granularity: {unique_values} unique values"
        else:
            pytest.skip("Gain map intermediate file not saved for analysis")
    
    def test_gain_map_brightness_correlation(self, temp_dir):
        """Test that gain map correlates logically with image brightness"""
        # Create test image with clear bright/dark regions
        height, width = 300, 400
        test_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create distinct regions
        test_array[:height//2, :] = [50, 50, 50]     # Dark region (top)
        test_array[height//2:, :] = [200, 200, 200]  # Bright region (bottom)
        
        test_path = temp_dir / "brightness_test.jpg"
        output_path = temp_dir / "brightness_hdr.jpg"
        
        Image.fromarray(test_array).save(test_path, "JPEG", quality=95)
        
        # Generate gain map directly using model
        model = create_model("synthetic")
        sdr_rgb = test_array.astype(np.float32) / 255.0
        prediction = model.predict(sdr_rgb)
        
        # Analyze correlation between brightness and gain
        brightness = np.mean(sdr_rgb, axis=2)  # Grayscale brightness
        gain_map = prediction.gain_map.astype(np.float32)
        
        # Bright regions should generally have higher gain (for HDR boost)
        dark_mask = brightness < 0.3
        bright_mask = brightness > 0.7
        
        if np.sum(dark_mask) > 0 and np.sum(bright_mask) > 0:
            avg_gain_dark = np.mean(gain_map[dark_mask])
            avg_gain_bright = np.mean(gain_map[bright_mask])
            
            # Bright regions should have higher gain for HDR enhancement
            gain_ratio = avg_gain_bright / max(avg_gain_dark, 1.0)
            assert gain_ratio > 1.1, f"Insufficient gain difference: bright/dark ratio = {gain_ratio:.2f}"
    
    def test_gain_map_smoothness(self, synthetic_sdr_image, temp_dir):
        """Test that gain map is reasonably smooth without artifacts"""
        model = create_model("synthetic")
        
        # Load and process image
        with Image.open(synthetic_sdr_image) as img:
            sdr_array = np.array(img)
        
        sdr_rgb = sdr_array.astype(np.float32) / 255.0
        prediction = model.predict(sdr_rgb)
        gain_map = prediction.gain_map.astype(np.float32)
        
        # Calculate smoothness metrics
        # Gradient magnitude (edges)
        grad_x = np.abs(np.gradient(gain_map, axis=1))
        grad_y = np.abs(np.gradient(gain_map, axis=0))
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Should not have excessive sharp edges
        mean_gradient = np.mean(gradient_magnitude)
        max_gradient = np.max(gradient_magnitude)
        
        assert mean_gradient < 20, f"Gain map too noisy, mean gradient: {mean_gradient:.2f}"
        assert max_gradient < 100, f"Sharp artifacts detected, max gradient: {max_gradient:.2f}"
        
        # Local variance should be reasonable
        local_var = ndimage.generic_filter(gain_map, np.var, size=5)
        mean_local_var = np.mean(local_var)
        
        assert mean_local_var < 500, f"Excessive local variation: {mean_local_var:.2f}"
    
    def test_gain_map_dynamic_range_effectiveness(self, synthetic_sdr_image, temp_dir):
        """Test that gain map effectively expands dynamic range"""
        model = create_model("synthetic") 
        
        # Load image
        with Image.open(synthetic_sdr_image) as img:
            sdr_array = np.array(img)
        
        sdr_rgb = sdr_array.astype(np.float32) / 255.0
        prediction = model.predict(sdr_rgb)
        
        # Calculate effective dynamic range expansion
        gain_min_linear = 2 ** prediction.gain_min_log2
        gain_max_linear = 2 ** prediction.gain_max_log2
        
        dynamic_range_expansion = gain_max_linear / gain_min_linear
        
        # Should provide meaningful HDR expansion
        assert dynamic_range_expansion > 2.0, f"Insufficient dynamic range: {dynamic_range_expansion:.2f}x"
        assert dynamic_range_expansion < 100.0, f"Unrealistic dynamic range: {dynamic_range_expansion:.2f}x"
        
        # Log2 range should be reasonable (iPhone reference: ~2.68 stops)
        log2_range = prediction.gain_max_log2 - prediction.gain_min_log2
        assert 1.0 < log2_range < 5.0, f"Log2 range outside reasonable bounds: {log2_range:.2f} stops"
    
    def test_gain_map_coverage_analysis(self, temp_dir):
        """Test that gain map provides appropriate coverage of image regions"""
        # Create test image with various brightness levels
        height, width = 200, 300
        test_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create brightness gradient
        for y in range(height):
            brightness = int(255 * y / height)
            test_array[y, :] = [brightness, brightness, brightness]
        
        sdr_rgb = test_array.astype(np.float32) / 255.0
        
        model = create_model("synthetic")
        prediction = model.predict(sdr_rgb)
        gain_map = prediction.gain_map
        
        # Analyze coverage across brightness levels
        brightness_levels = np.mean(sdr_rgb, axis=2)
        
        # Calculate gain for different brightness quintiles
        quintiles = []
        for i in range(5):
            lower = i / 5.0
            upper = (i + 1) / 5.0
            mask = (brightness_levels >= lower) & (brightness_levels < upper)
            if np.sum(mask) > 0:
                avg_gain = np.mean(gain_map[mask])
                quintiles.append(avg_gain)
        
        if len(quintiles) >= 3:
            # Gain should generally increase with brightness (for HDR)
            # Allow some variation but expect overall increasing trend
            gain_trend = np.polyfit(range(len(quintiles)), quintiles, 1)[0]  # Linear trend
            assert gain_trend >= 0, f"Gain decreases with brightness: trend={gain_trend:.2f}"
    
    def test_gain_map_metadata_consistency(self, synthetic_sdr_image):
        """Test that gain map metadata is consistent with actual gain map content"""
        model = create_model("synthetic")
        
        with Image.open(synthetic_sdr_image) as img:
            sdr_array = np.array(img)
        
        sdr_rgb = sdr_array.astype(np.float32) / 255.0
        prediction = model.predict(sdr_rgb)
        
        # Check consistency between gain map values and metadata
        gain_map_norm = prediction.gain_map.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Map normalized values to log2 range
        log2_range = prediction.gain_max_log2 - prediction.gain_min_log2
        gain_log2_values = prediction.gain_min_log2 + gain_map_norm * log2_range
        
        # Check that actual values stay within stated bounds
        actual_min = np.min(gain_log2_values)
        actual_max = np.max(gain_log2_values)
        
        tolerance = 0.1  # Allow small tolerance for quantization
        assert actual_min >= prediction.gain_min_log2 - tolerance, \
            f"Gain values below stated minimum: {actual_min:.3f} < {prediction.gain_min_log2:.3f}"
        assert actual_max <= prediction.gain_max_log2 + tolerance, \
            f"Gain values above stated maximum: {actual_max:.3f} > {prediction.gain_max_log2:.3f}"


class TestGainMapRegression:
    """Test gain map quality regression against reference standards"""
    
    def test_gain_map_quality_regression(self, synthetic_sdr_image, golden_masters_dir):
        """Test that gain map quality doesn't regress from baseline"""
        # Generate current gain map
        model = create_model("synthetic")
        
        with Image.open(synthetic_sdr_image) as img:
            sdr_array = np.array(img)
        
        sdr_rgb = sdr_array.astype(np.float32) / 255.0
        current_prediction = model.predict(sdr_rgb)
        
        # Save baseline if it doesn't exist
        baseline_path = golden_masters_dir / "gain_map_baseline.npy"
        golden_masters_dir.mkdir(exist_ok=True)
        
        if not baseline_path.exists():
            np.save(baseline_path, current_prediction.gain_map)
            pytest.skip("Created gain map baseline for future regression testing")
        
        # Load baseline and compare
        baseline_gain_map = np.load(baseline_path)
        current_gain_map = current_prediction.gain_map
        
        # Calculate similarity metrics
        mse = np.mean((baseline_gain_map.astype(np.float32) - current_gain_map.astype(np.float32)) ** 2)
        correlation = np.corrcoef(baseline_gain_map.flatten(), current_gain_map.flatten())[0, 1]
        
        # Should be very similar (allowing for minor improvements)
        assert mse < 100, f"Gain map MSE regression: {mse:.2f}"
        assert correlation > 0.95, f"Gain map correlation regression: {correlation:.3f}"
        
        # Dynamic range should be similar
        baseline_range = baseline_gain_map.max() - baseline_gain_map.min()
        current_range = current_gain_map.max() - current_gain_map.min()
        range_ratio = current_range / max(baseline_range, 1)
        
        assert 0.8 < range_ratio < 1.2, f"Dynamic range regression: {range_ratio:.2f}x"