"""Tests for automated visual validation and similarity metrics"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imagehash

from hdr.gainmap_pipeline import run_gainmap_pipeline, GainMapPipelineConfig


class TestVisualSimilarityMetrics:
    """Test automated visual similarity scoring"""
    
    def test_psnr_identical_images(self, temp_dir):
        """Test PSNR calculation for identical images"""
        # Create test image
        test_array = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        
        # PSNR between identical images should be infinite (or very high)
        psnr = peak_signal_noise_ratio(test_array, test_array)
        assert psnr > 100  # Practically infinite
    
    def test_psnr_similar_images(self, temp_dir):
        """Test PSNR calculation for similar images"""
        # Create base image
        base_array = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        
        # Create slightly modified version (add small noise)
        noise = np.random.randint(-5, 6, base_array.shape, dtype=np.int16)
        modified_array = np.clip(base_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        psnr = peak_signal_noise_ratio(base_array, modified_array)
        assert 25 < psnr < 50  # Should be good but not perfect
    
    def test_ssim_identical_images(self, temp_dir):
        """Test SSIM calculation for identical images"""
        # Create test image
        test_array = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        
        # SSIM between identical images should be 1.0
        ssim = structural_similarity(
            test_array, test_array, 
            win_size=7, channel_axis=2, data_range=255
        )
        assert abs(ssim - 1.0) < 0.001
    
    def test_ssim_similar_images(self, temp_dir):
        """Test SSIM calculation for similar images"""
        # Create base image with some structure
        x, y = np.meshgrid(np.linspace(0, 2*np.pi, 200), np.linspace(0, 2*np.pi, 300))
        base_pattern = (128 + 127 * np.sin(x) * np.cos(y)).astype(np.uint8)
        base_array = np.stack([base_pattern, base_pattern, base_pattern], axis=2)
        
        # Create slightly modified version
        noise = np.random.randint(-10, 11, base_array.shape, dtype=np.int16)
        modified_array = np.clip(base_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        ssim = structural_similarity(
            base_array, modified_array,
            win_size=7, channel_axis=2, data_range=255
        )
        assert 0.7 < ssim < 1.0  # Should be similar but not identical


class TestPerceptualHashing:
    """Test perceptual hash-based similarity detection"""
    
    def test_phash_identical_images(self, temp_dir):
        """Test perceptual hash for identical images"""
        # Create test image
        test_array = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        image = Image.fromarray(test_array)
        
        # Get perceptual hashes
        hash1 = imagehash.phash(image)
        hash2 = imagehash.phash(image)
        
        # Hashes should be identical
        assert hash1 == hash2
        assert hash1 - hash2 == 0
    
    def test_phash_similar_images(self, temp_dir):
        """Test perceptual hash for similar images"""
        # Create base image
        base_array = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        base_image = Image.fromarray(base_array)
        
        # Create slightly modified version
        noise = np.random.randint(-20, 21, base_array.shape, dtype=np.int16)
        modified_array = np.clip(base_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        modified_image = Image.fromarray(modified_array)
        
        # Get perceptual hashes
        hash1 = imagehash.phash(base_image)
        hash2 = imagehash.phash(modified_image)
        
        # Hashes should be similar (low Hamming distance)
        hamming_distance = hash1 - hash2
        assert hamming_distance < 10  # Should be similar
    
    def test_ahash_rotation_sensitivity(self, temp_dir):
        """Test average hash sensitivity to rotation"""
        # Create structured test image
        test_array = np.zeros((200, 200, 3), dtype=np.uint8)
        test_array[50:150, 50:150] = 255  # White square in center
        base_image = Image.fromarray(test_array)
        
        # Rotate image slightly
        rotated_image = base_image.rotate(5)
        
        # Get average hashes
        hash1 = imagehash.average_hash(base_image)
        hash2 = imagehash.average_hash(rotated_image)
        
        # Should detect the difference but still be somewhat similar
        hamming_distance = hash1 - hash2
        assert 0 < hamming_distance < 20


class TestHDRVisualValidation:
    """Test HDR-specific visual validation"""
    
    @pytest.mark.visual
    def test_hdr_enhancement_detection(self, synthetic_sdr_image, temp_dir):
        """Test detection of HDR enhancement in processed images"""
        output_path = temp_dir / "hdr_output.jpg"
        
        # Run HDR processing
        config = GainMapPipelineConfig(model_type="synthetic", export_quality=95)
        run_gainmap_pipeline(
            input_path=str(synthetic_sdr_image),
            output_path=str(output_path),
            config=config
        )
        
        # Load original and processed images
        with Image.open(synthetic_sdr_image) as sdr_img:
            sdr_array = np.array(sdr_img)
        
        with Image.open(output_path) as hdr_img:
            hdr_array = np.array(hdr_img)
        
        # Images should be similar but not identical
        psnr = peak_signal_noise_ratio(sdr_array, hdr_array)
        assert 15 < psnr < 50  # Similar but enhanced
        
        # Should have reasonable structural similarity
        ssim = structural_similarity(
            sdr_array, hdr_array,
            win_size=7, channel_axis=2, data_range=255
        )
        assert ssim > 0.8  # Should preserve structure
    
    @pytest.mark.visual
    def test_highlight_enhancement_measurement(self, temp_dir):
        """Test measurement of highlight enhancement in HDR processing"""
        # Create image with distinct highlight regions
        width, height = 400, 300
        sdr_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create bright highlight region
        sdr_array[100:200, 150:250] = [240, 240, 240]  # Bright highlights
        sdr_array[50:250, 50:150] = [120, 120, 120]    # Mid-tones
        sdr_array[:, :50] = [30, 30, 30]               # Shadows
        
        sdr_path = temp_dir / "highlight_test_sdr.jpg"
        hdr_path = temp_dir / "highlight_test_hdr.jpg"
        
        Image.fromarray(sdr_array).save(sdr_path, "JPEG", export_quality=95)
        
        # Process with HDR pipeline
        config = GainMapPipelineConfig(model_type="synthetic")
        run_gainmap_pipeline(
            input_path=str(sdr_path),
            output_path=str(hdr_path),
            config=config
        )
        
        # Analyze highlight regions
        with Image.open(hdr_path) as hdr_img:
            hdr_array = np.array(hdr_img)
        
        # Extract highlight region (should be enhanced)
        sdr_highlights = sdr_array[100:200, 150:250]
        hdr_highlights = hdr_array[100:200, 150:250]
        
        # HDR should maintain or enhance highlights
        sdr_brightness = np.mean(sdr_highlights)
        hdr_brightness = np.mean(hdr_highlights)
        
        # Brightness should be preserved (within processing tolerance)
        brightness_ratio = hdr_brightness / sdr_brightness
        assert 0.8 < brightness_ratio < 1.2  # Allow some processing variation
    
    def test_dynamic_range_measurement(self, synthetic_sdr_image, temp_dir):
        """Test automated measurement of dynamic range improvement"""
        output_path = temp_dir / "dynamic_range_test.jpg"
        
        config = GainMapPipelineConfig(model_type="synthetic")
        run_gainmap_pipeline(
            input_path=str(synthetic_sdr_image),
            output_path=str(output_path),
            config=config
        )
        
        # Load and analyze images
        with Image.open(synthetic_sdr_image) as sdr_img:
            sdr_array = np.array(sdr_img)
        
        with Image.open(output_path) as hdr_img:
            hdr_array = np.array(hdr_img)
        
        # Calculate dynamic range (ratio of max to min non-zero values)
        def calculate_dynamic_range(img_array):
            gray = np.mean(img_array, axis=2)
            non_zero = gray[gray > 5]  # Exclude near-black pixels
            if len(non_zero) == 0:
                return 1.0
            return np.max(non_zero) / np.max(np.min(non_zero), 1)
        
        sdr_range = calculate_dynamic_range(sdr_array)
        hdr_range = calculate_dynamic_range(hdr_array)
        
        # HDR should maintain reasonable dynamic range
        assert hdr_range >= sdr_range * 0.8  # Should not reduce range significantly


class TestGoldenMasterComparison:
    """Test comparison against golden master reference images"""
    
    def test_create_golden_master(self, synthetic_sdr_image, golden_masters_dir):
        """Create golden master for future regression testing"""
        golden_masters_dir.mkdir(exist_ok=True)
        
        golden_output = golden_masters_dir / "synthetic_model_golden.jpg"
        
        # Only create if it doesn't exist (to preserve existing golden masters)
        if not golden_output.exists():
            config = GainMapPipelineConfig(model_type="synthetic", export_quality=95)
            run_gainmap_pipeline(
                input_path=str(synthetic_sdr_image),
                output_path=str(golden_output),
                config=config
            )
        
        assert golden_output.exists()
        assert golden_output.stat().st_size > 10000  # Non-trivial size
    
    @pytest.mark.visual
    def test_regression_against_golden_master(self, synthetic_sdr_image, golden_masters_dir, temp_dir):
        """Test regression by comparing against golden master"""
        golden_output = golden_masters_dir / "synthetic_model_golden.jpg"
        
        if not golden_output.exists():
            pytest.skip("Golden master not available")
        
        # Generate current output
        current_output = temp_dir / "current_synthetic.jpg"
        config = GainMapPipelineConfig(model_type="synthetic", export_quality=95)
        run_gainmap_pipeline(
            input_path=str(synthetic_sdr_image),
            output_path=str(current_output),
            config=config
        )
        
        # Load both images
        with Image.open(golden_output) as golden_img:
            golden_array = np.array(golden_img)
        
        with Image.open(current_output) as current_img:
            current_array = np.array(current_img)
        
        # Compare using multiple metrics
        psnr = peak_signal_noise_ratio(golden_array, current_array)
        ssim = structural_similarity(
            golden_array, current_array,
            win_size=7, channel_axis=2, data_range=255
        )
        
        # Perceptual hash comparison
        golden_hash = imagehash.phash(Image.fromarray(golden_array))
        current_hash = imagehash.phash(Image.fromarray(current_array))
        hash_distance = golden_hash - current_hash
        
        # Assert similarity thresholds for regression detection
        assert psnr > 30, f"PSNR too low: {psnr}"
        assert ssim > 0.95, f"SSIM too low: {ssim}"
        assert hash_distance < 5, f"Perceptual hash distance too high: {hash_distance}"


class TestCrossPlatformConsistency:
    """Test visual consistency across different processing paths"""
    
    def test_quality_setting_consistency(self, synthetic_sdr_image, temp_dir):
        """Test that different quality settings produce consistent results"""
        config_90 = GainMapPipelineConfig(model_type="synthetic", export_quality=90)
        config_95 = GainMapPipelineConfig(model_type="synthetic", export_quality=95)
        
        output_90 = temp_dir / "quality_90.jpg"
        output_95 = temp_dir / "quality_95.jpg"
        
        # Generate outputs with different quality settings
        run_gainmap_pipeline(str(synthetic_sdr_image), str(output_90), config_90)
        run_gainmap_pipeline(str(synthetic_sdr_image), str(output_95), config_95)
        
        # Load and compare
        with Image.open(output_90) as img_90:
            array_90 = np.array(img_90)
        
        with Image.open(output_95) as img_95:
            array_95 = np.array(img_95)
        
        # Should be similar despite quality difference
        ssim = structural_similarity(
            array_90, array_95,
            win_size=7, channel_axis=2, data_range=255
        )
        assert ssim > 0.90  # Should be quite similar
        
        # Higher quality should not be significantly larger
        size_90 = output_90.stat().st_size
        size_95 = output_95.stat().st_size
        size_ratio = size_95 / size_90
        assert 1.0 <= size_ratio <= 2.0  # Reasonable size increase