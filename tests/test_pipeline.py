"""Tests for HDR pipeline end-to-end functionality"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from hdr.gainmap_pipeline import (
    run_gainmap_pipeline, 
    GainMapPipelineConfig,
    validate_ultrahdr_structure,
    GainMapPipelineError
)


class TestPipelineConfiguration:
    """Test pipeline configuration validation"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = GainMapPipelineConfig()
        
        # Should have sensible defaults
        assert config.model_type == "auto"
        assert config.export_quality >= 80
        assert config.strip_exif is True
        assert config.strict_mode is True
    
    def test_config_validation(self, temp_dir):
        """Test configuration parameter validation"""
        # Valid configuration should not raise
        valid_config = GainMapPipelineConfig(
            model_type="synthetic",
            export_quality=95,
            strip_exif=False
        )
        assert valid_config.model_type == "synthetic"
        
        # Configuration should accept reasonable values
        assert valid_config.export_quality == 95
        assert valid_config.strip_exif is False


class TestPipelineExecution:
    """Test end-to-end pipeline execution"""
    
    def test_synthetic_model_pipeline(self, synthetic_sdr_image, temp_dir):
        """Test complete pipeline with synthetic model"""
        output_path = temp_dir / "synthetic_output.jpg"
        
        config = GainMapPipelineConfig(
            model_type="synthetic",
            export_quality=90,
            save_intermediate=True
        )
        
        # Run pipeline
        result = run_gainmap_pipeline(
            img_path=str(synthetic_sdr_image),
            out_path=str(output_path),
            config=config
        )
        
        # Validate result
        assert result is not None
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Non-trivial file size
        
        # Validate HDR structure
        validation = validate_ultrahdr_structure(str(output_path))
        assert validation["valid"] is True
        assert validation["metadata"]["has_mpf"] is True
        assert validation["metadata"]["jpeg_count"] >= 2  # SDR + gain map
    
    @pytest.mark.slow 
    def test_gmnet_pipeline(self, synthetic_sdr_image, temp_dir):
        """Test pipeline with GMNet model (if available)"""
        from hdr.models import GMNetModel
        
        model = GMNetModel()
        if not model.is_available():
            pytest.skip("GMNet model not available")
        
        output_path = temp_dir / "gmnet_output.jpg"
        
        config = GainMapPipelineConfig(
            model_type="gmnet",
            export_quality=95,
            save_intermediate=True
        )
        
        # Run pipeline
        result = run_gainmap_pipeline(
            img_path=str(synthetic_sdr_image),
            out_path=str(output_path),
            config=config
        )
        
        # Validate result
        assert result is not None
        assert output_path.exists()
        
        # Validate HDR structure
        validation = validate_ultrahdr_structure(str(output_path))
        assert validation["valid"] is True
    
    def test_pipeline_file_formats(self, temp_dir):
        """Test pipeline with different input file formats"""
        from PIL import Image
        
        # Create test images in different formats
        test_array = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        
        formats = [
            ("jpg", "JPEG"),
            ("jpeg", "JPEG"), 
            ("png", "PNG")
        ]
        
        config = GainMapPipelineConfig(model_type="synthetic")
        
        for ext, pil_format in formats:
            # Create input image
            input_path = temp_dir / f"test_input.{ext}"
            output_path = temp_dir / f"test_output_{ext}.jpg"
            
            Image.fromarray(test_array).save(input_path, pil_format)
            
            # Run pipeline
            result = run_gainmap_pipeline(
                img_path=str(input_path),
                out_path=str(output_path), 
                config=config
            )
            
            assert result is not None
            assert output_path.exists()


class TestPipelineErrorHandling:
    """Test pipeline error handling and edge cases"""
    
    def test_missing_input_file(self, temp_dir):
        """Test error handling for missing input file"""
        nonexistent_input = temp_dir / "does_not_exist.jpg"
        output_path = temp_dir / "output.jpg"
        
        config = GainMapPipelineConfig()
        
        with pytest.raises(GainMapPipelineError, match="Input image not found"):
            run_gainmap_pipeline(
                img_path=str(nonexistent_input),
                out_path=str(output_path),
                config=config
            )
    
    def test_invalid_output_directory(self, synthetic_sdr_image, temp_dir):
        """Test error handling for invalid output directory"""
        # Create a file where we want a directory, making it impossible to create the path
        blocking_file = temp_dir / "blocking_file"
        blocking_file.write_text("block")
        
        # Try to write inside the file (impossible)
        invalid_output = str(blocking_file / "nested" / "output.jpg")
        
        config = GainMapPipelineConfig()
        
        with pytest.raises((GainMapPipelineError, NotADirectoryError, OSError)):
            run_gainmap_pipeline(
                img_path=str(synthetic_sdr_image),
                out_path=invalid_output,
                config=config
            )
    
    def test_unsupported_input_format(self, temp_dir):
        """Test error handling for unsupported input formats"""
        # Create a text file with wrong extension
        invalid_input = temp_dir / "not_an_image.jpg"
        invalid_input.write_text("This is not an image")
        
        output_path = temp_dir / "output.jpg"
        config = GainMapPipelineConfig()
        
        with pytest.raises(GainMapPipelineError):
            run_gainmap_pipeline(
                img_path=str(invalid_input),
                out_path=str(output_path),
                config=config
            )
    
    def test_corrupted_image_input(self, temp_dir):
        """Test error handling for corrupted image files"""
        # Create a file that looks like JPEG but is corrupted
        corrupted_input = temp_dir / "corrupted.jpg"
        corrupted_input.write_bytes(b"\\xff\\xd8\\xff\\xe0" + b"\\x00" * 100)  # Invalid JPEG
        
        output_path = temp_dir / "output.jpg"
        config = GainMapPipelineConfig()
        
        with pytest.raises(GainMapPipelineError):
            run_gainmap_pipeline(
                img_path=str(corrupted_input),
                out_path=str(output_path),
                config=config
            )


class TestMetadataPreservation:
    """Test that pipeline preserves important image metadata"""
    
    def test_dimensions_preserved(self, temp_dir):
        """Test that output dimensions match input"""
        from PIL import Image
        
        # Create input with specific dimensions
        width, height = 640, 480
        test_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        input_path = temp_dir / "input.jpg"
        output_path = temp_dir / "output.jpg"
        
        Image.fromarray(test_array).save(input_path, "JPEG")
        
        # Run pipeline
        config = GainMapPipelineConfig(model_type="synthetic")
        run_gainmap_pipeline(
            img_path=str(input_path),
            out_path=str(output_path),
            config=config
        )
        
        # Check output dimensions
        with Image.open(output_path) as output_img:
            assert output_img.size == (width, height)
    
    def test_colorspace_handling(self, temp_dir):
        """Test proper colorspace handling in pipeline"""
        from PIL import Image
        
        # Create sRGB input image
        test_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        input_path = temp_dir / "input.jpg"
        output_path = temp_dir / "output.jpg" 
        
        # Save with sRGB profile
        input_img = Image.fromarray(test_array)
        input_img.save(input_path, "JPEG")
        
        # Run pipeline
        config = GainMapPipelineConfig(model_type="synthetic")
        run_gainmap_pipeline(
            img_path=str(input_path),
            out_path=str(output_path),
            config=config
        )
        
        # Output should be valid and processable
        with Image.open(output_path) as output_img:
            assert output_img.mode == "RGB"
            # Should be able to convert to array without error
            output_array = np.array(output_img)
            assert output_array.shape == (300, 400, 3)


class TestValidationFunction:
    """Test Ultra HDR file structure validation"""
    
    def test_validate_reference_hdr(self, reference_hdr_image):
        """Test validation against iPhone HDR reference"""
        result = validate_ultrahdr_structure(str(reference_hdr_image))
        
        assert result["valid"] is True
        assert result["metadata"]["has_mpf"] is True
        assert result["metadata"]["file_size"] > 100000  # Should be substantial file
        assert result["metadata"]["jpeg_count"] >= 2     # SDR + gain map
        assert len(result["errors"]) == 0
    
    def test_validate_regular_jpeg(self, sample_sdr_image):
        """Test that regular JPEG fails Ultra HDR validation"""
        result = validate_ultrahdr_structure(str(sample_sdr_image))
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        # Should report missing Ultra HDR features
        error_text = " ".join(result["errors"])
        assert "MPF" in error_text or "Ultra HDR" in error_text
    
    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file"""
        result = validate_ultrahdr_structure("/path/that/does/not/exist.jpg")
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "does not exist" in result["errors"][0].lower()


class TestPipelinePerformance:
    """Test pipeline performance characteristics"""
    
    @pytest.mark.slow
    def test_batch_processing_memory(self, temp_dir):
        """Test that batch processing doesn't leak memory"""
        from PIL import Image
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        config = GainMapPipelineConfig(model_type="synthetic")
        
        # Process multiple images
        for i in range(5):
            # Create test image
            test_array = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
            input_path = temp_dir / f"batch_input_{i}.jpg"
            output_path = temp_dir / f"batch_output_{i}.jpg"
            
            Image.fromarray(test_array).save(input_path, "JPEG")
            
            # Run pipeline
            run_gainmap_pipeline(
                img_path=str(input_path),
                out_path=str(output_path),
                config=config
            )
            
            assert output_path.exists()
        
        # Memory should not have grown excessively (allow 50MB growth)
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        assert memory_growth < 50 * 1024 * 1024  # 50MB limit
    
    def test_processing_time_reasonable(self, synthetic_sdr_image, temp_dir):
        """Test that processing time is reasonable for typical images"""
        import time
        
        output_path = temp_dir / "timing_test.jpg"
        config = GainMapPipelineConfig(model_type="synthetic")
        
        start_time = time.time()
        
        run_gainmap_pipeline(
            img_path=str(synthetic_sdr_image),
            out_path=str(output_path),
            config=config
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (synthetic model should be fast)
        assert processing_time < 10.0  # 10 seconds max for synthetic model
        assert output_path.exists()