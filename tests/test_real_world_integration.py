"""Real-world integration tests using actual user-uploaded images

These tests use the ACTUAL images from the images/ directory to validate
that our test suite catches real-world usage patterns.
"""

import pytest
import requests
import time
import subprocess
import atexit
from pathlib import Path
from PIL import Image


class TestRealWorldWebIntegration:
    """Test web interface with real production example images"""
    
    @pytest.fixture(scope="class")
    def real_world_server(self):
        """Start web server for real-world testing"""
        # Use different port to avoid conflicts
        process = subprocess.Popen([
            "uv", "run", "uvicorn", "hdr_web.app:app", 
            "--host", "127.0.0.1", "--port", "8006"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        base_url = "http://127.0.0.1:8006"
        for _ in range(30):
            try:
                response = requests.get(base_url, timeout=2)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            process.terminate()
            raise RuntimeError("Real-world test server failed to start")
        
        # Register cleanup
        def cleanup():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        atexit.register(cleanup)
        
        return base_url
    
    @pytest.mark.integration
    def test_real_sdr_image_processing(self, real_world_server):
        """Test processing of actual SDR image from production examples"""
        # Use the actual SDR image that users would upload
        sdr_image_path = Path("images/02_edited_sdr.jpeg")
        
        if not sdr_image_path.exists():
            pytest.skip("Production SDR example not available")
        
        # Test the complete workflow
        with open(sdr_image_path, 'rb') as f:
            files = {'file': ('user_image.jpg', f, 'image/jpeg')}
            data = {'strength': '1.0'}
            
            # This should match exactly what a user does
            response = requests.post(f"{real_world_server}/process", files=files, data=data)
        
        # Should succeed without the crashes we saw in manual testing
        assert response.status_code in [200, 201], f"Real SDR processing failed: {response.status_code} - {response.text}"
        
        # Response should be substantial (not empty)
        assert len(response.content) > 50000, f"Response too small: {len(response.content)} bytes"
        
        # Should be valid HDR JPEG
        temp_output = Path("test_real_world_output.jpg")
        try:
            with open(temp_output, 'wb') as f:
                f.write(response.content)
            
            # Validate it's actually HDR
            from hdr.gainmap_pipeline import validate_ultrahdr_structure
            validation = validate_ultrahdr_structure(str(temp_output))
            
            assert validation['valid'], f"Real-world output not valid HDR: {validation['errors']}"
            
        finally:
            if temp_output.exists():
                temp_output.unlink()
    
    @pytest.mark.integration
    def test_preview_endpoint_with_real_image(self, real_world_server):
        """Test preview endpoint with real image (this was crashing)"""
        sdr_image_path = Path("images/02_edited_sdr.jpeg")
        
        if not sdr_image_path.exists():
            pytest.skip("Production SDR example not available")
        
        # Test preview generation (this was the failing endpoint)
        with open(sdr_image_path, 'rb') as f:
            files = {'file': ('user_image.jpg', f, 'image/jpeg')}
            data = {'strength': '1.5'}  # Test with different strength
            
            response = requests.post(f"{real_world_server}/preview", files=files, data=data)
        
        # This was returning 500 before our fixes
        assert response.status_code == 200, f"Preview endpoint failed: {response.status_code} - {response.text}"
        
        # Should return a valid image
        assert response.headers.get('content-type', '').startswith('image/'), "Preview not returning image"
        assert len(response.content) > 1000, f"Preview image too small: {len(response.content)} bytes"
    
    @pytest.mark.integration  
    def test_gmnet_with_real_image(self, real_world_server):
        """Test GMNet model with real-world image (not synthetic)"""
        from hdr.models import GMNetModel
        
        model = GMNetModel()
        if not model.is_available():
            pytest.skip("GMNet model not available")
        
        sdr_image_path = Path("images/02_edited_sdr.jpeg")
        if not sdr_image_path.exists():
            pytest.skip("Production SDR example not available")
        
        # Load actual image
        with Image.open(sdr_image_path) as img:
            sdr_array = img.array()
        
        # Test direct model inference
        sdr_rgb = sdr_array.astype(np.float32) / 255.0
        
        try:
            prediction = model.predict(sdr_rgb)
            
            # Validate prediction quality
            assert prediction.gain_map.shape == sdr_rgb.shape[:2], "Gain map size mismatch"
            assert 0 <= prediction.gain_min_log2 <= prediction.gain_max_log2, "Invalid gain range"
            assert prediction.confidence > 0.5, f"Low model confidence: {prediction.confidence}"
            
            # Gain map should have reasonable characteristics for real image
            gain_std = np.std(prediction.gain_map)
            assert gain_std > 1, f"Gain map too uniform for real image: std={gain_std}"
            
        except Exception as e:
            pytest.fail(f"GMNet failed on real-world image: {e}")


class TestRealWorldRegressionPrevention:
    """Test suite designed to catch real-world usage issues our synthetic tests miss"""
    
    def test_file_size_validation_realistic(self):
        """Test that our size validation logic works with real HDR files"""
        from hdr.gainmap_pipeline import run_gainmap_pipeline, GainMapPipelineConfig
        import tempfile
        
        sdr_path = Path("images/02_edited_sdr.jpeg")
        if not sdr_path.exists():
            pytest.skip("Real SDR image not available")
        
        # Test both synthetic and GMNet (if available) with real image
        models_to_test = ["synthetic"]
        
        from hdr.models import GMNetModel
        if GMNetModel().is_available():
            models_to_test.append("gmnet")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for model_type in models_to_test:
                output_path = temp_path / f"real_world_{model_type}.jpg"
                
                config = GainMapPipelineConfig(
                    model_type=model_type,
                    export_quality=95
                )
                
                # This should work without size validation failures
                try:
                    result = run_gainmap_pipeline(
                        img_path=str(sdr_path),
                        out_path=str(output_path),
                        config=config
                    )
                    
                    assert output_path.exists(), f"Output not created for {model_type}"
                    
                    # Validate with our existing validation
                    from hdr.gainmap_pipeline import validate_ultrahdr_structure
                    validation = validate_ultrahdr_structure(str(output_path))
                    assert validation['valid'], f"{model_type} output not valid HDR: {validation['errors']}"
                    
                except Exception as e:
                    pytest.fail(f"Real-world processing failed with {model_type}: {e}")
    
    def test_web_interface_error_paths(self):
        """Test error paths that only occur with real user behavior"""
        # This would test:
        # - Very large files (near the limit)
        # - Unusual aspect ratios  
        # - Files with complex EXIF data
        # - Images that compress differently than expected
        
        # For now, document what should be tested
        error_scenarios = [
            "Very large files (near 10MB limit)",
            "Unusual aspect ratios (very wide/tall)",
            "Files with complex EXIF metadata", 
            "Images that compress unexpectedly",
            "Files with embedded color profiles",
            "Progressive JPEG vs baseline JPEG"
        ]
        
        pytest.skip(f"Real-world error testing not yet implemented. Should test: {error_scenarios}")
    
    def test_preview_performance_with_real_images(self):
        """Test that preview generation is fast enough with real images"""
        sdr_path = Path("images/02_edited_sdr.jpeg") 
        
        if not sdr_path.exists():
            pytest.skip("Real SDR image not available")
        
        from hdr_web.utils.preview import create_hdr_preview
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "preview_test.jpg"
            
            # Time the preview generation
            start_time = time.time()
            
            try:
                create_hdr_preview(
                    str(sdr_path),
                    str(output_path), 
                    strength=1.0
                )
                
                processing_time = time.time() - start_time
                
                # Preview should be fast for real-time UI
                assert processing_time < 5.0, f"Preview too slow: {processing_time:.2f}s"
                assert output_path.exists(), "Preview not created"
                
                # Should be a reasonable size for web preview
                preview_size = output_path.stat().st_size
                assert 5000 < preview_size < 2000000, f"Preview size unreasonable: {preview_size} bytes"
                
            except Exception as e:
                pytest.fail(f"Preview generation failed with real image: {e}")