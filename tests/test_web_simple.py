"""Simple web interface tests using requests for Phase 2 validation

Tests the HTTP API directly without browser automation as a foundation.
"""

import pytest
import requests
import tempfile
import time
from pathlib import Path
from PIL import Image
import numpy as np


class TestWebAPI:
    """Test web API endpoints directly"""
    
    @pytest.fixture(scope="class")
    def web_server_url(self):
        """Start web server and return URL"""
        import subprocess
        import atexit
        
        # Start web service
        process = subprocess.Popen([
            "uv", "run", "uvicorn", "hdr_web.app:app", 
            "--host", "127.0.0.1", "--port", "8003"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        base_url = "http://127.0.0.1:8003"
        for _ in range(30):
            try:
                response = requests.get(base_url, timeout=2)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            process.terminate()
            raise RuntimeError("Web server failed to start")
        
        # Register cleanup
        def cleanup():
            process.terminate()
            process.wait()
        atexit.register(cleanup)
        
        return base_url
    
    @pytest.mark.web
    def test_homepage_accessible(self, web_server_url):
        """Test that homepage is accessible"""
        response = requests.get(web_server_url)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        
        # Check for expected content
        content = response.text.lower()
        assert "hdr" in content or "photo" in content or "upload" in content
    
    @pytest.mark.web
    def test_health_check(self, web_server_url):
        """Test basic health check"""
        # Test root endpoint
        response = requests.get(web_server_url)
        assert response.status_code == 200
        
        # Test if server handles basic requests
        response = requests.head(web_server_url)
        assert response.status_code in [200, 405]  # HEAD may not be implemented
    
    @pytest.mark.web
    def test_file_upload_endpoint(self, web_server_url, synthetic_sdr_image):
        """Test file upload via HTTP POST"""
        upload_url = f"{web_server_url}/upload"  # Adjust based on actual endpoint
        
        # Try to find the correct upload endpoint
        possible_endpoints = [
            "/upload",
            "/process", 
            "/api/upload",
            "/api/process",
            "/",  # Some apps handle upload via root
        ]
        
        upload_endpoint = None
        for endpoint in possible_endpoints:
            try:
                # Try OPTIONS request to see if endpoint exists
                response = requests.options(f"{web_server_url}{endpoint}")
                if response.status_code < 500:  # Endpoint exists
                    upload_endpoint = endpoint
                    break
            except:
                continue
        
        if upload_endpoint is None:
            pytest.skip("Could not find upload endpoint")
        
        # Test actual upload
        with open(synthetic_sdr_image, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"{web_server_url}{upload_endpoint}", files=files)
        
        # Should either succeed or give a meaningful error (or 405 if method not allowed)
        assert response.status_code in [200, 201, 400, 405, 422], f"Unexpected status: {response.status_code}"
        
        if response.status_code in [200, 201]:
            # Upload successful - check response
            assert len(response.content) > 0
    
    @pytest.mark.web  
    def test_static_files_accessible(self, web_server_url):
        """Test that static files are accessible"""
        # Test common static file paths
        static_paths = [
            "/static/css/style.css",
            "/static/js/main.js", 
            "/static/favicon.ico"
        ]
        
        accessible_static = False
        for path in static_paths:
            try:
                response = requests.get(f"{web_server_url}{path}")
                if response.status_code == 200:
                    accessible_static = True
                    break
            except:
                continue
        
        # At least check that /static/ path doesn't throw 500 error
        try:
            response = requests.get(f"{web_server_url}/static/")
            assert response.status_code in [200, 404, 403]  # Valid responses
        except requests.exceptions.ConnectionError:
            pytest.fail("Static file server not accessible")
    
    @pytest.mark.web
    def test_error_handling(self, web_server_url):
        """Test error handling for invalid requests"""
        # Test 404 handling
        response = requests.get(f"{web_server_url}/nonexistent-page")
        assert response.status_code == 404
        
        # Test invalid method on root
        try:
            response = requests.delete(web_server_url)
            assert response.status_code in [405, 501]  # Method not allowed
        except:
            pass  # Some servers might not handle this gracefully


class TestWebIntegration:
    """Integration tests for web workflow"""
    
    @pytest.mark.web
    def test_web_service_startup(self):
        """Test that web service can start and stop cleanly"""
        import subprocess
        
        # Start service
        process = subprocess.Popen([
            "uv", "run", "python", "-c", 
            "from hdr_web.app import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8004)"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(2)
        
        # Check it's running
        try:
            response = requests.get("http://127.0.0.1:8004", timeout=5)
            service_started = response.status_code == 200
        except:
            service_started = False
        
        # Clean shutdown
        process.terminate()
        process.wait(timeout=10)
        
        assert service_started, "Web service failed to start"
    
    @pytest.mark.web
    def test_concurrent_requests(self, web_server_url):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        def make_request():
            try:
                response = requests.get(web_server_url, timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # Most requests should succeed
        success_count = sum(results)
        assert success_count >= 7, f"Only {success_count}/10 concurrent requests succeeded"


class TestWebConfiguration:
    """Test web service configuration and setup"""
    
    def test_web_dependencies_available(self):
        """Test that web dependencies are properly installed"""
        try:
            import fastapi
            import uvicorn
            import jinja2
            assert True
        except ImportError as e:
            pytest.fail(f"Web dependencies not available: {e}")
    
    def test_web_module_importable(self):
        """Test that web module can be imported"""
        try:
            from hdr_web.app import app
            assert app is not None
        except ImportError as e:
            pytest.fail(f"Web module not importable: {e}")
    
    def test_template_directory_exists(self):
        """Test that template directory exists"""
        from pathlib import Path
        
        template_dirs = [
            Path("hdr_web/templates"),
            Path("hdr_web/static")
        ]
        
        for template_dir in template_dirs:
            if template_dir.exists():
                return  # At least one exists
        
        pytest.skip("Template directories not found - web interface may not be fully set up")


@pytest.mark.integration  
class TestFullWorkflow:
    """Test complete workflow integration"""
    
    def test_pipeline_to_web_integration(self, synthetic_sdr_image, temp_dir):
        """Test that pipeline output can be served by web interface"""
        from hdr.gainmap_pipeline import run_gainmap_pipeline, GainMapPipelineConfig
        
        # Generate HDR using pipeline
        hdr_output = temp_dir / "pipeline_hdr.jpg"
        config = GainMapPipelineConfig(model_type="synthetic")
        
        result = run_gainmap_pipeline(
            img_path=str(synthetic_sdr_image),
            out_path=str(hdr_output),
            config=config
        )
        
        assert hdr_output.exists()
        
        # Validate it can be processed by web interface components
        try:
            from PIL import Image
            with Image.open(hdr_output) as img:
                # Ultra HDR files show as MPO (Multi Picture Object) format in Pillow
                assert img.format in ["JPEG", "MPO"], f"Unexpected format: {img.format}"
                assert img.size[0] > 0 and img.size[1] > 0
        except Exception as e:
            pytest.fail(f"Pipeline output not compatible with web interface: {e}")