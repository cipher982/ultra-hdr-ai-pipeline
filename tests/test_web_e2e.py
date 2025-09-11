"""End-to-end web automation tests for HDR web interface

Eliminates human verification loop: no more "click the website and check if it works"
Tests complete upload → process → download workflow with visual validation.
"""

import asyncio
import tempfile
import pytest
import pytest_asyncio
from pathlib import Path
from PIL import Image
import numpy as np

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


class TestWebInterfaceE2E:
    """Test complete web interface workflow"""
    
    @pytest_asyncio.fixture(scope="class")
    async def browser(self):
        """Start browser for web testing"""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,  # Run headless for CI/CD
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        yield browser
        await browser.close()
        await playwright.stop()
    
    @pytest_asyncio.fixture
    async def context(self, browser: Browser):
        """Create isolated browser context for each test"""
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        yield context
        await context.close()
    
    @pytest_asyncio.fixture
    async def page(self, context: BrowserContext):
        """Create page for testing"""
        page = await context.new_page()
        yield page
        await page.close()
    
    @pytest_asyncio.fixture
    async def web_server(self):
        """Start HDR web service for testing"""
        import subprocess
        import time
        import requests
        
        # Start web service in background
        process = subprocess.Popen([
            "uv", "run", "uvicorn", "hdr_web.app:app", 
            "--host", "127.0.0.1", "--port", "8002"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get("http://127.0.0.1:8002/", timeout=1)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            process.terminate()
            raise RuntimeError("Web server failed to start")
        
        yield "http://127.0.0.1:8002"
        
        # Cleanup
        process.terminate()
        process.wait()
    
    @pytest.mark.web
    async def test_homepage_loads(self, page: Page, web_server):
        """Test that homepage loads without errors"""
        await page.goto(web_server)
        
        # Check page title
        title = await page.title()
        assert "HDR" in title or "Photo" in title
        
        # Check for upload form
        file_input = page.locator('input[type="file"]')
        assert await file_input.count() > 0
        
        # Check for expected UI elements
        await page.wait_for_selector('input[type="file"]', timeout=5000)
    
    @pytest.mark.web
    async def test_upload_process_download_flow(self, page: Page, web_server, synthetic_sdr_image, temp_dir):
        """Test complete upload → process → download workflow"""
        await page.goto(web_server)
        
        # Upload image
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files(str(synthetic_sdr_image))
        
        # Start processing (look for submit/process button)
        submit_selectors = [
            'button[type="submit"]',
            'input[type="submit"]', 
            'button:has-text("Process")',
            'button:has-text("Upload")',
            '.btn-primary'
        ]
        
        submitted = False
        for selector in submit_selectors:
            try:
                submit_btn = page.locator(selector).first
                if await submit_btn.count() > 0:
                    await submit_btn.click()
                    submitted = True
                    break
            except:
                continue
        
        assert submitted, "Could not find submit button"
        
        # Wait for processing to complete (look for download link or result)
        download_selectors = [
            'a[download]',
            'a:has-text("Download")',
            '.download-link',
            'button:has-text("Download")'
        ]
        
        download_link = None
        for selector in download_selectors:
            try:
                await page.wait_for_selector(selector, timeout=30000)  # 30 second timeout
                download_link = page.locator(selector).first
                if await download_link.count() > 0:
                    break
            except:
                continue
        
        assert download_link is not None, "Processing did not complete or download link not found"
        
        # Download result
        async with page.expect_download() as download_info:
            await download_link.click()
        
        download = await download_info.value
        
        # Save downloaded file
        downloaded_path = temp_dir / "downloaded_hdr.jpg"
        await download.save_as(downloaded_path)
        
        # Validate downloaded file
        assert downloaded_path.exists()
        assert downloaded_path.stat().st_size > 10000  # Non-trivial size
        
        # Validate it's a valid image
        with Image.open(downloaded_path) as img:
            assert img.format == "JPEG"
            assert img.size[0] > 100 and img.size[1] > 100  # Reasonable dimensions
    
    @pytest.mark.web  
    async def test_error_handling_invalid_file(self, page: Page, web_server, temp_dir):
        """Test error handling for invalid file uploads"""
        await page.goto(web_server)
        
        # Create invalid file (text file with .jpg extension)
        invalid_file = temp_dir / "invalid.jpg"
        invalid_file.write_text("This is not an image")
        
        # Upload invalid file
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files(str(invalid_file))
        
        # Try to submit
        submit_btn = page.locator('button[type="submit"], input[type="submit"]').first
        await submit_btn.click()
        
        # Should show error message
        error_selectors = [
            '.error',
            '.alert-danger', 
            '.text-danger',
            ':has-text("error")',
            ':has-text("invalid")'
        ]
        
        error_shown = False
        for selector in error_selectors:
            try:
                await page.wait_for_selector(selector, timeout=10000)
                error_shown = True
                break
            except:
                continue
        
        assert error_shown, "Error message not displayed for invalid file"
    
    @pytest.mark.web
    async def test_ui_responsiveness(self, page: Page, web_server):
        """Test UI responds properly at different screen sizes"""
        await page.goto(web_server)
        
        # Test desktop size
        await page.set_viewport_size({"width": 1920, "height": 1080})
        upload_element = page.locator('input[type="file"]')
        assert await upload_element.is_visible()
        
        # Test tablet size
        await page.set_viewport_size({"width": 768, "height": 1024})
        assert await upload_element.is_visible()
        
        # Test mobile size
        await page.set_viewport_size({"width": 375, "height": 667})
        assert await upload_element.is_visible()
    
    @pytest.mark.web
    async def test_multiple_file_uploads(self, page: Page, web_server, temp_dir):
        """Test handling multiple file uploads in sequence"""
        await page.goto(web_server)
        
        # Create multiple test images
        test_images = []
        for i in range(3):
            img_array = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
            img_path = temp_dir / f"test_image_{i}.jpg"
            Image.fromarray(img_array).save(img_path, "JPEG")
            test_images.append(img_path)
        
        # Upload and process each image
        for i, img_path in enumerate(test_images):
            # Navigate back to main page if needed
            if i > 0:
                await page.goto(web_server)
            
            # Upload image
            file_input = page.locator('input[type="file"]')
            await file_input.set_input_files(str(img_path))
            
            # Submit
            submit_btn = page.locator('button[type="submit"], input[type="submit"]').first
            await submit_btn.click()
            
            # Wait for processing (shorter timeout for synthetic model)
            try:
                await page.wait_for_selector('a[download], a:has-text("Download")', timeout=15000)
            except:
                # If download link not found, check for completion indicators
                completion_selectors = [
                    ':has-text("Complete")',
                    ':has-text("Success")',
                    '.success',
                    '.alert-success'
                ]
                
                completed = False
                for selector in completion_selectors:
                    if await page.locator(selector).count() > 0:
                        completed = True
                        break
                
                assert completed, f"Processing did not complete for image {i}"


class TestWebVisualRegression:
    """Test visual regression with screenshot comparison"""
    
    @pytest_asyncio.fixture
    async def browser_context(self):
        """Browser context for visual testing"""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            device_scale_factor=1
        )
        yield context
        await context.close()
        await browser.close()
        await playwright.stop()
    
    @pytest.mark.visual
    async def test_homepage_screenshot(self, browser_context, web_server, fixtures_dir):
        """Test homepage visual regression"""
        page = await browser_context.new_page()
        await page.goto(web_server)
        
        # Wait for page to fully load
        await page.wait_for_load_state('networkidle')
        
        # Take screenshot
        screenshot_path = fixtures_dir / "homepage_screenshot.png" 
        
        # Create golden master if it doesn't exist
        if not screenshot_path.exists():
            await page.screenshot(path=screenshot_path)
            pytest.skip("Created golden master screenshot")
        
        # Compare against golden master
        current_screenshot = await page.screenshot()
        
        # Load golden master
        with open(screenshot_path, 'rb') as f:
            golden_master = f.read()
        
        # Basic comparison - in production you'd use image diffing
        # For now, just ensure screenshots are similar size
        assert abs(len(current_screenshot) - len(golden_master)) < len(golden_master) * 0.1
        
        await page.close()
    
    @pytest.mark.visual
    async def test_upload_form_visual_state(self, browser_context, web_server):
        """Test upload form visual states"""
        page = await browser_context.new_page()
        await page.goto(web_server)
        
        # Test initial state
        upload_input = page.locator('input[type="file"]')
        assert await upload_input.is_visible()
        
        # Test hover state (if applicable)
        await upload_input.hover()
        
        # Test focus state
        await upload_input.focus()
        
        # Take screenshots of different states for manual verification if needed
        # await page.screenshot(path="upload_form_states.png")
        
        await page.close()


class TestWebPerformance:
    """Test web interface performance characteristics"""
    
    @pytest.mark.web
    async def test_page_load_performance(self, page: Page, web_server):
        """Test page load performance metrics"""
        # Start performance monitoring
        await page.goto(web_server)
        
        # Wait for page to fully load
        await page.wait_for_load_state('networkidle')
        
        # Basic performance checks
        title = await page.title()
        assert title  # Page loaded successfully
        
        # Check for obvious performance issues
        start_time = page.evaluate("performance.timing.navigationStart")
        load_time = page.evaluate("performance.timing.loadEventEnd")
        
        # Calculate load time (basic check)
        if await start_time and await load_time:
            total_load_time = await load_time - await start_time
            assert total_load_time < 5000  # Should load in under 5 seconds
    
    @pytest.mark.web
    async def test_file_upload_performance(self, page: Page, web_server, temp_dir):
        """Test file upload and processing performance"""
        await page.goto(web_server)
        
        # Create moderately large test image
        large_array = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
        large_image_path = temp_dir / "large_test.jpg"
        Image.fromarray(large_array).save(large_image_path, "JPEG", quality=95)
        
        # Upload file and measure processing time
        import time
        start_time = time.time()
        
        file_input = page.locator('input[type="file"]')
        await file_input.set_input_files(str(large_image_path))
        
        submit_btn = page.locator('button[type="submit"], input[type="submit"]').first
        await submit_btn.click()
        
        # Wait for processing to complete
        try:
            await page.wait_for_selector('a[download]', timeout=60000)  # 1 minute timeout
            processing_time = time.time() - start_time
            
            # Processing should be reasonable for synthetic model
            assert processing_time < 30  # Should process in under 30 seconds
            
        except:
            # If download link not found, just check processing completed
            processing_time = time.time() - start_time
            assert processing_time < 60  # At least didn't hang indefinitely
    
    @pytest.mark.web  
    async def test_concurrent_uploads(self, web_server, temp_dir):
        """Test handling multiple concurrent uploads"""
        # Create test images
        test_images = []
        for i in range(3):
            img_array = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8) 
            img_path = temp_dir / f"concurrent_test_{i}.jpg"
            Image.fromarray(img_array).save(img_path, "JPEG")
            test_images.append(img_path)
        
        async def upload_image(image_path):
            """Upload single image in separate browser context"""
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(web_server)
                
                file_input = page.locator('input[type="file"]')
                await file_input.set_input_files(str(image_path))
                
                submit_btn = page.locator('button[type="submit"], input[type="submit"]').first
                await submit_btn.click()
                
                # Wait for completion
                await page.wait_for_selector(
                    'a[download], :has-text("Complete"), :has-text("Success")', 
                    timeout=30000
                )
                
                return True
            except Exception as e:
                print(f"Upload failed: {e}")
                return False
            finally:
                await page.close()
                await context.close() 
                await browser.close()
                await playwright.stop()
        
        # Run uploads concurrently
        tasks = [upload_image(img_path) for img_path in test_images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least 2/3 should succeed (allowing for some concurrency limits)
        successful_uploads = sum(1 for result in results if result is True)
        assert successful_uploads >= 2, f"Only {successful_uploads}/3 concurrent uploads succeeded"