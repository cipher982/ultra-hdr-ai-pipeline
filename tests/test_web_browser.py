"""Browser automation tests using Playwright with proper async setup

Full E2E testing of web interface: upload → process → download workflow
Eliminates "click the website and check if it works" manual verification.
"""

import asyncio
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import subprocess
import time
import requests
import signal
import os

# Import Playwright 
pytest_plugins = ('pytest_asyncio',)


class WebServerManager:
    """Manages web server lifecycle for testing"""
    
    def __init__(self, port: int = 8005):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.process = None
    
    async def start(self):
        """Start web server"""
        # Start web service in background
        self.process = subprocess.Popen([
            "uv", "run", "uvicorn", "hdr_web.app:app", 
            "--host", "127.0.0.1", "--port", str(self.port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get(self.base_url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                await asyncio.sleep(1)
        
        await self.stop()
        raise RuntimeError("Web server failed to start")
    
    async def stop(self):
        """Stop web server"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


@pytest.fixture(scope="session")
def web_server():
    """Session-level web server for all browser tests"""
    import atexit
    
    # Start web service
    process = subprocess.Popen([
        "uv", "run", "uvicorn", "hdr_web.app:app", 
        "--host", "127.0.0.1", "--port", "8005"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    base_url = "http://127.0.0.1:8005"
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
        process.wait(timeout=5)
    atexit.register(cleanup)
    
    return base_url


@pytest.mark.asyncio
class TestWebBrowserAutomation:
    """Browser automation tests with Playwright"""
    
    async def test_homepage_loads_in_browser(self, web_server):
        """Test that homepage loads properly in browser"""
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Navigate to homepage
                await page.goto(web_server)
                
                # Wait for page to load
                await page.wait_for_load_state('networkidle')
                
                # Check page title
                title = await page.title()
                assert title is not None and len(title) > 0
                
                # Check for file upload element (may be hidden)
                file_input = page.locator('input[type="file"]')
                assert await file_input.count() >= 1, "No file upload input found"
                
                # Check for upload zone (the clickable area)
                upload_zone = page.locator('.upload-zone, #upload-zone, [id*="upload"]')
                upload_zone_count = await upload_zone.count()
                
                # Either file input is visible OR upload zone is present
                file_input_visible = await file_input.is_visible()
                upload_zone_visible = upload_zone_count > 0 and await upload_zone.first.is_visible() if upload_zone_count > 0 else False
                
                assert file_input_visible or upload_zone_visible, "No visible upload interface found"
                
            finally:
                await context.close()
                await browser.close()
    
    async def test_file_upload_workflow(self, web_server, synthetic_sdr_image):
        """Test complete file upload and processing workflow"""
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(web_server)
                await page.wait_for_load_state('networkidle')
                
                # Upload file
                file_input = page.locator('input[type="file"]')
                await file_input.set_input_files(str(synthetic_sdr_image))
                
                # Look for and click submit button
                submit_selectors = [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:has-text("Process")',
                    'button:has-text("Upload")',
                    '.btn-primary',
                    'button'
                ]
                
                submitted = False
                for selector in submit_selectors:
                    submit_btn = page.locator(selector).first
                    if await submit_btn.count() > 0 and await submit_btn.is_visible():
                        await submit_btn.click()
                        submitted = True
                        break
                
                assert submitted, "Could not find or click submit button"
                
                # Wait for processing completion (longer timeout for HDR processing)
                completion_indicators = [
                    'a[download]',
                    'a:has-text("Download")',
                    ':has-text("Complete")',
                    ':has-text("Success")',
                    '.success'
                ]
                
                completed = False
                for selector in completion_indicators:
                    try:
                        await page.wait_for_selector(selector, timeout=45000)  # 45 second timeout
                        if await page.locator(selector).count() > 0:
                            completed = True
                            break
                    except:
                        continue
                
                assert completed, "Processing did not complete within timeout"
                
            finally:
                await context.close()
                await browser.close()
    
    async def test_error_handling_browser(self, web_server, temp_dir):
        """Test error handling for invalid files in browser"""
        from playwright.async_api import async_playwright
        
        # Create invalid file
        invalid_file = temp_dir / "invalid.jpg"
        invalid_file.write_text("This is not an image file")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await page.goto(web_server)
                await page.wait_for_load_state('networkidle')
                
                # Upload invalid file
                file_input = page.locator('input[type="file"]')
                await file_input.set_input_files(str(invalid_file))
                
                # Submit
                submit_btn = page.locator('button[type="submit"], input[type="submit"], button').first
                await submit_btn.click()
                
                # Should show error (give it time to process and show error)
                error_indicators = [
                    ':has-text("error")',
                    ':has-text("invalid")', 
                    ':has-text("failed")',
                    '.error',
                    '.alert-danger'
                ]
                
                error_shown = False
                for selector in error_indicators:
                    try:
                        await page.wait_for_selector(selector, timeout=15000)
                        if await page.locator(selector).count() > 0:
                            error_shown = True
                            break
                    except:
                        continue
                
                # If no explicit error shown, at least processing shouldn't succeed
                success_indicators = [
                    'a[download]',
                    ':has-text("Complete")',
                    ':has-text("Success")'
                ]
                
                success_shown = False
                for selector in success_indicators:
                    if await page.locator(selector).count() > 0:
                        success_shown = True
                        break
                
                # Either error shown OR success not shown (invalid file shouldn't succeed)
                assert error_shown or not success_shown, "Invalid file processing should fail"
                
            finally:
                await context.close()
                await browser.close()


@pytest.mark.asyncio
class TestWebVisualRegression:
    """Visual regression testing with screenshot comparison"""
    
    async def test_homepage_visual_consistency(self, web_server, fixtures_dir):
        """Test homepage visual consistency with screenshot comparison"""
        from playwright.async_api import async_playwright
        
        fixtures_dir.mkdir(exist_ok=True)
        screenshot_path = fixtures_dir / "homepage_screenshot.png"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 720},
                device_scale_factor=1
            )
            page = await context.new_page()
            
            try:
                await page.goto(web_server)
                await page.wait_for_load_state('networkidle')
                
                # Take current screenshot
                current_screenshot = await page.screenshot()
                
                if not screenshot_path.exists():
                    # Create baseline
                    with open(screenshot_path, 'wb') as f:
                        f.write(current_screenshot)
                    pytest.skip("Created baseline screenshot for future regression testing")
                
                # Load baseline
                with open(screenshot_path, 'rb') as f:
                    baseline_screenshot = f.read()
                
                # Basic size comparison (exact pixel comparison would be too brittle)
                size_diff_ratio = abs(len(current_screenshot) - len(baseline_screenshot)) / len(baseline_screenshot)
                
                assert size_diff_ratio < 0.2, f"Screenshot size changed significantly: {size_diff_ratio:.2f}"
                
            finally:
                await context.close()
                await browser.close()


@pytest.mark.asyncio  
class TestWebPerformance:
    """Test web interface performance with browser automation"""
    
    async def test_page_load_performance(self, web_server):
        """Test page load performance metrics"""
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Measure page load time
                start_time = time.time()
                await page.goto(web_server)
                await page.wait_for_load_state('networkidle')
                load_time = time.time() - start_time
                
                # Should load reasonably quickly
                assert load_time < 10, f"Page load too slow: {load_time:.2f}s"
                
                # Page should be functional
                title = await page.title()
                assert title is not None
                
                # Essential elements should be present
                file_input = page.locator('input[type="file"]')
                assert await file_input.count() > 0, "File input missing"
                
            finally:
                await context.close()
                await browser.close()