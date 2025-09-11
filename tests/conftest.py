"""Pytest configuration and shared fixtures for HDR pipeline tests"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import pytest
import numpy as np
from PIL import Image


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent.parent / "images"


@pytest.fixture(scope="session") 
def golden_masters_dir() -> Path:
    """Path to golden master reference files"""
    return Path(__file__).parent / "golden_masters"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to test fixtures"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test outputs, cleaned up after test"""
    temp_path = Path(tempfile.mkdtemp(prefix="hdr_test_"))
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def sample_sdr_image(test_data_dir: Path) -> Path:
    """Path to sample SDR JPEG for testing"""
    sdr_path = test_data_dir / "02_edited_sdr.jpeg"
    if not sdr_path.exists():
        pytest.skip(f"Sample SDR image not found: {sdr_path}")
    return sdr_path


@pytest.fixture(scope="session")
def reference_hdr_image(test_data_dir: Path) -> Path:
    """Path to iPhone HDR reference image"""
    hdr_path = test_data_dir / "01_original_iphone_hdr.jpeg"
    if not hdr_path.exists():
        pytest.skip(f"Reference HDR image not found: {hdr_path}")
    return hdr_path


@pytest.fixture(scope="session") 
def example_hdr_output(test_data_dir: Path) -> Path:
    """Path to example pipeline HDR output"""
    output_path = test_data_dir / "example_hdr_enhanced.jpg"
    if not output_path.exists():
        pytest.skip(f"Example HDR output not found: {output_path}")
    return output_path


@pytest.fixture
def synthetic_sdr_image(temp_dir: Path) -> Path:
    """Generate synthetic SDR test image"""
    # Create a test image with interesting HDR characteristics
    # High contrast scene with bright highlights and dark shadows
    width, height = 800, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient with bright highlights
    for y in range(height):
        for x in range(width):
            # Bright sky area
            if y < height // 3:
                image[y, x] = [220 + (x % 35), 230 + (x % 25), 250]
            # Mid-tone area  
            elif y < 2 * height // 3:
                image[y, x] = [100 + (x % 50), 120 + (x % 40), 140 + (x % 30)]
            # Dark shadow area
            else:
                image[y, x] = [20 + (x % 15), 25 + (x % 12), 30 + (x % 10)]
    
    # Add some bright highlights that should benefit from HDR
    center_x, center_y = width // 2, height // 4
    for y in range(max(0, center_y - 50), min(height, center_y + 50)):
        for x in range(max(0, center_x - 50), min(width, center_x + 50)):
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if dist < 40:
                brightness = max(0, 255 - int(dist * 3))
                image[y, x] = [brightness, brightness, brightness]
    
    # Save as JPEG
    synthetic_path = temp_dir / "synthetic_sdr.jpg"
    Image.fromarray(image).save(synthetic_path, "JPEG", quality=95)
    return synthetic_path


@pytest.fixture
def mock_gmnet_available(monkeypatch):
    """Mock GMNet availability for testing without the actual model"""
    from hdr.models import GMNetModel
    
    def mock_is_available():
        return True
        
    def mock_predict(self, sdr_rgb):
        # Return a simple synthetic gain map
        h, w = sdr_rgb.shape[:2]
        # Create gain map with more gain in bright areas
        gain_map = np.ones((h, w), dtype=np.float32) * 128  # Neutral gain
        
        # Add higher gain to bright regions
        brightness = np.mean(sdr_rgb, axis=2)
        bright_mask = brightness > 0.7
        gain_map[bright_mask] = 200  # Higher gain for highlights
        
        from hdr.models import GainMapPrediction
        return GainMapPrediction(
            gain_map=gain_map.astype(np.uint8),
            gain_min_log2=0.0,
            gain_max_log2=2.68,  # iPhone reference headroom
            confidence=0.85
        )
    
    monkeypatch.setattr(GMNetModel, "is_available", mock_is_available)
    monkeypatch.setattr(GMNetModel, "predict", mock_predict)