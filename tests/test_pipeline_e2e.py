import os
import io
import sys
import tempfile
import time
import numpy as np
from PIL import Image

import pytest

try:
    import tensorflow as tf  # noqa: F401
except Exception:  # pragma: no cover
    import pytest
    pytest.skip('TensorFlow not available; skip pipeline e2e tests', allow_module_level=True)

from hdr.pipeline import PipelineConfig, run_pipeline


def make_test_image(path: str, size=(321, 245)):
    w, h = size
    # Simple gradient with a bright spot to validate luminance increase
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.2 + 0.6 * (xx / max(1, w-1))
    spot = np.exp(-(((xx - w*0.7)/(w*0.1))**2 + ((yy - h*0.3)/(h*0.1))**2))
    img = np.clip(base + 0.8*spot, 0.0, 1.0)
    rgb = np.stack([img, img*0.9, img*0.8], axis=-1)
    Image.fromarray((rgb*255).astype(np.uint8)).save(path, quality=95, subsampling=0)


@pytest.mark.parametrize('size', [(321,245), (1024,768)])
def test_pipeline_outputs_and_dimensions(tmp_path, size):
    # Skip if weights are missing
    weights = 'hdrcnn_tf2.weights.h5'
    if not os.path.exists(weights):
        pytest.skip('weights not found')

    img = tmp_path / 'in.jpg'
    make_test_image(str(img), size=size)
    out_path = tmp_path / 'out_gainmap.jpg'

    cfg = PipelineConfig(max_side=4096, auto_exposure=True)
    t0 = time.time()
    out = run_pipeline(str(img), str(out_path), weights, cfg)
    dt = time.time() - t0

    # Files exist (SDR + HDR TIFF + gain map PNG)
    assert os.path.exists(out.sdr_preview_path)
    assert os.path.exists(out.hdr_tiff_path)
    assert os.path.exists(out.gainmap_png_path)

    # Dimensions preserved (after orientation, which is identity here)
    im_in = Image.open(img)
    im_sdr = Image.open(out.sdr_preview_path)
    assert im_in.size == im_sdr.size

    # Luminance sanity: SDR preview highlights should be >= input
    arr_in = np.asarray(im_in).astype(np.float32)/255.0
    arr_sdr = np.asarray(im_sdr).astype(np.float32)/255.0
    L_in = 0.2126*arr_in[...,0] + 0.7152*arr_in[...,1] + 0.0722*arr_in[...,2]
    L_sdr = 0.2126*arr_sdr[...,0] + 0.7152*arr_sdr[...,1] + 0.0722*arr_sdr[...,2]
    p_in = float(np.quantile(L_in, 0.99))
    p_sdr = float(np.quantile(L_sdr, 0.99))
    assert p_sdr >= p_in - 1e-3

    # Exporter may be disabled in tests; skip container checks if not produced
    if out.gainmap_jpeg_path and os.path.exists(out.gainmap_jpeg_path):
        # Basic sanity: container exists and is non-empty. Detailed metadata
        # checks are covered in exporter-specific tests.
        assert os.path.getsize(out.gainmap_jpeg_path) > 0


def test_exporter_timeout_fallback(tmp_path):
    # Create input
    weights = 'hdrcnn_tf2.weights.h5'
    if not os.path.exists(weights):
        pytest.skip('weights not found')
    img = tmp_path / 'in.jpg'
    make_test_image(str(img), size=(200, 200))
    out_path = tmp_path / 'out_gainmap.jpg'

    # Fake exporter that sleeps longer than timeout
    script = tmp_path / 'slow.sh'
    script.write_text('#!/bin/sh\nsleep 5\n')
    os.chmod(script, 0o755)
    os.environ['HDR_EXPORTER_BIN'] = str(script)

    cfg = PipelineConfig(timeout_s=1)
    out = run_pipeline(str(img), str(out_path), weights, cfg)
    # Exporter should time out and no HDR container should be produced.
    assert out.gainmap_jpeg_path is None
    assert os.path.exists(out.sdr_preview_path)
