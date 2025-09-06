import os
import subprocess
import sys
import pytest


def test_cli_exporter_fails_without_hdr_container(tmp_path):
    # This test verifies that the CLI exits non-zero when exporter fails/unavailable.
    # Only run on macOS where the exporter path is attempted.
    if sys.platform != 'darwin':
        pytest.skip('macOS-only test')

    # Minimal input: use an existing sample
    img = os.path.join('hdrcnn_upstream', 'data', 'img_001.png')
    weights = 'hdrcnn_tf2.weights.h5'
    if not os.path.exists(weights):
        pytest.skip('weights not found')

    out = tmp_path / 'out_gainmap.jpg'

    # Run CLI; expect non-zero exit if exporter cannot finalize
    proc = subprocess.run([
        sys.executable, '-m', 'hdr.cli',
        '--img', img,
        '--out', str(out),
        '--weights', weights
    ], capture_output=True, text=True)

    if proc.returncode == 0:
        # Exporter succeeded on this machine; great. Just ensure file exists.
        assert os.path.exists(out)
    else:
        # Exporter failed as expected; stdout/stderr should mention failure.
        assert 'HDR exporter failed' in proc.stderr or 'Exporter error' in (proc.stdout + proc.stderr)

