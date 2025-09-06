import argparse
import os
import sys
import json
from .pipeline import PipelineConfig, run_pipeline


def main():
    ap = argparse.ArgumentParser(description='One-shot SDR→HDR gain-map JPEG pipeline')
    ap.add_argument('--img', required=True, help='Input SDR image path')
    ap.add_argument('--out', required=True, help='Output gain-map JPEG path')
    ap.add_argument('--weights', default='hdrcnn_tf2.weights.h5', help='Keras weights (hdrcnn_tf2)')
    ap.add_argument('--ref', default=None, help='Optional reference HDR image for metadata reuse (unused for now)')
    ap.add_argument('--tone', default='hable', choices=['hable','reinhard','gamma','none'])
    ap.add_argument('--gamma-trick', type=float, default=1.0)
    ap.add_argument('--ldr-gamma', type=float, default=2.2)
    ap.add_argument('--no-auto-exposure', dest='auto_exposure', action='store_false')
    ap.add_argument('--exposure-fstops', type=float, default=0.0)
    ap.add_argument('--max-side', type=int, default=4096)
    ap.add_argument('--timeout', type=int, default=20)
    args = ap.parse_args()

    cfg = PipelineConfig(
        max_side=args.max_side,
        gamma_trick=args.gamma_trick,
        ldr_gamma=args.ldr_gamma,
        tone=args.tone,
        auto_exposure=args.auto_exposure,
        exposure_fstops=args.exposure_fstops,
        timeout_s=args.timeout,
        export_enable=True,
        require_export_success=True,
    )

    print('[1/6] Load + orient + pad...')
    print('[2/6] Model inference (TF2/Keras)...')
    print('[3/6] Save HDR TIFF...')
    print('[4/6] Generate SDR preview...')
    print('[5/6] Compute gain map...')
    print('[6/6] Export gain‑map JPEG...')

    out = run_pipeline(args.img, args.out, args.weights, cfg, ref_path=args.ref)
    print('Done.')
    print(json.dumps({
        'sdr_preview': out.sdr_preview_path,
        'hdr_tiff': out.hdr_tiff_path,
        'gainmap_jpeg': out.gainmap_jpeg_path,
        'meta': out.meta_path,
    }, indent=2))


if __name__ == '__main__':
    main()
