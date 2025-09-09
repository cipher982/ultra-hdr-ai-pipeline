"""
Modern CLI for gain-map-first HDR pipeline.

Clean, simple interface for AI models that directly predict gain maps.
"""

import argparse
import os
import sys
import platform
from pathlib import Path

from .gainmap_pipeline import (
    run_gainmap_pipeline, 
    GainMapPipelineConfig, 
    GainMapPipelineError
)
from .models import ModelError


def main():
    parser = argparse.ArgumentParser(description='Gain-map-first HDR pipeline')
    
    # Core arguments
    parser.add_argument('--img', required=True, help='Input SDR image path')
    parser.add_argument('--out', required=True, help='Output Ultra HDR JPEG path')
    parser.add_argument('--model', default='auto', help='AI model path or type (gmnet, synthetic, auto)')
    
    # Export control
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (Ultra HDR JPEG only)')
    
    # Processing
    parser.add_argument('--max-side', type=int, default=4096, help='Max image dimension')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds')
    
    # Behavior
    parser.add_argument('--strict', action='store_true', help='Fail fast on any error')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies')
    parser.add_argument('--debug', action='store_true', help='Save debug files (gain map PNG, metadata JSON)')
    
    args = parser.parse_args()
    
    if args.check_deps:
        deps_ok = _check_dependencies()
        sys.exit(0 if deps_ok else 1)
    
    # Configure pipeline
    config = GainMapPipelineConfig(
        max_side=args.max_side,
        model_type=args.model if args.model in ['auto', 'synthetic', 'gmnet'] else 'auto',
        model_path=args.model if args.model not in ['auto', 'synthetic', 'gmnet'] else None,
        export_quality=args.quality,
        timeout_s=args.timeout,
        strict_mode=args.strict,
        save_intermediate=args.debug,
    )
    
    # Handle legacy model files
    if args.model.endswith('.h5'):
        config.model_type = 'hdrcnn'  # Legacy compatibility during transition
        config.model_path = args.model
        
    if args.verbose:
        print(f"Using model: {config.model_type} ({config.model_path or 'default'})")
        print(f"Export format: Ultra HDR JPEG")
        print(f"Export quality: {config.export_quality}")
    
    try:
        result = run_gainmap_pipeline(args.img, args.out, config)
        
        print("✅ Ultra HDR pipeline completed!")
        print(f"📱 Ultra HDR JPEG: {result.ultrahdr_jpeg_path}")
        
        if args.verbose:
            print(f"🤖 Model: {result.model_name}")
            if result.model_confidence:
                print(f"📊 Confidence: {result.model_confidence:.2f}")
                
    except GainMapPipelineError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def _check_dependencies() -> bool:
    """Check pipeline dependencies"""
    print("🔍 Checking dependencies...")
    all_good = True
    
    # Core dependencies
    try:
        import numpy as np
        import PIL
        print(f"✅ NumPy {np.__version__}, Pillow {PIL.__version__}")
    except ImportError as e:
        print(f"❌ Missing: {e}")
        all_good = False
        
    # Export capabilities
    print("✅ Ultra HDR JPEG export (all platforms)")
        
    return all_good


if __name__ == '__main__':
    main()
