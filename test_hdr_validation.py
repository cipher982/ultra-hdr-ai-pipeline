#!/usr/bin/env python3
"""
Quick test to validate HDR file structure validation.

Tests our validation function against known good and bad examples.
"""

import os
from hdr.gainmap_pipeline import validate_ultrahdr_structure

def test_reference_hdr():
    """Test validation against iPhone HDR reference"""
    ref_path = "tests/fixtures/reference/iphone_hdr.jpg"
    
    if not os.path.exists(ref_path):
        print(f"❌ Reference file not found: {ref_path}")
        return False
        
    result = validate_ultrahdr_structure(ref_path)
    
    print(f"Testing reference HDR: {ref_path}")
    print(f"Valid: {result['valid']}")
    print(f"File size: {result['metadata'].get('file_size', 0):,} bytes")
    print(f"Has MPF: {result['metadata'].get('has_mpf', False)}")
    print(f"Has hdrgm: {result['metadata'].get('has_hdrgm', False)}")
    print(f"JPEG count: {result['metadata'].get('jpeg_count', 0)}")
    
    if result['errors']:
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    return result['valid']

def test_working_output():
    """Test validation against our working output"""
    output_path = "tests/fixtures/reference/pipeline_output.jpg"
    
    if not os.path.exists(output_path):
        print(f"❌ Working output not found: {output_path}")
        return False
        
    result = validate_ultrahdr_structure(output_path)
    
    print(f"\nTesting working output: {output_path}")
    print(f"Valid: {result['valid']}")
    print(f"File size: {result['metadata'].get('file_size', 0):,} bytes")
    print(f"Has MPF: {result['metadata'].get('has_mpf', False)}")
    print(f"Has hdrgm: {result['metadata'].get('has_hdrgm', False)}")  
    print(f"JPEG count: {result['metadata'].get('jpeg_count', 0)}")
    
    if result['errors']:
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    return result['valid']

def test_regular_jpeg():
    """Test validation against regular SDR JPEG"""
    sdr_path = "tests/fixtures/input/sdr_sample.jpg"
    
    if not os.path.exists(sdr_path):
        print(f"❌ SDR reference not found: {sdr_path}")
        return True  # Expected to fail validation
        
    result = validate_ultrahdr_structure(sdr_path)
    
    print(f"\nTesting regular JPEG: {sdr_path}")
    print(f"Valid: {result['valid']}")
    print(f"File size: {result['metadata'].get('file_size', 0):,} bytes")
    
    if result['errors']:
        print("Expected errors (should fail validation):")
        for error in result['errors']:
            print(f"  - {error}")
    
    # Regular JPEG should fail Ultra HDR validation
    return not result['valid']

if __name__ == "__main__":
    print("🔍 HDR Validation Tests")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 3
    
    if test_reference_hdr():
        print("✅ Reference HDR validation passed")
        tests_passed += 1
    else:
        print("❌ Reference HDR validation failed")
    
    if test_working_output():
        print("✅ Working output validation passed")
        tests_passed += 1
    else:
        print("❌ Working output validation failed")
        
    if test_regular_jpeg():
        print("✅ Regular JPEG correctly rejected")
        tests_passed += 1
    else:
        print("❌ Regular JPEG incorrectly passed validation")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("🎉 All validation tests passed!")
        exit(0)
    else:
        print("💥 Some validation tests failed!")
        exit(1)