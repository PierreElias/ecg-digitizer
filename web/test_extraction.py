#!/usr/bin/env python3
"""
Test the Open-ECG-Digitizer integration with sample images.
"""

import sys
import numpy as np
from PIL import Image
from open_ecg_digitizer import process_with_open_ecg, HAS_OPEN_ECG

def test_image(image_path: str):
    """Test processing a single image."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print('='*60)

    # Load image
    img = Image.open(image_path)
    img_array = np.array(img.convert('RGB'))

    print(f"Image size: {img_array.shape}")

    # Process
    result = process_with_open_ecg(img_array, paper_speed=25, voltage_gain=10, layout=None)

    # Display results
    if result['success']:
        print(f"✓ Success!")
        print(f"  Method: {result.get('method', 'unknown')}")
        print(f"  Layout: {result.get('layout', 'unknown')}")
        print(f"  Detected layout: {result.get('detected_layout', 'unknown')}")
        print(f"  Lines detected: {result.get('num_lines_detected', 'unknown')}")
        print(f"  Leads extracted: {len(result.get('leads', []))}")

        if 'grid' in result:
            grid = result['grid']
            print(f"  Grid detection: {grid.get('detected', False)}")
            print(f"    Confidence: {grid.get('confidence', 0):.2f}")
            print(f"    mm/pixel: x={grid.get('mm_per_pixel_x', 0):.4f}, y={grid.get('mm_per_pixel_y', 0):.4f}")

        # Show sample lead info
        if result.get('leads'):
            print(f"\n  Sample leads:")
            for i, lead in enumerate(result['leads'][:3]):
                samples = lead['samples']
                if len(samples) > 0:
                    min_v = min(samples)
                    max_v = max(samples)
                    mean_v = sum(samples) / len(samples)
                    print(f"    {lead['name']}: {len(samples)} samples, range=[{min_v:.3f}, {max_v:.3f}]mV, mean={mean_v:.3f}mV")
    else:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        if 'traceback' in result:
            print(f"\nTraceback:\n{result['traceback']}")


if __name__ == '__main__':
    if not HAS_OPEN_ECG:
        print("ERROR: Open-ECG-Digitizer not available!")
        sys.exit(1)

    # Test images
    test_images = [
        '/Users/pae2/Desktop/ecg_app/ecg_images/ecg00005.jpg',
        '/Users/pae2/Desktop/ecg_app/ecg_images/ecg00008.jpg',
    ]

    for img_path in test_images:
        try:
            test_image(img_path)
        except Exception as e:
            import traceback
            print(f"ERROR processing {img_path}:")
            print(traceback.format_exc())

    print(f"\n{'='*60}")
    print("Testing complete!")
    print('='*60)
