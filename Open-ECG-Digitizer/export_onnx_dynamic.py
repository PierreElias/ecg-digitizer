#!/usr/bin/env python3
"""
Export UNet segmentation model to ONNX with dynamic input dimensions.

This allows the iOS app to:
1. Use any input resolution (not fixed to 1024x1024)
2. Preserve image aspect ratio (no stretching/padding)
3. Match Python preprocessing exactly

Usage:
    python export_onnx_dynamic.py

Output:
    weights/unet_dynamic.onnx - Dynamic dimension ONNX model
"""

import os
import sys
import torch
import onnx
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.model.unet import UNet


def export_onnx_dynamic():
    """Export UNet model to ONNX with dynamic spatial dimensions."""

    # Model configuration (from inference_wrapper.yml)
    # These must match the weights file exactly
    model_config = {
        "num_in_channels": 3,
        "num_out_channels": 4,  # grid, text, signal, background
        "depth": 2,  # From inference_wrapper.yml
        "dims": [32, 64, 128, 256, 320, 320, 320, 320],  # From inference_wrapper.yml
    }

    # Initialize model
    print("Loading UNet model...")
    model = UNet(**model_config)

    # Load weights
    weights_path = SCRIPT_DIR / "weights" / "unet_weights_07072025.pt"
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        print("Please ensure the weights file exists.")
        sys.exit(1)

    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)

    # Handle potential wrapper keys
    if isinstance(checkpoint, tuple):
        checkpoint = checkpoint[0]
    checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Model loaded successfully")

    # Create sample input (use smaller size for faster export, but dimensions are dynamic)
    # Using 512x512 as reference, but actual inference can use any size
    sample_height = 512
    sample_width = 512
    dummy_input = torch.randn(1, 3, sample_height, sample_width)

    # Output path
    output_path = SCRIPT_DIR / "weights" / "unet_dynamic.onnx"

    # Export to ONNX with dynamic axes
    print(f"\nExporting to ONNX with dynamic dimensions...")
    print(f"  Output: {output_path}")

    # Define dynamic axes for variable input/output sizes
    dynamic_axes = {
        'image': {
            0: 'batch_size',
            2: 'height',
            3: 'width'
        },
        'output': {
            0: 'batch_size',
            2: 'height',
            3: 'width'
        }
    }

    # Use legacy exporter to ensure weights are properly included
    # dynamo=False forces the older, more reliable export path
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=13,  # opset 13 is well-supported
        do_constant_folding=True,
        input_names=['image'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False,
        dynamo=False  # Use legacy exporter
    )

    print("✅ ONNX export complete")

    # Verify the exported model
    print("\nVerifying exported model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verification passed")

    # Print model info
    print(f"\nModel info:")
    print(f"  Input: {onnx_model.graph.input[0].name}")
    print(f"  Output: {onnx_model.graph.output[0].name}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Print dynamic dimension info
    input_shape = onnx_model.graph.input[0].type.tensor_type.shape
    print(f"\n  Input dimensions: [", end="")
    dims = []
    for dim in input_shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)  # Dynamic dimension name
        else:
            dims.append(str(dim.dim_value))  # Fixed dimension value
    print(", ".join(dims) + "]")

    print(f"\n✅ Dynamic ONNX model saved to: {output_path}")
    print("\nTo use in iOS:")
    print("  1. Rename to ECGSegmentation.onnx")
    print("  2. Replace the file in ECGDigitizer/Resources/")
    print("  3. Update ONNXInference.swift to remove fixed 1024x1024 constraint")

    return str(output_path)


def test_dynamic_inference():
    """Test that the exported model works with different input sizes."""
    import onnxruntime as ort
    import numpy as np

    model_path = SCRIPT_DIR / "weights" / "unet_dynamic.onnx"
    if not model_path.exists():
        print("Model not found. Run export first.")
        return

    print("\n" + "="*60)
    print("Testing dynamic inference...")
    print("="*60)

    session = ort.InferenceSession(str(model_path))

    # Test different input sizes
    test_sizes = [
        (512, 512),   # Small square
        (768, 768),   # Medium square
        (1024, 1024), # Original fixed size
        (800, 600),   # Landscape
        (600, 800),   # Portrait
        (1920, 1080), # HD landscape
    ]

    for h, w in test_sizes:
        try:
            dummy_input = np.random.randn(1, 3, h, w).astype(np.float32)
            outputs = session.run(None, {'image': dummy_input})
            output_shape = outputs[0].shape
            print(f"  ✅ Input {w}x{h} → Output {output_shape[3]}x{output_shape[2]} (4 channels)")
        except Exception as e:
            print(f"  ❌ Input {w}x{h} failed: {e}")

    print("\n✅ Dynamic inference test complete")


if __name__ == "__main__":
    output_path = export_onnx_dynamic()

    # Test the exported model
    try:
        import onnxruntime
        test_dynamic_inference()
    except ImportError:
        print("\nNote: Install onnxruntime to test inference: pip install onnxruntime")
