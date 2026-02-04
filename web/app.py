#!/usr/bin/env python3
"""
ECG Digitizer Web Application

A web-based interface for ECG image digitization using Open-ECG-Digitizer.
"""

import os
import sys
import base64
import tempfile
import uuid
import gc
from io import BytesIO
from datetime import datetime

from flask import Flask, render_template_string, request, jsonify, send_file
import numpy as np
import torch
from PIL import Image

# Maximum image size in bytes (2MB)
MAX_IMAGE_SIZE_BYTES = 2 * 1024 * 1024

# Maximum image dimensions for diagnostic endpoint (prevents OOM)
MAX_DIAGNOSTIC_DIMENSION = 2000


def ensure_image_under_limit(img: Image.Image, max_bytes: int = MAX_IMAGE_SIZE_BYTES) -> tuple[Image.Image, dict]:
    """
    Ensure an image is under the specified size limit while maintaining fidelity.

    Strategy:
    1. Try JPEG quality reduction (95%, 85%, 75%, 65%)
    2. If still too large, resize progressively (90%, 80%, 70% of original dimensions)
    3. Report what was done

    Args:
        img: PIL Image object
        max_bytes: Maximum size in bytes (default 2MB)

    Returns:
        Tuple of (processed_image, info_dict)
        info_dict contains: original_size, final_size, quality, scale, was_resized, was_recompressed
    """
    from io import BytesIO

    # Ensure RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    original_width, original_height = img.size
    info = {
        'original_size': None,
        'final_size': None,
        'quality': 95,
        'scale': 1.0,
        'was_resized': False,
        'was_recompressed': False,
        'original_dimensions': (original_width, original_height),
        'final_dimensions': (original_width, original_height)
    }

    # Check initial size at high quality
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    info['original_size'] = buffer.tell()

    # If already under limit, return as-is
    if buffer.tell() <= max_bytes:
        info['final_size'] = buffer.tell()
        return img, info

    # Try reducing JPEG quality
    quality_levels = [85, 75, 65, 55]
    for quality in quality_levels:
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        if buffer.tell() <= max_bytes:
            info['final_size'] = buffer.tell()
            info['quality'] = quality
            info['was_recompressed'] = True
            print(f"  📉 Image compressed: {info['original_size']/1024:.1f}KB → {info['final_size']/1024:.1f}KB (quality={quality})")
            return img, info

    # Quality reduction not enough - need to resize
    scale_factors = [0.9, 0.8, 0.7, 0.6, 0.5]
    for scale in scale_factors:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Try with quality 75 first (good balance)
        buffer = BytesIO()
        resized.save(buffer, format='JPEG', quality=75)
        if buffer.tell() <= max_bytes:
            info['final_size'] = buffer.tell()
            info['quality'] = 75
            info['scale'] = scale
            info['was_resized'] = True
            info['was_recompressed'] = True
            info['final_dimensions'] = (new_width, new_height)
            print(f"  📉 Image resized: {original_width}x{original_height} → {new_width}x{new_height} ({info['original_size']/1024:.1f}KB → {info['final_size']/1024:.1f}KB)")
            return resized, info

    # Last resort: aggressive resize and compression
    final_scale = 0.4
    new_width = int(original_width * final_scale)
    new_height = int(original_height * final_scale)
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    resized.save(buffer, format='JPEG', quality=60)
    info['final_size'] = buffer.tell()
    info['quality'] = 60
    info['scale'] = final_scale
    info['was_resized'] = True
    info['was_recompressed'] = True
    info['final_dimensions'] = (new_width, new_height)
    print(f"  📉 Image aggressively resized: {original_width}x{original_height} → {new_width}x{new_height}")
    return resized, info

# Add parent directory and current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Add web/ directory

# Import the proper Open-ECG-Digitizer wrapper
try:
    from digitizer_wrapper import process_ecg_image, get_digitizer, HAS_OPEN_ECG
    print(f"Open-ECG-Digitizer loaded: {HAS_OPEN_ECG}")
except ImportError as e:
    HAS_OPEN_ECG = False
    print(f"Open-ECG-Digitizer not available: {e}")
    import traceback
    traceback.print_exc()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload


# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint for load balancers and monitoring."""
    return jsonify({
        'status': 'healthy',
        'service': 'ecg-digitizer',
        'open_ecg_available': HAS_OPEN_ECG,
        'torch_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    """Main ECG processing endpoint."""
    data = request.json

    image_data = data.get('image')
    if not image_data:
        return jsonify({'success': False, 'error': 'No image provided'})

    try:
        # Decode base64 image and immediately clear original data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        del image_data  # Free the base64 string
        gc.collect()

        img = Image.open(BytesIO(img_bytes))
        del img_bytes  # Free the decoded bytes
        gc.collect()

        # Check and enforce 2MB size limit
        img, size_info = ensure_image_under_limit(img, MAX_IMAGE_SIZE_BYTES)
        if size_info['was_resized'] or size_info['was_recompressed']:
            print(f"📊 /api/process: Image optimized - {size_info}")
            gc.collect()  # Clean up after resize

        img_array = np.array(img.convert('RGB'))

        if img.size[0] < 128 or img.size[1] < 128:
            return jsonify({'success': False, 'error': 'Image too small (minimum 128x128)'})

        # Get processing settings
        layout = data.get('layout') or None
        use_signal_boundaries = data.get('signal_based_boundaries', True)

        # Process with Open-ECG-Digitizer
        if HAS_OPEN_ECG:
            result = process_ecg_image(
                img_array,
                layout=layout,
                use_signal_based_boundaries=use_signal_boundaries
            )
            if result['success']:
                result['timestamp'] = datetime.now().isoformat()
                result['image_size'] = {'width': img.size[0], 'height': img.size[1]}
                # Add extraction algorithm info (full pipeline always uses Advanced)
                result['extraction_algorithm'] = 'advanced'
                result['extraction_algorithm_display'] = 'SignalExtractorAdvanced (Hungarian + CC)'
                print(f"📊 Extraction algorithm: SignalExtractorAdvanced (Hungarian + CC)")
                return jsonify(result)
            else:
                return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': 'Open-ECG-Digitizer not available'
            })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route('/api/diagnostic', methods=['POST'])
def api_diagnostic():
    """Generate detailed diagnostic report."""
    try:
        from debug_pipeline import run_diagnostic

        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'success': False, 'error': 'No image provided'})

        settings = {
            'preprocessing': data.get('preprocessing', True),
            'signal_sensitivity': data.get('signal_sensitivity', 0.3),
            'vertical_margin': data.get('vertical_margin', 0.15),
            'min_width_squares': data.get('min_width_squares', 49.5),
            'signal_based_boundaries': data.get('signal_based_boundaries', True),
            'dilation_size': data.get('dilation_size', 3),
        }

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode and immediately clear original base64 data
        img_bytes = base64.b64decode(image_data)
        del image_data
        del data
        gc.collect()

        img = Image.open(BytesIO(img_bytes))
        del img_bytes
        gc.collect()

        # Enforce size limit first
        img, size_info = ensure_image_under_limit(img, MAX_IMAGE_SIZE_BYTES)
        if size_info['was_resized'] or size_info['was_recompressed']:
            print(f"📊 /api/diagnostic: Image optimized - {size_info}")

        # Additionally enforce max dimensions for diagnostic (prevents OOM)
        width, height = img.size
        if width > MAX_DIAGNOSTIC_DIMENSION or height > MAX_DIAGNOSTIC_DIMENSION:
            scale = min(MAX_DIAGNOSTIC_DIMENSION / width, MAX_DIAGNOSTIC_DIMENSION / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📊 /api/diagnostic: Resized for memory - {width}x{height} → {new_width}x{new_height}")
            gc.collect()

        temp_dir = tempfile.mkdtemp()
        temp_img = os.path.join(temp_dir, f'ecg_{uuid.uuid4().hex[:8]}.png')
        img.save(temp_img)
        del img
        gc.collect()

        report_path = run_diagnostic(temp_img, temp_dir, settings=settings)

        # Force cleanup after diagnostic
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with open(report_path, 'r') as f:
            html_content = f.read()

        # Cleanup temp files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

        return jsonify({'success': True, 'html': html_content})

    except Exception as e:
        import traceback
        gc.collect()  # Cleanup on error too
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route('/diagnostic')
def diagnostic_page():
    return render_template_string(DIAGNOSTIC_TEMPLATE)


def baseline_wander_removal(samples, sampling_frequency=500):
    """
    Remove baseline wander from ECG signal using two-pass median filtering.

    Step 5d: Baseline wander removal (applied after raw signal extraction)

    This uses a cascaded median filter approach:
    1. First pass: 0.2s window to capture high-frequency baseline
    2. Second pass: 0.6s window to smooth the baseline estimate
    3. Subtract the estimated baseline from the original signal

    Args:
        samples: List or numpy array of signal samples
        sampling_frequency: Sample rate in Hz (default 500 Hz)

    Returns:
        Baseline-corrected samples as list
    """
    import scipy.signal

    data = np.array(samples, dtype=np.float64)

    if len(data) < 10:
        return samples  # Too short for filtering

    # First pass: 0.2s window median filter
    win_size_1 = int(np.round(0.2 * sampling_frequency))
    if win_size_1 % 2 == 0:
        win_size_1 += 1  # medfilt requires odd window size
    win_size_1 = max(3, win_size_1)  # Minimum window size of 3

    baseline = scipy.signal.medfilt(data, win_size_1)

    # Second pass: 0.6s window median filter to smooth baseline
    win_size_2 = int(np.round(0.6 * sampling_frequency))
    if win_size_2 % 2 == 0:
        win_size_2 += 1
    win_size_2 = max(3, win_size_2)

    baseline = scipy.signal.medfilt(baseline, win_size_2)

    # Remove baseline from signal
    corrected = data - baseline

    return corrected.tolist()


def baseline_wander_removal_batch(leads, sampling_frequency=500):
    """
    Apply baseline wander removal to a list of leads.

    Args:
        leads: List of lead dictionaries with 'samples' key
        sampling_frequency: Sample rate in Hz

    Returns:
        List of leads with baseline-corrected samples
    """
    corrected_leads = []
    for lead in leads:
        corrected_lead = lead.copy()
        if 'samples' in lead and len(lead['samples']) > 0:
            corrected_lead['samples'] = baseline_wander_removal(
                lead['samples'],
                sampling_frequency
            )
        corrected_leads.append(corrected_lead)
    return corrected_leads


def extract_leads_by_sectioning(signal_prob, grid_prob, width, height):
    """
    Fallback lead extraction using 3x4 layout sectioning.

    This follows the working pattern from open_ecg_digitizer_old.py:
    1. Step 6a-b: Find row boundaries (where each signal row is)
    2. Step 6c: Extract FULL-WIDTH signal lines per row (weighted centroid)
    3. Step 6d: Convert to voltage values (baseline removal, scaling)
    4. Step 7: Split each row into 4 columns based on 3x4 layout

    Standard 12-lead ECG layout (3 rows x 4 columns):
        Column 0    Column 1    Column 2    Column 3
        (0-25%)    (25-50%)    (50-75%)    (75-100%)
    Row 0:  I         aVR         V1          V4
    Row 1:  II        aVL         V2          V5
    Row 2:  III       aVF         V3          V6

    Returns list of 12 lead dictionaries with samples.
    """
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    print(f"  Step 6 Fallback sectioning: {width}x{height}")

    # ===== Step 6a: Find row boundaries by analyzing signal probability =====
    row_sums = np.sum(signal_prob, axis=1)  # Sum along x for each y

    # Smooth to find row centers
    sigma = max(3, height // 30)
    row_sums_smooth = gaussian_filter1d(row_sums, sigma=sigma)

    # Find peaks (row centers) - expect 3-4 rows for standard ECG
    distance = max(10, height // 6)
    prominence = np.max(row_sums_smooth) * 0.1
    peaks, peak_props = find_peaks(row_sums_smooth, distance=distance, prominence=prominence)

    if len(peaks) < 3:
        # Fallback: divide into 3 equal rows
        peaks = np.array([height // 6, height // 2, 5 * height // 6])
        print(f"    Step 6a: Using default 3-row division at y={peaks.tolist()}")
    else:
        # Take top 3 peaks by prominence
        if len(peaks) > 3:
            prominences = peak_props['prominences']
            top_indices = np.argsort(prominences)[-3:]
            peaks = np.sort(peaks[top_indices])
        print(f"    Step 6a: Found {len(peaks)} row centers at y={peaks.tolist()}")

    # ===== Step 6b: Define row boundaries =====
    row_height = height // 3
    row_boundaries = []
    for i, peak in enumerate(peaks[:3]):
        y_min = max(0, peak - row_height // 2)
        y_max = min(height, peak + row_height // 2)
        row_boundaries.append((y_min, y_max))
    print(f"    Step 6b: Row boundaries: {row_boundaries}")

    # ===== Step 6c: Extract FULL-WIDTH signal lines per row =====
    # This matches the working pattern: extract full row, then split
    print(f"    Step 6c: Extracting full-width signal lines per row...")

    raw_lines = []  # Shape will be (num_rows, width)
    for row_idx, (y_min, y_max) in enumerate(row_boundaries):
        row_prob = signal_prob[y_min:y_max, :]
        row_height_local = y_max - y_min

        # Extract Y-position (centroid) for each X column
        line = []
        for x in range(width):
            col = row_prob[:, x]
            col_sum = np.sum(col)
            if col_sum > 0.01:
                # Weighted centroid (Y position)
                y_indices = np.arange(row_height_local)
                centroid_y = np.sum(y_indices * col) / col_sum
                line.append(centroid_y)
            else:
                line.append(np.nan)  # Mark as missing
        raw_lines.append(np.array(line))
        valid_count = np.sum(~np.isnan(line))
        print(f"      Row {row_idx}: {valid_count}/{width} valid samples")

    # ===== Step 6d: Convert to voltage values =====
    # Center around baseline (mean Y position) and scale
    print(f"    Step 6d: Converting to voltage values...")

    voltage_lines = []
    for row_idx, line in enumerate(raw_lines):
        valid_mask = ~np.isnan(line)
        if np.sum(valid_mask) > 10:
            # Remove baseline (mean Y position)
            baseline = np.nanmean(line)
            offset_pixels = line - baseline

            # Convert to voltage-like values
            # Y increases downward, so negative offset = positive voltage
            # Scale to reasonable mV range (assuming ~100 pixels per mV)
            row_height_local = row_boundaries[row_idx][1] - row_boundaries[row_idx][0]
            scale_factor = 2.0 / (row_height_local / 2)  # ±2mV range
            voltage = -offset_pixels * scale_factor

            # Fill NaN with interpolation
            if np.any(np.isnan(voltage)):
                valid_x = np.where(valid_mask)[0]
                valid_v = voltage[valid_mask]
                if len(valid_x) > 1:
                    f = interp1d(valid_x, valid_v, kind='linear', bounds_error=False, fill_value=0.0)
                    voltage = f(np.arange(len(voltage)))

            voltage_lines.append(voltage)
        else:
            voltage_lines.append(np.zeros(width))

    # ===== Step 7: Split each row into 4 columns (3x4 layout) =====
    print(f"    Step 7: Splitting rows into 4 columns for lead assignment...")

    lead_mapping = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6']
    ]

    target_samples = 5000
    samples_per_section = target_samples // 4  # 1250 samples per column
    cols = 4

    leads = []
    lead_names_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for lead_name in lead_names_order:
        # Find which row and column this lead belongs to
        row_idx = None
        col_idx = None
        for r, row in enumerate(lead_mapping):
            if lead_name in row:
                row_idx = r
                col_idx = row.index(lead_name)
                break

        if row_idx is not None and col_idx is not None and row_idx < len(voltage_lines):
            samples = voltage_lines[row_idx]

            # Split into column section
            col_width = len(samples) // cols
            start = col_idx * col_width
            end = (col_idx + 1) * col_width if col_idx < cols - 1 else len(samples)
            col_samples = samples[start:end]

            # Resample to target samples per section
            if len(col_samples) > 1:
                x_old = np.linspace(0, 1, len(col_samples))
                x_new = np.linspace(0, 1, samples_per_section)
                f = interp1d(x_old, col_samples, kind='linear', fill_value='extrapolate')
                resampled = f(x_new)
            else:
                resampled = np.zeros(samples_per_section)

            # Build full 5000-sample array with data at correct position
            full_samples = np.zeros(target_samples)
            start_idx = col_idx * samples_per_section
            full_samples[start_idx:start_idx + samples_per_section] = resampled

            leads.append({
                'name': lead_name,
                'samples': full_samples.tolist(),
                'duration_ms': 10000,
                'sample_count': target_samples
            })
        else:
            leads.append({
                'name': lead_name,
                'samples': [0.0] * target_samples,
                'duration_ms': 10000,
                'sample_count': target_samples
            })

    # ===== Step 7b: Handle rhythm strip (Lead II uses full row width) =====
    # Find Lead II in the list and replace with full-width data
    for i, lead in enumerate(leads):
        if lead['name'] == 'II':
            row_idx = 1  # Lead II is in row 1
            if row_idx < len(voltage_lines):
                full_row = voltage_lines[row_idx]
                # Resample to target_samples
                if len(full_row) > 1:
                    x_old = np.linspace(0, 1, len(full_row))
                    x_new = np.linspace(0, 1, target_samples)
                    f = interp1d(x_old, full_row, kind='linear', fill_value='extrapolate')
                    full_samples = f(x_new).tolist()
                    leads[i]['samples'] = full_samples
                    print(f"    Step 7b: Lead II rhythm strip using full row width")
            break

    print(f"    Extracted {len(leads)} leads: {[l['name'] for l in leads]}")

    # ===== Step 5d: Apply baseline wander removal to all leads =====
    print(f"    Step 5d: Applying baseline wander removal (500 Hz, win1=101, win2=301)")
    leads = baseline_wander_removal_batch(leads, sampling_frequency=500)

    return leads


@app.route('/api/postprocess', methods=['POST'])
def api_postprocess():
    """
    Hybrid processing endpoint: receives probability maps from iOS ONNX inference,
    runs full post-processing pipeline using InferenceWrapper.

    This enables fast on-device segmentation + accurate Python post-processing.

    Expected request format:
    {
        "signal_prob": "<base64 encoded float32 array>",
        "grid_prob": "<base64 encoded float32 array>",
        "text_prob": "<base64 encoded float32 array>",
        "width": 1024,
        "height": 1024,
        "format": "float32_base64"
    }
    """
    try:
        from digitizer_wrapper import get_digitizer, HAS_OPEN_ECG

        if not HAS_OPEN_ECG:
            return jsonify({
                'success': False,
                'error': 'Open-ECG-Digitizer not available'
            })

        data = request.json
        width = data.get('width', 1024)
        height = data.get('height', 1024)

        # Decode probability maps from base64
        import struct

        def decode_float_array(base64_str, expected_size):
            """Decode base64-encoded float32 array"""
            raw_bytes = base64.b64decode(base64_str)
            num_floats = len(raw_bytes) // 4
            if num_floats != expected_size:
                raise ValueError(f"Expected {expected_size} floats, got {num_floats}")
            return np.array(struct.unpack(f'{num_floats}f', raw_bytes), dtype=np.float32)

        expected_size = width * height
        signal_prob = decode_float_array(data['signal_prob'], expected_size)
        grid_prob = decode_float_array(data['grid_prob'], expected_size)
        text_prob = decode_float_array(data['text_prob'], expected_size)

        # Reshape to 2D arrays
        signal_prob = signal_prob.reshape(height, width)
        grid_prob = grid_prob.reshape(height, width)
        text_prob = text_prob.reshape(height, width)

        print(f"Postprocess: received prob maps {width}x{height}")
        print(f"  Signal: min={signal_prob.min():.3f}, max={signal_prob.max():.3f}, mean={signal_prob.mean():.3f}")
        print(f"  Grid: min={grid_prob.min():.3f}, max={grid_prob.max():.3f}")

        digitizer = get_digitizer()

        # Check if original image was provided (for better results)
        original_image_b64 = data.get('original_image')
        if original_image_b64:
            print(f"  Using provided original image for processing (full resolution)")
            # Decode original image
            if original_image_b64.startswith('data:'):
                original_image_b64 = original_image_b64.split(',', 1)[1]
            import io
            from PIL import Image as PILImage
            img_bytes = base64.b64decode(original_image_b64)
            pil_image = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')

            # Check and enforce 2MB size limit
            pil_image, size_info = ensure_image_under_limit(pil_image, MAX_IMAGE_SIZE_BYTES)
            if size_info['was_resized'] or size_info['was_recompressed']:
                print(f"  📊 /api/postprocess: Original image optimized - {size_info}")

            # Use full resolution - DON'T resize to probability map size
            # The pipeline handles resizing internally
            process_image = np.array(pil_image)
            print(f"  Original image size: {process_image.shape}")
        else:
            # Create synthetic ECG image from probability maps
            # This allows the full pipeline to run with the phone's segmentation
            print(f"  Creating synthetic ECG image from probability maps")
            # White background (255), dark signals (0), faint grid lines
            process_image = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Add grid lines (light gray where grid_prob is high)
            grid_intensity = 255 - (grid_prob * 100).clip(0, 100).astype(np.uint8)
            process_image[:, :, 0] = np.minimum(process_image[:, :, 0], grid_intensity)
            process_image[:, :, 1] = np.minimum(process_image[:, :, 1], grid_intensity)
            process_image[:, :, 2] = np.minimum(process_image[:, :, 2], grid_intensity)

            # Add signal (dark black where signal_prob is high)
            signal_mask = signal_prob > 0.1
            signal_intensity = 255 - (signal_prob * 255).clip(0, 255).astype(np.uint8)
            process_image[signal_mask, 0] = np.minimum(process_image[signal_mask, 0], signal_intensity[signal_mask])
            process_image[signal_mask, 1] = np.minimum(process_image[signal_mask, 1], signal_intensity[signal_mask])
            process_image[signal_mask, 2] = np.minimum(process_image[signal_mask, 2], signal_intensity[signal_mask])

        # Process through full pipeline
        result = digitizer.process_image(process_image, use_signal_based_boundaries=True)

        # process_image returns the full result dict directly
        if result.get('success', False):
            leads = result.get('leads', [])
            grid_info = result.get('grid', {})

            non_empty = len([l for l in leads if any(abs(s) > 0.0001 for s in l.get('samples', []))])
            print(f"  Pipeline returned {len(leads)} leads, {non_empty} non-empty")

            # Step 6d & 7: If pipeline didn't return enough leads, use fallback sectioning
            # This determines where each lead should be and cuts lines into 4 equal sections
            used_fallback = False
            if non_empty < 10:
                print(f"  ⚠ Insufficient leads ({non_empty}) from pipeline, using fallback 3x4 sectioning")
                leads = extract_leads_by_sectioning(signal_prob, grid_prob, width, height)
                non_empty = len([l for l in leads if any(abs(s) > 0.0001 for s in l.get('samples', []))])
                print(f"  Fallback extracted {non_empty} non-empty leads")
                used_fallback = True

            # Step 5d: Apply baseline wander removal to main pipeline leads
            # (fallback sectioning already applies this internally)
            if not used_fallback and leads:
                print(f"  Applying baseline wander removal to {len(leads)} leads")
                leads = baseline_wander_removal_batch(leads, sampling_frequency=500)

            # Determine which extraction algorithm was used
            extraction_algorithm = 'sectioning' if used_fallback else 'advanced'
            extraction_algorithm_display = 'SignalExtractorSectioning (Fallback)' if used_fallback else 'SignalExtractorAdvanced (Hungarian + CC)'
            print(f"  📊 Extraction algorithm: {extraction_algorithm_display}")

            return jsonify({
                'success': True,
                'leads': leads,
                'layout': result.get('layout', 'standard_3x4_with_r1'),
                'method': 'hybrid_onnx_postprocess',
                'extraction_algorithm': extraction_algorithm,
                'extraction_algorithm_display': extraction_algorithm_display,
                'grid': grid_info,
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Pipeline failed - try fallback sectioning method
            print(f"  Pipeline failed: {result.get('error')}, trying fallback sectioning")
            leads = extract_leads_by_sectioning(signal_prob, grid_prob, width, height)
            non_empty = len([l for l in leads if any(abs(s) > 0.0001 for s in l.get('samples', []))])

            if non_empty >= 6:
                print(f"  Fallback succeeded with {non_empty} non-empty leads")
                print(f"  📊 Extraction algorithm: SignalExtractorSectioning (Fallback)")
                return jsonify({
                    'success': True,
                    'leads': leads,
                    'layout': 'standard_3x4_with_r1',
                    'method': 'hybrid_fallback_sectioning',
                    'extraction_algorithm': 'sectioning',
                    'extraction_algorithm_display': 'SignalExtractorSectioning (Fallback)',
                    'grid': {'detected': False, 'confidence': 0.5},
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Pipeline failed')
                })

    except ImportError as e:
        # Fallback if components aren't properly imported
        return jsonify({
            'success': False,
            'error': f'Import error: {str(e)}. Postprocess endpoint requires full Open-ECG-Digitizer installation.'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


# ============================================================
# HTML Templates - Modern Professional Design
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Digitizer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --color-bg: #FAFAFA;
            --color-surface: #FFFFFF;
            --color-primary: #0066FF;
            --color-primary-dark: #0052CC;
            --color-primary-light: #E6F0FF;
            --color-accent: #7C3AED;
            --color-text: #111827;
            --color-text-muted: #6B7280;
            --color-border: #E5E7EB;
            --color-success: #10B981;
            --color-error: #EF4444;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --radius-xl: 20px;
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
            --transition: 0.15s ease;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 24px;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header-icon {
            width: 56px;
            height: 56px;
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
            border-radius: var(--radius-lg);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px;
            box-shadow: 0 4px 14px rgba(0, 102, 255, 0.25);
        }

        .header-icon svg {
            width: 28px;
            height: 28px;
            color: white;
        }

        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--color-text);
            margin-bottom: 6px;
            letter-spacing: -0.025em;
        }

        .header p {
            color: var(--color-text-muted);
            font-size: 1rem;
        }

        /* Layout Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 32px;
            align-items: start;
        }

        @media (max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        /* Cards */
        .card {
            background: var(--color-surface);
            border-radius: var(--radius-xl);
            padding: 24px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-border);
        }

        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-title svg {
            width: 18px;
            height: 18px;
            color: var(--color-primary);
        }

        /* Upload Area */
        .upload-area {
            border: 1.5px dashed var(--color-border);
            border-radius: var(--radius-lg);
            padding: 32px 20px;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
            background: var(--color-bg);
            margin-bottom: 20px;
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--color-primary);
            background: var(--color-primary-light);
        }

        .upload-area input { display: none; }

        .upload-icon {
            width: 44px;
            height: 44px;
            margin: 0 auto 12px;
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .upload-icon svg {
            width: 22px;
            height: 22px;
            color: white;
        }

        .upload-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--color-text);
            margin-bottom: 4px;
        }

        .upload-hint {
            font-size: 0.8rem;
            color: var(--color-text-muted);
        }

        .preview-img {
            max-width: 100%;
            max-height: 180px;
            border-radius: var(--radius-md);
            margin-top: 12px;
            object-fit: contain;
        }

        /* Form Controls */
        .form-group {
            margin-bottom: 16px;
        }

        .form-label {
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--color-text-muted);
            margin-bottom: 6px;
        }

        .form-select {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            font-size: 0.95rem;
            font-family: inherit;
            background: var(--color-surface);
            color: var(--color-text);
            cursor: pointer;
            transition: var(--transition);
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%236B6560' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
        }

        .form-select:focus {
            outline: none;
            border-color: var(--color-primary);
            box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.1);
        }

        /* Collapsible Settings */
        .settings-toggle {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 14px;
            background: var(--color-bg);
            border-radius: var(--radius-md);
            cursor: pointer;
            margin-bottom: 16px;
            transition: var(--transition);
            border: 1px solid var(--color-border);
        }

        .settings-toggle:hover {
            background: var(--color-primary-light);
            border-color: var(--color-primary);
        }

        .settings-toggle-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--color-text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .settings-toggle-icon {
            width: 18px;
            height: 18px;
            transition: transform var(--transition);
        }

        .settings-toggle.open .settings-toggle-icon {
            transform: rotate(180deg);
        }

        .settings-content {
            display: none;
            padding: 16px;
            background: var(--color-bg);
            border-radius: var(--radius-md);
            margin-bottom: 16px;
        }

        .settings-content.open { display: block; }

        /* Settings Row */
        .setting-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }

        .setting-row:last-child { margin-bottom: 0; }

        .setting-label {
            flex: 0 0 140px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--color-text);
        }

        .setting-slider {
            flex: 1;
            min-width: 120px;
            height: 6px;
            border-radius: 3px;
            background: var(--color-border);
            appearance: none;
            cursor: pointer;
        }

        .setting-slider::-webkit-slider-thumb {
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--color-primary);
            cursor: pointer;
            box-shadow: var(--shadow-sm);
        }

        .setting-value {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--color-primary);
            background: rgba(0, 102, 255, 0.1);
            padding: 4px 10px;
            border-radius: var(--radius-sm);
            min-width: 50px;
            text-align: center;
        }

        /* Toggle Switch */
        .toggle {
            position: relative;
            width: 44px;
            height: 24px;
            flex-shrink: 0;
        }

        .toggle input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .toggle-track {
            position: absolute;
            inset: 0;
            background: var(--color-border);
            border-radius: 12px;
            cursor: pointer;
            transition: var(--transition);
        }

        .toggle-track::before {
            content: "";
            position: absolute;
            width: 18px;
            height: 18px;
            left: 3px;
            top: 3px;
            background: white;
            border-radius: 50%;
            transition: var(--transition);
            box-shadow: var(--shadow-sm);
        }

        .toggle input:checked + .toggle-track {
            background: var(--color-primary);
        }

        .toggle input:checked + .toggle-track::before {
            transform: translateX(20px);
        }

        /* Primary Button */
        .btn-primary {
            width: 100%;
            padding: 12px 20px;
            background: var(--color-primary);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            font-size: 0.95rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-primary svg {
            width: 18px;
            height: 18px;
        }

        /* Status Messages */
        .status {
            padding: 12px 16px;
            border-radius: var(--radius-md);
            margin-top: 16px;
            font-size: 0.9rem;
            display: none;
        }

        .status.loading {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(0, 102, 255, 0.08);
            color: var(--color-primary-dark);
        }

        .status.success {
            display: block;
            background: rgba(90, 143, 90, 0.1);
            color: var(--color-success);
        }

        .status.error {
            display: block;
            background: rgba(184, 84, 80, 0.1);
            color: var(--color-error);
        }

        .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid var(--color-primary-light);
            border-top-color: var(--color-primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Results Card */
        .results-placeholder {
            text-align: center;
            padding: 60px 20px;
            color: var(--color-text-muted);
        }

        .results-placeholder svg {
            width: 48px;
            height: 48px;
            margin-bottom: 16px;
            opacity: 0.4;
        }

        .result-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }

        .stat-item {
            background: var(--color-bg);
            padding: 14px;
            border-radius: var(--radius-md);
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .stat-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--color-text);
        }

        /* Lead Grid */
        .lead-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px;
        }

        .lead-tag {
            background: var(--color-bg);
            padding: 10px 12px;
            border-radius: var(--radius-sm);
            text-align: center;
        }

        .lead-tag-name {
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--color-primary);
        }

        .lead-tag-info {
            font-size: 0.75rem;
            color: var(--color-text-muted);
            margin-top: 2px;
        }

        /* Waveform Card */
        .waveform-card {
            margin-top: 32px;
        }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 4px;
            border-bottom: 1px solid var(--color-border);
            margin-bottom: 24px;
        }

        .tab-btn {
            padding: 12px 20px;
            border: none;
            background: none;
            font-size: 0.9rem;
            font-weight: 500;
            font-family: inherit;
            color: var(--color-text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            transition: var(--transition);
        }

        .tab-btn:hover { color: var(--color-text); }

        .tab-btn.active {
            color: var(--color-primary);
            border-bottom-color: var(--color-primary);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active { display: block; }

        /* Waveform Canvas */
        .waveform-container {
            background: #FFF8F5;
            border-radius: var(--radius-lg);
            padding: 20px;
            position: relative;
        }

        .waveform-canvas {
            width: 100%;
            height: 400px;
        }

        /* Signal Table */
        .table-container {
            max-height: 400px;
            overflow: auto;
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
        }

        .signals-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }

        .signals-table th, .signals-table td {
            padding: 10px 14px;
            text-align: right;
            border-bottom: 1px solid var(--color-border);
        }

        .signals-table th {
            background: var(--color-bg);
            font-weight: 600;
            position: sticky;
            top: 0;
            color: var(--color-text);
        }

        .signals-table td:first-child, .signals-table th:first-child {
            text-align: left;
            position: sticky;
            left: 0;
            background: white;
        }

        .signals-table th:first-child { background: var(--color-bg); }

        /* Chart Container */
        .chart-container {
            position: relative;
            height: 400px;
        }

        .chart-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            align-items: center;
        }

        .lead-checkboxes {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .lead-checkbox {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            background: var(--color-bg);
            border-radius: var(--radius-sm);
            font-size: 0.85rem;
            cursor: pointer;
        }

        /* Export Buttons */
        .export-buttons {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }

        .btn-export {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid var(--color-border);
            background: var(--color-surface);
            color: var(--color-text);
            border-radius: var(--radius-md);
            font-size: 0.9rem;
            font-weight: 500;
            font-family: inherit;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-export:hover {
            background: var(--color-bg);
            border-color: var(--color-primary-light);
        }

        .btn-export svg {
            width: 16px;
            height: 16px;
        }

        .btn-diagnostic {
            width: 100%;
            padding: 10px 16px;
            border: 1px solid var(--color-border);
            background: var(--color-surface);
            color: var(--color-text);
            border-radius: var(--radius-md);
            font-size: 0.85rem;
            font-weight: 500;
            font-family: inherit;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-diagnostic:hover {
            background: var(--color-primary-light);
            border-color: var(--color-primary);
            color: var(--color-primary);
        }

        .btn-diagnostic:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Diagnostic Modal */
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(45, 42, 38, 0.5);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 24px;
        }

        .modal-overlay.active { display: flex; }

        .modal {
            background: var(--color-surface);
            border-radius: var(--radius-xl);
            max-width: 90vw;
            max-height: 90vh;
            overflow: auto;
            box-shadow: var(--shadow-lg);
        }

        .modal-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--color-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            background: var(--color-surface);
        }

        .modal-title {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .modal-close {
            background: none;
            border: none;
            padding: 8px;
            cursor: pointer;
            color: var(--color-text-muted);
            border-radius: var(--radius-sm);
            transition: var(--transition);
        }

        .modal-close:hover {
            background: var(--color-bg);
            color: var(--color-text);
        }

        .modal-body {
            padding: 0;
        }

        .modal-body iframe {
            width: 100%;
            min-width: 800px;
            height: 70vh;
            border: none;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 48px;
            padding: 24px;
            color: var(--color-text-muted);
            font-size: 0.85rem;
        }

        .footer a {
            color: var(--color-primary);
            text-decoration: none;
        }

        .footer a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </div>
            <h1>ECG Digitizer</h1>
            <p>Transform paper ECGs into digital waveforms</p>
        </header>

        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Input Card -->
            <div class="card">
                <h2 class="card-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="18" height="18" rx="2"/>
                        <circle cx="9" cy="9" r="2"/>
                        <path d="M21 15l-3.086-3.086a2 2 0 00-2.828 0L6 21"/>
                    </svg>
                    Upload ECG
                </h2>

                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                    </div>
                    <p class="upload-title">Drop ECG image here</p>
                    <p class="upload-hint">or click to browse</p>
                    <input type="file" id="fileInput" accept="image/*">
                    <img id="previewImg" class="preview-img" style="display: none;">
                </div>

                <div class="form-group">
                    <label class="form-label">ECG Layout</label>
                    <select class="form-select" id="layout">
                        <option value="">Auto-detect</option>
                        <option value="3x4">3x4 Layout</option>
                        <option value="6x2">6x2 Layout</option>
                        <option value="12x1">12x1 Layout</option>
                    </select>
                </div>

                <!-- Advanced Settings Toggle -->
                <div class="settings-toggle" id="settingsToggle">
                    <span class="settings-toggle-label">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                            <circle cx="12" cy="12" r="3"/>
                            <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
                        </svg>
                        Advanced Settings
                    </span>
                    <svg class="settings-toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M6 9l6 6 6-6"/>
                    </svg>
                </div>

                <div class="settings-content" id="settingsContent">
                    <div class="setting-row">
                        <span class="setting-label">Preprocessing</span>
                        <label class="toggle">
                            <input type="checkbox" id="preprocessingToggle" checked>
                            <span class="toggle-track"></span>
                        </label>
                    </div>

                    <div class="setting-row">
                        <span class="setting-label">Signal Sensitivity</span>
                        <input type="range" class="setting-slider" id="sensitivitySlider" min="0" max="1" step="0.05" value="0.3">
                        <span class="setting-value"><span id="sensitivityValue">0.3</span>x</span>
                    </div>

                    <div class="setting-row">
                        <span class="setting-label">Vertical Margin</span>
                        <input type="range" class="setting-slider" id="marginSlider" min="0" max="0.3" step="0.01" value="0.15">
                        <span class="setting-value"><span id="marginValue">15</span>%</span>
                    </div>

                    <div class="setting-row">
                        <span class="setting-label">Signal Boundaries</span>
                        <label class="toggle">
                            <input type="checkbox" id="signalBoundariesToggle" checked>
                            <span class="toggle-track"></span>
                        </label>
                    </div>

                    <div class="setting-row">
                        <span class="setting-label">Dilation Size</span>
                        <input type="range" class="setting-slider" id="dilationSlider" min="1" max="7" step="2" value="3">
                        <span class="setting-value"><span id="dilationValue">3</span>px</span>
                    </div>

                    <div class="setting-row">
                        <span class="setting-label">Min Width (squares)</span>
                        <input type="range" class="setting-slider" id="widthSlider" min="40" max="55" step="0.5" value="49.5">
                        <span class="setting-value"><span id="widthValue">49.5</span></span>
                    </div>

                    <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--color-border);">
                        <button class="btn-diagnostic" onclick="runDiagnostic()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                                <line x1="12" y1="18" x2="12" y2="12"/>
                                <line x1="9" y1="15" x2="15" y2="15"/>
                            </svg>
                            Generate Diagnostic Report
                        </button>
                    </div>
                </div>

                <button class="btn-primary" id="processBtn" disabled>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    Process ECG
                </button>

                <div class="status" id="status"></div>
            </div>

            <!-- Results Card -->
            <div class="card">
                <h2 class="card-title">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                    Results
                </h2>

                <div id="resultsContainer">
                    <div class="results-placeholder">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                        </svg>
                        <p>Upload an ECG image to see digitized results</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Waveform Card -->
        <div class="card waveform-card" id="waveformCard" style="display: none;">
            <h2 class="card-title">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="20" x2="18" y2="10"/>
                    <line x1="12" y1="20" x2="12" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
                ECG Reconstruction
            </h2>

            <div class="tabs">
                <button class="tab-btn active" data-tab="reconstruction">12-Lead ECG</button>
                <button class="tab-btn" data-tab="chart">Stacked Leads</button>
            </div>

            <div class="tab-content active" id="reconstructionTab">
                <div class="waveform-container">
                    <canvas id="waveformCanvas" class="waveform-canvas"></canvas>
                </div>
            </div>

            <div class="tab-content" id="chartTab">
                <div class="stacked-chart-container" style="height: 800px;">
                    <canvas id="stackedChart"></canvas>
                </div>
            </div>

            <div class="export-buttons">
                <button class="btn-export" onclick="exportCSV()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    Export CSV
                </button>
                <button class="btn-export" onclick="exportJSON()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    Export JSON
                </button>
            </div>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <p>Powered by <a href="https://github.com/Ahus-AIM/Open-ECG-Digitizer" target="_blank">Open-ECG-Digitizer</a></p>
        </footer>
    </div>

    <!-- Diagnostic Modal -->
    <div class="modal-overlay" id="diagnosticModal">
        <div class="modal">
            <div class="modal-header">
                <span class="modal-title">Pipeline Diagnostic Report</span>
                <button class="modal-close" onclick="closeDiagnosticModal()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div class="modal-body" id="diagnosticContent">
                <div style="padding: 60px; text-align: center; color: var(--color-text-muted);">
                    <div class="spinner" style="margin: 0 auto 16px;"></div>
                    <p>Generating diagnostic report...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentResult = null;
        let imageData = null;

        // Elements
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewImg = document.getElementById('previewImg');
        const processBtn = document.getElementById('processBtn');
        const status = document.getElementById('status');
        const settingsToggle = document.getElementById('settingsToggle');
        const settingsContent = document.getElementById('settingsContent');

        // Settings elements
        const sensitivitySlider = document.getElementById('sensitivitySlider');
        const marginSlider = document.getElementById('marginSlider');
        const dilationSlider = document.getElementById('dilationSlider');

        // Settings toggle
        settingsToggle.addEventListener('click', () => {
            settingsToggle.classList.toggle('open');
            settingsContent.classList.toggle('open');
        });

        // Sliders
        const widthSlider = document.getElementById('widthSlider');

        sensitivitySlider.addEventListener('input', () => {
            document.getElementById('sensitivityValue').textContent = sensitivitySlider.value;
        });
        marginSlider.addEventListener('input', () => {
            document.getElementById('marginValue').textContent = Math.round(marginSlider.value * 100);
        });
        dilationSlider.addEventListener('input', () => {
            document.getElementById('dilationValue').textContent = dilationSlider.value;
        });
        widthSlider.addEventListener('input', () => {
            document.getElementById('widthValue').textContent = widthSlider.value;
        });

        // Tab functionality
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(tab + 'Tab').classList.add('active');
                if (tab === 'chart' && stackedChart) stackedChart.resize();
            });
        });

        let stackedChart = null;

        // Upload handling
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });

        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imageData = e.target.result;
                previewImg.src = imageData;
                previewImg.style.display = 'block';
                processBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }

        processBtn.addEventListener('click', processImage);

        async function processImage() {
            if (!imageData) return;

            status.className = 'status loading';
            status.innerHTML = '<div class="spinner"></div> Processing ECG...';
            processBtn.disabled = true;

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image: imageData,
                        layout: document.getElementById('layout').value || null,
                        signal_based_boundaries: document.getElementById('signalBoundariesToggle').checked
                    })
                });

                const result = await response.json();
                currentResult = result;

                if (result.success) {
                    status.className = 'status success';
                    status.textContent = 'Processing complete!';
                    displayResults(result);
                } else {
                    status.className = 'status error';
                    status.textContent = 'Error: ' + result.error;
                }
            } catch (error) {
                status.className = 'status error';
                status.textContent = 'Error: ' + error.message;
            }

            processBtn.disabled = false;
        }

        function displayResults(result) {
            const container = document.getElementById('resultsContainer');

            // Determine algorithm badge color and icon
            const algorithm = result.extraction_algorithm || 'unknown';
            const algorithmDisplay = result.extraction_algorithm_display || algorithm;
            const isPrimary = algorithm === 'advanced' || algorithm === 'server';
            const algoBadgeColor = isPrimary ? '#10B981' : '#F59E0B';  // green vs amber
            const algoIcon = isPrimary ? '✓' : '⚠';

            let html = '<div class="result-stats">';
            html += `<div class="stat-item"><div class="stat-label">Image Size</div><div class="stat-value">${result.image_size.width} x ${result.image_size.height}</div></div>`;
            html += `<div class="stat-item"><div class="stat-label">Layout</div><div class="stat-value">${result.layout || 'Auto'}</div></div>`;
            html += `<div class="stat-item"><div class="stat-label">Leads</div><div class="stat-value">${result.leads.length}</div></div>`;
            html += `<div class="stat-item"><div class="stat-label">Extraction Algorithm</div><div class="stat-value" style="display: flex; align-items: center; gap: 6px;"><span style="background: ${algoBadgeColor}20; color: ${algoBadgeColor}; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">${algoIcon} ${algorithm === 'advanced' ? 'Advanced' : algorithm === 'sectioning' ? 'Sectioning' : algorithm}</span></div></div>`;
            html += `<div class="stat-item"><div class="stat-label">Method</div><div class="stat-value">${result.method || 'standard'}</div></div>`;
            html += '</div>';

            html += '<div class="lead-grid">';
            for (const lead of result.leads) {
                const amplitude = (Math.max(...lead.samples) - Math.min(...lead.samples)).toFixed(2);
                html += `<div class="lead-tag"><div class="lead-tag-name">${lead.name}</div><div class="lead-tag-info">${amplitude} mV</div></div>`;
            }
            html += '</div>';

            container.innerHTML = html;

            document.getElementById('waveformCard').style.display = 'block';
            drawWaveforms(result);
            updateStackedChart(result);
        }

        function updateStackedChart(result) {
            const ctx = document.getElementById('stackedChart').getContext('2d');
            if (stackedChart) stackedChart.destroy();
            if (!result.leads || result.leads.length === 0) return;

            const colors = ['#8B5A3C', '#B85450', '#5A8F5A', '#C4A484', '#6B4430', '#D4A574', '#4A6B8A', '#8A6B4A', '#5A6B8A', '#8A5A6B', '#6B8A5A', '#8A8A5A'];

            // Offset each lead vertically so they can all be seen at once
            const leadSpacing = 3; // mV offset between leads
            const datasets = result.leads.map((lead, idx) => {
                const offset = -idx * leadSpacing;
                const offsetData = lead.samples.map(v => v + offset);
                return {
                    label: lead.name,
                    data: offsetData,
                    borderColor: colors[idx % colors.length],
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                };
            });

            const maxLen = Math.max(...result.leads.map(l => l.samples.length));
            const labels = Array.from({ length: maxLen }, (_, i) => i);

            // Calculate y-axis range
            const yMax = 3;
            const yMin = -(result.leads.length - 1) * leadSpacing - 3;

            stackedChart = new Chart(ctx, {
                type: 'line',
                data: { labels, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                usePointStyle: true,
                                pointStyle: 'line',
                                font: { family: 'Inter', size: 11 },
                                padding: 8
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: ctx => {
                                    const leadIdx = ctx.datasetIndex;
                                    const originalValue = ctx.parsed.y + leadIdx * leadSpacing;
                                    return `${ctx.dataset.label}: ${originalValue.toFixed(3)} mV`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Sample', font: { family: 'Inter' } },
                            ticks: { maxTicksLimit: 10 }
                        },
                        y: {
                            min: yMin,
                            max: yMax,
                            title: { display: false },
                            ticks: { display: false },
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        function drawWaveforms(result) {
            const canvas = document.getElementById('waveformCanvas');
            const ctx = canvas.getContext('2d');

            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);

            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;

            // Background
            ctx.fillStyle = '#FFF8F5';
            ctx.fillRect(0, 0, width, height);

            // Grid - minor
            ctx.strokeStyle = '#F5E6E0';
            ctx.lineWidth = 0.5;
            for (let x = 0; x < width; x += 20) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            }
            for (let y = 0; y < height; y += 20) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            }

            // Grid - major
            ctx.strokeStyle = '#E8D5CC';
            ctx.lineWidth = 1;
            for (let x = 0; x < width; x += 100) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            }
            for (let y = 0; y < height; y += 100) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            }

            // Waveforms
            const leads = result.leads;
            const cols = 4;
            const rows = Math.ceil(leads.length / cols);
            const cellWidth = width / cols;
            const cellHeight = height / rows;

            ctx.strokeStyle = '#8B5A3C';
            ctx.lineWidth = 1.5;
            ctx.font = '600 11px Inter, sans-serif';
            ctx.fillStyle = '#6B4430';

            leads.forEach((lead, idx) => {
                const col = idx % cols;
                const row = Math.floor(idx / cols);
                const x0 = col * cellWidth + 10;
                const y0 = row * cellHeight;
                const w = cellWidth - 20;
                const h = cellHeight;
                const centerY = y0 + h / 2;

                ctx.fillText(lead.name, x0, y0 + 14);

                if (lead.samples.length > 0) {
                    ctx.beginPath();
                    const xScale = w / lead.samples.length;
                    const yScale = h / 4;

                    lead.samples.forEach((v, i) => {
                        const x = x0 + i * xScale;
                        const y = centerY - v * yScale * 20;
                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                }
            });
        }

        function exportCSV() {
            if (!currentResult) return;
            let csv = 'timestamp_ms,lead,voltage_mv\\n';
            for (const lead of currentResult.leads) {
                const interval = lead.duration_ms / lead.samples.length;
                lead.samples.forEach((v, i) => {
                    csv += `${(i * interval).toFixed(1)},${lead.name},${v.toFixed(4)}\\n`;
                });
            }
            downloadFile(csv, 'ecg_data.csv', 'text/csv');
        }

        function exportJSON() {
            if (!currentResult) return;
            downloadFile(JSON.stringify(currentResult, null, 2), 'ecg_data.json', 'application/json');
        }

        function downloadFile(content, filename, type) {
            const blob = new Blob([content], {type});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }

        // Diagnostic functions
        async function runDiagnostic() {
            if (!imageData) {
                alert('Please upload an image first');
                return;
            }

            const modal = document.getElementById('diagnosticModal');
            const content = document.getElementById('diagnosticContent');
            modal.classList.add('active');
            content.innerHTML = `
                <div style="padding: 60px; text-align: center; color: var(--color-text-muted);">
                    <div class="spinner" style="margin: 0 auto 16px;"></div>
                    <p>Generating diagnostic report... This may take 15-30 seconds.</p>
                </div>
            `;

            try {
                const response = await fetch('/api/diagnostic', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: imageData,
                        preprocessing: document.getElementById('preprocessingToggle').checked,
                        signal_sensitivity: parseFloat(sensitivitySlider.value),
                        vertical_margin: parseFloat(marginSlider.value),
                        min_width_squares: parseFloat(widthSlider.value),
                        signal_based_boundaries: document.getElementById('signalBoundariesToggle').checked,
                        dilation_size: parseInt(dilationSlider.value)
                    })
                });

                const data = await response.json();
                if (data.success) {
                    const blob = new Blob([data.html], { type: 'text/html' });
                    content.innerHTML = `<iframe src="${URL.createObjectURL(blob)}"></iframe>`;
                } else {
                    content.innerHTML = `
                        <div style="padding: 40px; text-align: center; color: var(--color-error);">
                            <p><strong>Error:</strong> ${data.error}</p>
                        </div>
                    `;
                }
            } catch (err) {
                content.innerHTML = `
                    <div style="padding: 40px; text-align: center; color: var(--color-error);">
                        <p><strong>Error:</strong> ${err.message}</p>
                    </div>
                `;
            }
        }

        function closeDiagnosticModal() {
            document.getElementById('diagnosticModal').classList.remove('active');
        }

        // Close modal on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeDiagnosticModal();
        });

        // Close modal on overlay click
        document.getElementById('diagnosticModal').addEventListener('click', (e) => {
            if (e.target.id === 'diagnosticModal') closeDiagnosticModal();
        });
    </script>
</body>
</html>
'''


DIAGNOSTIC_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Pipeline Diagnostic</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --color-bg: #FAFAFA;
            --color-surface: #FFFFFF;
            --color-primary: #0066FF;
            --color-primary-dark: #0052CC;
            --color-text: #111827;
            --color-text-muted: #6B7280;
            --color-border: #E5E7EB;
            --radius-md: 10px;
            --radius-lg: 14px;
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --transition: 0.15s ease;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.6;
            padding: 32px;
        }

        .container { max-width: 1000px; margin: 0 auto; }

        h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .subtitle {
            color: var(--color-text-muted);
            margin-bottom: 32px;
        }

        .card {
            background: var(--color-surface);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--color-border);
        }

        .card-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--color-border);
        }

        .upload-area {
            border: 2px dashed var(--color-border);
            border-radius: var(--radius-md);
            padding: 48px 24px;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
            margin-bottom: 20px;
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--color-primary);
            background: rgba(0, 102, 255, 0.04);
        }

        .upload-area input { display: none; }

        .setting-row {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }

        .setting-label {
            flex: 0 0 160px;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .setting-slider {
            flex: 1;
            min-width: 150px;
            height: 6px;
            border-radius: 3px;
            background: var(--color-border);
            appearance: none;
            cursor: pointer;
        }

        .setting-slider::-webkit-slider-thumb {
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--color-primary);
            cursor: pointer;
        }

        .setting-value {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--color-primary);
            background: rgba(0, 102, 255, 0.1);
            padding: 4px 12px;
            border-radius: 6px;
            min-width: 60px;
            text-align: center;
        }

        .toggle {
            position: relative;
            width: 44px;
            height: 24px;
        }

        .toggle input { opacity: 0; width: 0; height: 0; }

        .toggle-track {
            position: absolute;
            inset: 0;
            background: var(--color-border);
            border-radius: 12px;
            cursor: pointer;
            transition: var(--transition);
        }

        .toggle-track::before {
            content: "";
            position: absolute;
            width: 18px;
            height: 18px;
            left: 3px;
            top: 3px;
            background: white;
            border-radius: 50%;
            transition: var(--transition);
        }

        .toggle input:checked + .toggle-track { background: var(--color-primary); }
        .toggle input:checked + .toggle-track::before { transform: translateX(20px); }

        .btn-primary {
            width: 100%;
            padding: 12px 20px;
            background: var(--color-primary);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            font-size: 0.95rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: var(--transition);
        }

        .btn-primary:hover:not(:disabled) {
            background: var(--color-primary-dark);
            box-shadow: 0 4px 12px rgba(0, 102, 255, 0.3);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        #status {
            margin-top: 16px;
            padding: 12px 16px;
            border-radius: var(--radius-md);
            font-size: 0.9rem;
        }

        #report iframe {
            width: 100%;
            height: 800px;
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            margin-top: 24px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pipeline Diagnostic</h1>
        <p class="subtitle">Analyze ECG processing step-by-step</p>

        <div class="card">
            <div class="upload-area" id="uploadArea">
                <p><strong>Drop ECG image here</strong></p>
                <p style="color: var(--color-text-muted); font-size: 0.9rem;">or click to browse</p>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <div class="card-title">Processing Settings</div>

            <div class="setting-row">
                <span class="setting-label">Preprocessing</span>
                <label class="toggle">
                    <input type="checkbox" id="preprocessingToggle" checked>
                    <span class="toggle-track"></span>
                </label>
            </div>

            <div class="setting-row">
                <span class="setting-label">Signal Sensitivity</span>
                <input type="range" class="setting-slider" id="sensitivitySlider" min="0" max="1" step="0.05" value="0.3">
                <span class="setting-value"><span id="sensitivityValue">0.3</span>x</span>
            </div>

            <div class="setting-row">
                <span class="setting-label">Vertical Margin</span>
                <input type="range" class="setting-slider" id="marginSlider" min="0" max="0.3" step="0.01" value="0.15">
                <span class="setting-value"><span id="marginValue">15</span>%</span>
            </div>

            <div class="setting-row">
                <span class="setting-label">Min Width (squares)</span>
                <input type="range" class="setting-slider" id="widthSlider" min="40" max="55" step="0.5" value="49.5">
                <span class="setting-value"><span id="widthValue">49.5</span></span>
            </div>

            <div class="setting-row">
                <span class="setting-label">Signal Boundaries</span>
                <label class="toggle">
                    <input type="checkbox" id="signalBoundariesToggle" checked>
                    <span class="toggle-track"></span>
                </label>
            </div>

            <div class="setting-row">
                <span class="setting-label">Dilation Size</span>
                <input type="range" class="setting-slider" id="dilationSlider" min="1" max="7" step="2" value="3">
                <span class="setting-value"><span id="dilationValue">3</span>px</span>
            </div>

            <button class="btn-primary" id="runBtn" disabled>Generate Diagnostic Report</button>
            <div id="status"></div>
        </div>

        <div id="report"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const runBtn = document.getElementById('runBtn');
        const status = document.getElementById('status');
        const report = document.getElementById('report');
        let selectedFile = null;

        // Sliders
        document.getElementById('sensitivitySlider').addEventListener('input', e => {
            document.getElementById('sensitivityValue').textContent = e.target.value;
        });
        document.getElementById('marginSlider').addEventListener('input', e => {
            document.getElementById('marginValue').textContent = Math.round(e.target.value * 100);
        });
        document.getElementById('widthSlider').addEventListener('input', e => {
            document.getElementById('widthValue').textContent = e.target.value;
        });
        document.getElementById('dilationSlider').addEventListener('input', e => {
            document.getElementById('dilationValue').textContent = e.target.value;
        });

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', e => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });

        function handleFile(file) {
            selectedFile = file;
            uploadArea.innerHTML = `<p><strong>${file.name}</strong></p><p style="color: var(--color-text-muted);">${(file.size/1024/1024).toFixed(2)} MB</p>`;
            runBtn.disabled = false;
        }

        runBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            runBtn.disabled = true;
            status.innerHTML = '<p style="color: var(--color-primary);">Processing... This may take 15-30 seconds.</p>';
            report.innerHTML = '';

            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const response = await fetch('/api/diagnostic', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image: e.target.result,
                            preprocessing: document.getElementById('preprocessingToggle').checked,
                            signal_sensitivity: parseFloat(document.getElementById('sensitivitySlider').value),
                            vertical_margin: parseFloat(document.getElementById('marginSlider').value),
                            min_width_squares: parseFloat(document.getElementById('widthSlider').value),
                            signal_based_boundaries: document.getElementById('signalBoundariesToggle').checked,
                            dilation_size: parseInt(document.getElementById('dilationSlider').value)
                        })
                    });
                    const data = await response.json();

                    if (data.success) {
                        status.innerHTML = '<p style="color: #5A8F5A;">Report generated!</p>';
                        const blob = new Blob([data.html], { type: 'text/html' });
                        report.innerHTML = `<iframe src="${URL.createObjectURL(blob)}"></iframe>`;
                    } else {
                        status.innerHTML = `<p style="color: #B85450;">Error: ${data.error}</p>`;
                    }
                } catch (err) {
                    status.innerHTML = `<p style="color: #B85450;">Error: ${err.message}</p>`;
                }
                runBtn.disabled = false;
            };
            reader.readAsDataURL(selectedFile);
        });
    </script>
</body>
</html>
'''


if __name__ == '__main__':
    import os

    # Get port from environment (for cloud deployment) or use 8080
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'

    print("=" * 60)
    print("ECG Digitizer Web Application")
    print("=" * 60)
    print(f"\nStarting server on port {port}...")
    print(f"Debug mode: {debug}")
    print(f"Open http://localhost:{port} in your browser\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
