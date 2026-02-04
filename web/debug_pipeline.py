#!/usr/bin/env python3
"""
ECG Pipeline Diagnostic Visualizer

Visualizes each step of the Open-ECG-Digitizer pipeline to help debug issues.
Generates an HTML report with all intermediate outputs.
"""

import os
import sys
import base64
import io
import gc
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import scipy.signal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def cleanup_memory():
    """Force garbage collection and clear GPU cache if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import yaml
from digitizer_wrapper import get_digitizer, OPEN_ECG_PATH

# Load layout definitions
LAYOUTS_PATH = os.path.join(OPEN_ECG_PATH, 'src', 'config', 'lead_layouts_all.yml')
LAYOUT_DEFINITIONS = {}
if os.path.exists(LAYOUTS_PATH):
    with open(LAYOUTS_PATH, 'r') as f:
        LAYOUT_DEFINITIONS = yaml.safe_load(f)


def tensor_to_image(tensor, normalize=True):
    """Convert tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dim
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC

    arr = tensor.cpu().numpy()

    if normalize and arr.max() > 1:
        arr = arr / 255.0

    arr = np.clip(arr, 0, 1)

    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
        return Image.fromarray((arr * 255).astype(np.uint8), mode='L')
    else:
        return Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def baseline_wander_removal(samples, sampling_frequency=500):
    """
    Remove baseline wander from ECG signal using two-pass median filtering.

    Step 5d: Baseline wander removal (applied after raw signal extraction)

    Uses cascaded median filters:
    - First pass: 0.2s window (captures high-frequency baseline drift)
    - Second pass: 0.6s window (captures low-frequency baseline drift)

    Args:
        samples: 1D array of ECG samples
        sampling_frequency: Sample rate in Hz (default 500 Hz)

    Returns:
        Baseline-corrected samples
    """
    data = np.array(samples, dtype=np.float64)

    if len(data) < 10:
        return samples

    # First pass: 0.2s window median filter
    win_size_1 = int(np.round(0.2 * sampling_frequency))
    if win_size_1 % 2 == 0:
        win_size_1 += 1  # Must be odd
    win_size_1 = max(3, win_size_1)  # At least 3

    baseline = scipy.signal.medfilt(data, win_size_1)

    # Second pass: 0.6s window median filter
    win_size_2 = int(np.round(0.6 * sampling_frequency))
    if win_size_2 % 2 == 0:
        win_size_2 += 1  # Must be odd
    win_size_2 = max(3, win_size_2)

    baseline = scipy.signal.medfilt(baseline, win_size_2)

    # Remove baseline from signal
    corrected = data - baseline

    return corrected


def run_diagnostic(image_path, output_dir=None, settings=None):
    """
    Run full diagnostic on an ECG image.

    Args:
        image_path: Path to ECG image
        output_dir: Directory to save HTML report (default: same as image)
        settings: Optional dict with processing settings:
            - preprocessing: bool (default True)
            - signal_sensitivity: float (default 0.3)
            - vertical_margin: float (default 0.15)
            - min_width_squares: float (default 49.5)
            - signal_based_boundaries: bool (default True)
            - dilation_size: int (default 3)

    Returns:
        Path to HTML report
    """
    # Default settings
    if settings is None:
        settings = {}
    use_preprocessing = settings.get('preprocessing', True)
    signal_sensitivity = settings.get('signal_sensitivity', 0.1)  # Reduced from 0.3 to recover weak signal
    vertical_margin = settings.get('vertical_margin', 0.15)
    min_width_squares = settings.get('min_width_squares', 49.5)
    use_signal_boundaries = settings.get('signal_based_boundaries', True)
    dilation_size = settings.get('dilation_size', 3)

    if output_dir is None:
        output_dir = os.path.dirname(image_path)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading image: {image_path}")
    print(f"Settings: preprocessing={use_preprocessing}, sensitivity={signal_sensitivity}, "
          f"vertical_margin={vertical_margin}, min_width={min_width_squares}, "
          f"signal_boundaries={use_signal_boundaries}, dilation={dilation_size}")
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    print(f"Image size: {img_array.shape}")

    # Initialize digitizer
    print("Initializing digitizer...")
    digitizer = get_digitizer('cpu')

    # Apply custom settings to digitizer
    digitizer.VERTICAL_MARGIN_PERCENT = vertical_margin
    digitizer.MIN_WIDTH_LARGE_SQUARES = min_width_squares
    digitizer.DILATION_SIZE = dilation_size

    # Patch signal sensitivity with custom threshold
    # Store call count to verify patching works
    patch_call_count = [0]  # Use list to allow modification in closure

    # Note: Must accept 'self' as first arg since it's called as instance method
    def custom_sensitivity_process(self, signal_prob):
        patch_call_count[0] += 1
        mean_val = signal_prob.mean().item()
        max_before = signal_prob.max().item()
        print(f"    [PATCHED FUNCTION CALLED #{patch_call_count[0]}] sensitivity={signal_sensitivity}x, "
              f"mean={mean_val:.4f}, max_before={max_before:.4f}")
        signal_prob = signal_prob - signal_prob.mean() * signal_sensitivity
        signal_prob = torch.clamp(signal_prob, min=0)
        signal_prob = signal_prob / (signal_prob.max() + 1e-9)
        max_after = signal_prob.max().item()
        nonzero_pct = (signal_prob > 0.01).float().mean().item() * 100
        print(f"    [PATCHED FUNCTION RESULT] max={max_after:.4f}, nonzero={nonzero_pct:.1f}%")
        return signal_prob

    import types
    digitizer.wrapper.process_sparse_prob = types.MethodType(custom_sensitivity_process, digitizer.wrapper)
    print(f"  >>> PATCHED process_sparse_prob with sensitivity={signal_sensitivity}")

    # Apply preprocessing conditionally
    if use_preprocessing:
        print("Applying image preprocessing...")
        preprocessed_array = digitizer._preprocess_image(img_array)
        preprocessed_img = Image.fromarray(preprocessed_array)
    else:
        print("Preprocessing disabled")
        preprocessed_array = img_array
        preprocessed_img = img

    # Convert to tensor
    image_tensor = torch.from_numpy(preprocessed_array).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    # Run pipeline with timing
    print("Running pipeline...")
    import time
    start = time.time()
    output = digitizer.wrapper(image_tensor, layout_should_include_substring=None)

    # Store original Hough-detected points for visualization
    hough_source_points = output['source_points'].clone()

    # Compute signal-based boundaries if enabled
    signal_source_points = None
    if use_signal_boundaries:
        print("Computing signal-based boundaries...")
        # Re-get feature maps from input to compute boundaries
        with torch.no_grad():
            input_image = output.get('input_image')
            img_normalized = digitizer.wrapper.min_max_normalize(input_image)
            img_resampled = digitizer.wrapper._resample_image(img_normalized.to(digitizer.device))
            sig_prob, grd_prob, _ = digitizer.wrapper._get_feature_maps(img_resampled)

            signal_source_points = digitizer._compute_signal_based_boundaries(
                sig_prob, grd_prob, threshold=0.1, margin_percent=0.02
            )

            # Merge Hough and signal-based boundaries
            source_points = digitizer._merge_boundaries(hough_source_points, signal_source_points)
            print("  Merged Hough and signal-based boundaries")
    else:
        source_points = output['source_points']

    # Validate and expand boundaries (same as in process_image)
    pixel_spacing = output.get('pixel_spacing_mm', {})
    mm_per_pixel = pixel_spacing.get('x', 0.1)
    if hasattr(mm_per_pixel, 'item'):
        mm_per_pixel = mm_per_pixel.item()

    image_shape = (output['input_image'].shape[2], output['input_image'].shape[3])
    expanded_points, was_expanded = digitizer._validate_and_expand_boundaries(
        source_points, mm_per_pixel, image_shape
    )

    # Store original points for visualization (before expansion)
    original_source_points = source_points.clone()

    # If boundaries were modified, re-run alignment
    if was_expanded or use_signal_boundaries:
        print("Re-running alignment with adjusted boundaries...")
        output = digitizer._run_alignment_with_expanded_boundaries(
            image_tensor, expanded_points.to(digitizer.device), None
        )

    elapsed = time.time() - start
    print(f"Pipeline completed in {elapsed:.2f}s")
    print(f"  >>> Patched function was called {patch_call_count[0]} times")

    # Clear intermediate tensors
    del image_tensor
    cleanup_memory()

    # Build HTML report
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <title>ECG Pipeline Diagnostic - {os.path.basename(image_path)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 20px; background: #f5f5f5; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #007AFF; padding-bottom: 10px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .stat {{ background: #f0f0f0; padding: 10px; border-radius: 4px; margin: 5px 0; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f8f8f8; }}
        .lead-chart {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>ECG Pipeline Diagnostic Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Image: <code>{image_path}</code></p>
    <p>Processing time: <strong>{elapsed:.2f} seconds</strong></p>

    <div class="section" style="background: #e8f4ff;">
        <h3 style="margin-top: 0;">Processing Settings</h3>
        <table style="width: auto;">
            <tr><td>Preprocessing</td><td><strong>{'Enabled' if use_preprocessing else 'Disabled'}</strong></td></tr>
            <tr><td>Signal Sensitivity</td><td><strong>{signal_sensitivity}x mean</strong> (lower = more sensitive)</td></tr>
            <tr><td>Vertical Margin</td><td><strong>{vertical_margin*100:.0f}%</strong> each side</td></tr>
            <tr><td>Min Width</td><td><strong>{min_width_squares}</strong> large squares</td></tr>
            <tr><td>Signal-Based Boundaries</td><td><strong>{'Enabled' if use_signal_boundaries else 'Disabled'}</strong></td></tr>
            <tr><td>Dilation Size</td><td><strong>{dilation_size}x{dilation_size}</strong> pixels</td></tr>
        </table>
    </div>
""")

    # ===== Step 1: Input Image =====
    html_parts.append("""
    <div class="section">
        <h2>Step 1: Input Image</h2>
        <div class="grid">
            <div>
                <h3>Original Image</h3>
    """)
    html_parts.append(f'<img src="data:image/png;base64,{image_to_base64(img)}" alt="Original">')
    html_parts.append(f"""
            <div class="stat">Size: {img_array.shape[1]} x {img_array.shape[0]} pixels</div>
            </div>
            <div>
                <h3>Preprocessed Image</h3>
    """)
    html_parts.append(f'<img src="data:image/png;base64,{image_to_base64(preprocessed_img)}" alt="Preprocessed">')
    html_parts.append(f"""
            <div class="stat">Contrast +20%, Color +10%, Sharpness +30%</div>
            </div>
    """)

    # Resampled input
    input_img = tensor_to_image(output['input_image'])
    html_parts.append("""
            <div>
                <h3>Resampled Input (to model)</h3>
    """)
    html_parts.append(f'<img src="data:image/png;base64,{image_to_base64(input_img)}" alt="Resampled">')
    html_parts.append(f"""
            <div class="stat">Size: {output['input_image'].shape[3]} x {output['input_image'].shape[2]} pixels</div>
            </div>
        </div>
    </div>
    """)

    # ===== Step 2: UNet Segmentation =====
    html_parts.append("""
    <div class="section">
        <h2>Step 2: UNet Segmentation</h2>
        <p>The UNet model segments the image into signal (ECG traces), grid lines, and text regions.</p>
        <div class="grid">
    """)

    aligned = output['aligned']

    # Signal probability map
    if 'signal_prob' in aligned:
        fig, ax = plt.subplots(figsize=(12, 6))
        signal_prob = aligned['signal_prob']
        if signal_prob.dim() == 4:
            signal_prob = signal_prob[0, 0]
        elif signal_prob.dim() == 3:
            signal_prob = signal_prob[0]
        signal_np = signal_prob.cpu().numpy()
        ax.imshow(signal_np, cmap='hot')
        ax.set_title('Signal Probability Map')
        ax.axis('off')
        # Calculate stats
        sig_max = signal_np.max()
        sig_mean = signal_np.mean()
        sig_nonzero = (signal_np > 0.01).sum() / signal_np.size * 100
        html_parts.append(f"""
            <div>
                <h3>Signal Probability</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Signal Prob">
                <div class="stat">Max={sig_max:.3f}, Mean={sig_mean:.4f}, Nonzero={sig_nonzero:.1f}%</div>
                <div class="stat">Sensitivity: {signal_sensitivity}x mean (lower = more signal kept)</div>
            </div>
        """)

    # Dilated signal probability map (if available)
    if 'signal_prob_dilated' in aligned:
        fig, ax = plt.subplots(figsize=(12, 6))
        signal_dilated = aligned['signal_prob_dilated']
        if signal_dilated.dim() == 4:
            signal_dilated = signal_dilated[0, 0]
        elif signal_dilated.dim() == 3:
            signal_dilated = signal_dilated[0]
        dilated_np = signal_dilated.cpu().numpy()
        ax.imshow(dilated_np, cmap='hot')
        ax.set_title('Signal Probability (Dilated)')
        ax.axis('off')
        dil_max = dilated_np.max()
        dil_nonzero = (dilated_np > 0.01).sum() / dilated_np.size * 100
        html_parts.append(f"""
            <div>
                <h3>Signal Prob (Dilated)</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Signal Dilated">
                <div class="stat">Max={dil_max:.3f}, Nonzero={dil_nonzero:.1f}% (3x3 dilation)</div>
                <div class="stat">Used for signal extraction to connect broken traces</div>
            </div>
        """)

    # Grid probability map
    if 'grid_prob' in aligned:
        fig, ax = plt.subplots(figsize=(12, 6))
        grid_prob = aligned['grid_prob']
        if grid_prob.dim() == 4:
            grid_prob = grid_prob[0, 0]
        elif grid_prob.dim() == 3:
            grid_prob = grid_prob[0]
        ax.imshow(grid_prob.cpu().numpy(), cmap='Greens')
        ax.set_title('Grid Probability Map')
        ax.axis('off')
        html_parts.append(f"""
            <div>
                <h3>Grid Probability</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Grid Prob">
                <div class="stat">Shows detected grid line regions</div>
            </div>
        """)

    # Text probability map
    if 'text_prob' in aligned:
        fig, ax = plt.subplots(figsize=(12, 6))
        text_prob = aligned['text_prob']
        if text_prob.dim() == 4:
            text_prob = text_prob[0, 0]
        elif text_prob.dim() == 3:
            text_prob = text_prob[0]
        ax.imshow(text_prob.cpu().numpy(), cmap='Blues')
        ax.set_title('Text Probability Map')
        ax.axis('off')
        html_parts.append(f"""
            <div>
                <h3>Text Probability</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Text Prob">
                <div class="stat">Shows detected text/label regions</div>
            </div>
        """)

    html_parts.append("</div></div>")

    # Cleanup after segmentation visualization
    cleanup_memory()

    # ===== Step 3: Aligned/Dewarped Image =====
    html_parts.append("""
    <div class="section">
        <h2>Step 3: Perspective Correction & Dewarping</h2>
    """)

    if 'image' in aligned:
        aligned_img = tensor_to_image(aligned['image'])
        html_parts.append(f"""
        <div class="grid">
            <div>
                <h3>Aligned Image</h3>
                <img src="data:image/png;base64,{image_to_base64(aligned_img)}" alt="Aligned">
                <div class="stat">Image after perspective correction and dewarping</div>
            </div>
        """)

        # Source points (perspective corners) - show Hough, signal-based, and expanded
        if 'source_points' in output:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(preprocessed_array)

            # Draw Hough-based boundaries in yellow (dashed)
            hough_pts = hough_source_points.cpu().numpy()
            hough_pts_closed = np.vstack([hough_pts, hough_pts[0:1]])
            ax.plot(hough_pts_closed[:, 0], hough_pts_closed[:, 1], 'y--', linewidth=2, label='Hough')

            # Draw signal-based boundaries in cyan (dashed) if computed
            if signal_source_points is not None:
                sig_pts = signal_source_points.cpu().numpy()
                sig_pts_closed = np.vstack([sig_pts, sig_pts[0:1]])
                ax.plot(sig_pts_closed[:, 0], sig_pts_closed[:, 1], 'c--', linewidth=2, label='Signal-based')

            # Draw merged (before expansion) in orange (dotted)
            if use_signal_boundaries:
                merged_pts = original_source_points.cpu().numpy()
                merged_pts_closed = np.vstack([merged_pts, merged_pts[0:1]])
                ax.plot(merged_pts_closed[:, 0], merged_pts_closed[:, 1], 'orange', linewidth=2, linestyle=':', label='Merged')

            # Draw final expanded boundaries in green (solid)
            exp_pts = output['source_points'].cpu().numpy()
            exp_pts_closed = np.vstack([exp_pts, exp_pts[0:1]])
            ax.plot(exp_pts_closed[:, 0], exp_pts_closed[:, 1], 'g-', linewidth=3, label='Final')

            ax.scatter(exp_pts[:, 0], exp_pts[:, 1], c='green', s=100, zorder=5)
            for i, (x, y) in enumerate(exp_pts):
                ax.annotate(f'P{i}', (x, y), fontsize=12, color='white',
                           bbox=dict(boxstyle='round', facecolor='green'))
            ax.set_title('ECG Boundaries Detection')
            ax.legend(loc='upper right')
            ax.axis('off')

            boundary_notes = []
            boundary_notes.append(f"Signal-based: {'Enabled' if use_signal_boundaries else 'Disabled'}")
            boundary_notes.append(f"Min width: {min_width_squares} squares")
            boundary_notes.append(f"Vertical margin: {vertical_margin*100:.0f}%")
            boundary_note = " | ".join(boundary_notes)
            html_parts.append(f"""
            <div>
                <h3>Perspective Detection</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Perspective">
                <div class="stat">{boundary_note}</div>
                <div class="stat">Yellow=Hough, Cyan=Signal-based, Orange=Merged, Green=Final</div>
            </div>
        """)

        html_parts.append("</div>")

    html_parts.append("</div>")

    # ===== Step 4: Grid Calibration =====
    pixel_spacing = output.get('pixel_spacing_mm', {})
    html_parts.append(f"""
    <div class="section">
        <h2>Step 4: Grid Calibration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th><th>Expected</th></tr>
            <tr>
                <td>X spacing (mm/pixel)</td>
                <td>{pixel_spacing.get('x', 'N/A')}</td>
                <td>~0.1-0.3 for typical scans</td>
            </tr>
            <tr>
                <td>Y spacing (mm/pixel)</td>
                <td>{pixel_spacing.get('y', 'N/A')}</td>
                <td>~0.1-0.3 for typical scans</td>
            </tr>
            <tr>
                <td>Pixels per mm (avg)</td>
                <td>{pixel_spacing.get('average_pixel_per_mm', 'N/A')}</td>
                <td>~3-10 for typical scans</td>
            </tr>
        </table>
    </div>
    """)

    # ===== Step 5: Signal Probability Processing =====
    html_parts.append("""
    <div class="section">
        <h2>Step 5: Signal Probability Processing</h2>
        <p>The signal probability map goes through multiple processing steps before signal extraction.</p>
        <div class="grid">
    """)

    # Step 5a: Show processed signal probability (CLAHE + adaptive threshold)
    if 'signal_prob_processed' in aligned:
        fig, ax = plt.subplots(figsize=(12, 6))
        sig_proc = aligned['signal_prob_processed']
        if sig_proc.dim() == 4:
            sig_proc = sig_proc[0, 0]
        elif sig_proc.dim() == 3:
            sig_proc = sig_proc[0]
        sig_proc_np = sig_proc.cpu().numpy()
        ax.imshow(sig_proc_np, cmap='hot')
        ax.set_title('After CLAHE + Adaptive Threshold + Closing')
        ax.axis('off')
        proc_max = sig_proc_np.max()
        proc_nonzero = (sig_proc_np > 0.01).sum() / sig_proc_np.size * 100
        html_parts.append(f"""
            <div>
                <h3>5a: After Post-Processing</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Processed Signal">
                <div class="stat">Max={proc_max:.3f}, Nonzero={proc_nonzero:.1f}%</div>
                <div class="stat">CLAHE (clipLimit=2.0) + Adaptive Threshold (block=11, C=-5) + Morphological Closing (5x5)</div>
            </div>
        """)

    # Step 5b: Show final signal probability (after dilation)
    if 'signal_prob_final' in aligned:
        fig, ax = plt.subplots(figsize=(12, 6))
        sig_final = aligned['signal_prob_final']
        if sig_final.dim() == 4:
            sig_final = sig_final[0, 0]
        elif sig_final.dim() == 3:
            sig_final = sig_final[0]
        sig_final_np = sig_final.cpu().numpy()
        ax.imshow(sig_final_np, cmap='hot')
        ax.set_title(f'After {dilation_size}x{dilation_size} Dilation')
        ax.axis('off')
        final_max = sig_final_np.max()
        final_nonzero = (sig_final_np > 0.01).sum() / sig_final_np.size * 100
        html_parts.append(f"""
            <div>
                <h3>5b: After Dilation</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Final Signal">
                <div class="stat">Max={final_max:.3f}, Nonzero={final_nonzero:.1f}%</div>
                <div class="stat">{dilation_size}x{dilation_size} grey dilation to connect broken traces</div>
            </div>
        """)

    html_parts.append("</div></div>")

    # ===== Step 5c: Raw Signal Extraction =====
    html_parts.append("""
    <div class="section">
        <h2>Step 5c: Raw Signal Extraction</h2>
        <p>The signal extractor scans the processed probability map row-by-row to extract signal traces.</p>
    """)

    raw_lines = output['signal'].get('raw_lines')
    lines = output['signal'].get('lines')

    if raw_lines is not None:
        num_lines = raw_lines.shape[0] if hasattr(raw_lines, 'shape') else len(raw_lines)
        html_parts.append(f"<p><strong>{num_lines} signal lines detected</strong></p>")

        # Plot raw extracted lines with detailed stats
        fig, axes = plt.subplots(num_lines, 1, figsize=(14, 2*num_lines))
        if num_lines == 1:
            axes = [axes]

        line_stats = []
        for i in range(num_lines):
            if hasattr(raw_lines, 'shape'):
                line = raw_lines[i].cpu().numpy()
            else:
                line = raw_lines[i]

            # Remove NaN for plotting
            valid_mask = ~np.isnan(line)
            x = np.arange(len(line))

            axes[i].plot(x[valid_mask], line[valid_mask], 'b-', linewidth=0.5)
            axes[i].set_ylabel(f'Line {i}')
            axes[i].set_xlim(0, len(line))
            axes[i].grid(True, alpha=0.3)

            # Stats
            if valid_mask.sum() > 0:
                valid_line = line[valid_mask]
                min_val = valid_line.min()
                max_val = valid_line.max()
                std_val = valid_line.std()
                axes[i].set_title(f'Line {i}: {valid_mask.sum()} samples, '
                                 f'range [{min_val:.0f}, {max_val:.0f}] µV, std={std_val:.1f}')
                line_stats.append({
                    'line': i,
                    'samples': valid_mask.sum(),
                    'min': min_val,
                    'max': max_val,
                    'std': std_val,
                    'range': max_val - min_val
                })

        plt.tight_layout()
        html_parts.append(f'<img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Raw Lines">')

        # Add a table summarizing line stats
        html_parts.append("""
        <h3>Line Statistics</h3>
        <table>
            <tr><th>Line</th><th>Samples</th><th>Min (µV)</th><th>Max (µV)</th><th>Range (µV)</th><th>Std (µV)</th><th>Has Signal?</th></tr>
        """)
        for stat in line_stats:
            has_signal = "✓ Yes" if stat['std'] > 50 else "⚠ Low variance"
            signal_class = "success" if stat['std'] > 50 else "warning"
            html_parts.append(f"<tr><td>{stat['line']}</td><td>{stat['samples']}</td>"
                             f"<td>{stat['min']:.0f}</td><td>{stat['max']:.0f}</td>"
                             f"<td>{stat['range']:.0f}</td><td>{stat['std']:.1f}</td>"
                             f"<td class='{signal_class}'>{has_signal}</td></tr>")
        html_parts.append("</table>")

    html_parts.append("</div>")

    # ===== Step 5d: Baseline Wander Removal =====
    html_parts.append("""
    <div class="section">
        <h2>Step 5d: Baseline Wander Removal</h2>
        <p>Remove low-frequency baseline drift using two-pass cascaded median filtering (0.2s + 0.6s windows at 500Hz).</p>
    """)

    corrected_lines = []
    if raw_lines is not None:
        num_lines = raw_lines.shape[0] if hasattr(raw_lines, 'shape') else len(raw_lines)

        # Create before/after comparison plots
        fig, axes = plt.subplots(num_lines, 2, figsize=(16, 2.5*num_lines))
        if num_lines == 1:
            axes = axes.reshape(1, -1)

        baseline_stats = []
        for i in range(num_lines):
            if hasattr(raw_lines, 'shape'):
                line = raw_lines[i].cpu().numpy()
            else:
                line = np.array(raw_lines[i])

            # Remove NaN for processing
            valid_mask = ~np.isnan(line)
            x = np.arange(len(line))

            if valid_mask.sum() > 10:
                valid_line = line[valid_mask]
                valid_x = x[valid_mask]

                # Apply baseline wander removal
                corrected = baseline_wander_removal(valid_line, sampling_frequency=500)
                corrected_lines.append(corrected)

                # Before
                axes[i, 0].plot(valid_x, valid_line, 'b-', linewidth=0.5, alpha=0.7)
                axes[i, 0].set_ylabel(f'Line {i}')
                axes[i, 0].set_title(f'Before: mean={valid_line.mean():.1f}, std={valid_line.std():.1f}')
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].axhline(y=valid_line.mean(), color='r', linestyle='--', alpha=0.5, label='mean')

                # After
                axes[i, 1].plot(valid_x, corrected, 'g-', linewidth=0.5)
                axes[i, 1].set_title(f'After: mean={corrected.mean():.1f}, std={corrected.std():.1f}')
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='zero')

                # Compute drift removed
                drift_removed = valid_line.mean() - corrected.mean()
                baseline_stats.append({
                    'line': i,
                    'before_mean': valid_line.mean(),
                    'after_mean': corrected.mean(),
                    'drift_removed': abs(drift_removed),
                    'before_std': valid_line.std(),
                    'after_std': corrected.std()
                })
            else:
                corrected_lines.append(np.array([]))
                axes[i, 0].text(0.5, 0.5, 'No valid samples', ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 1].text(0.5, 0.5, 'No valid samples', ha='center', va='center', transform=axes[i, 1].transAxes)

        axes[0, 0].set_title('Before Baseline Removal', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('After Baseline Removal', fontsize=12, fontweight='bold')
        plt.tight_layout()
        html_parts.append(f'<img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Baseline Wander Removal">')

        # Stats table
        if baseline_stats:
            html_parts.append("""
            <h3>Baseline Removal Statistics</h3>
            <table>
                <tr><th>Line</th><th>Before Mean</th><th>After Mean</th><th>Drift Removed</th><th>Before Std</th><th>After Std</th></tr>
            """)
            for stat in baseline_stats:
                drift_class = "success" if stat['drift_removed'] > 10 else "warning"
                html_parts.append(f"<tr><td>{stat['line']}</td>"
                                 f"<td>{stat['before_mean']:.1f}</td>"
                                 f"<td>{stat['after_mean']:.1f}</td>"
                                 f"<td class='{drift_class}'>{stat['drift_removed']:.1f}</td>"
                                 f"<td>{stat['before_std']:.1f}</td>"
                                 f"<td>{stat['after_std']:.1f}</td></tr>")
            html_parts.append("</table>")
            html_parts.append("<p><em>Note: Drift Removed shows the absolute mean shift. Values > 10 indicate significant baseline correction.</em></p>")

    html_parts.append("</div>")

    # ===== Step 6: Layout Identification =====
    layout_name = output.get('layout_name', 'Unknown')
    matching_cost = output['signal'].get('layout_matching_cost', 'N/A')
    is_flipped = output['signal'].get('layout_is_flipped', False)

    cost_class = 'success' if isinstance(matching_cost, float) and matching_cost < 0.5 else 'warning' if isinstance(matching_cost, float) and matching_cost < 1.0 else 'error'
    matching_cost_str = f"{matching_cost:.3f}" if isinstance(matching_cost, (int, float)) else str(matching_cost)

    # Get layout definition from YAML
    layout_def = LAYOUT_DEFINITIONS.get(layout_name, {})
    layout_rows = layout_def.get('layout', {}).get('rows', '?')
    layout_cols = layout_def.get('layout', {}).get('cols', '?')
    total_rows = layout_def.get('total_rows', '?')
    rhythm_leads = layout_def.get('rhythm_leads', [])
    leads_grid = layout_def.get('leads', [])

    html_parts.append(f"""
    <div class="section">
        <h2>Step 6: Layout Identification & Lead Assignment</h2>

        <h3>6a: Detected Layout Summary</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th><th>Notes</th></tr>
            <tr><td>Detected Layout</td><td><strong>{layout_name}</strong></td><td>{layout_def.get('description', 'No description')}</td></tr>
            <tr><td>Grid Dimensions</td><td>{layout_rows} rows × {layout_cols} cols</td><td>Main ECG grid (excludes rhythm strips)</td></tr>
            <tr><td>Total Expected Rows</td><td>{total_rows}</td><td>Includes rhythm strips if any</td></tr>
            <tr><td>Rhythm Strips</td><td>{len(rhythm_leads)} ({', '.join(rhythm_leads) if rhythm_leads else 'None'})</td><td>Additional full-width leads below grid</td></tr>
            <tr><td>Raw Lines Extracted</td><td>{raw_lines.shape[0] if raw_lines is not None and hasattr(raw_lines, 'shape') else 'N/A'}</td><td>Should match total_rows</td></tr>
            <tr><td>Matching Cost</td><td class="{cost_class}">{matching_cost_str}</td><td>Lower = better (< 0.5 good, > 1.0 poor)</td></tr>
            <tr><td>Is Flipped</td><td>{is_flipped}</td><td>Image was vertically flipped</td></tr>
        </table>
    """)

    # Show layout grid visually
    if leads_grid:
        html_parts.append("""
        <h3>6b: Layout Grid Definition (from YAML)</h3>
        <p>This shows how the detected layout maps grid positions to lead names:</p>
        <table style="text-align: center; font-family: monospace;">
        """)

        # Header row with column numbers
        html_parts.append("<tr><th></th>")
        for col in range(layout_cols if isinstance(layout_cols, int) else 4):
            html_parts.append(f"<th>Col {col}<br><small>(samples {col}×chunk to {col+1}×chunk)</small></th>")
        html_parts.append("</tr>")

        # Grid rows
        if isinstance(leads_grid[0], list):
            # 2D grid (e.g., 3x4)
            for row_idx, row in enumerate(leads_grid):
                html_parts.append(f"<tr><th>Row {row_idx}</th>")
                for lead in row:
                    color = "#28a745" if not lead.startswith("-") else "#ffc107"
                    html_parts.append(f"<td style='background: {color}22; padding: 8px;'><strong>{lead}</strong></td>")
                html_parts.append("</tr>")
        else:
            # 1D grid (e.g., 12x1)
            for row_idx, lead in enumerate(leads_grid):
                html_parts.append(f"<tr><th>Row {row_idx}</th>")
                color = "#28a745" if not lead.startswith("-") else "#ffc107"
                html_parts.append(f"<td style='background: {color}22; padding: 8px;'><strong>{lead}</strong></td>")
                html_parts.append("</tr>")

        # Rhythm strips
        if rhythm_leads:
            for r_idx, r_lead in enumerate(rhythm_leads):
                html_parts.append(f"<tr><th>Rhythm {r_idx}</th>")
                html_parts.append(f"<td colspan='{layout_cols}' style='background: #17a2b822; padding: 8px;'>"
                                 f"<strong>{r_lead}</strong> (full width - matched via cosine similarity)</td></tr>")

        html_parts.append("</table>")

    # Sample length warning for layouts with rhythm strips
    if rhythm_leads:
        grid_samples = 5000 // layout_cols if isinstance(layout_cols, int) else 1250
        html_parts.append(f"""
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 16px; margin: 16px 0;">
            <h4 style="color: #856404; margin: 0 0 8px 0;">⚠️ Sample Length Warning</h4>
            <p style="margin: 0; color: #856404;">
                <strong>Issue:</strong> This layout has {len(rhythm_leads)} rhythm strip(s) which represent <strong>10 seconds</strong> of data (5000 samples at 500 Hz),
                while the grid leads only show <strong>{10/layout_cols:.1f} seconds</strong> each ({grid_samples} samples).
            </p>
            <p style="margin: 8px 0 0 0; color: #856404;">
                <strong>Current behavior:</strong> The system resamples ALL rows to 5000 samples before splitting into columns,
                which incorrectly treats the rhythm strip as having the same duration per column as the grid rows.
                This causes the rhythm strip data to be quartered instead of kept whole.
            </p>
            <p style="margin: 8px 0 0 0; color: #856404;">
                <strong>Expected:</strong> Grid rows should have {grid_samples} samples per lead. Rhythm strip should have 5000 samples for one lead.
            </p>
        </div>
        """)

    # Show the lines tensor if available
    lines_tensor = output['signal'].get('lines')
    if lines_tensor is not None:
        html_parts.append("""
        <h3>6c: Normalized Lines (after resampling)</h3>
        <p>These are the signal lines after normalization and resampling to 5000 samples. Each row represents one extracted signal line.</p>
        """)

        if hasattr(lines_tensor, 'shape'):
            lines_shape = lines_tensor.shape
            html_parts.append(f"<div class='stat'>Lines tensor shape: {lines_shape}</div>")
            html_parts.append(f"<div class='stat'>Each line has {lines_shape[1]} samples (resampled from original pixel width)</div>")

            # Plot the intermediate lines
            if len(lines_shape) >= 2:
                num_intermediate = lines_shape[0]
                fig, axes = plt.subplots(min(num_intermediate, 12), 1, figsize=(14, 2*min(num_intermediate, 12)))
                if num_intermediate == 1:
                    axes = [axes]

                # Determine which leads go in which row based on layout
                row_labels = []
                if isinstance(leads_grid, list) and len(leads_grid) > 0:
                    if isinstance(leads_grid[0], list):
                        for row in leads_grid:
                            row_labels.append(', '.join(row))
                    else:
                        row_labels = leads_grid
                for r in rhythm_leads:
                    row_labels.append(f"Rhythm: {r}")

                for i in range(min(num_intermediate, 12)):
                    line = lines_tensor[i].cpu().numpy() if hasattr(lines_tensor[i], 'cpu') else lines_tensor[i]
                    if len(line.shape) > 1:
                        line = line.flatten()
                    valid_mask = ~np.isnan(line)
                    x = np.arange(len(line))

                    axes[i].plot(x[valid_mask], line[valid_mask], 'g-', linewidth=0.5)

                    # Add column dividers for multi-column layouts
                    if isinstance(layout_cols, int) and layout_cols > 1:
                        chunk_width = len(line) // layout_cols
                        for col in range(1, layout_cols):
                            axes[i].axvline(x=col * chunk_width, color='red', linestyle='--', alpha=0.5)

                    row_label = row_labels[i] if i < len(row_labels) else f'Row {i}'
                    axes[i].set_ylabel(f'Row {i}')
                    axes[i].set_xlim(0, len(line))
                    axes[i].grid(True, alpha=0.3)

                    if valid_mask.sum() > 0:
                        valid_line = line[valid_mask]
                        axes[i].set_title(f'Row {i} → [{row_label}]: {valid_mask.sum()} samples, '
                                         f'range [{valid_line.min():.0f}, {valid_line.max():.0f}] µV')

                plt.tight_layout()
                html_parts.append(f'<img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Normalized Lines">')
                html_parts.append("<p><em>Red dashed lines show column boundaries where signals are split into individual leads.</em></p>")
    else:
        html_parts.append("<div class='stat warning'>Lines tensor not available - layout matching may have failed</div>")

    # Show the canonicalization mapping
    html_parts.append("""
    <h3>6d: Grid Position to Canonical Lead Mapping</h3>
    <p>This shows how each grid position maps to a canonical 12-lead index:</p>
    """)

    LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    html_parts.append("""
    <table>
        <tr><th>Grid Position</th><th>Lead Name</th><th>Canonical Index</th><th>Sample Range</th></tr>
    """)

    if isinstance(leads_grid, list) and len(leads_grid) > 0:
        cols = layout_cols if isinstance(layout_cols, int) else 1
        chunk_width = 5000 // cols

        if isinstance(leads_grid[0], list):
            for row_idx, row in enumerate(leads_grid):
                for col_idx, lead in enumerate(row):
                    lead_name = lead.lstrip('-')
                    canon_idx = LEAD_ORDER.index(lead_name) if lead_name in LEAD_ORDER else '?'
                    start = col_idx * chunk_width
                    end = (col_idx + 1) * chunk_width if col_idx < cols - 1 else 5000
                    inverted = "⚠️ inverted" if lead.startswith('-') else ""
                    html_parts.append(f"<tr><td>Row {row_idx}, Col {col_idx}</td><td><strong>{lead}</strong> {inverted}</td>"
                                     f"<td>{canon_idx}</td><td>{start} - {end}</td></tr>")
        else:
            for row_idx, lead in enumerate(leads_grid):
                lead_name = lead.lstrip('-')
                canon_idx = LEAD_ORDER.index(lead_name) if lead_name in LEAD_ORDER else '?'
                inverted = "⚠️ inverted" if lead.startswith('-') else ""
                html_parts.append(f"<tr><td>Row {row_idx}</td><td><strong>{lead}</strong> {inverted}</td>"
                                 f"<td>{canon_idx}</td><td>0 - 5000</td></tr>")

        # Rhythm strips
        for r_idx, r_lead in enumerate(rhythm_leads):
            html_parts.append(f"<tr><td>Rhythm {r_idx}</td><td><strong>{r_lead}</strong></td>"
                             f"<td>Matched via similarity</td><td>0 - 5000 (⚠️ full width)</td></tr>")

    html_parts.append("</table>")

    # ===== Step 6e: Sectioned Lead Visualization =====
    html_parts.append("""
        <h3>6e: Sectioned Lead Visualization (3×4 Grid)</h3>
        <p>Each row is divided into 4 columns, showing the individual lead waveforms after sectioning:</p>
    """)

    if lines_tensor is not None and isinstance(leads_grid, list) and len(leads_grid) > 0:
        num_rows = min(lines_tensor.shape[0], 3) if hasattr(lines_tensor, 'shape') else 0
        cols = layout_cols if isinstance(layout_cols, int) else 4

        if num_rows > 0 and isinstance(leads_grid[0], list):
            fig, axes = plt.subplots(num_rows, cols, figsize=(16, 2.5 * num_rows))
            if num_rows == 1:
                axes = axes.reshape(1, -1)

            for row_idx in range(num_rows):
                if hasattr(lines_tensor, 'shape'):
                    line = lines_tensor[row_idx].cpu().numpy() if hasattr(lines_tensor[row_idx], 'cpu') else lines_tensor[row_idx]
                else:
                    line = np.array(lines_tensor[row_idx])

                if len(line.shape) > 1:
                    line = line.flatten()

                chunk_width = len(line) // cols

                for col_idx in range(cols):
                    ax = axes[row_idx, col_idx]
                    start = col_idx * chunk_width
                    end = (col_idx + 1) * chunk_width if col_idx < cols - 1 else len(line)
                    section = line[start:end]

                    # Get lead name from grid
                    lead_name = leads_grid[row_idx][col_idx] if row_idx < len(leads_grid) and col_idx < len(leads_grid[row_idx]) else f"R{row_idx}C{col_idx}"

                    # Remove NaN for plotting
                    valid_mask = ~np.isnan(section)
                    x = np.arange(len(section))

                    if valid_mask.sum() > 0:
                        ax.plot(x[valid_mask], section[valid_mask], 'g-', linewidth=0.8)
                        valid_section = section[valid_mask]
                        ax.set_ylim(valid_section.min() - 10, valid_section.max() + 10)
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

                    ax.set_title(f'{lead_name}', fontweight='bold', fontsize=10)
                    ax.set_xlim(0, len(section))
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel(f'{len(section)} samples')

                    if col_idx == 0:
                        ax.set_ylabel(f'Row {row_idx}')

            plt.suptitle('Step 6e: Individual Leads After Sectioning', fontsize=12, fontweight='bold')
            plt.tight_layout()
            html_parts.append(f'<img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Sectioned Leads">')
            plt.close(fig)
        else:
            html_parts.append("<div class='stat warning'>Cannot generate sectioned view - layout is not 2D grid</div>")
    else:
        html_parts.append("<div class='stat warning'>Lines tensor not available for sectioned visualization</div>")

    html_parts.append("</div>")

    # ===== Step 7: Canonical Output =====
    html_parts.append("""
    <div class="section">
        <h2>Step 7: Canonical 12-Lead Output</h2>
        <p>The final step maps the extracted signal lines to the standard 12-lead format.</p>
    """)

    canonical = output['signal'].get('canonical_lines')
    lines_tensor = output['signal'].get('lines')
    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Calculate expected sample lengths based on layout
    grid_cols = layout_cols if isinstance(layout_cols, int) else 4
    grid_rows_count = layout_rows if isinstance(layout_rows, int) else 3
    expected_grid_samples = 5000 // grid_cols  # e.g., 1250 for 4 columns
    expected_rhythm_samples = 5000  # Full width for rhythm strips

    # Identify which canonical leads come from rhythm strips
    rhythm_lead_indices = set()
    for r_lead in rhythm_leads:
        if r_lead in LEAD_NAMES:
            rhythm_lead_indices.add(LEAD_NAMES.index(r_lead))

    if canonical is not None:
        # Show tensor info
        can_shape = canonical.shape if hasattr(canonical, 'shape') else 'Unknown'
        html_parts.append(f"<div class='stat'>Canonical tensor shape: {can_shape}</div>")

        # Expected sample lengths table
        html_parts.append("<h3>7a: Expected Sample Lengths by Layout</h3>")
        html_parts.append(f"""
        <div style="background: #e7f3ff; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #0066FF;">
            <p style="margin: 0;"><strong>Layout: {layout_name}</strong></p>
            <p style="margin: 8px 0 0 0;">
                <strong>Grid leads (11 of 12):</strong> {expected_grid_samples} samples each ({10/grid_cols:.1f} seconds at 500 Hz)
            </p>
            <p style="margin: 8px 0 0 0;">
                <strong>Rhythm strip leads ({len(rhythm_leads)}):</strong> {expected_rhythm_samples} samples ({', '.join(rhythm_leads) if rhythm_leads else 'None'}) = 10 seconds
            </p>
        </div>
        """)

        # ===== Step 7b: Rhythm Strip Matching Details =====
        if len(rhythm_leads) > 0 and lines_tensor is not None:
            html_parts.append("<h3>7b: Rhythm Strip Matching (Cosine Similarity)</h3>")
            html_parts.append("""
            <p>The rhythm strip is matched to a canonical lead using cosine similarity.
            Common rhythm leads (II, V1, V5) get an inflated similarity score.</p>
            """)

            # Compute cosine similarities between rhythm strip row and canonical leads
            num_rhythm = len(rhythm_leads)
            num_total_rows = lines_tensor.shape[0] if hasattr(lines_tensor, 'shape') else 0

            if num_total_rows > 0:
                html_parts.append("<table><tr><th>Rhythm Strip</th><th>Expected Lead</th>")
                for lead in LEAD_NAMES:
                    html_parts.append(f"<th>{lead}</th>")
                html_parts.append("<th>Best Match</th></tr>")

                for r_idx, r_lead_name in enumerate(rhythm_leads):
                    rhythm_row_idx = num_total_rows - num_rhythm + r_idx
                    if rhythm_row_idx < lines_tensor.shape[0]:
                        rhythm_vec = lines_tensor[rhythm_row_idx]
                        rhythm_vec_np = rhythm_vec.cpu().numpy() if hasattr(rhythm_vec, 'cpu') else rhythm_vec

                        html_parts.append(f"<tr><td>Row {rhythm_row_idx}</td><td><strong>{r_lead_name}</strong></td>")

                        similarities = []
                        for j, lead_name in enumerate(LEAD_NAMES):
                            # Compute cosine similarity with canonical lead
                            canonical_vec = canonical[j].cpu().numpy() if hasattr(canonical[j], 'cpu') else canonical[j]

                            # Handle NaN values
                            r_valid = ~np.isnan(rhythm_vec_np)
                            c_valid = ~np.isnan(canonical_vec)
                            both_valid = r_valid & c_valid

                            if both_valid.sum() > 100:
                                r_vals = rhythm_vec_np[both_valid]
                                c_vals = canonical_vec[both_valid]
                                # Cosine similarity
                                dot = np.dot(r_vals, c_vals)
                                norm_r = np.linalg.norm(r_vals)
                                norm_c = np.linalg.norm(c_vals)
                                if norm_r > 0 and norm_c > 0:
                                    sim = dot / (norm_r * norm_c)
                                else:
                                    sim = 0.0
                            else:
                                sim = 0.0

                            similarities.append((lead_name, sim))

                            # Apply inflation for common rhythm leads (II, V1, V5)
                            inflated = ""
                            if (num_rhythm == 1 and lead_name == "II") or \
                               (num_rhythm == 2 and lead_name in ["II", "V1"]) or \
                               (num_rhythm == 3 and lead_name in ["II", "V1", "V5"]):
                                inflated = " ⬆"

                            # Color coding
                            if sim > 0.8:
                                color = "#28a745"  # green
                            elif sim > 0.5:
                                color = "#ffc107"  # yellow
                            else:
                                color = "#dc3545"  # red

                            html_parts.append(f"<td style='background-color: {color}20; color: {color};'>{sim:.3f}{inflated}</td>")

                        # Find best match
                        best_lead, best_sim = max(similarities, key=lambda x: x[1])
                        html_parts.append(f"<td><strong>{best_lead}</strong> ({best_sim:.3f})</td></tr>")

                html_parts.append("</table>")
                html_parts.append("<p><em>⬆ = similarity inflated for common rhythm lead. Higher similarity = better match.</em></p>")

            # Warning about current behavior
            html_parts.append(f"""
            <div style="background: #fff3cd; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #856404;">
                <p style="margin: 0; color: #856404;">
                    <strong>⚠️ Current Issue:</strong> After matching, the rhythm strip row (5000 samples) overwrites the canonical lead.
                    However, the grid leads for that same canonical position had only {expected_grid_samples} samples.
                </p>
                <p style="margin: 8px 0 0 0; color: #856404;">
                    <strong>Expected behavior:</strong> The matched lead ({rhythm_leads[0] if rhythm_leads else '?'}) should retain the full 5000 samples from the rhythm strip,
                    while other leads keep their {expected_grid_samples} sample chunks.
                </p>
            </div>
            """)

        # Summary table with ORIGINAL valid sample lengths
        html_parts.append("<h3>7c: Lead Summary with Original Sample Lengths</h3>")
        html_parts.append("<p><strong>Note:</strong> Valid column shows actual non-NaN samples before resampling. Expected column shows correct length for this layout.</p>")
        html_parts.append("<table><tr><th>Lead</th><th>Total</th><th>Valid</th><th>Expected</th><th>Source</th><th>Min (mV)</th><th>Max (mV)</th><th>Range (mV)</th><th>Std (mV)</th><th>Status</th></tr>")

        leads_with_data = 0
        lead_problems = []
        for i, name in enumerate(LEAD_NAMES):
            row = canonical[i].cpu().numpy()
            valid_mask = ~np.isnan(row)
            valid_count = valid_mask.sum()

            # Determine expected samples and source based on whether this is a rhythm strip lead
            is_rhythm_lead = i in rhythm_lead_indices
            expected_samples = expected_rhythm_samples if is_rhythm_lead else expected_grid_samples
            source = "🎵 Rhythm Strip" if is_rhythm_lead else "📊 Grid"

            # Check if valid count matches expected
            sample_match = "✓" if abs(valid_count - expected_samples) < 50 else "⚠️"

            if valid_count > 0:
                valid_data = row[valid_mask] / 1000.0  # Convert µV to mV
                min_mv = valid_data.min()
                max_mv = valid_data.max()
                range_mv = max_mv - min_mv
                std_mv = valid_data.std()

                # Check for physiological plausibility
                is_physiological = -5 < min_mv < 5 and -5 < max_mv < 5
                has_variance = std_mv > 0.01
                has_good_range = 0.1 < range_mv < 6  # Typical ECG range

                if has_variance and is_physiological and has_good_range:
                    leads_with_data += 1
                    status = '<span class="success">✓ Good</span>'
                elif not is_physiological:
                    status = '<span class="error">✗ Out of range</span>'
                    lead_problems.append(f"{name}: values outside ±5mV")
                elif not has_variance:
                    status = '<span class="warning">⚠ Flat signal</span>'
                    lead_problems.append(f"{name}: std={std_mv:.4f}mV (too flat)")
                else:
                    status = '<span class="warning">⚠ Low amplitude</span>'
                    lead_problems.append(f"{name}: range={range_mv:.3f}mV")

                # Row highlighting for rhythm strip leads
                row_style = "background-color: #e7f3ff;" if is_rhythm_lead else ""

                html_parts.append(f"<tr style='{row_style}'><td><strong>{name}</strong></td><td>{len(row)}</td><td>{valid_count} {sample_match}</td>"
                                 f"<td>{expected_samples}</td><td>{source}</td>"
                                 f"<td>{min_mv:+.3f}</td><td>{max_mv:+.3f}</td><td>{range_mv:.3f}</td><td>{std_mv:.4f}</td><td>{status}</td></tr>")
            else:
                row_style = "background-color: #e7f3ff;" if is_rhythm_lead else ""
                html_parts.append(f"<tr style='{row_style}'><td><strong>{name}</strong></td><td>{len(row)}</td><td>0</td>"
                                 f"<td>{expected_samples}</td><td>{source}</td>"
                                 f"<td>-</td><td>-</td><td>-</td><td>-</td><td><span class='error'>✗ No data</span></td></tr>")
                lead_problems.append(f"{name}: no valid data")

        html_parts.append("</table>")
        html_parts.append(f"<p><strong>Leads with meaningful data: {leads_with_data}/12</strong></p>")

        # Show problems if any
        if lead_problems:
            html_parts.append("<h3>7d: Potential Issues Detected</h3>")
            html_parts.append("<ul style='color: #dc3545;'>")
            for prob in lead_problems:
                html_parts.append(f"<li>{prob}</li>")
            html_parts.append("</ul>")

        # Plot all 12 leads
        html_parts.append("<h3>7e: 12-Lead ECG Visualization</h3>")
        html_parts.append(f"<p><em>Blue highlighted rows = Rhythm strip leads ({', '.join(rhythm_leads) if rhythm_leads else 'None'}). Expected to have {expected_rhythm_samples} samples.</em></p>")
        fig, axes = plt.subplots(6, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, name in enumerate(LEAD_NAMES):
            row = canonical[i].cpu().numpy() / 1000.0  # µV to mV
            valid_mask = ~np.isnan(row)
            valid_count = valid_mask.sum()
            x = np.arange(len(row)) / 500.0  # Assuming 500 Hz = seconds

            # Determine expected samples based on whether this is a rhythm strip lead
            is_rhythm_lead = i in rhythm_lead_indices
            expected_samples = expected_rhythm_samples if is_rhythm_lead else expected_grid_samples
            source_indicator = "🎵" if is_rhythm_lead else ""

            ax = axes[i]
            if valid_count > 0:
                # Use different color for rhythm strip leads
                line_color = '#0066FF' if is_rhythm_lead else '#28a745'
                ax.plot(x[valid_mask], row[valid_mask], color=line_color, linewidth=0.5)
                ax.set_ylim(-1, 1)  # Changed from -3 to +3 to -1 to +1 mV
            ax.set_title(f'{name} {source_indicator} ({valid_count} samples, expected {expected_samples})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('mV')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        html_parts.append(f'<img src="data:image/png;base64,{fig_to_base64(fig)}" alt="12 Lead ECG">')

    html_parts.append("</div>")

    # ===== Summary =====
    confidence_str = f"{(1.0 - matching_cost):.1%}" if isinstance(matching_cost, (int, float)) else "N/A"
    lines_extracted = raw_lines.shape[0] if hasattr(raw_lines, 'shape') else 'N/A'
    html_parts.append(f"""
    <div class="section">
        <h2>Summary</h2>
        <ul>
            <li>Layout detected: <strong>{layout_name}</strong></li>
            <li>Layout confidence: <strong class="{cost_class}">{confidence_str} ({matching_cost_str} cost)</strong></li>
            <li>Signal lines extracted: <strong>{lines_extracted}</strong></li>
            <li>Leads with meaningful data: <strong>{leads_with_data}/12</strong></li>
            <li>Processing time: <strong>{elapsed:.2f}s</strong></li>
        </ul>
    </div>
    """)

    html_parts.append("</body></html>")

    # Write HTML report
    report_name = f"ecg_diagnostic_{os.path.splitext(os.path.basename(image_path))[0]}.html"
    report_path = os.path.join(output_dir, report_name)

    with open(report_path, 'w') as f:
        f.write('\n'.join(html_parts))

    # Cleanup memory before returning
    del html_parts
    del output
    del digitizer
    cleanup_memory()

    print(f"\nDiagnostic report saved to: {report_path}")
    return report_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ECG Pipeline Diagnostic Visualizer')
    parser.add_argument('image', help='Path to ECG image')
    parser.add_argument('--output', '-o', help='Output directory for report')

    args = parser.parse_args()

    run_diagnostic(args.image, args.output)
