"""
Proper wrapper around Open-ECG-Digitizer InferenceWrapper.

This module uses the COMPLETE pipeline from the original repo instead of
reimplementing it piecemeal. The InferenceWrapper handles all 8 steps:
1. Image normalization and resampling
2. UNet segmentation (signal/grid/text maps)
3. Perspective detection (Hough transform)
4. Cropping with perspective correction
5. Grid size extraction (autocorrelation)
6. Dewarping (TPS-based)
7. Signal extraction (iterative with line merging)
8. Layout identification and canonical output

Output: 12-lead ECG signals in µV, shape (12, 5000)
"""

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import grey_dilation, binary_dilation, binary_closing

# Add Open-ECG-Digitizer to path
OPEN_ECG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Open-ECG-Digitizer')
sys.path.insert(0, OPEN_ECG_PATH)

# Mock ray.tune to avoid import error (only needed for training, not inference)
import importlib.util
if importlib.util.find_spec('ray') is None:
    import types
    ray_mock = types.ModuleType('ray')
    ray_tune_mock = types.ModuleType('ray.tune')
    ray_tune_mock.Stopper = object  # Dummy class
    ray_mock.tune = ray_tune_mock
    sys.modules['ray'] = ray_mock
    sys.modules['ray.tune'] = ray_tune_mock

# Now import from Open-ECG-Digitizer
try:
    from src.model.inference_wrapper import InferenceWrapper
    from src.config.default import get_cfg
    HAS_INFERENCE_WRAPPER = True
except ImportError as e:
    print(f"Warning: Could not import InferenceWrapper: {e}")
    import traceback
    traceback.print_exc()
    HAS_INFERENCE_WRAPPER = False


class ECGDigitizerWrapper:
    """
    Wrapper around the Open-ECG-Digitizer InferenceWrapper.

    Uses the complete pipeline from the original repo for accurate ECG digitization.
    """

    # Minimum width in large squares (each large square = 200ms)
    # Standard 12-lead ECG is 10 seconds = 50 large squares
    MIN_WIDTH_LARGE_SQUARES = 49.5

    # Vertical margin expansion (15% on each side = 30% total)
    VERTICAL_MARGIN_PERCENT = 0.15

    # Morphological dilation kernel size for signal probability
    DILATION_SIZE = 3

    def __init__(self, device: str = 'cpu'):
        """
        Initialize the ECG digitizer.

        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        if not HAS_INFERENCE_WRAPPER:
            raise RuntimeError("Open-ECG-Digitizer not available. Check installation.")

        self.device = device
        self.wrapper = None
        self._initialize_wrapper()

    def _initialize_wrapper(self):
        """Load config and initialize InferenceWrapper."""
        # Load config from YAML
        config_path = os.path.join(OPEN_ECG_PATH, 'src/config/inference_wrapper.yml')
        cfg = get_cfg(config_path)

        # Get the model config (this is what InferenceWrapper expects)
        model_kwargs = cfg.MODEL.KWARGS

        # Fix all paths to be absolute
        weights_dir = os.path.join(OPEN_ECG_PATH, 'weights')
        config_dir = os.path.join(OPEN_ECG_PATH, 'src/config')

        # Fix segmentation model weight path
        model_kwargs.config.SEGMENTATION_MODEL.weight_path = os.path.join(
            weights_dir, 'unet_weights_07072025.pt'
        )

        # Fix layout identifier paths - use lead_layouts_all.yml for full layout support
        model_kwargs.config.LAYOUT_IDENTIFIER.config_path = os.path.join(
            config_dir, 'lead_layouts_all.yml'
        )
        model_kwargs.config.LAYOUT_IDENTIFIER.unet_config_path = os.path.join(
            config_dir, 'lead_name_unet.yml'
        )
        model_kwargs.config.LAYOUT_IDENTIFIER.unet_weight_path = os.path.join(
            weights_dir, 'lead_name_unet_weights_07072025.pt'
        )

        # Override device settings
        model_kwargs.device = self.device
        model_kwargs.config.LAYOUT_IDENTIFIER.KWARGS.device = self.device

        # Enable timing for debugging (optional)
        model_kwargs.enable_timing = False

        # Fix cropper to use wider boundaries - original [0.02, 0.98] crops too tight
        # Using [0.001, 0.999] to include 99.8% of signal probability mass
        model_kwargs.config.CROPPER.KWARGS.percentiles = [0.001, 0.999]
        # Lower alpha means less aggressive cropping (more towards image center)
        model_kwargs.config.CROPPER.KWARGS.alpha = 0.75

        # Tune SignalExtractor for better signal extraction
        # Lower thresholds to catch weaker signals (reduced further to fix undercalling)
        model_kwargs.config.SIGNAL_EXTRACTOR.KWARGS.label_thresh = 0.02  # Was 0.05, default 0.1
        model_kwargs.config.SIGNAL_EXTRACTOR.KWARGS.threshold_sum = 2.0  # Was 5.0, default 10.0
        model_kwargs.config.SIGNAL_EXTRACTOR.KWARGS.threshold_line_in_mask = 0.75  # Was 0.85, default 0.95
        model_kwargs.config.SIGNAL_EXTRACTOR.KWARGS.min_line_width = 20  # Default 30
        model_kwargs.config.SIGNAL_EXTRACTOR.KWARGS.max_iterations = 6  # Default 4
        model_kwargs.config.SIGNAL_EXTRACTOR.KWARGS.candidate_span = 15  # Default 10

        # Initialize the real InferenceWrapper
        self.wrapper = InferenceWrapper(
            config=model_kwargs.config,
            device=model_kwargs.device,
            resample_size=model_kwargs.resample_size,
            rotate_on_resample=model_kwargs.rotate_on_resample,
            enable_timing=model_kwargs.enable_timing,
            apply_dewarping=model_kwargs.apply_dewarping,
        )

        # Monkey-patch the signal probability processing to be more sensitive
        # Original uses mean * 1.0 as threshold, we use 0.7 to include more signal
        self._patch_signal_sensitivity()

        print(f"ECGDigitizerWrapper initialized on device: {self.device}")

    def _patch_signal_sensitivity(self):
        """
        Patch the InferenceWrapper's process_sparse_prob to use advanced post-processing.

        Uses:
        - Histogram equalization (CLAHE) for contrast stretching
        - Adaptive thresholding instead of fixed threshold
        - Morphological closing to fill gaps
        """
        import types

        # Keep reference to outer self for accessing _postprocess_signal_prob
        outer_self = self

        # Note: Must accept 'self_wrapper' as first arg since it's called as instance method
        def advanced_process(self_wrapper, signal_prob):
            # Apply advanced post-processing pipeline
            return outer_self._postprocess_signal_prob(signal_prob)

        self.wrapper.process_sparse_prob = types.MethodType(advanced_process, self.wrapper)
        print("  Signal processing: CLAHE + adaptive threshold + morphological closing")

    def _dilate_signal_prob(self, signal_prob: torch.Tensor, dilation_size: int = 3) -> torch.Tensor:
        """
        Apply morphological dilation to signal probability to connect broken traces.

        Args:
            signal_prob: Signal probability tensor (can be 2D, 3D, or 4D)
            dilation_size: Size of dilation kernel (default 3x3)

        Returns:
            Dilated signal probability tensor
        """
        # Handle different tensor dimensions
        original_shape = signal_prob.shape
        if signal_prob.dim() == 4:
            prob_2d = signal_prob[0, 0].cpu().numpy()
        elif signal_prob.dim() == 3:
            prob_2d = signal_prob[0].cpu().numpy()
        else:
            prob_2d = signal_prob.cpu().numpy()

        # Apply grey dilation (preserves probability values, expands high regions)
        dilated = grey_dilation(prob_2d, size=(dilation_size, dilation_size))

        # Convert back to tensor with original shape
        dilated_tensor = torch.from_numpy(dilated).float()
        if signal_prob.dim() == 4:
            dilated_tensor = dilated_tensor.unsqueeze(0).unsqueeze(0)
        elif signal_prob.dim() == 3:
            dilated_tensor = dilated_tensor.unsqueeze(0)

        return dilated_tensor.to(signal_prob.device)

    def _postprocess_signal_prob(self, signal_prob: torch.Tensor) -> torch.Tensor:
        """
        Apply advanced post-processing to signal probability map.

        Processing steps:
        1. Contrast stretching via histogram equalization
        2. Adaptive thresholding instead of fixed threshold
        3. Morphological closing to fill small gaps

        Args:
            signal_prob: Signal probability tensor (can be 2D, 3D, or 4D)

        Returns:
            Post-processed signal probability tensor
        """
        # Handle different tensor dimensions
        if signal_prob.dim() == 4:
            prob_2d = signal_prob[0, 0].cpu().numpy()
        elif signal_prob.dim() == 3:
            prob_2d = signal_prob[0].cpu().numpy()
        else:
            prob_2d = signal_prob.cpu().numpy()

        # Step 1: Contrast stretching via histogram equalization
        # Convert to 8-bit for OpenCV operations
        prob_normalized = (prob_2d - prob_2d.min()) / (prob_2d.max() - prob_2d.min() + 1e-9)
        prob_8bit = (prob_normalized * 255).astype(np.uint8)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is better than standard histogram equalization for local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        prob_equalized = clahe.apply(prob_8bit)

        # Step 2: Adaptive thresholding
        # Use Gaussian adaptive threshold - better for varying illumination
        # blockSize must be odd; smaller = better for thin traces
        # C is subtracted from the mean; more negative = more sensitive to faint signals
        adaptive_thresh = cv2.adaptiveThreshold(
            prob_equalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=7,  # Was 11, smaller catches thinner traces
            C=-10  # Was -5, more negative = more sensitive to weak signal
        )

        # Step 3: Morphological closing to fill small gaps
        # Closing = dilation followed by erosion - fills small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_close)

        # Combine adaptive threshold result with original probabilities
        # Use the closed mask to enhance original signal probabilities
        closed_mask = closed.astype(np.float32) / 255.0

        # Apply the mask to boost signal regions while preserving probability gradients
        # Original probabilities where mask is 1, reduced elsewhere
        prob_enhanced = prob_normalized * 0.3 + prob_normalized * closed_mask * 0.7

        # Normalize to [0, 1]
        prob_enhanced = (prob_enhanced - prob_enhanced.min()) / (prob_enhanced.max() - prob_enhanced.min() + 1e-9)

        print(f"  Signal prob post-processing: CLAHE + adaptive threshold (block=11, C=-5) + morphological closing (5x5)")

        # Convert back to tensor with original shape
        result_tensor = torch.from_numpy(prob_enhanced).float()
        if signal_prob.dim() == 4:
            result_tensor = result_tensor.unsqueeze(0).unsqueeze(0)
        elif signal_prob.dim() == 3:
            result_tensor = result_tensor.unsqueeze(0)

        return result_tensor.to(signal_prob.device)

    def _compute_signal_based_boundaries(
        self,
        signal_prob: torch.Tensor,
        grid_prob: torch.Tensor,
        threshold: float = 0.1,
        margin_percent: float = 0.02
    ) -> torch.Tensor:
        """
        Compute bounding box from signal and grid probability maps.
        This is an alternative to Hough-based perspective detection.

        Args:
            signal_prob: Signal probability tensor
            grid_prob: Grid probability tensor
            threshold: Minimum probability to consider as signal/grid
            margin_percent: Extra margin as percentage of dimensions

        Returns:
            Source points tensor [4, 2] in format [TL, TR, BR, BL]
        """
        # Combine signal and grid probabilities
        if signal_prob.dim() == 4:
            sig = signal_prob[0, 0].cpu().numpy()
            grid = grid_prob[0, 0].cpu().numpy()
        elif signal_prob.dim() == 3:
            sig = signal_prob[0].cpu().numpy()
            grid = grid_prob[0].cpu().numpy()
        else:
            sig = signal_prob.cpu().numpy()
            grid = grid_prob.cpu().numpy()

        # Create combined mask (signal OR grid above threshold)
        combined = np.maximum(sig, grid)
        mask = combined > threshold

        # Find rows and columns with content
        rows_with_content = np.any(mask, axis=1)
        cols_with_content = np.any(mask, axis=0)

        # Get bounding indices
        if rows_with_content.any() and cols_with_content.any():
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            top = row_indices[0]
            bottom = row_indices[-1]
            left = col_indices[0]
            right = col_indices[-1]
        else:
            # Fallback to full image
            H, W = sig.shape
            top, bottom, left, right = 0, H - 1, 0, W - 1

        # Add margin
        H, W = sig.shape
        h_margin = int(W * margin_percent)
        v_margin = int(H * margin_percent)

        top = max(0, top - v_margin)
        bottom = min(H - 1, bottom + v_margin)
        left = max(0, left - h_margin)
        right = min(W - 1, right + h_margin)

        # Create corner points [TL, TR, BR, BL]
        points = np.array([
            [left, top],      # Top-left
            [right, top],     # Top-right
            [right, bottom],  # Bottom-right
            [left, bottom],   # Bottom-left
        ], dtype=np.float32)

        print(f"  Signal-based boundaries: L={left}, R={right}, T={top}, B={bottom} "
              f"(W={right-left}, H={bottom-top})")

        return torch.from_numpy(points)

    def _merge_boundaries(
        self,
        hough_points: torch.Tensor,
        signal_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge Hough-based and signal-based boundaries by taking the union.
        This ensures we don't miss any signal content.

        Args:
            hough_points: Hough-detected corners [4, 2] as [TL, TR, BR, BL]
            signal_points: Signal-based corners [4, 2] as [TL, TR, BR, BL]

        Returns:
            Merged corners taking the outermost bounds from both
        """
        hp = hough_points.cpu().numpy()
        sp = signal_points.cpu().numpy()

        # For each boundary, take the more expansive one
        # TL: take minimum x, minimum y
        tl_x = min(hp[0, 0], sp[0, 0])
        tl_y = min(hp[0, 1], sp[0, 1])
        # TR: take maximum x, minimum y
        tr_x = max(hp[1, 0], sp[1, 0])
        tr_y = min(hp[1, 1], sp[1, 1])
        # BR: take maximum x, maximum y
        br_x = max(hp[2, 0], sp[2, 0])
        br_y = max(hp[2, 1], sp[2, 1])
        # BL: take minimum x, maximum y
        bl_x = min(hp[3, 0], sp[3, 0])
        bl_y = max(hp[3, 1], sp[3, 1])

        merged = np.array([
            [tl_x, tl_y],
            [tr_x, tr_y],
            [br_x, br_y],
            [bl_x, bl_y],
        ], dtype=np.float32)

        # Log if we expanded boundaries
        hough_width = (hp[1, 0] - hp[0, 0] + hp[2, 0] - hp[3, 0]) / 2
        merged_width = (merged[1, 0] - merged[0, 0] + merged[2, 0] - merged[3, 0]) / 2
        if merged_width > hough_width * 1.01:  # More than 1% expansion
            print(f"  Boundaries expanded by signal detection: {hough_width:.0f} → {merged_width:.0f}px")

        return torch.from_numpy(merged)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to enhance ECG signal visibility.

        - Increases contrast to make faint traces more visible
        - Applies slight sharpening to improve edge detection
        - Enhances color saturation to help distinguish signal from background

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Preprocessed image as numpy array
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Increase contrast slightly (1.2 = 20% more contrast)
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = contrast_enhancer.enhance(1.2)

        # Increase color saturation to help distinguish colored traces
        color_enhancer = ImageEnhance.Color(pil_image)
        pil_image = color_enhancer.enhance(1.1)

        # Apply slight sharpening to improve edge detection
        sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = sharpness_enhancer.enhance(1.3)

        # Convert back to numpy
        preprocessed = np.array(pil_image)

        print("  Image preprocessing applied: contrast +20%, color +10%, sharpness +30%")
        return preprocessed

    def _validate_and_expand_boundaries(
        self,
        source_points: torch.Tensor,
        mm_per_pixel: float,
        image_shape: tuple
    ) -> tuple[torch.Tensor, bool]:
        """
        Validate and expand boundaries:
        1. Ensure width spans at least MIN_WIDTH_LARGE_SQUARES
        2. Always add VERTICAL_MARGIN_PERCENT to top and bottom

        Args:
            source_points: Detected corner points [4, 2] in format [TL, TR, BR, BL]
            mm_per_pixel: Grid calibration (mm per pixel)
            image_shape: (H, W) of the image

        Returns:
            (adjusted_source_points, was_expanded)
        """
        # Calculate current dimensions in pixels
        # source_points format: [top-left, top-right, bottom-right, bottom-left]
        pts = source_points.cpu().numpy()
        H, W = image_shape

        width_top = pts[1, 0] - pts[0, 0]  # TR.x - TL.x
        width_bottom = pts[2, 0] - pts[3, 0]  # BR.x - BL.x
        # Use MAX width (either top or bottom needs to meet minimum, not both)
        max_width_pixels = max(width_top, width_bottom)
        avg_width_pixels = (width_top + width_bottom) / 2

        height_left = pts[3, 1] - pts[0, 1]  # BL.y - TL.y
        height_right = pts[2, 1] - pts[1, 1]  # BR.y - TR.y
        avg_height_pixels = (height_left + height_right) / 2

        # Convert width to large squares (use max for validation)
        width_top_mm = width_top * mm_per_pixel
        width_bottom_mm = width_bottom * mm_per_pixel
        width_top_squares = width_top_mm / 5.0
        width_bottom_squares = width_bottom_mm / 5.0
        max_width_squares = max(width_top_squares, width_bottom_squares)

        print(f"Boundary validation: top={width_top_squares:.1f}, bottom={width_bottom_squares:.1f} large squares "
              f"(max={max_width_squares:.1f}), height={avg_height_pixels:.0f}px")

        new_pts = pts.copy()
        was_expanded = False

        # Check horizontal width - passes if EITHER top or bottom meets minimum
        if max_width_squares < self.MIN_WIDTH_LARGE_SQUARES:
            target_width_mm = self.MIN_WIDTH_LARGE_SQUARES * 5.0
            target_width_pixels = target_width_mm / mm_per_pixel
            h_expansion = (target_width_pixels - avg_width_pixels) / 2
            print(f"  ⚠ Width too small (max {max_width_squares:.1f} < {self.MIN_WIDTH_LARGE_SQUARES}), "
                  f"expanding by {h_expansion:.0f}px horizontally")

            new_pts[0, 0] = max(0, pts[0, 0] - h_expansion)  # TL
            new_pts[3, 0] = max(0, pts[3, 0] - h_expansion)  # BL
            new_pts[1, 0] = min(W - 1, pts[1, 0] + h_expansion)  # TR
            new_pts[2, 0] = min(W - 1, pts[2, 0] + h_expansion)  # BR
            was_expanded = True
        else:
            print(f"  ✓ Width OK (max {max_width_squares:.1f} >= {self.MIN_WIDTH_LARGE_SQUARES})")

        # Always expand vertically by VERTICAL_MARGIN_PERCENT (5% on each side)
        v_expansion = avg_height_pixels * self.VERTICAL_MARGIN_PERCENT
        print(f"  Adding {self.VERTICAL_MARGIN_PERCENT*100:.0f}% vertical margin ({v_expansion:.0f}px each side)")

        # Expand top points upward
        new_pts[0, 1] = max(0, new_pts[0, 1] - v_expansion)  # TL
        new_pts[1, 1] = max(0, new_pts[1, 1] - v_expansion)  # TR
        # Expand bottom points downward
        new_pts[2, 1] = min(H - 1, new_pts[2, 1] + v_expansion)  # BR
        new_pts[3, 1] = min(H - 1, new_pts[3, 1] + v_expansion)  # BL
        was_expanded = True  # Always expanded due to vertical margin

        return torch.from_numpy(new_pts).to(source_points.device), was_expanded

    @torch.no_grad()
    def _run_alignment_with_expanded_boundaries(
        self,
        image_tensor: torch.Tensor,
        source_points: torch.Tensor,
        layout_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Re-run the pipeline with manually specified source points.
        This bypasses the automatic cropping.
        """
        # Normalize image
        image = self.wrapper.min_max_normalize(image_tensor)
        image = image.to(self.device)

        # Resample
        image = self.wrapper._resample_image(image)

        # Get feature maps
        signal_prob, grid_prob, text_prob = self.wrapper._get_feature_maps(image)

        # Apply alignment with our expanded source_points
        aligned_image = self.wrapper.cropper.apply_perspective(image, source_points, fill_value=0)
        aligned_signal_prob = self.wrapper.cropper.apply_perspective(signal_prob, source_points, fill_value=0)
        aligned_grid_prob = self.wrapper.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
        aligned_text_prob = self.wrapper.cropper.apply_perspective(text_prob, source_points, fill_value=0)

        # Get pixel spacing
        mm_per_pixel_x, mm_per_pixel_y = self.wrapper.pixel_size_finder(aligned_grid_prob)
        avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2

        # Apply dewarping if enabled
        if self.wrapper.apply_dewarping:
            self.wrapper.dewarper.fit(aligned_grid_prob.squeeze(), avg_pixel_per_mm)
            aligned_signal_prob = self.wrapper.dewarper.transform(aligned_signal_prob.squeeze())

        # Apply advanced post-processing to signal probability
        # This includes: CLAHE contrast stretching, adaptive thresholding, morphological closing
        aligned_signal_prob_processed = self._postprocess_signal_prob(aligned_signal_prob)

        # Also apply dilation to further connect broken traces
        aligned_signal_prob_final = self._dilate_signal_prob(aligned_signal_prob_processed, dilation_size=self.DILATION_SIZE)
        print(f"  Applied {self.DILATION_SIZE}x{self.DILATION_SIZE} dilation to signal probability")

        # Extract signals (use processed version for better trace detection)
        signals = self.wrapper.signal_extractor(aligned_signal_prob_final.squeeze())

        # Identify layout
        layout = self.wrapper.identifier(
            signals,
            aligned_text_prob,
            avg_pixel_per_mm,
            layout_should_include_substring=layout_hint,
        )

        try:
            layout_str = layout["layout"]
            layout_is_flipped = str(layout["flip"])
            layout_cost = layout.get("cost", 1.0)
        except KeyError:
            layout_str = "Unknown layout"
            layout_is_flipped = "False"
            layout_cost = 1.0

        return {
            "layout_name": layout_str,
            "input_image": image.cpu(),
            "aligned": {
                "image": aligned_image.cpu(),
                "signal_prob": aligned_signal_prob.cpu(),
                "signal_prob_processed": aligned_signal_prob_processed.cpu(),
                "signal_prob_final": aligned_signal_prob_final.cpu(),
                "grid_prob": aligned_grid_prob.cpu(),
                "text_prob": aligned_text_prob.cpu(),
            },
            "signal": {
                "raw_lines": signals.cpu(),
                "canonical_lines": layout.get("canonical_lines", None),
                "lines": layout.get("lines", None),
                "layout_matching_cost": layout_cost,
                "layout_is_flipped": layout_is_flipped,
            },
            "pixel_spacing_mm": {
                "x": mm_per_pixel_x,
                "y": mm_per_pixel_y,
                "average_pixel_per_mm": avg_pixel_per_mm,
            },
            "source_points": source_points.cpu(),
        }

    def process_from_probmaps(
        self,
        signal_prob: np.ndarray,
        grid_prob: np.ndarray,
        text_prob: np.ndarray,
        layout_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process pre-computed probability maps through the pipeline.

        This enables hybrid processing where segmentation runs on-device (phone ONNX)
        and the rest of the pipeline runs on the server.

        Args:
            signal_prob: Signal probability map (H, W) from ONNX segmentation
            grid_prob: Grid probability map (H, W) from ONNX segmentation
            text_prob: Text probability map (H, W) from ONNX segmentation
            layout_hint: Optional substring to filter layouts (e.g., '6x2', '3x4')

        Returns:
            Dictionary with:
                - success: bool
                - leads: list of lead data with samples in mV
                - grid: grid detection info
                - canonical_lines: raw tensor output (12, 5000) in µV
        """
        try:
            wrapper = self.wrapper

            # Convert to tensors
            signal_tensor = torch.tensor(signal_prob, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            grid_tensor = torch.tensor(grid_prob, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            text_tensor = torch.tensor(text_prob, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Apply post-processing to signal probability (same as full pipeline)
            processed_signal = self._postprocess_signal_prob(signal_tensor.squeeze())
            if isinstance(processed_signal, np.ndarray):
                processed_signal = torch.tensor(processed_signal, dtype=torch.float32)
            processed_signal = processed_signal.unsqueeze(0).unsqueeze(0)

            print(f"  Processed signal: shape={processed_signal.shape}, min={processed_signal.min():.3f}, max={processed_signal.max():.3f}")

            # Step 1: Perspective detection using grid probability
            alignment_params = wrapper.perspective_detector(grid_tensor.squeeze())

            # Step 2: Cropping to find source points
            source_points = wrapper.cropper(processed_signal.squeeze(), alignment_params)

            # Step 3: Align feature maps using perspective transform
            # Create dummy image from signal probability for alignment
            # Need shape (3, H, W) for the alignment function
            h, w = signal_prob.shape
            dummy_image = (1 - processed_signal.squeeze()) * 255  # (H, W)
            dummy_image = dummy_image.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)

            aligned_image, aligned_signal, aligned_grid, aligned_text = wrapper._align_feature_maps(
                dummy_image,  # (3, H, W)
                processed_signal.squeeze(),  # (1, H, W) or (H, W)
                grid_tensor.squeeze(),
                text_tensor.squeeze(),
                source_points
            )

            # Step 4: Find pixel size
            mm_per_pixel_x, mm_per_pixel_y = wrapper.pixel_size_finder(aligned_grid)
            if hasattr(mm_per_pixel_x, 'item'):
                mm_per_pixel_x = mm_per_pixel_x.item()
                mm_per_pixel_y = mm_per_pixel_y.item()
            avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2

            print(f"  Grid: {mm_per_pixel_x:.4f} x {mm_per_pixel_y:.4f} mm/px")

            # Step 5: Dewarping (if enabled)
            if wrapper.apply_dewarping:
                wrapper.dewarper.fit(aligned_grid.squeeze(), avg_pixel_per_mm)
                aligned_signal = wrapper.dewarper.transform(aligned_signal.squeeze())

            # Step 6: Signal extraction
            signals = wrapper.signal_extractor(aligned_signal.squeeze())
            print(f"  Extracted {signals.shape[0]} signal lines")

            # Step 7: Layout identification
            layout = wrapper.identifier(
                signals,
                aligned_text,
                avg_pixel_per_mm,
                layout_should_include_substring=layout_hint,
            )

            layout_str = layout.get("layout", "Unknown")
            canonical_lines = layout.get("canonical_lines")

            if canonical_lines is None:
                print("  Warning: No canonical lines found")
                return {'success': False, 'error': 'Layout identification failed'}

            print(f"  Layout: {layout_str}, canonical shape: {canonical_lines.shape}")

            # Convert to lead dictionary
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            leads_dict = {}

            # canonical_lines is (12, 5000) in µV - convert to mV
            for i, name in enumerate(lead_names):
                if i < canonical_lines.shape[0]:
                    samples = canonical_lines[i].cpu().numpy() / 1000.0  # µV to mV
                    leads_dict[name] = samples.tolist()

            return {
                'success': True,
                'layout': layout_str,
                'leads': leads_dict,
                'grid': {
                    'mm_per_pixel_x': mm_per_pixel_x,
                    'mm_per_pixel_y': mm_per_pixel_y,
                },
                'canonical_lines': canonical_lines.cpu().numpy() if hasattr(canonical_lines, 'cpu') else canonical_lines,
            }

        except Exception as e:
            import traceback
            print(f"  process_from_probmaps error: {e}")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def process_image(
        self,
        image: np.ndarray,
        layout_hint: Optional[str] = None,
        use_signal_based_boundaries: bool = True
    ) -> Dict[str, Any]:
        """
        Process an ECG image through the complete pipeline.

        Args:
            image: RGB image as numpy array, shape (H, W, 3)
            layout_hint: Optional substring to filter layouts (e.g., '6x2', '3x4')
            use_signal_based_boundaries: If True, merge signal-based boundaries with Hough

        Returns:
            Dictionary with:
                - success: bool
                - layout: detected layout name
                - leads: list of lead data with samples in mV
                - grid: grid detection info
                - method: 'open_ecg_digitizer'
                - canonical_lines: raw tensor output (12, 5000) in µV
        """
        try:
            # Apply preprocessing to enhance signal visibility
            if isinstance(image, np.ndarray):
                image = self._preprocess_image(image)
                image_tensor = torch.from_numpy(image).float()
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                image_tensor = image

            # Run the full pipeline
            output = self.wrapper(image_tensor, layout_should_include_substring=layout_hint)

            # Get Hough-based source points from initial run
            source_points = output['source_points']

            # Optionally merge with signal-based boundaries for better coverage
            if use_signal_based_boundaries:
                # Get the aligned feature maps from the output
                aligned = output.get('aligned', {})
                signal_prob = aligned.get('signal_prob')
                grid_prob = aligned.get('grid_prob')

                if signal_prob is not None and grid_prob is not None:
                    # Compute signal-based boundaries from aligned probability maps
                    # Use the input feature maps before alignment for boundary detection
                    input_image = output.get('input_image')
                    if input_image is not None:
                        # Re-get feature maps from input to compute boundaries
                        with torch.no_grad():
                            img_normalized = self.wrapper.min_max_normalize(input_image)
                            img_resampled = self.wrapper._resample_image(img_normalized.to(self.device))
                            sig_prob, grd_prob, _ = self.wrapper._get_feature_maps(img_resampled)

                            signal_bounds = self._compute_signal_based_boundaries(
                                sig_prob, grd_prob, threshold=0.1, margin_percent=0.02
                            )

                            # Merge Hough and signal-based boundaries
                            source_points = self._merge_boundaries(source_points, signal_bounds)
                            print("  Merged Hough and signal-based boundaries")

            # Validate boundary width and expand if needed
            pixel_spacing = output.get('pixel_spacing_mm', {})
            mm_per_pixel = pixel_spacing.get('x', 0.1)
            if hasattr(mm_per_pixel, 'item'):
                mm_per_pixel = mm_per_pixel.item()

            image_shape = (output['input_image'].shape[2], output['input_image'].shape[3])
            expanded_points, was_expanded = self._validate_and_expand_boundaries(
                source_points, mm_per_pixel, image_shape
            )

            # If boundaries were modified, re-run alignment with new boundaries
            if was_expanded or use_signal_based_boundaries:
                print("Re-running alignment with adjusted boundaries...")
                output = self._run_alignment_with_expanded_boundaries(
                    image_tensor, expanded_points.to(self.device), layout_hint
                )

            # Extract canonical lines (already in µV from LeadIdentifier)
            canonical_lines = output['signal']['canonical_lines']

            if canonical_lines is None:
                return {
                    'success': False,
                    'error': 'No canonical lines extracted - layout identification failed',
                    'layout': output.get('layout_name', 'Unknown'),
                }

            # Convert to mV for display
            canonical_mv = canonical_lines.cpu().numpy() / 1000.0

            # Build leads output
            LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"]

            leads = []
            for i, name in enumerate(LEAD_NAMES):
                samples = canonical_mv[i].tolist()
                # Handle NaN values - replace with 0 or interpolate
                samples = [0.0 if np.isnan(s) else s for s in samples]

                # 5000 samples at assumed 500 Hz = 10 seconds
                duration_ms = 10000.0

                leads.append({
                    'name': name,
                    'samples': samples,
                    'duration_ms': duration_ms,
                    'sample_count': len(samples)
                })

            # Extract pixel spacing info
            pixel_spacing = output.get('pixel_spacing_mm', {})
            mm_per_pixel_x = pixel_spacing.get('x', 0.1)
            mm_per_pixel_y = pixel_spacing.get('y', 0.1)

            # Convert tensor values to Python floats
            if hasattr(mm_per_pixel_x, 'item'):
                mm_per_pixel_x = mm_per_pixel_x.item()
            if hasattr(mm_per_pixel_y, 'item'):
                mm_per_pixel_y = mm_per_pixel_y.item()

            # Get layout matching cost (lower = better)
            matching_cost = output['signal'].get('layout_matching_cost', 1.0)
            if hasattr(matching_cost, 'item'):
                matching_cost = matching_cost.item()

            return {
                'success': True,
                'layout': output['layout_name'],
                'detected_layout': output['layout_name'],
                'leads': leads,
                'grid': {
                    'detected': True,
                    'confidence': max(0, 1.0 - matching_cost),
                    'mm_per_pixel_x': float(mm_per_pixel_x),
                    'mm_per_pixel_y': float(mm_per_pixel_y),
                },
                'method': 'open_ecg_digitizer',
                'layout_matching_cost': float(matching_cost),
                'layout_is_flipped': output['signal'].get('layout_is_flipped', 'False'),
            }

        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'method': 'open_ecg_digitizer',
            }


# Global singleton instance
_digitizer_instance = None


def get_digitizer(device: str = 'cpu') -> ECGDigitizerWrapper:
    """Get or create the global digitizer instance."""
    global _digitizer_instance
    if _digitizer_instance is None:
        _digitizer_instance = ECGDigitizerWrapper(device=device)
    return _digitizer_instance


def process_ecg_image(
    image: np.ndarray,
    paper_speed: int = 25,
    voltage_gain: int = 10,
    layout: Optional[str] = None,
    use_signal_based_boundaries: bool = True
) -> Dict[str, Any]:
    """
    Process an ECG image using the Open-ECG-Digitizer pipeline.

    Args:
        image: RGB image as numpy array
        paper_speed: Paper speed in mm/s (25 or 50) - not used by pipeline
        voltage_gain: Voltage gain in mm/mV (5 or 10) - not used by pipeline
        layout: Optional layout hint (e.g., '6x2', '3x4')
        use_signal_based_boundaries: If True, merge signal-based boundaries with Hough

    Returns:
        Processing result dictionary

    Note: paper_speed and voltage_gain are accepted for API compatibility but
    the Open-ECG-Digitizer pipeline determines these from the grid automatically.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    digitizer = get_digitizer(device=device)
    return digitizer.process_image(
        image,
        layout_hint=layout,
        use_signal_based_boundaries=use_signal_based_boundaries
    )


# For backwards compatibility
HAS_OPEN_ECG = HAS_INFERENCE_WRAPPER
