#!/usr/bin/env python3
"""
Open-ECG-Digitizer Integration Module

This module integrates the Open-ECG-Digitizer pipeline for accurate ECG digitization.
Uses their pre-trained UNet models for segmentation and signal extraction.

Based on: https://github.com/Ahus-AIM/Open-ECG-Digitizer
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, Dict, List, Any

# Add Open-ECG-Digitizer to path
OPEN_ECG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Open-ECG-Digitizer')
if os.path.exists(OPEN_ECG_PATH):
    sys.path.insert(0, OPEN_ECG_PATH)

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Using simplified extraction.")

# Try to import Open-ECG-Digitizer components
HAS_OPEN_ECG = False
if HAS_TORCH and os.path.exists(OPEN_ECG_PATH):
    try:
        from src.model.unet import UNet
        from src.model.signal_extractor import SignalExtractor
        from src.model.pixel_size_finder import PixelSizeFinder
        from src.model.perspective_detector import PerspectiveDetector
        from src.model.cropper import Cropper
        HAS_OPEN_ECG = True
        print("Successfully loaded Open-ECG-Digitizer components")
    except ImportError as e:
        print(f"Warning: Could not load Open-ECG-Digitizer: {e}")


# Lead names in standard order
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class OpenECGDigitizer:
    """
    ECG digitization using Open-ECG-Digitizer models.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.unet = None
        self.signal_extractor = None
        self.pixel_size_finder = None
        self.perspective_detector = None
        self.cropper = None

        if HAS_OPEN_ECG:
            self._load_models()

    def _load_models(self):
        """Load pre-trained models."""
        weights_dir = os.path.join(OPEN_ECG_PATH, 'weights')

        # Load segmentation UNet
        unet_weights = os.path.join(weights_dir, 'unet_weights_07072025.pt')
        if os.path.exists(unet_weights):
            print(f"Loading UNet from {unet_weights}")
            self.unet = UNet(
                num_in_channels=3,
                num_out_channels=4,  # grid, text_background, signal, background
                dims=[32, 64, 128, 256, 320, 320, 320, 320],
                depth=2
            )
            checkpoint = torch.load(unet_weights, map_location=self.device, weights_only=True)
            checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
            self.unet.load_state_dict(checkpoint)
            self.unet.to(self.device)
            self.unet.eval()
            print("UNet loaded successfully")

        # Initialize signal extractor
        self.signal_extractor = SignalExtractor(
            threshold_sum=10.0,
            threshold_line_in_mask=0.95,
            label_thresh=0.1,
            max_iterations=4,
            split_num_stripes=4
        )

        # Initialize pixel size finder
        self.pixel_size_finder = PixelSizeFinder(
            min_number_of_grid_lines=30,
            max_number_of_grid_lines=70,
            lower_grid_line_factor=0.3
        )

        # Initialize perspective detector
        self.perspective_detector = PerspectiveDetector(num_thetas=250)

        # Initialize cropper
        self.cropper = Cropper(
            granularity=80,
            percentiles=[0.02, 0.98],
            alpha=0.85
        )

    def preprocess_image(self, img_array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Normalize to 0-1
        img = img_array.astype(np.float32) / 255.0

        # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        # Min-max normalize
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)

        return img_tensor.to(self.device)

    def segment_image(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run UNet segmentation to get signal, grid, and text probability maps.

        Returns:
            signal_prob: Probability map for ECG signal
            grid_prob: Probability map for grid
            text_prob: Probability map for text/labels
        """
        if self.unet is None:
            raise RuntimeError("UNet model not loaded")

        with torch.no_grad():
            # Get logits from UNet
            logits = self.unet(img_tensor)

            # Convert to probabilities
            prob = torch.softmax(logits, dim=1)

            # Extract channels (based on Open-ECG-Digitizer config)
            # 0: grid, 1: text_background, 2: signal, 3: background
            grid_prob = prob[:, [0], :, :]
            text_prob = prob[:, [1], :, :]
            signal_prob = prob[:, [2], :, :]

            # Post-process to enhance sparse features
            signal_prob = self._process_sparse_prob(signal_prob)
            grid_prob = self._process_sparse_prob(grid_prob)
            text_prob = self._process_sparse_prob(text_prob)

        return signal_prob, grid_prob, text_prob

    def _process_sparse_prob(self, prob: torch.Tensor) -> torch.Tensor:
        """Post-process probability map for sparse features."""
        prob = prob - prob.mean()
        prob = torch.clamp(prob, min=0)
        prob = prob / (prob.max() + 1e-9)
        return prob

    def extract_signals(self, signal_prob: torch.Tensor) -> torch.Tensor:
        """
        Extract signal lines from probability map.

        Returns:
            lines: Tensor of shape (num_lines, width) with y-positions
        """
        if self.signal_extractor is None:
            raise RuntimeError("Signal extractor not initialized")

        # Signal extractor expects 2D tensor (H, W)
        signal_2d = signal_prob.squeeze()

        # Extract lines
        lines = self.signal_extractor(signal_2d)

        return lines

    def find_pixel_spacing(self, grid_prob: torch.Tensor) -> Tuple[float, float]:
        """
        Find pixel spacing (mm/pixel) from grid probability map.

        Returns:
            mm_per_pixel_x, mm_per_pixel_y
        """
        if self.pixel_size_finder is None:
            raise RuntimeError("Pixel size finder not initialized")

        mm_x, mm_y = self.pixel_size_finder(grid_prob)

        # Convert to float if torch tensor
        if hasattr(mm_x, 'item'):
            mm_x = mm_x.item()
        if hasattr(mm_y, 'item'):
            mm_y = mm_y.item()

        return float(mm_x), float(mm_y)

    def lines_to_voltage(
        self,
        lines: torch.Tensor,
        mm_per_pixel_y: float,
        voltage_gain_mm_per_mv: float = 10.0
    ) -> np.ndarray:
        """
        Convert y-position lines to voltage values.

        Args:
            lines: Tensor of shape (num_lines, width) with y-positions
            mm_per_pixel_y: Millimeters per pixel in y direction
            voltage_gain_mm_per_mv: Standard ECG calibration (default 10mm/mV)

        Returns:
            voltages: Array of shape (num_lines, width) with voltages in mV
        """
        lines_np = lines.cpu().numpy()

        # Remove mean (baseline) from each line
        baseline = np.nanmean(lines_np, axis=1, keepdims=True)
        offset_pixels = lines_np - baseline

        # Convert pixels to mm, then to mV
        # Note: y increases downward, so negative offset = positive voltage
        offset_mm = -offset_pixels * mm_per_pixel_y
        voltage_mv = offset_mm / voltage_gain_mm_per_mv

        # Scale to microvolts then back to mV for standard ECG display
        voltage_mv = voltage_mv * 1000  # to uV
        voltage_mv = voltage_mv / 1000  # back to mV

        return voltage_mv

    def process_image(
        self,
        img_array: np.ndarray,
        paper_speed: float = 25.0,
        voltage_gain: float = 10.0,
        layout: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full digitization pipeline.

        Args:
            img_array: Input image as numpy array (H, W, C)
            paper_speed: Paper speed in mm/s
            voltage_gain: Voltage gain in mm/mV
            layout: Optional layout override

        Returns:
            Dictionary with extracted leads and metadata
        """
        result = {
            'success': True,
            'image_size': {'width': img_array.shape[1], 'height': img_array.shape[0]},
            'method': 'open_ecg_digitizer' if HAS_OPEN_ECG else 'fallback'
        }

        if not HAS_OPEN_ECG or self.unet is None:
            # Fallback to basic extraction
            return self._fallback_extraction(img_array, paper_speed, voltage_gain, layout)

        try:
            # Preprocess
            img_tensor = self.preprocess_image(img_array)

            # Resample if too large
            max_size = 3000
            _, _, H, W = img_tensor.shape
            if max(H, W) > max_size:
                scale = max_size / max(H, W)
                new_size = (int(H * scale), int(W * scale))
                img_tensor = F.interpolate(img_tensor, size=new_size, mode='bilinear', align_corners=False)

            # Segment
            signal_prob, grid_prob, text_prob = self.segment_image(img_tensor)

            # Find pixel spacing
            try:
                mm_per_pixel_x, mm_per_pixel_y = self.find_pixel_spacing(grid_prob)
            except Exception as e:
                print(f"Pixel spacing detection failed: {e}, using defaults")
                mm_per_pixel_x = 0.1
                mm_per_pixel_y = 0.1

            result['grid'] = {
                'h_spacing': 1.0 / mm_per_pixel_x if mm_per_pixel_x > 0 else 10,
                'v_spacing': 1.0 / mm_per_pixel_y if mm_per_pixel_y > 0 else 10,
                'mm_per_pixel_x': mm_per_pixel_x,
                'mm_per_pixel_y': mm_per_pixel_y,
                'confidence': 0.9,
                'detected': True
            }

            # Extract signal lines
            lines = self.extract_signals(signal_prob)

            if lines.shape[0] == 0:
                result['success'] = False
                result['error'] = 'No signal lines detected'
                return result

            # Convert to voltage
            voltages = self.lines_to_voltage(lines, mm_per_pixel_y, voltage_gain)

            # Determine layout
            num_lines = voltages.shape[0]
            if layout:
                detected_layout = layout
            else:
                # Auto-detect based on number of lines
                if num_lines >= 12:
                    detected_layout = '3x4_r1'
                elif num_lines >= 6:
                    detected_layout = '6x2_r0'
                else:
                    detected_layout = '3x4_r0'

            result['layout'] = detected_layout
            result['detected_layout'] = detected_layout
            result['num_lines_detected'] = num_lines

            # Assign leads based on layout
            leads = self._assign_leads(voltages, detected_layout, mm_per_pixel_x, paper_speed)
            result['leads'] = leads

        except Exception as e:
            import traceback
            result['success'] = False
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()

        return result

    def _assign_leads(
        self,
        voltages: np.ndarray,
        layout: str,
        mm_per_pixel_x: float,
        paper_speed: float
    ) -> List[Dict[str, Any]]:
        """
        Assign extracted voltage lines to standard ECG leads based on layout.
        """
        leads = []
        num_lines = voltages.shape[0]
        width = voltages.shape[1]

        # Calculate duration
        total_mm = width * mm_per_pixel_x
        duration_ms = (total_mm / paper_speed) * 1000

        # Parse layout
        parts = layout.split('_')
        grid_parts = parts[0].split('x')

        if layout.startswith('3x4'):
            # 3x4 layout: 3 rows, 4 columns
            # Row 0: I, aVR, V1, V4
            # Row 1: II, aVL, V2, V5
            # Row 2: III, aVF, V3, V6
            lead_map = [
                ["I", "aVR", "V1", "V4"],
                ["II", "aVL", "V2", "V5"],
                ["III", "aVF", "V3", "V6"]
            ]
            cols = 4
            rows = 3

            for row_idx in range(min(rows, num_lines)):
                samples = voltages[row_idx]
                samples = np.nan_to_num(samples, nan=0.0)

                # Split into columns
                col_width = len(samples) // cols
                for col_idx in range(cols):
                    start = col_idx * col_width
                    end = (col_idx + 1) * col_width if col_idx < cols - 1 else len(samples)
                    col_samples = samples[start:end]

                    if row_idx < len(lead_map) and col_idx < len(lead_map[row_idx]):
                        lead_name = lead_map[row_idx][col_idx]

                        leads.append({
                            'name': lead_name,
                            'samples': col_samples.tolist(),
                            'duration_ms': duration_ms / cols,
                            'sample_count': len(col_samples)
                        })

            # Handle rhythm strips
            rhythm_strips = int(parts[1][1]) if len(parts) > 1 and parts[1].startswith('r') else 0
            rhythm_names = ["II", "V1", "V5"]

            for r in range(rhythm_strips):
                line_idx = rows + r
                if line_idx < num_lines:
                    samples = voltages[line_idx]
                    samples = np.nan_to_num(samples, nan=0.0)

                    lead_name = rhythm_names[r] if r < len(rhythm_names) else f"Rhythm{r+1}"
                    leads.append({
                        'name': f"{lead_name} (rhythm)",
                        'samples': samples.tolist(),
                        'duration_ms': duration_ms,
                        'sample_count': len(samples),
                        'is_rhythm': True
                    })

        elif layout.startswith('6x2'):
            # 6x2 layout: 6 rows, 2 columns
            lead_map = [
                ["I", "V1"],
                ["II", "V2"],
                ["III", "V3"],
                ["aVR", "V4"],
                ["aVL", "V5"],
                ["aVF", "V6"]
            ]
            cols = 2
            rows = 6

            for row_idx in range(min(rows, num_lines)):
                samples = voltages[row_idx]
                samples = np.nan_to_num(samples, nan=0.0)

                col_width = len(samples) // cols
                for col_idx in range(cols):
                    start = col_idx * col_width
                    end = (col_idx + 1) * col_width if col_idx < cols - 1 else len(samples)
                    col_samples = samples[start:end]

                    if row_idx < len(lead_map) and col_idx < len(lead_map[row_idx]):
                        lead_name = lead_map[row_idx][col_idx]

                        leads.append({
                            'name': lead_name,
                            'samples': col_samples.tolist(),
                            'duration_ms': duration_ms / cols,
                            'sample_count': len(col_samples)
                        })

        elif layout.startswith('12x1'):
            # 12x1 layout: 12 rows, 1 column each
            lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

            for row_idx in range(min(12, num_lines)):
                samples = voltages[row_idx]
                samples = np.nan_to_num(samples, nan=0.0)

                leads.append({
                    'name': lead_names[row_idx] if row_idx < len(lead_names) else f"Lead{row_idx+1}",
                    'samples': samples.tolist(),
                    'duration_ms': duration_ms,
                    'sample_count': len(samples)
                })

        else:
            # Default: just use the lines as-is
            for i in range(num_lines):
                samples = voltages[i]
                samples = np.nan_to_num(samples, nan=0.0)

                lead_name = LEAD_NAMES[i] if i < len(LEAD_NAMES) else f"Lead{i+1}"
                leads.append({
                    'name': lead_name,
                    'samples': samples.tolist(),
                    'duration_ms': duration_ms,
                    'sample_count': len(samples)
                })

        return leads

    def _fallback_extraction(
        self,
        img_array: np.ndarray,
        paper_speed: float,
        voltage_gain: float,
        layout: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback extraction when Open-ECG-Digitizer is not available."""
        # This is a simplified version - the main app.py has the full fallback
        return {
            'success': False,
            'error': 'Open-ECG-Digitizer not available, please use main app.py',
            'method': 'fallback'
        }


# Global instance
_digitizer = None


def get_digitizer(device: str = 'cpu') -> OpenECGDigitizer:
    """Get or create the global digitizer instance."""
    global _digitizer
    if _digitizer is None:
        _digitizer = OpenECGDigitizer(device=device)
    return _digitizer


def process_with_open_ecg(
    img_array: np.ndarray,
    paper_speed: float = 25.0,
    voltage_gain: float = 10.0,
    layout: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process an ECG image using Open-ECG-Digitizer.

    This is the main entry point for integration with the web app.
    """
    digitizer = get_digitizer()
    return digitizer.process_image(img_array, paper_speed, voltage_gain, layout)


if __name__ == '__main__':
    # Test
    print(f"HAS_TORCH: {HAS_TORCH}")
    print(f"HAS_OPEN_ECG: {HAS_OPEN_ECG}")
    print(f"OPEN_ECG_PATH exists: {os.path.exists(OPEN_ECG_PATH)}")

    if HAS_OPEN_ECG:
        digitizer = OpenECGDigitizer()
        print(f"UNet loaded: {digitizer.unet is not None}")
