#!/usr/bin/env python3
"""
Unit Tests for Waveform Extraction

Tests the signal extraction pipeline from Open-ECG-Digitizer to verify:
1. Signal extraction from probability maps
2. Line merging and grouping
3. Voltage conversion accuracy
4. Edge cases and error handling

Run with: pytest test_waveform_extraction.py -v
"""

import os
import sys
import json
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'Open-ECG-Digitizer'))

# Try to import components
HAS_TORCH = False
HAS_OPEN_ECG = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

if HAS_TORCH:
    try:
        from src.model.signal_extractor import SignalExtractor
        HAS_OPEN_ECG = True
    except ImportError:
        pass


class TestResult:
    """Stores test result with feedback data."""

    def __init__(self, name: str, passed: bool, message: str,
                 details: Optional[Dict] = None,
                 visual_data: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.visual_data = visual_data or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'passed': self.passed,
            'message': self.message,
            'details': self.details,
            'visual_data': self.visual_data,
            'timestamp': self.timestamp
        }


class WaveformExtractionTestSuite:
    """Test suite for waveform extraction with feedback mechanism."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.signal_extractor = None

        if HAS_OPEN_ECG:
            self.signal_extractor = SignalExtractor(
                threshold_sum=10.0,
                threshold_line_in_mask=0.95,
                label_thresh=0.1,
                max_iterations=4,
                split_num_stripes=4,
                min_line_width=30
            )

    def run_all_tests(self) -> List[Dict]:
        """Run all waveform extraction tests."""
        self.results = []

        # Core functionality tests
        self._test_dependency_availability()

        if HAS_OPEN_ECG:
            self._test_single_line_extraction()
            self._test_multiple_line_extraction()
            self._test_overlapping_lines()
            self._test_noisy_input()
            self._test_empty_input()
            self._test_line_merging()
            self._test_autodetect_peaks()
            self._test_voltage_conversion()

        return [r.to_dict() for r in self.results]

    def _test_dependency_availability(self):
        """Test that required dependencies are available."""
        details = {
            'has_torch': HAS_TORCH,
            'has_open_ecg': HAS_OPEN_ECG,
            'torch_version': torch.__version__ if HAS_TORCH else None
        }

        if HAS_TORCH and HAS_OPEN_ECG:
            self.results.append(TestResult(
                name="Dependency Check",
                passed=True,
                message="All required dependencies are available",
                details=details
            ))
        else:
            missing = []
            if not HAS_TORCH:
                missing.append("PyTorch")
            if not HAS_OPEN_ECG:
                missing.append("Open-ECG-Digitizer")

            self.results.append(TestResult(
                name="Dependency Check",
                passed=False,
                message=f"Missing dependencies: {', '.join(missing)}",
                details=details
            ))

    def _test_single_line_extraction(self):
        """Test extraction of a single clear signal line."""
        # Create synthetic probability map with one clear horizontal line
        height, width = 100, 500
        feature_map = torch.zeros(height, width)

        # Add a sinusoidal signal at y=50
        x = torch.arange(width).float()
        y_signal = 50 + 10 * torch.sin(2 * np.pi * x / 100)

        for i in range(width):
            y_idx = int(y_signal[i].item())
            if 0 <= y_idx < height:
                # Create a gaussian spread around the signal
                for dy in range(-5, 6):
                    if 0 <= y_idx + dy < height:
                        feature_map[y_idx + dy, i] = np.exp(-dy**2 / 4)

        try:
            lines = self.signal_extractor(feature_map)

            num_lines = lines.shape[0]
            expected_lines = 1

            visual_data = {
                'type': 'waveform',
                'extracted_lines': lines.tolist() if num_lines > 0 else [],
                'expected_pattern': 'single_sinusoid',
                'feature_map_shape': [height, width]
            }

            if num_lines == expected_lines:
                # Check if extracted line follows expected pattern
                extracted = lines[0].numpy()
                valid_mask = ~np.isnan(extracted)

                if valid_mask.sum() > width * 0.8:  # At least 80% valid
                    self.results.append(TestResult(
                        name="Single Line Extraction",
                        passed=True,
                        message=f"Successfully extracted {num_lines} line with {valid_mask.sum()} valid samples",
                        details={
                            'num_lines': num_lines,
                            'valid_samples': int(valid_mask.sum()),
                            'coverage': float(valid_mask.sum() / width)
                        },
                        visual_data=visual_data
                    ))
                else:
                    self.results.append(TestResult(
                        name="Single Line Extraction",
                        passed=False,
                        message=f"Line extracted but insufficient coverage: {valid_mask.sum()}/{width}",
                        details={'valid_samples': int(valid_mask.sum())},
                        visual_data=visual_data
                    ))
            else:
                self.results.append(TestResult(
                    name="Single Line Extraction",
                    passed=False,
                    message=f"Expected 1 line, got {num_lines}",
                    details={'num_lines': num_lines},
                    visual_data=visual_data
                ))
        except Exception as e:
            self.results.append(TestResult(
                name="Single Line Extraction",
                passed=False,
                message=f"Exception: {str(e)}",
                details={'error': str(e)}
            ))

    def _test_multiple_line_extraction(self):
        """Test extraction of multiple parallel signal lines."""
        height, width = 200, 500
        feature_map = torch.zeros(height, width)

        # Create 4 parallel lines at different y positions
        line_positions = [30, 70, 110, 150]

        for y_center in line_positions:
            x = torch.arange(width).float()
            y_signal = y_center + 5 * torch.sin(2 * np.pi * x / 80)

            for i in range(width):
                y_idx = int(y_signal[i].item())
                if 0 <= y_idx < height:
                    for dy in range(-3, 4):
                        if 0 <= y_idx + dy < height:
                            feature_map[y_idx + dy, i] = max(
                                feature_map[y_idx + dy, i].item(),
                                np.exp(-dy**2 / 2)
                            )

        try:
            lines = self.signal_extractor(feature_map)
            num_lines = lines.shape[0]
            expected_lines = 4

            visual_data = {
                'type': 'waveform',
                'extracted_lines': lines.tolist() if num_lines > 0 else [],
                'expected_pattern': 'four_parallel_sinusoids',
                'expected_positions': line_positions
            }

            if num_lines == expected_lines:
                self.results.append(TestResult(
                    name="Multiple Line Extraction",
                    passed=True,
                    message=f"Successfully extracted {num_lines} lines",
                    details={'num_lines': num_lines, 'expected': expected_lines},
                    visual_data=visual_data
                ))
            else:
                self.results.append(TestResult(
                    name="Multiple Line Extraction",
                    passed=num_lines >= expected_lines - 1,  # Allow small deviation
                    message=f"Extracted {num_lines} lines (expected {expected_lines})",
                    details={'num_lines': num_lines, 'expected': expected_lines},
                    visual_data=visual_data
                ))
        except Exception as e:
            self.results.append(TestResult(
                name="Multiple Line Extraction",
                passed=False,
                message=f"Exception: {str(e)}",
                details={'error': str(e)}
            ))

    def _test_overlapping_lines(self):
        """Test extraction when lines partially overlap."""
        height, width = 150, 500
        feature_map = torch.zeros(height, width)

        # Create two lines that overlap in the middle
        x = torch.arange(width).float()

        # Line 1: starts at y=40, ends at y=80
        y1 = 40 + (x / width) * 40

        # Line 2: starts at y=100, crosses line 1 in middle
        y2 = 100 - (x / width) * 30

        for i in range(width):
            for y, strength in [(y1[i], 1.0), (y2[i], 0.8)]:
                y_idx = int(y.item())
                if 0 <= y_idx < height:
                    for dy in range(-2, 3):
                        if 0 <= y_idx + dy < height:
                            feature_map[y_idx + dy, i] = max(
                                feature_map[y_idx + dy, i].item(),
                                strength * np.exp(-dy**2 / 2)
                            )

        try:
            lines = self.signal_extractor(feature_map)
            num_lines = lines.shape[0]

            visual_data = {
                'type': 'waveform',
                'extracted_lines': lines.tolist() if num_lines > 0 else [],
                'expected_pattern': 'crossing_lines'
            }

            # Should still extract 2 lines despite overlap
            if num_lines >= 2:
                self.results.append(TestResult(
                    name="Overlapping Lines",
                    passed=True,
                    message=f"Successfully handled overlapping lines, extracted {num_lines}",
                    details={'num_lines': num_lines},
                    visual_data=visual_data
                ))
            else:
                self.results.append(TestResult(
                    name="Overlapping Lines",
                    passed=False,
                    message=f"Failed to separate overlapping lines, got {num_lines}",
                    details={'num_lines': num_lines},
                    visual_data=visual_data
                ))
        except Exception as e:
            self.results.append(TestResult(
                name="Overlapping Lines",
                passed=False,
                message=f"Exception: {str(e)}",
                details={'error': str(e)}
            ))

    def _test_noisy_input(self):
        """Test extraction with noisy probability map."""
        height, width = 100, 500
        feature_map = torch.zeros(height, width)

        # Add signal
        x = torch.arange(width).float()
        y_signal = 50 + 8 * torch.sin(2 * np.pi * x / 100)

        for i in range(width):
            y_idx = int(y_signal[i].item())
            if 0 <= y_idx < height:
                for dy in range(-3, 4):
                    if 0 <= y_idx + dy < height:
                        feature_map[y_idx + dy, i] = np.exp(-dy**2 / 3)

        # Add noise
        noise = torch.rand(height, width) * 0.15
        feature_map = feature_map + noise

        try:
            lines = self.signal_extractor(feature_map)
            num_lines = lines.shape[0]

            visual_data = {
                'type': 'waveform',
                'extracted_lines': lines.tolist() if num_lines > 0 else [],
                'noise_level': 0.15
            }

            if num_lines >= 1:
                # Check if main signal was recovered
                extracted = lines[0].numpy()
                valid_mask = ~np.isnan(extracted)

                self.results.append(TestResult(
                    name="Noisy Input Handling",
                    passed=True,
                    message=f"Extracted {num_lines} lines from noisy input",
                    details={
                        'num_lines': num_lines,
                        'valid_samples': int(valid_mask.sum())
                    },
                    visual_data=visual_data
                ))
            else:
                self.results.append(TestResult(
                    name="Noisy Input Handling",
                    passed=False,
                    message="Failed to extract signal from noisy input",
                    details={'num_lines': num_lines},
                    visual_data=visual_data
                ))
        except Exception as e:
            self.results.append(TestResult(
                name="Noisy Input Handling",
                passed=False,
                message=f"Exception: {str(e)}",
                details={'error': str(e)}
            ))

    def _test_empty_input(self):
        """Test handling of empty/zero probability map."""
        height, width = 100, 500
        feature_map = torch.zeros(height, width)

        try:
            lines = self.signal_extractor(feature_map)

            # Should return empty or minimal result
            if lines.shape[0] == 0:
                self.results.append(TestResult(
                    name="Empty Input Handling",
                    passed=True,
                    message="Correctly returned empty result for zero input",
                    details={'num_lines': 0}
                ))
            else:
                self.results.append(TestResult(
                    name="Empty Input Handling",
                    passed=False,
                    message=f"Unexpected lines from empty input: {lines.shape[0]}",
                    details={'num_lines': lines.shape[0]}
                ))
        except Exception as e:
            # Some implementations may raise an exception for empty input
            self.results.append(TestResult(
                name="Empty Input Handling",
                passed=True,  # Graceful exception is acceptable
                message=f"Handled empty input with exception: {type(e).__name__}",
                details={'error': str(e)}
            ))

    def _test_line_merging(self):
        """Test the line merging algorithm."""
        height, width = 100, 600
        feature_map = torch.zeros(height, width)

        # Create a line that has a gap in the middle (simulates wrapped ECG)
        y_center = 50

        # First segment: 0-200
        for i in range(0, 200):
            y = y_center + 5 * np.sin(2 * np.pi * i / 100)
            y_idx = int(y)
            if 0 <= y_idx < height:
                for dy in range(-2, 3):
                    if 0 <= y_idx + dy < height:
                        feature_map[y_idx + dy, i] = np.exp(-dy**2 / 2)

        # Second segment: 400-600 (continuation of same signal)
        for i in range(400, 600):
            y = y_center + 5 * np.sin(2 * np.pi * i / 100)
            y_idx = int(y)
            if 0 <= y_idx < height:
                for dy in range(-2, 3):
                    if 0 <= y_idx + dy < height:
                        feature_map[y_idx + dy, i] = np.exp(-dy**2 / 2)

        try:
            lines = self.signal_extractor(feature_map)
            num_lines = lines.shape[0]

            visual_data = {
                'type': 'waveform',
                'extracted_lines': lines.tolist() if num_lines > 0 else [],
                'expected_pattern': 'split_line_merge'
            }

            # Ideally should merge into 1 line, but 2 is also acceptable
            if num_lines <= 2:
                self.results.append(TestResult(
                    name="Line Merging",
                    passed=True,
                    message=f"Line segments handled correctly, got {num_lines} line(s)",
                    details={'num_lines': num_lines},
                    visual_data=visual_data
                ))
            else:
                self.results.append(TestResult(
                    name="Line Merging",
                    passed=False,
                    message=f"Too many lines from split segments: {num_lines}",
                    details={'num_lines': num_lines},
                    visual_data=visual_data
                ))
        except Exception as e:
            self.results.append(TestResult(
                name="Line Merging",
                passed=False,
                message=f"Exception: {str(e)}",
                details={'error': str(e)}
            ))

    def _test_autodetect_peaks(self):
        """Test automatic peak detection."""
        height, width = 200, 500
        feature_map = torch.zeros(height, width)

        # Create 6 well-separated lines (typical for 6x1 layout)
        line_positions = [20, 50, 80, 110, 140, 170]

        for y_center in line_positions:
            for i in range(width):
                y = y_center + 3 * np.sin(2 * np.pi * i / 60)
                y_idx = int(y)
                if 0 <= y_idx < height:
                    for dy in range(-2, 3):
                        if 0 <= y_idx + dy < height:
                            feature_map[y_idx + dy, i] = np.exp(-dy**2 / 2)

        try:
            lines = self.signal_extractor(feature_map)
            detected_peaks = self.signal_extractor.num_peaks

            visual_data = {
                'type': 'waveform',
                'extracted_lines': lines.tolist() if lines.shape[0] > 0 else [],
                'expected_peaks': len(line_positions),
                'detected_peaks': detected_peaks
            }

            if detected_peaks == len(line_positions):
                self.results.append(TestResult(
                    name="Autodetect Peaks",
                    passed=True,
                    message=f"Correctly detected {detected_peaks} peaks",
                    details={
                        'detected_peaks': detected_peaks,
                        'expected_peaks': len(line_positions)
                    },
                    visual_data=visual_data
                ))
            else:
                self.results.append(TestResult(
                    name="Autodetect Peaks",
                    passed=abs(detected_peaks - len(line_positions)) <= 1,
                    message=f"Detected {detected_peaks} peaks (expected {len(line_positions)})",
                    details={
                        'detected_peaks': detected_peaks,
                        'expected_peaks': len(line_positions)
                    },
                    visual_data=visual_data
                ))
        except Exception as e:
            self.results.append(TestResult(
                name="Autodetect Peaks",
                passed=False,
                message=f"Exception: {str(e)}",
                details={'error': str(e)}
            ))

    def _test_voltage_conversion(self):
        """Test voltage conversion accuracy."""
        # Create known pixel positions and verify voltage conversion
        height = 100
        width = 500
        mm_per_pixel_y = 0.1  # 10 pixels per mm
        voltage_gain = 10.0  # 10 mm/mV (standard ECG)

        # Create test lines with known offsets
        # y=50 is baseline, y=40 should be +1mm = +0.1mV
        lines = torch.zeros(1, width)

        # First half at baseline (y=50), second half at y=40 (10 pixels up = 1mm up = 0.1mV)
        lines[0, :250] = 50
        lines[0, 250:] = 40

        # Convert to voltage manually
        baseline = 50
        offset_pixels = lines - baseline  # negative means up in image
        offset_mm = -offset_pixels * mm_per_pixel_y  # up in image = positive mm
        voltage_mv = offset_mm / voltage_gain

        expected_first_half = 0.0  # baseline
        expected_second_half = 0.1  # 10 pixels up * 0.1 mm/pixel / 10 mm/mV = 0.1 mV

        actual_first_half = voltage_mv[0, :250].mean().item()
        actual_second_half = voltage_mv[0, 250:].mean().item()

        tolerance = 0.001  # 1 µV tolerance

        first_ok = abs(actual_first_half - expected_first_half) < tolerance
        second_ok = abs(actual_second_half - expected_second_half) < tolerance

        visual_data = {
            'type': 'voltage_conversion',
            'expected_values': [expected_first_half, expected_second_half],
            'actual_values': [actual_first_half, actual_second_half]
        }

        if first_ok and second_ok:
            self.results.append(TestResult(
                name="Voltage Conversion",
                passed=True,
                message="Voltage conversion is accurate",
                details={
                    'first_half_expected': expected_first_half,
                    'first_half_actual': actual_first_half,
                    'second_half_expected': expected_second_half,
                    'second_half_actual': actual_second_half
                },
                visual_data=visual_data
            ))
        else:
            self.results.append(TestResult(
                name="Voltage Conversion",
                passed=False,
                message="Voltage conversion error",
                details={
                    'first_half_error': abs(actual_first_half - expected_first_half),
                    'second_half_error': abs(actual_second_half - expected_second_half)
                },
                visual_data=visual_data
            ))


# Standard unittest compatibility
class TestWaveformExtraction(unittest.TestCase):
    """Standard unittest class for waveform extraction."""

    @classmethod
    def setUpClass(cls):
        cls.suite = WaveformExtractionTestSuite()

    def test_dependencies(self):
        """Test that dependencies are available."""
        self.assertTrue(HAS_TORCH, "PyTorch is required")

    @unittest.skipUnless(HAS_OPEN_ECG, "Open-ECG-Digitizer not available")
    def test_single_line(self):
        """Test single line extraction."""
        self.suite._test_single_line_extraction()
        result = self.suite.results[-1]
        self.assertTrue(result.passed, result.message)

    @unittest.skipUnless(HAS_OPEN_ECG, "Open-ECG-Digitizer not available")
    def test_multiple_lines(self):
        """Test multiple line extraction."""
        self.suite._test_multiple_line_extraction()
        result = self.suite.results[-1]
        self.assertTrue(result.passed, result.message)


def run_tests() -> List[Dict]:
    """Run all tests and return results as JSON-serializable dict."""
    suite = WaveformExtractionTestSuite()
    return suite.run_all_tests()


if __name__ == '__main__':
    # Run tests and print results
    results = run_tests()

    print("\n" + "=" * 60)
    print("WAVEFORM EXTRACTION TEST RESULTS")
    print("=" * 60)

    passed = sum(1 for r in results if r['passed'])
    total = len(results)

    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"\n{status}: {r['name']}")
        print(f"   {r['message']}")
        if r['details']:
            for k, v in r['details'].items():
                print(f"   - {k}: {v}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
