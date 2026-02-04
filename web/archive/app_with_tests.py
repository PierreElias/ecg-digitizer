#!/usr/bin/env python3
"""
ECG Digitizer Web Application with Interactive Testing

A comprehensive web interface for:
1. ECG image digitization using Open-ECG-Digitizer
2. Interactive unit testing with visual feedback
3. Waveform extraction validation
4. Layout identification testing

Run with: python app_with_tests.py
Then open: http://localhost:8080

Requirements:
    pip install flask pillow numpy scipy scikit-image torch torchvision networkx
"""

import os
import sys
import json
import base64
import tempfile
from pathlib import Path
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List, Optional

from flask import Flask, render_template_string, request, jsonify, send_file
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Open-ECG-Digitizer'))

# Import test modules
try:
    from tests.test_waveform_extraction import WaveformExtractionTestSuite, run_tests as run_waveform_tests
    from tests.test_layout_identification import LayoutIdentificationTestSuite, run_tests as run_layout_tests
    HAS_TESTS = True
except ImportError as e:
    print(f"Warning: Could not import test modules: {e}")
    HAS_TESTS = False

# Try to import Open-ECG-Digitizer
HAS_TORCH = False
HAS_OPEN_ECG = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

if HAS_TORCH:
    try:
        from open_ecg_digitizer import process_with_open_ecg, HAS_OPEN_ECG as _HAS_OE
        HAS_OPEN_ECG = _HAS_OE
    except ImportError:
        pass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


# ============================================================
# Simplified ECG Processing (Fallback)
# ============================================================

def detect_grid(img_array):
    """Detect ECG grid pattern."""
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    height, width = gray.shape[:2]
    return {
        'h_spacing': width / 50,
        'v_spacing': height / 40,
        'confidence': 0.3,
        'detected': False
    }


def classify_layout(img_array):
    """Classify ECG layout type."""
    height, width = img_array.shape[:2]
    aspect = width / height
    if aspect > 2.5:
        return "12x1_r1"
    elif aspect > 1.8:
        return "6x2_r1"
    else:
        return "3x4_r1"


def process_image(image_data, paper_speed=25, voltage_gain=10, layout=None):
    """Main processing pipeline."""
    try:
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)

        img = Image.open(BytesIO(image_data))
        img_array = np.array(img.convert('RGB'))

        result = {
            'success': True,
            'image_size': {'width': img.size[0], 'height': img.size[1]},
            'timestamp': datetime.now().isoformat()
        }

        if img.size[0] < 128 or img.size[1] < 128:
            return {'success': False, 'error': 'Image too small (minimum 128x128)'}

        # Try Open-ECG-Digitizer first
        if HAS_OPEN_ECG:
            try:
                open_ecg_result = process_with_open_ecg(img_array, paper_speed, voltage_gain, layout)
                if open_ecg_result['success']:
                    open_ecg_result['timestamp'] = datetime.now().isoformat()
                    return open_ecg_result
            except Exception as e:
                print(f"Open-ECG-Digitizer error: {e}")

        # Fallback
        grid_info = detect_grid(img_array)
        result['grid'] = grid_info
        result['layout'] = layout or classify_layout(img_array)
        result['detected_layout'] = classify_layout(img_array)
        result['method'] = 'fallback'
        result['leads'] = []

        return result

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# ============================================================
# HTML Template with Test Interface
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Digitizer with Testing</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f7;
            color: #1d1d1f;
            line-height: 1.5;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 16px;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { opacity: 0.9; }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 12px 24px;
            border: none;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: all 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .tab:hover { background: #f0f0f0; }
        .tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }

        .card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .card h2 {
            font-size: 1.3em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }

        /* Test Results Styles */
        .test-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .test-btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        .test-btn.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .test-btn.secondary {
            background: #e0e0e0;
            color: #333;
        }
        .test-btn:hover { transform: translateY(-2px); }
        .test-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .test-result {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid #ccc;
            transition: all 0.3s;
        }
        .test-result.pass { border-left-color: #28a745; background: #e8f5e9; }
        .test-result.fail { border-left-color: #dc3545; background: #ffebee; }
        .test-result.running { border-left-color: #007bff; background: #e3f2fd; }

        .test-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .test-name { font-weight: 600; font-size: 1.1em; }
        .test-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .test-status.pass { background: #28a745; color: white; }
        .test-status.fail { background: #dc3545; color: white; }
        .test-status.running { background: #007bff; color: white; }

        .test-message { color: #666; margin-bottom: 10px; }
        .test-details {
            background: rgba(0,0,0,0.05);
            padding: 12px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.85em;
            max-height: 200px;
            overflow-y: auto;
        }
        .test-details pre { white-space: pre-wrap; word-wrap: break-word; }

        /* Summary Stats */
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stat-value { font-size: 2em; font-weight: 700; }
        .stat-label { color: #666; font-size: 0.9em; }
        .stat-card.passed .stat-value { color: #28a745; }
        .stat-card.failed .stat-value { color: #dc3545; }
        .stat-card.total .stat-value { color: #667eea; }

        /* Visualization */
        .visualization-container {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .visualization-container canvas {
            max-width: 100%;
            height: auto;
        }

        /* Feedback Form */
        .feedback-section {
            margin-top: 20px;
            padding: 20px;
            background: #f0f4f8;
            border-radius: 12px;
        }
        .feedback-section h3 { margin-bottom: 15px; }
        .feedback-textarea {
            width: 100%;
            min-height: 100px;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1em;
            resize: vertical;
        }
        .feedback-textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        /* Upload Area */
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-area input { display: none; }
        .preview-img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            margin-top: 15px;
        }

        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: 500; color: #666; }
        .form-group select, .form-group input {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            width: 100%;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
        .btn:disabled { background: #ccc; cursor: not-allowed; transform: none; }

        .status {
            padding: 10px 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
        .status.error { background: #fee; color: #c00; display: block; }
        .status.success { background: #efe; color: #060; display: block; }
        .status.loading { background: #eef; color: #006; display: block; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

        /* Waveform Display */
        .waveform-container {
            background: #fff5f5;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .waveform-canvas {
            width: 100%;
            height: 400px;
            background:
                repeating-linear-gradient(0deg, transparent, transparent 19px, #ffcccc 19px, #ffcccc 20px),
                repeating-linear-gradient(90deg, transparent, transparent 19px, #ffcccc 19px, #ffcccc 20px);
        }

        .export-buttons { display: flex; gap: 10px; margin-top: 20px; }
        .export-btn {
            flex: 1;
            padding: 10px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
        }
        .export-btn:hover { background: #667eea; color: white; }

        /* System Info */
        .system-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        .info-item {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
        }
        .info-label { font-size: 0.85em; color: #666; }
        .info-value { font-weight: 600; }
        .info-value.available { color: #28a745; }
        .info-value.unavailable { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ECG Digitizer with Testing</h1>
            <p>Upload ECG images, run validation tests, and provide feedback</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="showTab('digitize')">Digitize ECG</button>
            <button class="tab" onclick="showTab('tests')">Run Tests</button>
            <button class="tab" onclick="showTab('feedback')">Provide Feedback</button>
        </div>

        <!-- Digitize Tab -->
        <div id="digitize-tab" class="tab-content active">
            <div class="grid">
                <div class="card">
                    <h2>Input</h2>
                    <div class="upload-area" id="uploadArea">
                        <div style="font-size: 48px;">📄</div>
                        <p><strong>Drop ECG image here</strong></p>
                        <p style="color: #888; font-size: 0.9em;">or click to browse</p>
                        <input type="file" id="fileInput" accept="image/*">
                        <img id="previewImg" class="preview-img" style="display: none;">
                    </div>

                    <div class="form-group">
                        <label>Paper Speed</label>
                        <select id="paperSpeed">
                            <option value="25">25 mm/s (standard)</option>
                            <option value="50">50 mm/s</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Voltage Gain</label>
                        <select id="voltageGain">
                            <option value="10">10 mm/mV (standard)</option>
                            <option value="5">5 mm/mV</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Layout</label>
                        <select id="layout">
                            <option value="">Auto-detect</option>
                            <option value="3x4_r1">3×4 + 1 rhythm strip</option>
                            <option value="3x4_r0">3×4 (no rhythm)</option>
                            <option value="6x2_r1">6×2 + 1 rhythm strip</option>
                            <option value="12x1_r0">12×1</option>
                        </select>
                    </div>

                    <button class="btn" id="processBtn" disabled>Process ECG</button>
                    <div class="status" id="status"></div>
                </div>

                <div class="card">
                    <h2>Results</h2>
                    <div id="resultsContainer">
                        <p style="color: #888; text-align: center; padding: 40px;">
                            Upload an ECG image to see results
                        </p>
                    </div>
                </div>
            </div>

            <div class="card" style="margin-top: 20px;" id="waveformCard" hidden>
                <h2>Waveform Visualization</h2>
                <div class="waveform-container">
                    <canvas id="waveformCanvas" class="waveform-canvas"></canvas>
                </div>
                <div class="export-buttons">
                    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
                    <button class="export-btn" onclick="exportJSON()">Export JSON</button>
                </div>
            </div>
        </div>

        <!-- Tests Tab -->
        <div id="tests-tab" class="tab-content">
            <div class="card">
                <h2>System Status</h2>
                <div class="system-info" id="systemInfo">
                    <div class="info-item">
                        <div class="info-label">PyTorch</div>
                        <div class="info-value" id="torchStatus">Checking...</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Open-ECG-Digitizer</div>
                        <div class="info-value" id="openEcgStatus">Checking...</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Test Modules</div>
                        <div class="info-value" id="testModulesStatus">Checking...</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">OpenCV</div>
                        <div class="info-value" id="cv2Status">Checking...</div>
                    </div>
                </div>
            </div>

            <div class="summary-stats" id="testSummary" style="display: none;">
                <div class="stat-card total">
                    <div class="stat-value" id="totalTests">0</div>
                    <div class="stat-label">Total Tests</div>
                </div>
                <div class="stat-card passed">
                    <div class="stat-value" id="passedTests">0</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat-card failed">
                    <div class="stat-value" id="failedTests">0</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="passRate">0%</div>
                    <div class="stat-label">Pass Rate</div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>Waveform Extraction Tests</h2>
                    <div class="test-controls">
                        <button class="test-btn primary" onclick="runWaveformTests()">Run Waveform Tests</button>
                    </div>
                    <div id="waveformTestResults">
                        <p style="color: #888; text-align: center; padding: 20px;">
                            Click "Run Waveform Tests" to validate signal extraction
                        </p>
                    </div>
                </div>

                <div class="card">
                    <h2>Layout Identification Tests</h2>
                    <div class="test-controls">
                        <button class="test-btn primary" onclick="runLayoutTests()">Run Layout Tests</button>
                    </div>
                    <div id="layoutTestResults">
                        <p style="color: #888; text-align: center; padding: 20px;">
                            Click "Run Layout Tests" to validate layout detection
                        </p>
                    </div>
                </div>
            </div>

            <div class="card" style="margin-top: 20px;">
                <h2>Test Visualizations</h2>
                <div id="testVisualization">
                    <p style="color: #888; text-align: center; padding: 20px;">
                        Run tests to see visualizations of extracted waveforms
                    </p>
                </div>
            </div>
        </div>

        <!-- Feedback Tab -->
        <div id="feedback-tab" class="tab-content">
            <div class="card">
                <h2>Provide Feedback on Test Results</h2>
                <p style="margin-bottom: 20px;">
                    Help improve the ECG digitizer by providing feedback on waveform extraction
                    and layout identification accuracy.
                </p>

                <div class="feedback-section">
                    <h3>Waveform Extraction Feedback</h3>
                    <div class="form-group">
                        <label>How accurate was the waveform extraction?</label>
                        <select id="waveformAccuracy">
                            <option value="">Select...</option>
                            <option value="excellent">Excellent - Very accurate</option>
                            <option value="good">Good - Minor issues</option>
                            <option value="fair">Fair - Some noticeable errors</option>
                            <option value="poor">Poor - Significant errors</option>
                            <option value="failed">Failed - Did not work</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Specific issues observed:</label>
                        <textarea class="feedback-textarea" id="waveformIssues" placeholder="Describe any issues with waveform extraction (e.g., missed peaks, noise, baseline wander)..."></textarea>
                    </div>
                </div>

                <div class="feedback-section">
                    <h3>Layout Identification Feedback</h3>
                    <div class="form-group">
                        <label>Was the layout correctly identified?</label>
                        <select id="layoutAccuracy">
                            <option value="">Select...</option>
                            <option value="correct">Yes - Correct layout detected</option>
                            <option value="partial">Partial - Some leads misidentified</option>
                            <option value="wrong">Wrong - Incorrect layout detected</option>
                            <option value="failed">Failed - Could not identify</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Expected layout (if different):</label>
                        <select id="expectedLayout">
                            <option value="">Select expected layout...</option>
                            <option value="3x4_r1">3×4 + 1 rhythm strip</option>
                            <option value="3x4_r0">3×4 (no rhythm)</option>
                            <option value="6x2_r1">6×2 + 1 rhythm strip</option>
                            <option value="6x2_r0">6×2 (no rhythm)</option>
                            <option value="12x1_r0">12×1</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Additional notes:</label>
                        <textarea class="feedback-textarea" id="layoutNotes" placeholder="Any additional observations about layout detection..."></textarea>
                    </div>
                </div>

                <div class="feedback-section">
                    <h3>General Feedback</h3>
                    <textarea class="feedback-textarea" id="generalFeedback" placeholder="Any other feedback, suggestions, or issues..."></textarea>
                </div>

                <button class="btn" style="margin-top: 20px;" onclick="submitFeedback()">Submit Feedback</button>
                <div class="status" id="feedbackStatus"></div>
            </div>

            <div class="card" style="margin-top: 20px;">
                <h2>Feedback History</h2>
                <div id="feedbackHistory">
                    <p style="color: #888; text-align: center; padding: 20px;">
                        No feedback submitted yet
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentResult = null;
        let imageData = null;
        let allTestResults = [];
        let feedbackHistory = [];

        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');

            if (tabName === 'tests') {
                loadSystemStatus();
            }
        }

        // Load system status
        async function loadSystemStatus() {
            try {
                const response = await fetch('/api/system-status');
                const status = await response.json();

                updateStatusBadge('torchStatus', status.has_torch);
                updateStatusBadge('openEcgStatus', status.has_open_ecg);
                updateStatusBadge('testModulesStatus', status.has_tests);
                updateStatusBadge('cv2Status', status.has_cv2);
            } catch (e) {
                console.error('Failed to load system status:', e);
            }
        }

        function updateStatusBadge(elementId, available) {
            const el = document.getElementById(elementId);
            el.textContent = available ? 'Available' : 'Not Available';
            el.className = 'info-value ' + (available ? 'available' : 'unavailable');
        }

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewImg = document.getElementById('previewImg');
        const processBtn = document.getElementById('processBtn');
        const status = document.getElementById('status');

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
            status.textContent = 'Processing...';
            processBtn.disabled = true;

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image: imageData,
                        paper_speed: parseInt(document.getElementById('paperSpeed').value),
                        voltage_gain: parseInt(document.getElementById('voltageGain').value),
                        layout: document.getElementById('layout').value || null
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
            let html = `
                <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="color: #666; font-size: 0.9em;">Image Size</div>
                    <div style="font-weight: 600;">${result.image_size.width} × ${result.image_size.height}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="color: #666; font-size: 0.9em;">Detected Layout</div>
                    <div style="font-weight: 600;">${result.layout || 'Unknown'}</div>
                </div>
                <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="color: #666; font-size: 0.9em;">Method</div>
                    <div style="font-weight: 600;">${result.method || 'standard'}</div>
                </div>
            `;

            if (result.leads && result.leads.length > 0) {
                html += `<div style="background: #f8f9fa; padding: 12px; border-radius: 8px;">
                    <div style="color: #666; font-size: 0.9em;">Leads Extracted</div>
                    <div style="font-weight: 600;">${result.leads.length} leads</div>
                </div>`;
            }

            container.innerHTML = html;

            if (result.leads && result.leads.length > 0) {
                document.getElementById('waveformCard').hidden = false;
                drawWaveforms(result);
            }
        }

        function drawWaveforms(result) {
            const canvas = document.getElementById('waveformCanvas');
            const ctx = canvas.getContext('2d');

            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);

            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;

            ctx.fillStyle = '#fff5f5';
            ctx.fillRect(0, 0, width, height);

            // Draw grid
            ctx.strokeStyle = '#ffcccc';
            ctx.lineWidth = 0.5;
            for (let x = 0; x < width; x += 20) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            for (let y = 0; y < height; y += 20) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            // Draw waveforms
            const leads = result.leads;
            const cols = 4;
            const rows = Math.ceil(leads.length / cols);
            const cellWidth = width / cols;
            const cellHeight = height / rows;

            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.font = '12px sans-serif';
            ctx.fillStyle = '#333';

            leads.forEach((lead, idx) => {
                const col = idx % cols;
                const row = Math.floor(idx / cols);
                const x0 = col * cellWidth + 10;
                const y0 = row * cellHeight;
                const w = cellWidth - 20;
                const h = cellHeight;
                const centerY = y0 + h / 2;

                ctx.fillText(lead.name, x0, y0 + 15);

                if (lead.samples && lead.samples.length > 0) {
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

        // Test functions
        async function runWaveformTests() {
            const container = document.getElementById('waveformTestResults');
            container.innerHTML = '<div class="test-result running"><div class="test-header"><span class="test-name">Running tests...</span><span class="test-status running">Running</span></div></div>';

            try {
                const response = await fetch('/api/tests/waveform');
                const results = await response.json();
                displayTestResults(results, 'waveformTestResults');
                updateTestSummary();
            } catch (e) {
                container.innerHTML = `<div class="test-result fail"><div class="test-message">Failed to run tests: ${e.message}</div></div>`;
            }
        }

        async function runLayoutTests() {
            const container = document.getElementById('layoutTestResults');
            container.innerHTML = '<div class="test-result running"><div class="test-header"><span class="test-name">Running tests...</span><span class="test-status running">Running</span></div></div>';

            try {
                const response = await fetch('/api/tests/layout');
                const results = await response.json();
                displayTestResults(results, 'layoutTestResults');
                updateTestSummary();
            } catch (e) {
                container.innerHTML = `<div class="test-result fail"><div class="test-message">Failed to run tests: ${e.message}</div></div>`;
            }
        }

        function displayTestResults(results, containerId) {
            const container = document.getElementById(containerId);
            allTestResults = allTestResults.concat(results);

            let html = '';
            results.forEach(r => {
                const statusClass = r.passed ? 'pass' : 'fail';
                const statusText = r.passed ? 'PASS' : 'FAIL';

                html += `
                    <div class="test-result ${statusClass}">
                        <div class="test-header">
                            <span class="test-name">${r.name}</span>
                            <span class="test-status ${statusClass}">${statusText}</span>
                        </div>
                        <div class="test-message">${r.message}</div>
                        ${r.details ? `<div class="test-details"><pre>${JSON.stringify(r.details, null, 2)}</pre></div>` : ''}
                    </div>
                `;
            });

            container.innerHTML = html;

            // Update visualization if available
            const vizResults = results.filter(r => r.visual_data && r.visual_data.extracted_lines);
            if (vizResults.length > 0) {
                displayTestVisualization(vizResults);
            }
        }

        function displayTestVisualization(results) {
            const container = document.getElementById('testVisualization');
            let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">';

            results.forEach(r => {
                if (r.visual_data.extracted_lines && r.visual_data.extracted_lines.length > 0) {
                    const lines = r.visual_data.extracted_lines;
                    html += `
                        <div style="background: white; padding: 15px; border-radius: 8px;">
                            <h4 style="margin-bottom: 10px;">${r.name}</h4>
                            <canvas id="viz-${r.name.replace(/\\s+/g, '-')}" width="400" height="150" style="width: 100%; border: 1px solid #eee;"></canvas>
                        </div>
                    `;
                }
            });

            html += '</div>';
            container.innerHTML = html;

            // Draw visualizations
            results.forEach(r => {
                if (r.visual_data.extracted_lines && r.visual_data.extracted_lines.length > 0) {
                    const canvasId = `viz-${r.name.replace(/\\s+/g, '-')}`;
                    const canvas = document.getElementById(canvasId);
                    if (canvas) {
                        drawTestVisualization(canvas, r.visual_data.extracted_lines);
                    }
                }
            });
        }

        function drawTestVisualization(canvas, lines) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const colors = ['#667eea', '#764ba2', '#28a745', '#dc3545', '#ffc107', '#17a2b8'];

            lines.forEach((line, idx) => {
                ctx.strokeStyle = colors[idx % colors.length];
                ctx.lineWidth = 2;
                ctx.beginPath();

                const xScale = canvas.width / line.length;
                const yMin = Math.min(...line.filter(v => !isNaN(v)));
                const yMax = Math.max(...line.filter(v => !isNaN(v)));
                const yRange = yMax - yMin || 1;

                let started = false;
                line.forEach((y, x) => {
                    if (!isNaN(y)) {
                        const px = x * xScale;
                        const py = canvas.height - ((y - yMin) / yRange) * (canvas.height - 20) - 10;
                        if (!started) {
                            ctx.moveTo(px, py);
                            started = true;
                        } else {
                            ctx.lineTo(px, py);
                        }
                    }
                });
                ctx.stroke();
            });
        }

        function updateTestSummary() {
            const total = allTestResults.length;
            const passed = allTestResults.filter(r => r.passed).length;
            const failed = total - passed;
            const rate = total > 0 ? Math.round((passed / total) * 100) : 0;

            document.getElementById('totalTests').textContent = total;
            document.getElementById('passedTests').textContent = passed;
            document.getElementById('failedTests').textContent = failed;
            document.getElementById('passRate').textContent = rate + '%';
            document.getElementById('testSummary').style.display = 'grid';
        }

        // Feedback functions
        function submitFeedback() {
            const feedback = {
                timestamp: new Date().toISOString(),
                waveform: {
                    accuracy: document.getElementById('waveformAccuracy').value,
                    issues: document.getElementById('waveformIssues').value
                },
                layout: {
                    accuracy: document.getElementById('layoutAccuracy').value,
                    expected: document.getElementById('expectedLayout').value,
                    notes: document.getElementById('layoutNotes').value
                },
                general: document.getElementById('generalFeedback').value,
                testResults: allTestResults,
                processedImage: currentResult
            };

            feedbackHistory.push(feedback);

            // Save to localStorage
            localStorage.setItem('ecg_feedback', JSON.stringify(feedbackHistory));

            const status = document.getElementById('feedbackStatus');
            status.className = 'status success';
            status.textContent = 'Feedback submitted successfully!';

            // Clear form
            document.getElementById('waveformAccuracy').value = '';
            document.getElementById('waveformIssues').value = '';
            document.getElementById('layoutAccuracy').value = '';
            document.getElementById('expectedLayout').value = '';
            document.getElementById('layoutNotes').value = '';
            document.getElementById('generalFeedback').value = '';

            displayFeedbackHistory();
        }

        function displayFeedbackHistory() {
            const container = document.getElementById('feedbackHistory');

            if (feedbackHistory.length === 0) {
                container.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No feedback submitted yet</p>';
                return;
            }

            let html = '';
            feedbackHistory.forEach((fb, idx) => {
                html += `
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <strong>Feedback #${idx + 1}</strong>
                            <span style="color: #888; font-size: 0.85em;">${new Date(fb.timestamp).toLocaleString()}</span>
                        </div>
                        <div style="font-size: 0.9em;">
                            <div>Waveform: ${fb.waveform.accuracy || 'Not rated'}</div>
                            <div>Layout: ${fb.layout.accuracy || 'Not rated'}</div>
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
        }

        function exportCSV() {
            if (!currentResult || !currentResult.leads) return;

            let csv = '# ECG Digitization Result\\n';
            csv += '# Layout: ' + currentResult.layout + '\\n\\n';
            csv += 'timestamp_ms,lead,voltage_mv\\n';

            currentResult.leads.forEach(lead => {
                const interval = lead.duration_ms / lead.samples.length;
                lead.samples.forEach((v, i) => {
                    csv += `${(i * interval).toFixed(1)},${lead.name},${v.toFixed(4)}\\n`;
                });
            });

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

        // Load saved feedback on page load
        window.onload = function() {
            const saved = localStorage.getItem('ecg_feedback');
            if (saved) {
                feedbackHistory = JSON.parse(saved);
                displayFeedbackHistory();
            }
            loadSystemStatus();
        };
    </script>
</body>
</html>
'''


# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/system-status')
def system_status():
    return jsonify({
        'has_torch': HAS_TORCH,
        'has_open_ecg': HAS_OPEN_ECG,
        'has_tests': HAS_TESTS,
        'has_cv2': HAS_CV2
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.json
    result = process_image(
        data.get('image'),
        data.get('paper_speed', 25),
        data.get('voltage_gain', 10),
        data.get('layout')
    )
    return jsonify(result)


@app.route('/api/tests/waveform')
def api_waveform_tests():
    if not HAS_TESTS:
        return jsonify([{
            'name': 'Test Module Check',
            'passed': False,
            'message': 'Test modules not available'
        }])

    try:
        results = run_waveform_tests()
        return jsonify(results)
    except Exception as e:
        return jsonify([{
            'name': 'Test Execution',
            'passed': False,
            'message': f'Error running tests: {str(e)}'
        }])


@app.route('/api/tests/layout')
def api_layout_tests():
    if not HAS_TESTS:
        return jsonify([{
            'name': 'Test Module Check',
            'passed': False,
            'message': 'Test modules not available'
        }])

    try:
        results = run_layout_tests()
        return jsonify(results)
    except Exception as e:
        return jsonify([{
            'name': 'Test Execution',
            'passed': False,
            'message': f'Error running tests: {str(e)}'
        }])


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.json

    # Save feedback to file
    feedback_dir = Path(__file__).parent / 'feedback'
    feedback_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    feedback_file = feedback_dir / f'feedback_{timestamp}.json'

    with open(feedback_file, 'w') as f:
        json.dump(data, f, indent=2)

    return jsonify({'success': True, 'saved_to': str(feedback_file)})


if __name__ == '__main__':
    print("=" * 60)
    print("ECG Digitizer with Interactive Testing")
    print("=" * 60)
    print(f"\nSystem Status:")
    print(f"  PyTorch: {'Available' if HAS_TORCH else 'Not Available'}")
    print(f"  Open-ECG-Digitizer: {'Available' if HAS_OPEN_ECG else 'Not Available'}")
    print(f"  Test Modules: {'Available' if HAS_TESTS else 'Not Available'}")
    print(f"  OpenCV: {'Available' if HAS_CV2 else 'Not Available'}")
    print("\nStarting server...")
    print("Open http://localhost:8080 in your browser\n")

    app.run(host='0.0.0.0', port=8080, debug=True)
