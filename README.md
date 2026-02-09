# ECG Digitizer

A complete ECG digitization solution that converts paper ECG images into machine-readable waveform data. Supports both **on-device iOS processing** using ONNX Runtime and **server-based processing** via a Fly.io-hosted API.

## Overview

This project provides two processing modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **On-Device (iOS)** | 100% local ONNX Runtime inference on iPhone | Privacy-critical, offline, low latency (~2s) |
| **Server (Fly.io)** | PyTorch-based processing via REST API | Full pipeline with TPS dewarping, diagnostic output |

**Live Server**: https://ecg-digitizer.fly.dev

## Repository Structure

```
ecg_app/
├── ECGDigitizer/           # iOS app (SwiftUI + ONNX Runtime)
│   └── ECGDigitizer/
│       ├── App/            # Entry point, dependency injection
│       ├── Domain/         # Business logic, models
│       ├── Infrastructure/ # ONNX inference, image processing
│       ├── Presentation/   # SwiftUI views
│       └── Resources/      # ONNX model files
│
├── Open-ECG-Digitizer/     # Python/PyTorch digitization engine
│   └── src/
│       ├── model/          # UNet models, dewarping, calibration
│       └── digitizer.py    # Main processing pipeline
│
├── web/                    # Flask server for Fly.io
│   ├── app.py              # API endpoints
│   ├── digitizer_wrapper.py
│   ├── debug_pipeline.py   # Diagnostic visualization
│   └── requirements.txt
│
├── scripts/                # Utility scripts
│   └── convert_to_onnx.py  # PyTorch → ONNX conversion
│
├── fly.toml                # Fly.io deployment config
├── Dockerfile.fly          # Server container
└── .dockerignore
```

---

## Part 1: iOS App (On-Device Processing)

### Features

- **100% On-Device Processing** - No network required after initial setup
- **ONNX Runtime Integration** - PyTorch models converted to ONNX format
- **Neural Engine Acceleration** - Hardware-accelerated on A12+ chips
- **12-Lead ECG Support** - Extracts all standard leads (I, II, III, aVR, aVL, aVF, V1-V6)
- **Multiple Export Formats** - CSV, PDF, HL7 FHIR

### Requirements

- Xcode 15.0+
- iOS 17.0+
- Swift 5.9+
- ONNX Runtime Swift Package (1.20.0) - auto-fetched

### Build Steps

1. Open Xcode project:
   ```bash
   open ECGDigitizer/ECGDigitizer.xcodeproj
   ```

2. Build and run on device (iOS 17.0+)

### Processing Pipeline

```swift
// 1. Load ONNX models (at app startup)
ONNXInference.shared.loadModels()

// 2. Run Segmentation UNet (on Neural Engine)
let segmentation = try await ONNXInference.runSegmentation(image)
// → Returns 4-channel probability map: Grid, Text, Signal, Background

// 3. Extract Grid Calibration via autocorrelation
let calibration = GridSizeFinder.findGridCalibration(gridProb)
// → Detects 5mm grid spacing in pixels

// 4. Extract Waveforms using weighted centroid
let waveforms = SignalExtractor.extractWaveforms(signalProb, calibration)
// → Converts pixel positions to voltage (µV)

// 5. Return 12-lead ECG Recording
return ECGRecording(leads: leads, calibration: calibration)
```

### Performance (iPhone 14 Pro)

| Operation | Time |
|-----------|------|
| Model Loading | 3-5s (once) |
| Segmentation | 0.8s |
| Calibration | 0.1s |
| Extraction | 0.3s |
| **Total** | **~2s** |

---

## Part 2: Server API (Fly.io)

### Deployment

The server runs on Fly.io with the following configuration:

- **Region**: San Jose (sjc)
- **VM**: shared-cpu-4x, 8GB RAM
- **Framework**: Flask + Gunicorn
- **ML Stack**: PyTorch + OpenCV

### Deploy Commands

```bash
cd /Users/pae2/Desktop/ecg_app
fly deploy
```

### API Endpoints

#### `GET /`
Web interface for manual ECG upload and processing.

#### `GET /api/health`
Health check for load balancers.

```json
{
  "status": "healthy",
  "service": "ecg-digitizer",
  "open_ecg_available": true
}
```

#### `POST /api/process`
Main ECG processing endpoint.

**Request:**
```json
{
  "image": "<base64-encoded-image>",
  "layout": "3x4_1",  // optional: "3x4_1", "6x2", "12x1"
  "signal_based_boundaries": true
}
```

**Response:**
```json
{
  "success": true,
  "leads": {
    "I": { "samples": [...], "sample_rate": 500 },
    "II": { "samples": [...], "sample_rate": 500 },
    ...
  },
  "calibration": {
    "pixels_per_mm": 10.5,
    "quality_score": 0.85
  }
}
```

#### `POST /api/diagnostic`
Full diagnostic output with visualization (for debugging).

**Response includes:**
- Segmentation probability maps (grid, text, signal)
- Grid calibration details
- Lead bounding boxes
- Extracted waveforms as images

### Memory Management

The server enforces a 2MB image size limit and uses aggressive garbage collection to stay within 8GB RAM:

```python
MAX_IMAGE_SIZE_BYTES = 2 * 1024 * 1024
MAX_DIAGNOSTIC_DIMENSION = 2000

# After processing
del image_data
gc.collect()
```

---

## Part 3: Processing Algorithm

### UNet Models

| Model | Input | Output | Size |
|-------|-------|--------|------|
| Segmentation | RGB (1,3,H,W) | 4-channel prob map | 81.7 MB |
| Lead Identifier | Text prob (1,1,H,W) | 13-channel lead map | 20.5 MB |

### Grid Calibration (Autocorrelation)

```
Grid Probability Map → Column Sum → Autocorrelation → Peak Spacing
                                                      ↓
                                              pixels_per_mm = spacing / 5mm
```

Standard ECG paper: 5mm grid squares, 1mm = 0.1mV

### Signal Extraction (Weighted Centroid)

For each column:
1. Extract signal probability values
2. Compute weighted centroid: `y = Σ(y_i × prob_i) / Σ(prob_i)`
3. Convert to voltage: `voltage_µV = (y - baseline) × 0.1mV/mm × 1000`

---

## Configuration

### iOS Processing Mode

Edit `CaptureFlowView.swift`:
```swift
@Published var useServerProcessing = false  // On-device (default)
@Published var useServerProcessing = true   // Server fallback
```

### Server URL

If using server mode, configure in `ECGAPIClient.swift`:
```swift
private let baseURL = "https://ecg-digitizer.fly.dev"
```

---

## Troubleshooting

### iOS: "Model not loaded"
- Verify `.onnx` files are in Resources folder
- Check files are added to Xcode target

### iOS: Slow inference (>3s)
- Ensure CoreML EP is enabled in `ONNXInference.swift`
- Check for Neural Engine activity in Instruments

### Server: HTTP 502 errors
- Check Fly.io logs: `fly logs`
- Usually OOM - image may be too large
- Server auto-scales with 8GB RAM limit

### Server: Deploy fails (image too large)
- Docker image must be <8GB uncompressed
- Check `.dockerignore` excludes test/assets folders

---

## Development

### Convert PyTorch to ONNX

```bash
python scripts/convert_to_onnx.py
# Outputs: ECGSegmentation.onnx, ECGLeadIdentifier.onnx
```

### Run Server Locally

```bash
cd web
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

### Run with Gunicorn (Production)

```bash
cd web
gunicorn --bind 0.0.0.0:8080 --timeout 300 --workers 1 app:app
```

---

## Cost

| Component | Cost |
|-----------|------|
| On-device (iOS) | Free |
| Fly.io Server | ~$48/month (shared-cpu-4x, 8GB RAM) |

---

## References

- **ONNX Runtime**: https://onnxruntime.ai/
- **Fly.io**: https://fly.io/
- **ECG Standards**: 1mm = 0.1mV amplitude, 1mm = 0.04s at 25mm/s

---

## License

[Specify license]

---

**GitHub**: https://github.com/PierreElias/ecg-digitizer

**Last Updated**: 2026-02-09

**Version**: 0.6.0-a1
