import Foundation
import UIKit
import CoreML
import Vision
import Accelerate

/// On-device ECG processing using Core ML models
/// This implements the full ECG digitization pipeline without network dependency
///
/// # SERVER-BASED IMPLEMENTATION ALTERNATIVE:
/// To use server-based processing instead of on-device:
/// 1. Replace the processECG() method to call ECGAPIClient.digitize()
/// 2. Keep offline queue for when network is unavailable
/// 3. Remove Core ML model loading and inference code
/// 4. See marked sections below with "# CHANGE FOR SERVER-BASED IMPLEMENTATION"
///
/// Note: Heavy processing is done on background threads to avoid blocking the UI
class OnDeviceECGProcessor: ObservableObject {

    // MARK: - Singleton

    nonisolated(unsafe) static let shared = OnDeviceECGProcessor()

    // MARK: - Properties

    /// Current processing mode
    enum ProcessingMode: String {
        case onDevice = "onDevice"   // Use Core ML models locally
        case server = "server"       // Use remote server processing
        case hybrid = "hybrid"       // ONNX on device + server post-processing
    }

    // Processing mode synced with AppStorage
    // Default is hybrid: ONNX segmentation on device + Python post-processing on server
    @Published var processingMode: ProcessingMode {
        didSet {
            UserDefaults.standard.set(processingMode.rawValue, forKey: "processingMode")
        }
    }

    // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    // This client would be used when processingMode is .server or .hybrid
    private let apiClient = ECGAPIClient.shared

    // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    // Add offline queue manager here
    // private let offlineQueue = OfflineProcessingQueue()

    // MARK: - ONNX Models (On-Device Only)

    // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    // These models are only needed for on-device processing
    // Remove if using server-based processing exclusively
    private let onnxInference = ONNXInference.shared

    // MARK: - Initialization

    // Models loaded on-demand flag
    private var modelsLoaded = false

    init() {
        // Load processing mode from UserDefaults (default to hybrid)
        let savedMode = UserDefaults.standard.string(forKey: "processingMode") ?? "hybrid"
        self.processingMode = ProcessingMode(rawValue: savedMode) ?? .hybrid

        // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
        // Models are now loaded on-demand when first needed, not at init time
        // This prevents blocking the main thread when the capture view appears
    }

    // MARK: - Model Loading (On-Device Only)

    /// Load ONNX models for on-device processing (lazy loading)
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// This entire function can be removed if using server-based processing
    private func loadModels() async {
        // Only load once
        if modelsLoaded {
            return
        }

        do {
            try await onnxInference.loadModels()
            modelsLoaded = true
            print("✅ ONNX models loaded successfully")
        } catch {
            print("⚠️ Failed to load ONNX models: \(error)")
            print("   Falling back to server processing")
            await MainActor.run {
                processingMode = .server
            }
        }
    }

    // MARK: - Main Processing

    /// Process ECG image and extract waveforms
    /// - Parameters:
    ///   - image: Input ECG image
    ///   - parameters: Processing parameters
    ///   - progressCallback: Progress updates
    /// - Returns: Processed ECG recording
    func processECG(
        image: UIImage,
        parameters: ProcessingParameters,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> ECGRecording {

        // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
        // Replace entire method body with:
        /*
        // Check network connectivity
        if NetworkMonitor.shared.isConnected {
            // Use server processing
            return try await processWithServer(
                image: image,
                parameters: parameters,
                progressCallback: progressCallback
            )
        } else {
            // Queue for offline processing
            offlineQueue.add(image: image, parameters: parameters)
            throw ECGProcessingError.offline(
                message: "No network connection. ECG queued for processing when online."
            )
        }
        */

        // Current on-device implementation:
        switch processingMode {
        case .onDevice:
            return try await processOnDevice(
                image: image,
                parameters: parameters,
                progressCallback: progressCallback
            )

        case .server:
            // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
            // This would be the primary path
            return try await processWithServer(
                image: image,
                parameters: parameters,
                progressCallback: progressCallback
            )

        case .hybrid:
            // ONNX segmentation on device + Python post-processing on server
            // This gives the best of both worlds: fast segmentation + accurate extraction
            return try await processHybrid(
                image: image,
                parameters: parameters,
                progressCallback: progressCallback
            )
        }
    }

    // MARK: - On-Device Processing

    /// Process ECG using on-device ONNX models
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// This entire method can be removed if using server-based processing
    private func processOnDevice(
        image: UIImage,
        parameters: ProcessingParameters,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> ECGRecording {
        let logger = DiagnosticLogger.shared

        // Load ONNX models on-demand (only once, on background thread)
        await loadModels()

        // Step 1: Validate image
        progressCallback(.validatingImage)
        guard let cgImage = image.cgImage else {
            throw ECGProcessingError.invalidImage
        }

        // Save input image for debugging
        logger.saveInputImage(image, name: "ecg_input")

        // Step 2: Run segmentation model
        progressCallback(.detectingGrid(progress: 0.2))
        let segmentationResult = try await runSegmentation(cgImage)

        // Save probability maps for debugging visualization
        logger.saveProbabilityMap(
            segmentationResult.signalProb,
            width: segmentationResult.width,
            height: segmentationResult.height,
            name: "signal_prob"
        )
        logger.saveProbabilityMap(
            segmentationResult.gridProb,
            width: segmentationResult.width,
            height: segmentationResult.height,
            name: "grid_prob"
        )
        logger.saveProbabilityMap(
            segmentationResult.textProb,
            width: segmentationResult.width,
            height: segmentationResult.height,
            name: "text_prob"
        )
        // Color-coded overlay (Red=Grid, Green=Signal, Blue=Text)
        logger.saveOverlayVisualization(
            signalProb: segmentationResult.signalProb,
            gridProb: segmentationResult.gridProb,
            textProb: segmentationResult.textProb,
            width: segmentationResult.width,
            height: segmentationResult.height
        )

        logger.log("📊 Debug images saved to: \(logger.getDebugImagesPath())")

        // Step 3: Extract grid parameters
        progressCallback(.detectingGrid(progress: 0.4))
        let gridCalibration = try extractGridCalibration(from: segmentationResult)

        // Step 4: Extract waveforms from signal probability map
        progressCallback(.extractingWaveforms(currentLead: 1, totalLeads: 12))
        let (waveforms, extractionAlgorithm) = try extractWaveforms(
            from: segmentationResult,
            calibration: gridCalibration
        )

        // Step 5: Classify layout
        progressCallback(.detectingGrid(progress: 0.7))
        let layout = try classifyLayout(waveforms: waveforms)

        // Step 6: Match leads to canonical positions
        progressCallback(.validatingResults)
        let leads = try matchLeadsToCanonical(
            waveforms: waveforms,
            layout: layout
        )

        // Step 7: Create recording with extraction algorithm metadata
        let metadata = ECGMetadata(extractionAlgorithm: extractionAlgorithm)
        let recording = ECGRecording(
            id: UUID(),
            timestamp: Date(),
            originalImageData: image.jpegData(compressionQuality: 0.9),
            parameters: parameters,
            layout: layout,
            leads: leads,
            gridCalibration: gridCalibration,
            metadata: metadata,
            validationStatus: .valid
        )

        return recording
    }

    // MARK: - Hybrid Processing (ONNX on device + Python post-processing)

    /// Process ECG using hybrid approach:
    /// 1. Run ONNX segmentation on device (fast, ~1s)
    /// 2. Send probability maps to server for post-processing (accurate extraction)
    ///
    /// This combines the speed of on-device inference with the accuracy of
    /// Python post-processing (perspective correction, dewarping, signal extraction)
    private func processHybrid(
        image: UIImage,
        parameters: ProcessingParameters,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> ECGRecording {
        let logger = DiagnosticLogger.shared
        logger.log("🔀 Starting hybrid processing (ONNX + server)")

        // Load ONNX models on-demand
        await loadModels()

        // Step 1: Validate image
        progressCallback(.validatingImage)
        guard let cgImage = image.cgImage else {
            throw ECGProcessingError.invalidImage
        }

        // Save input for debugging (low quality to save memory)
        logger.saveInputImage(image, name: "hybrid_input")

        // Step 2: Run ONNX segmentation on device
        progressCallback(.detectingGrid(progress: 0.2))
        logger.log("⚙️ Running ONNX segmentation on device...")
        logger.log("   Input image: \(cgImage.width)x\(cgImage.height)")

        guard onnxInference.isReady else {
            logger.log("⚠️ ONNX not ready, falling back to full server processing")
            return try await processWithServer(
                image: image,
                parameters: parameters,
                progressCallback: progressCallback
            )
        }

        // Run ONNX segmentation (memory is managed inside ONNXInference with autoreleasepool)
        let segmentationResult = try await onnxInference.runSegmentation(image: cgImage)

        logger.log("✅ ONNX segmentation complete: \(segmentationResult.width)x\(segmentationResult.height)")

        // Save probability maps for debugging
        logger.saveProbabilityMap(
            segmentationResult.signalProb,
            width: segmentationResult.width,
            height: segmentationResult.height,
            name: "hybrid_signal"
        )
        logger.saveProbabilityMap(
            segmentationResult.gridProb,
            width: segmentationResult.width,
            height: segmentationResult.height,
            name: "hybrid_grid"
        )

        // Step 3: Send probability maps to server for post-processing
        progressCallback(.classifyingLayout)
        logger.log("📤 Sending probability maps to server for post-processing...")

        // Store results in local variables that can be released
        let signalProb = segmentationResult.signalProb
        let gridProb = segmentationResult.gridProb
        let textProb = segmentationResult.textProb
        let width = segmentationResult.width
        let height = segmentationResult.height

        // Release segmentationResult to free memory before server call
        // (The probability arrays are now held by local variables)

        do {
            let response = try await apiClient.postprocess(
                signalProb: signalProb,
                gridProb: gridProb,
                textProb: textProb,
                width: width,
                height: height,
                originalImage: image,  // Send original image for better extraction
                progressCallback: progressCallback
            )

            logger.log("✅ Server post-processing complete")

            // Convert response to recording
            guard let recording = response.toRecording(
                originalImage: image,
                parameters: parameters
            ) else {
                throw ECGProcessingError.serverError("Failed to create recording from hybrid response")
            }

            return recording

        } catch {
            // Server handles all processing including fallback to sectioning
            // If server fails, report the error - don't use unreliable on-device processing
            logger.log("❌ Server processing failed: \(error.localizedDescription)")
            logger.log("   Ensure Python server is running on the correct port")
            logger.log("   Check Settings → Server URL (default: http://localhost:8080)")

            throw ECGProcessingError.serverError(
                "Server processing failed. Please ensure the server is running. Error: \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Server Processing

    /// Process ECG using remote server
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// This would be the primary processing method
    /// Move network connectivity check here and add offline queue
    private func processWithServer(
        image: UIImage,
        parameters: ProcessingParameters,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> ECGRecording {

        // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
        // Add network connectivity check:
        /*
        guard NetworkMonitor.shared.isConnected else {
            // Queue for offline processing
            let queuedItem = OfflineQueueItem(
                image: image,
                parameters: parameters,
                timestamp: Date()
            )
            offlineQueue.add(queuedItem)

            throw ECGProcessingError.offline(
                message: "No network connection. ECG will be processed when connection is restored."
            )
        }
        */

        // Use existing API client
        let response = try await apiClient.digitize(
            image: image,
            parameters: parameters,
            progressCallback: progressCallback
        )

        guard let recording = response.toRecording(
            originalImage: image,
            parameters: parameters
        ) else {
            throw ECGProcessingError.serverError("Failed to create recording from server response")
        }

        return recording
    }


    // MARK: - Diagnostic Processing

    /// Run ONNX segmentation with full diagnostic report
    ///
    /// This method provides detailed step-by-step metrics for debugging
    /// and validating the on-device processing pipeline.
    ///
    /// - Parameter image: Input ECG image
    /// - Returns: Tuple of (SegmentationResult, ONNXDiagnosticReport)
    func runSegmentationWithDiagnostics(image: UIImage) async throws -> (SegmentationResult, ONNXDiagnosticReport) {
        let logger = DiagnosticLogger.shared
        logger.log("🔬 Running diagnostic segmentation...")

        // Load ONNX models on-demand
        await loadModels()

        guard let cgImage = image.cgImage else {
            throw ECGProcessingError.invalidImage
        }

        guard onnxInference.isReady else {
            throw ECGProcessingError.modelNotLoaded
        }

        // Run segmentation with diagnostics
        let (result, report) = try await onnxInference.runSegmentationWithDiagnostics(image: cgImage)

        // Log the report
        logger.log(report.generateReport())

        // Save diagnostic images
        logger.saveInputImage(image, name: "diag_input")
        logger.saveProbabilityMap(
            result.signalProb,
            width: result.width,
            height: result.height,
            name: "diag_signal"
        )
        logger.saveProbabilityMap(
            result.gridProb,
            width: result.width,
            height: result.height,
            name: "diag_grid"
        )
        logger.saveProbabilityMap(
            result.textProb,
            width: result.width,
            height: result.height,
            name: "diag_text"
        )

        // Color-coded overlay
        logger.saveOverlayVisualization(
            signalProb: result.signalProb,
            gridProb: result.gridProb,
            textProb: result.textProb,
            width: result.width,
            height: result.height
        )

        logger.log("📊 Diagnostic images saved to: \(logger.getDebugImagesPath())")

        return (result, report)
    }

    /// Generate a full diagnostic report as a formatted string
    ///
    /// Runs the ONNX pipeline and returns a detailed report of each step
    func generateDiagnosticReport(image: UIImage) async throws -> String {
        let (_, report) = try await runSegmentationWithDiagnostics(image: image)
        return report.generateReport()
    }

    /// Generate a visual diagnostic report with step-by-step images
    ///
    /// Returns a VisualDiagnosticReport containing:
    /// - Input image
    /// - Signal probability heatmap (green)
    /// - Grid probability heatmap (red)
    /// - Text probability heatmap (blue)
    /// - RGB overlay
    /// - Text metrics
    func generateVisualDiagnosticReport(image: UIImage) async throws -> VisualDiagnosticReport {
        let logger = DiagnosticLogger.shared
        logger.log("🔬 Running visual diagnostic segmentation...")

        // Load ONNX models on-demand
        await loadModels()

        guard let cgImage = image.cgImage else {
            throw ECGProcessingError.invalidImage
        }

        guard onnxInference.isReady else {
            throw ECGProcessingError.modelNotLoaded
        }

        // Run visual diagnostic
        let visualReport = try await onnxInference.runVisualDiagnostic(image: cgImage, inputUIImage: image)

        logger.log("✅ Visual diagnostic complete")
        logger.log(visualReport.textReport)

        return visualReport
    }

    // MARK: - Segmentation (On-Device Only)

    /// Run UNet segmentation model
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Remove this method - server handles segmentation
    private func runSegmentation(_ image: CGImage) async throws -> SegmentationResult {
        guard onnxInference.isReady else {
            throw ECGProcessingError.modelNotLoaded
        }

        return try await onnxInference.runSegmentation(image: image)
    }

    // MARK: - Grid Extraction (On-Device Only)

    /// Extract grid calibration from segmentation result
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Remove this method - server handles grid detection
    private func extractGridCalibration(from segmentation: SegmentationResult) throws -> GridCalibration {
        // Use GridSizeFinder to detect grid calibration from grid probability map
        let gridFinder = GridSizeFinder()

        let result = try gridFinder.findGridCalibration(
            gridProb: segmentation.gridProb,
            width: segmentation.width,
            height: segmentation.height
        )

        return GridCalibration(
            smallSquareWidthPixels: result.smallSquareWidthPixels,
            smallSquareHeightPixels: result.smallSquareHeightPixels,
            angleInDegrees: result.angleInDegrees,
            qualityScore: result.qualityScore,
            gridBounds: result.gridBounds
        )
    }

    // MARK: - Waveform Extraction (On-Device Only)

    /// Extract waveforms from segmentation result using advanced pipeline
    /// Pipeline: Perspective Detection → Cropping → Advanced Signal Extraction → Sectioning Fallback
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Remove this method - server handles waveform extraction
    /// - Returns: Tuple of (waveforms, algorithm used)
    private func extractWaveforms(
        from segmentation: SegmentationResult,
        calibration: GridCalibration
    ) throws -> (waveforms: [[Double]], algorithm: ExtractionAlgorithm) {

        // Step 1: Detect perspective distortion from grid
        let perspectiveDetector = PerspectiveDetector()
        let perspective = try perspectiveDetector.detectPerspective(
            gridProb: segmentation.gridProb,
            width: segmentation.width,
            height: segmentation.height
        )

        print("📐 Perspective: rotation=\(perspective.rotationAngle * 180 / .pi)°, confidence=\(perspective.qualityScore)")

        // Step 2: Use SignalExtractorAdvanced with Hungarian algorithm
        // This handles connected component labeling and optimal line matching
        // Note: For best results, use server-based processing (hybrid or server mode)
        let advancedExtractor = SignalExtractorAdvanced()

        let waveforms = try advancedExtractor.extractLeads(
            signalProb: segmentation.signalProb,
            width: segmentation.width,
            height: segmentation.height,
            calibration: calibration
        )

        // Check quality - count non-empty waveforms
        let nonEmptyCount = waveforms.filter { samples in
            samples.contains { abs($0) > 0.0001 }
        }.count

        print("✅ Advanced extraction completed (\(nonEmptyCount)/\(waveforms.count) waveforms)")
        print("📊 Extraction algorithm: SignalExtractorAdvanced (Hungarian + CC)")

        // Return whatever we got - server mode is recommended for best results
        return (waveforms, .advanced)
    }

    // MARK: - Layout Classification (On-Device Only)

    /// Classify ECG layout using Procrustes alignment
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Remove this method - server handles layout classification
    private func classifyLayout(waveforms: [[Double]]) throws -> ECGLayout {
        // For now, use rule-based approach based on waveform count and samples
        // TODO: Implement full Procrustes alignment when lead positions are available from extraction

        let leadCount = waveforms.count
        let avgSamples = waveforms.isEmpty ? 0 : waveforms.map { $0.count }.reduce(0, +) / waveforms.count

        print("📊 Layout detection: \(leadCount) leads, avg \(avgSamples) samples per lead")

        // Rule-based heuristics:
        // - 3×4 layout: 12 leads, ~1250 samples each (5000 total / 4 columns)
        // - 3×4 + rhythm: 12 leads with one full-width (5000 samples)
        // - 6×2 layout: 12 leads, ~2500 samples each (5000 total / 2 columns)

        if leadCount >= 12 {
            // Check for full-width rhythm strip (Lead II typically has 5000 samples)
            let hasFullWidthLead = waveforms.contains { samples in
                let nonZero = samples.filter { abs($0) > 0.0001 }.count
                return nonZero > 3000  // More than 3000 non-zero samples = full width
            }

            if hasFullWidthLead {
                print("✅ Detected 3×4 layout with 1 rhythm strip")
                return .threeByFour_r1
            }

            // Check average samples per lead
            if avgSamples > 2000 {
                print("✅ Detected 6×2 layout (wider columns)")
                return .sixByTwo_r0
            } else {
                print("✅ Detected 3×4 layout (standard)")
                return .threeByFour_r0
            }
        }

        // Fallback - most common layout
        print("⚠️ Using default layout: 3×4 with rhythm strip")
        return .threeByFour_r1
    }

    // MARK: - Lead Matching (On-Device Only)

    /// Match extracted waveforms to canonical lead positions
    /// Uses cosine similarity for rhythm strip matching
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Remove this method - server handles lead matching
    private func matchLeadsToCanonical(
        waveforms: [[Double]],
        layout: ECGLayout
    ) throws -> [ECGLead] {
        print("🔗 Matching leads to canonical positions...")

        // Determine rhythm lead count from layout
        let rhythmCount = layout.rhythmLeads

        guard rhythmCount > 0 else {
            // No rhythm leads - direct mapping
            print("  No rhythm leads, using direct mapping")
            return directMapLeads(waveforms: waveforms, layout: layout)
        }

        guard waveforms.count >= 12 + rhythmCount else {
            print("⚠️ Insufficient waveforms (\(waveforms.count)), expected \(12 + rhythmCount)")
            return directMapLeads(waveforms: waveforms, layout: layout)
        }

        // Split into grid leads (first 12) and rhythm leads (remaining)
        let gridLeadCount = 12
        let gridWaveforms = Array(waveforms.prefix(gridLeadCount))
        let rhythmWaveforms = Array(waveforms.suffix(rhythmCount))

        print("  Grid leads: \(gridLeadCount), Rhythm leads: \(rhythmCount)")

        // Convert to ECGLead objects
        var canonicalLeads = directMapLeads(waveforms: gridWaveforms, layout: layout)

        // Map rhythm waveforms to R1, R2, R3 lead types
        let rhythmLeadTypes: [LeadType] = [.R1, .R2, .R3]
        let rhythmLeads = rhythmWaveforms.enumerated().map { idx, samples in
            let leadType = idx < rhythmLeadTypes.count ? rhythmLeadTypes[idx] : .R1
            return ECGLead(type: leadType, samples: samples, samplingRate: 500.0)
        }

        // Match rhythm leads using cosine similarity
        let matcher = RhythmStripMatcher()
        let assignments = matcher.matchRhythmLeads(
            rhythmLeads: rhythmLeads,
            canonicalLeads: canonicalLeads
        )

        // Replace matched canonical leads with rhythm strip data
        for (rhythmIdx, canonicalIdx) in assignments {
            guard rhythmIdx < rhythmLeads.count, canonicalIdx < canonicalLeads.count else {
                continue
            }

            let originalType = canonicalLeads[canonicalIdx].type
            print("  Replacing \(originalType.rawValue) with rhythm strip [\(rhythmIdx)]")

            canonicalLeads[canonicalIdx] = ECGLead(
                type: originalType,
                samples: rhythmLeads[rhythmIdx].samples,
                samplingRate: 500.0
            )
        }

        return canonicalLeads
    }

    /// Direct mapping of waveforms to lead types based on layout
    private func directMapLeads(waveforms: [[Double]], layout: ECGLayout) -> [ECGLead] {
        let leadTypes = layout.standardLeadOrder
        return zip(leadTypes, waveforms).map { type, samples in
            ECGLead(type: type, samples: samples, samplingRate: 500.0)
        }
    }

    // MARK: - Memory Optimization Helpers

    /// Downsample probability maps to reduce memory usage on physical devices
    /// Uses bilinear interpolation for smooth downsampling
    private func downsampleProbMaps(
        signalProb: [Float],
        gridProb: [Float],
        width: Int,
        height: Int,
        maxDimension: Int
    ) -> (signalProb: [Float], gridProb: [Float], width: Int, height: Int) {

        // Check if downsampling is needed
        if width <= maxDimension && height <= maxDimension {
            return (signalProb, gridProb, width, height)
        }

        // Calculate new dimensions maintaining aspect ratio
        let scale = Float(maxDimension) / Float(max(width, height))
        let newWidth = Int(Float(width) * scale)
        let newHeight = Int(Float(height) * scale)

        print("📉 Downsampling probability maps: \(width)x\(height) → \(newWidth)x\(newHeight)")

        // Downsample using simple box filter (fast, memory efficient)
        func downsample(_ input: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> [Float] {
            var output = [Float](repeating: 0, count: dstW * dstH)

            let scaleX = Float(srcW) / Float(dstW)
            let scaleY = Float(srcH) / Float(dstH)

            for dstY in 0..<dstH {
                for dstX in 0..<dstW {
                    // Map to source coordinates
                    let srcX = Float(dstX) * scaleX
                    let srcY = Float(dstY) * scaleY

                    // Bilinear interpolation
                    let x0 = Int(srcX)
                    let y0 = Int(srcY)
                    let x1 = min(x0 + 1, srcW - 1)
                    let y1 = min(y0 + 1, srcH - 1)

                    let fx = srcX - Float(x0)
                    let fy = srcY - Float(y0)

                    let v00 = input[y0 * srcW + x0]
                    let v10 = input[y0 * srcW + x1]
                    let v01 = input[y1 * srcW + x0]
                    let v11 = input[y1 * srcW + x1]

                    let v0 = v00 * (1 - fx) + v10 * fx
                    let v1 = v01 * (1 - fx) + v11 * fx

                    output[dstY * dstW + dstX] = v0 * (1 - fy) + v1 * fy
                }
            }

            return output
        }

        let dsSignal = downsample(signalProb, srcW: width, srcH: height, dstW: newWidth, dstH: newHeight)
        let dsGrid = downsample(gridProb, srcW: width, srcH: height, dstW: newWidth, dstH: newHeight)

        return (dsSignal, dsGrid, newWidth, newHeight)
    }
}

// MARK: - Processing Errors

enum ECGProcessingError: LocalizedError {
    case invalidImage
    case modelNotLoaded
    case segmentationFailed
    case gridDetectionFailed
    case waveformExtractionFailed
    case layoutClassificationFailed
    case leadMatchingFailed
    case serverError(String)
    case offline(message: String)

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid ECG image"
        case .modelNotLoaded:
            return "Core ML models not loaded"
        case .segmentationFailed:
            return "ECG segmentation failed"
        case .gridDetectionFailed:
            return "Grid detection failed"
        case .waveformExtractionFailed:
            return "Waveform extraction failed"
        case .layoutClassificationFailed:
            return "Layout classification failed"
        case .leadMatchingFailed:
            return "Lead matching failed"
        case .serverError(let message):
            return "Server error: \(message)"
        case .offline(let message):
            return message
        }
    }
}

// MARK: - Offline Queue (For Server-Based Implementation)

/// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
/// Uncomment and implement this class for offline queue functionality
/*
class OfflineProcessingQueue {
    struct QueueItem: Codable {
        let id: UUID
        let imageData: Data
        let parameters: ProcessingParameters
        let timestamp: Date
    }

    private var queue: [QueueItem] = []
    private let queueKey = "offline_processing_queue"

    func add(_ item: QueueItem) {
        queue.append(item)
        save()

        // Show notification
        NotificationCenter.default.post(
            name: .ecgQueuedForProcessing,
            object: nil,
            userInfo: ["count": queue.count]
        )
    }

    func processQueue() async {
        guard NetworkMonitor.shared.isConnected else { return }

        for item in queue {
            do {
                guard let image = UIImage(data: item.imageData) else { continue }

                let _ = try await ECGAPIClient.shared.digitize(
                    image: image,
                    parameters: item.parameters,
                    progressCallback: { _ in }
                )

                // Remove from queue on success
                queue.removeAll { $0.id == item.id }
                save()
            } catch {
                print("Failed to process queued item: \(error)")
            }
        }
    }

    private func save() {
        if let encoded = try? JSONEncoder().encode(queue) {
            UserDefaults.standard.set(encoded, forKey: queueKey)
        }
    }

    private func load() {
        if let data = UserDefaults.standard.data(forKey: queueKey),
           let decoded = try? JSONDecoder().decode([QueueItem].self, from: data) {
            queue = decoded
        }
    }
}

extension Notification.Name {
    static let ecgQueuedForProcessing = Notification.Name("ecgQueuedForProcessing")
}
*/
