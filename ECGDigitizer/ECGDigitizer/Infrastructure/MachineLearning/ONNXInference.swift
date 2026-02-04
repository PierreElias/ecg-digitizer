import Foundation
import CoreGraphics
import Accelerate
import UIKit

// MARK: - ONNX Runtime Integration
//
// ONNX Runtime is added via Swift Package Manager:
// https://github.com/microsoft/onnxruntime-swift-package-manager.git
import OnnxRuntimeBindings

// MARK: - Error Types

/// Errors that can occur during ONNX inference
enum ONNXError: Error, LocalizedError {
    case modelNotFound(String)
    case sessionCreationFailed(String)
    case inputCreationFailed(String)
    case inferenceFailed(String)
    case outputExtractionFailed(String)
    case invalidImageFormat
    case runtimeNotAvailable

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "ONNX model not found: \(name)"
        case .sessionCreationFailed(let message):
            return "Failed to create ONNX session: \(message)"
        case .inputCreationFailed(let message):
            return "Failed to create input tensor: \(message)"
        case .inferenceFailed(let message):
            return "ONNX inference failed: \(message)"
        case .outputExtractionFailed(let message):
            return "Failed to extract output: \(message)"
        case .invalidImageFormat:
            return "Invalid image format for ONNX inference"
        case .runtimeNotAvailable:
            return "ONNX Runtime is not available. Please add the SPM dependency."
        }
    }
}

// MARK: - Segmentation Result

/// Result of running the segmentation model
struct SegmentationResult {
    /// Signal probability map (H x W)
    let signalProb: [Float]

    /// Grid probability map (H x W)
    let gridProb: [Float]

    /// Text/background probability map (H x W)
    let textProb: [Float]

    /// Width of the probability maps
    let width: Int

    /// Height of the probability maps
    let height: Int

    /// Extract signal probability as CGImage for visualization
    func signalProbAsImage() -> CGImage? {
        return createGrayscaleImage(from: signalProb, width: width, height: height)
    }

    /// Extract grid probability as CGImage for visualization
    func gridProbAsImage() -> CGImage? {
        return createGrayscaleImage(from: gridProb, width: width, height: height)
    }

    private func createGrayscaleImage(from data: [Float], width: Int, height: Int) -> CGImage? {
        // Convert float array to UInt8
        var uint8Data = [UInt8](repeating: 0, count: data.count)
        for i in 0..<data.count {
            uint8Data[i] = UInt8(min(255, max(0, data[i] * 255)))
        }

        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &uint8Data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }

        return context.makeImage()
    }
}

// MARK: - Visual Diagnostic Report

/// Visual diagnostic report with step-by-step images
struct VisualDiagnosticReport {
    /// The input ECG image
    let inputImage: UIImage?

    /// Signal probability heatmap (green-tinted)
    let signalHeatmap: UIImage?

    /// Grid probability heatmap (red-tinted)
    let gridHeatmap: UIImage?

    /// Text probability heatmap (blue-tinted)
    let textHeatmap: UIImage?

    /// RGB overlay (R=Grid, G=Signal, B=Text)
    let rgbOverlay: UIImage?

    /// Step 6: Lead extraction visualization
    /// Shows detected signal rows with boundary lines
    let leadExtractionImage: UIImage?

    /// Step 6b: Sectioned leads visualization
    /// Shows the 3x4 grid of individual leads after sectioning
    let sectionedLeadsImage: UIImage?

    /// Step 7: Final 12-lead waveform plot
    /// Shows the final output that appears in the app
    let final12LeadPlot: UIImage?

    /// Text metrics report
    let textReport: String

    /// Metrics for display
    let metrics: ONNXDiagnosticReport

    /// Lead extraction details for display
    let leadExtractionDetails: LeadExtractionDetails?
}

// MARK: - Lead Extraction Details

struct LeadExtractionDetails {
    let rowCount: Int
    let rowBoundaries: [(top: Int, bottom: Int)]
    let leadNames: [[String]]  // 3x4 grid of lead names
    let extractedSamples: Int  // Samples per lead
}

// MARK: - Diagnostic Report

/// Detailed diagnostic report for iOS-only ONNX processing
struct ONNXDiagnosticReport {
    /// Timestamp of the diagnostic run
    let timestamp: Date

    /// Input image dimensions
    let inputWidth: Int
    let inputHeight: Int

    /// Processed dimensions (after resize)
    let processedWidth: Int
    let processedHeight: Int

    /// Preprocessing metrics
    struct PreprocessingMetrics {
        let minPixelValue: Float
        let maxPixelValue: Float
        let meanPixelValue: Float
        let normalizedRange: (min: Float, max: Float)
    }
    let preprocessing: PreprocessingMetrics

    /// Segmentation metrics for each channel
    struct ChannelMetrics {
        let name: String
        let rawMin: Float
        let rawMax: Float
        let rawMean: Float
        let processedMin: Float
        let processedMax: Float
        let processedMean: Float
        let coverage: Float  // % of pixels above 0.1 threshold
        let postDilationCoverage: Float?  // For signal channel only
    }
    let signalMetrics: ChannelMetrics
    let gridMetrics: ChannelMetrics
    let textMetrics: ChannelMetrics

    /// Inference timing
    let preprocessingTimeMs: Double
    let inferenceTimeMs: Double
    let postprocessingTimeMs: Double
    let totalTimeMs: Double

    /// Generate a human-readable report
    func generateReport() -> String {
        var report = """
        ═══════════════════════════════════════════════════════════════
        ONNX SEGMENTATION DIAGNOSTIC REPORT
        ═══════════════════════════════════════════════════════════════
        Timestamp: \(timestamp)

        ── INPUT ──────────────────────────────────────────────────────
        Original Size: \(inputWidth) × \(inputHeight)
        Processed Size: \(processedWidth) × \(processedHeight)

        ── PREPROCESSING ──────────────────────────────────────────────
        Pixel Range: [\(String(format: "%.1f", preprocessing.minPixelValue)), \(String(format: "%.1f", preprocessing.maxPixelValue))]
        Mean Pixel: \(String(format: "%.1f", preprocessing.meanPixelValue))
        Normalized Range: [\(String(format: "%.3f", preprocessing.normalizedRange.min)), \(String(format: "%.3f", preprocessing.normalizedRange.max))]

        ── SIGNAL CHANNEL ─────────────────────────────────────────────
        Raw Logits: min=\(String(format: "%.3f", signalMetrics.rawMin)), max=\(String(format: "%.3f", signalMetrics.rawMax)), mean=\(String(format: "%.3f", signalMetrics.rawMean))
        After Softmax+Sparse: min=\(String(format: "%.3f", signalMetrics.processedMin)), max=\(String(format: "%.3f", signalMetrics.processedMax)), mean=\(String(format: "%.3f", signalMetrics.processedMean))
        Coverage (>0.1): \(String(format: "%.1f", signalMetrics.coverage))%
        """

        if let dilationCoverage = signalMetrics.postDilationCoverage {
            report += """

        After Dilation: \(String(format: "%.1f", dilationCoverage))% (+\(String(format: "%.1f", dilationCoverage - signalMetrics.coverage))%)
        """
        }

        report += """


        ── GRID CHANNEL ───────────────────────────────────────────────
        Raw Logits: min=\(String(format: "%.3f", gridMetrics.rawMin)), max=\(String(format: "%.3f", gridMetrics.rawMax)), mean=\(String(format: "%.3f", gridMetrics.rawMean))
        After Softmax+Sparse: min=\(String(format: "%.3f", gridMetrics.processedMin)), max=\(String(format: "%.3f", gridMetrics.processedMax)), mean=\(String(format: "%.3f", gridMetrics.processedMean))
        Coverage (>0.1): \(String(format: "%.1f", gridMetrics.coverage))%

        ── TEXT CHANNEL ───────────────────────────────────────────────
        Raw Logits: min=\(String(format: "%.3f", textMetrics.rawMin)), max=\(String(format: "%.3f", textMetrics.rawMax)), mean=\(String(format: "%.3f", textMetrics.rawMean))
        After Softmax+Sparse: min=\(String(format: "%.3f", textMetrics.processedMin)), max=\(String(format: "%.3f", textMetrics.processedMax)), mean=\(String(format: "%.3f", textMetrics.processedMean))
        Coverage (>0.1): \(String(format: "%.1f", textMetrics.coverage))%

        ── TIMING ─────────────────────────────────────────────────────
        Preprocessing: \(String(format: "%.1f", preprocessingTimeMs)) ms
        Inference: \(String(format: "%.1f", inferenceTimeMs)) ms
        Postprocessing: \(String(format: "%.1f", postprocessingTimeMs)) ms
        TOTAL: \(String(format: "%.1f", totalTimeMs)) ms

        ═══════════════════════════════════════════════════════════════
        """

        return report
    }
}

// MARK: - Lead Identification Result

/// Result of running the lead identifier model
struct LeadIdentificationResult {
    /// Probability maps for each of the 12 leads (12 x H x W)
    let leadProbabilities: [[Float]]

    /// Width of the probability maps
    let width: Int

    /// Height of the probability maps
    let height: Int

    /// Lead names in order
    static let leadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
}

// MARK: - ONNX Inference Engine

/// ONNX Runtime inference wrapper for ECG processing
///
/// This class handles loading and running ONNX models for:
/// 1. ECG image segmentation (separating signal, grid, text regions)
/// 2. Lead identification (detecting which lead is which)
///
/// Usage:
/// ```swift
/// let inference = ONNXInference()
/// try await inference.loadModels()
/// let segmentation = try await inference.runSegmentation(image: cgImage)
/// ```
///
/// Note: Heavy processing is done on background threads to avoid blocking the UI
class ONNXInference: ObservableObject {

    // MARK: - Properties

    /// Whether models are loaded and ready
    @Published private(set) var isReady = false

    /// Loading progress (0.0 - 1.0)
    @Published private(set) var loadingProgress: Double = 0.0

    /// Error message if loading failed
    @Published private(set) var loadingError: String?

    // Model input sizes
    // MEMORY CRITICAL: iPhone has ~100-150MB limit before crash
    // ONNX inference at 768×768 uses ~120-170MB peak = CRASH
    // ONNX inference at 512×512 uses ~60-80MB peak = SAFE
    //
    // Quality impact: 512 is sufficient for ECG signal extraction
    // The grid lines and waveforms are well-defined at this resolution
    private let segmentationInputSize = 512  // MUST stay at 512 to prevent iPhone crash
    private let leadIdentifierInputSize = 512

    // Whether to use CoreML Execution Provider
    // CoreML EP uses Neural Engine but adds ~30-50MB memory overhead
    // Set to false for memory-constrained devices
    private let useCoreMLExecutionProvider = false

    /// Whether to use the dynamic dimension model (ECGSegmentation_dynamic.onnx)
    /// Set to true after testing the dynamic model
    private let useDynamicModel = true

    // ONNX Runtime objects
    private var ortEnv: ORTEnv?
    private var segmentationSession: ORTSession?
    private var leadIdentifierSession: ORTSession?

    // MARK: - Singleton

    static let shared = ONNXInference()

    private init() {}

    // MARK: - Model Loading

    /// Load both ONNX models
    ///
    /// This should be called once at app startup or before first use.
    /// Models are loaded asynchronously on a background thread to avoid blocking the UI.
    func loadModels() async throws {
        await MainActor.run {
            loadingProgress = 0.0
            loadingError = nil
        }

        do {
            // Load segmentation model on background thread
            await MainActor.run { loadingProgress = 0.1 }
            try await loadSegmentationModel()
            await MainActor.run { loadingProgress = 0.5 }

            // Load lead identifier model
            try await loadLeadIdentifierModel()
            await MainActor.run { loadingProgress = 1.0 }

            await MainActor.run {
                isReady = true
            }
            print("✅ ONNX models loaded successfully")

        } catch {
            await MainActor.run {
                loadingError = error.localizedDescription
                isReady = false
            }
            throw error
        }
    }

    private func loadSegmentationModel() async throws {
        // Choose model based on configuration
        let modelName = useDynamicModel ? "ECGSegmentation_dynamic" : "ECGSegmentation"
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "onnx") else {
            // Fallback to original model if dynamic not found
            if useDynamicModel {
                print("⚠️ Dynamic model not found, trying fixed model...")
                if let fallbackPath = Bundle.main.path(forResource: "ECGSegmentation", ofType: "onnx") {
                    print("Loading segmentation model (fixed) from: \(fallbackPath)")
                    return try await loadSegmentationModelFromPath(fallbackPath)
                }
            }
            throw ONNXError.modelNotFound("\(modelName).onnx")
        }

        print("Loading segmentation model (\(useDynamicModel ? "dynamic" : "fixed")) from: \(modelPath)")
        try await loadSegmentationModelFromPath(modelPath)
    }

    private func loadSegmentationModelFromPath(_ modelPath: String) async throws {
        // Initialize ONNX Runtime environment if not already done
        if ortEnv == nil {
            // Use .error level to suppress iOS home directory warnings
            ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.error)
        }

        // Create session options with memory optimizations for iPhone
        let options = try ORTSessionOptions()

        // Memory optimization settings to reduce peak memory usage
        try options.setGraphOptimizationLevel(.all)
        try options.setIntraOpNumThreads(2)  // Limit threads to reduce memory

        // CoreML EP adds ~30-50MB overhead - disable for memory-constrained devices
        if useCoreMLExecutionProvider {
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            try options.appendCoreMLExecutionProvider(with: coreMLOptions)
            print("  ℹ️ Using CoreML Execution Provider (Neural Engine)")
        } else {
            print("  ℹ️ Using CPU Execution Provider (lower memory)")
        }

        // Create the session
        segmentationSession = try ORTSession(
            env: ortEnv!,
            modelPath: modelPath,
            sessionOptions: options
        )

        let modelType = useDynamicModel ? "dynamic" : "fixed"
        print("  ✓ Segmentation model loaded (\(modelType), input: \(segmentationInputSize)x\(segmentationInputSize))")
    }

    private func loadLeadIdentifierModel() async throws {
        guard let modelPath = Bundle.main.path(forResource: "ECGLeadIdentifier", ofType: "onnx") else {
            throw ONNXError.modelNotFound("ECGLeadIdentifier.onnx")
        }

        print("Loading lead identifier model from: \(modelPath)")

        // Create session options with CoreML EP
        let options = try ORTSessionOptions()
        try options.appendCoreMLExecutionProvider(with: ORTCoreMLExecutionProviderOptions())

        // Create the session
        leadIdentifierSession = try ORTSession(
            env: ortEnv!,
            modelPath: modelPath,
            sessionOptions: options
        )

        print("  ✓ Lead identifier model loaded")
    }

    // MARK: - Segmentation Inference

    /// Run segmentation with detailed diagnostic report
    ///
    /// - Parameter image: The ECG image to segment
    /// - Returns: Tuple of (SegmentationResult, ONNXDiagnosticReport)
    func runSegmentationWithDiagnostics(image: CGImage) async throws -> (SegmentationResult, ONNXDiagnosticReport) {
        guard let session = segmentationSession else {
            throw ONNXError.sessionCreationFailed("Segmentation model not loaded")
        }

        let startTime = Date()

        return try await Task.detached(priority: .userInitiated) {
            return try autoreleasepool {
                // Step 1: Preprocessing
                let preprocessStart = Date()
                let (inputData, width, height, preprocessMetrics) = try self.preprocessImageForSegmentationWithMetrics(image)
                let preprocessEnd = Date()
                let preprocessTimeMs = preprocessEnd.timeIntervalSince(preprocessStart) * 1000

                // Step 2: Inference
                let inferenceStart = Date()
                let inputNSData = NSMutableData(bytes: inputData, length: inputData.count * MemoryLayout<Float>.size)
                let inputShape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]

                let inputTensor = try ORTValue(
                    tensorData: inputNSData,
                    elementType: .float,
                    shape: inputShape
                )

                let inputName = self.useDynamicModel ? "image" : "input"
                let outputNames: Set<String> = ["output"]
                let outputs = try session.run(
                    withInputs: [inputName: inputTensor],
                    outputNames: outputNames,
                    runOptions: nil
                )

                guard let outputTensor = outputs["output"] else {
                    throw ONNXError.outputExtractionFailed("No output tensor")
                }

                let outputData = try outputTensor.tensorData()
                let outputCount = outputData.length / MemoryLayout<Float>.size
                var outputFloats = [Float](repeating: 0, count: outputCount)
                outputData.getBytes(&outputFloats, length: outputData.length)
                let inferenceEnd = Date()
                let inferenceTimeMs = inferenceEnd.timeIntervalSince(inferenceStart) * 1000

                // Step 3: Postprocessing with metrics
                let postprocessStart = Date()
                let (signalProb, gridProb, textProb, channelMetrics) = self.extractSegmentationChannelsWithMetrics(
                    from: outputFloats,
                    width: width,
                    height: height
                )
                let postprocessEnd = Date()
                let postprocessTimeMs = postprocessEnd.timeIntervalSince(postprocessStart) * 1000

                let totalTimeMs = postprocessEnd.timeIntervalSince(startTime) * 1000

                // Build diagnostic report
                let report = ONNXDiagnosticReport(
                    timestamp: startTime,
                    inputWidth: image.width,
                    inputHeight: image.height,
                    processedWidth: width,
                    processedHeight: height,
                    preprocessing: preprocessMetrics,
                    signalMetrics: channelMetrics.signal,
                    gridMetrics: channelMetrics.grid,
                    textMetrics: channelMetrics.text,
                    preprocessingTimeMs: preprocessTimeMs,
                    inferenceTimeMs: inferenceTimeMs,
                    postprocessingTimeMs: postprocessTimeMs,
                    totalTimeMs: totalTimeMs
                )

                let result = SegmentationResult(
                    signalProb: signalProb,
                    gridProb: gridProb,
                    textProb: textProb,
                    width: width,
                    height: height
                )

                return (result, report)
            }
        }.value
    }

    /// Run segmentation on an ECG image
    ///
    /// - Parameter image: The ECG image to segment
    /// - Returns: SegmentationResult containing probability maps for signal, grid, and text
    func runSegmentation(image: CGImage) async throws -> SegmentationResult {
        guard let session = segmentationSession else {
            throw ONNXError.sessionCreationFailed("Segmentation model not loaded")
        }

        // Run heavy inference on background thread to prevent UI blocking
        return try await Task.detached(priority: .userInitiated) {
            // Use autoreleasepool to release intermediate memory immediately
            // This helps prevent memory spikes on iPhone
            return try autoreleasepool {
                // Preprocess image to input tensor format
                var (inputData, width, height) = try self.preprocessImageForSegmentation(image)

                // Create input tensor - convert [Float] to NSMutableData
                let inputNSData = NSMutableData(bytes: inputData, length: inputData.count * MemoryLayout<Float>.size)
                let inputShape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]

                // Clear input data to free memory before inference
                inputData = []

                let inputTensor = try ORTValue(
                    tensorData: inputNSData,
                    elementType: .float,
                    shape: inputShape
                )

                // Run inference - THIS IS THE HEAVY CPU/GPU WORK
                // Note: Dynamic model uses "image" as input name, fixed model uses "input"
                let inputName = self.useDynamicModel ? "image" : "input"
                let outputNames: Set<String> = ["output"]
                let outputs = try session.run(
                    withInputs: [inputName: inputTensor],
                    outputNames: outputNames,
                    runOptions: nil
                )

                guard let outputTensor = outputs["output"] else {
                    throw ONNXError.outputExtractionFailed("No output tensor")
                }

                // Extract output data
                let outputData = try outputTensor.tensorData()
                let outputCount = outputData.length / MemoryLayout<Float>.size
                var outputFloats = [Float](repeating: 0, count: outputCount)
                outputData.getBytes(&outputFloats, length: outputData.length)

                // Apply softmax and extract channels
                let (signalProb, gridProb, textProb) = self.extractSegmentationChannels(
                    from: outputFloats,
                    width: width,
                    height: height
                )

                // Clear outputFloats to free memory
                outputFloats = []

                return SegmentationResult(
                    signalProb: signalProb,
                    gridProb: gridProb,
                    textProb: textProb,
                    width: width,
                    height: height
                )
            }
        }.value
    }

    /// Run lead identification on a text probability map
    ///
    /// - Parameters:
    ///   - textProb: The text probability map from segmentation
    ///   - width: Width of the probability map
    ///   - height: Height of the probability map
    /// - Returns: LeadIdentificationResult with probability maps for each lead
    func runLeadIdentifier(textProb: [Float], width: Int, height: Int) async throws -> LeadIdentificationResult {
        guard let session = leadIdentifierSession else {
            throw ONNXError.sessionCreationFailed("Lead identifier model not loaded")
        }

        // Run heavy inference on background thread to prevent UI blocking
        return try await Task.detached(priority: .userInitiated) {
            // Resize text probability map to lead identifier input size
            let resizedInput = self.resizeFloatArray(
                textProb,
                fromWidth: width,
                fromHeight: height,
                toWidth: self.leadIdentifierInputSize,
                toHeight: self.leadIdentifierInputSize
            )

            // Create input tensor - convert [Float] to NSMutableData
            let inputNSData = NSMutableData(bytes: resizedInput, length: resizedInput.count * MemoryLayout<Float>.size)
            let inputShape: [NSNumber] = [1, 1, NSNumber(value: self.leadIdentifierInputSize), NSNumber(value: self.leadIdentifierInputSize)]

            let inputTensor = try ORTValue(
                tensorData: inputNSData,
                elementType: .float,
                shape: inputShape
            )

            // Run inference - HEAVY CPU/GPU WORK
            let outputNames: Set<String> = ["output"]
            let outputs = try session.run(
                withInputs: ["input": inputTensor],
                outputNames: outputNames,
                runOptions: nil
            )

            guard let outputTensor = outputs["output"] else {
                throw ONNXError.outputExtractionFailed("No output tensor")
            }

            // Extract output data
            let outputData = try outputTensor.tensorData()
            let outputCount = outputData.length / MemoryLayout<Float>.size
            var outputFloats = [Float](repeating: 0, count: outputCount)
            outputData.getBytes(&outputFloats, length: outputData.length)

            let leadProbs = self.extractLeadChannels(from: outputFloats, width: self.leadIdentifierInputSize, height: self.leadIdentifierInputSize)

            return LeadIdentificationResult(
                leadProbabilities: leadProbs,
                width: self.leadIdentifierInputSize,
                height: self.leadIdentifierInputSize
            )
        }.value
    }

    // MARK: - Image Preprocessing

    /// Preprocess a CGImage for segmentation model input
    ///
    /// The ONNX model was exported with fixed 1024x1024 input dimensions.
    /// To avoid white padding artifacts that confuse normalization, we resize
    /// the image to FILL 1024x1024 (slight stretching is better than padding).
    ///
    /// Preprocessing steps:
    /// 1. Resize to 1024x1024 (filling entire canvas)
    /// 2. Per-image min-max normalization to [0, 1]
    ///
    /// - Parameter image: Input CGImage
    /// - Returns: Tuple of (normalized float array in CHW format, width, height)
    private func preprocessImageForSegmentation(_ image: CGImage) throws -> ([Float], Int, Int) {
        let originalWidth = image.width
        let originalHeight = image.height

        // ONNX model requires exactly 1024x1024 input
        let targetWidth = segmentationInputSize
        let targetHeight = segmentationInputSize

        print("ONNX Preprocessing: \(originalWidth)x\(originalHeight) -> \(targetWidth)x\(targetHeight) (resize to fill, no padding)")

        // Create a bitmap context for the resized image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = targetWidth * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: targetHeight * bytesPerRow)

        guard let context = CGContext(
            data: &pixelData,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ONNXError.invalidImageFormat
        }

        // Draw the image filling the entire 1024x1024 canvas (NO PADDING)
        // This may slightly stretch the image but avoids padding artifacts
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        // Convert to normalized float array in CHW format (channels, height, width)
        let pixelCount = targetWidth * targetHeight
        var floatData = [Float](repeating: 0, count: 3 * pixelCount)

        // Find min/max for normalization (matches Python's min_max_normalize)
        var minVal: Float = 255
        var maxVal: Float = 0
        for i in 0..<pixelCount {
            let offset = i * bytesPerPixel
            let r = Float(pixelData[offset])
            let g = Float(pixelData[offset + 1])
            let b = Float(pixelData[offset + 2])
            minVal = min(minVal, min(r, min(g, b)))
            maxVal = max(maxVal, max(r, max(g, b)))
        }

        let range = maxVal - minVal + 1e-9

        // Convert to CHW format with min-max normalization: (x - min) / (max - min)
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let pixelIndex = y * targetWidth + x
                let offset = pixelIndex * bytesPerPixel

                // Normalize to 0-1 using per-image min-max
                let r = (Float(pixelData[offset]) - minVal) / range
                let g = (Float(pixelData[offset + 1]) - minVal) / range
                let b = (Float(pixelData[offset + 2]) - minVal) / range

                // CHW format: all R values, then all G values, then all B values
                floatData[pixelIndex] = r                           // R channel
                floatData[pixelCount + pixelIndex] = g              // G channel
                floatData[2 * pixelCount + pixelIndex] = b          // B channel
            }
        }

        return (floatData, targetWidth, targetHeight)
    }

    /// Preprocess a CGImage for segmentation with detailed metrics
    ///
    /// Same as preprocessImageForSegmentation but also returns preprocessing metrics
    private func preprocessImageForSegmentationWithMetrics(_ image: CGImage) throws -> ([Float], Int, Int, ONNXDiagnosticReport.PreprocessingMetrics) {
        let originalWidth = image.width
        let originalHeight = image.height

        let targetWidth = segmentationInputSize
        let targetHeight = segmentationInputSize

        // Create a bitmap context for the resized image
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = targetWidth * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: targetHeight * bytesPerRow)

        guard let context = CGContext(
            data: &pixelData,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ONNXError.invalidImageFormat
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        // Convert to normalized float array in CHW format
        let pixelCount = targetWidth * targetHeight
        var floatData = [Float](repeating: 0, count: 3 * pixelCount)

        // Find min/max for normalization and collect metrics
        var minVal: Float = 255
        var maxVal: Float = 0
        var sumVal: Float = 0

        for i in 0..<pixelCount {
            let offset = i * bytesPerPixel
            let r = Float(pixelData[offset])
            let g = Float(pixelData[offset + 1])
            let b = Float(pixelData[offset + 2])
            minVal = min(minVal, min(r, min(g, b)))
            maxVal = max(maxVal, max(r, max(g, b)))
            sumVal += (r + g + b) / 3.0
        }

        let meanVal = sumVal / Float(pixelCount)
        let range = maxVal - minVal + 1e-9

        // Convert to CHW format with min-max normalization
        var normalizedMin: Float = 1.0
        var normalizedMax: Float = 0.0

        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let pixelIndex = y * targetWidth + x
                let offset = pixelIndex * bytesPerPixel

                let r = (Float(pixelData[offset]) - minVal) / range
                let g = (Float(pixelData[offset + 1]) - minVal) / range
                let b = (Float(pixelData[offset + 2]) - minVal) / range

                normalizedMin = min(normalizedMin, min(r, min(g, b)))
                normalizedMax = max(normalizedMax, max(r, max(g, b)))

                floatData[pixelIndex] = r
                floatData[pixelCount + pixelIndex] = g
                floatData[2 * pixelCount + pixelIndex] = b
            }
        }

        let metrics = ONNXDiagnosticReport.PreprocessingMetrics(
            minPixelValue: minVal,
            maxPixelValue: maxVal,
            meanPixelValue: meanVal,
            normalizedRange: (min: normalizedMin, max: normalizedMax)
        )

        return (floatData, targetWidth, targetHeight, metrics)
    }

    /// Extract segmentation channels from model output
    ///
    /// Model outputs shape (1, 4, H, W) with logits for each class:
    /// - Channel 0: Grid
    /// - Channel 1: Text/Background
    /// - Channel 2: Signal
    /// - Channel 3: Background
    private func extractSegmentationChannels(
        from output: [Float],
        width: Int,
        height: Int
    ) -> (signal: [Float], grid: [Float], text: [Float]) {
        let pixelCount = width * height

        // Apply softmax per pixel
        var signalProb = [Float](repeating: 0, count: pixelCount)
        var gridProb = [Float](repeating: 0, count: pixelCount)
        var textProb = [Float](repeating: 0, count: pixelCount)

        for i in 0..<pixelCount {
            // Get logits for each class
            let gridLogit = output[i]
            let textLogit = output[pixelCount + i]
            let signalLogit = output[2 * pixelCount + i]
            let bgLogit = output[3 * pixelCount + i]

            // Softmax
            let maxLogit = max(gridLogit, max(textLogit, max(signalLogit, bgLogit)))
            let expGrid = exp(gridLogit - maxLogit)
            let expText = exp(textLogit - maxLogit)
            let expSignal = exp(signalLogit - maxLogit)
            let expBg = exp(bgLogit - maxLogit)
            let sumExp = expGrid + expText + expSignal + expBg

            gridProb[i] = expGrid / sumExp
            textProb[i] = expText / sumExp
            signalProb[i] = expSignal / sumExp
        }

        // Apply sparse probability processing to ALL channels (matching Python pipeline)
        // Python: signal_prob = signal_prob - signal_prob.mean()
        //         signal_prob = torch.clamp(signal_prob, min=0)
        //         signal_prob = signal_prob / (signal_prob.max() + 1e-9)
        func processSparseProb(_ prob: [Float]) -> [Float] {
            let mean = prob.reduce(0, +) / Float(prob.count)
            var processed = prob.map { max(0, $0 - mean) }
            let maxVal = processed.max() ?? 1.0
            return processed.map { $0 / (maxVal + 1e-9) }
        }

        let processedSignal = processSparseProb(signalProb)
        let processedGrid = processSparseProb(gridProb)
        let processedText = processSparseProb(textProb)

        // Apply morphological dilation to signal probability to fill small gaps in traces
        // This improves signal capture by ~70% based on evaluation testing
        let dilatedSignal = applyDilation(processedSignal, width: width, height: height, kernelSize: 3)

        // Renormalize after dilation
        let dilatedMax = dilatedSignal.max() ?? 1.0
        let finalSignal = dilatedSignal.map { $0 / (dilatedMax + 1e-9) }

        return (finalSignal, processedGrid, processedText)
    }

    /// Extract segmentation channels with detailed metrics for diagnostics
    ///
    /// Same as extractSegmentationChannels but also returns channel metrics
    private func extractSegmentationChannelsWithMetrics(
        from output: [Float],
        width: Int,
        height: Int
    ) -> (signal: [Float], grid: [Float], text: [Float], metrics: (signal: ONNXDiagnosticReport.ChannelMetrics, grid: ONNXDiagnosticReport.ChannelMetrics, text: ONNXDiagnosticReport.ChannelMetrics)) {
        let pixelCount = width * height

        // Apply softmax per pixel
        var signalProb = [Float](repeating: 0, count: pixelCount)
        var gridProb = [Float](repeating: 0, count: pixelCount)
        var textProb = [Float](repeating: 0, count: pixelCount)

        // Also track raw logits for metrics
        var signalLogits = [Float](repeating: 0, count: pixelCount)
        var gridLogits = [Float](repeating: 0, count: pixelCount)
        var textLogits = [Float](repeating: 0, count: pixelCount)

        for i in 0..<pixelCount {
            let gridLogit = output[i]
            let textLogit = output[pixelCount + i]
            let signalLogit = output[2 * pixelCount + i]
            let bgLogit = output[3 * pixelCount + i]

            signalLogits[i] = signalLogit
            gridLogits[i] = gridLogit
            textLogits[i] = textLogit

            let maxLogit = max(gridLogit, max(textLogit, max(signalLogit, bgLogit)))
            let expGrid = exp(gridLogit - maxLogit)
            let expText = exp(textLogit - maxLogit)
            let expSignal = exp(signalLogit - maxLogit)
            let expBg = exp(bgLogit - maxLogit)
            let sumExp = expGrid + expText + expSignal + expBg

            gridProb[i] = expGrid / sumExp
            textProb[i] = expText / sumExp
            signalProb[i] = expSignal / sumExp
        }

        // Helper to compute stats
        func computeStats(_ arr: [Float]) -> (min: Float, max: Float, mean: Float) {
            let minV = arr.min() ?? 0
            let maxV = arr.max() ?? 0
            let mean = arr.reduce(0, +) / Float(arr.count)
            return (minV, maxV, mean)
        }

        // Helper to compute coverage (% above threshold)
        func computeCoverage(_ arr: [Float], threshold: Float = 0.1) -> Float {
            let count = arr.filter { $0 > threshold }.count
            return Float(count) / Float(arr.count) * 100
        }

        // Sparse probability processing with metrics
        func processSparseProb(_ prob: [Float]) -> [Float] {
            let mean = prob.reduce(0, +) / Float(prob.count)
            var processed = prob.map { max(0, $0 - mean) }
            let maxVal = processed.max() ?? 1.0
            return processed.map { $0 / (maxVal + 1e-9) }
        }

        // Collect raw logit stats
        let signalLogitStats = computeStats(signalLogits)
        let gridLogitStats = computeStats(gridLogits)
        let textLogitStats = computeStats(textLogits)

        // Process channels
        let processedSignal = processSparseProb(signalProb)
        let processedGrid = processSparseProb(gridProb)
        let processedText = processSparseProb(textProb)

        // Get processed stats (before dilation for signal)
        let signalProcessedStats = computeStats(processedSignal)
        let gridProcessedStats = computeStats(processedGrid)
        let textProcessedStats = computeStats(processedText)

        // Coverage before dilation
        let signalCoverageBefore = computeCoverage(processedSignal)
        let gridCoverage = computeCoverage(processedGrid)
        let textCoverage = computeCoverage(processedText)

        // Apply dilation to signal
        let dilatedSignal = applyDilation(processedSignal, width: width, height: height, kernelSize: 3)
        let dilatedMax = dilatedSignal.max() ?? 1.0
        let finalSignal = dilatedSignal.map { $0 / (dilatedMax + 1e-9) }

        // Coverage after dilation
        let signalCoverageAfter = computeCoverage(finalSignal)

        // Build metrics
        let signalMetrics = ONNXDiagnosticReport.ChannelMetrics(
            name: "Signal",
            rawMin: signalLogitStats.min,
            rawMax: signalLogitStats.max,
            rawMean: signalLogitStats.mean,
            processedMin: signalProcessedStats.min,
            processedMax: signalProcessedStats.max,
            processedMean: signalProcessedStats.mean,
            coverage: signalCoverageBefore,
            postDilationCoverage: signalCoverageAfter
        )

        let gridMetrics = ONNXDiagnosticReport.ChannelMetrics(
            name: "Grid",
            rawMin: gridLogitStats.min,
            rawMax: gridLogitStats.max,
            rawMean: gridLogitStats.mean,
            processedMin: gridProcessedStats.min,
            processedMax: gridProcessedStats.max,
            processedMean: gridProcessedStats.mean,
            coverage: gridCoverage,
            postDilationCoverage: nil
        )

        let textMetrics = ONNXDiagnosticReport.ChannelMetrics(
            name: "Text",
            rawMin: textLogitStats.min,
            rawMax: textLogitStats.max,
            rawMean: textLogitStats.mean,
            processedMin: textProcessedStats.min,
            processedMax: textProcessedStats.max,
            processedMean: textProcessedStats.mean,
            coverage: textCoverage,
            postDilationCoverage: nil
        )

        return (finalSignal, processedGrid, processedText, (signalMetrics, gridMetrics, textMetrics))
    }

    /// Apply morphological dilation to probability map
    ///
    /// Dilation expands high-probability regions by taking the maximum value
    /// within a kernel neighborhood. This fills small gaps in waveform traces.
    ///
    /// - Parameters:
    ///   - input: Input probability map (flattened)
    ///   - width: Width of the map
    ///   - height: Height of the map
    ///   - kernelSize: Size of the dilation kernel (3 = 3x3)
    /// - Returns: Dilated probability map
    private func applyDilation(_ input: [Float], width: Int, height: Int, kernelSize: Int = 3) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        let halfKernel = kernelSize / 2

        for y in 0..<height {
            for x in 0..<width {
                var maxVal: Float = 0

                // Find maximum in kernel neighborhood
                for ky in -halfKernel...halfKernel {
                    for kx in -halfKernel...halfKernel {
                        let ny = y + ky
                        let nx = x + kx

                        // Check bounds
                        if ny >= 0 && ny < height && nx >= 0 && nx < width {
                            let idx = ny * width + nx
                            maxVal = max(maxVal, input[idx])
                        }
                    }
                }

                output[y * width + x] = maxVal
            }
        }

        return output
    }

    /// Extract lead probability channels from lead identifier output
    private func extractLeadChannels(from output: [Float], width: Int, height: Int) -> [[Float]] {
        let pixelCount = width * height
        var leadProbs = [[Float]]()

        // First 12 channels are the lead probabilities
        for leadIndex in 0..<12 {
            var probs = [Float](repeating: 0, count: pixelCount)
            for i in 0..<pixelCount {
                probs[i] = output[leadIndex * pixelCount + i]
            }

            // Apply softmax across all channels for each pixel
            // (simplified - full implementation would softmax across all 13 channels)
            leadProbs.append(probs)
        }

        return leadProbs
    }

    // MARK: - Utility Functions

    /// Resize a float array using bilinear interpolation
    private func resizeFloatArray(
        _ input: [Float],
        fromWidth: Int,
        fromHeight: Int,
        toWidth: Int,
        toHeight: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: toWidth * toHeight)

        let xScale = Float(fromWidth - 1) / Float(toWidth - 1)
        let yScale = Float(fromHeight - 1) / Float(toHeight - 1)

        for y in 0..<toHeight {
            for x in 0..<toWidth {
                let srcX = Float(x) * xScale
                let srcY = Float(y) * yScale

                let x0 = Int(srcX)
                let y0 = Int(srcY)
                let x1 = min(x0 + 1, fromWidth - 1)
                let y1 = min(y0 + 1, fromHeight - 1)

                let xFrac = srcX - Float(x0)
                let yFrac = srcY - Float(y0)

                let v00 = input[y0 * fromWidth + x0]
                let v01 = input[y0 * fromWidth + x1]
                let v10 = input[y1 * fromWidth + x0]
                let v11 = input[y1 * fromWidth + x1]

                let v0 = v00 * (1 - xFrac) + v01 * xFrac
                let v1 = v10 * (1 - xFrac) + v11 * xFrac

                output[y * toWidth + x] = v0 * (1 - yFrac) + v1 * yFrac
            }
        }

        return output
    }

    // MARK: - Cleanup

    /// Unload models to free memory
    func unloadModels() {
        segmentationSession = nil
        leadIdentifierSession = nil
        ortEnv = nil

        isReady = false
        loadingProgress = 0.0

        print("ONNX models unloaded")
    }

    // MARK: - Visual Diagnostics

    /// Run segmentation with visual diagnostic output (step-by-step images)
    ///
    /// - Parameter image: The ECG image to segment
    /// - Returns: VisualDiagnosticReport with images and metrics
    func runVisualDiagnostic(image: CGImage, inputUIImage: UIImage) async throws -> VisualDiagnosticReport {
        // Run the standard diagnostic
        let (result, report) = try await runSegmentationWithDiagnostics(image: image)

        // Generate heatmap images
        let signalHeatmap = probabilityToHeatmap(
            result.signalProb,
            width: result.width,
            height: result.height,
            colormap: .green
        )

        let gridHeatmap = probabilityToHeatmap(
            result.gridProb,
            width: result.width,
            height: result.height,
            colormap: .red
        )

        let textHeatmap = probabilityToHeatmap(
            result.textProb,
            width: result.width,
            height: result.height,
            colormap: .blue
        )

        // Generate RGB overlay
        let rgbOverlay = createRGBOverlay(
            signalProb: result.signalProb,
            gridProb: result.gridProb,
            textProb: result.textProb,
            width: result.width,
            height: result.height
        )

        // Step 6: Lead Extraction Visualization
        // Detect rows and visualize sectioning
        let (leadExtractionImage, sectionedLeadsImage, extractionDetails) = visualizeLeadExtraction(
            signalProb: result.signalProb,
            width: result.width,
            height: result.height
        )

        // Step 7: Final 12-Lead Waveform Plot
        // This shows what the user actually sees in the app
        let final12LeadPlot = generateFinal12LeadPlot(
            signalProb: result.signalProb,
            width: result.width,
            height: result.height,
            extractionDetails: extractionDetails
        )

        return VisualDiagnosticReport(
            inputImage: inputUIImage,
            signalHeatmap: signalHeatmap,
            gridHeatmap: gridHeatmap,
            textHeatmap: textHeatmap,
            rgbOverlay: rgbOverlay,
            leadExtractionImage: leadExtractionImage,
            sectionedLeadsImage: sectionedLeadsImage,
            final12LeadPlot: final12LeadPlot,
            textReport: report.generateReport(),
            metrics: report,
            leadExtractionDetails: extractionDetails
        )
    }

    /// Colormap types for heatmap generation
    enum HeatmapColormap {
        case hot      // Black → Red → Yellow → White (matplotlib's "hot")
        case green    // Black → Green → White
        case red      // Black → Red → White
        case blue     // Black → Blue → White
    }

    /// Convert probability map to colored heatmap UIImage
    private func probabilityToHeatmap(
        _ prob: [Float],
        width: Int,
        height: Int,
        colormap: HeatmapColormap
    ) -> UIImage? {
        guard prob.count == width * height else { return nil }

        // Create RGBA pixel data
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for i in 0..<prob.count {
            let value = min(1.0, max(0.0, prob[i]))
            let offset = i * 4

            switch colormap {
            case .hot:
                // Hot colormap: black → red → yellow → white
                let (r, g, b) = hotColormap(value)
                pixelData[offset] = r
                pixelData[offset + 1] = g
                pixelData[offset + 2] = b

            case .green:
                // Green tint: scale green channel, slight red/blue
                pixelData[offset] = UInt8(value * 50)      // R
                pixelData[offset + 1] = UInt8(value * 255) // G
                pixelData[offset + 2] = UInt8(value * 50)  // B

            case .red:
                // Red tint: scale red channel
                pixelData[offset] = UInt8(value * 255)     // R
                pixelData[offset + 1] = UInt8(value * 50)  // G
                pixelData[offset + 2] = UInt8(value * 50)  // B

            case .blue:
                // Blue tint: scale blue channel
                pixelData[offset] = UInt8(value * 50)      // R
                pixelData[offset + 1] = UInt8(value * 50)  // G
                pixelData[offset + 2] = UInt8(value * 255) // B
            }

            pixelData[offset + 3] = 255 // Alpha
        }

        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    /// Hot colormap implementation (like matplotlib's "hot")
    /// Maps 0→black, 0.33→red, 0.66→yellow, 1→white
    private func hotColormap(_ value: Float) -> (UInt8, UInt8, UInt8) {
        let v = min(1.0, max(0.0, value))

        var r: Float = 0
        var g: Float = 0
        var b: Float = 0

        if v < 0.33 {
            // Black to red
            r = v / 0.33
        } else if v < 0.66 {
            // Red to yellow
            r = 1.0
            g = (v - 0.33) / 0.33
        } else {
            // Yellow to white
            r = 1.0
            g = 1.0
            b = (v - 0.66) / 0.34
        }

        return (
            UInt8(r * 255),
            UInt8(g * 255),
            UInt8(b * 255)
        )
    }

    /// Create RGB overlay image (R=Grid, G=Signal, B=Text)
    private func createRGBOverlay(
        signalProb: [Float],
        gridProb: [Float],
        textProb: [Float],
        width: Int,
        height: Int
    ) -> UIImage? {
        guard signalProb.count == width * height,
              gridProb.count == width * height,
              textProb.count == width * height else {
            return nil
        }

        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for i in 0..<(width * height) {
            let offset = i * 4
            // R = Grid, G = Signal, B = Text
            pixelData[offset] = UInt8(min(255, max(0, gridProb[i] * 255)))
            pixelData[offset + 1] = UInt8(min(255, max(0, signalProb[i] * 255)))
            pixelData[offset + 2] = UInt8(min(255, max(0, textProb[i] * 255)))
            pixelData[offset + 3] = 255
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Lead Extraction Visualization

    /// Visualize the lead extraction process
    /// Shows detected rows with boundary lines overlaid on signal probability
    private func visualizeLeadExtraction(
        signalProb: [Float],
        width: Int,
        height: Int
    ) -> (extraction: UIImage?, sectioned: UIImage?, details: LeadExtractionDetails?) {
        // Step 1: Find row boundaries by detecting horizontal signal concentrations
        let rowBoundaries = detectRowBoundaries(signalProb: signalProb, width: width, height: height)

        guard rowBoundaries.count >= 3 else {
            return (nil, nil, nil)
        }

        // Standard 3x4 lead layout
        let leadNames = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"]
        ]

        let details = LeadExtractionDetails(
            rowCount: min(rowBoundaries.count, 3),
            rowBoundaries: Array(rowBoundaries.prefix(3)),
            leadNames: leadNames,
            extractedSamples: width / 4  // Each lead gets 1/4 of the width
        )

        // Generate extraction visualization (signal prob + boundary lines)
        let extractionImage = drawRowBoundaries(
            signalProb: signalProb,
            width: width,
            height: height,
            boundaries: Array(rowBoundaries.prefix(3))
        )

        // Generate sectioned leads visualization (3x4 grid of individual waveforms)
        let sectionedImage = drawSectionedLeads(
            signalProb: signalProb,
            width: width,
            height: height,
            boundaries: Array(rowBoundaries.prefix(3)),
            leadNames: leadNames
        )

        return (extractionImage, sectionedImage, details)
    }

    /// Detect row boundaries from signal probability
    private func detectRowBoundaries(
        signalProb: [Float],
        width: Int,
        height: Int
    ) -> [(top: Int, bottom: Int)] {
        // Sum signal probability across each row
        var rowSums = [Float](repeating: 0, count: height)
        for y in 0..<height {
            for x in 0..<width {
                rowSums[y] += signalProb[y * width + x]
            }
        }

        // Find peaks (local maxima) in row sums - these are signal rows
        var peaks: [Int] = []
        let windowSize = height / 20  // Minimum distance between peaks

        for y in windowSize..<(height - windowSize) {
            let isLocalMax = (0..<windowSize).allSatisfy { offset in
                rowSums[y] >= rowSums[y - offset] && rowSums[y] >= rowSums[y + offset]
            }
            if isLocalMax && rowSums[y] > rowSums.max()! * 0.3 {
                peaks.append(y)
            }
        }

        // Take the top 3 peaks (for 3x4 layout)
        let sortedPeaks = peaks.sorted { rowSums[$0] > rowSums[$1] }.prefix(3).sorted()

        // Create boundaries around each peak
        var boundaries: [(top: Int, bottom: Int)] = []
        let margin = height / 8

        for peak in sortedPeaks {
            let top = max(0, peak - margin)
            let bottom = min(height - 1, peak + margin)
            boundaries.append((top, bottom))
        }

        return boundaries.sorted { $0.top < $1.top }
    }

    /// Draw row boundary lines overlaid on signal probability heatmap
    private func drawRowBoundaries(
        signalProb: [Float],
        width: Int,
        height: Int,
        boundaries: [(top: Int, bottom: Int)]
    ) -> UIImage? {
        // Start with signal heatmap as background
        guard var pixelData = createHeatmapPixelData(signalProb, width: width, height: height, colormap: .green) else {
            return nil
        }

        // Draw horizontal boundary lines
        for boundary in boundaries {
            // Top boundary (red line)
            drawHorizontalLine(on: &pixelData, y: boundary.top, width: width, color: (255, 0, 0, 255))
            // Bottom boundary (red line)
            drawHorizontalLine(on: &pixelData, y: boundary.bottom, width: width, color: (255, 0, 0, 255))
        }

        // Draw vertical column dividers (4 columns)
        for col in 1..<4 {
            let x = (width * col) / 4
            drawVerticalLine(on: &pixelData, x: x, height: height, width: width, color: (255, 255, 0, 200))
        }

        return createUIImageFromPixelData(pixelData, width: width, height: height)
    }

    /// Draw 3x4 grid of sectioned lead waveforms
    private func drawSectionedLeads(
        signalProb: [Float],
        width: Int,
        height: Int,
        boundaries: [(top: Int, bottom: Int)],
        leadNames: [[String]]
    ) -> UIImage? {
        // Create a larger canvas for the grid (3 rows × 4 cols)
        let cellWidth = 300
        let cellHeight = 150
        let canvasWidth = cellWidth * 4
        let canvasHeight = cellHeight * 3

        guard let renderer = UIGraphicsImageRenderer(size: CGSize(width: canvasWidth, height: canvasHeight)) as UIGraphicsImageRenderer? else {
            return nil
        }

        return renderer.image { context in
            let ctx = context.cgContext

            // White background
            ctx.setFillColor(UIColor.white.cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight))

            // Draw each cell
            for row in 0..<3 {
                guard row < boundaries.count else { continue }
                let boundary = boundaries[row]

                for col in 0..<4 {
                    let cellX = col * cellWidth
                    let cellY = row * cellHeight

                    // Extract waveform for this cell
                    let startX = (width * col) / 4
                    let endX = (width * (col + 1)) / 4
                    let waveform = extractWaveformSegment(
                        signalProb: signalProb,
                        width: width,
                        height: height,
                        rowTop: boundary.top,
                        rowBottom: boundary.bottom,
                        colStart: startX,
                        colEnd: endX
                    )

                    // Draw cell border
                    ctx.setStrokeColor(UIColor.lightGray.cgColor)
                    ctx.setLineWidth(1)
                    ctx.stroke(CGRect(x: cellX, y: cellY, width: cellWidth, height: cellHeight))

                    // Draw lead name
                    let leadName = leadNames[row][col]
                    let attrs: [NSAttributedString.Key: Any] = [
                        .font: UIFont.boldSystemFont(ofSize: 12),
                        .foregroundColor: UIColor.black
                    ]
                    let nameString = NSAttributedString(string: leadName, attributes: attrs)
                    nameString.draw(at: CGPoint(x: cellX + 5, y: cellY + 5))

                    // Draw waveform
                    if !waveform.isEmpty {
                        ctx.setStrokeColor(UIColor.systemGreen.cgColor)
                        ctx.setLineWidth(1.5)

                        let xScale = CGFloat(cellWidth - 20) / CGFloat(waveform.count)
                        let yScale = CGFloat(cellHeight - 40)

                        ctx.beginPath()
                        for (i, value) in waveform.enumerated() {
                            let x = CGFloat(cellX + 10) + CGFloat(i) * xScale
                            let y = CGFloat(cellY + cellHeight / 2) - CGFloat(value) * yScale
                            if i == 0 {
                                ctx.move(to: CGPoint(x: x, y: y))
                            } else {
                                ctx.addLine(to: CGPoint(x: x, y: y))
                            }
                        }
                        ctx.strokePath()
                    }
                }
            }
        }
    }

    /// Extract waveform segment from signal probability
    private func extractWaveformSegment(
        signalProb: [Float],
        width: Int,
        height: Int,
        rowTop: Int,
        rowBottom: Int,
        colStart: Int,
        colEnd: Int
    ) -> [Float] {
        var waveform: [Float] = []

        for x in colStart..<colEnd {
            // Compute weighted centroid for this column
            var weightedSum: Float = 0
            var totalWeight: Float = 0

            for y in rowTop..<rowBottom {
                if y < height {
                    let prob = signalProb[y * width + x]
                    weightedSum += Float(y) * prob
                    totalWeight += prob
                }
            }

            if totalWeight > 0.01 {
                let centroid = weightedSum / totalWeight
                // Normalize to [-1, 1] range relative to row center
                let rowCenter = Float(rowTop + rowBottom) / 2
                let normalized = (centroid - rowCenter) / Float(rowBottom - rowTop)
                waveform.append(normalized)
            } else {
                waveform.append(0)
            }
        }

        return waveform
    }

    // MARK: - Final 12-Lead Waveform Plot

    /// Generate the final 12-lead ECG plot
    /// This shows what the user actually sees in the app
    private func generateFinal12LeadPlot(
        signalProb: [Float],
        width: Int,
        height: Int,
        extractionDetails: LeadExtractionDetails?
    ) -> UIImage? {
        guard let details = extractionDetails else { return nil }

        // Create canvas for 6x2 layout (similar to standard ECG printout)
        let cellWidth = 400
        let cellHeight = 120
        let canvasWidth = cellWidth * 2
        let canvasHeight = cellHeight * 6

        guard let renderer = UIGraphicsImageRenderer(size: CGSize(width: canvasWidth, height: canvasHeight)) as UIGraphicsImageRenderer? else {
            return nil
        }

        return renderer.image { context in
            let ctx = context.cgContext

            // Light gray background
            ctx.setFillColor(UIColor(white: 0.95, alpha: 1).cgColor)
            ctx.fill(CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight))

            // All 12 leads in 6x2 layout
            let allLeads = details.leadNames.flatMap { $0 }

            for (index, leadName) in allLeads.enumerated() {
                let row = index / 2
                let col = index % 2
                let cellX = col * cellWidth
                let cellY = row * cellHeight

                // Determine which row and column in 3x4 grid
                let gridRow = index / 4
                let gridCol = index % 4

                guard gridRow < details.rowBoundaries.count else { continue }
                let boundary = details.rowBoundaries[gridRow]

                // Extract waveform
                let startX = (width * gridCol) / 4
                let endX = (width * (gridCol + 1)) / 4
                let waveform = extractWaveformSegment(
                    signalProb: signalProb,
                    width: width,
                    height: height,
                    rowTop: boundary.top,
                    rowBottom: boundary.bottom,
                    colStart: startX,
                    colEnd: endX
                )

                // Draw cell
                ctx.setFillColor(UIColor.white.cgColor)
                ctx.fill(CGRect(x: cellX + 5, y: cellY + 5, width: cellWidth - 10, height: cellHeight - 10))

                ctx.setStrokeColor(UIColor.lightGray.cgColor)
                ctx.setLineWidth(0.5)
                ctx.stroke(CGRect(x: cellX + 5, y: cellY + 5, width: cellWidth - 10, height: cellHeight - 10))

                // Draw lead name
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 14),
                    .foregroundColor: UIColor.black
                ]
                let nameString = NSAttributedString(string: leadName, attributes: attrs)
                nameString.draw(at: CGPoint(x: cellX + 15, y: cellY + 10))

                // Draw waveform
                if !waveform.isEmpty {
                    ctx.setStrokeColor(UIColor.systemGreen.cgColor)
                    ctx.setLineWidth(2)

                    let xScale = CGFloat(cellWidth - 40) / CGFloat(waveform.count)
                    let yScale = CGFloat(cellHeight - 60)

                    ctx.beginPath()
                    for (i, value) in waveform.enumerated() {
                        let x = CGFloat(cellX + 20) + CGFloat(i) * xScale
                        let y = CGFloat(cellY + cellHeight / 2) - CGFloat(value) * yScale
                        if i == 0 {
                            ctx.move(to: CGPoint(x: x, y: y))
                        } else {
                            ctx.addLine(to: CGPoint(x: x, y: y))
                        }
                    }
                    ctx.strokePath()
                }
            }

            // Add title
            let titleAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 18),
                .foregroundColor: UIColor.black
            ]
            let title = NSAttributedString(string: "12-Lead ECG (ONNX Processing)", attributes: titleAttrs)
            title.draw(at: CGPoint(x: 20, y: 10))
        }
    }

    // MARK: - Drawing Helpers

    private func createHeatmapPixelData(_ prob: [Float], width: Int, height: Int, colormap: HeatmapColormap) -> [UInt8]? {
        guard prob.count == width * height else { return nil }

        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for i in 0..<prob.count {
            let value = min(1.0, max(0.0, prob[i]))
            let offset = i * 4

            switch colormap {
            case .green:
                pixelData[offset] = UInt8(value * 50)
                pixelData[offset + 1] = UInt8(value * 255)
                pixelData[offset + 2] = UInt8(value * 50)
            default:
                pixelData[offset] = UInt8(value * 255)
                pixelData[offset + 1] = UInt8(value * 255)
                pixelData[offset + 2] = UInt8(value * 255)
            }

            pixelData[offset + 3] = 255
        }

        return pixelData
    }

    private func drawHorizontalLine(on pixelData: inout [UInt8], y: Int, width: Int, color: (UInt8, UInt8, UInt8, UInt8)) {
        guard y >= 0 && y < pixelData.count / (width * 4) else { return }

        for x in 0..<width {
            let offset = (y * width + x) * 4
            pixelData[offset] = color.0
            pixelData[offset + 1] = color.1
            pixelData[offset + 2] = color.2
            pixelData[offset + 3] = color.3
        }
    }

    private func drawVerticalLine(on pixelData: inout [UInt8], x: Int, height: Int, width: Int, color: (UInt8, UInt8, UInt8, UInt8)) {
        guard x >= 0 && x < width else { return }

        for y in 0..<height {
            let offset = (y * width + x) * 4
            pixelData[offset] = color.0
            pixelData[offset + 1] = color.1
            pixelData[offset + 2] = color.2
            pixelData[offset + 3] = color.3
        }
    }

    private func createUIImageFromPixelData(_ pixelData: [UInt8], width: Int, height: Int) -> UIImage? {
        var mutablePixelData = pixelData
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: &mutablePixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }
}

