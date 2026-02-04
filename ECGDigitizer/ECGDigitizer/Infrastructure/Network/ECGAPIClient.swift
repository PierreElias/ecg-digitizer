import Foundation
import UIKit

// MARK: - Image Size Utilities

/// Maximum image size for API upload (2MB)
private let maxImageSizeBytes = 2 * 1024 * 1024

/// Result of image compression operation
struct ImageCompressionResult {
    let data: Data
    let originalSize: Int
    let finalSize: Int
    let quality: CGFloat
    let scale: CGFloat
    let wasResized: Bool
    let wasRecompressed: Bool
}

/// Ensures an image is under the specified size limit while maintaining fidelity
/// - Parameters:
///   - image: The UIImage to compress
///   - maxBytes: Maximum size in bytes (default 2MB)
/// - Returns: ImageCompressionResult with compressed data and metadata
private func ensureImageUnderLimit(_ image: UIImage, maxBytes: Int = maxImageSizeBytes) -> ImageCompressionResult {
    var result = ImageCompressionResult(
        data: Data(),
        originalSize: 0,
        finalSize: 0,
        quality: 0.95,
        scale: 1.0,
        wasResized: false,
        wasRecompressed: false
    )

    // Try high quality first
    guard let initialData = image.jpegData(compressionQuality: 0.95) else {
        return result
    }
    result = ImageCompressionResult(
        data: initialData,
        originalSize: initialData.count,
        finalSize: initialData.count,
        quality: 0.95,
        scale: 1.0,
        wasResized: false,
        wasRecompressed: false
    )

    // If already under limit, return as-is
    if initialData.count <= maxBytes {
        return result
    }

    // Try reducing quality progressively
    let qualityLevels: [CGFloat] = [0.85, 0.75, 0.65, 0.55]
    for quality in qualityLevels {
        if let data = image.jpegData(compressionQuality: quality), data.count <= maxBytes {
            return ImageCompressionResult(
                data: data,
                originalSize: initialData.count,
                finalSize: data.count,
                quality: quality,
                scale: 1.0,
                wasResized: false,
                wasRecompressed: true
            )
        }
    }

    // Quality reduction not enough - need to resize
    let scaleFactors: [CGFloat] = [0.9, 0.8, 0.7, 0.6, 0.5]
    let originalSize = image.size

    for scale in scaleFactors {
        let newSize = CGSize(
            width: originalSize.width * scale,
            height: originalSize.height * scale
        )

        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: newSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        if let resized = resizedImage, let data = resized.jpegData(compressionQuality: 0.75), data.count <= maxBytes {
            return ImageCompressionResult(
                data: data,
                originalSize: initialData.count,
                finalSize: data.count,
                quality: 0.75,
                scale: scale,
                wasResized: true,
                wasRecompressed: true
            )
        }
    }

    // Last resort: aggressive compression
    let finalScale: CGFloat = 0.4
    let finalSize = CGSize(
        width: originalSize.width * finalScale,
        height: originalSize.height * finalScale
    )

    UIGraphicsBeginImageContextWithOptions(finalSize, false, 1.0)
    image.draw(in: CGRect(origin: .zero, size: finalSize))
    let finalImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    if let final = finalImage, let data = final.jpegData(compressionQuality: 0.60) {
        return ImageCompressionResult(
            data: data,
            originalSize: initialData.count,
            finalSize: data.count,
            quality: 0.60,
            scale: finalScale,
            wasResized: true,
            wasRecompressed: true
        )
    }

    // Return best effort
    return result
}

// MARK: - ECG API Error Types

/// Error types for ECG API operations
enum ECGAPIError: Error, LocalizedError {
    case invalidURL
    case encodingFailed
    case networkError(Error)
    case serverError(String)
    case decodingFailed(Error)
    case processingFailed(String)
    case serverUnavailable

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid server URL"
        case .encodingFailed:
            return "Failed to encode image"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .serverError(let message):
            return "Server error: \(message)"
        case .decodingFailed(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .serverUnavailable:
            return "Server is not available"
        }
    }
}

/// API client for communicating with the ECG Digitizer web server
class ECGAPIClient: ObservableObject {

    // MARK: - Singleton

    nonisolated(unsafe) static let shared = ECGAPIClient()

    // MARK: - Types

    struct ProcessingResponse: Codable {
        let success: Bool
        let error: String?
        let layout: String?
        let detectedLayout: String?
        let leads: [LeadData]?
        let grid: GridInfo?
        let method: String?
        let layoutMatchingCost: Double?
        let imageSize: ImageSize?
        let timestamp: String?
        let extractionAlgorithm: String?          // "advanced" or "sectioning"
        let extractionAlgorithmDisplay: String?   // Human-readable algorithm name

        enum CodingKeys: String, CodingKey {
            case success, error, layout, leads, grid, method, timestamp
            case detectedLayout = "detected_layout"
            case layoutMatchingCost = "layout_matching_cost"
            case imageSize = "image_size"
            case extractionAlgorithm = "extraction_algorithm"
            case extractionAlgorithmDisplay = "extraction_algorithm_display"
        }

        struct LeadData: Codable {
            let name: String
            let samples: [Double]
            let durationMs: Double
            let sampleCount: Int

            enum CodingKeys: String, CodingKey {
                case name, samples
                case durationMs = "duration_ms"
                case sampleCount = "sample_count"
            }
        }

        struct GridInfo: Codable {
            let detected: Bool
            let confidence: Double
            let mmPerPixelX: Double
            let mmPerPixelY: Double

            enum CodingKeys: String, CodingKey {
                case detected, confidence
                case mmPerPixelX = "mm_per_pixel_x"
                case mmPerPixelY = "mm_per_pixel_y"
            }
        }

        struct ImageSize: Codable {
            let width: Int
            let height: Int
        }

        /// Map server layout names to app ECGLayout enum
        /// Server returns names like "standard_3x4_with_r1", app uses "3x4_r1"
        static func mapServerLayoutToAppLayout(_ serverLayout: String) -> ECGLayout {
            let normalized = serverLayout.lowercased()

            // Map common server layout names to app layout
            if normalized.contains("12x1") || normalized.contains("12_1") {
                if normalized.contains("r1") || normalized.contains("rhythm") {
                    return .twelveByOne_r1
                }
                return .twelveByOne_r0
            } else if normalized.contains("6x2") || normalized.contains("6_2") {
                if normalized.contains("r1") || normalized.contains("rhythm") {
                    return .sixByTwo_r1
                }
                return .sixByTwo_r0
            } else if normalized.contains("3x4") || normalized.contains("3_4") {
                if normalized.contains("r3") || normalized.contains("3_rhythm") {
                    return .threeByFour_r3
                } else if normalized.contains("r2") || normalized.contains("2_rhythm") {
                    return .threeByFour_r2
                } else if normalized.contains("r1") || normalized.contains("rhythm") || normalized.contains("with_r") {
                    return .threeByFour_r1
                }
                return .threeByFour_r0
            } else if normalized.contains("6x1") || normalized.contains("6_1") {
                return .sixByOne_r0
            } else if normalized.contains("3x2") || normalized.contains("3_2") {
                return .threeByTwo_r0
            } else if normalized.contains("3x1") || normalized.contains("3_1") {
                return .threeByOne_r0
            }

            // Default to 3x4 with rhythm strip (most common)
            return .threeByFour_r1
        }

        /// Convert API response to ECGRecording
        func toRecording(originalImage: UIImage, parameters: ProcessingParameters) -> ECGRecording? {
            guard success, let leadsData = leads else { return nil }

            var ecgLeads = leadsData.compactMap { leadData -> ECGLead? in
                guard let leadType = LeadType(rawValue: leadData.name) else { return nil }
                return ECGLead(
                    type: leadType,
                    samples: leadData.samples,
                    samplingRate: Double(leadData.sampleCount) / (leadData.durationMs / 1000.0)
                )
            }

            guard ecgLeads.count >= 12 else { return nil }

            // Determine layout from response - map server format to app format
            let detectedLayoutStr = detectedLayout ?? layout ?? "standard_3x4_with_r1"
            let ecgLayout = Self.mapServerLayoutToAppLayout(detectedLayoutStr)

            // Add rhythm leads if layout requires them
            // Rhythm leads are typically Lead II data shown at full width
            if ecgLayout.rhythmLeads > 0 {
                if let leadII = ecgLeads.first(where: { $0.type == .II }) {
                    // R1 rhythm strip uses Lead II's full data
                    if ecgLayout.rhythmLeads >= 1 {
                        ecgLeads.append(ECGLead(
                            type: .R1,
                            samples: leadII.samples,
                            samplingRate: leadII.samplingRate
                        ))
                    }
                    // R2 and R3 can use other leads (V1, V5) or Lead II
                    if ecgLayout.rhythmLeads >= 2 {
                        let r2Source = ecgLeads.first(where: { $0.type == .V1 }) ?? leadII
                        ecgLeads.append(ECGLead(
                            type: .R2,
                            samples: r2Source.samples,
                            samplingRate: r2Source.samplingRate
                        ))
                    }
                    if ecgLayout.rhythmLeads >= 3 {
                        let r3Source = ecgLeads.first(where: { $0.type == .V5 }) ?? leadII
                        ecgLeads.append(ECGLead(
                            type: .R3,
                            samples: r3Source.samples,
                            samplingRate: r3Source.samplingRate
                        ))
                    }
                }
            }

            // Create grid calibration from response
            let gridCalibration: GridCalibration?
            if let gridInfo = grid, gridInfo.detected {
                // Convert mm per pixel to pixels per mm (small square width)
                let pixelsPerMmX = 1.0 / gridInfo.mmPerPixelX
                let pixelsPerMmY = 1.0 / gridInfo.mmPerPixelY

                gridCalibration = GridCalibration(
                    smallSquareWidthPixels: pixelsPerMmX,
                    smallSquareHeightPixels: pixelsPerMmY,
                    angleInDegrees: 0.0,  // Server handles dewarping
                    qualityScore: gridInfo.confidence,
                    gridBounds: CGRect(
                        x: 0, y: 0,
                        width: Double(imageSize?.width ?? 1000),
                        height: Double(imageSize?.height ?? 1000)
                    )
                )
            } else {
                gridCalibration = nil
            }

            // Determine extraction algorithm from server response
            let algorithm: ExtractionAlgorithm
            if let algoStr = extractionAlgorithm {
                algorithm = ExtractionAlgorithm(rawValue: algoStr) ?? .server
            } else {
                // Default to server if not specified
                algorithm = .server
            }

            let metadata = ECGMetadata(
                notes: "Processed with \(method ?? "open_ecg_digitizer")",
                extractionAlgorithm: algorithm
            )

            // Compress original image with 2MB limit
            let compressionResult = ensureImageUnderLimit(originalImage, maxBytes: maxImageSizeBytes)
            let compressedImageData = compressionResult.data.isEmpty ? nil : compressionResult.data

            return ECGRecording(
                originalImageData: compressedImageData,
                parameters: parameters,
                layout: ecgLayout,
                leads: ecgLeads,
                gridCalibration: gridCalibration,
                metadata: metadata,
                validationStatus: .valid
            )
        }
    }

    // MARK: - Properties

    @MainActor @Published var baseURL: URL = URL(string: "http://localhost:8080")!
    @MainActor @Published var isConnected: Bool = false

    private let session: URLSession

    // MARK: - Initialization

    private init() {
        // Default to localhost for development; can be changed in settings
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120 // ECG processing can take time
        config.timeoutIntervalForResource = 300
        self.session = URLSession(configuration: config)
    }

    // MARK: - Public Methods

    /// Check if the server is reachable
    func checkHealth() async -> Bool {
        let url = await MainActor.run { baseURL }
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        request.timeoutInterval = 5

        do {
            let (_, response) = try await session.data(for: request)
            if let httpResponse = response as? HTTPURLResponse {
                let connected = httpResponse.statusCode == 200
                await MainActor.run { isConnected = connected }
                return connected
            }
            await MainActor.run { isConnected = false }
            return false
        } catch {
            await MainActor.run { isConnected = false }
            return false
        }
    }

    /// Process an ECG image through the server API
    func digitize(
        image: UIImage,
        parameters: ProcessingParameters,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> ProcessingResponse {
        // Update progress
        progressCallback(.validatingImage)

        // Convert image to base64 with size limit enforcement
        let compressionResult = ensureImageUnderLimit(image, maxBytes: maxImageSizeBytes)
        guard !compressionResult.data.isEmpty else {
            throw ECGAPIError.encodingFailed
        }

        if compressionResult.wasResized || compressionResult.wasRecompressed {
            print("📊 digitize: Image optimized - \(compressionResult.originalSize / 1024)KB → \(compressionResult.finalSize / 1024)KB (quality=\(compressionResult.quality), scale=\(compressionResult.scale))")
        }

        let base64Image = "data:image/jpeg;base64," + compressionResult.data.base64EncodedString()

        progressCallback(.detectingGrid(progress: 0.2))

        // Build request body
        let requestBody: [String: Any] = [
            "image": base64Image,
            "signal_based_boundaries": true
        ]

        // Create request
        let url = await MainActor.run { baseURL.appendingPathComponent("api/process") }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        progressCallback(.detectingGrid(progress: 0.4))

        // Send request
        let (data, response) = try await session.data(for: request)

        progressCallback(.classifyingLayout)

        // Check HTTP status
        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
            throw ECGAPIError.serverError("HTTP \(httpResponse.statusCode)")
        }

        progressCallback(.extractingWaveforms(currentLead: 6, totalLeads: 12))

        // Decode response
        let decoder = JSONDecoder()
        do {
            let result = try decoder.decode(ProcessingResponse.self, from: data)

            progressCallback(.validatingResults)

            if !result.success, let error = result.error {
                throw ECGAPIError.processingFailed(error)
            }

            return result
        } catch let decodingError as DecodingError {
            throw ECGAPIError.decodingFailed(decodingError)
        }
    }

    /// Update the server URL
    func setServerURL(_ urlString: String) {
        if let url = URL(string: urlString) {
            Task { @MainActor in
                baseURL = url
                _ = await checkHealth()
            }
        }
    }

    /// Generate diagnostic report for an ECG image
    func generateDiagnostic(
        image: UIImage,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> DiagnosticResponse {
        progressCallback(.validatingImage)

        // Convert image to base64 with size limit enforcement
        let compressionResult = ensureImageUnderLimit(image, maxBytes: maxImageSizeBytes)
        guard !compressionResult.data.isEmpty else {
            throw ECGAPIError.encodingFailed
        }

        if compressionResult.wasResized || compressionResult.wasRecompressed {
            print("📊 generateDiagnostic: Image optimized - \(compressionResult.originalSize / 1024)KB → \(compressionResult.finalSize / 1024)KB")
        }

        let base64Image = "data:image/jpeg;base64," + compressionResult.data.base64EncodedString()

        // Build request body
        let requestBody: [String: Any] = [
            "image": base64Image,
            "signal_based_boundaries": true
        ]

        // Create request
        let url = await MainActor.run { baseURL.appendingPathComponent("api/diagnostic") }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        progressCallback(.detectingGrid(progress: 0.5))

        // Send request
        let (data, response) = try await session.data(for: request)

        // Check HTTP status
        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
            throw ECGAPIError.serverError("HTTP \(httpResponse.statusCode)")
        }

        progressCallback(.validatingResults)

        // Decode response
        let decoder = JSONDecoder()
        do {
            let result = try decoder.decode(DiagnosticResponse.self, from: data)

            if !result.success {
                throw ECGAPIError.processingFailed(result.error ?? "Unknown error")
            }

            return result
        } catch let decodingError as DecodingError {
            throw ECGAPIError.decodingFailed(decodingError)
        }
    }

    /// Diagnostic response containing HTML report
    struct DiagnosticResponse: Codable {
        let success: Bool
        let html: String?
        let error: String?
    }

    // MARK: - Aptible API Integration

    /// Upload waveform data to Aptible API and receive a diagnostic report
    /// This sends the digitized ECG waveform data to an external diagnostic service
    func uploadWaveformForDiagnosis(
        recording: ECGRecording,
        aptibleURL: URL,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> DiagnosticReportResponse {
        progressCallback(.validatingImage)

        // Build request body with waveform data
        let requestBody: [String: Any] = [
            // Add patient metadata
            "patient_id": recording.metadata.patientId ?? "",
            "notes": recording.metadata.notes ?? "",
            "timestamp": recording.timestamp.ISO8601Format(),
            "recording_id": recording.id.uuidString,

            // Add processing parameters
            "paper_speed": recording.parameters.paperSpeed.rawValue,
            "voltage_gain": recording.parameters.voltageGain.rawValue,

            // Add layout information
            "layout": recording.layout.rawValue,

            // Add waveform data for all 12 leads
            "leads": recording.leads.map { lead in
                [
                    "name": lead.type.rawValue,
                    "samples": lead.samples,
                    "sampling_rate": lead.samplingRate,
                    "duration_ms": Double(lead.samples.count) / lead.samplingRate * 1000.0
                ]
            },

            // Add original image if needed for visual analysis
            "original_image": recording.originalImageData.map { imageData in
                "data:image/jpeg;base64," + imageData.base64EncodedString()
            } ?? ""
        ]

        progressCallback(.detectingGrid(progress: 0.3))

        // Create request
        var request = URLRequest(url: aptibleURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        progressCallback(.classifyingLayout)

        // Send request
        let (data, response) = try await session.data(for: request)

        // Check HTTP status
        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
            throw ECGAPIError.serverError("HTTP \(httpResponse.statusCode)")
        }

        progressCallback(.validatingResults)

        // Decode response
        let decoder = JSONDecoder()
        do {
            let result = try decoder.decode(DiagnosticReportResponse.self, from: data)

            if !result.success {
                throw ECGAPIError.processingFailed(result.error ?? "Unknown error")
            }

            return result
        } catch let decodingError as DecodingError {
            throw ECGAPIError.decodingFailed(decodingError)
        }
    }

    // MARK: - Hybrid Processing (ONNX on device + Python post-processing)

    /// Send probability maps to server for post-processing
    /// This is the hybrid approach: ONNX segmentation runs on device (fast),
    /// then server handles perspective/cropping/dewarping/extraction (accurate)
    ///
    /// - Parameters:
    ///   - signalProb: Signal probability map from ONNX segmentation
    ///   - gridProb: Grid probability map from ONNX segmentation
    ///   - textProb: Text probability map from ONNX segmentation
    ///   - width: Width of probability maps (typically 512)
    ///   - height: Height of probability maps (typically 512)
    ///   - originalImage: Original full-resolution image for better extraction
    ///   - progressCallback: Progress update callback
    func postprocess(
        signalProb: [Float],
        gridProb: [Float],
        textProb: [Float],
        width: Int,
        height: Int,
        originalImage: UIImage?,
        progressCallback: @escaping (ProcessingState) -> Void
    ) async throws -> ProcessingResponse {
        progressCallback(.detectingGrid(progress: 0.3))

        // Compress probability maps using gzip to reduce transfer size
        // Each map is ~4MB uncompressed, ~200-400KB compressed
        let signalData = Data(bytes: signalProb, count: signalProb.count * MemoryLayout<Float>.size)
        let gridData = Data(bytes: gridProb, count: gridProb.count * MemoryLayout<Float>.size)
        let textData = Data(bytes: textProb, count: textProb.count * MemoryLayout<Float>.size)

        // Base64 encode the binary data
        let signalBase64 = signalData.base64EncodedString()
        let gridBase64 = gridData.base64EncodedString()
        let textBase64 = textData.base64EncodedString()

        // Build request body
        var requestBody: [String: Any] = [
            "signal_prob": signalBase64,
            "grid_prob": gridBase64,
            "text_prob": textBase64,
            "width": width,
            "height": height,
            "format": "float32_base64"  // Tell server the encoding format
        ]

        // Add original image if provided (for much better extraction results)
        // Compress with size limit enforcement
        if let image = originalImage {
            let compressionResult = ensureImageUnderLimit(image, maxBytes: maxImageSizeBytes)
            if !compressionResult.data.isEmpty {
                requestBody["original_image"] = compressionResult.data.base64EncodedString()
                if compressionResult.wasResized || compressionResult.wasRecompressed {
                    print("📊 postprocess: Original image optimized - \(compressionResult.originalSize / 1024)KB → \(compressionResult.finalSize / 1024)KB")
                }
            }
        }

        // Create request
        let url = await MainActor.run { baseURL.appendingPathComponent("api/postprocess") }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        progressCallback(.classifyingLayout)

        // Send request
        let (data, response) = try await session.data(for: request)

        // Check HTTP status
        if let httpResponse = response as? HTTPURLResponse {
            if httpResponse.statusCode == 404 {
                // Endpoint doesn't exist yet - fall back to full server processing
                throw ECGAPIError.serverError("Postprocess endpoint not available (404). Use full server processing.")
            }
            if httpResponse.statusCode != 200 {
                throw ECGAPIError.serverError("HTTP \(httpResponse.statusCode)")
            }
        }

        progressCallback(.extractingWaveforms(currentLead: 6, totalLeads: 12))

        // Decode response
        let decoder = JSONDecoder()
        do {
            let result = try decoder.decode(ProcessingResponse.self, from: data)

            progressCallback(.validatingResults)

            if !result.success, let error = result.error {
                throw ECGAPIError.processingFailed(error)
            }

            return result
        } catch let decodingError as DecodingError {
            throw ECGAPIError.decodingFailed(decodingError)
        }
    }

    /// Response from Aptible diagnostic API
    struct DiagnosticReportResponse: Codable {
        let success: Bool
        let error: String?

        // Diagnostic findings
        let findings: [DiagnosticFinding]?
        let summary: String?
        let confidence: Double?

        // Measurements
        let measurements: ECGMeasurements?

        // Full report
        let reportHtml: String?
        let reportPdf: String? // Base64 encoded PDF

        enum CodingKeys: String, CodingKey {
            case success, error, findings, summary, confidence, measurements
            case reportHtml = "report_html"
            case reportPdf = "report_pdf"
        }

        struct DiagnosticFinding: Codable {
            let category: String
            let severity: String // "normal", "abnormal", "critical"
            let description: String
            let confidence: Double
        }

        struct ECGMeasurements: Codable {
            let heartRate: Int?
            let prInterval: Int? // milliseconds
            let qrsDuration: Int? // milliseconds
            let qtInterval: Int? // milliseconds
            let qtcInterval: Int? // corrected QT, milliseconds
            let pWave: WaveMeasurement?
            let qrsComplex: WaveMeasurement?
            let tWave: WaveMeasurement?

            enum CodingKeys: String, CodingKey {
                case heartRate = "heart_rate"
                case prInterval = "pr_interval"
                case qrsDuration = "qrs_duration"
                case qtInterval = "qt_interval"
                case qtcInterval = "qtc_interval"
                case pWave = "p_wave"
                case qrsComplex = "qrs_complex"
                case tWave = "t_wave"
            }

            struct WaveMeasurement: Codable {
                let amplitude: Double? // millivolts
                let duration: Int? // milliseconds
                let axis: Int? // degrees
            }
        }
    }
}

