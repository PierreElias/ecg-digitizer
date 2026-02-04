import Foundation
import UIKit
import CoreGraphics

/// ECG Validator implementing all 14 validation checks from PMcardio spec
final class ECGValidator {

    // MARK: - Configuration Constants

    struct Config {
        // Image validation
        static let minimumImageDimension: Int = 128
        static let supportedImageFormats: Set<String> = ["jpeg", "jpg", "png"]
        static let uniformityThreshold: Double = 0.02  // Variance threshold for blank image detection

        // Grid validation
        static let maximumGridAngle: Double = 5.0  // degrees
        static let minimumGridSquares: Int = 100
        static let maximumGridPoints: Int = 50000
        static let gridSpacingVarianceThreshold: Double = 0.05  // 5% variance allowed

        // Lead validation
        static let minimumLeadDurationMs: Double = 1500.0
        static let expectedBaseLeads: Int = 12
        static let minimumSignalAmplitude: Double = 0.001  // 1 µV threshold for flatline
    }

    // MARK: - Image Validation

    /// Validates image dimensions (Error 1)
    func validateImageSize(_ image: UIImage) throws {
        let width = Int(image.size.width * image.scale)
        let height = Int(image.size.height * image.scale)

        if width < Config.minimumImageDimension || height < Config.minimumImageDimension {
            throw ValidationError.inputImageTooSmall(width: width, height: height)
        }
    }

    /// Validates image is not uniform/blank (Error 2)
    func validateImageContent(_ image: UIImage) throws {
        guard let cgImage = image.cgImage else {
            throw ValidationError.uniformInputImage
        }

        let variance = calculateImageVariance(cgImage)
        if variance < Config.uniformityThreshold {
            throw ValidationError.uniformInputImage
        }
    }

    /// Validates image format (Error 12)
    func validateImageFormat(_ image: UIImage, sourceURL: URL? = nil) throws {
        // Check if we can determine format from URL
        if let url = sourceURL {
            let ext = url.pathExtension.lowercased()
            if !Config.supportedImageFormats.contains(ext) {
                throw ValidationError.invalidImageFormat(format: ext)
            }
        }

        // Also verify we can get valid image data
        guard image.jpegData(compressionQuality: 1.0) != nil ||
              image.pngData() != nil else {
            throw ValidationError.invalidImageFormat(format: nil)
        }
    }

    /// Validates image format from Data
    func validateImageFormat(data: Data) throws {
        // Check magic bytes for JPEG/PNG
        let bytes = [UInt8](data.prefix(8))

        // JPEG magic bytes: 0xFF 0xD8 0xFF
        let isJPEG = bytes.count >= 3 &&
                     bytes[0] == 0xFF &&
                     bytes[1] == 0xD8 &&
                     bytes[2] == 0xFF

        // PNG magic bytes: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
        let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        let isPNG = bytes.count >= 8 && Array(bytes.prefix(8)) == pngSignature

        if !isJPEG && !isPNG {
            let detectedFormat = isJPEG ? "JPEG" : (isPNG ? "PNG" : "Unknown")
            throw ValidationError.invalidImageFormat(format: detectedFormat)
        }
    }

    // MARK: - Grid Validation

    /// Validates grid detection result (Errors 3, 4, 5, 6)
    func validateGridDetection(_ gridResult: GridDetectionResult?) throws {
        guard let grid = gridResult else {
            throw ValidationError.noGridDetected
        }

        // Error 4: Bad grid angle
        if abs(grid.angleInDegrees) > Config.maximumGridAngle {
            throw ValidationError.badGridAngle(angle: grid.angleInDegrees)
        }

        // Error 5: Not enough squares
        if grid.detectedSquareCount < Config.minimumGridSquares {
            throw ValidationError.notEnoughSquares(
                detected: grid.detectedSquareCount,
                minimum: Config.minimumGridSquares
            )
        }

        // Error 6: Too many points (noise)
        if grid.detectedPointCount > Config.maximumGridPoints {
            throw ValidationError.tooManyPoints(
                detected: grid.detectedPointCount,
                maximum: Config.maximumGridPoints
            )
        }

        // Validate grid spacing uniformity
        if grid.spacingVariance > Config.gridSpacingVarianceThreshold {
            throw ValidationError.badGridAngle(angle: grid.angleInDegrees)
        }
    }

    /// Validates image rotation (Error 7)
    func validateRotation(angle: Double) throws {
        if abs(angle) > Config.maximumGridAngle {
            throw ValidationError.wrongRotation(detectedAngle: angle)
        }
    }

    // MARK: - Layout Validation

    /// Validates detected layout (Error 9)
    func validateLayout(_ layout: ECGLayout?, detectedLayoutString: String? = nil) throws {
        guard layout != nil else {
            throw ValidationError.wrongLayout(detected: detectedLayoutString)
        }
    }

    /// Validates layout classification confidence
    func validateLayoutConfidence(confidence: Double, threshold: Double = 0.8) throws {
        if confidence < threshold {
            throw ValidationError.wrongLayout(detected: "low confidence: \(String(format: "%.0f", confidence * 100))%")
        }
    }

    // MARK: - Lead Validation

    /// Validates leads were detected (Error 8)
    func validateLeadsDetected(_ leads: [ECGLead]?) throws {
        guard let leads = leads, !leads.isEmpty else {
            throw ValidationError.noLeadsDetected
        }
    }

    /// Validates lead count matches expected (Error 10)
    func validateLeadCount(_ leads: [ECGLead], layout: ECGLayout) throws {
        let baseLeadCount = leads.filter { !$0.isRhythmLead }.count
        let rhythmLeadCount = leads.filter { $0.isRhythmLead }.count

        let expectedBase = 12
        let expectedRhythm = layout.rhythmLeads

        // Check base leads
        if baseLeadCount != expectedBase {
            throw ValidationError.wrongLeadNumber(detected: baseLeadCount, expected: expectedBase)
        }

        // Check rhythm leads (more lenient - can have 0 to expected)
        if rhythmLeadCount > expectedRhythm {
            throw ValidationError.wrongLeadNumber(
                detected: baseLeadCount + rhythmLeadCount,
                expected: expectedBase + expectedRhythm
            )
        }
    }

    /// Validates all base leads are present
    func validateAllBaseLeadsPresent(_ leads: [ECGLead]) throws {
        let presentTypes = Set(leads.filter { !$0.isRhythmLead }.map { $0.type })
        let requiredTypes = Set(LeadType.baseLeads)

        let missingTypes = requiredTypes.subtracting(presentTypes)
        if !missingTypes.isEmpty {
            throw ValidationError.wrongLeadNumber(
                detected: presentTypes.count,
                expected: requiredTypes.count
            )
        }
    }

    /// Validates lead durations (Error 11)
    func validateLeadDurations(_ leads: [ECGLead]) throws {
        for lead in leads where !lead.isRhythmLead {
            if lead.durationMs < Config.minimumLeadDurationMs {
                throw ValidationError.leadsTooShort(shortestDurationMs: lead.durationMs)
            }
        }
    }

    /// Validates leads don't have flatlines (disconnected electrodes)
    func validateNoFlatlines(_ leads: [ECGLead]) throws {
        for lead in leads where !lead.isRhythmLead {
            if lead.amplitude < Config.minimumSignalAmplitude {
                // Flatline detected - this is a warning in PMcardio spec
                // but we'll let it pass with a warning rather than error
                // The validation status will be set to .warning
            }
        }
    }

    // MARK: - Parameter Validation

    /// Validates processing parameters (Error 13)
    func validateParameters(_ parameters: ProcessingParameters) throws {
        // Both paper speed and voltage gain are enum values, so they're always valid
        // This function exists for future extensibility and custom parameter validation

        // Example: could add validation for custom parameters
        guard parameters.isValid else {
            throw ValidationError.invalidParameters(reason: "Invalid parameter combination")
        }
    }

    // MARK: - Complete Validation Pipeline

    /// Runs all image validations
    func validateImage(_ image: UIImage, sourceURL: URL? = nil) throws {
        try validateImageSize(image)
        try validateImageContent(image)
        try validateImageFormat(image, sourceURL: sourceURL)
    }

    /// Runs all grid validations
    func validateGrid(_ gridResult: GridDetectionResult?) throws {
        try validateGridDetection(gridResult)
        if let grid = gridResult {
            try validateRotation(angle: grid.angleInDegrees)
        }
    }

    /// Runs all lead validations
    func validateLeads(_ leads: [ECGLead]?, layout: ECGLayout) throws {
        try validateLeadsDetected(leads)
        guard let leads = leads else { return }

        try validateLeadCount(leads, layout: layout)
        try validateAllBaseLeadsPresent(leads)
        try validateLeadDurations(leads)
        try validateNoFlatlines(leads)
    }

    /// Complete validation pipeline
    func validate(
        image: UIImage,
        gridResult: GridDetectionResult?,
        layout: ECGLayout?,
        leads: [ECGLead]?,
        parameters: ProcessingParameters
    ) throws -> ValidationStatus {
        // Image validation
        try validateImage(image)

        // Parameter validation
        try validateParameters(parameters)

        // Grid validation
        try validateGrid(gridResult)

        // Layout validation
        try validateLayout(layout)
        guard let layout = layout else {
            throw ValidationError.wrongLayout(detected: nil)
        }

        // Lead validation
        try validateLeads(leads, layout: layout)

        // Check for warnings (flatlines, etc.)
        if let leads = leads {
            let hasWarnings = leads.contains { lead in
                !lead.isRhythmLead && lead.amplitude < Config.minimumSignalAmplitude
            }
            return hasWarnings ? .warning : .valid
        }

        return .valid
    }

    // MARK: - Helper Methods

    /// Calculates pixel variance to detect blank/uniform images
    private func calculateImageVariance(_ cgImage: CGImage) -> Double {
        let width = cgImage.width
        let height = cgImage.height

        // Sample a subset of pixels for efficiency
        let sampleSize = min(width * height, 10000)
        let step = max(1, (width * height) / sampleSize)

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return 0.0
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else { return 0.0 }

        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

        var sum: Double = 0
        var sumSquares: Double = 0
        var count: Double = 0

        for i in stride(from: 0, to: width * height, by: step) {
            let idx = i * 4
            // Convert to grayscale using luminosity
            let gray = Double(pixels[idx]) * 0.299 +
                       Double(pixels[idx + 1]) * 0.587 +
                       Double(pixels[idx + 2]) * 0.114
            sum += gray
            sumSquares += gray * gray
            count += 1
        }

        let mean = sum / count
        let variance = (sumSquares / count) - (mean * mean)

        // Normalize variance to 0-1 range
        return variance / (255.0 * 255.0)
    }
}

// MARK: - Grid Detection Result

/// Result from grid detection for validation
struct GridDetectionResult {
    let angleInDegrees: Double
    let detectedSquareCount: Int
    let detectedPointCount: Int
    let spacingVariance: Double
    let horizontalSpacing: Double
    let verticalSpacing: Double
    let gridBounds: CGRect
    let confidence: Double

    /// Convert to GridCalibration
    func toCalibration() -> GridCalibration {
        GridCalibration(
            smallSquareWidthPixels: horizontalSpacing,
            smallSquareHeightPixels: verticalSpacing,
            angleInDegrees: angleInDegrees,
            qualityScore: confidence,
            gridBounds: gridBounds
        )
    }
}
