import Foundation
import Vision
import UIKit
import CoreImage

/// Manages Vision framework operations for image processing
final class VisionManager {

    private let ciContext: CIContext

    init() {
        self.ciContext = CIContext(options: [.useSoftwareRenderer: false])
    }

    // MARK: - Line Detection

    /// Detects horizontal and vertical lines in the image for grid detection
    func detectLines(in image: UIImage) async throws -> [VNContoursObservation] {
        guard let cgImage = image.cgImage else {
            throw VisionError.invalidImage
        }

        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = true

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results else {
            return []
        }

        return results
    }

    /// Detects rectangles in the image (useful for finding ECG paper boundaries)
    func detectRectangles(in image: UIImage) async throws -> [VNRectangleObservation] {
        guard let cgImage = image.cgImage else {
            throw VisionError.invalidImage
        }

        let request = VNDetectRectanglesRequest()
        request.minimumConfidence = 0.5
        request.maximumObservations = 10
        request.minimumAspectRatio = 0.3
        request.maximumAspectRatio = 3.0
        request.minimumSize = 0.1

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        return request.results ?? []
    }

    // MARK: - Text Recognition

    /// Recognizes text in the image (for detecting lead labels, metadata)
    func recognizeText(in image: UIImage) async throws -> [VNRecognizedTextObservation] {
        guard let cgImage = image.cgImage else {
            throw VisionError.invalidImage
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US"]
        request.usesLanguageCorrection = false  // ECG labels are short codes

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        return request.results ?? []
    }

    /// Extract lead labels from recognized text
    func extractLeadLabels(from observations: [VNRecognizedTextObservation]) -> [(label: String, boundingBox: CGRect)] {
        let leadPatterns = Set(LeadType.allCases.map { $0.rawValue.uppercased() })

        var labels: [(String, CGRect)] = []

        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }

            let text = candidate.string.uppercased().trimmingCharacters(in: .whitespaces)

            // Check if it matches a lead label
            if leadPatterns.contains(text) || text.hasPrefix("V") || text.hasPrefix("AVR") ||
               text.hasPrefix("AVL") || text.hasPrefix("AVF") {
                labels.append((text, observation.boundingBox))
            }
        }

        return labels
    }

    // MARK: - Edge Detection

    /// Applies Canny edge detection using Core Image
    func detectEdges(in image: UIImage) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        // Apply edge detection filter
        guard let edgeFilter = CIFilter(name: "CIEdges") else { return nil }
        edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)
        edgeFilter.setValue(1.0, forKey: kCIInputIntensityKey)

        guard let outputImage = edgeFilter.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Horizon Detection

    /// Detects the dominant horizontal angle in the image
    func detectHorizonAngle(in image: UIImage) async throws -> Double {
        guard let cgImage = image.cgImage else {
            throw VisionError.invalidImage
        }

        let request = VNDetectHorizonRequest()

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let result = request.results?.first else {
            return 0.0
        }

        // Convert radians to degrees
        return result.angle * 180.0 / Double.pi
    }

    // MARK: - Perspective Correction

    /// Applies perspective correction based on detected rectangle
    func applyPerspectiveCorrection(
        to image: UIImage,
        using rectangle: VNRectangleObservation
    ) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        let imageSize = ciImage.extent.size

        // Convert normalized coordinates to image coordinates
        let topLeft = CGPoint(
            x: rectangle.topLeft.x * imageSize.width,
            y: rectangle.topLeft.y * imageSize.height
        )
        let topRight = CGPoint(
            x: rectangle.topRight.x * imageSize.width,
            y: rectangle.topRight.y * imageSize.height
        )
        let bottomLeft = CGPoint(
            x: rectangle.bottomLeft.x * imageSize.width,
            y: rectangle.bottomLeft.y * imageSize.height
        )
        let bottomRight = CGPoint(
            x: rectangle.bottomRight.x * imageSize.width,
            y: rectangle.bottomRight.y * imageSize.height
        )

        // Apply perspective correction filter
        guard let filter = CIFilter(name: "CIPerspectiveCorrection") else { return nil }

        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(CIVector(cgPoint: topLeft), forKey: "inputTopLeft")
        filter.setValue(CIVector(cgPoint: topRight), forKey: "inputTopRight")
        filter.setValue(CIVector(cgPoint: bottomLeft), forKey: "inputBottomLeft")
        filter.setValue(CIVector(cgPoint: bottomRight), forKey: "inputBottomRight")

        guard let outputImage = filter.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Image Rotation

    /// Rotates image by the specified angle (degrees)
    func rotateImage(_ image: UIImage, byDegrees degrees: Double) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        let radians = degrees * .pi / 180.0
        let transform = CGAffineTransform(rotationAngle: radians)

        let rotatedImage = ciImage.transformed(by: transform)

        guard let cgImage = ciContext.createCGImage(rotatedImage, from: rotatedImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Classification with Core ML

    /// Runs a Core ML model for layout classification
    func classifyWithModel<T: MLModel>(
        _ model: T,
        image: UIImage
    ) async throws -> [VNClassificationObservation] {
        guard let cgImage = image.cgImage else {
            throw VisionError.invalidImage
        }

        guard let vnModel = try? VNCoreMLModel(for: model) else {
            throw VisionError.modelError
        }

        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results as? [VNClassificationObservation] else {
            return []
        }

        return results
    }
}

// MARK: - Vision Errors

enum VisionError: Error, LocalizedError {
    case invalidImage
    case processingFailed
    case modelError
    case noResults

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid image provided for Vision processing"
        case .processingFailed:
            return "Vision processing failed"
        case .modelError:
            return "Failed to load Core ML model"
        case .noResults:
            return "No results from Vision processing"
        }
    }
}

// MARK: - VNRectangleObservation Extension

extension VNRectangleObservation {
    /// Returns the center point of the rectangle
    var center: CGPoint {
        CGPoint(
            x: (topLeft.x + bottomRight.x) / 2,
            y: (topLeft.y + bottomRight.y) / 2
        )
    }

    /// Returns the width of the rectangle
    var width: CGFloat {
        let topWidth = hypot(topRight.x - topLeft.x, topRight.y - topLeft.y)
        let bottomWidth = hypot(bottomRight.x - bottomLeft.x, bottomRight.y - bottomLeft.y)
        return (topWidth + bottomWidth) / 2
    }

    /// Returns the height of the rectangle
    var height: CGFloat {
        let leftHeight = hypot(bottomLeft.x - topLeft.x, bottomLeft.y - topLeft.y)
        let rightHeight = hypot(bottomRight.x - topRight.x, bottomRight.y - topRight.y)
        return (leftHeight + rightHeight) / 2
    }
}
