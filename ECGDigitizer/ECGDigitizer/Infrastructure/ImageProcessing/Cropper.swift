import Foundation
import CoreImage
import CoreGraphics
import Accelerate

/// Crops and perspective-corrects ECG images using CoreImage
///
/// This implements the Python cropper.py algorithm:
/// 1. Use perspective parameters to find corner points
/// 2. Apply perspective transformation to make grid rectangular
/// 3. Crop to signal-containing region using cumulative probability
class Cropper {

    // MARK: - Types

    struct CropResult {
        /// Cropped and corrected image (CGImage)
        let image: CGImage

        /// Cropped signal probability map
        let signalProb: [Float]

        /// Cropped grid probability map
        let gridProb: [Float]

        /// Output dimensions
        let width: Int
        let height: Int

        /// Crop bounds in original image coordinates
        let cropBounds: CGRect
    }

    // MARK: - Configuration

    /// Margin around detected signal region (fraction of image)
    private let marginFraction: CGFloat = 0.02

    /// Minimum cumulative probability threshold for signal detection
    private let signalThreshold: Float = 0.01

    /// CoreImage context for processing
    private let ciContext = CIContext()

    // MARK: - Cropping

    /// Crop and perspective-correct an ECG image
    /// - Parameters:
    ///   - image: Original image
    ///   - signalProb: Signal probability map
    ///   - gridProb: Grid probability map
    ///   - perspective: Perspective parameters from PerspectiveDetector
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Cropped result with corrected image and probability maps
    func crop(
        image: CGImage,
        signalProb: [Float],
        gridProb: [Float],
        perspective: PerspectiveDetector.PerspectiveParams,
        width: Int,
        height: Int
    ) throws -> CropResult {

        print("✂️ Cropper: Processing \(width)x\(height) image")

        // Step 1: Find signal bounds using cumulative probability
        let signalBounds = findSignalBounds(
            signalProb: signalProb,
            width: width,
            height: height
        )

        print("   Signal bounds: \(signalBounds)")

        // Step 2: Calculate perspective-corrected corners
        let corners = calculateCorrectedCorners(
            bounds: signalBounds,
            perspective: perspective,
            imageWidth: width,
            imageHeight: height
        )

        // Step 3: Apply perspective correction using CoreImage
        let correctedImage = try applyPerspectiveCorrection(
            image: image,
            corners: corners
        )

        // Step 4: Crop probability maps to match
        let (croppedSignal, croppedGrid, cropBounds) = cropProbabilityMaps(
            signalProb: signalProb,
            gridProb: gridProb,
            bounds: signalBounds,
            width: width,
            height: height
        )

        let outputWidth = Int(cropBounds.width)
        let outputHeight = Int(cropBounds.height)

        print("   Output size: \(outputWidth)x\(outputHeight)")

        return CropResult(
            image: correctedImage,
            signalProb: croppedSignal,
            gridProb: croppedGrid,
            width: outputWidth,
            height: outputHeight,
            cropBounds: cropBounds
        )
    }

    // MARK: - Signal Bounds Detection

    /// Find bounds of signal region using cumulative probability
    private func findSignalBounds(
        signalProb: [Float],
        width: Int,
        height: Int
    ) -> CGRect {

        // Compute cumulative probability along X (sum columns)
        var xCumulative = [Float](repeating: 0, count: width)
        for x in 0..<width {
            var sum: Float = 0
            for y in 0..<height {
                sum += signalProb[y * width + x]
            }
            xCumulative[x] = sum
        }

        // Compute cumulative probability along Y (sum rows)
        var yCumulative = [Float](repeating: 0, count: height)
        for y in 0..<height {
            var sum: Float = 0
            for x in 0..<width {
                sum += signalProb[y * width + x]
            }
            yCumulative[y] = sum
        }

        // Find total probability
        let totalX = xCumulative.reduce(0, +)
        let totalY = yCumulative.reduce(0, +)

        guard totalX > 0 && totalY > 0 else {
            // No signal detected, return full image
            return CGRect(x: 0, y: 0, width: width, height: height)
        }

        // Find X bounds (left and right edges of signal)
        var minX = 0
        var cumSum: Float = 0
        for x in 0..<width {
            cumSum += xCumulative[x]
            if cumSum / totalX > signalThreshold {
                minX = x
                break
            }
        }

        var maxX = width - 1
        cumSum = 0
        for x in stride(from: width - 1, through: 0, by: -1) {
            cumSum += xCumulative[x]
            if cumSum / totalX > signalThreshold {
                maxX = x
                break
            }
        }

        // Find Y bounds (top and bottom edges of signal)
        var minY = 0
        cumSum = 0
        for y in 0..<height {
            cumSum += yCumulative[y]
            if cumSum / totalY > signalThreshold {
                minY = y
                break
            }
        }

        var maxY = height - 1
        cumSum = 0
        for y in stride(from: height - 1, through: 0, by: -1) {
            cumSum += yCumulative[y]
            if cumSum / totalY > signalThreshold {
                maxY = y
                break
            }
        }

        // Add margin
        let marginX = Int(CGFloat(width) * marginFraction)
        let marginY = Int(CGFloat(height) * marginFraction)

        minX = max(0, minX - marginX)
        maxX = min(width - 1, maxX + marginX)
        minY = max(0, minY - marginY)
        maxY = min(height - 1, maxY + marginY)

        return CGRect(
            x: minX,
            y: minY,
            width: maxX - minX + 1,
            height: maxY - minY + 1
        )
    }

    // MARK: - Perspective Correction

    private struct Corners {
        let topLeft: CGPoint
        let topRight: CGPoint
        let bottomRight: CGPoint
        let bottomLeft: CGPoint
    }

    /// Calculate perspective-corrected corner points
    private func calculateCorrectedCorners(
        bounds: CGRect,
        perspective: PerspectiveDetector.PerspectiveParams,
        imageWidth: Int,
        imageHeight: Int
    ) -> Corners {

        let rotationAngle = CGFloat(perspective.rotationAngle)

        // Calculate rotation-adjusted corners
        let centerX = bounds.midX
        let centerY = bounds.midY

        func rotatePoint(_ point: CGPoint) -> CGPoint {
            let dx = point.x - centerX
            let dy = point.y - centerY
            let cos_a = cos(rotationAngle)
            let sin_a = sin(rotationAngle)
            return CGPoint(
                x: centerX + dx * cos_a - dy * sin_a,
                y: centerY + dx * sin_a + dy * cos_a
            )
        }

        // Original corners
        let tl = CGPoint(x: bounds.minX, y: bounds.minY)
        let tr = CGPoint(x: bounds.maxX, y: bounds.minY)
        let br = CGPoint(x: bounds.maxX, y: bounds.maxY)
        let bl = CGPoint(x: bounds.minX, y: bounds.maxY)

        // If rotation is significant, rotate corners
        if abs(rotationAngle) > 0.01 {
            return Corners(
                topLeft: rotatePoint(tl),
                topRight: rotatePoint(tr),
                bottomRight: rotatePoint(br),
                bottomLeft: rotatePoint(bl)
            )
        } else {
            return Corners(
                topLeft: tl,
                topRight: tr,
                bottomRight: br,
                bottomLeft: bl
            )
        }
    }

    /// Apply perspective correction using CoreImage
    private func applyPerspectiveCorrection(
        image: CGImage,
        corners: Corners
    ) throws -> CGImage {

        let ciImage = CIImage(cgImage: image)

        // CIPerspectiveCorrection expects points in Core Image coordinate system
        // (origin at bottom-left, Y increases upward)
        let imageHeight = CGFloat(image.height)

        let filter = CIFilter(name: "CIPerspectiveCorrection")!
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(CIVector(cgPoint: CGPoint(x: corners.topLeft.x, y: imageHeight - corners.topLeft.y)),
                       forKey: "inputTopLeft")
        filter.setValue(CIVector(cgPoint: CGPoint(x: corners.topRight.x, y: imageHeight - corners.topRight.y)),
                       forKey: "inputTopRight")
        filter.setValue(CIVector(cgPoint: CGPoint(x: corners.bottomRight.x, y: imageHeight - corners.bottomRight.y)),
                       forKey: "inputBottomRight")
        filter.setValue(CIVector(cgPoint: CGPoint(x: corners.bottomLeft.x, y: imageHeight - corners.bottomLeft.y)),
                       forKey: "inputBottomLeft")

        guard let outputImage = filter.outputImage else {
            throw CropperError.perspectiveCorrectionFailed
        }

        guard let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            throw CropperError.imageCreationFailed
        }

        return cgImage
    }

    // MARK: - Probability Map Cropping

    /// Crop probability maps to match image crop
    private func cropProbabilityMaps(
        signalProb: [Float],
        gridProb: [Float],
        bounds: CGRect,
        width: Int,
        height: Int
    ) -> (signal: [Float], grid: [Float], bounds: CGRect) {

        let minX = Int(bounds.minX)
        let minY = Int(bounds.minY)
        let cropWidth = Int(bounds.width)
        let cropHeight = Int(bounds.height)

        var croppedSignal = [Float](repeating: 0, count: cropWidth * cropHeight)
        var croppedGrid = [Float](repeating: 0, count: cropWidth * cropHeight)

        for y in 0..<cropHeight {
            for x in 0..<cropWidth {
                let srcX = minX + x
                let srcY = minY + y

                guard srcX >= 0 && srcX < width && srcY >= 0 && srcY < height else {
                    continue
                }

                let srcIdx = srcY * width + srcX
                let dstIdx = y * cropWidth + x

                croppedSignal[dstIdx] = signalProb[srcIdx]
                croppedGrid[dstIdx] = gridProb[srcIdx]
            }
        }

        return (croppedSignal, croppedGrid, bounds)
    }
}

// MARK: - Errors

enum CropperError: LocalizedError {
    case perspectiveCorrectionFailed
    case imageCreationFailed
    case noSignalDetected

    var errorDescription: String? {
        switch self {
        case .perspectiveCorrectionFailed:
            return "Failed to apply perspective correction"
        case .imageCreationFailed:
            return "Failed to create CGImage from perspective-corrected result"
        case .noSignalDetected:
            return "No signal detected in probability map"
        }
    }
}
