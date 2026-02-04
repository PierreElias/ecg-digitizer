import Foundation
import CoreGraphics
import Accelerate

/// Detects perspective distortion in ECG grid images using Hough transform
///
/// This implements the Python perspective_detector.py algorithm:
/// 1. Apply Sobel edge detection on grid probability map
/// 2. Hough transform to find dominant line angles
/// 3. Two-pass refinement: coarse then fine
/// 4. Output theta ranges for horizontal and vertical grid lines
class PerspectiveDetector {

    // MARK: - Types

    struct PerspectiveParams {
        /// Rotation angle in radians (0 = no rotation)
        let rotationAngle: Float

        /// Minimum angle for horizontal lines (radians)
        let thetaMinHorizontal: Float

        /// Maximum angle for horizontal lines (radians)
        let thetaMaxHorizontal: Float

        /// Minimum angle for vertical lines (radians)
        let thetaMinVertical: Float

        /// Maximum angle for vertical lines (radians)
        let thetaMaxVertical: Float

        /// Quality score (0-1) indicating confidence in detection
        let qualityScore: Float

        /// Bounding rectangle of detected grid area
        let gridBounds: CGRect

        static var identity: PerspectiveParams {
            PerspectiveParams(
                rotationAngle: 0.0,
                thetaMinHorizontal: 0.0,
                thetaMaxHorizontal: 0.0,
                thetaMinVertical: Float.pi / 2,
                thetaMaxVertical: Float.pi / 2,
                qualityScore: 1.0,
                gridBounds: .zero
            )
        }
    }

    // MARK: - Configuration

    /// Number of angle bins for coarse Hough transform
    private let coarseAngleBins: Int = 180

    /// Angle range for refinement (degrees)
    private let refinementRange: Float = 5.0

    /// Minimum edge magnitude threshold
    private let edgeThreshold: Float = 0.1

    // MARK: - Detection

    /// Detect perspective distortion from grid probability map
    /// - Parameters:
    ///   - gridProb: Grid probability map (height × width)
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Perspective parameters
    func detectPerspective(
        gridProb: [Float],
        width: Int,
        height: Int
    ) throws -> PerspectiveParams {

        print("🔍 PerspectiveDetector: Processing \(width)x\(height) grid probability map")

        // Step 1: Compute edge gradients using Sobel
        let (gradX, gradY) = computeSobelGradients(
            image: gridProb,
            width: width,
            height: height
        )

        // Step 2: Compute edge magnitude and angle
        let (magnitude, angle) = computeMagnitudeAndAngle(
            gradX: gradX,
            gradY: gradY,
            count: width * height
        )

        // Step 3: Coarse Hough transform (0-180 degrees)
        let coarseHistogram = computeAngleHistogram(
            magnitude: magnitude,
            angle: angle,
            count: width * height
        )

        // Step 4: Find dominant peaks (horizontal ~0/180°, vertical ~90°)
        let (horizontalPeak, verticalPeak) = findDominantPeaks(histogram: coarseHistogram)

        print("   Coarse peaks: horizontal=\(horizontalPeak)°, vertical=\(verticalPeak)°")

        // Step 5: Refined Hough around peaks
        let refinedHorizontal = refineAngle(
            magnitude: magnitude,
            angle: angle,
            count: width * height,
            centerAngle: horizontalPeak
        )

        let refinedVertical = refineAngle(
            magnitude: magnitude,
            angle: angle,
            count: width * height,
            centerAngle: verticalPeak
        )

        print("   Refined: horizontal=\(refinedHorizontal.center)°, vertical=\(refinedVertical.center)°")

        // Step 6: Calculate rotation correction (horizontal should be ~0°)
        let dominantAngle = refinedHorizontal.center

        // Step 7: Calculate confidence from peak clarity
        let confidence = calculateConfidence(
            histogram: coarseHistogram,
            horizontalPeak: Int(horizontalPeak),
            verticalPeak: Int(verticalPeak)
        )

        // Convert to radians
        let toRadians = { (deg: Float) -> Float in deg * .pi / 180.0 }

        return PerspectiveParams(
            rotationAngle: toRadians(dominantAngle),
            thetaMinHorizontal: toRadians(refinedHorizontal.min),
            thetaMaxHorizontal: toRadians(refinedHorizontal.max),
            thetaMinVertical: toRadians(refinedVertical.min),
            thetaMaxVertical: toRadians(refinedVertical.max),
            qualityScore: confidence,
            gridBounds: CGRect(x: 0, y: 0, width: width, height: height)
        )
    }

    // MARK: - Sobel Edge Detection

    /// Compute Sobel gradients
    private func computeSobelGradients(
        image: [Float],
        width: Int,
        height: Int
    ) -> (gradX: [Float], gradY: [Float]) {

        var gradX = [Float](repeating: 0, count: width * height)
        var gradY = [Float](repeating: 0, count: width * height)

        // Sobel kernels
        let sobelX: [Float] = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
        let sobelY: [Float] = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

        // Apply convolution (3x3 kernel)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var sumX: Float = 0
                var sumY: Float = 0

                for ky in 0..<3 {
                    for kx in 0..<3 {
                        let px = x + kx - 1
                        let py = y + ky - 1
                        let pixel = image[py * width + px]
                        let ki = ky * 3 + kx

                        sumX += pixel * sobelX[ki]
                        sumY += pixel * sobelY[ki]
                    }
                }

                gradX[y * width + x] = sumX
                gradY[y * width + x] = sumY
            }
        }

        return (gradX, gradY)
    }

    // MARK: - Magnitude and Angle

    /// Compute edge magnitude and angle from gradients
    private func computeMagnitudeAndAngle(
        gradX: [Float],
        gradY: [Float],
        count: Int
    ) -> (magnitude: [Float], angle: [Float]) {

        var magnitude = [Float](repeating: 0, count: count)
        var angle = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let gx = gradX[i]
            let gy = gradY[i]

            // Magnitude = sqrt(gx² + gy²)
            magnitude[i] = sqrt(gx * gx + gy * gy)

            // Angle = atan2(gy, gx) in degrees [0, 180)
            var theta = atan2(gy, gx) * 180.0 / .pi
            if theta < 0 { theta += 180.0 }
            angle[i] = theta
        }

        return (magnitude, angle)
    }

    // MARK: - Angle Histogram

    /// Compute angle histogram weighted by edge magnitude
    private func computeAngleHistogram(
        magnitude: [Float],
        angle: [Float],
        count: Int
    ) -> [Float] {

        var histogram = [Float](repeating: 0, count: coarseAngleBins)
        let binWidth = 180.0 / Float(coarseAngleBins)

        for i in 0..<count {
            let mag = magnitude[i]
            guard mag > edgeThreshold else { continue }

            let ang = angle[i]
            let bin = min(coarseAngleBins - 1, Int(ang / binWidth))
            histogram[bin] += mag
        }

        return histogram
    }

    // MARK: - Peak Finding

    /// Find dominant horizontal and vertical peaks
    private func findDominantPeaks(histogram: [Float]) -> (horizontal: Float, vertical: Float) {
        let numBins = histogram.count
        let binWidth = 180.0 / Float(numBins)

        // Find global maximum
        var maxBin = 0
        var maxValue: Float = 0

        for i in 0..<numBins {
            if histogram[i] > maxValue {
                maxValue = histogram[i]
                maxBin = i
            }
        }

        let firstPeak = Float(maxBin) * binWidth

        // Find second peak ~90° away from first
        let targetSecondBin = (maxBin + numBins / 2) % numBins
        let searchRange = numBins / 6  // ±30°

        var secondMaxBin = targetSecondBin
        var secondMaxValue: Float = 0

        for offset in -searchRange...searchRange {
            let bin = (targetSecondBin + offset + numBins) % numBins
            if histogram[bin] > secondMaxValue {
                secondMaxValue = histogram[bin]
                secondMaxBin = bin
            }
        }

        let secondPeak = Float(secondMaxBin) * binWidth

        // Assign to horizontal/vertical based on angle
        let horizontal: Float
        let vertical: Float

        if abs(firstPeak - 90) < abs(secondPeak - 90) {
            vertical = firstPeak
            horizontal = secondPeak
        } else {
            horizontal = firstPeak
            vertical = secondPeak
        }

        return (horizontal, vertical)
    }

    // MARK: - Angle Refinement

    private struct RefinedAngle {
        let min: Float
        let max: Float
        let center: Float
    }

    /// Refine angle estimate with higher resolution
    private func refineAngle(
        magnitude: [Float],
        angle: [Float],
        count: Int,
        centerAngle: Float
    ) -> RefinedAngle {

        let numBins = 100
        let minAngle = centerAngle - refinementRange
        let maxAngle = centerAngle + refinementRange
        let binWidth = (refinementRange * 2) / Float(numBins)

        var histogram = [Float](repeating: 0, count: numBins)

        for i in 0..<count {
            let mag = magnitude[i]
            guard mag > edgeThreshold else { continue }

            var ang = angle[i]

            // Handle wrap-around near 0/180
            if centerAngle < refinementRange && ang > 180 - refinementRange {
                ang -= 180
            } else if centerAngle > 180 - refinementRange && ang < refinementRange {
                ang += 180
            }

            if ang >= minAngle && ang <= maxAngle {
                let bin = min(numBins - 1, Int((ang - minAngle) / binWidth))
                histogram[bin] += mag
            }
        }

        // Find peak in refined histogram
        var peakBin = 0
        var peakValue: Float = 0
        for i in 0..<numBins {
            if histogram[i] > peakValue {
                peakValue = histogram[i]
                peakBin = i
            }
        }

        let refinedCenter = minAngle + (Float(peakBin) + 0.5) * binWidth

        // Find FWHM for min/max
        let halfMax = peakValue / 2
        var minBin = peakBin
        var maxBin = peakBin

        while minBin > 0 && histogram[minBin] > halfMax {
            minBin -= 1
        }
        while maxBin < numBins - 1 && histogram[maxBin] > halfMax {
            maxBin += 1
        }

        let refinedMin = minAngle + Float(minBin) * binWidth
        let refinedMax = minAngle + Float(maxBin + 1) * binWidth

        return RefinedAngle(min: refinedMin, max: refinedMax, center: refinedCenter)
    }

    // MARK: - Confidence Score

    /// Calculate confidence score from peak clarity
    private func calculateConfidence(
        histogram: [Float],
        horizontalPeak: Int,
        verticalPeak: Int
    ) -> Float {

        guard !histogram.isEmpty else { return 0 }

        let total = histogram.reduce(0, +)
        guard total > 0 else { return 0 }

        // Sum values near peaks (±5 bins)
        let peakRange = 5
        var peakSum: Float = 0

        for offset in -peakRange...peakRange {
            let hBin = (horizontalPeak + offset + histogram.count) % histogram.count
            let vBin = (verticalPeak + offset + histogram.count) % histogram.count
            peakSum += histogram[hBin] + histogram[vBin]
        }

        // Confidence = fraction of edge energy in peaks
        let confidence = peakSum / total

        return min(1.0, confidence * 2)
    }
}

// MARK: - Errors

enum PerspectiveDetectionError: LocalizedError {
    case insufficientEdges
    case noPeaksFound

    var errorDescription: String? {
        switch self {
        case .insufficientEdges:
            return "Insufficient edge information in grid probability map"
        case .noPeaksFound:
            return "Could not find dominant line angles in histogram"
        }
    }
}
