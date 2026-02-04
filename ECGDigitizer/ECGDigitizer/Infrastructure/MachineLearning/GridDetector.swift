import Foundation
import UIKit
import Vision
import Accelerate

/// Detects and analyzes ECG grid pattern in images
final class GridDetector {

    private let visionManager: VisionManager
    private let preprocessor: ImagePreprocessor

    // MARK: - Configuration

    struct Config {
        /// Expected small square size in mm (standard ECG paper)
        static let smallSquareMm: Double = 1.0

        /// Expected large square size in mm
        static let largeSquareMm: Double = 5.0

        /// Minimum number of grid intersections to consider valid
        static let minimumIntersections: Int = 100

        /// Maximum angle deviation from horizontal (degrees)
        static let maximumAngle: Double = 5.0

        /// Tolerance for grid spacing variance
        static let spacingTolerance: Double = 0.1  // 10%

        /// Minimum grid line length (as fraction of image dimension)
        static let minimumLineLength: Double = 0.1

        /// Maximum grid line length (as fraction of image dimension)
        static let maximumLineLength: Double = 0.95
    }

    // MARK: - Initialization

    init(visionManager: VisionManager) {
        self.visionManager = visionManager
        self.preprocessor = ImagePreprocessor()
    }

    // MARK: - Grid Detection

    /// Detects ECG grid in the image
    func detectGrid(in image: UIImage) async throws -> GridDetectionResult {
        // Step 1: Preprocess image for grid detection
        let preprocessedImage = preprocessor.preprocessForGridDetection(image)

        // Step 2: Detect horizon angle
        _ = try await visionManager.detectHorizonAngle(in: image) // horizonAngle for future deskewing

        // Step 3: Detect grid lines using Hough transform approximation
        let gridLines = try await detectGridLines(in: preprocessedImage)

        // Step 4: Find grid intersections
        let intersections = findGridIntersections(gridLines)

        // Step 5: Analyze grid pattern
        let gridAnalysis = analyzeGridPattern(intersections: intersections, imageSize: image.size)

        // Step 6: Build result
        return GridDetectionResult(
            angleInDegrees: gridAnalysis.angle,
            detectedSquareCount: gridAnalysis.squareCount,
            detectedPointCount: intersections.count,
            spacingVariance: gridAnalysis.spacingVariance,
            horizontalSpacing: gridAnalysis.horizontalSpacing,
            verticalSpacing: gridAnalysis.verticalSpacing,
            gridBounds: gridAnalysis.bounds,
            confidence: gridAnalysis.confidence
        )
    }

    /// Detects grid using enhanced method with edge detection
    func detectGridEnhanced(in image: UIImage) async throws -> GridDetectionResult {
        // Apply edge detection first
        guard let edgeImage = visionManager.detectEdges(in: image) else {
            throw GridDetectionError.preprocessingFailed
        }

        // Detect contours
        let contours = try await visionManager.detectLines(in: edgeImage)

        // Extract line segments from contours
        let lines = extractLinesFromContours(contours, imageSize: image.size)

        // Separate horizontal and vertical lines
        let (horizontal, vertical) = separateLines(lines)

        // Find intersections
        let intersections = findLineIntersections(horizontal: horizontal, vertical: vertical)

        // Analyze grid
        let analysis = analyzeGridPattern(intersections: intersections, imageSize: image.size)

        return GridDetectionResult(
            angleInDegrees: analysis.angle,
            detectedSquareCount: analysis.squareCount,
            detectedPointCount: intersections.count,
            spacingVariance: analysis.spacingVariance,
            horizontalSpacing: analysis.horizontalSpacing,
            verticalSpacing: analysis.verticalSpacing,
            gridBounds: analysis.bounds,
            confidence: analysis.confidence
        )
    }

    // MARK: - Private Methods

    /// Detects grid lines using Vision contours
    private func detectGridLines(in image: UIImage) async throws -> [Line] {
        let contours = try await visionManager.detectLines(in: image)
        return extractLinesFromContours(contours, imageSize: image.size)
    }

    /// Extracts line segments from contour observations
    private func extractLinesFromContours(
        _ contours: [VNContoursObservation],
        imageSize: CGSize
    ) -> [Line] {
        var lines: [Line] = []

        for observation in contours {
            // Process contours at all levels
            let contourCount = observation.contourCount

            for i in 0..<contourCount {
                if let contour = try? observation.contour(at: i) {
                    let points = extractPointsFromContour(contour)

                    if points.count >= 2 {
                        // Fit line to points
                        if let line = fitLine(to: points, imageSize: imageSize) {
                            lines.append(line)
                        }
                    }
                }
            }
        }

        return lines
    }

    /// Extracts points from a contour
    private func extractPointsFromContour(_ contour: VNContour) -> [CGPoint] {
        var points: [CGPoint] = []
        let pointCount = contour.pointCount

        for i in 0..<pointCount {
            let point = contour.normalizedPoints[i]
            points.append(CGPoint(x: CGFloat(point.x), y: CGFloat(point.y)))
        }

        return points
    }

    /// Fits a line to a set of points using least squares
    private func fitLine(to points: [CGPoint], imageSize: CGSize) -> Line? {
        guard points.count >= 2 else { return nil }

        // Calculate line parameters using least squares
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        var sumXY: CGFloat = 0
        var sumX2: CGFloat = 0

        for point in points {
            sumX += point.x
            sumY += point.y
            sumXY += point.x * point.y
            sumX2 += point.x * point.x
        }

        let n = CGFloat(points.count)
        let denominator = n * sumX2 - sumX * sumX

        guard abs(denominator) > 0.0001 else {
            // Vertical line
            let x = sumX / n
            return Line(
                start: CGPoint(x: x * imageSize.width, y: 0),
                end: CGPoint(x: x * imageSize.width, y: imageSize.height),
                angle: 90
            )
        }

        let slope = (n * sumXY - sumX * sumY) / denominator
        let intercept = (sumY - slope * sumX) / n

        // Calculate angle
        let angle = atan(slope) * 180 / .pi

        // Convert to image coordinates
        let startX: CGFloat = 0
        let startY = intercept * imageSize.height
        let endX = imageSize.width
        let endY = (slope + intercept) * imageSize.height

        return Line(
            start: CGPoint(x: startX, y: startY),
            end: CGPoint(x: endX, y: endY),
            angle: angle
        )
    }

    /// Separates lines into horizontal and vertical
    private func separateLines(_ lines: [Line]) -> (horizontal: [Line], vertical: [Line]) {
        var horizontal: [Line] = []
        var vertical: [Line] = []

        for line in lines {
            if abs(line.angle) < 45 {
                horizontal.append(line)
            } else {
                vertical.append(line)
            }
        }

        return (horizontal, vertical)
    }

    /// Finds intersections between horizontal and vertical lines
    private func findLineIntersections(horizontal: [Line], vertical: [Line]) -> [CGPoint] {
        var intersections: [CGPoint] = []

        for hLine in horizontal {
            for vLine in vertical {
                if let intersection = lineIntersection(hLine, vLine) {
                    intersections.append(intersection)
                }
            }
        }

        return intersections
    }

    /// Calculates intersection point of two lines
    private func lineIntersection(_ line1: Line, _ line2: Line) -> CGPoint? {
        let x1 = line1.start.x
        let y1 = line1.start.y
        let x2 = line1.end.x
        let y2 = line1.end.y
        let x3 = line2.start.x
        let y3 = line2.start.y
        let x4 = line2.end.x
        let y4 = line2.end.y

        let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        guard abs(denominator) > 0.0001 else { return nil }

        let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator

        let x = x1 + t * (x2 - x1)
        let y = y1 + t * (y2 - y1)

        return CGPoint(x: x, y: y)
    }

    /// Finds grid intersections from detected points
    private func findGridIntersections(_ lines: [Line]) -> [CGPoint] {
        let (horizontal, vertical) = separateLines(lines)
        return findLineIntersections(horizontal: horizontal, vertical: vertical)
    }

    /// Analyzes grid pattern from intersections
    private func analyzeGridPattern(
        intersections: [CGPoint],
        imageSize: CGSize
    ) -> GridAnalysis {
        guard intersections.count >= Config.minimumIntersections else {
            return GridAnalysis(
                angle: 0,
                squareCount: intersections.count / 4,
                spacingVariance: 1.0,
                horizontalSpacing: 0,
                verticalSpacing: 0,
                bounds: .zero,
                confidence: 0
            )
        }

        // Sort points by position
        let sortedByX = intersections.sorted { $0.x < $1.x }
        let sortedByY = intersections.sorted { $0.y < $1.y }

        // Calculate horizontal spacing
        var horizontalSpacings: [CGFloat] = []
        for i in 1..<min(sortedByX.count, 100) {
            let spacing = sortedByX[i].x - sortedByX[i-1].x
            if spacing > 1 {  // Ignore very small spacings
                horizontalSpacings.append(spacing)
            }
        }

        // Calculate vertical spacing
        var verticalSpacings: [CGFloat] = []
        for i in 1..<min(sortedByY.count, 100) {
            let spacing = sortedByY[i].y - sortedByY[i-1].y
            if spacing > 1 {
                verticalSpacings.append(spacing)
            }
        }

        // Find most common spacing (grid spacing)
        let hSpacing = findMostCommonValue(horizontalSpacings)
        let vSpacing = findMostCommonValue(verticalSpacings)

        // Calculate spacing variance
        let hVariance = calculateVariance(horizontalSpacings, around: hSpacing)
        let vVariance = calculateVariance(verticalSpacings, around: vSpacing)
        let avgVariance = (hVariance + vVariance) / 2

        // Calculate bounds
        let minX = sortedByX.first?.x ?? 0
        let maxX = sortedByX.last?.x ?? imageSize.width
        let minY = sortedByY.first?.y ?? 0
        let maxY = sortedByY.last?.y ?? imageSize.height
        let bounds = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)

        // Estimate number of squares
        let squareCount = Int((bounds.width / hSpacing) * (bounds.height / vSpacing))

        // Calculate grid angle from point alignment
        let angle = calculateGridAngle(intersections)

        // Calculate confidence
        let confidence = calculateConfidence(
            intersectionCount: intersections.count,
            variance: avgVariance,
            angle: angle
        )

        return GridAnalysis(
            angle: angle,
            squareCount: squareCount,
            spacingVariance: avgVariance,
            horizontalSpacing: Double(hSpacing),
            verticalSpacing: Double(vSpacing),
            bounds: bounds,
            confidence: confidence
        )
    }

    /// Finds the most common value in an array (mode)
    private func findMostCommonValue(_ values: [CGFloat]) -> CGFloat {
        guard !values.isEmpty else { return 0 }

        // Bucket values (quantize to nearest integer)
        var buckets: [Int: Int] = [:]
        for value in values {
            let bucket = Int(value)
            buckets[bucket, default: 0] += 1
        }

        // Find bucket with most entries
        let mostCommon = buckets.max { $0.value < $1.value }
        return CGFloat(mostCommon?.key ?? 0)
    }

    /// Calculates variance of values around expected value
    private func calculateVariance(_ values: [CGFloat], around expected: CGFloat) -> Double {
        guard !values.isEmpty, expected > 0 else { return 1.0 }

        var sumSquaredDiff: CGFloat = 0
        for value in values {
            let diff = (value - expected) / expected
            sumSquaredDiff += diff * diff
        }

        return Double(sumSquaredDiff / CGFloat(values.count))
    }

    /// Calculates grid angle from point alignment
    private func calculateGridAngle(_ points: [CGPoint]) -> Double {
        guard points.count >= 10 else { return 0 }

        // Sample points and fit line
        let sampleSize = min(points.count, 100)
        let sampledPoints = Array(points.prefix(sampleSize))

        // Use linear regression to find dominant angle
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        var sumXY: CGFloat = 0
        var sumX2: CGFloat = 0

        for point in sampledPoints {
            sumX += point.x
            sumY += point.y
            sumXY += point.x * point.y
            sumX2 += point.x * point.x
        }

        let n = CGFloat(sampledPoints.count)
        let denominator = n * sumX2 - sumX * sumX

        guard abs(denominator) > 0.0001 else { return 0 }

        let slope = (n * sumXY - sumX * sumY) / denominator
        let angle = atan(slope) * 180 / .pi

        return Double(angle)
    }

    /// Calculates detection confidence
    private func calculateConfidence(
        intersectionCount: Int,
        variance: Double,
        angle: Double
    ) -> Double {
        // Factors affecting confidence
        let countScore = min(1.0, Double(intersectionCount) / Double(Config.minimumIntersections * 2))
        let varianceScore = max(0, 1.0 - variance * 5)
        let angleScore = max(0, 1.0 - abs(angle) / Config.maximumAngle)

        return (countScore + varianceScore + angleScore) / 3.0
    }
}

// MARK: - Supporting Types

struct Line {
    let start: CGPoint
    let end: CGPoint
    let angle: Double  // in degrees

    var length: CGFloat {
        hypot(end.x - start.x, end.y - start.y)
    }

    var isHorizontal: Bool {
        abs(angle) < 45
    }

    var isVertical: Bool {
        abs(angle) >= 45
    }
}

struct GridAnalysis {
    let angle: Double
    let squareCount: Int
    let spacingVariance: Double
    let horizontalSpacing: Double
    let verticalSpacing: Double
    let bounds: CGRect
    let confidence: Double
}

enum GridDetectionError: Error, LocalizedError {
    case preprocessingFailed
    case noGridFound
    case insufficientPoints
    case invalidAngle

    var errorDescription: String? {
        switch self {
        case .preprocessingFailed:
            return "Failed to preprocess image for grid detection"
        case .noGridFound:
            return "No ECG grid found in image"
        case .insufficientPoints:
            return "Not enough grid points detected"
        case .invalidAngle:
            return "Grid angle is too large"
        }
    }
}
