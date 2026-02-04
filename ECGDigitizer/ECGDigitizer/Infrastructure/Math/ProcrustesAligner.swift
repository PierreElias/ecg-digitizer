//
//  ProcrustesAligner.swift
//  ECGDigitizer
//
//  Procrustes alignment for ECG layout detection
//  Matches detected lead positions to known layout templates
//

import Foundation

/// Procrustes alignment for shape matching
struct ProcrustesAligner {

    // MARK: - Result Structure

    struct AlignmentResult {
        let scale: Float
        let translation: (x: Float, y: Float)
        let cost: Float
        let flip: Bool

        /// Quality score (0-1, higher is better)
        var quality: Float {
            // Convert cost to quality score
            // Lower cost = better quality
            return max(0, 1.0 - cost / 100.0)
        }
    }

    // MARK: - Main Alignment Function

    /// Align detected points to grid template using Procrustes analysis
    /// - Parameters:
    ///   - detectedPoints: Detected lead positions (normalized 0-1)
    ///   - gridPoints: Template grid positions (normalized 0-1)
    /// - Returns: Alignment result with scale, translation, and cost
    func align(
        detectedPoints: [(x: Float, y: Float)],
        gridPoints: [(x: Float, y: Float)]
    ) -> AlignmentResult {
        guard detectedPoints.count >= 3, gridPoints.count >= 3 else {
            return AlignmentResult(
                scale: 1.0,
                translation: (0, 0),
                cost: Float.infinity,
                flip: false
            )
        }

        guard detectedPoints.count == gridPoints.count else {
            print("  [Procrustes] ⚠️ Point count mismatch: detected=\(detectedPoints.count), grid=\(gridPoints.count)")
            return AlignmentResult(
                scale: 1.0,
                translation: (0, 0),
                cost: Float.infinity,
                flip: false
            )
        }

        // Compute centroids
        let pMean = computeCentroid(detectedPoints)
        let gMean = computeCentroid(gridPoints)

        // Center points (subtract mean)
        let pCentered = detectedPoints.map { (x: $0.x - pMean.x, y: $0.y - pMean.y) }
        let gCentered = gridPoints.map { (x: $0.x - gMean.x, y: $0.y - gMean.y) }

        // Compute optimal scale using least squares
        // scale = Σ(p · g) / Σ(p · p)
        var numerator: Float = 0
        var denominator: Float = 0

        for (p, g) in zip(pCentered, gCentered) {
            numerator += p.x * g.x + p.y * g.y
            denominator += p.x * p.x + p.y * p.y
        }

        let scale = denominator > 1e-6 ? numerator / denominator : 1.0

        // Compute translation
        // t = g_mean - scale * p_mean
        let translation = (
            x: gMean.x - scale * pMean.x,
            y: gMean.y - scale * pMean.y
        )

        // Apply transformation and compute residuals
        var residuals: [Float] = []
        for (p, g) in zip(detectedPoints, gridPoints) {
            let transformed = (
                x: scale * p.x + translation.x,
                y: scale * p.y + translation.y
            )
            let distance = sqrt(pow(transformed.x - g.x, 2) + pow(transformed.y - g.y, 2))
            residuals.append(distance)
        }

        // Compute mean residual as cost
        let cost = residuals.reduce(0, +) / Float(residuals.count)

        // Check for flip (negative scale indicates reflection)
        let flip = scale < 0

        return AlignmentResult(
            scale: abs(scale),
            translation: translation,
            cost: cost,
            flip: flip
        )
    }

    // MARK: - Centroid Computation

    /// Compute centroid (mean position) of points
    private func computeCentroid(_ points: [(x: Float, y: Float)]) -> (x: Float, y: Float) {
        guard !points.isEmpty else { return (0, 0) }

        let sumX = points.reduce(0.0) { $0 + $1.x }
        let sumY = points.reduce(0.0) { $0 + $1.y }

        return (
            x: sumX / Float(points.count),
            y: sumY / Float(points.count)
        )
    }

    // MARK: - Layout Template Generation

    /// Generate normalized grid positions for a given layout
    /// - Parameter layout: ECG layout type
    /// - Returns: Array of (x, y) positions in range [0, 1]
    static func generateGridPositions(for layout: ECGLayout) -> [(x: Float, y: Float)] {
        let rows = layout.rows
        let cols = layout.columns

        var positions: [(x: Float, y: Float)] = []

        for row in 0..<rows {
            for col in 0..<cols {
                let x = (cols > 1) ? Float(col) / Float(cols - 1) : 0.5
                let y = (rows > 1) ? Float(row) / Float(rows - 1) : 0.5
                positions.append((x, y))
            }
        }

        return positions
    }

    // MARK: - Layout Detection

    /// Detect ECG layout by testing all templates
    /// - Parameter detectedPoints: Detected lead positions (normalized)
    /// - Returns: Best matching layout and alignment quality
    static func detectLayout(
        detectedPoints: [(x: Float, y: Float)]
    ) -> (layout: ECGLayout, quality: Float) {
        let candidateLayouts: [ECGLayout] = [
            .threeByFour_r0,
            .threeByFour_r1,
            .threeByFour_r2,
            .sixByTwo_r0,
            .sixByTwo_r1
        ]

        let aligner = ProcrustesAligner()
        var bestMatch: (layout: ECGLayout, quality: Float) = (.threeByFour_r1, 0.0)

        for layout in candidateLayouts {
            let gridPoints = generateGridPositions(for: layout)

            // Only compare if point counts match
            guard gridPoints.count == detectedPoints.count else {
                continue
            }

            let result = aligner.align(
                detectedPoints: detectedPoints,
                gridPoints: gridPoints
            )

            let quality = result.quality

            if quality > bestMatch.quality {
                bestMatch = (layout, quality)
            }

            print("  [Procrustes] \(layout): cost=\(String(format: "%.4f", result.cost)), quality=\(String(format: "%.3f", quality))")
        }

        print("  [Procrustes] Best match: \(bestMatch.layout) (quality: \(String(format: "%.3f", bestMatch.quality)))")

        return bestMatch
    }

    // MARK: - Point Normalization

    /// Normalize points to [0, 1] range
    /// - Parameter points: Input points in arbitrary coordinates
    /// - Returns: Normalized points
    static func normalizePoints(_ points: [(x: Float, y: Float)]) -> [(x: Float, y: Float)] {
        guard !points.isEmpty else { return [] }

        // Find bounding box
        let xValues = points.map { $0.x }
        let yValues = points.map { $0.y }

        guard let minX = xValues.min(),
              let maxX = xValues.max(),
              let minY = yValues.min(),
              let maxY = yValues.max() else {
            return points
        }

        let rangeX = maxX - minX
        let rangeY = maxY - minY

        // Avoid division by zero
        guard rangeX > 1e-6, rangeY > 1e-6 else {
            return points.map { _ in (x: 0.5, y: 0.5) }
        }

        // Normalize to [0, 1]
        return points.map { point in
            (
                x: (point.x - minX) / rangeX,
                y: (point.y - minY) / rangeY
            )
        }
    }
}

// Note: ECGLayout properties (rows, columns, rhythmLeads, standardLeadOrder)
// are already defined in ECGLayout.swift - do not duplicate here
