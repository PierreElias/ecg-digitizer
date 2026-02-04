import Foundation
import Accelerate

/// Finds grid calibration using autocorrelation-based peak detection
/// Ports the Python PixelSizeFinder algorithm to Swift
class GridSizeFinder {

    // MARK: - Types

    struct GridCalibrationResult {
        let pixelsPerMm: Double
        let smallSquareWidthPixels: Double
        let smallSquareHeightPixels: Double
        let angleInDegrees: Double
        let qualityScore: Double
        let gridBounds: CGRect
    }

    // MARK: - Constants

    private let standardGridSpacingMm: Double = 5.0  // ECG paper has 5mm large squares
    private let minPixelsPerMm: Double = 2.0   // Minimum expected pixels/mm (lowered for low-res images)
    private let maxPixelsPerMm: Double = 50.0  // Maximum expected pixels/mm (increased for high-res images)
    private let zoomIterations: Int = 3

    // MARK: - Main Method

    /// Find grid calibration from grid probability map
    /// - Parameters:
    ///   - gridProb: Grid probability map (height × width)
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Grid calibration parameters
    func findGridCalibration(
        gridProb: [Float],
        width: Int,
        height: Int
    ) throws -> GridCalibrationResult {

        // Step 1: Sum columns to get 1D horizontal signal
        let horizontalSignal = sumColumns(gridProb: gridProb, width: width, height: height)

        // Step 2: Compute autocorrelation
        let autocorr = computeAutocorrelation(signal: horizontalSignal)

        // Step 3: Find peak spacing using zoom-based grid search
        let peakSpacing = findPeakSpacing(autocorr: autocorr)

        // Step 4: Calculate pixels per mm
        let pixelsPerMm = peakSpacing / standardGridSpacingMm

        // Validate result
        guard pixelsPerMm >= minPixelsPerMm && pixelsPerMm <= maxPixelsPerMm else {
            throw GridFinderError.calibrationOutOfRange(pixelsPerMm)
        }

        // Step 5: Calculate quality score from autocorrelation peak height
        let qualityScore = calculateQualityScore(autocorr: autocorr, peakSpacing: peakSpacing)

        return GridCalibrationResult(
            pixelsPerMm: pixelsPerMm,
            smallSquareWidthPixels: pixelsPerMm,   // 1mm squares
            smallSquareHeightPixels: pixelsPerMm,
            angleInDegrees: 0.0,  // TODO: Implement rotation detection
            qualityScore: qualityScore,
            gridBounds: CGRect(x: 0, y: 0, width: width, height: height)
        )
    }

    // MARK: - Column Summation

    /// Sum grid probabilities along columns to get 1D horizontal signal
    private func sumColumns(gridProb: [Float], width: Int, height: Int) -> [Float] {
        var columnSums = [Float](repeating: 0.0, count: width)

        for x in 0..<width {
            var sum: Float = 0.0
            for y in 0..<height {
                sum += gridProb[y * width + x]
            }
            columnSums[x] = sum
        }

        return columnSums
    }

    // MARK: - Autocorrelation

    /// Compute autocorrelation of signal using vDSP
    /// Matches scipy.signal.correlate(signal, signal, mode='same')
    private func computeAutocorrelation(signal: [Float]) -> [Float] {
        let n = signal.count
        var autocorr = [Float](repeating: 0.0, count: n)

        // Normalize signal (zero mean)
        var normalizedSignal = signal
        var mean: Float = 0.0
        vDSP_meanv(signal, 1, &mean, vDSP_Length(n))

        var negativeMean = -mean
        vDSP_vsadd(signal, 1, &negativeMean, &normalizedSignal, 1, vDSP_Length(n))

        // Compute autocorrelation using vDSP convolution
        // autocorr[lag] = sum(signal[i] * signal[i + lag])
        // For 'same' mode, we only compute lags up to n/2 to stay within bounds
        normalizedSignal.withUnsafeBufferPointer { signalPtr in
            let maxLag = n / 2
            for lag in 0..<maxLag {
                let validLength = n - lag
                var dotProduct: Float = 0.0

                vDSP_dotpr(
                    signalPtr.baseAddress!, 1,
                    signalPtr.baseAddress! + lag, 1,
                    &dotProduct,
                    vDSP_Length(validLength)
                )

                // Store in center-aligned output (like scipy 'same' mode)
                autocorr[n / 2 + lag] = dotProduct
            }

            // Mirror for negative lags
            for lag in 1..<(n / 2) {
                autocorr[n / 2 - lag] = autocorr[n / 2 + lag]
            }
        }

        return autocorr
    }

    // MARK: - Peak Finding with Zoom

    /// Find peak spacing using zoom-based grid search
    /// Matches Python's zoom-based optimization
    private func findPeakSpacing(autocorr: [Float]) -> Double {
        // Initial search range in pixels (spacing = pixels per 5mm grid square)
        let minInitialSpacing = minPixelsPerMm * standardGridSpacingMm  // 10 pixels minimum
        let maxInitialSpacing = maxPixelsPerMm * standardGridSpacingMm  // 250 pixels maximum

        var minSpacing = minInitialSpacing
        var maxSpacing = maxInitialSpacing

        // Zoom in over multiple iterations
        for _ in 0..<zoomIterations {
            let spacing = findPeakInRange(
                autocorr: autocorr,
                minSpacing: minSpacing,
                maxSpacing: maxSpacing,
                numSamples: 50
            )

            // Zoom in: narrow the search range by 50%
            let range = maxSpacing - minSpacing
            let newMin = spacing - range * 0.25
            let newMax = spacing + range * 0.25

            // Ensure we never go below minimum or above maximum
            minSpacing = max(minInitialSpacing, newMin)
            maxSpacing = min(maxInitialSpacing, newMax)

            // Prevent inverted range
            if minSpacing >= maxSpacing {
                minSpacing = minInitialSpacing
                maxSpacing = maxInitialSpacing
                break
            }
        }

        // Final high-resolution search
        return findPeakInRange(
            autocorr: autocorr,
            minSpacing: minSpacing,
            maxSpacing: maxSpacing,
            numSamples: 100
        )
    }

    /// Find the spacing that maximizes autocorrelation peak
    private func findPeakInRange(
        autocorr: [Float],
        minSpacing: Double,
        maxSpacing: Double,
        numSamples: Int
    ) -> Double {
        let centerIndex = autocorr.count / 2

        // Ensure valid search range
        let safeMinSpacing = max(1.0, minSpacing)
        let safeMaxSpacing = max(safeMinSpacing + 1.0, maxSpacing)

        var bestSpacing = safeMinSpacing
        var bestScore: Float = -Float.infinity

        for i in 0..<numSamples {
            let t = Double(i) / Double(numSamples - 1)
            let spacing = safeMinSpacing + t * (safeMaxSpacing - safeMinSpacing)

            // Check autocorrelation at this lag
            let lagIndex = centerIndex + Int(spacing.rounded())
            guard lagIndex > 0 && lagIndex < autocorr.count else { continue }

            let score = autocorr[lagIndex]

            if score > bestScore {
                bestScore = score
                bestSpacing = spacing
            }
        }

        // Ensure we return a positive value
        return max(1.0, bestSpacing)
    }

    // MARK: - Quality Score

    /// Calculate quality score from autocorrelation peak height
    /// Higher peaks indicate clearer grid structure
    private func calculateQualityScore(autocorr: [Float], peakSpacing: Double) -> Double {
        let centerIndex = autocorr.count / 2
        let peakIndex = centerIndex + Int(peakSpacing.rounded())

        guard peakIndex < autocorr.count else { return 0.5 }

        let peakValue = autocorr[peakIndex]
        let centerValue = autocorr[centerIndex]

        // Normalize: peak ratio relative to zero-lag autocorrelation
        guard centerValue > 0 else { return 0.5 }

        let ratio = Double(peakValue / centerValue)

        // Map ratio to 0-1 quality score
        // Good grids typically have ratio > 0.3
        let quality = min(1.0, max(0.0, (ratio - 0.1) / 0.4))

        return quality
    }
}

// MARK: - Errors

enum GridFinderError: LocalizedError {
    case calibrationOutOfRange(Double)
    case noPeakFound

    var errorDescription: String? {
        switch self {
        case .calibrationOutOfRange(let pixelsPerMm):
            return "Grid calibration out of expected range: \(pixelsPerMm) pixels/mm"
        case .noPeakFound:
            return "No autocorrelation peak found in grid probability map"
        }
    }
}
