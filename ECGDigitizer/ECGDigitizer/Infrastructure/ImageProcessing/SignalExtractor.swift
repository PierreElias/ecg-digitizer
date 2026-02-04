import Foundation
import Accelerate

/// Extracts ECG waveforms from signal probability maps
///
/// This class implements the centroid-based signal extraction algorithm from the
/// Open-ECG-Digitizer project, converting 2D probability maps to 1D voltage signals.
///
/// Algorithm overview:
/// 1. For each column in the probability map, compute weighted centroid
/// 2. Weighted centroid = sum(probability[y] * y) / sum(probability[y])
/// 3. Convert pixel Y-positions to voltage using grid calibration
/// 4. Handle multiple leads by detecting horizontal regions
class SignalExtractor {

    // MARK: - Configuration

    /// Minimum probability threshold for signal detection
    private let probabilityThreshold: Float = 0.1

    /// Minimum consecutive valid samples to consider a lead
    private let minLeadWidth: Int = 100

    /// Standard ECG calibration: 10mm = 1mV
    private let mvPerMm: Double = 0.1

    // MARK: - Extraction

    /// Extract ECG waveforms from signal probability map
    ///
    /// - Parameters:
    ///   - signalProb: 2D array of signal probabilities (height × width)
    ///   - width: Width of the probability map
    ///   - height: Height of the probability map
    ///   - calibration: Grid calibration for voltage conversion
    /// - Returns: Array of waveforms, one per detected lead
    func extractWaveforms(
        signalProb: [Float],
        width: Int,
        height: Int,
        calibration: GridCalibration
    ) throws -> [[Double]] {

        // Debug: Log signal probability statistics
        let signalStats = signalProb.filter { $0 > 0.01 }
        print("📊 SignalExtractor: Input \(width)x\(height), non-zero pixels: \(signalStats.count)/\(signalProb.count)")
        if !signalStats.isEmpty {
            print("   Min: \(signalStats.min()!), Max: \(signalStats.max()!), Mean: \(signalStats.reduce(0, +) / Float(signalStats.count))")
        }

        // Step 1: Extract raw signal lines (Y-positions) using centroid method
        let rawLines = extractSignalLines(
            signalProb: signalProb,
            width: width,
            height: height
        )

        print("📊 SignalExtractor: Extracted \(rawLines.count) raw lines")
        for (i, line) in rawLines.enumerated() {
            let validCount = line.filter { !$0.isNaN }.count
            print("   Line \(i): \(validCount)/\(line.count) valid samples")
        }

        guard !rawLines.isEmpty else {
            throw SignalExtractionError.noSignalsDetected
        }

        // Step 2: Convert pixel positions to voltage
        let voltageSignals = convertToVoltage(
            lines: rawLines,
            calibration: calibration,
            imageHeight: height
        )

        // Step 3: Resample to standard sample rate (500 Hz, 10 seconds = 5000 samples)
        let resampledSignals = resampleSignals(
            signals: voltageSignals,
            targetLength: 5000
        )

        print("📊 SignalExtractor: Output \(resampledSignals.count) leads, each \(resampledSignals.first?.count ?? 0) samples")

        return resampledSignals
    }

    // MARK: - Centroid Extraction

    /// Extract signal lines using weighted centroid method
    ///
    /// For each column:
    /// - Compute weighted average Y-position using probabilities as weights
    /// - Y_centroid = sum(P[y] * y) / sum(P[y])
    ///
    /// - Parameters:
    ///   - signalProb: Flattened probability map (row-major: y * width + x)
    ///   - width: Image width
    ///   - height: Image height
    /// - Returns: Array of signal lines, each line is array of Y-positions
    private func extractSignalLines(
        signalProb: [Float],
        width: Int,
        height: Int
    ) -> [[Float]] {

        // Extract centroids for each column
        var centroids = [Float](repeating: Float.nan, count: width)

        for x in 0..<width {
            // Extract column probabilities
            var columnProbs = [Float](repeating: 0, count: height)
            for y in 0..<height {
                columnProbs[y] = signalProb[y * width + x]
            }

            // Compute weighted centroid
            if let centroid = computeWeightedCentroid(
                probabilities: columnProbs,
                threshold: probabilityThreshold
            ) {
                centroids[x] = centroid
            }
        }

        // Always use grid-based extraction for standard 3x4 ECG layouts
        // The gap-based detection only works for vertically-stacked rhythm strips
        let lines = extractLeadsByHorizontalSlicing(
            centroids: centroids,
            imageHeight: Float(height)
        )

        return lines
    }

    /// Compute weighted centroid for a column
    ///
    /// - Parameters:
    ///   - probabilities: Probability values for each Y position
    ///   - threshold: Minimum probability to consider
    /// - Returns: Weighted average Y-position, or nil if insufficient signal
    private func computeWeightedCentroid(
        probabilities: [Float],
        threshold: Float
    ) -> Float? {

        var weightedSum: Float = 0.0
        var totalWeight: Float = 0.0

        for (y, prob) in probabilities.enumerated() {
            if prob > threshold {
                weightedSum += Float(y) * prob
                totalWeight += prob
            }
        }

        guard totalWeight > 0.01 else { return nil }

        return weightedSum / totalWeight
    }

    /// Split centroids into individual lead lines
    ///
    /// ECGs typically have 3-4 rows of leads. This method detects gaps
    /// and splits the continuous centroid array into separate leads.
    ///
    /// - Parameters:
    ///   - centroids: Array of Y-positions (may contain NaN for gaps)
    ///   - imageHeight: Total image height for normalization
    /// - Returns: Array of lead lines
    private func splitIntoLeads(centroids: [Float], imageHeight: Float) -> [[Float]] {
        var leads: [[Float]] = []
        var currentLead: [Float] = []
        var consecutiveValid = 0

        for centroid in centroids {
            if centroid.isNaN {
                // Gap in signal
                if consecutiveValid >= minLeadWidth {
                    leads.append(currentLead)
                }
                currentLead = []
                consecutiveValid = 0
            } else {
                currentLead.append(centroid)
                consecutiveValid += 1
            }
        }

        // Add final lead if valid
        if consecutiveValid >= minLeadWidth {
            leads.append(currentLead)
        }

        // If we detected too few leads, fall back to horizontal slicing
        if leads.count < 3 {
            return extractLeadsByHorizontalSlicing(
                centroids: centroids,
                imageHeight: imageHeight
            )
        }

        return leads
    }

    /// Fallback method: Extract leads by dividing image into a 3×4 grid
    ///
    /// Assumes standard 3×4 layout: 3 rows of 4 leads each
    /// Each cell contains one lead's waveform
    ///
    /// - Parameters:
    ///   - centroids: Array of Y-positions for each column
    ///   - imageHeight: Total image height
    /// - Returns: Array of 12 lead lines
    private func extractLeadsByHorizontalSlicing(
        centroids: [Float],
        imageHeight: Float
    ) -> [[Float]] {

        // Standard 3×4 ECG layout: 3 rows, 4 columns
        let numRows = 3
        let numCols = 4
        let rowHeight = imageHeight / Float(numRows)
        let colWidth = centroids.count / numCols

        var leads: [[Float]] = []

        // Extract leads in standard ECG order (row by row, left to right)
        // Row 1: I, aVR, V1, V4
        // Row 2: II, aVL, V2, V5
        // Row 3: III, aVF, V3, V6
        for row in 0..<numRows {
            let yMin = Float(row) * rowHeight
            let yMax = Float(row + 1) * rowHeight

            for col in 0..<numCols {
                let xMin = col * colWidth
                let xMax = (col + 1) * colWidth

                // Extract centroids within this cell
                var leadLine: [Float] = []
                for x in xMin..<min(xMax, centroids.count) {
                    let centroid = centroids[x]
                    // Only include if centroid is in this row's Y range
                    if !centroid.isNaN && centroid >= yMin && centroid < yMax {
                        // Normalize Y position relative to row center
                        let normalizedY = centroid - (yMin + yMax) / 2
                        leadLine.append(normalizedY)
                    } else {
                        leadLine.append(Float.nan)
                    }
                }

                // Only add if we have some valid data
                let validCount = leadLine.filter { !$0.isNaN }.count
                if validCount > minLeadWidth / 4 {
                    leads.append(leadLine)
                } else {
                    // Add empty lead
                    leads.append([Float](repeating: Float.nan, count: max(1, colWidth)))
                }
            }
        }

        // Ensure we have exactly 12 leads
        while leads.count < 12 {
            leads.append([Float](repeating: Float.nan, count: max(1, colWidth)))
        }

        return Array(leads.prefix(12))
    }

    // MARK: - Voltage Conversion

    /// Convert pixel Y-positions to voltage values
    ///
    /// Conversion formula:
    /// voltage_µV = (pixel_y - baseline) * (mv_per_mm / pixels_per_mm) * 1000
    ///
    /// - Parameters:
    ///   - lines: Array of signal lines (pixel Y-positions)
    ///   - calibration: Grid calibration
    ///   - imageHeight: Image height for baseline calculation
    /// - Returns: Array of voltage signals in microvolts
    private func convertToVoltage(
        lines: [[Float]],
        calibration: GridCalibration,
        imageHeight: Int
    ) -> [[Double]] {

        let pixelsPerMm = Double(calibration.smallSquareHeightPixels)

        return lines.map { line in
            // Remove baseline (center the signal)
            let validValues = line.filter { !$0.isNaN }
            guard !validValues.isEmpty else {
                return [Double](repeating: 0.0, count: line.count)
            }

            let baseline = validValues.reduce(0.0, +) / Float(validValues.count)

            // Convert to voltage
            return line.map { pixel in
                guard !pixel.isNaN else { return 0.0 }

                let centeredPixel = Double(pixel - baseline)
                // Invert Y-axis (higher pixel = more negative voltage)
                let voltage_mV = -centeredPixel * (mvPerMm / pixelsPerMm)
                let voltage_µV = voltage_mV * 1000.0

                return voltage_µV
            }
        }
    }

    // MARK: - Resampling

    /// Resample signals to target length using linear interpolation
    ///
    /// ECG standard is 500 Hz sample rate. For 10-second recording = 5000 samples.
    ///
    /// - Parameters:
    ///   - signals: Input voltage signals
    ///   - targetLength: Desired output length
    /// - Returns: Resampled signals
    private func resampleSignals(
        signals: [[Double]],
        targetLength: Int
    ) -> [[Double]] {

        return signals.map { signal in
            guard signal.count > 1 else {
                // Too short to resample, pad with zeros
                return [Double](repeating: 0.0, count: targetLength)
            }

            return resampleLinear(signal: signal, targetLength: targetLength)
        }
    }

    /// Linear interpolation resampling using vDSP
    ///
    /// - Parameters:
    ///   - signal: Input signal
    ///   - targetLength: Output length
    /// - Returns: Resampled signal
    private func resampleLinear(signal: [Double], targetLength: Int) -> [Double] {
        guard signal.count != targetLength else { return signal }

        let scale = Double(signal.count - 1) / Double(targetLength - 1)
        var resampled = [Double](repeating: 0.0, count: targetLength)

        for i in 0..<targetLength {
            let srcIndex = Double(i) * scale
            let i0 = Int(srcIndex)
            let i1 = min(i0 + 1, signal.count - 1)
            let fraction = srcIndex - Double(i0)

            // Linear interpolation
            resampled[i] = signal[i0] * (1.0 - fraction) + signal[i1] * fraction
        }

        return resampled
    }
}

// MARK: - Error Types

enum SignalExtractionError: LocalizedError {
    case noSignalsDetected
    case insufficientSignalQuality
    case invalidCalibration

    var errorDescription: String? {
        switch self {
        case .noSignalsDetected:
            return "No ECG signals detected in the image"
        case .insufficientSignalQuality:
            return "Signal quality too low for extraction"
        case .invalidCalibration:
            return "Invalid grid calibration parameters"
        }
    }
}
