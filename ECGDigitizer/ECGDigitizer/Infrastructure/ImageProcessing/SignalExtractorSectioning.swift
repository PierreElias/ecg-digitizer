//
//  SignalExtractorSectioning.swift
//  ECGDigitizer
//
//  Implements lead sectioning algorithm (Python's extract_leads_by_sectioning)
//  Extracts 12 ECG leads from probability maps using 3×4 grid sectioning
//

import Foundation
import UIKit

/// Sectioning-based lead extraction for 3×4 ECG layouts
class SignalExtractorSectioning {

    // MARK: - Configuration

    private let targetSamples: Int = 5000
    private let samplingFrequency: Double = 500.0
    private let voltageRange: Double = 2.0  // ±2mV per row

    // MARK: - Lead Mapping (3×4 Grid)

    private let leadMapping: [[String]] = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ]

    // MARK: - Main Entry Point

    /// Extract 12-15 leads using sectioning algorithm
    /// - Parameters:
    ///   - signalProb: Signal probability map (flattened row-major)
    ///   - gridProb: Grid probability map (flattened row-major)
    ///   - width: Image width
    ///   - height: Image height
    ///   - calibration: Grid calibration parameters
    /// - Returns: Array of 12-15 ECGLead objects (12 base + 0-3 rhythm leads)
    func extractLeads(
        signalProb: [Float],
        gridProb: [Float],
        width: Int,
        height: Int,
        calibration: GridCalibration
    ) throws -> [ECGLead] {
        print("  [Sectioning] Starting lead extraction")
        print("  [Sectioning] Image size: \(width)×\(height)")

        // Step 6a-b: Detect all row boundaries (main + rhythm)
        let (mainBoundaries, rhythmBoundaries) = detectAllRowBoundaries(
            signalProb: signalProb,
            width: width,
            height: height
        )

        print("  [Sectioning] Detected \(mainBoundaries.count) main rows, \(rhythmBoundaries.count) rhythm strips")
        for (idx, boundary) in mainBoundaries.enumerated() {
            print("    Main row \(idx): y=[\(boundary.top), \(boundary.bottom)]")
        }
        for (idx, boundary) in rhythmBoundaries.enumerated() {
            print("    Rhythm strip \(idx): y=[\(boundary.top), \(boundary.bottom)]")
        }

        // Step 6c: Extract full-width signal lines for main rows
        let rawLines = extractFullWidthLines(
            signalProb: signalProb,
            width: width,
            height: height,
            boundaries: mainBoundaries
        )

        print("  [Sectioning] Extracted \(rawLines.count) full-width lines")

        // Step 6d: Convert to voltage
        let voltageLines = convertToVoltage(
            rawLines: rawLines,
            boundaries: mainBoundaries,
            height: height
        )

        // Step 7: Split rows into 4 columns
        var leads = splitIntoLeads(
            voltageLines: voltageLines,
            width: width
        )

        print("  [Sectioning] Split into \(leads.count) leads")

        // Step 7b: Extract rhythm strips (if present)
        if !rhythmBoundaries.isEmpty {
            let rhythmLines = extractFullWidthLines(
                signalProb: signalProb,
                width: width,
                height: height,
                boundaries: rhythmBoundaries
            )

            let rhythmVoltages = convertToVoltage(
                rawLines: rhythmLines,
                boundaries: rhythmBoundaries,
                height: height
            )

            // Add rhythm leads (R1, R2, R3)
            let rhythmTypes: [LeadType] = [.R1, .R2, .R3]
            for (idx, rhythmVoltage) in rhythmVoltages.enumerated() {
                guard idx < rhythmTypes.count else { break }

                let fullWidthSamples = VectorMathUtilities.resample(
                    rhythmVoltage,
                    targetLength: targetSamples
                )

                leads.append(ECGLead(
                    type: rhythmTypes[idx],
                    samples: fullWidthSamples,
                    samplingRate: samplingFrequency
                ))

                print("  [Sectioning] Added rhythm lead \(rhythmTypes[idx].rawValue)")
            }
        }

        // Step 5d: Apply baseline wander removal
        let correctedLeads = applyBaselineRemoval(leads: leads)

        print("  [Sectioning] Applied baseline wander removal, total leads: \(correctedLeads.count)")

        return correctedLeads
    }

    // MARK: - Step 6a-b: Row Boundary Detection

    /// Detect all row boundaries, separating main rows from rhythm strips
    /// - Returns: Tuple of (main row boundaries, rhythm strip boundaries)
    private func detectAllRowBoundaries(
        signalProb: [Float],
        width: Int,
        height: Int
    ) -> (main: [(top: Int, bottom: Int)], rhythm: [(top: Int, bottom: Int)]) {
        // Sum signal probability across each row
        var rowSums = [Float](repeating: 0, count: height)
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if idx < signalProb.count {
                    rowSums[y] += signalProb[idx]
                }
            }
        }

        // Gaussian smoothing (adaptive sigma - larger for noisy images)
        let sigma = max(5.0, Float(height) / 25.0)
        let smoothed = VectorMathUtilities.gaussianFilter1D(rowSums, sigma: sigma)

        // Find peaks - use distance appropriate for up to 6 rows
        let distance = max(15, height / 7)
        let maxValue = smoothed.max() ?? 1.0
        let prominence = maxValue * 0.05

        let (peaks, _) = VectorMathUtilities.findPeaks(
            in: smoothed,
            distance: distance,
            prominence: prominence,
            height: nil
        )

        print("  [Sectioning] Peak detection found \(peaks.count) peaks at: \(peaks)")

        // Classify peaks into main rows (top ~55%) and rhythm strips (bottom ~45%)
        let mainAreaThreshold = Int(Float(height) * 0.55)

        var mainPeaks: [Int] = []
        var rhythmPeaks: [Int] = []

        for peak in peaks {
            if peak < mainAreaThreshold {
                mainPeaks.append(peak)
            } else {
                rhythmPeaks.append(peak)
            }
        }

        // Sort main peaks by signal strength, take top 3
        mainPeaks = mainPeaks.sorted { smoothed[$0] > smoothed[$1] }
        if mainPeaks.count > 3 {
            mainPeaks = Array(mainPeaks.prefix(3))
        }
        mainPeaks.sort()  // Sort by position

        // Sort rhythm peaks by position, take up to 3
        rhythmPeaks.sort()
        if rhythmPeaks.count > 3 {
            rhythmPeaks = Array(rhythmPeaks.prefix(3))
        }

        print("  [Sectioning] Main peaks: \(mainPeaks)")
        print("  [Sectioning] Rhythm peaks: \(rhythmPeaks)")

        // Fallback for main rows if not enough detected
        if mainPeaks.count < 3 {
            print("  [Sectioning] ⚠️ Only found \(mainPeaks.count) main peaks, using equal division")
            let mainHeight = Int(Float(height) * 0.5)
            mainPeaks = [mainHeight / 6, mainHeight / 2, 5 * mainHeight / 6]
        }

        // Calculate row heights
        let totalRows = mainPeaks.count + rhythmPeaks.count
        let rowSpacing = totalRows > 0 ? height / max(totalRows, 1) : height / 3
        let rowHeight = Int(Float(rowSpacing) * 0.85)

        // Convert peaks to boundaries
        let mainBoundaries = mainPeaks.map { peak -> (top: Int, bottom: Int) in
            let yMin = max(0, peak - rowHeight / 2)
            let yMax = min(height, peak + rowHeight / 2)
            return (yMin, yMax)
        }

        let rhythmBoundaries = rhythmPeaks.map { peak -> (top: Int, bottom: Int) in
            let yMin = max(0, peak - rowHeight / 2)
            let yMax = min(height, peak + rowHeight / 2)
            return (yMin, yMax)
        }

        return (mainBoundaries, rhythmBoundaries)
    }

    // MARK: - Step 6c: Full-Width Centroid Extraction

    /// Extract full-width signal lines using weighted centroid method
    private func extractFullWidthLines(
        signalProb: [Float],
        width: Int,
        height: Int,
        boundaries: [(top: Int, bottom: Int)]
    ) -> [[Float]] {
        var rawLines: [[Float]] = []

        for (rowIdx, boundary) in boundaries.enumerated() {
            var line = [Float](repeating: Float.nan, count: width)
            let y_min = boundary.top
            let y_max = boundary.bottom
            let rowHeight = y_max - y_min

            guard rowHeight > 0 else {
                rawLines.append(line)
                continue
            }

            for x in 0..<width {
                // Extract column probabilities within this row
                var colProbs = [Float](repeating: 0, count: rowHeight)
                var maxProb: Float = 0
                var maxProbY: Int = rowHeight / 2

                for localY in 0..<rowHeight {
                    let globalY = y_min + localY
                    if globalY < height {
                        let idx = globalY * width + x
                        if idx < signalProb.count {
                            let prob = signalProb[idx]
                            colProbs[localY] = prob
                            if prob > maxProb {
                                maxProb = prob
                                maxProbY = localY
                            }
                        }
                    }
                }

                // Compute weighted centroid: centroid = Σ(y * prob) / Σ(prob)
                var weightedSum: Float = 0
                var totalProb: Float = 0

                for (localY, prob) in colProbs.enumerated() {
                    weightedSum += Float(localY) * prob
                    totalProb += prob
                }

                // Only assign centroid if sufficient signal
                if totalProb > 0.01 {
                    let centroid = weightedSum / totalProb

                    // For tall QRS complexes (high vertical spread), use max probability location
                    // instead of centroid to avoid jumping between top/bottom of QRS
                    let verticalSpread = calculateVerticalSpread(colProbs)
                    let spreadThreshold = Float(rowHeight) * 0.4  // 40% of row height

                    if verticalSpread > spreadThreshold && maxProb > 0.3 {
                        // Use peak location for tall QRS complexes
                        line[x] = Float(maxProbY)
                    } else {
                        line[x] = centroid
                    }
                }
            }

            rawLines.append(line)

            // Calculate coverage for debugging
            let validCount = line.filter { !$0.isNaN }.count
            let coverage = Float(validCount) / Float(width) * 100
            print("    Row \(rowIdx): \(validCount)/\(width) pixels valid (\(String(format: "%.1f", coverage))%)")
        }

        return rawLines
    }

    // MARK: - Step 6d: Voltage Conversion

    /// Convert raw Y-positions to voltage values with baseline removal
    private func convertToVoltage(
        rawLines: [[Float]],
        boundaries: [(top: Int, bottom: Int)],
        height: Int
    ) -> [[Double]] {
        var voltageLines: [[Double]] = []

        for (idx, line) in rawLines.enumerated() {
            // Check if line has enough valid data
            let validMask = line.map { !$0.isNaN }
            let validCount = validMask.filter { $0 }.count

            guard validCount > 10 else {
                print("    Row \(idx): ⚠️ Insufficient valid pixels (\(validCount)), filling with zeros")
                voltageLines.append([Double](repeating: 0, count: line.count))
                continue
            }

            // Compute baseline (mean Y position of valid pixels)
            let validValues = line.enumerated()
                .filter { !$0.element.isNaN }
                .map { Double($0.element) }

            let baseline = validValues.reduce(0, +) / Double(validValues.count)

            // Convert to voltage
            let rowHeight = boundaries[idx].bottom - boundaries[idx].top
            let scaleFactor = voltageRange / (Double(rowHeight) / 2.0)

            var voltage = line.map { value -> Double in
                if value.isNaN {
                    return Double.nan
                }
                let offset = Double(value) - baseline
                return -offset * scaleFactor  // Invert Y-axis (down = positive voltage)
            }

            // Interpolate missing values
            voltage = VectorMathUtilities.linearInterpolate(
                values: voltage,
                mask: validMask,
                outputLength: voltage.count
            )

            voltageLines.append(voltage)

            // Calculate voltage range for debugging
            let validVoltages = voltage.filter { !$0.isNaN }
            if !validVoltages.isEmpty {
                let minV = validVoltages.min() ?? 0
                let maxV = validVoltages.max() ?? 0
                print("    Row \(idx): Voltage range [\(String(format: "%.2f", minV)), \(String(format: "%.2f", maxV))] mV")
            }
        }

        return voltageLines
    }

    // MARK: - Step 7: Split Into Leads (3×4 Grid)

    /// Split each row into 4 columns to create 12 leads
    private func splitIntoLeads(
        voltageLines: [[Double]],
        width: Int
    ) -> [ECGLead] {
        let cols = 4
        let samplesPerSection = targetSamples / cols  // 1250 samples per column
        var leads: [ECGLead] = []

        for (rowIdx, leadNames) in leadMapping.enumerated() {
            guard rowIdx < voltageLines.count else {
                print("  [Sectioning] ⚠️ Missing row \(rowIdx)")
                continue
            }

            let samples = voltageLines[rowIdx]
            let colWidth = samples.count / cols

            for (colIdx, leadName) in leadNames.enumerated() {
                // Extract column section
                let start = colIdx * colWidth
                let end = (colIdx < cols - 1) ? (colIdx + 1) * colWidth : samples.count
                let section = Array(samples[start..<end])

                // Resample to target length (1250 samples)
                let resampled = VectorMathUtilities.resample(
                    section,
                    targetLength: samplesPerSection
                )

                // Build full 5000-sample array
                var fullSamples = [Double](repeating: 0, count: targetSamples)
                let startIdx = colIdx * samplesPerSection
                fullSamples.replaceSubrange(startIdx..<(startIdx + samplesPerSection), with: resampled)

                // Create lead
                guard let leadType = LeadType(rawValue: leadName) else {
                    print("  [Sectioning] ⚠️ Unknown lead name: \(leadName)")
                    continue
                }

                leads.append(ECGLead(
                    type: leadType,
                    samples: fullSamples,
                    samplingRate: samplingFrequency
                ))
            }
        }

        return leads
    }

    // MARK: - Step 5d: Baseline Wander Removal

    /// Apply baseline wander removal using median filtering
    private func applyBaselineRemoval(leads: [ECGLead]) -> [ECGLead] {
        return leads.map { lead in
            let corrected = removeBaselineWander(
                from: lead.samples,
                samplingFrequency: lead.samplingRate
            )

            return ECGLead(
                type: lead.type,
                samples: corrected,
                samplingRate: lead.samplingRate
            )
        }
    }

    /// Two-pass median filtering for baseline wander removal
    private func removeBaselineWander(
        from samples: [Double],
        samplingFrequency: Double
    ) -> [Double] {
        // Two-pass median filtering
        let winSize1 = Int(round(0.2 * samplingFrequency))  // 0.2s window = 100 samples @ 500Hz
        let winSize2 = Int(round(0.6 * samplingFrequency))  // 0.6s window = 300 samples @ 500Hz

        // First pass
        var baseline = VectorMathUtilities.medianFilter(samples, windowSize: winSize1)

        // Second pass
        baseline = VectorMathUtilities.medianFilter(baseline, windowSize: winSize2)

        // Subtract baseline
        var corrected = [Double](repeating: 0, count: samples.count)
        for i in 0..<samples.count {
            corrected[i] = samples[i] - baseline[i]
        }

        return corrected
    }

    // MARK: - Helper Functions

    /// Calculate vertical spread of signal probability in a column
    /// Returns the distance between the highest and lowest significant probability positions
    private func calculateVerticalSpread(_ probs: [Float]) -> Float {
        let threshold: Float = 0.1  // Minimum probability to consider

        var minY: Int?
        var maxY: Int?

        for (y, prob) in probs.enumerated() {
            if prob > threshold {
                if minY == nil { minY = y }
                maxY = y
            }
        }

        guard let min = minY, let max = maxY else {
            return 0
        }

        return Float(max - min)
    }
}
