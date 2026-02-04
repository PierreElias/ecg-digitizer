import Foundation
import Accelerate

/// Advanced signal extraction using connected components and Hungarian algorithm
///
/// This implements the Python signal_extractor.py algorithm:
/// 1. Connected component labeling to find signal regions
/// 2. Weighted centroid extraction per column
/// 3. Iterative refinement for overlapping traces
/// 4. Hungarian algorithm for line matching and merging
/// 5. Output 12 leads with proper ordering
class SignalExtractorAdvanced {

    // MARK: - Configuration

    /// Minimum probability threshold for signal detection
    private let probabilityThreshold: Float = 0.1

    /// Minimum width (pixels) for a valid lead segment
    private let minSegmentWidth: Int = 50

    /// Maximum vertical gap to consider lines as same lead
    private let maxVerticalGap: Float = 20.0

    /// Weight for horizontal distance in cost matrix
    private let horizontalWeight: Float = 1.0

    /// Weight for vertical distance in cost matrix
    private let verticalWeight: Float = 30.0

    // MARK: - Extraction

    /// Extract 12 ECG leads from signal probability map
    /// - Parameters:
    ///   - signalProb: Signal probability map (height × width)
    ///   - width: Image width
    ///   - height: Image height
    ///   - calibration: Grid calibration for voltage conversion
    /// - Returns: 12 leads as voltage arrays
    func extractLeads(
        signalProb: [Float],
        width: Int,
        height: Int,
        calibration: GridCalibration
    ) throws -> [[Double]] {

        print("📊 SignalExtractorAdvanced: Processing \(width)x\(height) image")

        // Step 1: Label connected components
        let labels = ConnectedComponentLabeler.label(
            mask: signalProb,
            width: width,
            height: height,
            threshold: probabilityThreshold
        )

        let boxes = ConnectedComponentLabeler.getBoundingBoxes(
            labels: labels,
            width: width,
            height: height
        )

        print("   Found \(boxes.count) connected components")

        // Step 2: Extract candidate lines from each component
        var candidateLines: [CandidateLine] = []

        for (labelId, box) in boxes {
            let line = extractLineFromComponent(
                signalProb: signalProb,
                labels: labels,
                labelId: labelId,
                box: box,
                width: width,
                height: height
            )

            if line.validCount >= minSegmentWidth {
                candidateLines.append(line)
            }
        }

        print("   Extracted \(candidateLines.count) candidate lines")

        // Step 3: Compute cost matrix for line matching
        let costMatrix = computeCostMatrix(lines: candidateLines, imageWidth: width)

        // Step 4: Run Hungarian algorithm to find optimal matching
        let assignments = HungarianAlgorithm.solve(costMatrix: costMatrix)

        // Step 5: Build match graph and find connected components
        let matchedLines = mergeMatchedLines(
            lines: candidateLines,
            assignments: assignments,
            imageWidth: width
        )

        print("   Merged into \(matchedLines.count) leads")

        // Step 6: Convert to 12 leads with proper layout
        let leads = assignToStandardLayout(
            lines: matchedLines,
            imageHeight: height,
            imageWidth: width
        )

        // Step 7: Convert to voltage
        let voltageLeads = convertToVoltage(
            leads: leads,
            calibration: calibration,
            imageHeight: height
        )

        // Step 8: Resample to standard length
        let resampledLeads = resampleToStandardLength(voltageLeads, targetLength: 5000)

        return resampledLeads
    }

    // MARK: - Component Extraction

    private struct CandidateLine {
        var centroids: [Float]  // Y-position per X
        var startX: Int
        var endX: Int
        var validCount: Int
        var meanY: Float

        var startY: Float { centroids.first { !$0.isNaN } ?? 0 }
        var endY: Float { centroids.last { !$0.isNaN } ?? 0 }
    }

    /// Extract a line from a connected component using weighted centroids
    private func extractLineFromComponent(
        signalProb: [Float],
        labels: [Int],
        labelId: Int,
        box: (minX: Int, minY: Int, maxX: Int, maxY: Int),
        width: Int,
        height: Int
    ) -> CandidateLine {

        var centroids = [Float](repeating: Float.nan, count: box.maxX - box.minX + 1)
        var validCount = 0
        var sumY: Float = 0

        for x in box.minX...box.maxX {
            var weightedSum: Float = 0
            var totalWeight: Float = 0

            for y in box.minY...box.maxY {
                let idx = y * width + x
                if labels[idx] == labelId {
                    let prob = signalProb[idx]
                    weightedSum += Float(y) * prob
                    totalWeight += prob
                }
            }

            if totalWeight > 0.01 {
                let centroid = weightedSum / totalWeight
                centroids[x - box.minX] = centroid
                validCount += 1
                sumY += centroid
            }
        }

        let meanY = validCount > 0 ? sumY / Float(validCount) : 0

        return CandidateLine(
            centroids: centroids,
            startX: box.minX,
            endX: box.maxX,
            validCount: validCount,
            meanY: meanY
        )
    }

    // MARK: - Cost Matrix

    /// Compute cost matrix for Hungarian algorithm
    /// Cost = horizontal_distance + vertical_weight * vertical_distance
    private func computeCostMatrix(lines: [CandidateLine], imageWidth: Int) -> [[Float]] {
        let n = lines.count
        guard n > 0 else { return [] }

        var costMatrix = [[Float]](repeating: [Float](repeating: Float.infinity, count: n), count: n)

        for i in 0..<n {
            for j in 0..<n {
                if i == j { continue }

                let line1 = lines[i]
                let line2 = lines[j]

                // Check if line2 could be a continuation of line1
                // (line2 should start after line1 ends)
                if line2.startX > line1.endX {
                    let horizontalGap = Float(line2.startX - line1.endX)
                    let verticalDiff = abs(line2.startY - line1.endY)

                    // Wrap-around distance for ECG rhythm strips
                    let wrappedHorizontal = min(horizontalGap, Float(imageWidth) - horizontalGap)

                    let cost = horizontalWeight * wrappedHorizontal +
                               verticalWeight * verticalDiff * verticalDiff

                    // Only match if vertical distance is reasonable
                    if verticalDiff < maxVerticalGap {
                        costMatrix[i][j] = cost
                    }
                }
            }
        }

        return costMatrix
    }

    // MARK: - Line Merging

    /// Merge matched lines using graph connected components
    private func mergeMatchedLines(
        lines: [CandidateLine],
        assignments: [(row: Int, col: Int)],
        imageWidth: Int
    ) -> [[Float]] {

        let n = lines.count
        guard n > 0 else { return [] }

        // Build adjacency list
        var adj = [[Int]](repeating: [], count: n)
        for (i, j) in assignments {
            if i < n && j < n {
                adj[i].append(j)
                adj[j].append(i)
            }
        }

        // Find connected components using DFS
        var visited = [Bool](repeating: false, count: n)
        var components: [[Int]] = []

        for i in 0..<n {
            if !visited[i] {
                var component: [Int] = []
                var stack = [i]

                while !stack.isEmpty {
                    let node = stack.removeLast()
                    if !visited[node] {
                        visited[node] = true
                        component.append(node)
                        stack.append(contentsOf: adj[node])
                    }
                }

                components.append(component)
            }
        }

        // Merge lines in each component
        var mergedLines: [[Float]] = []

        for component in components {
            // Sort lines by startX
            let sortedIndices = component.sorted { lines[$0].startX < lines[$1].startX }

            // Create merged line spanning full width
            var mergedLine = [Float](repeating: Float.nan, count: imageWidth)

            for idx in sortedIndices {
                let line = lines[idx]
                for (i, centroid) in line.centroids.enumerated() {
                    let x = line.startX + i
                    if x < imageWidth && !centroid.isNaN {
                        // Take first valid value (don't overwrite)
                        if mergedLine[x].isNaN {
                            mergedLine[x] = centroid
                        }
                    }
                }
            }

            mergedLines.append(mergedLine)
        }

        return mergedLines
    }

    // MARK: - Layout Assignment

    /// Assign merged lines to standard 12-lead layout
    /// Layout: 3 rows × 4 columns
    /// Row 1: I, aVR, V1, V4
    /// Row 2: II, aVL, V2, V5
    /// Row 3: III, aVF, V3, V6
    private func assignToStandardLayout(
        lines: [[Float]],
        imageHeight: Int,
        imageWidth: Int
    ) -> [[Float]] {

        // Sort lines by mean Y position (top to bottom)
        let sortedLines = lines.sorted { meanY($0) < meanY($1) }

        // Group into 3 rows
        let rowHeight = Float(imageHeight) / 3.0
        var rows: [[[Float]]] = [[], [], []]

        for line in sortedLines {
            let y = meanY(line)
            let rowIndex = min(2, Int(y / rowHeight))
            rows[rowIndex].append(line)
        }

        // Sort each row by X position (left to right)
        for i in 0..<3 {
            rows[i].sort { meanX($0, width: imageWidth) < meanX($1, width: imageWidth) }
        }

        // Build 12-lead output
        var leads: [[Float]] = []

        // Standard layout order: 3 rows × 4 columns, read row by row
        for row in rows {
            // Ensure 4 leads per row
            var rowLeads = row
            while rowLeads.count < 4 {
                rowLeads.append([Float](repeating: Float.nan, count: imageWidth / 4))
            }
            leads.append(contentsOf: rowLeads.prefix(4))
        }

        // Ensure exactly 12 leads
        while leads.count < 12 {
            leads.append([Float](repeating: Float.nan, count: imageWidth / 4))
        }

        return Array(leads.prefix(12))
    }

    private func meanY(_ line: [Float]) -> Float {
        let valid = line.filter { !$0.isNaN }
        return valid.isEmpty ? 0 : valid.reduce(0, +) / Float(valid.count)
    }

    private func meanX(_ line: [Float], width: Int) -> Float {
        var sumX: Float = 0
        var count: Float = 0
        for (x, y) in line.enumerated() {
            if !y.isNaN {
                sumX += Float(x)
                count += 1
            }
        }
        return count > 0 ? sumX / count : Float(width) / 2
    }

    // MARK: - Voltage Conversion

    private func convertToVoltage(
        leads: [[Float]],
        calibration: GridCalibration,
        imageHeight: Int
    ) -> [[Double]] {

        let pixelsPerMm = calibration.smallSquareHeightPixels
        let mvPerMm: Double = 0.1  // Standard ECG: 10mm = 1mV

        return leads.map { line in
            // Find baseline (median of valid values)
            let valid = line.filter { !$0.isNaN }
            guard !valid.isEmpty else {
                return [Double](repeating: 0, count: line.count)
            }

            let sorted = valid.sorted()
            let baseline = sorted[sorted.count / 2]

            // Convert to voltage
            return line.map { pixel in
                guard !pixel.isNaN else { return 0.0 }

                let centered = Double(pixel - baseline)
                // Invert Y (higher pixel = more negative voltage)
                let voltage_mV = -centered * mvPerMm / pixelsPerMm
                return voltage_mV * 1000.0  // Convert to µV
            }
        }
    }

    // MARK: - Resampling

    private func resampleToStandardLength(_ signals: [[Double]], targetLength: Int) -> [[Double]] {
        return signals.map { signal in
            guard signal.count > 1 else {
                return [Double](repeating: 0, count: targetLength)
            }

            let scale = Double(signal.count - 1) / Double(targetLength - 1)
            var resampled = [Double](repeating: 0, count: targetLength)

            for i in 0..<targetLength {
                let srcIdx = Double(i) * scale
                let i0 = Int(srcIdx)
                let i1 = min(i0 + 1, signal.count - 1)
                let frac = srcIdx - Double(i0)

                resampled[i] = signal[i0] * (1.0 - frac) + signal[i1] * frac
            }

            return resampled
        }
    }
}
