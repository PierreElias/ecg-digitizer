//
//  VectorMathUtilities.swift
//  ECGDigitizer
//
//  Centralized Accelerate/vDSP operations for signal processing
//  Provides optimized vector operations matching Python's scipy/numpy functions
//

import Foundation
import Accelerate

struct PeakProperties {
    let prominences: [Float]?
    let widths: [Float]?
}

struct VectorMathUtilities {

    // MARK: - Peak Detection (mirrors scipy.signal.find_peaks)

    /// Finds peaks in a 1D signal
    /// - Parameters:
    ///   - data: Input signal
    ///   - distance: Minimum horizontal distance between peaks (in samples)
    ///   - prominence: Minimum prominence of peaks
    ///   - height: Minimum height of peaks
    /// - Returns: Tuple of (peak indices, properties)
    static func findPeaks(
        in data: [Float],
        distance: Int? = nil,
        prominence: Float? = nil,
        height: Float? = nil
    ) -> (peaks: [Int], properties: PeakProperties) {
        guard data.count > 2 else { return ([], PeakProperties(prominences: nil, widths: nil)) }

        var candidatePeaks: [Int] = []

        // Step 1: Find all local maxima
        for i in 1..<(data.count - 1) {
            if data[i] > data[i - 1] && data[i] > data[i + 1] {
                candidatePeaks.append(i)
            }
        }

        // Step 2: Filter by height threshold
        if let minHeight = height {
            candidatePeaks = candidatePeaks.filter { data[$0] >= minHeight }
        }

        // Step 3: Filter by prominence
        var prominences: [Float]?
        if let minProminence = prominence {
            let peakProminences = candidatePeaks.map { calculateProminence(data: data, peakIndex: $0) }
            prominences = peakProminences

            let filtered = zip(candidatePeaks, peakProminences)
                .filter { $0.1 >= minProminence }
                .map { $0.0 }
            candidatePeaks = filtered
        }

        // Step 4: Filter by minimum distance
        if let minDistance = distance, minDistance > 0 {
            candidatePeaks = filterByDistance(peaks: candidatePeaks, data: data, minDistance: minDistance)
        }

        return (candidatePeaks, PeakProperties(prominences: prominences, widths: nil))
    }

    /// Calculate prominence of a peak (height difference to lowest contour)
    private static func calculateProminence(data: [Float], peakIndex: Int) -> Float {
        let peakValue = data[peakIndex]

        // Find lowest contour on left side
        var leftMin = peakValue
        for i in stride(from: peakIndex - 1, through: 0, by: -1) {
            if data[i] < leftMin {
                leftMin = data[i]
            }
            if data[i] > peakValue {
                break  // Found a higher peak
            }
        }

        // Find lowest contour on right side
        var rightMin = peakValue
        for i in (peakIndex + 1)..<data.count {
            if data[i] < rightMin {
                rightMin = data[i]
            }
            if data[i] > peakValue {
                break  // Found a higher peak
            }
        }

        // Prominence is height above the highest of the two bases
        let base = max(leftMin, rightMin)
        return peakValue - base
    }

    /// Filter peaks by minimum distance, keeping highest peaks
    private static func filterByDistance(peaks: [Int], data: [Float], minDistance: Int) -> [Int] {
        guard !peaks.isEmpty else { return [] }

        // Sort by peak height (descending)
        let sortedPeaks = peaks.sorted { data[$0] > data[$1] }

        var selected: [Int] = []
        for peak in sortedPeaks {
            // Check if this peak is far enough from all selected peaks
            let isFarEnough = selected.allSatisfy { abs($0 - peak) >= minDistance }
            if isFarEnough {
                selected.append(peak)
            }
        }

        // Return in original order
        return selected.sorted()
    }

    // MARK: - Gaussian Smoothing (mirrors scipy.ndimage.gaussian_filter1d)

    /// Apply Gaussian smoothing to 1D signal using convolution
    /// - Parameters:
    ///   - data: Input signal
    ///   - sigma: Standard deviation of Gaussian kernel
    /// - Returns: Smoothed signal
    static func gaussianFilter1D(_ data: [Float], sigma: Float) -> [Float] {
        guard data.count > 1, sigma > 0 else { return data }

        // Create Gaussian kernel
        let radius = max(1, Int(ceil(3.0 * sigma)))  // 3-sigma rule
        let kernelSize = 2 * radius + 1
        var kernel = [Float](repeating: 0, count: kernelSize)

        var sum: Float = 0
        for i in 0..<kernelSize {
            let x = Float(i - radius)
            let value = exp(-(x * x) / (2 * sigma * sigma))
            kernel[i] = value
            sum += value
        }

        // Normalize kernel
        vDSP_vsdiv(kernel, 1, [sum], &kernel, 1, vDSP_Length(kernelSize))

        // Pad data to handle boundaries (reflect mode)
        let paddedSize = data.count + 2 * radius
        var padded = [Float](repeating: 0, count: paddedSize)

        // Copy original data to center
        padded.replaceSubrange(radius..<(radius + data.count), with: data)

        // Reflect left boundary
        for i in 0..<radius {
            padded[i] = data[min(radius - i, data.count - 1)]
        }

        // Reflect right boundary
        for i in 0..<radius {
            padded[radius + data.count + i] = data[max(0, data.count - 2 - i)]
        }

        // Convolve
        var result = [Float](repeating: 0, count: data.count)
        vDSP_conv(padded, 1, kernel, 1, &result, 1, vDSP_Length(data.count), vDSP_Length(kernelSize))

        return result
    }

    // MARK: - Cosine Similarity (NaN-safe)

    /// Compute cosine similarity between two vectors, ignoring NaN values
    /// - Parameters:
    ///   - x: First vector
    ///   - y: Second vector
    /// - Returns: Cosine similarity in range [-1, 1], or NaN if no valid pairs
    static func nanSafeCosineSimilarity(_ x: [Double], _ y: [Double]) -> Double {
        guard x.count == y.count else { return Double.nan }

        // Find valid (non-NaN) pairs
        var validX: [Double] = []
        var validY: [Double] = []

        for i in 0..<x.count {
            if !x[i].isNaN && !y[i].isNaN && x[i].isFinite && y[i].isFinite {
                validX.append(x[i])
                validY.append(y[i])
            }
        }

        guard validX.count > 0 else { return Double.nan }

        // Compute dot product
        var dotProduct: Double = 0
        vDSP_dotprD(validX, 1, validY, 1, &dotProduct, vDSP_Length(validX.count))

        // Compute magnitudes
        var normX: Double = 0
        var normY: Double = 0
        vDSP_svesqD(validX, 1, &normX, vDSP_Length(validX.count))
        vDSP_svesqD(validY, 1, &normY, vDSP_Length(validY.count))

        normX = sqrt(normX)
        normY = sqrt(normY)

        // Avoid division by zero
        guard normX > 1e-10, normY > 1e-10 else { return 0.0 }

        return dotProduct / (normX * normY)
    }

    // MARK: - Median Filter

    /// Apply median filter for baseline wander removal
    /// - Parameters:
    ///   - data: Input signal
    ///   - windowSize: Size of median window (should be odd)
    /// - Returns: Median-filtered signal
    static func medianFilter(_ data: [Double], windowSize: Int) -> [Double] {
        guard data.count > 0 else { return [] }
        guard windowSize > 1 else { return data }

        let halfWindow = windowSize / 2
        var result = [Double](repeating: 0, count: data.count)
        var window = [Double](repeating: 0, count: windowSize)

        for i in 0..<data.count {
            // Extract window centered at i
            var validCount = 0
            for j in 0..<windowSize {
                let idx = i - halfWindow + j
                if idx >= 0 && idx < data.count {
                    window[validCount] = data[idx]
                    validCount += 1
                }
            }

            // Compute median of valid values
            if validCount > 0 {
                let validWindow = Array(window.prefix(validCount))
                result[i] = median(of: validWindow)
            } else {
                result[i] = data[i]
            }
        }

        return result
    }

    /// Compute median of an array
    private static func median(of values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }

        var sorted = values
        sorted.sort()

        let count = sorted.count
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        } else {
            return sorted[count / 2]
        }
    }

    // MARK: - Linear Interpolation

    /// Linearly interpolate missing (NaN) values in a signal
    /// - Parameters:
    ///   - values: Input signal with NaN gaps
    ///   - mask: Boolean mask (true = valid, false = NaN)
    ///   - outputLength: Length of output (usually same as input)
    /// - Returns: Interpolated signal
    static func linearInterpolate(
        values: [Double],
        mask: [Bool],
        outputLength: Int
    ) -> [Double] {
        guard values.count == mask.count else { return values }
        guard outputLength > 0 else { return [] }

        var result = [Double](repeating: 0, count: outputLength)

        // Find valid indices and values
        var validIndices: [Int] = []
        var validValues: [Double] = []

        for i in 0..<values.count {
            if mask[i] && !values[i].isNaN {
                validIndices.append(i)
                validValues.append(values[i])
            }
        }

        guard validIndices.count >= 2 else {
            // Not enough points for interpolation
            if validIndices.count == 1 {
                // Fill with constant
                result = [Double](repeating: validValues[0], count: outputLength)
            }
            return result
        }

        // Interpolate for each output index
        for i in 0..<outputLength {
            let x = Double(i) * Double(values.count - 1) / Double(max(1, outputLength - 1))

            // Find surrounding valid points
            if x <= Double(validIndices.first!) {
                result[i] = validValues.first!
            } else if x >= Double(validIndices.last!) {
                result[i] = validValues.last!
            } else {
                // Find bracketing points
                var leftIdx = 0
                var rightIdx = validIndices.count - 1

                for j in 0..<(validIndices.count - 1) {
                    if Double(validIndices[j]) <= x && x <= Double(validIndices[j + 1]) {
                        leftIdx = j
                        rightIdx = j + 1
                        break
                    }
                }

                let x0 = Double(validIndices[leftIdx])
                let x1 = Double(validIndices[rightIdx])
                let y0 = validValues[leftIdx]
                let y1 = validValues[rightIdx]

                // Linear interpolation
                let t = (x - x0) / (x1 - x0)
                result[i] = y0 + t * (y1 - y0)
            }
        }

        return result
    }

    // MARK: - Resampling

    /// Resample signal to target length using linear interpolation
    /// - Parameters:
    ///   - signal: Input signal
    ///   - targetLength: Desired output length
    /// - Returns: Resampled signal
    static func resample(_ signal: [Double], targetLength: Int) -> [Double] {
        guard signal.count > 0, targetLength > 0 else { return [] }
        guard signal.count != targetLength else { return signal }

        var result = [Double](repeating: 0, count: targetLength)

        for i in 0..<targetLength {
            // Map output index to input coordinate
            let x = Double(i) * Double(signal.count - 1) / Double(max(1, targetLength - 1))

            let leftIdx = Int(floor(x))
            let rightIdx = min(leftIdx + 1, signal.count - 1)

            if leftIdx == rightIdx {
                result[i] = signal[leftIdx]
            } else {
                let t = x - Double(leftIdx)
                result[i] = signal[leftIdx] + t * (signal[rightIdx] - signal[leftIdx])
            }
        }

        return result
    }

    // MARK: - Vector Operations

    /// Subtract mean from vector (in-place operation on copy)
    static func centerVector(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }

        var result = vector
        var mean: Float = 0
        vDSP_meanv(vector, 1, &mean, vDSP_Length(vector.count))

        let negMean = -mean
        vDSP_vsadd(vector, 1, [negMean], &result, 1, vDSP_Length(vector.count))

        return result
    }

    /// Compute mean of vector
    static func mean(_ vector: [Float]) -> Float {
        guard !vector.isEmpty else { return 0 }

        var result: Float = 0
        vDSP_meanv(vector, 1, &result, vDSP_Length(vector.count))
        return result
    }

    /// Compute sum of vector
    static func sum(_ vector: [Double]) -> Double {
        guard !vector.isEmpty else { return 0 }

        var result: Double = 0
        vDSP_sveD(vector, 1, &result, vDSP_Length(vector.count))
        return result
    }
}
