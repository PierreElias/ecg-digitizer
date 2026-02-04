import Foundation
import UIKit
import Accelerate

/// Extracts ECG waveforms from images
final class WaveformExtractor {

    // MARK: - Configuration

    struct Config {
        /// Target sampling rate for output waveforms (Hz)
        static let targetSamplingRate: Double = 500.0

        /// Minimum signal amplitude to consider valid (mV)
        static let minimumAmplitude: Double = 0.001

        /// Smoothing window size for noise reduction
        static let smoothingWindow: Int = 5

        /// Threshold for detecting dark pixels (signal)
        static let signalThreshold: UInt8 = 128

        /// Maximum gap to interpolate (in pixels)
        static let maxInterpolationGap: Int = 10
    }

    // MARK: - Properties

    private let preprocessor: ImagePreprocessor

    // MARK: - Initialization

    init() {
        self.preprocessor = ImagePreprocessor()
    }

    // MARK: - Extraction

    /// Extracts all leads from an ECG image
    func extractWaveforms(
        from image: UIImage,
        layout: ECGLayout,
        gridCalibration: GridCalibration,
        parameters: ProcessingParameters
    ) async throws -> [ECGLead] {
        // Step 1: Preprocess image for waveform extraction
        let processedImage = preprocessor.preprocessForWaveformExtraction(image)

        // Step 2: Get lead regions based on layout
        let leadRegions = calculateLeadRegions(
            layout: layout,
            imageSize: image.size,
            gridBounds: gridCalibration.gridBounds
        )

        // Step 3: Extract each lead
        var leads: [ECGLead] = []

        for (leadType, region) in leadRegions {
            let lead = try extractSingleLead(
                from: processedImage,
                region: region,
                leadType: leadType,
                gridCalibration: gridCalibration,
                parameters: parameters
            )
            leads.append(lead)
        }

        return leads
    }

    /// Extracts a single lead from a region
    func extractSingleLead(
        from image: UIImage,
        region: CGRect,
        leadType: LeadType,
        gridCalibration: GridCalibration,
        parameters: ProcessingParameters
    ) throws -> ECGLead {
        guard let cgImage = image.cgImage else {
            throw WaveformExtractionError.invalidImage
        }

        // Crop to lead region
        let scaledRegion = CGRect(
            x: region.origin.x * image.scale,
            y: region.origin.y * image.scale,
            width: region.width * image.scale,
            height: region.height * image.scale
        )

        guard let croppedImage = cgImage.cropping(to: scaledRegion) else {
            throw WaveformExtractionError.cropFailed
        }

        // Extract signal trace
        let rawSamples = extractSignalTrace(from: croppedImage)

        // Convert pixels to voltage
        let voltageSamples = convertToVoltage(
            pixelSamples: rawSamples,
            regionHeight: region.height,
            gridCalibration: gridCalibration,
            parameters: parameters
        )

        // Apply smoothing
        let smoothedSamples = applyMedianFilter(voltageSamples, windowSize: Config.smoothingWindow)

        // Resample to target sampling rate
        let resampledSamples = resampleSignal(
            samples: smoothedSamples,
            regionWidth: region.width,
            gridCalibration: gridCalibration,
            parameters: parameters
        )

        return ECGLead(
            type: leadType,
            samples: resampledSamples,
            samplingRate: Config.targetSamplingRate
        )
    }

    // MARK: - Signal Extraction

    /// Extracts signal trace using column-wise scanning
    private func extractSignalTrace(from cgImage: CGImage) -> [Double] {
        let width = cgImage.width
        let height = cgImage.height

        // Create grayscale buffer
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return []
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else { return [] }

        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height)

        var samples: [Double] = []
        var lastValidY: Double?

        // Scan each column
        for x in 0..<width {
            // Find dark pixels in this column
            var darkPixelYs: [Int] = []

            for y in 0..<height {
                let pixelValue = pixels[y * width + x]
                if pixelValue < Config.signalThreshold {
                    darkPixelYs.append(y)
                }
            }

            // Calculate centroid of dark pixels
            var sampleY: Double

            if darkPixelYs.isEmpty {
                // No signal found - interpolate if possible
                if let last = lastValidY {
                    sampleY = last  // Hold last value
                } else {
                    sampleY = Double(height) / 2  // Default to center
                }
            } else {
                // Weighted centroid (more weight to central pixels)
                var sum: Double = 0
                var weightSum: Double = 0

                for y in darkPixelYs {
                    let weight = 1.0  // Could add intensity-based weighting
                    sum += Double(y) * weight
                    weightSum += weight
                }

                sampleY = sum / weightSum
                lastValidY = sampleY
            }

            samples.append(sampleY)
        }

        // Interpolate gaps
        samples = interpolateGaps(samples)

        return samples
    }

    /// Interpolates gaps in the signal
    private func interpolateGaps(_ samples: [Double]) -> [Double] {
        var result = samples
        var gapStart: Int?

        for i in 0..<result.count {
            let isBaseline = abs(result[i] - Double(samples.count) / 2) < 5  // Near center = gap

            if isBaseline && gapStart == nil {
                gapStart = i
            } else if !isBaseline && gapStart != nil {
                let gapEnd = i
                let gapLength = gapEnd - gapStart!

                // Only interpolate small gaps
                if gapLength < Config.maxInterpolationGap {
                    let startValue = gapStart! > 0 ? result[gapStart! - 1] : result[gapEnd]
                    let endValue = result[gapEnd]

                    // Linear interpolation
                    for j in gapStart!..<gapEnd {
                        let t = Double(j - gapStart!) / Double(gapLength)
                        result[j] = startValue + t * (endValue - startValue)
                    }
                }
                gapStart = nil
            }
        }

        return result
    }

    // MARK: - Conversion

    /// Converts pixel Y positions to voltage values
    private func convertToVoltage(
        pixelSamples: [Double],
        regionHeight: CGFloat,
        gridCalibration: GridCalibration,
        parameters: ProcessingParameters
    ) -> [Double] {
        let centerY = regionHeight / 2

        return pixelSamples.map { pixelY in
            // Convert pixel offset from center to mm
            let offsetPixels = centerY - pixelY  // Positive = above center = positive voltage
            let offsetMm = offsetPixels * gridCalibration.mmPerPixelVertical

            // Convert mm to mV using voltage gain
            let voltageMv = offsetMm / parameters.voltageGain.mmPerMillivolt

            return voltageMv
        }
    }

    /// Resamples signal to target sampling rate
    private func resampleSignal(
        samples: [Double],
        regionWidth: CGFloat,
        gridCalibration: GridCalibration,
        parameters: ProcessingParameters
    ) -> [Double] {
        guard !samples.isEmpty else { return [] }

        // Calculate original sample interval in ms
        let widthMm = regionWidth * gridCalibration.mmPerPixelHorizontal
        let durationMs = widthMm * parameters.paperSpeed.msPerMm
        let originalSampleIntervalMs = durationMs / Double(samples.count)

        // Calculate target sample interval
        let targetSampleIntervalMs = 1000.0 / Config.targetSamplingRate

        // Calculate number of output samples
        let outputSampleCount = Int(durationMs / targetSampleIntervalMs)

        // Resample using linear interpolation
        var resampled: [Double] = []

        for i in 0..<outputSampleCount {
            let targetTimeMs = Double(i) * targetSampleIntervalMs
            let sourceIndex = targetTimeMs / originalSampleIntervalMs

            let lowerIndex = Int(floor(sourceIndex))
            let upperIndex = min(lowerIndex + 1, samples.count - 1)
            let fraction = sourceIndex - Double(lowerIndex)

            if lowerIndex >= 0 && lowerIndex < samples.count {
                let interpolated = samples[lowerIndex] + fraction * (samples[upperIndex] - samples[lowerIndex])
                resampled.append(interpolated)
            }
        }

        return resampled
    }

    // MARK: - Filtering

    /// Applies median filter for noise reduction
    private func applyMedianFilter(_ samples: [Double], windowSize: Int) -> [Double] {
        guard samples.count > windowSize else { return samples }

        var filtered: [Double] = []
        let halfWindow = windowSize / 2

        for i in 0..<samples.count {
            let start = max(0, i - halfWindow)
            let end = min(samples.count, i + halfWindow + 1)

            var window = Array(samples[start..<end])
            window.sort()

            let median = window[window.count / 2]
            filtered.append(median)
        }

        return filtered
    }

    /// Applies Butterworth bandpass filter
    func applyBandpassFilter(
        _ samples: [Double],
        lowCutoff: Double = 0.5,
        highCutoff: Double = 100.0,
        samplingRate: Double = 500.0
    ) -> [Double] {
        // Simplified implementation - in production would use proper digital filter
        // Apply high-pass to remove baseline wander
        var filtered = samples

        // Simple moving average subtraction for baseline removal
        let windowSize = Int(samplingRate / lowCutoff)
        var movingAverage: [Double] = []

        for i in 0..<samples.count {
            let start = max(0, i - windowSize / 2)
            let end = min(samples.count, i + windowSize / 2)
            let avg = Array(samples[start..<end]).reduce(0, +) / Double(end - start)
            movingAverage.append(avg)
        }

        for i in 0..<filtered.count {
            filtered[i] = samples[i] - movingAverage[i]
        }

        // Low-pass using simple averaging
        let lpWindowSize = max(1, Int(samplingRate / highCutoff / 2))
        return applyMedianFilter(filtered, windowSize: lpWindowSize)
    }

    // MARK: - Layout Regions

    /// Calculates lead regions based on layout
    private func calculateLeadRegions(
        layout: ECGLayout,
        imageSize: CGSize,
        gridBounds: CGRect
    ) -> [(LeadType, CGRect)] {
        var regions: [(LeadType, CGRect)] = []

        let leadOrder = layout.standardLeadOrder
        let rows = layout.rows
        let columns = layout.columns

        // Calculate cell dimensions
        let cellWidth = gridBounds.width / CGFloat(columns)
        let cellHeight = gridBounds.height / CGFloat(rows + layout.rhythmLeads)

        // Map leads to grid positions
        for (index, leadType) in leadOrder.enumerated() {
            let row = index / columns
            let col = index % columns

            let region = CGRect(
                x: gridBounds.origin.x + CGFloat(col) * cellWidth,
                y: gridBounds.origin.y + CGFloat(row) * cellHeight,
                width: cellWidth,
                height: cellHeight
            )

            regions.append((leadType, region))
        }

        // Add rhythm leads at bottom
        if layout.rhythmLeads > 0 {
            for i in 0..<layout.rhythmLeads {
                let rhythmType: LeadType = [.R1, .R2, .R3][i]
                let rhythmRow = rows + i

                let region = CGRect(
                    x: gridBounds.origin.x,
                    y: gridBounds.origin.y + CGFloat(rhythmRow) * cellHeight,
                    width: gridBounds.width,  // Rhythm leads span full width
                    height: cellHeight
                )

                regions.append((rhythmType, region))
            }
        }

        return regions
    }
}

// MARK: - Errors

enum WaveformExtractionError: Error, LocalizedError {
    case invalidImage
    case cropFailed
    case noSignalFound
    case processingFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid image for waveform extraction"
        case .cropFailed:
            return "Failed to crop lead region"
        case .noSignalFound:
            return "No signal found in lead region"
        case .processingFailed:
            return "Waveform processing failed"
        }
    }
}
