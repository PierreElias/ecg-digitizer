//
//  SignalExtractionTests.swift
//  ECGDigitizerTests
//
//  Unit tests for signal extraction and lead sectioning components
//

import XCTest
@testable import ECGDigitizer

class SignalExtractionTests: XCTestCase {

    // MARK: - VectorMathUtilities Tests

    func testPeakDetection() {
        // Simple test case with clear peaks
        let data: [Float] = [0, 1, 0, 0, 5, 0, 0, 3, 0, 0, 4, 0]

        let (peaks, _) = VectorMathUtilities.findPeaks(
            in: data,
            distance: 2,
            prominence: 0.5,
            height: nil
        )

        // Should find peaks at indices 4 (value=5), 10 (value=4), 7 (value=3)
        // Sorted by height: 5, 4, 3
        XCTAssertEqual(peaks.count, 3, "Should find 3 peaks")
        XCTAssertTrue(peaks.contains(4), "Should find peak at index 4")
        XCTAssertTrue(peaks.contains(10), "Should find peak at index 10")
        XCTAssertTrue(peaks.contains(7), "Should find peak at index 7")
    }

    func testPeakDetectionWithDistance() {
        // Test that distance filtering works
        let data: [Float] = [0, 5, 0, 4, 0, 6, 0]

        let (peaks, _) = VectorMathUtilities.findPeaks(
            in: data,
            distance: 3,  // Peaks must be at least 3 samples apart
            prominence: nil,
            height: nil
        )

        // With distance=3, should keep only the highest peaks that are far apart
        // Peaks at indices: 1 (value=5), 3 (value=4), 5 (value=6)
        // After distance filtering, keep: 5 (value=6) and 1 (value=5)
        XCTAssertLessThanOrEqual(peaks.count, 2, "Distance filtering should reduce peak count")
    }

    func testGaussianSmoothing() {
        let data: [Float] = [1, 2, 3, 2, 1]
        let smoothed = VectorMathUtilities.gaussianFilter1D(data, sigma: 1.0)

        XCTAssertEqual(smoothed.count, data.count, "Output length should match input")

        // Smoothing should reduce the peak
        let originalMax = data.max() ?? 0
        let smoothedMax = smoothed.max() ?? 0
        XCTAssertLessThanOrEqual(smoothedMax, originalMax, "Smoothing should not increase peak")

        // Smoothing should preserve the general shape
        let smoothedPeakIndex = smoothed.enumerated().max { $0.element < $1.element }?.offset ?? 0
        let originalPeakIndex = data.enumerated().max { $0.element < $1.element }?.offset ?? 0
        XCTAssertEqual(smoothedPeakIndex, originalPeakIndex, "Peak position should be preserved")
    }

    func testCosineSimilarity() {
        // Test identical vectors
        let x = [1.0, 2.0, 3.0, 4.0]
        let y = [1.0, 2.0, 3.0, 4.0]
        let similarity = VectorMathUtilities.nanSafeCosineSimilarity(x, y)

        XCTAssertEqual(similarity, 1.0, accuracy: 0.01, "Identical vectors should have similarity = 1.0")
    }

    func testCosineSimilarityOpposite() {
        // Test opposite vectors
        let x = [1.0, 2.0, 3.0, 4.0]
        let y = [-1.0, -2.0, -3.0, -4.0]
        let similarity = VectorMathUtilities.nanSafeCosineSimilarity(x, y)

        XCTAssertEqual(similarity, -1.0, accuracy: 0.01, "Opposite vectors should have similarity = -1.0")
    }

    func testCosineSimilarityOrthogonal() {
        // Test orthogonal vectors
        let x = [1.0, 0.0, 0.0, 0.0]
        let y = [0.0, 1.0, 0.0, 0.0]
        let similarity = VectorMathUtilities.nanSafeCosineSimilarity(x, y)

        XCTAssertEqual(similarity, 0.0, accuracy: 0.01, "Orthogonal vectors should have similarity = 0.0")
    }

    func testCosineSimilarityWithNaN() {
        // Test NaN-safe behavior
        let x = [1.0, Double.nan, 3.0, 4.0]
        let y = [1.0, 2.0, 3.0, 4.0]
        let similarity = VectorMathUtilities.nanSafeCosineSimilarity(x, y)

        XCTAssertFalse(similarity.isNaN, "Should handle NaN values gracefully")
        XCTAssertGreaterThan(similarity, 0.5, "Should still compute meaningful similarity")
    }

    func testMedianFilter() {
        let data = [1.0, 5.0, 1.0, 5.0, 1.0]
        let filtered = VectorMathUtilities.medianFilter(data, windowSize: 3)

        XCTAssertEqual(filtered.count, data.count, "Output length should match input")

        // Middle value should be median of [1, 5, 1] = 1
        XCTAssertEqual(filtered[2], 1.0, accuracy: 0.01, "Median should filter out outliers")
    }

    func testLinearInterpolation() {
        // Test gap filling
        let values = [1.0, Double.nan, Double.nan, 4.0]
        let mask = [true, false, false, true]
        let interpolated = VectorMathUtilities.linearInterpolate(
            values: values,
            mask: mask,
            outputLength: 4
        )

        XCTAssertEqual(interpolated.count, 4, "Output length should match requested")

        // Check interpolated values
        XCTAssertEqual(interpolated[0], 1.0, accuracy: 0.01, "First value should be preserved")
        XCTAssertEqual(interpolated[3], 4.0, accuracy: 0.01, "Last value should be preserved")

        // Middle values should be interpolated
        XCTAssertGreaterThan(interpolated[1], 1.0, "Should interpolate upward")
        XCTAssertLessThan(interpolated[1], 4.0, "Should interpolate between bounds")
        XCTAssertGreaterThan(interpolated[2], 1.0, "Should interpolate upward")
        XCTAssertLessThan(interpolated[2], 4.0, "Should interpolate between bounds")
    }

    func testResample() {
        // Downsample
        let signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        let downsampled = VectorMathUtilities.resample(signal, targetLength: 3)

        XCTAssertEqual(downsampled.count, 3, "Should downsample to target length")
        XCTAssertEqual(downsampled[0], 1.0, accuracy: 0.01, "First sample should match")
        XCTAssertEqual(downsampled[2], 5.0, accuracy: 0.01, "Last sample should match")

        // Upsample
        let upsampled = VectorMathUtilities.resample(signal, targetLength: 10)
        XCTAssertEqual(upsampled.count, 10, "Should upsample to target length")
        XCTAssertEqual(upsampled[0], 1.0, accuracy: 0.01, "First sample should match")
        XCTAssertEqual(upsampled[9], 5.0, accuracy: 0.01, "Last sample should match")
    }

    // MARK: - SignalExtractorSectioning Tests

    func testRowBoundaryDetection() {
        // Create synthetic signal probability with 3 clear rows
        let width = 800
        let height = 600
        var signalProb = [Float](repeating: 0, count: width * height)

        // Add 3 horizontal bands of signal at y = 150, 300, 450
        for y in [150, 300, 450] {
            for x in 0..<width {
                for dy in -10...10 {
                    let actualY = y + dy
                    if actualY >= 0 && actualY < height {
                        signalProb[actualY * width + x] = 0.8
                    }
                }
            }
        }

        let extractor = SignalExtractorSectioning()
        let gridCalibration = GridCalibration(
            smallSquareWidthPixels: 10.0,
            smallSquareHeightPixels: 10.0,
            angleInDegrees: 0.0,
            qualityScore: 0.9,
            gridBounds: nil
        )

        // Note: We can't directly test private methods, but we can test the full extraction
        do {
            let leads = try extractor.extractLeads(
                signalProb: signalProb,
                gridProb: [Float](repeating: 0, count: width * height),
                width: width,
                height: height,
                calibration: gridCalibration
            )

            XCTAssertEqual(leads.count, 12, "Should extract 12 leads")
            XCTAssertEqual(leads[0].samples.count, 5000, "Each lead should have 5000 samples")
        } catch {
            XCTFail("Extraction should not throw: \(error)")
        }
    }

    // MARK: - RhythmStripMatcher Tests

    func testRhythmStripMatchingWithIdenticalLeads() {
        // Create test leads with identical patterns
        let samples = Array(stride(from: 0.0, to: 5000.0, by: 1.0))

        let canonicalLeads = [
            ECGLead(type: .I, samples: samples, samplingRate: 500.0),
            ECGLead(type: .II, samples: samples, samplingRate: 500.0),
            ECGLead(type: .III, samples: samples, samplingRate: 500.0)
        ]

        let rhythmLeads = [
            ECGLead(type: .R1, samples: samples, samplingRate: 500.0)
        ]

        let matcher = RhythmStripMatcher()
        let assignments = matcher.matchRhythmLeads(
            rhythmLeads: rhythmLeads,
            canonicalLeads: canonicalLeads
        )

        XCTAssertEqual(assignments.count, 1, "Should produce 1 assignment")

        // With inflation, rhythm[0] should match Lead II (canonical index 1)
        let (rhythmIdx, canonicalIdx) = assignments[0]
        XCTAssertEqual(rhythmIdx, 0, "Rhythm index should be 0")
        XCTAssertEqual(canonicalIdx, 1, "Should match Lead II (index 1) due to inflation")
    }

    // MARK: - ProcrustesAligner Tests

    func testProcrustesAlignmentIdenticalPoints() {
        let aligner = ProcrustesAligner()

        let points = [
            (x: 0.0 as Float, y: 0.0 as Float),
            (x: 1.0 as Float, y: 0.0 as Float),
            (x: 0.0 as Float, y: 1.0 as Float)
        ]

        let result = aligner.align(detectedPoints: points, gridPoints: points)

        XCTAssertEqual(result.scale, 1.0, accuracy: 0.01, "Scale should be 1 for identical points")
        XCTAssertEqual(result.translation.x, 0.0, accuracy: 0.01, "Translation X should be 0")
        XCTAssertEqual(result.translation.y, 0.0, accuracy: 0.01, "Translation Y should be 0")
        XCTAssertEqual(result.cost, 0.0, accuracy: 0.01, "Cost should be 0 for perfect match")
        XCTAssertGreaterThan(result.quality, 0.95, "Quality should be very high")
    }

    func testProcrustesAlignmentScaled() {
        let aligner = ProcrustesAligner()

        let detectedPoints = [
            (x: 0.0 as Float, y: 0.0 as Float),
            (x: 2.0 as Float, y: 0.0 as Float),
            (x: 0.0 as Float, y: 2.0 as Float)
        ]

        let gridPoints = [
            (x: 0.0 as Float, y: 0.0 as Float),
            (x: 1.0 as Float, y: 0.0 as Float),
            (x: 0.0 as Float, y: 1.0 as Float)
        ]

        let result = aligner.align(detectedPoints: detectedPoints, gridPoints: gridPoints)

        XCTAssertEqual(result.scale, 0.5, accuracy: 0.01, "Scale should be 0.5")
        XCTAssertLessThan(result.cost, 0.1, "Cost should be low for scaled match")
    }

    func testLayoutGeneration() {
        // Test 3×4 layout
        let positions3x4 = ProcrustesAligner.generateGridPositions(for: .threeByFour_r1)
        XCTAssertEqual(positions3x4.count, 12, "3×4 layout should have 12 positions")

        // Test 6×2 layout
        let positions6x2 = ProcrustesAligner.generateGridPositions(for: .sixByTwo_r0)
        XCTAssertEqual(positions6x2.count, 12, "6×2 layout should have 12 positions")

        // Verify normalized coordinates
        for pos in positions3x4 {
            XCTAssertGreaterThanOrEqual(pos.x, 0.0, "X should be >= 0")
            XCTAssertLessThanOrEqual(pos.x, 1.0, "X should be <= 1")
            XCTAssertGreaterThanOrEqual(pos.y, 0.0, "Y should be >= 0")
            XCTAssertLessThanOrEqual(pos.y, 1.0, "Y should be <= 1")
        }
    }

    func testPointNormalization() {
        let points = [
            (x: 10.0 as Float, y: 20.0 as Float),
            (x: 50.0 as Float, y: 40.0 as Float),
            (x: 30.0 as Float, y: 60.0 as Float)
        ]

        let normalized = ProcrustesAligner.normalizePoints(points)

        // Check range
        let xValues = normalized.map { $0.x }
        let yValues = normalized.map { $0.y }

        XCTAssertEqual(xValues.min() ?? 0, 0.0, accuracy: 0.01, "Min X should be 0")
        XCTAssertEqual(xValues.max() ?? 0, 1.0, accuracy: 0.01, "Max X should be 1")
        XCTAssertEqual(yValues.min() ?? 0, 0.0, accuracy: 0.01, "Min Y should be 0")
        XCTAssertEqual(yValues.max() ?? 0, 1.0, accuracy: 0.01, "Max Y should be 1")
    }

    // MARK: - Integration Tests

    func testEndToEndExtractionWithSyntheticData() {
        // Create synthetic ECG signal probability map
        let width = 800
        let height = 600

        // Simple sine wave pattern for testing
        var signalProb = [Float](repeating: 0, count: width * height)
        var gridProb = [Float](repeating: 0, count: width * height)

        // Add 3 rows of sine wave signal
        for rowIdx in 0..<3 {
            let rowY = 100 + rowIdx * 200  // Rows at y=100, 300, 500

            for x in 0..<width {
                // Sine wave centered at rowY
                let amplitude: Float = 20.0
                let frequency: Float = 0.02
                let yOffset = sin(Float(x) * frequency) * amplitude
                let y = Int(Float(rowY) + yOffset)

                if y >= 0 && y < height {
                    signalProb[y * width + x] = 1.0
                }
            }
        }

        let extractor = SignalExtractorSectioning()
        let calibration = GridCalibration(
            smallSquareWidthPixels: 5.0,
            smallSquareHeightPixels: 5.0,
            angleInDegrees: 0.0,
            qualityScore: 0.9,
            gridBounds: nil
        )

        do {
            let leads = try extractor.extractLeads(
                signalProb: signalProb,
                gridProb: gridProb,
                width: width,
                height: height,
                calibration: calibration
            )

            // Verify output structure
            XCTAssertEqual(leads.count, 12, "Should extract 12 leads")

            for (idx, lead) in leads.enumerated() {
                XCTAssertEqual(lead.samples.count, 5000, "Lead \(idx) should have 5000 samples")
                XCTAssertEqual(lead.samplingRate, 500.0, "Sample rate should be 500 Hz")

                // Check for non-zero samples (indicating signal was extracted)
                let nonZeroCount = lead.samples.filter { abs($0) > 0.0001 }.count
                XCTAssertGreaterThan(nonZeroCount, 100, "Lead \(idx) should have significant signal")
            }

            // Verify lead types are correct
            XCTAssertEqual(leads[0].type, .I, "First lead should be Lead I")
            XCTAssertEqual(leads[1].type, .II, "Second lead should be Lead II")
            XCTAssertEqual(leads[11].type, .V6, "Last lead should be Lead V6")

        } catch {
            XCTFail("Extraction failed: \(error)")
        }
    }
}
