//
//  AlgorithmComparisonTest.swift
//  ECGDigitizerTests
//
//  Side-by-side comparison of SignalExtractorAdvanced vs SignalExtractorSectioning
//

import XCTest
@testable import ECGDigitizer

class AlgorithmComparisonTest: XCTestCase {

    /// Test both algorithms on synthetic ECG probability map
    func testSideBySideComparison() {
        print("\n" + String(repeating: "=", count: 70))
        print("ALGORITHM COMPARISON: Advanced (Hungarian+CC) vs Sectioning")
        print(String(repeating: "=", count: 70))

        // Create synthetic signal probability map (3 rows of ECG signals)
        let width = 800
        let height = 600
        var signalProb = [Float](repeating: 0, count: width * height)
        var gridProb = [Float](repeating: 0, count: width * height)

        // Create 3 horizontal signal traces with some variation
        let rowCenters = [100, 300, 500]  // Y positions for 3 rows

        for (rowIdx, centerY) in rowCenters.enumerated() {
            for x in 0..<width {
                // Create a sine wave pattern for each row
                let phase = Double(rowIdx) * 0.5
                let amplitude = 20.0 + Double(rowIdx) * 5.0
                let y = centerY + Int(amplitude * sin(Double(x) * 0.02 + phase))

                // Add signal probability in a small vertical range
                for dy in -3...3 {
                    let py = y + dy
                    if py >= 0 && py < height {
                        let idx = py * width + x
                        let prob = Float(1.0 - Double(abs(dy)) / 4.0) * 0.9
                        signalProb[idx] = max(signalProb[idx], prob)
                    }
                }
            }
        }

        // Add some grid probability (vertical and horizontal lines)
        for y in stride(from: 0, to: height, by: 50) {
            for x in 0..<width {
                gridProb[y * width + x] = 0.7
            }
        }
        for x in stride(from: 0, to: width, by: 50) {
            for y in 0..<height {
                gridProb[y * width + x] = 0.7
            }
        }

        let calibration = GridCalibration(
            smallSquareWidthPixels: 10.0,
            smallSquareHeightPixels: 10.0,
            angleInDegrees: 0.0,
            qualityScore: 0.9,
            gridBounds: nil
        )

        // ============================================================
        // Run SignalExtractorAdvanced
        // ============================================================
        print("\n📊 ALGORITHM 1: SignalExtractorAdvanced (Hungarian + CC)")
        print(String(repeating: "-", count: 50))

        let advancedExtractor = SignalExtractorAdvanced()
        var advancedLeads: [[Double]] = []
        var advancedTime: TimeInterval = 0

        do {
            let startTime = Date()
            advancedLeads = try advancedExtractor.extractLeads(
                signalProb: signalProb,
                width: width,
                height: height,
                calibration: calibration
            )
            advancedTime = Date().timeIntervalSince(startTime)

            let nonEmpty = advancedLeads.filter { $0.contains { abs($0) > 0.0001 } }.count
            print("  ✅ Extraction complete")
            print("  📈 Leads extracted: \(advancedLeads.count)")
            print("  📊 Non-empty leads: \(nonEmpty)")
            print("  ⏱️  Time: \(String(format: "%.3f", advancedTime * 1000)) ms")

            // Show sample counts per lead
            print("\n  Lead details:")
            for (i, lead) in advancedLeads.prefix(6).enumerated() {
                let nonZero = lead.filter { abs($0) > 0.0001 }.count
                let maxVal = lead.max() ?? 0
                let minVal = lead.min() ?? 0
                print("    Lead \(i): \(lead.count) samples, \(nonZero) non-zero, range: [\(String(format: "%.2f", minVal)), \(String(format: "%.2f", maxVal))]")
            }

        } catch {
            print("  ❌ FAILED: \(error)")
        }

        // ============================================================
        // Run SignalExtractorSectioning
        // ============================================================
        print("\n📊 ALGORITHM 2: SignalExtractorSectioning (Row-based)")
        print(String(repeating: "-", count: 50))

        let sectioningExtractor = SignalExtractorSectioning()
        var sectioningLeads: [ECGLead] = []
        var sectioningTime: TimeInterval = 0

        do {
            let startTime = Date()
            sectioningLeads = try sectioningExtractor.extractLeads(
                signalProb: signalProb,
                gridProb: gridProb,
                width: width,
                height: height,
                calibration: calibration
            )
            sectioningTime = Date().timeIntervalSince(startTime)

            let nonEmpty = sectioningLeads.filter { $0.samples.contains { abs($0) > 0.0001 } }.count
            print("  ✅ Extraction complete")
            print("  📈 Leads extracted: \(sectioningLeads.count)")
            print("  📊 Non-empty leads: \(nonEmpty)")
            print("  ⏱️  Time: \(String(format: "%.3f", sectioningTime * 1000)) ms")

            // Show sample counts per lead
            print("\n  Lead details:")
            for (i, lead) in sectioningLeads.prefix(6).enumerated() {
                let nonZero = lead.samples.filter { abs($0) > 0.0001 }.count
                let maxVal = lead.samples.max() ?? 0
                let minVal = lead.samples.min() ?? 0
                print("    \(lead.type.rawValue): \(lead.samples.count) samples, \(nonZero) non-zero, range: [\(String(format: "%.2f", minVal)), \(String(format: "%.2f", maxVal))]")
            }

        } catch {
            print("  ❌ FAILED: \(error)")
        }

        // ============================================================
        // Comparison Summary
        // ============================================================
        print("\n" + String(repeating: "=", count: 70))
        print("COMPARISON SUMMARY")
        print(String(repeating: "=", count: 70))

        let advNonEmpty = advancedLeads.filter { $0.contains { abs($0) > 0.0001 } }.count
        let secNonEmpty = sectioningLeads.filter { $0.samples.contains { abs($0) > 0.0001 } }.count

        print("""

        | Metric                | Advanced (H+CC) | Sectioning |
        |-----------------------|-----------------|------------|
        | Leads extracted       | \(String(format: "%15d", advancedLeads.count)) | \(String(format: "%10d", sectioningLeads.count)) |
        | Non-empty leads       | \(String(format: "%15d", advNonEmpty)) | \(String(format: "%10d", secNonEmpty)) |
        | Processing time (ms)  | \(String(format: "%15.2f", advancedTime * 1000)) | \(String(format: "%10.2f", sectioningTime * 1000)) |

        """)

        if advNonEmpty >= secNonEmpty {
            print("✅ WINNER: SignalExtractorAdvanced (Hungarian + Connected Components)")
        } else {
            print("⚠️ SignalExtractorSectioning performed better on this test")
        }

        print(String(repeating: "=", count: 70) + "\n")

        // Assertions
        XCTAssertGreaterThanOrEqual(advancedLeads.count, 1, "Advanced should extract at least 1 lead")
        XCTAssertGreaterThanOrEqual(sectioningLeads.count, 1, "Sectioning should extract at least 1 lead")
    }

    /// Performance comparison on larger image
    func testPerformanceComparison() {
        print("\n" + String(repeating: "=", count: 70))
        print("PERFORMANCE TEST: 2000x1000 Image")
        print(String(repeating: "=", count: 70))

        let width = 2000
        let height = 1000
        var signalProb = [Float](repeating: 0, count: width * height)
        var gridProb = [Float](repeating: 0.1, count: width * height)

        // Create more complex signal pattern
        let rowCenters = [166, 500, 833]

        for centerY in rowCenters {
            for x in 0..<width {
                // QRS-like pattern
                let xNorm = Double(x % 200) / 200.0
                var y = Double(centerY)

                if xNorm < 0.1 {
                    y += 5.0 * sin(xNorm * 10 * .pi)
                } else if xNorm < 0.15 {
                    y -= 30.0 * (xNorm - 0.1) / 0.05
                } else if xNorm < 0.2 {
                    y += 50.0 * (xNorm - 0.15) / 0.05 - 30
                } else if xNorm < 0.25 {
                    y -= 20.0 * (xNorm - 0.2) / 0.05 + 20
                } else {
                    y += 3.0 * sin((xNorm - 0.25) * 4 * .pi)
                }

                let yInt = Int(y)
                for dy in -2...2 {
                    let py = yInt + dy
                    if py >= 0 && py < height {
                        signalProb[py * width + x] = Float(0.9 - Double(abs(dy)) * 0.2)
                    }
                }
            }
        }

        let calibration = GridCalibration(
            smallSquareWidthPixels: 20.0,
            smallSquareHeightPixels: 20.0,
            angleInDegrees: 0.0,
            qualityScore: 0.9,
            gridBounds: nil
        )

        // Time Advanced
        let advStart = Date()
        let advExtractor = SignalExtractorAdvanced()
        let advLeads = try? advExtractor.extractLeads(
            signalProb: signalProb,
            width: width,
            height: height,
            calibration: calibration
        )
        let advTime = Date().timeIntervalSince(advStart)

        // Time Sectioning
        let secStart = Date()
        let secExtractor = SignalExtractorSectioning()
        let secLeads = try? secExtractor.extractLeads(
            signalProb: signalProb,
            gridProb: gridProb,
            width: width,
            height: height,
            calibration: calibration
        )
        let secTime = Date().timeIntervalSince(secStart)

        print("""

        Performance Results (2000x1000 image):

        | Algorithm          | Time (ms) | Leads | Status |
        |--------------------|-----------|-------|--------|
        | Advanced (H+CC)    | \(String(format: "%9.2f", advTime * 1000)) | \(String(format: "%5d", advLeads?.count ?? 0)) | \(advLeads != nil ? "✅" : "❌") |
        | Sectioning         | \(String(format: "%9.2f", secTime * 1000)) | \(String(format: "%5d", secLeads?.count ?? 0)) | \(secLeads != nil ? "✅" : "❌") |

        """)

        print(String(repeating: "=", count: 70) + "\n")
    }
}
