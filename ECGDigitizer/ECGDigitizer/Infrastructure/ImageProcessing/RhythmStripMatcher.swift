//
//  RhythmStripMatcher.swift
//  ECGDigitizer
//
//  Matches rhythm strip leads to canonical leads using cosine similarity
//  Implements inflation bias for common rhythm leads (II, V1, V5)
//

import Foundation

/// Rhythm strip matching using cosine similarity
class RhythmStripMatcher {

    // MARK: - Main Entry Point

    /// Match rhythm leads to canonical leads using cosine similarity
    /// - Parameters:
    ///   - rhythmLeads: Rhythm strip leads to match
    ///   - canonicalLeads: Standard 12-lead ECG leads
    /// - Returns: Array of (rhythmIndex, canonicalIndex) assignments
    func matchRhythmLeads(
        rhythmLeads: [ECGLead],
        canonicalLeads: [ECGLead]
    ) -> [(rhythmIndex: Int, canonicalIndex: Int)] {
        guard !rhythmLeads.isEmpty else {
            print("  [RhythmMatcher] No rhythm leads to match")
            return []
        }

        let numRhythm = rhythmLeads.count
        let numCanonical = canonicalLeads.count

        print("  [RhythmMatcher] Matching \(numRhythm) rhythm leads to \(numCanonical) canonical leads")

        // Compute cosine similarity matrix
        var similarityMatrix = [[Double]](
            repeating: [Double](repeating: -1.0, count: numCanonical),
            count: numRhythm
        )

        for (i, rhythmLead) in rhythmLeads.enumerated() {
            for (j, canonicalLead) in canonicalLeads.enumerated() {
                let similarity = VectorMathUtilities.nanSafeCosineSimilarity(
                    rhythmLead.samples,
                    canonicalLead.samples
                )

                similarityMatrix[i][j] = similarity.isNaN ? -1.0 : similarity

                // Debug: Print top similarities
                if similarity > 0.7 {
                    print("    Rhythm[\(i)] vs Canonical[\(j)] (\(canonicalLead.type.rawValue)): \(String(format: "%.3f", similarity))")
                }
            }
        }

        // Apply inflation bias to common rhythm leads
        let commonLeadIndices = getCommonRhythmLeadIndices(count: numRhythm)
        for (rhythmIdx, canonicalIdx) in commonLeadIndices {
            if rhythmIdx < numRhythm && canonicalIdx < numCanonical {
                let original = similarityMatrix[rhythmIdx][canonicalIdx]
                similarityMatrix[rhythmIdx][canonicalIdx] = inflateSimilarity(original, factor: 0.75)

                let leadName = canonicalLeads[canonicalIdx].type.rawValue
                print("    Inflated Rhythm[\(rhythmIdx)] → \(leadName): \(String(format: "%.3f", original)) → \(String(format: "%.3f", similarityMatrix[rhythmIdx][canonicalIdx]))")
            }
        }

        // Use Hungarian algorithm for optimal assignment
        // Convert to Float for HungarianAlgorithm which expects [[Float]]
        let costMatrix: [[Float]] = similarityMatrix.map { row in
            row.map { Float(-$0) }  // Negate for minimization (Hungarian minimizes cost)
        }

        let hungarianAssignments = HungarianAlgorithm.solve(costMatrix: costMatrix)

        // Convert tuple labels from (row, col) to (rhythmIndex, canonicalIndex)
        let assignments: [(rhythmIndex: Int, canonicalIndex: Int)] = hungarianAssignments.map {
            (rhythmIndex: $0.row, canonicalIndex: $0.col)
        }

        // Log assignments
        print("  [RhythmMatcher] Assignments:")
        for (rhythmIdx, canonicalIdx) in assignments {
            if canonicalIdx < canonicalLeads.count {
                let leadName = canonicalLeads[canonicalIdx].type.rawValue
                let similarity = similarityMatrix[rhythmIdx][canonicalIdx]
                print("    Rhythm[\(rhythmIdx)] → \(leadName) (similarity: \(String(format: "%.3f", similarity)))")
            }
        }

        return assignments
    }

    // MARK: - Common Rhythm Lead Mapping

    /// Get typical rhythm lead indices for inflation bias
    /// - Parameter count: Number of rhythm leads (1, 2, or 3)
    /// - Returns: Array of (rhythmIndex, canonicalIndex) for common leads
    private func getCommonRhythmLeadIndices(count: Int) -> [(Int, Int)] {
        // Common rhythm leads in standard ECG layouts:
        // Lead II = canonical index 1
        // V1 = canonical index 6
        // V5 = canonical index 10

        switch count {
        case 1:
            // Single rhythm strip: usually Lead II
            return [(0, 1)]  // Rhythm[0] → Lead II

        case 2:
            // Two rhythm strips: usually Lead II and V1
            return [
                (0, 1),   // Rhythm[0] → Lead II
                (1, 6)    // Rhythm[1] → V1
            ]

        case 3:
            // Three rhythm strips: usually Lead II, V1, and V5
            return [
                (0, 1),   // Rhythm[0] → Lead II
                (1, 6),   // Rhythm[1] → V1
                (2, 10)   // Rhythm[2] → V5
            ]

        default:
            return []
        }
    }

    // MARK: - Similarity Inflation

    /// Inflate cosine similarity for common rhythm leads
    /// Formula: inflated = 1.0 - factor + factor * original
    /// - Parameters:
    ///   - value: Original cosine similarity
    ///   - factor: Inflation factor (0.75 = 25% boost to baseline)
    /// - Returns: Inflated similarity score
    private func inflateSimilarity(_ value: Double, factor: Double = 0.75) -> Double {
        guard !value.isNaN else { return value }

        // Formula gives a baseline boost:
        // - If similarity = 1.0: inflated = 1.0
        // - If similarity = 0.0: inflated = 0.25 (with factor=0.75)
        // - If similarity = 0.5: inflated = 0.625
        return 1.0 - factor + factor * value
    }

    // MARK: - Canonical Lead Index Lookup

    /// Get canonical index for a specific lead type
    /// - Parameter leadType: Lead type to find
    /// - Returns: Canonical index (0-11) or nil if not found
    static func canonicalIndex(for leadType: LeadType) -> Int? {
        let standardOrder: [LeadType] = [
            .I, .II, .III, .aVR, .aVL, .aVF,
            .V1, .V2, .V3, .V4, .V5, .V6
        ]

        return standardOrder.firstIndex(of: leadType)
    }
}
