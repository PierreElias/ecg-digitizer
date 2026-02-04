import Foundation
import UIKit
import Vision
import CoreML

/// Classifies ECG layout type from images
final class LayoutClassifier {

    // MARK: - Configuration

    struct Config {
        /// Minimum confidence threshold for classification
        static let minimumConfidence: Float = 0.8

        /// Input image size for model (if using ML model)
        static let inputSize: CGSize = CGSize(width: 512, height: 512)

        /// Supported layouts from PMcardio spec
        static let supportedLayouts = ECGLayout.allCases
    }

    // MARK: - Properties

    private var mlModel: VNCoreMLModel?
    private let visionManager: VisionManager

    // MARK: - Initialization

    init() {
        self.visionManager = VisionManager()
        loadModel()
    }

    private func loadModel() {
        // Attempt to load Core ML model if available
        // In production, this would load a trained model
        // For now, we use heuristic-based classification
    }

    // MARK: - Classification

    /// Classifies the ECG layout from an image
    func classify(image: UIImage, gridInfo: GridDetectionResult?) async throws -> LayoutClassificationResult {
        // If we have a Core ML model, use it
        if let model = mlModel {
            return try await classifyWithModel(image: image, model: model)
        }

        // Otherwise, use heuristic-based classification
        return try await classifyWithHeuristics(image: image, gridInfo: gridInfo)
    }

    /// Classifies using Core ML model
    private func classifyWithModel(
        image: UIImage,
        model: VNCoreMLModel
    ) async throws -> LayoutClassificationResult {
        guard let cgImage = image.cgImage else {
            throw LayoutClassificationError.invalidImage
        }

        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results as? [VNClassificationObservation],
              let topResult = results.first else {
            throw LayoutClassificationError.classificationFailed
        }

        // Convert string classification to ECGLayout
        guard let layout = ECGLayout(rawValue: topResult.identifier) else {
            throw LayoutClassificationError.unsupportedLayout(topResult.identifier)
        }

        return LayoutClassificationResult(
            layout: layout,
            confidence: topResult.confidence,
            alternativeLayouts: results.dropFirst().prefix(3).compactMap { result in
                guard let layout = ECGLayout(rawValue: result.identifier) else { return nil }
                return (layout, result.confidence)
            }
        )
    }

    /// Classifies using heuristic-based analysis
    private func classifyWithHeuristics(
        image: UIImage,
        gridInfo: GridDetectionResult?
    ) async throws -> LayoutClassificationResult {
        // Step 1: Detect text labels to identify lead positions
        let textObservations = try await visionManager.recognizeText(in: image)
        let leadLabels = visionManager.extractLeadLabels(from: textObservations)

        // Step 2: Analyze label positions to determine layout
        let layoutAnalysis = analyzeLeadPositions(labels: leadLabels, imageSize: image.size)

        // Step 3: Count rows and columns
        let (rows, columns) = countRowsAndColumns(labels: leadLabels, imageSize: image.size)

        // Step 4: Detect rhythm leads
        let rhythmLeads = detectRhythmLeads(labels: leadLabels, imageSize: image.size)

        // Step 5: Match to known layout
        let (layout, confidence) = matchLayout(
            detectedRows: rows,
            detectedColumns: columns,
            detectedRhythmLeads: rhythmLeads,
            analysis: layoutAnalysis
        )

        return LayoutClassificationResult(
            layout: layout,
            confidence: confidence,
            alternativeLayouts: suggestAlternativeLayouts(detectedRows: rows, detectedColumns: columns, detectedRhythmLeads: rhythmLeads)
        )
    }

    // MARK: - Heuristic Analysis

    /// Analyzes lead label positions
    private func analyzeLeadPositions(
        labels: [(label: String, boundingBox: CGRect)],
        imageSize: CGSize
    ) -> LeadPositionAnalysis {
        // Group labels by row (similar Y position)
        var rowGroups: [[String]] = []
        var currentRow: [String] = []
        var lastY: CGFloat = -1

        let sortedByY = labels.sorted { $0.boundingBox.midY > $1.boundingBox.midY }

        for (label, box) in sortedByY {
            if lastY < 0 || abs(box.midY - lastY) < 0.1 {
                currentRow.append(label)
            } else {
                if !currentRow.isEmpty {
                    rowGroups.append(currentRow)
                }
                currentRow = [label]
            }
            lastY = box.midY
        }
        if !currentRow.isEmpty {
            rowGroups.append(currentRow)
        }

        // Determine presentation system (Standard vs Cabrera)
        let isCabrera = labels.contains { $0.label.uppercased() == "-AVR" || $0.label.uppercased() == "AVR-" }

        return LeadPositionAnalysis(
            rowCount: rowGroups.count,
            labelsPerRow: rowGroups.map { $0.count },
            isCabrera: isCabrera,
            detectedLabels: labels.map { $0.label }
        )
    }

    /// Counts rows and columns based on label positions
    private func countRowsAndColumns(
        labels: [(label: String, boundingBox: CGRect)],
        imageSize: CGSize
    ) -> (rows: Int, columns: Int) {
        guard !labels.isEmpty else {
            return (3, 4)  // Default to most common layout
        }

        // Calculate unique Y positions (rows)
        let yPositions = labels.map { $0.boundingBox.midY }
        let uniqueYs = clusterValues(yPositions, threshold: 0.08)

        // Calculate unique X positions (columns)
        let xPositions = labels.map { $0.boundingBox.midX }
        let uniqueXs = clusterValues(xPositions, threshold: 0.15)

        return (uniqueYs.count, uniqueXs.count)
    }

    /// Clusters similar values together
    private func clusterValues(_ values: [CGFloat], threshold: CGFloat) -> [CGFloat] {
        guard !values.isEmpty else { return [] }

        let sorted = values.sorted()
        var clusters: [CGFloat] = [sorted[0]]

        for value in sorted.dropFirst() {
            if let last = clusters.last, abs(value - last) > threshold {
                clusters.append(value)
            }
        }

        return clusters
    }

    /// Detects rhythm leads
    private func detectRhythmLeads(
        labels: [(label: String, boundingBox: CGRect)],
        imageSize: CGSize
    ) -> Int {
        // Rhythm leads are typically at the bottom and span the full width
        let bottomLabels = labels.filter { $0.boundingBox.midY < 0.15 }

        // Check for R1, R2, R3 or II rhythm strip
        let rhythmLabels = bottomLabels.filter { label in
            let text = label.label.uppercased()
            return text == "R1" || text == "R2" || text == "R3" ||
                   text == "II" || text == "RHYTHM"
        }

        return min(rhythmLabels.count, 3)
    }

    /// Matches detected parameters to known layout
    private func matchLayout(
        detectedRows: Int,
        detectedColumns: Int,
        detectedRhythmLeads: Int,
        analysis: LeadPositionAnalysis
    ) -> (ECGLayout, Float) {
        // Match based on rows x columns format
        var bestMatch: ECGLayout = .threeByFour_r1  // Default
        var bestConfidence: Float = 0.5

        for layout in ECGLayout.allCases {
            var score: Float = 0

            // Check row match
            if layout.rows == detectedRows {
                score += 0.3
            } else if abs(layout.rows - detectedRows) == 1 {
                score += 0.15
            }

            // Check column match
            if layout.columns == detectedColumns {
                score += 0.3
            } else if abs(layout.columns - detectedColumns) == 1 {
                score += 0.15
            }

            // Check rhythm lead match
            if layout.rhythmLeads == detectedRhythmLeads {
                score += 0.2
            } else if abs(layout.rhythmLeads - detectedRhythmLeads) == 1 {
                score += 0.1
            }

            // Bonus for detecting correct number of leads
            let expectedLeads = 12 + layout.rhythmLeads
            let detectedLeads = analysis.detectedLabels.count
            if detectedLeads >= expectedLeads - 2 && detectedLeads <= expectedLeads + 2 {
                score += 0.2
            }

            if score > bestConfidence {
                bestConfidence = score
                bestMatch = layout
            }
        }

        return (bestMatch, bestConfidence)
    }

    /// Suggests alternative layouts
    private func suggestAlternativeLayouts(
        detectedRows: Int,
        detectedColumns: Int,
        detectedRhythmLeads: Int
    ) -> [(ECGLayout, Float)] {
        var alternatives: [(ECGLayout, Float)] = []

        for layout in ECGLayout.allCases {
            var score: Float = 0

            if layout.rows == detectedRows { score += 0.3 }
            if layout.columns == detectedColumns { score += 0.3 }
            if layout.rhythmLeads == detectedRhythmLeads { score += 0.2 }

            if score > 0.3 && score < 0.8 {
                alternatives.append((layout, score))
            }
        }

        return alternatives.sorted { $0.1 > $1.1 }.prefix(3).map { $0 }
    }
}

// MARK: - Supporting Types

struct LayoutClassificationResult {
    let layout: ECGLayout
    let confidence: Float
    let alternativeLayouts: [(ECGLayout, Float)]

    var isHighConfidence: Bool {
        confidence >= LayoutClassifier.Config.minimumConfidence
    }
}

struct LeadPositionAnalysis {
    let rowCount: Int
    let labelsPerRow: [Int]
    let isCabrera: Bool
    let detectedLabels: [String]
}

enum LayoutClassificationError: Error, LocalizedError {
    case invalidImage
    case classificationFailed
    case unsupportedLayout(String)
    case lowConfidence(Float)

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid image for layout classification"
        case .classificationFailed:
            return "Layout classification failed"
        case .unsupportedLayout(let layout):
            return "Unsupported ECG layout: \(layout)"
        case .lowConfidence(let confidence):
            return "Low classification confidence: \(String(format: "%.0f", confidence * 100))%"
        }
    }
}
