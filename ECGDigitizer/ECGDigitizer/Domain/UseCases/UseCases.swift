import Foundation
import UIKit

// MARK: - Validate Image Use Case

/// Validates an input image for ECG digitization
final class ValidateImageUseCase {
    private let validator: ECGValidator

    init(validator: ECGValidator) {
        self.validator = validator
    }

    func execute(image: UIImage, sourceURL: URL? = nil) throws {
        try validator.validateImage(image, sourceURL: sourceURL)
    }
}

// MARK: - Detect Grid Use Case

/// Detects ECG grid in an image
final class DetectGridUseCase {
    private let preprocessor: ImagePreprocessor
    private let gridDetector: GridDetector
    private let validator: ECGValidator

    init(
        preprocessor: ImagePreprocessor,
        gridDetector: GridDetector,
        validator: ECGValidator
    ) {
        self.preprocessor = preprocessor
        self.gridDetector = gridDetector
        self.validator = validator
    }

    func execute(image: UIImage) async throws -> GridDetectionResult {
        // Preprocess
        let preprocessed = preprocessor.preprocessForGridDetection(image)

        // Detect grid
        let result = try await gridDetector.detectGrid(in: preprocessed)

        // Validate
        try validator.validateGrid(result)

        return result
    }
}

// MARK: - Classify Layout Use Case

/// Classifies ECG layout from an image
final class ClassifyLayoutUseCase {
    private let classifier: LayoutClassifier
    private let validator: ECGValidator

    init(classifier: LayoutClassifier, validator: ECGValidator) {
        self.classifier = classifier
        self.validator = validator
    }

    func execute(
        image: UIImage,
        gridInfo: GridDetectionResult?
    ) async throws -> LayoutClassificationResult {
        let result = try await classifier.classify(image: image, gridInfo: gridInfo)

        // Validate confidence
        if !result.isHighConfidence {
            throw LayoutClassificationError.lowConfidence(result.confidence)
        }

        // Validate layout is supported
        try validator.validateLayout(result.layout)

        return result
    }
}

// MARK: - Extract Waveforms Use Case

/// Extracts waveforms from an ECG image
final class ExtractWaveformsUseCase {
    private let extractor: WaveformExtractor
    private let validator: ECGValidator

    init(extractor: WaveformExtractor, validator: ECGValidator) {
        self.extractor = extractor
        self.validator = validator
    }

    func execute(
        image: UIImage,
        layout: ECGLayout,
        gridCalibration: GridCalibration,
        parameters: ProcessingParameters
    ) async throws -> [ECGLead] {
        let leads = try await extractor.extractWaveforms(
            from: image,
            layout: layout,
            gridCalibration: gridCalibration,
            parameters: parameters
        )

        // Validate leads
        try validator.validateLeads(leads, layout: layout)

        return leads
    }
}

// MARK: - Validate Result Use Case

/// Validates complete digitization result
final class ValidateResultUseCase {
    private let validator: ECGValidator

    init(validator: ECGValidator) {
        self.validator = validator
    }

    func execute(
        image: UIImage,
        gridResult: GridDetectionResult?,
        layout: ECGLayout?,
        leads: [ECGLead]?,
        parameters: ProcessingParameters
    ) throws -> ValidationStatus {
        return try validator.validate(
            image: image,
            gridResult: gridResult,
            layout: layout,
            leads: leads,
            parameters: parameters
        )
    }
}

// MARK: - Save Recording Use Case

/// Saves an ECG recording
final class SaveRecordingUseCase {
    private let repository: ECGRecordingRepository

    init(repository: ECGRecordingRepository) {
        self.repository = repository
    }

    func execute(_ recording: ECGRecording) async throws {
        try await repository.save(recording)
    }
}

// MARK: - Load Recordings Use Case

/// Loads all ECG recordings
final class LoadRecordingsUseCase {
    private let repository: ECGRecordingRepository

    init(repository: ECGRecordingRepository) {
        self.repository = repository
    }

    func execute() async throws -> [ECGRecording] {
        return try await repository.fetchAll()
    }

    func execute(id: UUID) async throws -> ECGRecording {
        return try await repository.fetch(id: id)
    }
}

// MARK: - Delete Recording Use Case

/// Deletes an ECG recording
final class DeleteRecordingUseCase {
    private let repository: ECGRecordingRepository

    init(repository: ECGRecordingRepository) {
        self.repository = repository
    }

    func execute(id: UUID) async throws {
        try await repository.delete(id: id)
    }
}

// MARK: - Complete Digitization Use Case

/// Orchestrates the complete digitization process
final class DigitizeImageUseCase {
    private let validateImageUseCase: ValidateImageUseCase
    private let detectGridUseCase: DetectGridUseCase
    private let classifyLayoutUseCase: ClassifyLayoutUseCase
    private let extractWaveformsUseCase: ExtractWaveformsUseCase
    private let validateResultUseCase: ValidateResultUseCase

    init(
        validateImageUseCase: ValidateImageUseCase,
        detectGridUseCase: DetectGridUseCase,
        classifyLayoutUseCase: ClassifyLayoutUseCase,
        extractWaveformsUseCase: ExtractWaveformsUseCase,
        validateResultUseCase: ValidateResultUseCase
    ) {
        self.validateImageUseCase = validateImageUseCase
        self.detectGridUseCase = detectGridUseCase
        self.classifyLayoutUseCase = classifyLayoutUseCase
        self.extractWaveformsUseCase = extractWaveformsUseCase
        self.validateResultUseCase = validateResultUseCase
    }

    func execute(
        image: UIImage,
        parameters: ProcessingParameters,
        progressHandler: ((ProcessingState) -> Void)? = nil
    ) async throws -> ECGRecording {
        _ = Date() // startTime for potential future logging

        // Step 1: Validate image
        progressHandler?(.validatingImage)
        try validateImageUseCase.execute(image: image)

        // Step 2: Detect grid
        progressHandler?(.detectingGrid(progress: 0))
        let gridResult = try await detectGridUseCase.execute(image: image)
        progressHandler?(.detectingGrid(progress: 1.0))

        // Step 3: Classify layout
        progressHandler?(.classifyingLayout)
        let layoutResult = try await classifyLayoutUseCase.execute(
            image: image,
            gridInfo: gridResult
        )

        // Step 4: Extract waveforms
        let gridCalibration = gridResult.toCalibration()
        let totalLeads = layoutResult.layout.totalLeads

        progressHandler?(.extractingWaveforms(currentLead: 0, totalLeads: totalLeads))

        let leads = try await extractWaveformsUseCase.execute(
            image: image,
            layout: layoutResult.layout,
            gridCalibration: gridCalibration,
            parameters: parameters
        )

        progressHandler?(.extractingWaveforms(currentLead: totalLeads, totalLeads: totalLeads))

        // Step 5: Validate results
        progressHandler?(.validatingResults)
        let validationStatus = try validateResultUseCase.execute(
            image: image,
            gridResult: gridResult,
            layout: layoutResult.layout,
            leads: leads,
            parameters: parameters
        )

        // Create recording
        let recording = ECGRecording(
            timestamp: Date(),
            originalImageData: image.jpegData(compressionQuality: 0.9),
            parameters: parameters,
            layout: layoutResult.layout,
            leads: leads,
            gridCalibration: gridCalibration,
            validationStatus: validationStatus
        )

        progressHandler?(.complete)

        return recording
    }
}
