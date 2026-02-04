import Foundation
import SwiftUI

/// Dependency injection container for managing app dependencies
@MainActor
final class DependencyContainer: ObservableObject {

    // MARK: - Network

    lazy var apiClient: ECGAPIClient = {
        ECGAPIClient.shared
    }()

    // MARK: - Infrastructure

    lazy var imagePreprocessor: ImagePreprocessor = {
        ImagePreprocessor()
    }()

    lazy var gridDetector: GridDetector = {
        GridDetector(visionManager: visionManager)
    }()

    lazy var layoutClassifier: LayoutClassifier = {
        LayoutClassifier()
    }()

    lazy var waveformExtractor: WaveformExtractor = {
        WaveformExtractor()
    }()

    lazy var ecgValidator: ECGValidator = {
        ECGValidator()
    }()

    lazy var visionManager: VisionManager = {
        VisionManager()
    }()

    lazy var cameraManager: CameraManager = {
        CameraManager()
    }()

    // MARK: - Data Layer

    lazy var fileStorageManager: FileStorageManager = {
        FileStorageManager()
    }()

    lazy var ecgRecordingRepository: ECGRecordingRepository = {
        ECGRecordingRepository(fileStorage: fileStorageManager)
    }()

    // MARK: - Use Cases

    lazy var validateImageUseCase: ValidateImageUseCase = {
        ValidateImageUseCase(validator: ecgValidator)
    }()

    lazy var detectGridUseCase: DetectGridUseCase = {
        DetectGridUseCase(
            preprocessor: imagePreprocessor,
            gridDetector: gridDetector,
            validator: ecgValidator
        )
    }()

    lazy var classifyLayoutUseCase: ClassifyLayoutUseCase = {
        ClassifyLayoutUseCase(
            classifier: layoutClassifier,
            validator: ecgValidator
        )
    }()

    lazy var extractWaveformsUseCase: ExtractWaveformsUseCase = {
        ExtractWaveformsUseCase(
            extractor: waveformExtractor,
            validator: ecgValidator
        )
    }()

    lazy var saveRecordingUseCase: SaveRecordingUseCase = {
        SaveRecordingUseCase(repository: ecgRecordingRepository)
    }()

    // MARK: - Initialization

    init() {
        // Any initialization logic
    }
}
