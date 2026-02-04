import Foundation
import AVFoundation
import UIKit
import Combine

/// Manages camera capture for ECG photos
@MainActor
final class CameraManager: NSObject, ObservableObject {

    // MARK: - Published Properties

    @Published var capturedImage: UIImage?
    @Published var isSessionRunning = false
    @Published var error: CameraError?
    @Published var flashMode: AVCaptureDevice.FlashMode = .auto
    @Published var isCameraAvailable = true

    // MARK: - Private Properties

    private let captureSession = AVCaptureSession()
    private var photoOutput = AVCapturePhotoOutput()
    private var videoDeviceInput: AVCaptureDeviceInput?
    private var photoContinuation: CheckedContinuation<UIImage, Error>?

    private let sessionQueue = DispatchQueue(label: "com.ecgdigitizer.camera")

    // MARK: - Configuration

    struct Config {
        // Use .high instead of .photo to reduce memory usage
        // .photo = 12MP (4032×3024) = ~48MB uncompressed
        // .high = ~2MP (1920×1080) = ~8MB uncompressed
        // ECG images don't need 12MP - 2MP is plenty for signal extraction
        static let sessionPreset: AVCaptureSession.Preset = .high
        static let photoQualityPrioritization: AVCapturePhotoOutput.QualityPrioritization = .balanced

        // Maximum image dimension for processing (saves memory)
        static let maxImageDimension: CGFloat = 2048
    }

    // MARK: - Platform Detection

    static var isRunningOnMac: Bool {
        #if targetEnvironment(macCatalyst)
        return true
        #else
        return ProcessInfo.processInfo.isMacCatalystApp
        #endif
    }

    // MARK: - Initialization

    override init() {
        super.init()
        checkCameraAvailability()
    }

    private func checkCameraAvailability() {
        #if targetEnvironment(macCatalyst)
        // On Mac Catalyst, camera may not be available
        isCameraAvailable = AVCaptureDevice.default(for: .video) != nil
        #else
        isCameraAvailable = true
        #endif
    }

    // MARK: - Session Management

    /// Configures and starts the camera session
    func startSession() async throws {
        // Check authorization
        let status = AVCaptureDevice.authorizationStatus(for: .video)

        switch status {
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if !granted {
                throw CameraError.notAuthorized
            }
        case .denied, .restricted:
            throw CameraError.notAuthorized
        case .authorized:
            break
        @unknown default:
            throw CameraError.notAuthorized
        }

        // Configure session
        try await configureSession()

        // Start session
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.captureSession.startRunning()
            Task { @MainActor in
                self.isSessionRunning = true
            }
        }
    }

    /// Stops the camera session
    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.captureSession.stopRunning()
            Task { @MainActor in
                self.isSessionRunning = false
            }
        }
    }

    /// Configures the capture session
    private func configureSession() async throws {
        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        captureSession.sessionPreset = Config.sessionPreset

        // Add video input
        guard let videoDevice = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ) else {
            throw CameraError.cameraUnavailable
        }

        do {
            let videoInput = try AVCaptureDeviceInput(device: videoDevice)

            if captureSession.canAddInput(videoInput) {
                captureSession.addInput(videoInput)
                videoDeviceInput = videoInput
            } else {
                throw CameraError.inputError
            }
        } catch {
            throw CameraError.inputError
        }

        // Add photo output
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)

            // DON'T enable high resolution capture - it causes memory crashes on iPhone
            // The session preset .high gives us 1920x1080 which is sufficient for ECG
            photoOutput.maxPhotoQualityPrioritization = Config.photoQualityPrioritization

            // Configure for document capture
            if let connection = photoOutput.connection(with: .video) {
                connection.videoRotationAngle = 90  // Portrait orientation
            }
        } else {
            throw CameraError.outputError
        }

        // Configure video device
        try configureVideoDevice(videoDevice)
    }

    /// Configures video device settings
    private func configureVideoDevice(_ device: AVCaptureDevice) throws {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        // Enable auto-focus
        if device.isFocusModeSupported(.continuousAutoFocus) {
            device.focusMode = .continuousAutoFocus
        }

        // Enable auto-exposure
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }

        // Enable auto white balance
        if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            device.whiteBalanceMode = .continuousAutoWhiteBalance
        }
    }

    // MARK: - Photo Capture

    /// Captures a photo
    func capturePhoto() async throws -> UIImage {
        return try await withCheckedThrowingContinuation { continuation in
            self.photoContinuation = continuation

            let settings = AVCapturePhotoSettings()
            settings.flashMode = flashMode

            // DON'T request high resolution - it uses too much memory on iPhone
            // The .high session preset already gives us 1920x1080 which is plenty
            // Requesting max dimensions would give 12MP which causes memory crashes

            // Use HEIF for better compression if available
            if photoOutput.availablePhotoCodecTypes.contains(.hevc) {
                settings.photoQualityPrioritization = .balanced  // Balance quality vs memory
            }

            photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }

    /// Toggles flash mode
    func toggleFlash() {
        switch flashMode {
        case .off:
            flashMode = .auto
        case .auto:
            flashMode = .on
        case .on:
            flashMode = .off
        @unknown default:
            flashMode = .auto
        }
    }

    // MARK: - Focus and Exposure

    /// Focuses on a specific point
    func focus(at point: CGPoint) {
        guard let device = videoDeviceInput?.device else { return }

        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }

            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = point
                device.focusMode = .autoFocus
            }

            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = point
                device.exposureMode = .autoExpose
            }
        } catch {
            print("Failed to set focus: \(error)")
        }
    }

    // MARK: - Preview Layer

    /// Returns a preview layer for the camera
    func makePreviewLayer() -> AVCaptureVideoPreviewLayer {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        return layer
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension CameraManager: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        Task { @MainActor in
            if let error = error {
                self.photoContinuation?.resume(throwing: CameraError.captureError(error.localizedDescription))
                self.photoContinuation = nil
                return
            }

            guard let imageData = photo.fileDataRepresentation(),
                  let image = UIImage(data: imageData) else {
                self.photoContinuation?.resume(throwing: CameraError.processingError)
                self.photoContinuation = nil
                return
            }

            // Fix orientation and resize to save memory
            // This is critical for iPhone memory limits
            let correctedImage = image.fixOrientationAndResize(maxDimension: Config.maxImageDimension)

            self.capturedImage = correctedImage
            self.photoContinuation?.resume(returning: correctedImage)
            self.photoContinuation = nil
        }
    }
}

// MARK: - Camera Errors

enum CameraError: Error, LocalizedError {
    case notAuthorized
    case cameraUnavailable
    case inputError
    case outputError
    case captureError(String)
    case processingError

    var errorDescription: String? {
        switch self {
        case .notAuthorized:
            return "Camera access not authorized. Please enable camera access in Settings."
        case .cameraUnavailable:
            return "Camera is not available on this device."
        case .inputError:
            return "Failed to configure camera input."
        case .outputError:
            return "Failed to configure photo output."
        case .captureError(let message):
            return "Photo capture failed: \(message)"
        case .processingError:
            return "Failed to process captured photo."
        }
    }
}

// MARK: - UIImage Extension

extension UIImage {
    /// Fixes image orientation to always be .up
    func fixOrientation() -> UIImage {
        if imageOrientation == .up {
            return self
        }

        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return normalizedImage ?? self
    }

    /// Fixes orientation AND resizes to save memory
    /// Combines both operations in one pass to avoid creating intermediate copies
    func fixOrientationAndResize(maxDimension: CGFloat) -> UIImage {
        // Calculate new size maintaining aspect ratio
        var newSize = size

        if size.width > maxDimension || size.height > maxDimension {
            let scale = maxDimension / max(size.width, size.height)
            newSize = CGSize(width: size.width * scale, height: size.height * scale)
        }

        // If already correct orientation and size, return self
        if imageOrientation == .up && newSize == size {
            return self
        }

        // Use scale 1.0 to avoid Retina scaling which uses more memory
        UIGraphicsBeginImageContextWithOptions(newSize, true, 1.0)
        draw(in: CGRect(origin: .zero, size: newSize))
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return result ?? self
    }

    /// Rotates image by degrees
    func rotated(by degrees: CGFloat) -> UIImage? {
        let radians = degrees * .pi / 180

        var newSize = CGRect(origin: .zero, size: size)
            .applying(CGAffineTransform(rotationAngle: radians))
            .integral.size

        newSize.width = floor(newSize.width)
        newSize.height = floor(newSize.height)

        UIGraphicsBeginImageContextWithOptions(newSize, false, scale)
        guard let context = UIGraphicsGetCurrentContext() else { return nil }

        context.translateBy(x: newSize.width / 2, y: newSize.height / 2)
        context.rotate(by: radians)
        draw(in: CGRect(
            x: -size.width / 2,
            y: -size.height / 2,
            width: size.width,
            height: size.height
        ))

        let rotatedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return rotatedImage
    }
}
