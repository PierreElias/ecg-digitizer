import SwiftUI
import AVFoundation
import UniformTypeIdentifiers
import PhotosUI

/// Main capture flow view coordinating camera -> parameters -> processing -> results
struct CaptureFlowView: View {
    @StateObject private var viewModel = CaptureFlowViewModel()
    @State private var currentStep: CaptureStep = .camera

    enum CaptureStep {
        case camera
        case parameters
        case processing
        case results
    }

    var body: some View {
        NavigationStack {
            Group {
                switch currentStep {
                case .camera:
                    CameraView(
                        onCapture: { image in
                            viewModel.capturedImage = image
                            currentStep = .parameters
                        }
                    )

                case .parameters:
                    ParameterInputView(
                        image: viewModel.capturedImage,
                        onSubmit: { parameters in
                            viewModel.parameters = parameters
                            currentStep = .processing
                            Task {
                                await viewModel.processImage()
                            }
                        },
                        onRetake: {
                            viewModel.capturedImage = nil
                            currentStep = .camera
                        }
                    )

                case .processing:
                    ProcessingView(state: viewModel.processingState)
                        .onChange(of: viewModel.processingState) { _, newState in
                            if case .complete = newState {
                                currentStep = .results
                            } else if case .failed = newState {
                                // Stay on processing view to show error
                            }
                        }

                case .results:
                    if let recording = viewModel.recording {
                        ResultsView(
                            recording: recording,
                            onSave: {
                                Task {
                                    await viewModel.saveRecording()
                                }
                            },
                            onNewCapture: {
                                viewModel.reset()
                                currentStep = .camera
                            }
                        )
                    } else {
                        // Error state - show error and retry option
                        ErrorView(
                            error: viewModel.lastError,
                            onRetry: {
                                currentStep = .camera
                                viewModel.reset()
                            }
                        )
                    }
                }
            }
            .navigationTitle(navigationTitle)
            .navigationBarTitleDisplayMode(.inline)
            .alert("Error", isPresented: $viewModel.showError) {
                Button("Retry") {
                    currentStep = .camera
                    viewModel.reset()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                if let error = viewModel.lastError {
                    Text(error.message)
                }
            }
        }
    }

    private var navigationTitle: String {
        switch currentStep {
        case .camera:
            return "Capture ECG"
        case .parameters:
            return "Settings"
        case .processing:
            return "Processing"
        case .results:
            return "Results"
        }
    }
}

// MARK: - View Model

@MainActor
class CaptureFlowViewModel: ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var parameters: ProcessingParameters = .standard
    @Published var processingState: ProcessingState = .idle
    @Published var recording: ECGRecording?
    @Published var lastError: ValidationError?
    @Published var showError = false
    @Published var useServerProcessing = false // Use ONNX on-device processing
    @Published var wasQueuedOffline = false // Track if ECG was queued

    private let validator = ECGValidator()
    private let onDeviceProcessor = OnDeviceECGProcessor()
    private let apiClient = ECGAPIClient.shared

    // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    // Network monitoring and offline queue are CRITICAL for server-based processing
    private let networkMonitor = NetworkMonitor.shared
    private let offlineQueue = OfflineQueue.shared

    // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    // Main processing method now checks network and queues if offline
    func processImage() async {
        guard let image = capturedImage else { return }

        // Try server-side processing first (Open-ECG-Digitizer)
        if useServerProcessing {
            // Check network connectivity before attempting server processing
            if !networkMonitor.isConnected {
                // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
                // No network - queue for later processing
                offlineQueue.add(image: image, parameters: parameters)
                wasQueuedOffline = true
                processingState = .idle

                // Show user that ECG was queued
                lastError = ValidationError.unexpectedError(
                    message: "No network connection. ECG has been queued and will be processed when connection is restored. You have \(offlineQueue.pendingCount) ECG\(offlineQueue.pendingCount == 1 ? "" : "s") queued."
                )
                showError = true
                return
            }

            // Network available - check server health
            let serverAvailable = await apiClient.checkHealth()
            if serverAvailable {
                await processWithServer(image: image)
                return
            } else {
                // Server not available but network is - queue it
                offlineQueue.add(image: image, parameters: parameters)
                wasQueuedOffline = true
                processingState = .idle

                lastError = ValidationError.unexpectedError(
                    message: "Server is temporarily unavailable. ECG has been queued for processing. You have \(offlineQueue.pendingCount) ECG\(offlineQueue.pendingCount == 1 ? "" : "s") queued."
                )
                showError = true
                return
            }
        }

        // Fall back to local processing
        await processLocally(image: image)
    }

    /// Process image using server-side Open-ECG-Digitizer
    private func processWithServer(image: UIImage) async {
        do {
            let response = try await apiClient.digitize(
                image: image,
                parameters: parameters
            ) { [weak self] state in
                Task { @MainActor in
                    self?.processingState = state
                }
            }

            // Convert API response to ECGRecording
            recording = response.toRecording(
                originalImage: image,
                parameters: parameters
            )

            processingState = .complete

        } catch let error as ECGAPIError {
            // If server processing fails, try local processing
            print("Server processing failed: \(error.localizedDescription). Falling back to local processing.")
            await processLocally(image: image)
        } catch {
            let validationError = ValidationError.unexpectedError(message: error.localizedDescription)
            lastError = validationError
            processingState = .failed(validationError)
            showError = true
        }
    }

    /// Process image using local on-device ONNX processing
    private func processLocally(image: UIImage) async {
        do {
            // Use OnDeviceECGProcessor with ONNX Runtime
            let recording = try await onDeviceProcessor.processECG(
                image: image,
                parameters: parameters,
                progressCallback: { [weak self] state in
                    Task { @MainActor in
                        self?.processingState = state
                    }
                }
            )

            self.recording = recording
            processingState = .complete

        } catch let error as ECGProcessingError {
            let validationError = ValidationError.unexpectedError(message: error.localizedDescription)
            lastError = validationError
            processingState = .failed(validationError)
            showError = true
        } catch {
            let validationError = ValidationError.unexpectedError(message: error.localizedDescription)
            lastError = validationError
            processingState = .failed(validationError)
            showError = true
        }
    }

    func saveRecording() async {
        // Save to repository
        // In production, this would use the repository
    }

    func reset() {
        capturedImage = nil
        parameters = .standard
        processingState = .idle
        recording = nil
        lastError = nil
        showError = false
        wasQueuedOffline = false
    }
}

// MARK: - Camera View

struct CameraView: View {
    let onCapture: (UIImage) -> Void

    @StateObject private var cameraManager = CameraManager()
    @State private var showingImagePicker = false
    @State private var showingFilePicker = false
    @State private var showingCropView = false
    @State private var selectedImage: UIImage?
    @State private var isDragging = false
    @State private var cameraError: String?

    var body: some View {
        Group {
            #if targetEnvironment(macCatalyst)
            macCatalystView
            #else
            if let error = cameraError {
                errorView(error: error)
            } else if cameraManager.isCameraAvailable {
                cameraView
            } else {
                macCatalystView
            }
            #endif
        }
        .sheet(isPresented: $showingImagePicker) {
            ImagePicker(
                image: $selectedImage,
                onImageSelected: { image in
                    // Stop camera before transitioning to crop view
                    cameraManager.stopSession()
                    selectedImage = image
                    showingCropView = true
                }
            )
        }
        .fullScreenCover(isPresented: $showingCropView) {
            // Restart camera when crop view is dismissed
            Task {
                try? await cameraManager.startSession()
            }
        } content: {
            if let image = selectedImage {
                ImageCropView(
                    image: image,
                    onCrop: { croppedImage in
                        showingCropView = false
                        selectedImage = nil
                        onCapture(croppedImage)
                    },
                    onCancel: {
                        showingCropView = false
                        selectedImage = nil
                    }
                )
            }
        }
    }

    // MARK: - Mac Catalyst View

    private var macCatalystView: some View {
        ZStack {
            Color.backgroundPrimary
                .ignoresSafeArea()

            VStack(spacing: AppSpacing.xl) {
                Spacer()

                // App icon/header
                VStack(spacing: AppSpacing.md) {
                    ZStack {
                        Circle()
                            .fill(Color.brandPrimaryLight.opacity(0.2))
                            .frame(width: 120, height: 120)

                        Image(systemName: "waveform.path.ecg.rectangle")
                            .font(.system(size: 50))
                            .foregroundColor(.brandPrimary)
                    }

                    Text("ECG Digitizer")
                        .font(AppTypography.largeTitle)
                        .foregroundColor(.textPrimary)

                    Text("Transform paper ECGs into digital signals")
                        .font(AppTypography.body)
                        .foregroundColor(.textSecondary)
                }

                Spacer()

                // Drop zone
                VStack(spacing: AppSpacing.lg) {
                    Image(systemName: isDragging ? "arrow.down.doc.fill" : "doc.badge.plus")
                        .font(.system(size: 50))
                        .foregroundColor(isDragging ? .brandPrimary : .brandPrimaryLight)

                    Text(isDragging ? "Drop to import" : "Drag and drop ECG image here")
                        .font(AppTypography.headline)
                        .foregroundColor(isDragging ? .brandPrimary : .textSecondary)

                    Text("or")
                        .font(AppTypography.caption)
                        .foregroundColor(.textMuted)

                    Button {
                        showingImagePicker = true
                    } label: {
                        HStack(spacing: AppSpacing.sm) {
                            Image(systemName: "photo.on.rectangle")
                            Text("Choose from Photos")
                        }
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    .padding(.horizontal, AppSpacing.xl)
                }
                .frame(maxWidth: 400)
                .padding(AppSpacing.xxl)
                .background(Color.backgroundSecondary)
                .cornerRadius(AppRadius.xl)
                .overlay(
                    RoundedRectangle(cornerRadius: AppRadius.xl)
                        .strokeBorder(
                            isDragging ? Color.brandPrimary : Color.borderLight,
                            style: StrokeStyle(lineWidth: 2, dash: [8])
                        )
                )
                .primaryShadow()
                .onDrop(of: [.image], isTargeted: $isDragging) { providers in
                    handleDrop(providers: providers)
                }

                Spacer()

                // Supported formats
                Text("Supported formats: PNG, JPEG, HEIC, PDF")
                    .font(AppTypography.caption)
                    .foregroundColor(.textMuted)

                Spacer()
            }
            .padding(AppSpacing.lg)
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }

        provider.loadObject(ofClass: UIImage.self) { item, error in
            if let image = item as? UIImage {
                DispatchQueue.main.async {
                    onCapture(image)
                }
            }
        }
        return true
    }

    // MARK: - Camera View (iOS)

    private var cameraView: some View {
        ZStack {
            // Camera preview
            CameraPreviewView(cameraManager: cameraManager)
                .ignoresSafeArea()

            // Overlay
            VStack {
                Spacer()

                // Grid alignment guide
                GridOverlayGuide()
                    .padding(.horizontal, 20)

                Spacer()

                // Controls
                HStack(spacing: 60) {
                    // Gallery button
                    Button {
                        showingImagePicker = true
                    } label: {
                        Image(systemName: "photo.on.rectangle")
                            .font(.title)
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    // Capture button
                    Button {
                        Task {
                            do {
                                let image = try await cameraManager.capturePhoto()
                                // Stop camera before showing crop view
                                cameraManager.stopSession()
                                // Show crop view for camera photos too
                                await MainActor.run {
                                    selectedImage = image
                                    showingCropView = true
                                }
                            } catch {
                                print("Capture error: \(error)")
                                await MainActor.run {
                                    cameraError = error.localizedDescription
                                }
                            }
                        }
                    } label: {
                        Circle()
                            .fill(Color.white)
                            .frame(width: 80, height: 80)
                            .overlay(
                                Circle()
                                    .stroke(Color.white, lineWidth: 4)
                                    .frame(width: 70, height: 70)
                            )
                    }

                    // Flash button
                    Button {
                        cameraManager.toggleFlash()
                    } label: {
                        Image(systemName: flashIconName)
                            .font(.title)
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                }
                .padding(.bottom, 40)
            }
        }
        .onAppear {
            Task {
                do {
                    try await cameraManager.startSession()
                } catch {
                    await MainActor.run {
                        cameraError = error.localizedDescription
                    }
                }
            }
        }
        .onDisappear {
            cameraManager.stopSession()
        }
    }

    private var flashIconName: String {
        switch cameraManager.flashMode {
        case .off:
            return "bolt.slash.fill"
        case .auto:
            return "bolt.badge.automatic.fill"
        case .on:
            return "bolt.fill"
        @unknown default:
            return "bolt.badge.automatic.fill"
        }
    }

    private func errorView(error: String) -> some View {
        VStack(spacing: AppSpacing.lg) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundColor(.statusError)

            Text("Camera Error")
                .font(AppTypography.title2)
                .foregroundColor(.textPrimary)

            Text(error)
                .font(AppTypography.body)
                .foregroundColor(.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Button {
                showingImagePicker = true
            } label: {
                HStack {
                    Image(systemName: "photo.on.rectangle")
                    Text("Choose from Photos Instead")
                }
            }
            .buttonStyle(PrimaryButtonStyle())
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.backgroundPrimary)
    }
}

// MARK: - Camera Preview View

struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager

    func makeUIView(context: Context) -> CameraPreviewUIView {
        let view = CameraPreviewUIView()
        let previewLayer = cameraManager.makePreviewLayer()
        view.previewLayer = previewLayer
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: CameraPreviewUIView, context: Context) {
        // Frame will be updated in layoutSubviews
        DispatchQueue.main.async {
            uiView.setNeedsLayout()
        }
    }
}

// Custom UIView that properly manages preview layer frame
class CameraPreviewUIView: UIView {
    var previewLayer: AVCaptureVideoPreviewLayer?

    override func layoutSubviews() {
        super.layoutSubviews()
        // Update preview layer frame to match view bounds
        previewLayer?.frame = bounds
    }
}

// MARK: - Grid Overlay Guide

struct GridOverlayGuide: View {
    var body: some View {
        Rectangle()
            .stroke(Color.white.opacity(0.5), lineWidth: 2)
            .overlay(
                VStack(spacing: 0) {
                    ForEach(0..<3) { _ in
                        HStack(spacing: 0) {
                            ForEach(0..<4) { _ in
                                Rectangle()
                                    .stroke(Color.white.opacity(0.3), lineWidth: 1)
                            }
                        }
                    }
                }
            )
            .aspectRatio(4/3, contentMode: .fit)
    }
}

// MARK: - Image Picker (using PHPickerViewController for better scrolling)

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    let onImageSelected: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        config.preferredAssetRepresentationMode = .current

        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            guard let result = results.first else {
                parent.dismiss()
                return
            }

            if result.itemProvider.canLoadObject(ofClass: UIImage.self) {
                result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, error in
                    guard let self = self, let image = object as? UIImage else {
                        DispatchQueue.main.async {
                            self?.parent.dismiss()
                        }
                        return
                    }

                    DispatchQueue.main.async {
                        self.parent.image = image
                        self.parent.dismiss()

                        // Wait for sheet dismissal to complete before showing fullScreenCover
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                            self.parent.onImageSelected(image)
                        }
                    }
                }
            } else {
                parent.dismiss()
            }
        }
    }
}

// MARK: - Error View

struct ErrorView: View {
    let error: ValidationError?
    let onRetry: () -> Void

    var body: some View {
        ZStack {
            Color.backgroundPrimary
                .ignoresSafeArea()

            VStack(spacing: AppSpacing.lg) {
                Spacer()

                ZStack {
                    Circle()
                        .fill(Color.statusError.opacity(0.1))
                        .frame(width: 100, height: 100)

                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.statusError)
                }

                if let error = error {
                    VStack(spacing: AppSpacing.sm) {
                        Text(error.title)
                            .font(AppTypography.title2)
                            .foregroundColor(.textPrimary)

                        Text(error.message)
                            .font(AppTypography.body)
                            .foregroundColor(.textSecondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.horizontal, AppSpacing.lg)

                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        Text("Suggestions")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        ForEach(error.recoverySuggestions, id: \.self) { suggestion in
                            HStack(alignment: .top, spacing: AppSpacing.sm) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.statusSuccess)
                                    .font(.system(size: 16))
                                Text(suggestion)
                                    .font(AppTypography.subheadline)
                                    .foregroundColor(.textSecondary)
                            }
                        }
                    }
                    .padding(AppSpacing.md)
                    .background(Color.backgroundTertiary)
                    .cornerRadius(AppRadius.md)
                    .padding(.horizontal, AppSpacing.lg)
                }

                Spacer()

                Button {
                    onRetry()
                } label: {
                    HStack(spacing: AppSpacing.sm) {
                        Image(systemName: "camera.fill")
                        Text("Try Again")
                    }
                }
                .buttonStyle(PrimaryButtonStyle())
                .padding(.horizontal, AppSpacing.lg)
                .padding(.bottom, AppSpacing.xl)
            }
        }
    }
}

#Preview {
    CaptureFlowView()
}
