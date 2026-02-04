import SwiftUI
import PhotosUI

/// Settings view for configuring server connection and processing mode
struct SettingsView: View {
    @StateObject private var apiClient = ECGAPIClient.shared
    @ObservedObject private var processor = OnDeviceECGProcessor.shared
    @AppStorage("serverURL") private var serverURL: String = "http://localhost:8080"
    @AppStorage("aptibleURL") private var aptibleURL: String = "https://api-uat.pathwaihealth.com/ecg/predict"
    @State private var showingSuccess = false
    @State private var showingError = false
    @State private var errorMessage = ""
    @Environment(\.dismiss) private var dismiss

    // Diagnostic states
    @State private var showingDiagnostic = false
    @State private var diagnosticReport: String = ""
    @State private var visualDiagnosticReport: VisualDiagnosticReport?
    @State private var isRunningDiagnostic = false
    @State private var showingPhotoPicker = false
    @State private var selectedPhotoItem: PhotosPickerItem?

    private var processingModeBinding: Binding<String> {
        Binding(
            get: { processor.processingMode.rawValue },
            set: { newValue in
                if let mode = OnDeviceECGProcessor.ProcessingMode(rawValue: newValue) {
                    processor.processingMode = mode
                }
            }
        )
    }

    private var processingModeDescription: String {
        switch processor.processingMode {
        case .onDevice:
            return "All processing on iPhone using ONNX models. Works offline but may be less accurate."
        case .server:
            return "Full image sent to Python server. Requires connection but most accurate."
        case .hybrid:
            return "ONNX segmentation on iPhone + Python extraction on server. Best balance of speed and accuracy."
        }
    }

    var body: some View {
        NavigationView {
            ZStack {
                Color.backgroundPrimary
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: AppSpacing.xl) {
                        // Connection Status
                        VStack(spacing: AppSpacing.md) {
                        Image(systemName: apiClient.isConnected ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .font(.system(size: 48))
                            .foregroundColor(apiClient.isConnected ? .statusSuccess : .statusError)

                        Text(apiClient.isConnected ? "Connected" : "Disconnected")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        Text(apiClient.baseURL.absoluteString)
                            .font(AppTypography.caption)
                            .foregroundColor(.textSecondary)
                    }
                    .padding(AppSpacing.xl)
                    .background(Color.backgroundSecondary)
                    .cornerRadius(AppRadius.lg)

                    // Processing Mode Toggle
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        Text("Processing Mode")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        Picker("Mode", selection: processingModeBinding) {
                            Text("Hybrid (Recommended)").tag("hybrid")
                            Text("On-Device Only").tag("onDevice")
                            Text("Server Only").tag("server")
                        }
                        .pickerStyle(.segmented)

                        Text(processingModeDescription)
                            .font(AppTypography.caption)
                            .foregroundColor(.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(AppSpacing.md)
                    .background(Color.backgroundSecondary)
                    .cornerRadius(AppRadius.lg)

                    // Quick Select
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        Text("Quick Select")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        Button {
                            serverURL = "http://localhost:8080"
                            connectToServer()
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                                    Text("Simulator / Mac")
                                        .font(AppTypography.body)
                                        .foregroundColor(.textPrimary)
                                    Text("http://localhost:8080")
                                        .font(AppTypography.caption)
                                        .foregroundColor(.textSecondary)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .foregroundColor(.textMuted)
                            }
                            .padding(AppSpacing.md)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.md)
                        }

                        Button {
                            serverURL = "http://10.118.154.141:8080"
                            connectToServer()
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                                    Text("Physical Device")
                                        .font(AppTypography.body)
                                        .foregroundColor(.textPrimary)
                                    Text("http://10.118.154.141:8080")
                                        .font(AppTypography.caption)
                                        .foregroundColor(.textSecondary)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .foregroundColor(.textMuted)
                            }
                            .padding(AppSpacing.md)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.md)
                        }

                        Button {
                            serverURL = "https://ecg-digitizer.fly.dev"
                            connectToServer()
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                                    Text("Cloud Server (Fly.io)")
                                        .font(AppTypography.body)
                                        .foregroundColor(.textPrimary)
                                    Text("https://ecg-digitizer.fly.dev")
                                        .font(AppTypography.caption)
                                        .foregroundColor(.textSecondary)
                                }
                                Spacer()
                                Image(systemName: "cloud.fill")
                                    .foregroundColor(.blue)
                                Image(systemName: "chevron.right")
                                    .foregroundColor(.textMuted)
                            }
                            .padding(AppSpacing.md)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.md)
                        }
                    }

                    // Custom URL
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        Text("Digitizer Server URL")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        TextField("http://192.168.1.x:8080", text: $serverURL)
                            .textFieldStyle(.plain)
                            .padding(AppSpacing.md)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.md)
                            .autocapitalization(.none)
                            .keyboardType(.URL)

                        Button {
                            connectToServer()
                        } label: {
                            Text("Connect")
                        }
                        .buttonStyle(PrimaryButtonStyle())
                    }

                    // Aptible Diagnostic API URL
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        Text("Diagnostic API URL (Aptible)")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        Text("Enter the URL of your Aptible-hosted diagnostic service endpoint")
                            .font(AppTypography.caption)
                            .foregroundColor(.textSecondary)

                        TextField("https://your-app.aptible.in/api/diagnose", text: $aptibleURL)
                            .textFieldStyle(.plain)
                            .padding(AppSpacing.md)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.md)
                            .autocapitalization(.none)
                            .keyboardType(.URL)

                        if !aptibleURL.isEmpty {
                            HStack(spacing: AppSpacing.xs) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.statusSuccess)
                                    .font(.caption)
                                Text("Diagnostic API configured")
                                    .font(AppTypography.caption)
                                    .foregroundColor(.textSecondary)
                            }
                        }
                    }

                    // ONNX Diagnostic Section
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        Text("ONNX Diagnostics")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        Text("Run detailed diagnostic on ONNX segmentation pipeline to analyze preprocessing, inference, and postprocessing steps.")
                            .font(AppTypography.caption)
                            .foregroundColor(.textSecondary)

                        PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                            HStack {
                                if isRunningDiagnostic {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle())
                                        .scaleEffect(0.8)
                                    Text("Running Diagnostic...")
                                } else {
                                    Image(systemName: "waveform.path.ecg")
                                    Text("Run ONNX Diagnostic")
                                }
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(isRunningDiagnostic)
                        .onChange(of: selectedPhotoItem) { oldValue, newValue in
                            if let newValue = newValue {
                                runDiagnostic(with: newValue)
                            }
                        }
                    }
                    .padding(AppSpacing.md)
                    .background(Color.backgroundSecondary)
                    .cornerRadius(AppRadius.lg)
                    }
                    .padding(AppSpacing.xl)
                }
            }
            .sheet(isPresented: $showingDiagnostic) {
                if let visualReport = visualDiagnosticReport {
                    VisualDiagnosticReportView(report: visualReport)
                } else {
                    DiagnosticReportView(report: diagnosticReport)
                }
            }
            .navigationTitle("Server Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .alert("Connected", isPresented: $showingSuccess) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("Successfully connected to server")
            }
            .alert("Connection Failed", isPresented: $showingError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(errorMessage)
            }
        }
        .onAppear {
            serverURL = apiClient.baseURL.absoluteString
        }
    }

    private func connectToServer() {
        Task {
            apiClient.setServerURL(serverURL)

            // Wait a bit for connection check
            try? await Task.sleep(nanoseconds: 1_000_000_000)

            if apiClient.isConnected {
                showingSuccess = true
            } else {
                errorMessage = "Could not connect to server. Make sure the server is running and the URL is correct."
                showingError = true
            }
        }
    }

    private func runDiagnostic(with item: PhotosPickerItem) {
        isRunningDiagnostic = true

        Task {
            do {
                // Load the image
                guard let data = try await item.loadTransferable(type: Data.self),
                      let image = UIImage(data: data) else {
                    await MainActor.run {
                        isRunningDiagnostic = false
                        errorMessage = "Failed to load selected image"
                        showingError = true
                    }
                    return
                }

                // Run visual diagnostic (with step-by-step images)
                let report = try await processor.generateVisualDiagnosticReport(image: image)

                await MainActor.run {
                    isRunningDiagnostic = false
                    visualDiagnosticReport = report
                    diagnosticReport = report.textReport
                    showingDiagnostic = true
                    selectedPhotoItem = nil
                }

            } catch {
                await MainActor.run {
                    isRunningDiagnostic = false
                    errorMessage = "Diagnostic failed: \(error.localizedDescription)"
                    showingError = true
                    selectedPhotoItem = nil
                }
            }
        }
    }
}

// MARK: - Visual Diagnostic Report View

struct VisualDiagnosticReportView: View {
    let report: VisualDiagnosticReport
    @Environment(\.dismiss) private var dismiss
    @State private var showingShareSheet = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: AppSpacing.lg) {
                    // Step 1: Input Image
                    DiagnosticImageSection(
                        title: "Step 1: Input Image",
                        subtitle: "\(report.metrics.inputWidth) × \(report.metrics.inputHeight) → \(report.metrics.processedWidth) × \(report.metrics.processedHeight)",
                        image: report.inputImage
                    )

                    // Step 2: Signal Probability
                    DiagnosticImageSection(
                        title: "Step 2: Signal Probability",
                        subtitle: "Coverage: \(String(format: "%.1f", report.metrics.signalMetrics.coverage))% → \(String(format: "%.1f", report.metrics.signalMetrics.postDilationCoverage ?? 0))% (after dilation)",
                        image: report.signalHeatmap
                    )

                    // Step 3: Grid Probability
                    DiagnosticImageSection(
                        title: "Step 3: Grid Probability",
                        subtitle: "Coverage: \(String(format: "%.1f", report.metrics.gridMetrics.coverage))%",
                        image: report.gridHeatmap
                    )

                    // Step 4: Text Probability
                    DiagnosticImageSection(
                        title: "Step 4: Text Probability",
                        subtitle: "Coverage: \(String(format: "%.1f", report.metrics.textMetrics.coverage))%",
                        image: report.textHeatmap
                    )

                    // Step 5: RGB Overlay
                    DiagnosticImageSection(
                        title: "Step 5: RGB Overlay",
                        subtitle: "R=Grid, G=Signal, B=Text",
                        image: report.rgbOverlay
                    )

                    // Step 6: Lead Extraction
                    DiagnosticImageSection(
                        title: "Step 6: Lead Extraction",
                        subtitle: leadExtractionSubtitle(report.leadExtractionDetails),
                        image: report.leadExtractionImage
                    )

                    // Step 6b: Sectioned Leads
                    DiagnosticImageSection(
                        title: "Step 6b: Sectioned Leads (3×4 Grid)",
                        subtitle: "Individual leads after sectioning",
                        image: report.sectionedLeadsImage
                    )

                    // Step 7: Final 12-Lead Output
                    DiagnosticImageSection(
                        title: "Step 7: Final 12-Lead ECG",
                        subtitle: "This is what appears in the app results",
                        image: report.final12LeadPlot
                    )

                    // Timing Metrics
                    VStack(alignment: .leading, spacing: AppSpacing.sm) {
                        Text("Timing")
                            .font(AppTypography.headline)
                            .foregroundColor(.textPrimary)

                        HStack {
                            MetricPill(label: "Preprocess", value: "\(Int(report.metrics.preprocessingTimeMs))ms")
                            MetricPill(label: "Inference", value: "\(Int(report.metrics.inferenceTimeMs))ms")
                            MetricPill(label: "Postprocess", value: "\(Int(report.metrics.postprocessingTimeMs))ms")
                        }

                        Text("Total: \(Int(report.metrics.totalTimeMs))ms")
                            .font(AppTypography.caption)
                            .foregroundColor(.textSecondary)
                    }
                    .padding()
                    .background(Color.backgroundSecondary)
                    .cornerRadius(AppRadius.md)
                }
                .padding()
            }
            .background(Color.backgroundPrimary)
            .navigationTitle("ONNX Visual Diagnostic")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showingShareSheet = true
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
            .sheet(isPresented: $showingShareSheet) {
                ShareSheet(items: [report.textReport])
            }
        }
    }

    private func leadExtractionSubtitle(_ details: LeadExtractionDetails?) -> String {
        guard let details = details else {
            return "Lead extraction failed"
        }
        return "\(details.rowCount) rows detected, \(details.extractedSamples) samples per lead"
    }
}

// MARK: - Diagnostic Image Section

struct DiagnosticImageSection: View {
    let title: String
    let subtitle: String
    let image: UIImage?

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.sm) {
            Text(title)
                .font(AppTypography.headline)
                .foregroundColor(.textPrimary)

            Text(subtitle)
                .font(AppTypography.caption)
                .foregroundColor(.textSecondary)

            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .cornerRadius(AppRadius.md)
                    .overlay(
                        RoundedRectangle(cornerRadius: AppRadius.md)
                            .stroke(Color.borderLight, lineWidth: 1)
                    )
            } else {
                Text("Image not available")
                    .font(AppTypography.caption)
                    .foregroundColor(.textMuted)
                    .frame(maxWidth: .infinity, minHeight: 100)
                    .background(Color.backgroundSecondary)
                    .cornerRadius(AppRadius.md)
            }
        }
        .padding()
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.lg)
    }
}

// MARK: - Metric Pill

struct MetricPill: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(.caption, design: .monospaced).bold())
                .foregroundColor(.textPrimary)
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(.textSecondary)
        }
        .padding(.horizontal, AppSpacing.sm)
        .padding(.vertical, AppSpacing.xs)
        .background(Color.backgroundPrimary)
        .cornerRadius(AppRadius.sm)
    }
}

// MARK: - Text Diagnostic Report View

struct DiagnosticReportView: View {
    let report: String
    @Environment(\.dismiss) private var dismiss
    @State private var showingShareSheet = false

    var body: some View {
        NavigationView {
            ScrollView {
                Text(report)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.textPrimary)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color.backgroundPrimary)
            .navigationTitle("ONNX Diagnostic Report")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showingShareSheet = true
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
            .sheet(isPresented: $showingShareSheet) {
                ShareSheet(items: [report])
            }
        }
    }
}

#Preview {
    SettingsView()
}
