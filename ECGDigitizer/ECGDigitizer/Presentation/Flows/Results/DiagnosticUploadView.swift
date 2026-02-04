import SwiftUI

/// View for uploading ECG waveform data to Aptible diagnostic API
struct DiagnosticUploadView: View {
    let recording: ECGRecording
    @Environment(\.dismiss) private var dismiss
    @AppStorage("aptibleURL") private var aptibleURL: String = ""

    @State private var isUploading = false
    @State private var uploadProgress: Double = 0.0
    @State private var diagnosticReport: ECGAPIClient.DiagnosticReportResponse?
    @State private var error: Error?
    @State private var showError = false

    @StateObject private var apiClient = ECGAPIClient.shared

    var body: some View {
        NavigationStack {
            ZStack {
                Color.backgroundPrimary
                    .ignoresSafeArea()

                if isUploading {
                    uploadingView
                } else if let report = diagnosticReport {
                    reportView(report: report)
                } else {
                    readyToUploadView
                }
            }
            .navigationTitle("Diagnostic Analysis")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                    .foregroundColor(.brandPrimary)
                }
            }
            .alert("Upload Error", isPresented: $showError) {
                Button("OK", role: .cancel) {}
            } message: {
                if let error = error {
                    Text(error.localizedDescription)
                }
            }
        }
    }

    // MARK: - Ready to Upload View

    private var readyToUploadView: some View {
        VStack(spacing: AppSpacing.xl) {
            Spacer()

            // Icon
            ZStack {
                Circle()
                    .fill(Color.brandPrimaryLight.opacity(0.2))
                    .frame(width: 120, height: 120)

                Image(systemName: "waveform.path.ecg")
                    .font(.system(size: 50))
                    .foregroundColor(.brandPrimary)
            }

            // Info
            VStack(spacing: AppSpacing.md) {
                Text("AI-Powered ECG Analysis")
                    .font(AppTypography.title2)
                    .foregroundColor(.textPrimary)

                Text("Upload your digitized ECG waveform data to receive a comprehensive diagnostic report with AI-powered analysis.")
                    .font(AppTypography.body)
                    .foregroundColor(.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, AppSpacing.xl)
            }

            // Recording details
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                HStack {
                    Image(systemName: "waveform")
                        .foregroundColor(.brandPrimary)
                    Text("\(recording.leads.count) Leads")
                        .font(AppTypography.subheadline)
                        .foregroundColor(.textPrimary)
                }

                HStack {
                    Image(systemName: "clock")
                        .foregroundColor(.brandPrimary)
                    Text(String(format: "%.1f seconds", recording.totalDuration))
                        .font(AppTypography.subheadline)
                        .foregroundColor(.textPrimary)
                }

                HStack {
                    Image(systemName: "square.grid.3x3")
                        .foregroundColor(.brandPrimary)
                    Text(recording.layout.displayName)
                        .font(AppTypography.subheadline)
                        .foregroundColor(.textPrimary)
                }
            }
            .padding(AppSpacing.md)
            .background(Color.backgroundSecondary)
            .cornerRadius(AppRadius.lg)
            .primaryShadow()

            Spacer()

            // Upload button
            if aptibleURL.isEmpty {
                VStack(spacing: AppSpacing.sm) {
                    Text("No diagnostic API configured")
                        .font(AppTypography.caption)
                        .foregroundColor(.statusError)

                    Text("Please configure the Aptible diagnostic API URL in Settings")
                        .font(AppTypography.caption)
                        .foregroundColor(.textMuted)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, AppSpacing.xl)
            } else {
                Button {
                    uploadForDiagnosis()
                } label: {
                    HStack(spacing: AppSpacing.sm) {
                        Image(systemName: "arrow.up.circle.fill")
                        Text("Upload for Analysis")
                    }
                }
                .buttonStyle(PrimaryButtonStyle())
                .padding(.horizontal, AppSpacing.xl)
            }

            Spacer()
        }
    }

    // MARK: - Uploading View

    private var uploadingView: some View {
        VStack(spacing: AppSpacing.xl) {
            Spacer()

            BrandLoadingView()
                .scaleEffect(1.5)

            VStack(spacing: AppSpacing.sm) {
                Text("Analyzing ECG...")
                    .font(AppTypography.headline)
                    .foregroundColor(.textPrimary)

                Text("This may take a few moments")
                    .font(AppTypography.caption)
                    .foregroundColor(.textSecondary)
            }

            // Progress indicator
            ProgressView(value: uploadProgress)
                .progressViewStyle(.linear)
                .tint(.brandPrimary)
                .frame(width: 200)

            Spacer()
        }
    }

    // MARK: - Report View

    private func reportView(report: ECGAPIClient.DiagnosticReportResponse) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: AppSpacing.lg) {
                // Summary
                if let summary = report.summary {
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        SectionHeader("Summary")

                        Text(summary)
                            .font(AppTypography.body)
                            .foregroundColor(.textPrimary)
                            .padding(AppSpacing.md)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.lg)
                            .primaryShadow()
                    }
                }

                // Confidence score
                if let confidence = report.confidence {
                    HStack {
                        Text("Confidence Score")
                            .font(AppTypography.subheadline)
                            .foregroundColor(.textSecondary)
                        Spacer()
                        Text(String(format: "%.0f%%", confidence * 100))
                            .font(AppTypography.headline)
                            .foregroundColor(confidenceColor(confidence))
                    }
                    .padding(AppSpacing.md)
                    .background(Color.backgroundSecondary)
                    .cornerRadius(AppRadius.lg)
                    .primaryShadow()
                }

                // Findings
                if let findings = report.findings, !findings.isEmpty {
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        SectionHeader("Findings")

                        ForEach(Array(findings.enumerated()), id: \.offset) { _, finding in
                            FindingCard(finding: finding)
                        }
                    }
                }

                // Measurements
                if let measurements = report.measurements {
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        SectionHeader("Measurements")

                        VStack(spacing: AppSpacing.sm) {
                            if let hr = measurements.heartRate {
                                MeasurementRow(
                                    label: "Heart Rate",
                                    value: "\(hr) BPM",
                                    icon: "heart.fill"
                                )
                            }

                            if let pr = measurements.prInterval {
                                MeasurementRow(
                                    label: "PR Interval",
                                    value: "\(pr) ms",
                                    icon: "waveform.path"
                                )
                            }

                            if let qrs = measurements.qrsDuration {
                                MeasurementRow(
                                    label: "QRS Duration",
                                    value: "\(qrs) ms",
                                    icon: "waveform.path"
                                )
                            }

                            if let qt = measurements.qtInterval {
                                MeasurementRow(
                                    label: "QT Interval",
                                    value: "\(qt) ms",
                                    icon: "waveform.path"
                                )
                            }

                            if let qtc = measurements.qtcInterval {
                                MeasurementRow(
                                    label: "QTc Interval",
                                    value: "\(qtc) ms",
                                    icon: "waveform.path"
                                )
                            }
                        }
                        .padding(AppSpacing.md)
                        .background(Color.backgroundSecondary)
                        .cornerRadius(AppRadius.lg)
                        .primaryShadow()
                    }
                }

                // HTML Report
                if report.reportHtml != nil {
                    VStack(alignment: .leading, spacing: AppSpacing.md) {
                        SectionHeader("Full Report")

                        // TODO: Display HTML report in a WebView
                        Text("HTML report available")
                            .font(AppTypography.caption)
                            .foregroundColor(.textMuted)
                            .padding(AppSpacing.md)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.lg)
                    }
                }

                // Download PDF button
                if let _ = report.reportPdf {
                    Button {
                        downloadPDFReport(report)
                    } label: {
                        HStack(spacing: AppSpacing.sm) {
                            Image(systemName: "arrow.down.doc.fill")
                            Text("Download PDF Report")
                        }
                    }
                    .buttonStyle(SecondaryButtonStyle())
                }
            }
            .padding(AppSpacing.md)
        }
    }

    // MARK: - Helper Functions

    private func uploadForDiagnosis() {
        guard let url = URL(string: aptibleURL) else {
            error = ECGAPIError.invalidURL
            showError = true
            return
        }

        isUploading = true
        uploadProgress = 0.0

        Task {
            do {
                let report = try await apiClient.uploadWaveformForDiagnosis(
                    recording: recording,
                    aptibleURL: url,
                    progressCallback: { state in
                        Task { @MainActor in
                            // Update progress based on state
                            switch state {
                            case .validatingImage:
                                uploadProgress = 0.2
                            case .detectingGrid:
                                uploadProgress = 0.4
                            case .classifyingLayout:
                                uploadProgress = 0.6
                            case .extractingWaveforms:
                                uploadProgress = 0.8
                            case .validatingResults:
                                uploadProgress = 0.9
                            case .complete:
                                uploadProgress = 1.0
                            default:
                                break
                            }
                        }
                    }
                )

                await MainActor.run {
                    isUploading = false
                    diagnosticReport = report
                }
            } catch {
                await MainActor.run {
                    isUploading = false
                    self.error = error
                    showError = true
                }
            }
        }
    }

    private func downloadPDFReport(_ report: ECGAPIClient.DiagnosticReportResponse) {
        guard let pdfBase64 = report.reportPdf,
              let pdfData = Data(base64Encoded: pdfBase64) else {
            return
        }

        // Save PDF to temporary directory and share
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("diagnostic_report.pdf")

        do {
            try pdfData.write(to: tempURL)
            // TODO: Present share sheet with PDF
        } catch {
            self.error = error
            showError = true
        }
    }

    private func confidenceColor(_ confidence: Double) -> Color {
        if confidence >= 0.8 {
            return .statusSuccess
        } else if confidence >= 0.6 {
            return .statusWarning
        } else {
            return .statusError
        }
    }
}

// MARK: - Finding Card

struct FindingCard: View {
    let finding: ECGAPIClient.DiagnosticReportResponse.DiagnosticFinding

    var body: some View {
        HStack(alignment: .top, spacing: AppSpacing.md) {
            // Severity indicator
            Circle()
                .fill(severityColor)
                .frame(width: 12, height: 12)
                .padding(.top, 4)

            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                // Category
                Text(finding.category)
                    .font(AppTypography.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.textPrimary)

                // Description
                Text(finding.description)
                    .font(AppTypography.body)
                    .foregroundColor(.textSecondary)

                // Confidence
                HStack(spacing: AppSpacing.xs) {
                    Text("Confidence:")
                        .font(AppTypography.caption)
                        .foregroundColor(.textMuted)
                    Text(String(format: "%.0f%%", finding.confidence * 100))
                        .font(AppTypography.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.textSecondary)
                }
            }

            Spacer()
        }
        .padding(AppSpacing.md)
        .background(severityColor.opacity(0.1))
        .cornerRadius(AppRadius.md)
    }

    private var severityColor: Color {
        switch finding.severity.lowercased() {
        case "normal":
            return .statusSuccess
        case "abnormal":
            return .statusWarning
        case "critical":
            return .statusError
        default:
            return .textMuted
        }
    }
}

// MARK: - Measurement Row

struct MeasurementRow: View {
    let label: String
    let value: String
    let icon: String

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.brandPrimary)
                .frame(width: 20)

            Text(label)
                .font(AppTypography.subheadline)
                .foregroundColor(.textSecondary)

            Spacer()

            Text(value)
                .font(AppTypography.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.textPrimary)
        }
    }
}

// MARK: - Preview

#Preview {
    let sampleRecording = ECGRecording(
        originalImageData: nil,
        parameters: .standard,
        layout: .threeByFour_r1,
        leads: LeadType.baseLeads.map { type in
            ECGLead(
                type: type,
                samples: (0..<500).map { _ in Double.random(in: -0.5...0.5) },
                samplingRate: 500
            )
        },
        gridCalibration: nil,
        validationStatus: .valid
    )

    return DiagnosticUploadView(recording: sampleRecording)
}
