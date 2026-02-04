import SwiftUI

/// View displaying digitization results
struct ResultsView: View {
    let recording: ECGRecording
    let onSave: () -> Void
    let onNewCapture: () -> Void

    @State private var showingExportSheet = false
    @State private var showingShareSheet = false
    @State private var showingDiagnostic = false
    @State private var showingAPIUpload = false
    @State private var selectedTab: ResultsTab = .visualization
    @State private var exportURL: URL?

    enum ResultsTab: String, CaseIterable {
        case visualization = "Visualization"
        case original = "Original"
        case details = "Details"
    }

    var body: some View {
        ZStack {
            Color.backgroundPrimary
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Header with status
                ResultsHeader(recording: recording)

                // Tab selector
                Picker("View", selection: $selectedTab) {
                    ForEach(ResultsTab.allCases, id: \.self) { tab in
                        Text(tab.rawValue).tag(tab)
                    }
                }
                .pickerStyle(.segmented)
                .padding(AppSpacing.md)

                // Content
                TabView(selection: $selectedTab) {
                    // Visualization tab
                    ECGVisualizationTab(recording: recording)
                        .tag(ResultsTab.visualization)

                    // Original image tab
                    OriginalImageTab(recording: recording)
                        .tag(ResultsTab.original)

                    // Details tab
                    DetailsTab(recording: recording)
                        .tag(ResultsTab.details)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))

                // Action buttons
                ResultsActionBar(
                    onSave: onSave,
                    onExport: { showingExportSheet = true },
                    onDiagnostic: { showingDiagnostic = true },
                    onAPIUpload: { showingAPIUpload = true },
                    onNewCapture: onNewCapture
                )
            }
        }
        .sheet(isPresented: $showingExportSheet) {
            ExportOptionsSheet(recording: recording) { url in
                exportURL = url
                showingShareSheet = true
            }
        }
        .sheet(isPresented: $showingShareSheet) {
            if let url = exportURL {
                ShareSheet(items: [url])
            }
        }
        .fullScreenCover(isPresented: $showingDiagnostic) {
            if let image = recording.originalImage {
                DiagnosticView(image: image)
            }
        }
        .sheet(isPresented: $showingAPIUpload) {
            DiagnosticUploadView(recording: recording)
        }
    }
}

// MARK: - Results Header

struct ResultsHeader: View {
    let recording: ECGRecording

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                Text(recording.reportName)
                    .font(AppTypography.headline)
                    .foregroundColor(.textPrimary)

                HStack(spacing: AppSpacing.sm) {
                    Text(recording.formattedDate)
                        .font(AppTypography.caption)
                        .foregroundColor(.textMuted)

                    // Algorithm indicator badge
                    if let algorithm = recording.metadata.extractionAlgorithm {
                        HStack(spacing: 2) {
                            Image(systemName: algorithm.isPrimary ? "cpu" : "arrow.triangle.2.circlepath")
                                .font(.system(size: 9))
                            Text(algorithm.shortName)
                                .font(.system(size: 10, weight: .medium))
                        }
                        .foregroundColor(algorithm.isPrimary ? .brandPrimary : .statusWarning)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            (algorithm.isPrimary ? Color.brandPrimary : Color.statusWarning)
                                .opacity(0.15)
                        )
                        .cornerRadius(AppRadius.full)
                    }
                }
            }

            Spacer()

            // Validation status badge
            StatusBadge(
                text: recording.validationStatus.displayName,
                status: statusType(for: recording.validationStatus)
            )

            // Heart rate if available
            if let hr = recording.estimatedHeartRate {
                HStack(spacing: AppSpacing.xs) {
                    Image(systemName: "heart.fill")
                        .font(.system(size: 12))
                    Text("\(hr) BPM")
                        .font(AppTypography.caption)
                }
                .foregroundColor(.statusError)
                .padding(.horizontal, AppSpacing.sm)
                .padding(.vertical, AppSpacing.xs)
                .background(Color.statusError.opacity(0.1))
                .cornerRadius(AppRadius.full)
            }
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
    }

    private func statusType(for status: ValidationStatus) -> StatusBadge.Status {
        switch status {
        case .valid: return .success
        case .warning: return .warning
        case .invalid: return .error
        }
    }
}

// MARK: - ECG Visualization Tab

struct ECGVisualizationTab: View {
    let recording: ECGRecording

    var body: some View {
        ScrollView {
            VStack(spacing: AppSpacing.md) {
                // Main ECG view
                ECGWaveformView(
                    leads: recording.leads,
                    layout: recording.layout,
                    parameters: recording.parameters
                )
                .frame(height: 400)
                .cornerRadius(AppRadius.lg)
                .primaryShadow()

                // Calibration info
                HStack {
                    CalibrationMarkView(parameters: recording.parameters)

                    Spacer()

                    VStack(alignment: .trailing, spacing: AppSpacing.xs) {
                        Text(recording.parameters.paperSpeed.displayName)
                            .font(AppTypography.caption)
                            .foregroundColor(.textSecondary)
                        Text(recording.parameters.voltageGain.displayName)
                            .font(AppTypography.caption)
                            .foregroundColor(.textSecondary)
                    }
                }
                .padding(.horizontal, AppSpacing.md)

                // Individual lead views
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible())
                ], spacing: AppSpacing.md) {
                    ForEach(recording.leads.filter { !$0.isRhythmLead }) { lead in
                        SingleLeadWaveformView(lead: lead, showGrid: true)
                            .frame(height: 80)
                            .cornerRadius(AppRadius.sm)
                            .overlay(
                                RoundedRectangle(cornerRadius: AppRadius.sm)
                                    .stroke(Color.borderLight, lineWidth: 1)
                            )
                    }
                }
                .padding(.horizontal, AppSpacing.md)
            }
            .padding(.vertical, AppSpacing.md)
        }
    }
}

// MARK: - Original Image Tab

struct OriginalImageTab: View {
    let recording: ECGRecording

    var body: some View {
        ScrollView {
            VStack(spacing: AppSpacing.md) {
                if let image = recording.originalImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(AppRadius.lg)
                        .primaryShadow()
                } else {
                    EmptyStateView(
                        icon: "photo",
                        title: "No Image",
                        message: "Original image not available"
                    )
                }

                // Layout info
                VStack(alignment: .leading, spacing: AppSpacing.sm) {
                    SectionHeader("Detected Layout")

                    HStack {
                        HStack(spacing: AppSpacing.sm) {
                            Image(systemName: "square.grid.3x3")
                                .foregroundColor(.brandPrimary)
                            Text(recording.layout.displayName)
                                .font(AppTypography.body)
                                .foregroundColor(.textPrimary)
                        }
                        Spacer()
                        Text(recording.layout.rawValue)
                            .font(AppTypography.caption)
                            .foregroundColor(.textMuted)
                    }
                }
                .padding(AppSpacing.md)
                .background(Color.backgroundSecondary)
                .cornerRadius(AppRadius.lg)
                .primaryShadow()
            }
            .padding(AppSpacing.md)
        }
    }
}

// MARK: - Details Tab

struct DetailsTab: View {
    let recording: ECGRecording

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                // Recording info
                DetailSection(title: "Recording") {
                    DetailRow(label: "ID", value: recording.shortId)
                    DetailRow(label: "Date", value: recording.formattedDate)
                    DetailRow(label: "Layout", value: recording.layout.displayName)
                    DetailRow(label: "Total Leads", value: "\(recording.leads.count)")
                    DetailRow(label: "Duration", value: String(format: "%.1f s", recording.totalDuration))
                }

                // Parameters
                DetailSection(title: "Parameters") {
                    DetailRow(label: "Paper Speed", value: recording.parameters.paperSpeed.displayName)
                    DetailRow(label: "Voltage Gain", value: recording.parameters.voltageGain.displayName)
                }

                // Processing info with algorithm used
                DetailSection(title: "Processing") {
                    if let algorithm = recording.metadata.extractionAlgorithm {
                        HStack {
                            Text("Extraction Algorithm")
                                .font(AppTypography.subheadline)
                                .foregroundColor(.textSecondary)
                            Spacer()
                            HStack(spacing: AppSpacing.xs) {
                                Image(systemName: algorithm.isPrimary ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                                    .foregroundColor(algorithm.isPrimary ? .statusSuccess : .statusWarning)
                                    .font(.system(size: 12))
                                Text(algorithm.shortName)
                                    .font(AppTypography.subheadline)
                                    .fontWeight(.medium)
                                    .foregroundColor(.textPrimary)
                            }
                        }
                        Text(algorithm.displayName)
                            .font(AppTypography.caption)
                            .foregroundColor(.textMuted)
                    } else {
                        DetailRow(label: "Extraction Algorithm", value: "Unknown")
                    }
                    if let appVersion = recording.metadata.appVersion {
                        DetailRow(label: "App Version", value: appVersion)
                    }
                }

                // Grid calibration
                if let calibration = recording.gridCalibration {
                    DetailSection(title: "Grid Calibration") {
                        DetailRow(label: "Horizontal Spacing", value: String(format: "%.1f px/mm", calibration.smallSquareWidthPixels))
                        DetailRow(label: "Vertical Spacing", value: String(format: "%.1f px/mm", calibration.smallSquareHeightPixels))
                        DetailRow(label: "Grid Angle", value: String(format: "%.2f\u{00B0}", calibration.angleInDegrees))
                        DetailRow(label: "Quality Score", value: String(format: "%.0f%%", calibration.qualityScore * 100))
                    }
                }

                // Lead summary
                DetailSection(title: "Lead Summary") {
                    ForEach(recording.leads.filter { !$0.isRhythmLead }) { lead in
                        HStack {
                            Text(lead.type.displayName)
                                .font(AppTypography.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.textPrimary)
                            Spacer()
                            Text(String(format: "%.0f ms", lead.durationMs))
                                .font(AppTypography.caption)
                                .foregroundColor(.textSecondary)
                            Text(String(format: "%.2f mV", lead.amplitude))
                                .font(AppTypography.caption)
                                .foregroundColor(.textMuted)
                        }
                    }
                }

                // Validation
                if !recording.leads.validationIssues.isEmpty {
                    DetailSection(title: "Validation Issues") {
                        ForEach(Array(recording.leads.validationIssues.keys), id: \.self) { leadName in
                            if let issues = recording.leads.validationIssues[leadName] {
                                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                                    Text(leadName)
                                        .font(AppTypography.subheadline)
                                        .fontWeight(.medium)
                                        .foregroundColor(.textPrimary)
                                    ForEach(issues, id: \.self) { issue in
                                        HStack(alignment: .top, spacing: AppSpacing.xs) {
                                            Circle()
                                                .fill(Color.statusWarning)
                                                .frame(width: 6, height: 6)
                                                .padding(.top, 6)
                                            Text(issue)
                                                .font(AppTypography.caption)
                                                .foregroundColor(.statusWarning)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            .padding(AppSpacing.md)
        }
    }
}

// MARK: - Detail Section

struct DetailSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            Text(title)
                .font(AppTypography.headline)
                .foregroundColor(.textPrimary)

            VStack(spacing: AppSpacing.sm) {
                content()
            }
            .padding(AppSpacing.md)
            .background(Color.backgroundSecondary)
            .cornerRadius(AppRadius.lg)
            .primaryShadow()
        }
    }
}

// MARK: - Detail Row

struct DetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
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

// MARK: - Results Action Bar

struct ResultsActionBar: View {
    let onSave: () -> Void
    let onExport: () -> Void
    let onDiagnostic: () -> Void
    let onAPIUpload: () -> Void
    let onNewCapture: () -> Void

    var body: some View {
        HStack(spacing: AppSpacing.sm) {
            // Save button
            Button {
                onSave()
            } label: {
                VStack(spacing: AppSpacing.xs) {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 20))
                    Text("Save")
                        .font(AppTypography.caption2)
                }
            }
            .foregroundColor(.brandPrimary)
            .frame(maxWidth: .infinity)

            // Export & Share button (combined)
            Button {
                onExport()
            } label: {
                VStack(spacing: AppSpacing.xs) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 20))
                    Text("Export")
                        .font(AppTypography.caption2)
                }
            }
            .foregroundColor(.brandPrimary)
            .frame(maxWidth: .infinity)

            // Diagnostic button (runs full diagnostic report)
            Button {
                onDiagnostic()
            } label: {
                VStack(spacing: AppSpacing.xs) {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.system(size: 20))
                    Text("Diagnostic")
                        .font(AppTypography.caption2)
                }
            }
            .foregroundColor(.brandPrimary)
            .frame(maxWidth: .infinity)

            // API Upload button (Aptible)
            Button {
                onAPIUpload()
            } label: {
                VStack(spacing: AppSpacing.xs) {
                    Image(systemName: "cloud.fill")
                        .font(.system(size: 20))
                    Text("API")
                        .font(AppTypography.caption2)
                }
            }
            .foregroundColor(.brandPrimary)
            .frame(maxWidth: .infinity)

            // New Capture button
            Button {
                onNewCapture()
            } label: {
                VStack(spacing: AppSpacing.xs) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 20))
                    Text("New")
                        .font(AppTypography.caption2)
                }
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, AppSpacing.sm)
            .background(Color.brandPrimary)
            .cornerRadius(AppRadius.md)
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
    }
}

// MARK: - Export Options Sheet

struct ExportOptionsSheet: View {
    let recording: ECGRecording
    let onExport: (URL) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var selectedFormat: ExportFormat = .csv
    @State private var isExporting = false
    @State private var exportError: Error?
    @State private var showingError = false

    // Exporters
    private let csvExporter = CSVExporter()
    private let pdfExporter = PDFExporter()
    private let hl7Exporter = HL7Exporter()

    enum ExportFormat: String, CaseIterable {
        case csv = "CSV"
        case csvWide = "CSV (Wide)"
        case pdf = "PDF"
        case hl7 = "HL7"
        case hl7XML = "HL7 XML"

        var icon: String {
            switch self {
            case .csv, .csvWide: return "tablecells"
            case .pdf: return "doc.richtext"
            case .hl7, .hl7XML: return "cross.case"
            }
        }

        var description: String {
            switch self {
            case .csv: return "Long format - one row per sample"
            case .csvWide: return "Wide format - one column per lead"
            case .pdf: return "Printable report with visualization"
            case .hl7: return "HL7 v2.x message format"
            case .hl7XML: return "HL7 aECG XML format"
            }
        }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.backgroundPrimary
                    .ignoresSafeArea()

                VStack(spacing: AppSpacing.md) {
                    ScrollView {
                        VStack(spacing: AppSpacing.sm) {
                            ForEach(ExportFormat.allCases, id: \.self) { format in
                                Button {
                                    selectedFormat = format
                                } label: {
                                    HStack {
                                        Image(systemName: format.icon)
                                            .font(.system(size: 24))
                                            .foregroundColor(.brandPrimary)
                                            .frame(width: 40)

                                        VStack(alignment: .leading, spacing: AppSpacing.xs) {
                                            Text(format.rawValue)
                                                .font(AppTypography.headline)
                                                .foregroundColor(.textPrimary)
                                            Text(format.description)
                                                .font(AppTypography.caption)
                                                .foregroundColor(.textSecondary)
                                        }

                                        Spacer()

                                        if selectedFormat == format {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundColor(.brandPrimary)
                                        }
                                    }
                                    .padding(AppSpacing.md)
                                    .background(selectedFormat == format ? Color.brandPrimary.opacity(0.1) : Color.backgroundSecondary)
                                    .cornerRadius(AppRadius.md)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(AppSpacing.md)
                    }

                    Button {
                        exportRecording()
                    } label: {
                        if isExporting {
                            HStack(spacing: AppSpacing.sm) {
                                BrandLoadingView()
                                    .scaleEffect(0.5)
                                Text("Exporting...")
                            }
                        } else {
                            Text("Export as \(selectedFormat.rawValue)")
                        }
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(isExporting)
                    .padding(.horizontal, AppSpacing.md)
                    .padding(.bottom, AppSpacing.md)
                }
            }
            .navigationTitle("Export")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .foregroundColor(.brandPrimary)
                }
            }
            .alert("Export Error", isPresented: $showingError) {
                Button("OK", role: .cancel) {}
            } message: {
                if let error = exportError {
                    Text(error.localizedDescription)
                }
            }
        }
    }

    private func exportRecording() {
        isExporting = true

        Task {
            do {
                let url: URL

                switch selectedFormat {
                case .csv:
                    url = try csvExporter.exportToFile(recording)
                case .csvWide:
                    url = try csvExporter.exportWideFormatToFile(recording)
                case .pdf:
                    url = try pdfExporter.exportToFile(recording)
                case .hl7:
                    url = try hl7Exporter.exportToFile(recording)
                case .hl7XML:
                    url = try hl7Exporter.exportXMLToFile(recording)
                }

                await MainActor.run {
                    isExporting = false
                    onExport(url)
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    isExporting = false
                    exportError = error
                    showingError = true
                }
            }
        }
    }
}

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

    return ResultsView(
        recording: sampleRecording,
        onSave: {},
        onNewCapture: {}
    )
}
