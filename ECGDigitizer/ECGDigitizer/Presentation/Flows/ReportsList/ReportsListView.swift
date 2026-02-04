import SwiftUI

/// View for listing all saved ECG reports
struct ReportsListView: View {
    @StateObject private var viewModel = ReportsListViewModel()
    @State private var searchText = ""
    @State private var selectedRecording: ECGRecording?
    @State private var showingDeleteConfirmation = false
    @State private var recordingToDelete: ECGRecording?

    var body: some View {
        NavigationStack {
            ZStack {
                Color.backgroundPrimary
                    .ignoresSafeArea()

                Group {
                    if viewModel.recordings.isEmpty {
                        EmptyReportsView()
                    } else {
                        reportsList
                    }
                }
            }
            .navigationTitle("Reports")
            .searchable(text: $searchText, prompt: "Search reports")
            .refreshable {
                await viewModel.loadRecordings()
            }
            .task {
                await viewModel.loadRecordings()
            }
            .sheet(item: $selectedRecording) { recording in
                NavigationStack {
                    ResultsView(
                        recording: recording,
                        onSave: {},
                        onNewCapture: {
                            selectedRecording = nil
                        }
                    )
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") {
                                selectedRecording = nil
                            }
                            .foregroundColor(.brandPrimary)
                        }
                    }
                }
            }
            .alert("Delete Report", isPresented: $showingDeleteConfirmation) {
                Button("Cancel", role: .cancel) {}
                Button("Delete", role: .destructive) {
                    if let recording = recordingToDelete {
                        Task {
                            await viewModel.deleteRecording(recording)
                        }
                    }
                }
            } message: {
                Text("Are you sure you want to delete this report? This action cannot be undone.")
            }
        }
        .tint(.brandPrimary)
    }

    private var reportsList: some View {
        ScrollView {
            LazyVStack(spacing: AppSpacing.sm) {
                ForEach(filteredRecordings) { recording in
                    ReportRow(recording: recording)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            selectedRecording = recording
                        }
                        .contextMenu {
                            Button {
                                shareRecording(recording)
                            } label: {
                                Label("Share", systemImage: "square.and.arrow.up")
                            }

                            Button(role: .destructive) {
                                recordingToDelete = recording
                                showingDeleteConfirmation = true
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                }
            }
            .padding(AppSpacing.md)
        }
    }

    private var filteredRecordings: [ECGRecording] {
        if searchText.isEmpty {
            return viewModel.recordings
        } else {
            return viewModel.recordings.filter { recording in
                recording.reportName.localizedCaseInsensitiveContains(searchText) ||
                recording.layout.displayName.localizedCaseInsensitiveContains(searchText)
            }
        }
    }

    private func shareRecording(_ recording: ECGRecording) {
        // Share functionality
    }
}

// MARK: - Reports List View Model

@MainActor
class ReportsListViewModel: ObservableObject {
    @Published var recordings: [ECGRecording] = []
    @Published var isLoading = false
    @Published var error: Error?

    func loadRecordings() async {
        isLoading = true

        // In production, this would load from repository
        // For now, create sample data
        recordings = createSampleRecordings()

        isLoading = false
    }

    func deleteRecording(_ recording: ECGRecording) async {
        recordings.removeAll { $0.id == recording.id }
        // In production, also delete from repository
    }

    private func createSampleRecordings() -> [ECGRecording] {
        let layouts: [ECGLayout] = [.threeByFour_r1, .sixByTwo_r0, .twelveByOne_r1]

        return (0..<5).map { i in
            ECGRecording(
                timestamp: Date().addingTimeInterval(-Double(i) * 86400),
                originalImageData: nil,
                parameters: .standard,
                layout: layouts[i % layouts.count],
                leads: LeadType.baseLeads.map { type in
                    ECGLead(
                        type: type,
                        samples: generateSampleECG(),
                        samplingRate: 500
                    )
                },
                gridCalibration: nil,
                validationStatus: i == 2 ? .warning : .valid
            )
        }
    }

    private func generateSampleECG() -> [Double] {
        // Generate realistic-looking ECG samples
        var samples: [Double] = []
        let cycleLength = 200  // ~60 BPM at 500Hz

        for i in 0..<750 {  // 1.5 seconds
            let phase = Double(i % cycleLength) / Double(cycleLength)

            var value: Double = 0

            // P wave
            if phase >= 0.1 && phase < 0.2 {
                let pPhase = (phase - 0.1) / 0.1
                value = 0.15 * sin(pPhase * .pi)
            }
            // QRS complex
            else if phase >= 0.25 && phase < 0.35 {
                let qrsPhase = (phase - 0.25) / 0.1
                if qrsPhase < 0.2 {
                    value = -0.1 * (qrsPhase / 0.2)  // Q
                } else if qrsPhase < 0.5 {
                    value = -0.1 + 1.2 * ((qrsPhase - 0.2) / 0.3)  // R up
                } else if qrsPhase < 0.7 {
                    value = 1.1 - 1.4 * ((qrsPhase - 0.5) / 0.2)  // R down to S
                } else {
                    value = -0.3 + 0.3 * ((qrsPhase - 0.7) / 0.3)  // S recovery
                }
            }
            // T wave
            else if phase >= 0.45 && phase < 0.65 {
                let tPhase = (phase - 0.45) / 0.2
                value = 0.3 * sin(tPhase * .pi)
            }

            // Add small noise
            value += Double.random(in: -0.02...0.02)

            samples.append(value)
        }

        return samples
    }
}

// MARK: - Report Row

struct ReportRow: View {
    let recording: ECGRecording

    var body: some View {
        HStack(spacing: AppSpacing.md) {
            // Thumbnail preview
            ECGThumbnailView(recording: recording)
                .frame(width: 70, height: 50)
                .cornerRadius(AppRadius.sm)

            // Info
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                HStack {
                    Text(recording.reportName)
                        .font(AppTypography.headline)
                        .foregroundColor(.textPrimary)

                    Spacer()

                    StatusBadge(
                        text: recording.validationStatus.displayName,
                        status: statusType(for: recording.validationStatus)
                    )
                }

                HStack {
                    Text(recording.layout.displayName)
                        .font(AppTypography.caption)
                        .foregroundColor(.textSecondary)

                    Spacer()

                    Text(recording.formattedDate)
                        .font(AppTypography.caption)
                        .foregroundColor(.textMuted)
                }

                // Lead count and heart rate
                HStack(spacing: AppSpacing.md) {
                    HStack(spacing: AppSpacing.xs) {
                        Image(systemName: "waveform.path.ecg")
                            .font(.system(size: 10))
                        Text("\(recording.leads.count) leads")
                    }
                    .font(AppTypography.caption2)
                    .foregroundColor(.textMuted)

                    if let hr = recording.estimatedHeartRate {
                        HStack(spacing: AppSpacing.xs) {
                            Image(systemName: "heart.fill")
                                .font(.system(size: 10))
                            Text("\(hr) BPM")
                        }
                        .font(AppTypography.caption2)
                        .foregroundColor(.statusError)
                    }
                }
            }

            // Chevron
            Image(systemName: "chevron.right")
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.textMuted)
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.lg)
        .primaryShadow()
    }

    private func statusType(for status: ValidationStatus) -> StatusBadge.Status {
        switch status {
        case .valid: return .success
        case .warning: return .warning
        case .invalid: return .error
        }
    }
}

// MARK: - ECG Thumbnail View

struct ECGThumbnailView: View {
    let recording: ECGRecording

    var body: some View {
        Canvas { context, size in
            // Draw mini waveform
            if let lead = recording.leads.first(where: { $0.type == .II }) {
                drawMiniWaveform(context: context, size: size, samples: lead.samples)
            }
        }
        .background(Color.backgroundTertiary)
    }

    private func drawMiniWaveform(
        context: GraphicsContext,
        size: CGSize,
        samples: [Double]
    ) {
        guard !samples.isEmpty else { return }

        var path = Path()

        let xScale = size.width / CGFloat(samples.count)
        let yScale = size.height / 4.0
        let yCenter = size.height / 2

        for (index, voltage) in samples.enumerated() {
            let x = CGFloat(index) * xScale
            let y = yCenter - CGFloat(voltage) * yScale

            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }

        context.stroke(path, with: .color(.brandPrimary), lineWidth: 1.5)
    }
}

// MARK: - Empty Reports View

struct EmptyReportsView: View {
    var body: some View {
        EmptyStateView(
            icon: "doc.text.magnifyingglass",
            title: "No Reports Yet",
            message: "Captured ECG reports will appear here.\nUse the Capture tab to digitize an ECG."
        )
    }
}

#Preview {
    ReportsListView()
}
