import SwiftUI

/// View for inputting ECG processing parameters
struct ParameterInputView: View {
    let image: UIImage?
    let onSubmit: (ProcessingParameters) -> Void
    let onRetake: () -> Void

    @State private var paperSpeed: ProcessingParameters.PaperSpeed = .twentyFive
    @State private var voltageGain: ProcessingParameters.VoltageGain = .ten
    @State private var showAdvancedSettings = false

    var body: some View {
        ZStack {
            Color.backgroundPrimary
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: AppSpacing.lg) {
                    // Image preview
                    if let image = image {
                        ImagePreviewSection(image: image, onRetake: onRetake)
                    }

                    // Paper Speed Selection
                    ParameterCard(
                        title: "Paper Speed",
                        icon: "arrow.left.and.right",
                        description: "Standard ECG paper speed is 25 mm/s"
                    ) {
                        Picker("Paper Speed", selection: $paperSpeed) {
                            ForEach(ProcessingParameters.PaperSpeed.allCases) { speed in
                                Text(speed.displayName).tag(speed)
                            }
                        }
                        .pickerStyle(.segmented)
                        .tint(.blue)
                    }

                    // Voltage Gain Selection
                    ParameterCard(
                        title: "Voltage Gain",
                        icon: "arrow.up.and.down",
                        description: "Standard ECG calibration is 10 mm/mV"
                    ) {
                        Picker("Voltage Gain", selection: $voltageGain) {
                            ForEach(ProcessingParameters.VoltageGain.allCases) { gain in
                                Text(gain.displayName).tag(gain)
                            }
                        }
                        .pickerStyle(.segmented)
                        .tint(.blue)
                    }

                    // Calibration Info
                    CalibrationInfoCard(paperSpeed: paperSpeed, voltageGain: voltageGain)

                    Spacer(minLength: AppSpacing.lg)
                }
                .padding(AppSpacing.md)
            }

            // Submit Button - Fixed at bottom
            VStack {
                Spacer()

                Button {
                    let parameters = ProcessingParameters(
                        paperSpeed: paperSpeed,
                        voltageGain: voltageGain
                    )
                    onSubmit(parameters)
                } label: {
                    HStack(spacing: AppSpacing.sm) {
                        Image(systemName: "waveform.path.ecg")
                        Text("Digitize ECG")
                    }
                }
                .buttonStyle(PrimaryButtonStyle())
                .padding(.horizontal, AppSpacing.lg)
                .padding(.vertical, AppSpacing.md)
                .background(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color.backgroundPrimary.opacity(0),
                            Color.backgroundPrimary
                        ]),
                        startPoint: .top,
                        endPoint: .center
                    )
                )
            }
        }
    }
}

// MARK: - Image Preview Section

struct ImagePreviewSection: View {
    let image: UIImage
    let onRetake: () -> Void

    var body: some View {
        VStack(spacing: AppSpacing.md) {
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 200)
                .cornerRadius(AppRadius.md)
                .primaryShadow()

            Button {
                onRetake()
            } label: {
                HStack(spacing: AppSpacing.xs) {
                    Image(systemName: "arrow.counterclockwise")
                    Text("Retake Photo")
                }
            }
            .buttonStyle(OutlineButtonStyle())
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.lg)
        .primaryShadow()
    }
}

// MARK: - Parameter Card

struct ParameterCard<Content: View>: View {
    let title: String
    let icon: String
    let description: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            HStack(spacing: AppSpacing.sm) {
                Image(systemName: icon)
                    .font(.system(size: 18))
                    .foregroundColor(.brandPrimary)
                    .frame(width: 24)

                Text(title)
                    .font(AppTypography.headline)
                    .foregroundColor(.textPrimary)
            }

            content()

            Text(description)
                .font(AppTypography.caption)
                .foregroundColor(.textMuted)
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.lg)
        .primaryShadow()
    }
}

// MARK: - Parameter Section (Legacy compatibility)

struct ParameterSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            Text(title)
                .font(AppTypography.headline)
                .foregroundColor(.textPrimary)

            content()
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.lg)
    }
}

// MARK: - Calibration Info Card

struct CalibrationInfoCard: View {
    let paperSpeed: ProcessingParameters.PaperSpeed
    let voltageGain: ProcessingParameters.VoltageGain

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            HStack(spacing: AppSpacing.sm) {
                Image(systemName: "ruler")
                    .font(.system(size: 18))
                    .foregroundColor(.brandPrimary)
                    .frame(width: 24)

                Text("Calibration Reference")
                    .font(AppTypography.headline)
                    .foregroundColor(.textPrimary)
            }

            HStack(spacing: AppSpacing.md) {
                // Time calibration
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    HStack(spacing: AppSpacing.xs) {
                        Image(systemName: "arrow.left.and.right")
                            .foregroundColor(.brandPrimary)
                            .font(.system(size: 14))
                        Text("Time")
                            .font(AppTypography.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(.textPrimary)
                    }

                    Text("1 large = \(String(format: "%.0f", paperSpeed.msPerLargeSquare)) ms")
                        .font(AppTypography.caption)
                        .foregroundColor(.textSecondary)

                    Text("1 small = \(String(format: "%.0f", paperSpeed.msPerMm)) ms")
                        .font(AppTypography.caption)
                        .foregroundColor(.textMuted)
                }

                Divider()

                // Voltage calibration
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    HStack(spacing: AppSpacing.xs) {
                        Image(systemName: "arrow.up.and.down")
                            .foregroundColor(.statusError)
                            .font(.system(size: 14))
                        Text("Voltage")
                            .font(AppTypography.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(.textPrimary)
                    }

                    Text("1 large = \(String(format: "%.1f", voltageGain.mvPerLargeSquare)) mV")
                        .font(AppTypography.caption)
                        .foregroundColor(.textSecondary)

                    Text("1 small = \(String(format: "%.2f", voltageGain.mvPerMm)) mV")
                        .font(AppTypography.caption)
                        .foregroundColor(.textMuted)
                }

                Divider()

                // Grid diagram - now on the same line
                CalibrationGridDiagram()
                    .frame(width: 80, height: 80)
            }
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.lg)
        .primaryShadow()
    }
}

// MARK: - Calibration Info Section (Legacy compatibility)

struct CalibrationInfoSection: View {
    let paperSpeed: ProcessingParameters.PaperSpeed
    let voltageGain: ProcessingParameters.VoltageGain

    var body: some View {
        CalibrationInfoCard(paperSpeed: paperSpeed, voltageGain: voltageGain)
    }
}

// MARK: - Calibration Grid Diagram

struct CalibrationGridDiagram: View {
    var body: some View {
        GeometryReader { geometry in
            let gridSize = min(geometry.size.width, geometry.size.height)
            let smallSquare = gridSize / 10

            ZStack {
                // Grid background - ECG pink
                Rectangle()
                    .fill(Color(hex: "FCE4EC"))

                // Small squares
                ForEach(0..<10, id: \.self) { row in
                    ForEach(0..<10, id: \.self) { col in
                        Rectangle()
                            .stroke(Color(hex: "F8BBD9").opacity(0.6), lineWidth: 0.5)
                            .frame(width: smallSquare, height: smallSquare)
                            .position(
                                x: CGFloat(col) * smallSquare + smallSquare / 2,
                                y: CGFloat(row) * smallSquare + smallSquare / 2
                            )
                    }
                }

                // Large squares
                ForEach(0..<2, id: \.self) { row in
                    ForEach(0..<2, id: \.self) { col in
                        Rectangle()
                            .stroke(Color(hex: "E91E63").opacity(0.4), lineWidth: 1)
                            .frame(width: smallSquare * 5, height: smallSquare * 5)
                            .position(
                                x: CGFloat(col) * smallSquare * 5 + smallSquare * 2.5,
                                y: CGFloat(row) * smallSquare * 5 + smallSquare * 2.5
                            )
                    }
                }

                // Labels
                Text("5mm")
                    .font(.system(size: 10))
                    .foregroundColor(.textSecondary)
                    .position(x: smallSquare * 2.5, y: gridSize + 10)

                Text("1mm")
                    .font(.system(size: 8))
                    .foregroundColor(.textMuted)
                    .position(x: smallSquare * 0.5, y: gridSize + 10)
            }
            .frame(width: gridSize, height: gridSize)
            .cornerRadius(AppRadius.sm)
        }
    }
}

#Preview {
    ParameterInputView(
        image: nil,
        onSubmit: { _ in },
        onRetake: {}
    )
}
