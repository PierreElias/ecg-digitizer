import SwiftUI

/// View showing processing progress
struct ProcessingView: View {
    let state: ProcessingState

    var body: some View {
        ZStack {
            Color.backgroundPrimary
                .ignoresSafeArea()

            VStack(spacing: AppSpacing.xl) {
                Spacer()

                // Progress indicator
                if state.isProcessing {
                    ProgressIndicator(progress: state.progress)
                } else if case .failed(let error) = state {
                    ErrorIndicator(error: error)
                } else if case .complete = state {
                    SuccessIndicator()
                }

                // Status message
                Text(state.displayMessage)
                    .font(AppTypography.title3)
                    .foregroundColor(.textPrimary)
                    .multilineTextAlignment(.center)

                // Progress bar
                if let progress = state.progress, state.isProcessing {
                    VStack(spacing: AppSpacing.sm) {
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: AppRadius.full)
                                    .fill(Color.borderLight)
                                    .frame(height: 8)

                                RoundedRectangle(cornerRadius: AppRadius.full)
                                    .fill(Color.brandPrimary)
                                    .frame(width: geometry.size.width * progress, height: 8)
                                    .animation(.easeInOut(duration: 0.3), value: progress)
                            }
                        }
                        .frame(height: 8)

                        Text("\(Int(progress * 100))%")
                            .font(AppTypography.caption)
                            .foregroundColor(.textMuted)
                    }
                    .padding(.horizontal, AppSpacing.xxl)
                }

                // Processing steps
                ProcessingStepsList(currentState: state)
                    .padding(.horizontal, AppSpacing.lg)

                Spacer()
            }
            .padding(AppSpacing.lg)
        }
    }
}

// MARK: - Progress Indicator

struct ProgressIndicator: View {
    let progress: Double?

    @State private var isAnimating = false

    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .stroke(Color.brandPrimaryLight.opacity(0.3), lineWidth: 8)
                .frame(width: 100, height: 100)

            // Progress circle
            if let progress = progress {
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(Color.brandPrimary, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .frame(width: 100, height: 100)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.3), value: progress)
            } else {
                // Indeterminate spinner
                Circle()
                    .trim(from: 0, to: 0.7)
                    .stroke(Color.brandPrimary, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                    .frame(width: 100, height: 100)
                    .rotationEffect(.degrees(isAnimating ? 360 : 0))
                    .animation(.linear(duration: 1).repeatForever(autoreverses: false), value: isAnimating)
            }

            // Icon
            Image(systemName: "waveform.path.ecg")
                .font(.system(size: 30))
                .foregroundColor(.brandPrimary)
        }
        .onAppear {
            isAnimating = true
        }
    }
}

// MARK: - Error Indicator

struct ErrorIndicator: View {
    let error: ValidationError

    var body: some View {
        VStack(spacing: AppSpacing.md) {
            ZStack {
                Circle()
                    .fill(Color.statusError.opacity(0.1))
                    .frame(width: 100, height: 100)

                Image(systemName: "exclamationmark.circle.fill")
                    .font(.system(size: 50))
                    .foregroundColor(.statusError)
            }

            Text(error.title)
                .font(AppTypography.headline)
                .foregroundColor(.statusError)
        }
    }
}

// MARK: - Success Indicator

struct SuccessIndicator: View {
    @State private var showCheckmark = false

    var body: some View {
        ZStack {
            Circle()
                .fill(Color.statusSuccess)
                .frame(width: 100, height: 100)

            Image(systemName: "checkmark")
                .font(.system(size: 50, weight: .bold))
                .foregroundColor(.white)
                .scaleEffect(showCheckmark ? 1 : 0)
        }
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                showCheckmark = true
            }
        }
    }
}

// MARK: - Processing Steps List

struct ProcessingStepsList: View {
    let currentState: ProcessingState

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            ProcessingStepRow(
                title: "Validating image",
                state: stepState(for: .validatingImage)
            )

            ProcessingStepRow(
                title: "Detecting ECG grid",
                state: stepState(for: .detectingGrid(progress: 0))
            )

            ProcessingStepRow(
                title: "Classifying layout",
                state: stepState(for: .classifyingLayout)
            )

            ProcessingStepRow(
                title: "Extracting waveforms",
                state: stepState(for: .extractingWaveforms(currentLead: 0, totalLeads: 12))
            )

            ProcessingStepRow(
                title: "Validating results",
                state: stepState(for: .validatingResults)
            )
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.lg)
        .primaryShadow()
    }

    private func stepState(for step: ProcessingState) -> StepState {
        let currentIndex = stepIndex(currentState)
        let targetIndex = stepIndex(step)

        if case .failed = currentState {
            if currentIndex == targetIndex {
                return .failed
            } else if currentIndex > targetIndex {
                return .completed
            }
        }

        if case .complete = currentState {
            return .completed
        }

        if currentIndex > targetIndex {
            return .completed
        } else if currentIndex == targetIndex {
            return .inProgress
        } else {
            return .pending
        }
    }

    private func stepIndex(_ state: ProcessingState) -> Int {
        switch state {
        case .idle, .capturing:
            return -1
        case .validatingImage:
            return 0
        case .detectingGrid:
            return 1
        case .classifyingLayout:
            return 2
        case .extractingWaveforms:
            return 3
        case .validatingResults:
            return 4
        case .complete, .failed:
            return 5
        }
    }

    enum StepState {
        case pending
        case inProgress
        case completed
        case failed
    }
}

// MARK: - Processing Step Row

struct ProcessingStepRow: View {
    let title: String
    let state: ProcessingStepsList.StepState

    var body: some View {
        HStack(spacing: AppSpacing.md) {
            // Status icon
            Group {
                switch state {
                case .pending:
                    Circle()
                        .stroke(Color.borderMedium, lineWidth: 2)
                        .frame(width: 24, height: 24)

                case .inProgress:
                    BrandLoadingView()
                        .scaleEffect(0.6)
                        .frame(width: 24, height: 24)

                case .completed:
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 24))
                        .foregroundColor(.statusSuccess)

                case .failed:
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 24))
                        .foregroundColor(.statusError)
                }
            }
            .frame(width: 24, height: 24)

            // Title
            Text(title)
                .font(AppTypography.subheadline)
                .foregroundColor(state == .pending ? .textMuted : .textPrimary)

            Spacer()
        }
    }
}

#Preview {
    ProcessingView(state: .extractingWaveforms(currentLead: 5, totalLeads: 12))
}
