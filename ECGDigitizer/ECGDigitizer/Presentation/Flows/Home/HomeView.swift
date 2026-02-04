import SwiftUI

/// Enhanced home view showing example ECG and call to action
struct HomeView: View {
    @Binding var selectedTab: ContentView.Tab

    var body: some View {
        NavigationStack {
            ZStack {
                Color.backgroundPrimary
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: AppSpacing.xl) {
                        // Hero section
                        VStack(spacing: AppSpacing.md) {
                            // App icon/logo
                            ZStack {
                                Circle()
                                    .fill(
                                        LinearGradient(
                                            colors: [Color.brandPrimary.opacity(0.1), Color.brandPrimary.opacity(0.05)],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                                    .frame(width: 100, height: 100)

                                Image(systemName: "waveform.path.ecg.rectangle.fill")
                                    .font(.system(size: 50))
                                    .foregroundStyle(
                                        LinearGradient(
                                            colors: [Color.brandPrimary, Color.brandAccent],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                            }
                            .padding(.top, AppSpacing.xl)

                            VStack(spacing: AppSpacing.xs) {
                                Text("EchoNext")
                                    .font(AppTypography.largeTitle)
                                    .foregroundColor(.textPrimary)

                                Text("Transform Paper ECGs into Digital Signals")
                                    .font(AppTypography.body)
                                    .foregroundColor(.textSecondary)
                                    .multilineTextAlignment(.center)
                            }
                        }

                        // Input ECG call-to-action box
                        Button {
                            selectedTab = .capture
                        } label: {
                            VStack(spacing: AppSpacing.md) {
                                Image(systemName: "doc.text.image")
                                    .font(.system(size: 60))
                                    .foregroundColor(.brandPrimary)

                                Text("Input an ECG")
                                    .font(AppTypography.title2)
                                    .foregroundColor(.textPrimary)
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 200)
                            .background(Color.backgroundSecondary)
                            .cornerRadius(AppRadius.lg)
                            .overlay(
                                RoundedRectangle(cornerRadius: AppRadius.lg)
                                    .stroke(Color.brandPrimary.opacity(0.3), lineWidth: 2)
                            )
                            .primaryShadow()
                        }
                        .padding(.horizontal, AppSpacing.md)

                        // Features list
                        VStack(alignment: .leading, spacing: AppSpacing.md) {
                            Text("How It Works")
                                .font(AppTypography.title3)
                                .foregroundColor(.textPrimary)
                                .padding(.horizontal, AppSpacing.md)

                            VStack(spacing: AppSpacing.sm) {
                                FeatureCard(
                                    icon: "camera.fill",
                                    title: "1. Capture",
                                    description: "Take a photo of a paper ECG or import from your library"
                                )

                                FeatureCard(
                                    icon: "gearshape.2.fill",
                                    title: "2. Configure",
                                    description: "Set paper speed and voltage gain parameters"
                                )

                                FeatureCard(
                                    icon: "cpu.fill",
                                    title: "3. Process",
                                    description: "AI digitizes all 12 leads in ~2 seconds, 100% on-device"
                                )

                                FeatureCard(
                                    icon: "square.and.arrow.up.fill",
                                    title: "4. Export",
                                    description: "Save as CSV, PDF, or HL7 FHIR format"
                                )
                            }
                            .padding(.horizontal, AppSpacing.md)
                        }

                        // Key benefits
                        VStack(spacing: AppSpacing.sm) {
                            BenefitRow(icon: "checkmark.circle.fill", text: "100% On-Device Processing")
                            BenefitRow(icon: "checkmark.circle.fill", text: "Complete Privacy - Data Never Leaves Device")
                            BenefitRow(icon: "checkmark.circle.fill", text: "Works Offline - No Internet Required")
                            BenefitRow(icon: "checkmark.circle.fill", text: "Fast - 2 Second Processing Time")
                        }
                        .padding(AppSpacing.md)
                        .background(Color.brandPrimaryLight.opacity(0.3))
                        .cornerRadius(AppRadius.lg)
                        .padding(.horizontal, AppSpacing.md)

                        // Call to action
                        VStack(spacing: AppSpacing.md) {
                            Button {
                                selectedTab = .capture
                            } label: {
                                HStack(spacing: AppSpacing.sm) {
                                    Image(systemName: "camera.fill")
                                    Text("Get Started")
                                }
                            }
                            .buttonStyle(PrimaryButtonStyle())
                            .padding(.horizontal, AppSpacing.md)

                            Text("Powered by ONNX Runtime & Apple Neural Engine")
                                .font(AppTypography.caption)
                                .foregroundColor(.textMuted)
                                .multilineTextAlignment(.center)
                        }

                        Spacer(minLength: AppSpacing.xxl)
                    }
                }
            }
            .navigationTitle("Home")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

// MARK: - Feature Card

struct FeatureCard: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        HStack(alignment: .top, spacing: AppSpacing.md) {
            ZStack {
                Circle()
                    .fill(Color.brandPrimaryLight)
                    .frame(width: 44, height: 44)

                Image(systemName: icon)
                    .font(.system(size: 20))
                    .foregroundColor(.brandPrimary)
            }

            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                Text(title)
                    .font(AppTypography.headline)
                    .foregroundColor(.textPrimary)

                Text(description)
                    .font(AppTypography.subheadline)
                    .foregroundColor(.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(AppSpacing.md)
        .background(Color.backgroundSecondary)
        .cornerRadius(AppRadius.md)
        .primaryShadow()
    }
}

// MARK: - Benefit Row

struct BenefitRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: AppSpacing.sm) {
            Image(systemName: icon)
                .foregroundColor(.statusSuccess)
                .font(.system(size: 16))

            Text(text)
                .font(AppTypography.subheadline)
                .foregroundColor(.textPrimary)

            Spacer()
        }
    }
}

// MARK: - Example ECG Preview

struct ExampleECGPreview: View {
    var body: some View {
        VStack(spacing: 4) {
            // Simulate 12-lead grid (3x4 layout)
            ForEach(0..<3, id: \.self) { row in
                HStack(spacing: 4) {
                    ForEach(0..<4, id: \.self) { col in
                        ExampleECGLead(
                            leadName: leadNames[row * 4 + col],
                            waveform: sampleWaveforms[row * 4 + col]
                        )
                    }
                }
            }
        }
        .padding(8)
    }

    private let leadNames = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    private let sampleWaveforms: [[CGFloat]] = {
        // Generate 12 sample ECG waveforms
        (0..<12).map { leadIndex in
            (0..<100).map { i in
                let t = CGFloat(i) / 100.0
                // Simulate P-QRS-T complex
                let p = 0.1 * sin(t * .pi * 8)
                let qrs = leadIndex % 2 == 0 ? 0.8 * sin(t * .pi * 20) : -0.6 * sin(t * .pi * 20)
                let t_wave = 0.15 * sin(t * .pi * 6)
                return p + qrs + t_wave
            }
        }
    }()
}

// MARK: - Example ECG Lead

struct ExampleECGLead: View {
    let leadName: String
    let waveform: [CGFloat]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(leadName)
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(.textSecondary)

            GeometryReader { geometry in
                Path { path in
                    let width = geometry.size.width
                    let height = geometry.size.height
                    let stepX = width / CGFloat(waveform.count - 1)

                    path.move(to: CGPoint(x: 0, y: height / 2 - waveform[0] * height / 2))

                    for (index, value) in waveform.enumerated() {
                        let x = CGFloat(index) * stepX
                        let y = height / 2 - value * height / 2
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                .stroke(Color.brandPrimary, lineWidth: 1)
            }
            .frame(height: 40)
            .background(
                // ECG grid background
                Rectangle()
                    .fill(Color(hex: "FCE4EC").opacity(0.3))
            )
        }
        .padding(4)
        .background(Color.backgroundTertiary)
        .cornerRadius(4)
    }
}

#Preview {
    HomeView(selectedTab: .constant(.capture))
}
