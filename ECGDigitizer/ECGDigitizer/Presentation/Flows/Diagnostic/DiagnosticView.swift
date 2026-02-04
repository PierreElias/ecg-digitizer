import SwiftUI
import WebKit

/// View for displaying detailed diagnostic analysis of ECG processing
struct DiagnosticView: View {
    let image: UIImage
    @StateObject private var apiClient = ECGAPIClient.shared
    @State private var diagnosticHTML: String?
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var processingState: ProcessingState = .idle
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            Color.backgroundPrimary
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                HStack {
                    Button {
                        dismiss()
                    } label: {
                        HStack(spacing: AppSpacing.xs) {
                            Image(systemName: "chevron.left")
                            Text("Back")
                        }
                        .foregroundColor(.brandPrimary)
                    }

                    Spacer()

                    Text("Pipeline Diagnostic")
                        .font(AppTypography.headline)
                        .foregroundColor(.textPrimary)

                    Spacer()

                    // Invisible button for balance
                    Button {} label: {
                        HStack(spacing: AppSpacing.xs) {
                            Image(systemName: "chevron.left")
                            Text("Back")
                        }
                    }
                    .opacity(0)
                    .disabled(true)
                }
                .padding(AppSpacing.md)
                .background(Color.backgroundSecondary)

                // Content
                if isLoading {
                    VStack(spacing: AppSpacing.xl) {
                        Spacer()

                        BrandLoadingView()

                        Text(processingState.displayMessage)
                            .font(AppTypography.body)
                            .foregroundColor(.textSecondary)
                            .multilineTextAlignment(.center)

                        Spacer()
                    }
                    .padding(AppSpacing.xl)
                } else if let html = diagnosticHTML {
                    DiagnosticWebView(html: html)
                } else if let error = errorMessage {
                    VStack(spacing: AppSpacing.md) {
                        Spacer()

                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 48))
                            .foregroundColor(.statusError)

                        Text("Diagnostic Failed")
                            .font(AppTypography.title2)
                            .foregroundColor(.textPrimary)

                        Text(error)
                            .font(AppTypography.body)
                            .foregroundColor(.textSecondary)
                            .multilineTextAlignment(.center)

                        Button {
                            Task {
                                await runDiagnostic()
                            }
                        } label: {
                            Text("Retry")
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .padding(.top, AppSpacing.md)

                        Spacer()
                    }
                    .padding(AppSpacing.xl)
                } else {
                    VStack(spacing: AppSpacing.md) {
                        Spacer()

                        Image(systemName: "doc.text.magnifyingglass")
                            .font(.system(size: 48))
                            .foregroundColor(.brandPrimary)

                        Text("Ready to Analyze")
                            .font(AppTypography.title2)
                            .foregroundColor(.textPrimary)

                        Text("Run a detailed diagnostic analysis of the ECG processing pipeline")
                            .font(AppTypography.body)
                            .foregroundColor(.textSecondary)
                            .multilineTextAlignment(.center)

                        Button {
                            Task {
                                await runDiagnostic()
                            }
                        } label: {
                            Text("Run Diagnostic")
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .padding(.top, AppSpacing.md)

                        Spacer()
                    }
                    .padding(AppSpacing.xl)
                }
            }
        }
        .navigationBarHidden(true)
    }

    private func runDiagnostic() async {
        isLoading = true
        errorMessage = nil
        diagnosticHTML = nil

        do {
            let response = try await apiClient.generateDiagnostic(
                image: image,
                progressCallback: { state in
                    processingState = state
                }
            )

            diagnosticHTML = response.html
        } catch {
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }
}

// MARK: - Diagnostic Web View

struct DiagnosticWebView: UIViewRepresentable {
    let html: String

    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.navigationDelegate = context.coordinator
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        webView.loadHTMLString(html, baseURL: nil)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, WKNavigationDelegate {
        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            // Allow all navigation within the HTML
            decisionHandler(.allow)
        }
    }
}

// MARK: - Preview

#Preview {
    let sampleImage = UIImage(systemName: "heart.text.square")!
    return DiagnosticView(image: sampleImage)
}
