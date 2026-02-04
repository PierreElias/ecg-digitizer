import SwiftUI

@main
struct ECGDigitizerApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .preferredColorScheme(.light)
                // Automated test disabled - was causing confusion
        }
    }

    // Automated test that runs on simulator
    private func runAutomatedTest() async {
        var output = "\n"
        output += "═══════════════════════════════════════\n"
        output += "   AUTOMATED TEST STARTING\n"
        output += "═══════════════════════════════════════\n"
        output += "\n"

        print(output, terminator: "")

        // Step 1: Create test image
        print("📝 Step 1: Creating test ECG image...")
        guard let testImage = createTestECGImage() else {
            print("❌ FAIL: Could not create test image\n")
            return
        }
        print("✅ PASS: Test image created (1000x800)\n")

        // Step 2: Initialize view model
        print("📝 Step 2: Initializing CaptureFlowViewModel...")
        let viewModel = await MainActor.run { CaptureFlowViewModel() }
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        print("✅ PASS: ViewModel initialized (no crash after 2 seconds)\n")

        // Step 3: Set captured image
        print("📝 Step 3: Setting captured image...")
        await MainActor.run {
            viewModel.capturedImage = testImage
            viewModel.parameters = .standard
        }
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        print("✅ PASS: Image set (no crash)\n")

        // Step 4: Start processing
        print("📝 Step 4: Starting ECG processing...")
        print("   ⏳ Loading ONNX models on-demand...")
        print("   ⏳ Running inference on background thread...")

        let startTime = Date()
        await viewModel.processImage()
        let duration = Date().timeIntervalSince(startTime)

        print("   ⏱️  Processing took: \(String(format: "%.2f", duration)) seconds\n")

        // Step 5: Check results
        print("📝 Step 5: Checking results...")
        let state = await MainActor.run { viewModel.processingState }
        let recording = await MainActor.run { viewModel.recording }
        let error = await MainActor.run { viewModel.lastError }

        print("   State: \(state)")

        if let recording = recording {
            print("\n🎉 SUCCESS: ECG Processing Completed!")
            print("   ├─ Leads detected: \(recording.leads.count)")
            print("   ├─ Layout: \(recording.layout.rawValue)")
            print("   ├─ Validation: \(recording.validationStatus)")
            if let grid = recording.gridCalibration {
                print("   └─ Grid calibration: \(String(format: "%.2f", grid.smallSquareWidthPixels))px/mm")
            }
        } else if let error = error {
            print("\n❌ FAIL: Processing failed")
            print("   Error: \(error.message)")
        } else {
            print("\n⚠️  WARNING: No recording and no error (unexpected state)")
        }

        print("\n")
        print("═══════════════════════════════════════")
        print("   AUTOMATED TEST COMPLETE")
        print("═══════════════════════════════════════")
        print("\n")

        // Write results to a file we can read
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let testResultFile = documentsPath.appendingPathComponent("test_results.txt")
        let resultText = """
        Test Results:
        - App Launch: PASS
        - ViewModel Init: PASS
        - Image Set: PASS
        - Processing State: \(state)
        - Recording: \(recording != nil ? "SUCCESS - \(recording!.leads.count) leads" : "NONE")
        - Error: \(error?.message ?? "NONE")
        """
        try? resultText.write(to: testResultFile, atomically: true, encoding: .utf8)
        print("📄 Results written to: \(testResultFile.path)")
    }

    private func createTestECGImage() -> UIImage? {
        let size = CGSize(width: 1000, height: 800)
        let renderer = UIGraphicsImageRenderer(size: size)

        return renderer.image { context in
            let ctx = context.cgContext

            // White background
            ctx.setFillColor(UIColor.white.cgColor)
            ctx.fill(CGRect(origin: .zero, size: size))

            // Red grid (5mm squares)
            ctx.setStrokeColor(UIColor.red.withAlphaComponent(0.3).cgColor)
            ctx.setLineWidth(1)

            for i in stride(from: 0, through: Int(size.height), by: 20) {
                ctx.move(to: CGPoint(x: 0, y: CGFloat(i)))
                ctx.addLine(to: CGPoint(x: size.width, y: CGFloat(i)))
            }

            for i in stride(from: 0, through: Int(size.width), by: 20) {
                ctx.move(to: CGPoint(x: CGFloat(i), y: 0))
                ctx.addLine(to: CGPoint(x: CGFloat(i), y: size.height))
            }

            ctx.strokePath()

            // Black ECG waveforms (12 leads)
            ctx.setStrokeColor(UIColor.black.cgColor)
            ctx.setLineWidth(2)

            for row in 0..<12 {
                let baseY = CGFloat(row) * (size.height / 12) + (size.height / 24)
                ctx.move(to: CGPoint(x: 0, y: baseY))

                // Simulate ECG waveform with P, QRS, T waves
                for x in stride(from: 0.0, through: Double(size.width), by: 2.0) {
                    let t = x / 50.0
                    let pWave = sin(t * 2 + Double(row)) * 10
                    let qrsComplex = sin(t * 4 + Double(row)) * 40
                    let tWave = sin(t + Double(row)) * 15
                    let y = baseY + CGFloat(pWave + qrsComplex + tWave)
                    ctx.addLine(to: CGPoint(x: x, y: y))
                }
            }

            ctx.strokePath()
        }
    }
}

/// Global application state
@MainActor
class AppState: ObservableObject {
    @Published var isProcessing: Bool = false
    @Published var currentRecording: ECGRecording?

    // Dependency container
    let container: DependencyContainer

    init() {
        self.container = DependencyContainer()
    }
}
