import SwiftUI
import UIKit

/// Debug view to test each component and identify crashes
struct DebugTestView: View {
    @State private var testLog: [String] = []
    @State private var isRunning = false
    @State private var currentTest = ""

    var body: some View {
        VStack(spacing: 20) {
            Text("Debug Test Suite")
                .font(.title)
                .padding()

            if isRunning {
                ProgressView()
                Text("Running: \(currentTest)")
                    .font(.caption)
            }

            Button("Run All Tests") {
                runAllTests()
            }
            .disabled(isRunning)
            .buttonStyle(.borderedProminent)

            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(testLog, id: \.self) { log in
                        Text(log)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(log.contains("PASS") ? .green : log.contains("FAIL") ? .red : .primary)
                    }
                }
                .padding()
            }
            .frame(maxHeight: .infinity)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            .padding()

            Button("Clear Log") {
                testLog.removeAll()
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }

    private func runAllTests() {
        isRunning = true
        testLog.removeAll()
        testLog.append("=== Starting Debug Tests ===")

        Task {
            await runTests()
            await MainActor.run {
                isRunning = false
                currentTest = ""
                testLog.append("=== Tests Complete ===")
            }
        }
    }

    @MainActor
    private func runTests() async {
        // Test 1: ViewModel Creation
        currentTest = "Create CaptureFlowViewModel"
        testLog.append("Testing: \(currentTest)...")
        do {
            let _ = CaptureFlowViewModel()
            try await Task.sleep(nanoseconds: 500_000_000)
            testLog.append("PASS: \(currentTest)")
        } catch {
            testLog.append("FAIL: \(currentTest) - \(error.localizedDescription)")
        }

        // Test 2: OnDeviceProcessor Creation
        currentTest = "Create OnDeviceECGProcessor"
        testLog.append("Testing: \(currentTest)...")
        do {
            let _ = OnDeviceECGProcessor()
            try await Task.sleep(nanoseconds: 500_000_000)
            testLog.append("PASS: \(currentTest)")
        } catch {
            testLog.append("FAIL: \(currentTest) - \(error.localizedDescription)")
        }

        // Test 3: CameraManager Creation
        currentTest = "Create CameraManager"
        testLog.append("Testing: \(currentTest)...")
        do {
            let _ = CameraManager()
            try await Task.sleep(nanoseconds: 500_000_000)
            testLog.append("PASS: \(currentTest)")
        } catch {
            testLog.append("FAIL: \(currentTest) - \(error.localizedDescription)")
        }

        // Test 4: Create Test Image
        currentTest = "Create Test ECG Image"
        testLog.append("Testing: \(currentTest)...")
        let testImage = createTestECGImage()
        if testImage != nil {
            testLog.append("PASS: \(currentTest)")
        } else {
            testLog.append("FAIL: \(currentTest) - Image creation failed")
        }

        // Test 5: ONNX Inference Check
        currentTest = "Check ONNX Inference Singleton"
        testLog.append("Testing: \(currentTest)...")
        do {
            let _ = ONNXInference.shared
            try await Task.sleep(nanoseconds: 500_000_000)
            testLog.append("PASS: \(currentTest)")
        } catch {
            testLog.append("FAIL: \(currentTest) - \(error.localizedDescription)")
        }

        // Test 6: DiagnosticLogger
        currentTest = "Check DiagnosticLogger"
        testLog.append("Testing: \(currentTest)...")
        DiagnosticLogger.shared.log("Test log from DebugTestView")
        testLog.append("PASS: \(currentTest)")
        testLog.append("   Debug images path: \(DiagnosticLogger.shared.getDebugImagesPath())")
    }

    private func createTestECGImage() -> UIImage? {
        let size = CGSize(width: 1000, height: 800)
        let renderer = UIGraphicsImageRenderer(size: size)

        return renderer.image { context in
            let ctx = context.cgContext

            // White background
            ctx.setFillColor(UIColor.white.cgColor)
            ctx.fill(CGRect(origin: .zero, size: size))

            // Red grid
            ctx.setStrokeColor(UIColor.red.withAlphaComponent(0.3).cgColor)
            ctx.setLineWidth(1)

            for i in stride(from: 0, to: Int(size.height), by: 20) {
                ctx.move(to: CGPoint(x: 0, y: i))
                ctx.addLine(to: CGPoint(x: size.width, y: CGFloat(i)))
            }

            for i in stride(from: 0, to: Int(size.width), by: 20) {
                ctx.move(to: CGPoint(x: i, y: 0))
                ctx.addLine(to: CGPoint(x: CGFloat(i), y: size.height))
            }

            ctx.strokePath()

            // Black ECG waveforms
            ctx.setStrokeColor(UIColor.black.cgColor)
            ctx.setLineWidth(2)

            for row in 0..<12 {
                let baseY = CGFloat(row) * (size.height / 12) + (size.height / 24)
                ctx.move(to: CGPoint(x: 0, y: baseY))

                for x in stride(from: 0, to: size.width, by: 5) {
                    let y = baseY + sin(x / 50 + Double(row)) * 30
                    ctx.addLine(to: CGPoint(x: x, y: y))
                }
            }

            ctx.strokePath()
        }
    }
}

#Preview {
    DebugTestView()
}
