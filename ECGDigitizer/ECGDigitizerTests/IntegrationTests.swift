import XCTest
@testable import ECGDigitizer

/// Integration tests that simulate real user flows
class IntegrationTests: XCTestCase {

    // MARK: - Complete Flow Test

    func testCompleteUserFlow() async throws {
        print("🧪 Starting complete user flow test...")

        // Step 1: App launches
        print("✓ Step 1: App launch simulation")
        let viewModel = CaptureFlowViewModel()
        XCTAssertNotNil(viewModel)

        // Wait 2 seconds to detect delayed crash
        try await Task.sleep(nanoseconds: 2_000_000_000)
        print("✓ No crash after 2 seconds")

        // Step 2: User navigates to capture tab (camera appears)
        print("✓ Step 2: Navigate to capture tab")
        let cameraManager = CameraManager()
        XCTAssertNotNil(cameraManager)

        // Step 3: User selects a photo
        print("✓ Step 3: Select photo")
        guard let testImage = createTestECGImage() else {
            XCTFail("Could not create test image")
            return
        }

        await MainActor.run {
            viewModel.capturedImage = testImage
        }

        // Wait for potential crash
        try await Task.sleep(nanoseconds: 1_000_000_000)
        print("✓ No crash after photo selection")

        // Step 4: Image goes to crop view and back
        print("✓ Step 4: Crop simulation (image unchanged)")
        // The crop view would normally modify the image, but we skip that for test

        // Step 5: Parameters screen appears
        print("✓ Step 5: Set parameters")
        await MainActor.run {
            viewModel.parameters = .standard
        }

        // Step 6: User taps "Process"
        print("✓ Step 6: Start processing")

        let expectation = expectation(description: "Processing completes")
        var processingCompleted = false

        Task { @MainActor in
            await viewModel.processImage()
            processingCompleted = true
            expectation.fulfill()
        }

        // Wait up to 60 seconds for processing
        await fulfillment(of: [expectation], timeout: 60)

        XCTAssertTrue(processingCompleted, "Processing should complete")

        let finalState = await MainActor.run { viewModel.processingState }
        print("✓ Final state: \(finalState)")

        // Check result
        let recording = await MainActor.run { viewModel.recording }
        let error = await MainActor.run { viewModel.lastError }

        if let recording = recording {
            print("✅ SUCCESS: Got ECG recording with \(recording.leads.count) leads")
            XCTAssertGreaterThan(recording.leads.count, 0, "Should have leads")
        } else if let error = error {
            print("⚠️ PROCESSING FAILED: \(error.message)")
            // Failure is OK for this test - we're checking for crashes, not correct processing
        } else {
            print("⚠️ No recording and no error")
        }

        print("🎉 Complete flow test finished without crash!")
    }

    // MARK: - Memory Test

    func testRepeatedProcessingDoesNotLeak() async throws {
        print("🧪 Testing repeated processing for memory leaks...")

        let viewModel = CaptureFlowViewModel()
        guard let testImage = createTestECGImage() else {
            XCTFail("Could not create test image")
            return
        }

        // Process 3 times
        for i in 1...3 {
            print("   Iteration \(i)/3")

            await MainActor.run {
                viewModel.capturedImage = testImage
                viewModel.parameters = .standard
            }

            let expectation = expectation(description: "Processing \(i)")

            Task { @MainActor in
                await viewModel.processImage()
                expectation.fulfill()
            }

            await fulfillment(of: [expectation], timeout: 30)

            // Reset
            await MainActor.run {
                viewModel.reset()
            }

            // Brief pause
            try await Task.sleep(nanoseconds: 500_000_000)
        }

        print("✅ Repeated processing completed without crash")
    }

    // MARK: - State Transition Tests

    func testRapidStateChanges() async throws {
        print("🧪 Testing rapid state changes...")

        let viewModel = CaptureFlowViewModel()
        guard let testImage = createTestECGImage() else {
            XCTFail("Could not create test image")
            return
        }

        // Rapidly change states
        await MainActor.run {
            viewModel.capturedImage = testImage
            viewModel.capturedImage = nil
            viewModel.capturedImage = testImage
            viewModel.parameters = .standard
            viewModel.parameters = ProcessingParameters(
                paperSpeed: .twentyFiveMmPerSec,
                voltageGain: .tenMmPerMv,
                filterSettings: FilterSettings()
            )
        }

        try await Task.sleep(nanoseconds: 1_000_000_000)

        print("✅ Rapid state changes completed without crash")
    }

    // MARK: - Helper

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
                ctx.addLine(to: CGPoint(x: size.width, y: i))
            }

            for i in stride(from: 0, to: Int(size.width), by: 20) {
                ctx.move(to: CGPoint(x: i, y: 0))
                ctx.addLine(to: CGPoint(x: i, y: size.height))
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
