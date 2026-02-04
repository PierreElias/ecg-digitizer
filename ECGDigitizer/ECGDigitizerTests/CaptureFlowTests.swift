import XCTest
@testable import ECGDigitizer

/// Tests for the complete capture flow to identify crashes
class CaptureFlowTests: XCTestCase {

    var viewModel: CaptureFlowViewModel!

    override func setUp() {
        super.setUp()
        viewModel = CaptureFlowViewModel()
    }

    override func tearDown() {
        viewModel = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testViewModelInitialization() {
        XCTAssertNotNil(viewModel, "ViewModel should initialize without crashing")
        XCTAssertNil(viewModel.capturedImage, "Captured image should be nil initially")
        XCTAssertNil(viewModel.recording, "Recording should be nil initially")
        XCTAssertEqual(viewModel.processingState, .idle, "Processing state should be idle")
    }

    func testOnDeviceProcessorInitialization() {
        let processor = OnDeviceECGProcessor()
        XCTAssertNotNil(processor, "OnDeviceECGProcessor should initialize without crashing")
    }

    // MARK: - Image Processing Tests

    func testImageSelectionDoesNotCrash() async throws {
        // Create a test ECG image
        guard let testImage = createTestECGImage() else {
            XCTFail("Could not create test image")
            return
        }

        await MainActor.run {
            viewModel.capturedImage = testImage
        }

        // Wait a bit to see if delayed crash happens
        try await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds

        let capturedImage = await MainActor.run { viewModel.capturedImage }
        XCTAssertNotNil(capturedImage, "Captured image should still be set after 3 seconds")
    }

    func testProcessingDoesNotCrash() async throws {
        // Create a test ECG image
        guard let testImage = createTestECGImage() else {
            XCTFail("Could not create test image")
            return
        }

        await MainActor.run {
            viewModel.capturedImage = testImage
            viewModel.parameters = .standard
        }

        // Start processing
        let expectation = expectation(description: "Processing completes or fails")

        Task { @MainActor in
            await viewModel.processImage()
            expectation.fulfill()
        }

        // Wait up to 30 seconds for processing
        await fulfillment(of: [expectation], timeout: 30)

        // Check if we got a result or error (both are OK, just shouldn't crash)
        let processingState = await MainActor.run { viewModel.processingState }
        print("Processing state: \(processingState)")

        // The important thing is we got here without crashing
        XCTAssertTrue(true, "Processing completed without crash")
    }

    // MARK: - Camera Tests

    func testCameraManagerInitialization() {
        let cameraManager = CameraManager()
        XCTAssertNotNil(cameraManager, "CameraManager should initialize without crashing")
    }

    // MARK: - Helper Methods

    private func createTestECGImage() -> UIImage? {
        // Create a simple test image representing an ECG
        let size = CGSize(width: 1000, height: 800)
        UIGraphicsBeginImageContext(size)
        defer { UIGraphicsEndImageContext() }

        guard let context = UIGraphicsGetCurrentContext() else { return nil }

        // White background
        context.setFillColor(UIColor.white.cgColor)
        context.fill(CGRect(origin: .zero, size: size))

        // Draw grid lines (red)
        context.setStrokeColor(UIColor.red.cgColor)
        context.setLineWidth(1)

        // Horizontal lines
        for i in stride(from: 0, to: Int(size.height), by: 20) {
            context.move(to: CGPoint(x: 0, y: i))
            context.addLine(to: CGPoint(x: size.width, y: i))
        }

        // Vertical lines
        for i in stride(from: 0, to: Int(size.width), by: 20) {
            context.move(to: CGPoint(x: i, y: 0))
            context.addLine(to: CGPoint(x: i, y: size.height))
        }

        context.strokePath()

        // Draw some ECG waveforms (black)
        context.setStrokeColor(UIColor.black.cgColor)
        context.setLineWidth(2)

        let waveformCount = 12
        let waveformHeight = size.height / CGFloat(waveformCount)

        for row in 0..<waveformCount {
            let baseY = CGFloat(row) * waveformHeight + waveformHeight / 2

            context.move(to: CGPoint(x: 0, y: baseY))

            // Simple sine wave pattern
            for x in stride(from: 0, to: size.width, by: 5) {
                let y = baseY + sin(x / 50) * 30
                context.addLine(to: CGPoint(x: x, y: y))
            }

            context.strokePath()
        }

        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
