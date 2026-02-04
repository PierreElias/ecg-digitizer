import Foundation
import UIKit

/// Comprehensive diagnostic logger to track app lifecycle and detect crashes
class DiagnosticLogger {
    static let shared = DiagnosticLogger()

    private var logs: [String] = []
    private let logFile: URL

    private init() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        logFile = documentsPath.appendingPathComponent("diagnostic_log.txt")

        // Clear previous log
        try? "=== NEW SESSION ===\n".write(to: logFile, atomically: true, encoding: .utf8)

        log("DiagnosticLogger initialized")
    }

    func log(_ message: String, function: String = #function, file: String = #file, line: Int = #line) {
        let fileName = (file as NSString).lastPathComponent
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let logMessage = "[\(timestamp)] [\(fileName):\(line)] \(function): \(message)"

        logs.append(logMessage)
        print("📊 \(logMessage)")

        // Append to file
        if let data = (logMessage + "\n").data(using: .utf8) {
            if let fileHandle = try? FileHandle(forWritingTo: logFile) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.closeFile()
            }
        }
    }

    func logError(_ error: Error, context: String, function: String = #function, file: String = #file, line: Int = #line) {
        let errorMessage = "ERROR in \(context): \(error.localizedDescription)"
        log(errorMessage, function: function, file: file, line: line)

        // Also print stack trace
        let stackTrace = Thread.callStackSymbols.prefix(10).joined(separator: "\n  ")
        log("Stack trace:\n  \(stackTrace)", function: function, file: file, line: line)
    }

    func getAllLogs() -> String {
        logs.joined(separator: "\n")
    }

    func getLogFilePath() -> String {
        logFile.path
    }

    func exportLogs() -> String {
        (try? String(contentsOf: logFile)) ?? "No logs available"
    }
}

/// Helper for logging lifecycle events
extension DiagnosticLogger {
    func logViewAppear(_ viewName: String) {
        log("✅ View appeared: \(viewName)")
    }

    func logViewDisappear(_ viewName: String) {
        log("❌ View disappeared: \(viewName)")
    }

    func logUserAction(_ action: String) {
        log("👆 User action: \(action)")
    }

    func logProcessingStep(_ step: String) {
        log("⚙️ Processing: \(step)")
    }

    func logMemoryWarning() {
        log("⚠️ MEMORY WARNING RECEIVED")
    }

    func logCrashImminent(_ reason: String) {
        log("🔴 CRASH IMMINENT: \(reason)")
        // Force write to disk
        if let handle = try? FileHandle(forWritingTo: logFile) {
            try? handle.synchronize()
            try? handle.close()
        }
    }
}

// MARK: - Debug Visualization for Probability Maps

extension DiagnosticLogger {

    /// Directory for saving debug images
    private var debugImagesDir: URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let debugDir = documentsPath.appendingPathComponent("debug_images")
        try? FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
        return debugDir
    }

    /// Save a probability map as a grayscale image for debugging
    ///
    /// - Parameters:
    ///   - probabilities: Float array of probability values (0-1)
    ///   - width: Width of the probability map
    ///   - height: Height of the probability map
    ///   - name: Name for the saved file
    /// - Returns: URL where the image was saved, or nil if failed
    @discardableResult
    func saveProbabilityMap(
        _ probabilities: [Float],
        width: Int,
        height: Int,
        name: String
    ) -> URL? {
        guard probabilities.count == width * height else {
            log("⚠️ Probability map size mismatch: \(probabilities.count) vs \(width)x\(height)")
            return nil
        }

        // Convert to UIImage
        var pixelData = [UInt8](repeating: 0, count: width * height)
        for i in 0..<probabilities.count {
            pixelData[i] = UInt8(min(255, max(0, probabilities[i] * 255)))
        }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ), let cgImage = context.makeImage() else {
            log("⚠️ Failed to create CGImage from probability map")
            return nil
        }

        let uiImage = UIImage(cgImage: cgImage)

        // Save to file
        let timestamp = Int(Date().timeIntervalSince1970)
        let fileName = "\(name)_\(timestamp).png"
        let fileURL = debugImagesDir.appendingPathComponent(fileName)

        guard let pngData = uiImage.pngData() else {
            log("⚠️ Failed to create PNG data")
            return nil
        }

        do {
            try pngData.write(to: fileURL)
            log("✅ Saved debug image: \(fileName) (\(width)x\(height))")
            return fileURL
        } catch {
            log("⚠️ Failed to save debug image: \(error.localizedDescription)")
            return nil
        }
    }

    /// Save the original input image for comparison
    @discardableResult
    func saveInputImage(_ image: UIImage, name: String = "input") -> URL? {
        let timestamp = Int(Date().timeIntervalSince1970)
        let fileName = "\(name)_\(timestamp).png"
        let fileURL = debugImagesDir.appendingPathComponent(fileName)

        guard let pngData = image.pngData() else {
            log("⚠️ Failed to create PNG data for input image")
            return nil
        }

        do {
            try pngData.write(to: fileURL)
            log("✅ Saved input image: \(fileName)")
            return fileURL
        } catch {
            log("⚠️ Failed to save input image: \(error.localizedDescription)")
            return nil
        }
    }

    /// Save a CGImage (useful for segmentation results)
    @discardableResult
    func saveCGImage(_ cgImage: CGImage, name: String) -> URL? {
        let uiImage = UIImage(cgImage: cgImage)
        return saveInputImage(uiImage, name: name)
    }

    /// Create a color-coded overlay of all probability maps
    ///
    /// - Parameters:
    ///   - signalProb: Signal probability (displayed as green)
    ///   - gridProb: Grid probability (displayed as red)
    ///   - textProb: Text probability (displayed as blue)
    ///   - width: Width of the probability maps
    ///   - height: Height of the probability maps
    /// - Returns: URL where the overlay image was saved
    @discardableResult
    func saveOverlayVisualization(
        signalProb: [Float],
        gridProb: [Float],
        textProb: [Float],
        width: Int,
        height: Int
    ) -> URL? {
        guard signalProb.count == width * height,
              gridProb.count == width * height,
              textProb.count == width * height else {
            log("⚠️ Overlay size mismatch")
            return nil
        }

        // Create RGBA pixel data
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for i in 0..<(width * height) {
            let offset = i * 4
            // Red = Grid, Green = Signal, Blue = Text
            pixelData[offset] = UInt8(min(255, max(0, gridProb[i] * 255)))      // R
            pixelData[offset + 1] = UInt8(min(255, max(0, signalProb[i] * 255))) // G
            pixelData[offset + 2] = UInt8(min(255, max(0, textProb[i] * 255)))   // B
            pixelData[offset + 3] = 255                                          // A
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else {
            log("⚠️ Failed to create overlay CGImage")
            return nil
        }

        return saveCGImage(cgImage, name: "overlay_rgb")
    }

    /// Get list of all debug images
    func listDebugImages() -> [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: debugImagesDir,
            includingPropertiesForKeys: [.creationDateKey],
            options: .skipsHiddenFiles
        )) ?? []

        return files.sorted { url1, url2 in
            let date1 = (try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            let date2 = (try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
            return date1 > date2
        }
    }

    /// Clear all debug images
    func clearDebugImages() {
        let files = listDebugImages()
        for file in files {
            try? FileManager.default.removeItem(at: file)
        }
        log("🗑️ Cleared \(files.count) debug images")
    }

    /// Get the debug images directory path
    func getDebugImagesPath() -> String {
        debugImagesDir.path
    }
}
