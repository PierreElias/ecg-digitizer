import Foundation
import UIKit

/// Crash reporter to help diagnose issues
class CrashReporter {
    static let shared = CrashReporter()

    private var crashLogs: [CrashLog] = []

    struct CrashLog {
        let timestamp: Date
        let location: String
        let error: String
        let stackTrace: [String]
    }

    private init() {
        setupCrashHandler()
    }

    private func setupCrashHandler() {
        NSSetUncaughtExceptionHandler { exception in
            let log = CrashLog(
                timestamp: Date(),
                location: "UncaughtException",
                error: exception.description,
                stackTrace: exception.callStackSymbols
            )
            CrashReporter.shared.crashLogs.append(log)
            CrashReporter.shared.saveCrashLog(log)
            print("🔴 CRASH DETECTED: \(exception.description)")
            print("🔴 Stack trace: \(exception.callStackSymbols)")
        }
    }

    func logError(_ error: Error, location: String) {
        let log = CrashLog(
            timestamp: Date(),
            location: location,
            error: error.localizedDescription,
            stackTrace: Thread.callStackSymbols
        )
        crashLogs.append(log)
        saveCrashLog(log)
        print("⚠️ Error at \(location): \(error.localizedDescription)")
    }

    private func saveCrashLog(_ log: CrashLog) {
        let logString = """
        ===== CRASH LOG =====
        Time: \(log.timestamp)
        Location: \(log.location)
        Error: \(log.error)
        Stack Trace:
        \(log.stackTrace.joined(separator: "\n"))
        ====================

        """

        // Save to documents directory
        if let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let logFile = documentsPath.appendingPathComponent("crash_logs.txt")
            try? logString.appendToFile(url: logFile)
        }

        // Also print to console
        print(logString)
    }

    func getAllLogs() -> String {
        crashLogs.map { log in
            """
            Time: \(log.timestamp)
            Location: \(log.location)
            Error: \(log.error)
            ---
            """
        }.joined(separator: "\n")
    }
}

extension String {
    func appendToFile(url: URL) throws {
        if let data = (self + "\n").data(using: .utf8) {
            if FileManager.default.fileExists(atPath: url.path) {
                if let fileHandle = try? FileHandle(forWritingTo: url) {
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(data)
                    fileHandle.closeFile()
                }
            } else {
                try data.write(to: url)
            }
        }
    }
}
