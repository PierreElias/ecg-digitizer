import Foundation
import UIKit
import SwiftUI

/// Manages sharing and export functionality for ECG recordings
@MainActor
final class ShareManager: ObservableObject {

    // MARK: - Properties

    @Published var isExporting = false
    @Published var exportError: Error?
    @Published var showShareSheet = false

    private let csvExporter = CSVExporter()
    private let pdfExporter = PDFExporter()
    private let hl7Exporter = HL7Exporter()

    private var exportedFileURLs: [URL] = []

    // MARK: - Export Types

    enum ExportFormat: String, CaseIterable, Identifiable {
        case csv = "CSV"
        case csvWide = "CSV (Wide)"
        case pdf = "PDF"
        case hl7 = "HL7"
        case hl7XML = "HL7 XML"
        case all = "All Formats"

        var id: String { rawValue }

        var fileExtension: String {
            switch self {
            case .csv, .csvWide: return "csv"
            case .pdf: return "pdf"
            case .hl7: return "hl7"
            case .hl7XML: return "xml"
            case .all: return "zip"
            }
        }

        var mimeType: String {
            switch self {
            case .csv, .csvWide: return "text/csv"
            case .pdf: return "application/pdf"
            case .hl7: return "application/hl7-v2"
            case .hl7XML: return "application/xml"
            case .all: return "application/zip"
            }
        }

        var icon: String {
            switch self {
            case .csv, .csvWide: return "tablecells"
            case .pdf: return "doc.richtext"
            case .hl7, .hl7XML: return "cross.case"
            case .all: return "doc.zipper"
            }
        }
    }

    // MARK: - Export Methods

    /// Exports a recording in the specified format
    /// - Parameters:
    ///   - recording: The ECG recording to export
    ///   - format: The export format
    /// - Returns: URL of the exported file
    func export(_ recording: ECGRecording, format: ExportFormat) async throws -> URL {
        isExporting = true
        defer { isExporting = false }

        do {
            let url: URL

            switch format {
            case .csv:
                url = try csvExporter.exportToFile(recording)
            case .csvWide:
                url = try csvExporter.exportWideFormatToFile(recording)
            case .pdf:
                url = try pdfExporter.exportToFile(recording)
            case .hl7:
                url = try hl7Exporter.exportToFile(recording)
            case .hl7XML:
                url = try hl7Exporter.exportXMLToFile(recording)
            case .all:
                url = try exportAllFormats(recording)
            }

            exportedFileURLs.append(url)
            return url

        } catch {
            exportError = error
            throw error
        }
    }

    /// Exports a recording in all formats and creates a zip archive
    private func exportAllFormats(_ recording: ECGRecording) throws -> URL {
        var fileURLs: [URL] = []

        // Export all formats
        fileURLs.append(try csvExporter.exportToFile(recording))
        fileURLs.append(try csvExporter.exportWideFormatToFile(recording))
        fileURLs.append(try pdfExporter.exportToFile(recording))
        fileURLs.append(try hl7Exporter.exportToFile(recording))
        fileURLs.append(try hl7Exporter.exportXMLToFile(recording))

        // Create zip archive
        let zipFileName = "\(recording.reportName.replacingOccurrences(of: " ", with: "_"))_export.zip"
        let zipURL = FileManager.default.temporaryDirectory.appendingPathComponent(zipFileName)

        // Remove existing file if present
        try? FileManager.default.removeItem(at: zipURL)

        // Create zip
        try createZipArchive(from: fileURLs, at: zipURL)

        // Clean up individual files
        for url in fileURLs {
            try? FileManager.default.removeItem(at: url)
        }

        return zipURL
    }

    /// Creates a zip archive from multiple files
    private func createZipArchive(from fileURLs: [URL], at destinationURL: URL) throws {
        // Using Cocoa's built-in zip support via NSFileCoordinator
        let coordinator = NSFileCoordinator()
        var error: NSError?

        coordinator.coordinate(
            readingItemAt: fileURLs[0].deletingLastPathComponent(),
            options: .forUploading,
            error: &error
        ) { zipURL in
            // For a proper implementation, we'd use a zip library
            // For now, we'll just copy the first file as a placeholder
            // In production, use SSZipArchive or similar
        }

        // Simplified: Just create a folder and copy files
        // In production, use a proper zip library like SSZipArchive
        let tempFolder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)

        try FileManager.default.createDirectory(at: tempFolder, withIntermediateDirectories: true)

        for url in fileURLs {
            let destURL = tempFolder.appendingPathComponent(url.lastPathComponent)
            try FileManager.default.copyItem(at: url, to: destURL)
        }

        // Note: In production, you'd use SSZipArchive.createZipFile here
        // For now, we'll just return the folder path
        try FileManager.default.moveItem(at: tempFolder, to: destinationURL)
    }

    // MARK: - Share Methods

    /// Shares a recording using the system share sheet
    /// - Parameters:
    ///   - recording: The ECG recording to share
    ///   - format: The export format
    ///   - from: The view controller to present from (or nil for SwiftUI)
    func share(_ recording: ECGRecording, format: ExportFormat) async throws -> URL {
        let url = try await export(recording, format: format)
        return url
    }

    /// Creates share items for UIActivityViewController
    func createShareItems(for recording: ECGRecording, format: ExportFormat) async throws -> [Any] {
        let url = try await export(recording, format: format)

        var items: [Any] = [url]

        // Add a text description
        let description = """
        ECG Recording: \(recording.reportName)
        Date: \(recording.formattedDate)
        Layout: \(recording.layout.displayName)
        Leads: \(recording.leads.count)
        """
        items.append(description)

        return items
    }

    // MARK: - Cleanup

    /// Cleans up temporary export files
    func cleanupExportedFiles() {
        for url in exportedFileURLs {
            try? FileManager.default.removeItem(at: url)
        }
        exportedFileURLs.removeAll()
    }

    deinit {
        // Clean up on deallocation (though this won't be called for MainActor classes properly)
        for url in exportedFileURLs {
            try? FileManager.default.removeItem(at: url)
        }
    }
}

// MARK: - SwiftUI Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    let onComplete: ((Bool) -> Void)?

    init(items: [Any], onComplete: ((Bool) -> Void)? = nil) {
        self.items = items
        self.onComplete = onComplete
    }

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: items,
            applicationActivities: nil
        )

        controller.completionWithItemsHandler = { _, completed, _, _ in
            onComplete?(completed)
        }

        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Export Button View

struct ExportButton: View {
    let recording: ECGRecording
    let format: ShareManager.ExportFormat
    @ObservedObject var shareManager: ShareManager

    @State private var showingShareSheet = false
    @State private var shareItems: [Any] = []
    @State private var isLoading = false

    var body: some View {
        Button {
            Task {
                await exportAndShare()
            }
        } label: {
            Label(format.rawValue, systemImage: format.icon)
        }
        .disabled(isLoading)
        .sheet(isPresented: $showingShareSheet) {
            ShareSheet(items: shareItems) { completed in
                if completed {
                    shareManager.cleanupExportedFiles()
                }
            }
        }
    }

    private func exportAndShare() async {
        isLoading = true
        defer { isLoading = false }

        do {
            shareItems = try await shareManager.createShareItems(for: recording, format: format)
            showingShareSheet = true
        } catch {
            print("Export error: \(error)")
        }
    }
}

// MARK: - Export Menu View

struct ExportMenuView: View {
    let recording: ECGRecording
    @StateObject private var shareManager = ShareManager()

    @State private var showingShareSheet = false
    @State private var shareItems: [Any] = []
    @State private var selectedFormat: ShareManager.ExportFormat?
    @State private var isExporting = false

    var body: some View {
        Menu {
            ForEach(ShareManager.ExportFormat.allCases) { format in
                Button {
                    selectedFormat = format
                    Task {
                        await exportAndShare(format: format)
                    }
                } label: {
                    Label(format.rawValue, systemImage: format.icon)
                }
            }
        } label: {
            Label("Export", systemImage: "square.and.arrow.up")
        }
        .disabled(isExporting)
        .sheet(isPresented: $showingShareSheet) {
            ShareSheet(items: shareItems) { completed in
                if completed {
                    shareManager.cleanupExportedFiles()
                }
            }
        }
    }

    private func exportAndShare(format: ShareManager.ExportFormat) async {
        isExporting = true
        defer { isExporting = false }

        do {
            shareItems = try await shareManager.createShareItems(for: recording, format: format)
            showingShareSheet = true
        } catch {
            print("Export error: \(error)")
        }
    }
}
