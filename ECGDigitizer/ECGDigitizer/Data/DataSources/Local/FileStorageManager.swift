import Foundation
import UIKit

/// Manages file storage for ECG recordings
final class FileStorageManager {

    // MARK: - Properties

    private let fileManager = FileManager.default

    private var documentsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var recordingsDirectory: URL {
        documentsDirectory.appendingPathComponent("ECGRecordings", isDirectory: true)
    }

    private var exportsDirectory: URL {
        documentsDirectory.appendingPathComponent("Exports", isDirectory: true)
    }

    // MARK: - Initialization

    init() {
        createDirectoriesIfNeeded()
    }

    private func createDirectoriesIfNeeded() {
        try? fileManager.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true)
        try? fileManager.createDirectory(at: exportsDirectory, withIntermediateDirectories: true)
    }

    // MARK: - Recording Storage

    /// Returns the directory for a specific recording
    func recordingDirectory(for recordingId: UUID) -> URL {
        recordingsDirectory.appendingPathComponent(recordingId.uuidString, isDirectory: true)
    }

    /// Creates directory for a recording
    func createRecordingDirectory(for recordingId: UUID) throws -> URL {
        let directory = recordingDirectory(for: recordingId)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Saves original image for a recording
    func saveOriginalImage(_ image: UIImage, recordingId: UUID) throws -> String {
        let directory = try createRecordingDirectory(for: recordingId)
        let imagePath = directory.appendingPathComponent("original.jpg")

        guard let imageData = image.jpegData(compressionQuality: 0.9) else {
            throw StorageError.imageConversionFailed
        }

        try imageData.write(to: imagePath, options: .completeFileProtection)
        return imagePath.lastPathComponent
    }

    /// Saves processed image for a recording
    func saveProcessedImage(_ image: UIImage, recordingId: UUID) throws -> String {
        let directory = recordingDirectory(for: recordingId)
        let imagePath = directory.appendingPathComponent("processed.jpg")

        guard let imageData = image.jpegData(compressionQuality: 0.9) else {
            throw StorageError.imageConversionFailed
        }

        try imageData.write(to: imagePath, options: .completeFileProtection)
        return imagePath.lastPathComponent
    }

    /// Saves lead data as JSON
    func saveLeadData(_ leads: [ECGLead], recordingId: UUID) throws {
        let directory = recordingDirectory(for: recordingId)
        let dataPath = directory.appendingPathComponent("leads.json")

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(leads)

        try data.write(to: dataPath, options: .completeFileProtection)
    }

    /// Saves metadata as JSON
    func saveMetadata(_ recording: ECGRecording, recordingId: UUID) throws {
        let directory = recordingDirectory(for: recordingId)
        let metadataPath = directory.appendingPathComponent("metadata.json")

        let metadata = RecordingMetadata(
            id: recording.id,
            timestamp: recording.timestamp,
            layout: recording.layout,
            parameters: recording.parameters,
            validationStatus: recording.validationStatus,
            gridCalibration: recording.gridCalibration
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(metadata)

        try data.write(to: metadataPath, options: .completeFileProtection)
    }

    // MARK: - Loading

    /// Loads original image for a recording
    func loadOriginalImage(recordingId: UUID) -> UIImage? {
        let imagePath = recordingDirectory(for: recordingId).appendingPathComponent("original.jpg")
        guard let data = try? Data(contentsOf: imagePath) else { return nil }
        return UIImage(data: data)
    }

    /// Loads lead data for a recording
    func loadLeadData(recordingId: UUID) throws -> [ECGLead] {
        let dataPath = recordingDirectory(for: recordingId).appendingPathComponent("leads.json")
        let data = try Data(contentsOf: dataPath)

        let decoder = JSONDecoder()
        return try decoder.decode([ECGLead].self, from: data)
    }

    /// Loads metadata for a recording
    func loadMetadata(recordingId: UUID) throws -> RecordingMetadata {
        let metadataPath = recordingDirectory(for: recordingId).appendingPathComponent("metadata.json")
        let data = try Data(contentsOf: metadataPath)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(RecordingMetadata.self, from: data)
    }

    // MARK: - Deletion

    /// Deletes all files for a recording
    func deleteRecording(recordingId: UUID) throws {
        let directory = recordingDirectory(for: recordingId)
        try fileManager.removeItem(at: directory)
    }

    /// Deletes all recordings
    func deleteAllRecordings() throws {
        try fileManager.removeItem(at: recordingsDirectory)
        createDirectoriesIfNeeded()
    }

    // MARK: - Listing

    /// Lists all recording IDs
    func listRecordingIds() -> [UUID] {
        guard let contents = try? fileManager.contentsOfDirectory(
            at: recordingsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ) else {
            return []
        }

        return contents.compactMap { url in
            UUID(uuidString: url.lastPathComponent)
        }
    }

    // MARK: - Export Storage

    /// Saves export file
    func saveExport(data: Data, fileName: String) throws -> URL {
        let exportPath = exportsDirectory.appendingPathComponent(fileName)
        try data.write(to: exportPath)
        return exportPath
    }

    /// Creates temporary export file
    func createTempExportFile(fileName: String) -> URL {
        fileManager.temporaryDirectory.appendingPathComponent(fileName)
    }

    /// Cleans up old export files
    func cleanupOldExports(olderThan days: Int = 7) {
        guard let contents = try? fileManager.contentsOfDirectory(
            at: exportsDirectory,
            includingPropertiesForKeys: [.creationDateKey],
            options: .skipsHiddenFiles
        ) else {
            return
        }

        let cutoffDate = Calendar.current.date(byAdding: .day, value: -days, to: Date())!

        for url in contents {
            if let attributes = try? fileManager.attributesOfItem(atPath: url.path),
               let creationDate = attributes[.creationDate] as? Date,
               creationDate < cutoffDate {
                try? fileManager.removeItem(at: url)
            }
        }
    }

    // MARK: - Storage Info

    /// Returns total storage used by recordings
    func totalStorageUsed() -> Int64 {
        var totalSize: Int64 = 0

        if let enumerator = fileManager.enumerator(
            at: recordingsDirectory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: .skipsHiddenFiles
        ) {
            for case let fileURL as URL in enumerator {
                if let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
                   let size = attributes[.size] as? Int64 {
                    totalSize += size
                }
            }
        }

        return totalSize
    }

    /// Returns formatted storage size
    func formattedStorageUsed() -> String {
        let bytes = totalStorageUsed()
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

// MARK: - Supporting Types

struct RecordingMetadata: Codable {
    let id: UUID
    let timestamp: Date
    let layout: ECGLayout
    let parameters: ProcessingParameters
    let validationStatus: ValidationStatus
    let gridCalibration: GridCalibration?
}

enum StorageError: Error, LocalizedError {
    case imageConversionFailed
    case fileNotFound
    case writeError
    case readError

    var errorDescription: String? {
        switch self {
        case .imageConversionFailed:
            return "Failed to convert image to JPEG"
        case .fileNotFound:
            return "File not found"
        case .writeError:
            return "Failed to write file"
        case .readError:
            return "Failed to read file"
        }
    }
}
