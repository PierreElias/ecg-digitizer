import Foundation
import UIKit

/// Repository for managing ECG recordings
final class ECGRecordingRepository {

    // MARK: - Properties

    private let fileStorage: FileStorageManager

    // In-memory cache
    private var cachedRecordings: [UUID: ECGRecording] = [:]

    // MARK: - Initialization

    init(fileStorage: FileStorageManager) {
        self.fileStorage = fileStorage
    }

    // MARK: - CRUD Operations

    /// Saves a recording
    func save(_ recording: ECGRecording) async throws {
        // Save image
        if let image = recording.originalImage {
            _ = try fileStorage.saveOriginalImage(image, recordingId: recording.id)
        }

        // Save lead data
        try fileStorage.saveLeadData(recording.leads, recordingId: recording.id)

        // Save metadata
        try fileStorage.saveMetadata(recording, recordingId: recording.id)

        // Update cache
        cachedRecordings[recording.id] = recording
    }

    /// Fetches a recording by ID
    func fetch(id: UUID) async throws -> ECGRecording {
        // Check cache first
        if let cached = cachedRecordings[id] {
            return cached
        }

        // Load from storage
        let metadata = try fileStorage.loadMetadata(recordingId: id)
        let leads = try fileStorage.loadLeadData(recordingId: id)
        let originalImage = fileStorage.loadOriginalImage(recordingId: id)

        let recording = ECGRecording(
            id: metadata.id,
            timestamp: metadata.timestamp,
            originalImageData: originalImage?.jpegData(compressionQuality: 0.9),
            parameters: metadata.parameters,
            layout: metadata.layout,
            leads: leads,
            gridCalibration: metadata.gridCalibration,
            validationStatus: metadata.validationStatus
        )

        // Update cache
        cachedRecordings[id] = recording

        return recording
    }

    /// Fetches all recordings
    func fetchAll() async throws -> [ECGRecording] {
        let ids = fileStorage.listRecordingIds()

        var recordings: [ECGRecording] = []

        for id in ids {
            do {
                let recording = try await fetch(id: id)
                recordings.append(recording)
            } catch {
                // Skip corrupted recordings
                print("Failed to load recording \(id): \(error)")
            }
        }

        // Sort by timestamp (newest first)
        return recordings.sorted { $0.timestamp > $1.timestamp }
    }

    /// Deletes a recording
    func delete(id: UUID) async throws {
        try fileStorage.deleteRecording(recordingId: id)
        cachedRecordings.removeValue(forKey: id)
    }

    /// Deletes all recordings
    func deleteAll() async throws {
        try fileStorage.deleteAllRecordings()
        cachedRecordings.removeAll()
    }

    // MARK: - Query Operations

    /// Fetches recordings with pagination
    func fetch(offset: Int, limit: Int) async throws -> [ECGRecording] {
        let all = try await fetchAll()
        let end = min(offset + limit, all.count)

        guard offset < all.count else {
            return []
        }

        return Array(all[offset..<end])
    }

    /// Fetches recordings by layout
    func fetch(layout: ECGLayout) async throws -> [ECGRecording] {
        let all = try await fetchAll()
        return all.filter { $0.layout == layout }
    }

    /// Fetches recordings within date range
    func fetch(from startDate: Date, to endDate: Date) async throws -> [ECGRecording] {
        let all = try await fetchAll()
        return all.filter { recording in
            recording.timestamp >= startDate && recording.timestamp <= endDate
        }
    }

    /// Fetches recordings by validation status
    func fetch(status: ValidationStatus) async throws -> [ECGRecording] {
        let all = try await fetchAll()
        return all.filter { $0.validationStatus == status }
    }

    // MARK: - Search

    /// Searches recordings by text query
    func search(query: String) async throws -> [ECGRecording] {
        let all = try await fetchAll()
        let lowercaseQuery = query.lowercased()

        return all.filter { recording in
            recording.reportName.lowercased().contains(lowercaseQuery) ||
            recording.layout.displayName.lowercased().contains(lowercaseQuery) ||
            recording.shortId.lowercased().contains(lowercaseQuery)
        }
    }

    // MARK: - Statistics

    /// Returns count of all recordings
    func count() -> Int {
        fileStorage.listRecordingIds().count
    }

    /// Returns storage used
    func storageUsed() -> String {
        fileStorage.formattedStorageUsed()
    }

    // MARK: - Cache Management

    /// Clears the in-memory cache
    func clearCache() {
        cachedRecordings.removeAll()
    }

    /// Preloads recordings into cache
    func preloadCache() async {
        _ = try? await fetchAll()
    }
}

// MARK: - Repository Protocol

protocol ECGRecordingRepositoryProtocol {
    func save(_ recording: ECGRecording) async throws
    func fetch(id: UUID) async throws -> ECGRecording
    func fetchAll() async throws -> [ECGRecording]
    func delete(id: UUID) async throws
}

extension ECGRecordingRepository: ECGRecordingRepositoryProtocol {}
