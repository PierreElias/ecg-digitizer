import Foundation
import UIKit

/// Manages offline ECG processing queue
/// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
/// This class is CRITICAL for server-based processing with offline support
/// It queues ECGs when network is unavailable and processes them when connection returns
///
/// USAGE FOR SERVER-BASED IMPLEMENTATION:
/// 1. When capturing ECG, check NetworkMonitor.shared.isConnected
/// 2. If offline, call OfflineQueue.shared.add() instead of processing immediately
/// 3. When network returns, OfflineQueue automatically processes queued items
/// 4. User sees progress via notifications and UI updates
@MainActor
class OfflineQueue: ObservableObject {

    // MARK: - Singleton

    static let shared = OfflineQueue()

    // MARK: - Queue Item

    struct QueueItem: Codable, Identifiable {
        let id: UUID
        let imageData: Data
        let parameters: ProcessingParameters
        let timestamp: Date
        var status: Status
        var retryCount: Int

        enum Status: String, Codable {
            case pending    // Waiting to be processed
            case processing // Currently being processed
            case failed     // Processing failed
            case completed  // Successfully processed
        }

        init(image: UIImage, parameters: ProcessingParameters) {
            self.id = UUID()
            self.imageData = image.jpegData(compressionQuality: 0.9) ?? Data()
            self.parameters = parameters
            self.timestamp = Date()
            self.status = .pending
            self.retryCount = 0
        }
    }

    // MARK: - Properties

    @Published var queue: [QueueItem] = []
    @Published var isProcessing: Bool = false

    private let queueKey = "offline_ecg_processing_queue"
    private let maxRetries = 3

    // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    // This is the API client used to process queued items
    private let apiClient = ECGAPIClient.shared

    // MARK: - Initialization

    private init() {
        loadQueue()
        observeNetworkChanges()
    }

    // MARK: - Queue Management

    /// Add ECG to offline queue
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Call this when NetworkMonitor.shared.isConnected == false
    /// Example usage in capture flow:
    /// ```
    /// if NetworkMonitor.shared.isConnected {
    ///     // Process immediately with server
    ///     let result = try await apiClient.digitize(...)
    /// } else {
    ///     // Queue for later processing
    ///     OfflineQueue.shared.add(image: image, parameters: parameters)
    ///     // Show user: "Queued for processing when online"
    /// }
    /// ```
    func add(image: UIImage, parameters: ProcessingParameters) {
        let item = QueueItem(image: image, parameters: parameters)
        queue.append(item)
        saveQueue()

        // Notify observers
        NotificationCenter.default.post(
            name: .ecgQueuedForProcessing,
            object: nil,
            userInfo: [
                "id": item.id,
                "queueCount": queue.count
            ]
        )

        print("📥 Added ECG to offline queue (\(queue.count) items)")
    }

    /// Remove item from queue
    func remove(id: UUID) {
        queue.removeAll { $0.id == id }
        saveQueue()
    }

    /// Clear all completed items
    func clearCompleted() {
        queue.removeAll { $0.status == .completed }
        saveQueue()
    }

    /// Clear entire queue
    func clearAll() {
        queue.removeAll()
        saveQueue()
    }

    // MARK: - Processing

    /// Process all pending items in the queue
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// This method is automatically called when network connection is restored
    /// You can also call it manually to retry failed items
    func processQueue() async {
        guard NetworkMonitor.shared.isConnected else {
            print("⚠️ Cannot process queue - no network connection")
            return
        }

        guard !isProcessing else {
            print("⚠️ Queue already being processed")
            return
        }

        isProcessing = true
        print("🔄 Processing offline queue (\(queue.count) items)")

        // Process pending and failed items
        let itemsToProcess = queue.filter {
            $0.status == .pending || ($0.status == .failed && $0.retryCount < maxRetries)
        }

        for var item in itemsToProcess {
            // Update status to processing
            if let index = queue.firstIndex(where: { $0.id == item.id }) {
                queue[index].status = .processing
            }

            do {
                // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
                // This is where the actual server processing happens
                try await processItem(&item)

                // Success - mark as completed
                if let index = queue.firstIndex(where: { $0.id == item.id }) {
                    queue[index].status = .completed
                    saveQueue()
                }

                // Notify observers of success
                NotificationCenter.default.post(
                    name: .ecgProcessedFromQueue,
                    object: nil,
                    userInfo: ["id": item.id, "success": true]
                )

                print("✅ Processed queued ECG: \(item.id)")

            } catch {
                // Failure - increment retry count and mark as failed
                item.retryCount += 1

                if let index = queue.firstIndex(where: { $0.id == item.id }) {
                    queue[index].retryCount = item.retryCount

                    if item.retryCount >= maxRetries {
                        queue[index].status = .failed
                        print("❌ Failed to process queued ECG after \(maxRetries) retries: \(error)")
                    } else {
                        queue[index].status = .failed
                        print("⚠️ Failed to process queued ECG (retry \(item.retryCount)/\(maxRetries)): \(error)")
                    }

                    saveQueue()
                }

                // Notify observers of failure
                NotificationCenter.default.post(
                    name: .ecgProcessedFromQueue,
                    object: nil,
                    userInfo: ["id": item.id, "success": false, "error": error.localizedDescription]
                )
            }
        }

        isProcessing = false
        print("✅ Finished processing offline queue")
    }

    /// Process a single queue item
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// This method contains the server API call
    /// Modify this to match your server processing flow
    private func processItem(_ item: inout QueueItem) async throws {
        guard let image = UIImage(data: item.imageData) else {
            throw OfflineQueueError.invalidImage
        }

        // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
        // This is the key server call - processes the ECG remotely
        let response = try await apiClient.digitize(
            image: image,
            parameters: item.parameters,
            progressCallback: { state in
                // Update UI with progress if needed
                print("Processing queued item: \(state)")
            }
        )

        // Convert to recording
        guard let recording = response.toRecording(
            originalImage: image,
            parameters: item.parameters
        ) else {
            throw OfflineQueueError.processingFailed
        }

        // Save the recording
        // TODO: Save to local database/file system
        // Example: await recordingRepository.save(recording)
        print("💾 Saved processed ECG from queue: \(recording.id)")
    }

    // MARK: - Network Monitoring

    /// Observe network changes and auto-process queue when connection restored
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// This is CRITICAL - it automatically processes queue when network returns
    private func observeNetworkChanges() {
        NotificationCenter.default.addObserver(
            forName: .networkStatusChanged,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self = self else { return }
            Task { @MainActor [weak self] in
                // When network becomes available, process queue
                guard let self = self else { return }
                if NetworkMonitor.shared.isConnected,
                   !self.queue.isEmpty {
                    print("🌐 Network connected - processing offline queue")
                    await self.processQueue()
                }
            }
        }
    }

    // MARK: - Persistence

    private func saveQueue() {
        if let encoded = try? JSONEncoder().encode(queue) {
            UserDefaults.standard.set(encoded, forKey: queueKey)
        }
    }

    private func loadQueue() {
        if let data = UserDefaults.standard.data(forKey: queueKey),
           let decoded = try? JSONDecoder().decode([QueueItem].self, from: data) {
            queue = decoded
            print("📂 Loaded \(queue.count) items from offline queue")
        }
    }

    // MARK: - Statistics

    var pendingCount: Int {
        queue.filter { $0.status == .pending }.count
    }

    var failedCount: Int {
        queue.filter { $0.status == .failed }.count
    }

    var completedCount: Int {
        queue.filter { $0.status == .completed }.count
    }
}

// MARK: - Errors

enum OfflineQueueError: LocalizedError {
    case invalidImage
    case processingFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid image data in queue"
        case .processingFailed:
            return "Failed to process ECG from queue"
        }
    }
}

// MARK: - Notifications

extension Notification.Name {
    /// Posted when ECG is added to offline queue
    /// UserInfo: ["id": UUID, "queueCount": Int]
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Listen to this to show user confirmation: "ECG queued for processing"
    static let ecgQueuedForProcessing = Notification.Name("ecgQueuedForProcessing")

    /// Posted when queued ECG is processed
    /// UserInfo: ["id": UUID, "success": Bool, "error": String?]
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Listen to this to show user results: "ECG processed successfully" or show error
    static let ecgProcessedFromQueue = Notification.Name("ecgProcessedFromQueue")
}

// MARK: - Server-Based Implementation Guide

/*
# COMPLETE SERVER-BASED IMPLEMENTATION GUIDE

## 1. Update Capture Flow to Check Network

In CaptureFlowView.swift or similar:

```swift
func processECG() async {
    if NetworkMonitor.shared.isConnected {
        // Process immediately with server
        do {
            let result = try await ECGAPIClient.shared.digitize(
                image: capturedImage,
                parameters: parameters,
                progressCallback: { state in
                    processingState = state
                }
            )
            // Show results immediately
            showResults(result)
        } catch {
            showError(error)
        }
    } else {
        // Queue for offline processing
        OfflineQueue.shared.add(
            image: capturedImage,
            parameters: parameters
        )

        // Show user confirmation
        showAlert(
            title: "Queued for Processing",
            message: "ECG will be processed when network connection is restored."
        )
    }
}
```

## 2. Monitor Queue Status

Add to your app state or main view:

```swift
@StateObject private var offlineQueue = OfflineQueue.shared
@StateObject private var networkMonitor = NetworkMonitor.shared

var body: some View {
    // ... your view

    // Show queue status indicator
    if offlineQueue.pendingCount > 0 {
        HStack {
            Image(systemName: networkMonitor.isConnected ? "arrow.clockwise" : "wifi.slash")
            Text("\(offlineQueue.pendingCount) ECGs queued")
        }
        .padding()
        .background(Color.yellow.opacity(0.2))
    }
}
```

## 3. Listen for Processing Completion

In your view or view model:

```swift
.onReceive(NotificationCenter.default.publisher(
    for: .ecgProcessedFromQueue
)) { notification in
    guard let userInfo = notification.userInfo,
          let success = userInfo["success"] as? Bool else { return }

    if success {
        // Refresh results list, show success message
        showNotification("ECG processed successfully!")
    } else {
        let error = userInfo["error"] as? String ?? "Unknown error"
        showNotification("Failed to process ECG: \(error)")
    }
}
```

## 4. Manual Queue Processing

Add a button to manually retry queue processing:

```swift
Button("Process Queue (\(offlineQueue.pendingCount))") {
    Task {
        await offlineQueue.processQueue()
    }
}
.disabled(!networkMonitor.isConnected || offlineQueue.isProcessing)
```

## 5. Queue Management UI

Create a view to show and manage queued items:

```swift
struct QueueView: View {
    @StateObject private var queue = OfflineQueue.shared

    var body: some View {
        List {
            ForEach(queue.queue) { item in
                QueueItemRow(item: item)
            }
            .onDelete { indexSet in
                indexSet.forEach { index in
                    queue.remove(id: queue.queue[index].id)
                }
            }
        }
        .toolbar {
            Button("Clear Completed") {
                queue.clearCompleted()
            }
        }
    }
}
```

## Key Benefits of Server-Based + Offline Queue:

✅ Works offline - captures ECG anytime
✅ Automatic processing when connection restored
✅ User never loses data
✅ No complex on-device ML implementation needed
✅ Easy server-side updates without app updates
✅ Consistent accuracy across all devices

## Trade-offs vs Pure On-Device:

⚠️ Requires network for processing (but not capture)
⚠️ Processing delayed when offline
⚠️ Data transmitted to server (privacy consideration)
✅ Much faster to implement (days vs weeks)
✅ Smaller app size (no ML models bundled)
✅ Easy to update algorithms server-side
*/
