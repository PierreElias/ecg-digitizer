import Foundation
import Network

/// Monitors network connectivity status
/// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
/// This class is ESSENTIAL for server-based processing
/// It determines when to use server vs queue for offline processing
@MainActor
class NetworkMonitor: ObservableObject {

    // MARK: - Singleton

    static let shared = NetworkMonitor()

    // MARK: - Properties

    @Published var isConnected: Bool = false
    @Published var connectionType: ConnectionType = .unknown

    enum ConnectionType {
        case wifi
        case cellular
        case ethernet
        case unknown
    }

    private let monitor: NWPathMonitor
    private let queue = DispatchQueue(label: "NetworkMonitor")

    // MARK: - Initialization

    private init() {
        monitor = NWPathMonitor()
        startMonitoring()
    }

    // MARK: - Monitoring

    private func startMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self = self else { return }
            Task { @MainActor [weak self] in
                self?.isConnected = path.status == .satisfied
                self?.updateConnectionType(path)
            }
        }
        monitor.start(queue: queue)
    }

    private func updateConnectionType(_ path: NWPath) {
        if path.usesInterfaceType(.wifi) {
            connectionType = .wifi
        } else if path.usesInterfaceType(.cellular) {
            connectionType = .cellular
        } else if path.usesInterfaceType(.wiredEthernet) {
            connectionType = .ethernet
        } else {
            connectionType = .unknown
        }
    }

    func stopMonitoring() {
        monitor.cancel()
    }

    // MARK: - Server-Based Implementation Helpers

    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Use this method before attempting server communication
    /// Returns true if network is available for server requests
    var canUseServer: Bool {
        return isConnected
    }

    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Determines if we should prefer on-device processing based on connection
    /// For example, prefer on-device if on cellular to save data
    var shouldPreferOnDevice: Bool {
        // If no connection, must use on-device
        guard isConnected else { return true }

        // Prefer on-device on cellular to save data (optional policy)
        if connectionType == .cellular {
            // Could check user preference here
            return false // Change to true to prefer on-device on cellular
        }

        return false
    }
}

// MARK: - Notification Names

extension Notification.Name {
    /// Posted when network connectivity changes
    /// # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    /// Listen to this notification to process offline queue when connection restored
    static let networkStatusChanged = Notification.Name("networkStatusChanged")
}
