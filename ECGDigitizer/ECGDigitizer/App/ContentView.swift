import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var offlineQueue = OfflineQueue.shared
    @StateObject private var networkMonitor = NetworkMonitor.shared
    @State private var selectedTab: Tab = .home
    @State private var showingSettings = false

    enum Tab {
        case home
        case capture
        case reports
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView(selectedTab: $selectedTab)
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
                .tag(Tab.home)

            CaptureFlowView()
                .tabItem {
                    Label("Capture", systemImage: "camera.fill")
                }
                .tag(Tab.capture)

            ReportsListView()
                .tabItem {
                    Label("Reports", systemImage: "list.bullet.rectangle.portrait")
                }
                .tag(Tab.reports)
        }
        .tint(.brandPrimary)
        .safeAreaInset(edge: .top) {
            // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
            // Queue status banner - shows when ECGs are queued for processing
            if offlineQueue.pendingCount > 0 {
                queueStatusBanner
            }
        }
        .overlay(alignment: .topTrailing) {
            Button {
                showingSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 20))
                    .foregroundColor(.brandPrimary)
                    .padding(AppSpacing.md)
                    .background(Color.backgroundSecondary.opacity(0.9))
                    .clipShape(Circle())
                    .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
            }
            .padding(AppSpacing.md)
            .offset(y: offlineQueue.pendingCount > 0 ? 50 : 0) // Move down when queue banner is visible
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
        .onAppear {
            // Configure tab bar appearance
            let appearance = UITabBarAppearance()
            appearance.configureWithOpaqueBackground()
            appearance.backgroundColor = UIColor(Color.backgroundSecondary)

            UITabBar.appearance().standardAppearance = appearance
            if #available(iOS 15.0, *) {
                UITabBar.appearance().scrollEdgeAppearance = appearance
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .ecgProcessedFromQueue)) { notification in
            // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
            // Handle completed queue processing
            guard let userInfo = notification.userInfo,
                  let success = userInfo["success"] as? Bool else { return }

            if success {
                // Could show a toast notification here
                print("ECG processed successfully from queue")
            } else if let error = userInfo["error"] as? String {
                print("Failed to process ECG from queue: \(error)")
            }
        }
    }

    // # CHANGE FOR SERVER-BASED IMPLEMENTATION:
    // Queue status banner shown at top of screen
    private var queueStatusBanner: some View {
        HStack(spacing: AppSpacing.sm) {
            Image(systemName: networkMonitor.isConnected ? "arrow.clockwise" : "wifi.slash")
                .foregroundColor(networkMonitor.isConnected ? .statusSuccess : .statusWarning)
                .font(.system(size: 16))

            Text("\(offlineQueue.pendingCount) ECG\(offlineQueue.pendingCount == 1 ? "" : "s") queued")
                .font(AppTypography.subheadline)
                .foregroundColor(.textPrimary)

            if offlineQueue.isProcessing {
                ProgressView()
                    .scaleEffect(0.7)
            }

            Spacer()

            if networkMonitor.isConnected {
                Button("Process Now") {
                    Task {
                        await offlineQueue.processQueue()
                    }
                }
                .font(AppTypography.caption)
                .foregroundColor(.brandPrimary)
            } else {
                Text("Waiting for network")
                    .font(AppTypography.caption)
                    .foregroundColor(.textSecondary)
            }
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.vertical, AppSpacing.sm)
        .background(Color.statusWarning.opacity(0.15))
        .overlay(
            Rectangle()
                .fill(Color.statusWarning.opacity(0.3))
                .frame(height: 1),
            alignment: .bottom
        )
    }
}

#Preview {
    ContentView()
        .environmentObject(AppState())
}
