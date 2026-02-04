import SwiftUI

// MARK: - Status Badge

struct StatusBadge: View {
    let text: String
    let status: Status

    enum Status {
        case success
        case warning
        case error

        var color: Color {
            switch self {
            case .success: return .statusSuccess
            case .warning: return .statusWarning
            case .error: return .statusError
            }
        }
    }

    var body: some View {
        Text(text)
            .font(AppTypography.caption)
            .fontWeight(.medium)
            .foregroundColor(status.color)
            .padding(.horizontal, AppSpacing.sm)
            .padding(.vertical, AppSpacing.xs)
            .background(status.color.opacity(0.1))
            .cornerRadius(AppRadius.full)
    }
}

// MARK: - Section Header

struct SectionHeader: View {
    let title: String

    init(_ title: String) {
        self.title = title
    }

    var body: some View {
        Text(title)
            .font(AppTypography.headline)
            .foregroundColor(.textPrimary)
    }
}

// MARK: - Empty State View

struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String

    var body: some View {
        VStack(spacing: AppSpacing.md) {
            Image(systemName: icon)
                .font(.system(size: 48))
                .foregroundColor(.textMuted)

            VStack(spacing: AppSpacing.xs) {
                Text(title)
                    .font(AppTypography.headline)
                    .foregroundColor(.textPrimary)

                Text(message)
                    .font(AppTypography.subheadline)
                    .foregroundColor(.textSecondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(AppSpacing.xl)
    }
}

#Preview {
    VStack(spacing: 20) {
        StatusBadge(text: "Valid", status: .success)
        StatusBadge(text: "Warning", status: .warning)
        StatusBadge(text: "Error", status: .error)
    }
}
