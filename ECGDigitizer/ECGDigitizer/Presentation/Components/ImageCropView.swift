import SwiftUI
import UIKit

/// View for cropping images before processing
struct ImageCropView: View {
    let image: UIImage
    let onCrop: (UIImage) -> Void
    let onCancel: () -> Void

    @State private var cropRect: CGRect = .zero
    @State private var imageDisplayRect: CGRect = .zero
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ZStack {
                Color.black
                    .ignoresSafeArea()

                GeometryReader { geometry in
                    ZStack {
                        // Image
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)

                        // Crop overlay with image bounds
                        CropOverlay(
                            cropRect: $cropRect,
                            imageBounds: imageDisplayRect,
                            containerSize: geometry.size
                        )
                    }
                    .onAppear {
                        setupCropRect(in: geometry.size)
                    }
                    .onChange(of: geometry.size) { newSize in
                        setupCropRect(in: newSize)
                    }
                }
            }
            .navigationTitle("Crop Image")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        onCancel()
                        dismiss()
                    }
                    .foregroundColor(.white)
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        cropImage()
                    }
                    .foregroundColor(.brandPrimary)
                    .fontWeight(.semibold)
                }
            }
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
        }
    }

    private func setupCropRect(in containerSize: CGSize) {
        // Calculate actual image display size within the geometry
        let imageAspect = image.size.width / image.size.height
        let containerAspect = containerSize.width / containerSize.height

        var imageDisplayWidth: CGFloat
        var imageDisplayHeight: CGFloat
        var imageOffsetX: CGFloat
        var imageOffsetY: CGFloat

        if imageAspect > containerAspect {
            // Image is wider - limited by width
            imageDisplayWidth = containerSize.width
            imageDisplayHeight = containerSize.width / imageAspect
            imageOffsetX = 0
            imageOffsetY = (containerSize.height - imageDisplayHeight) / 2
        } else {
            // Image is taller - limited by height
            imageDisplayHeight = containerSize.height
            imageDisplayWidth = containerSize.height * imageAspect
            imageOffsetX = (containerSize.width - imageDisplayWidth) / 2
            imageOffsetY = 0
        }

        // Store the image display rect for crop calculations
        imageDisplayRect = CGRect(
            x: imageOffsetX,
            y: imageOffsetY,
            width: imageDisplayWidth,
            height: imageDisplayHeight
        )

        // Set crop rect to exactly match the displayed image (full coverage)
        cropRect = imageDisplayRect
    }

    private func cropImage() {
        guard let croppedImage = cropImageToRect() else { return }
        dismiss()
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 100_000_000)
            onCrop(croppedImage)
        }
    }

    private func cropImageToRect() -> UIImage? {
        let imageSize = image.size

        // Guard against zero-sized display rect
        guard imageDisplayRect.width > 0 && imageDisplayRect.height > 0 else {
            return image
        }

        // Calculate the crop rect relative to the displayed image area
        let relativeX = cropRect.origin.x - imageDisplayRect.origin.x
        let relativeY = cropRect.origin.y - imageDisplayRect.origin.y

        // Calculate the scale factor from display size to actual image size
        let scaleX = imageSize.width / imageDisplayRect.width
        let scaleY = imageSize.height / imageDisplayRect.height

        // Convert to actual image coordinates
        let cropRectInImage = CGRect(
            x: max(0, relativeX * scaleX),
            y: max(0, relativeY * scaleY),
            width: min(cropRect.width * scaleX, imageSize.width - max(0, relativeX * scaleX)),
            height: min(cropRect.height * scaleY, imageSize.height - max(0, relativeY * scaleY))
        )

        guard let cgImage = image.cgImage,
              let croppedCGImage = cgImage.cropping(to: cropRectInImage) else {
            return nil
        }

        return UIImage(cgImage: croppedCGImage, scale: image.scale, orientation: image.imageOrientation)
    }
}

// MARK: - Crop Overlay

struct CropOverlay: View {
    @Binding var cropRect: CGRect
    let imageBounds: CGRect
    let containerSize: CGSize

    private let handleSize: CGFloat = 24
    private let minCropSize: CGFloat = 50

    var body: some View {
        ZStack {
            // Dark overlay outside crop area
            Color.black.opacity(0.5)
                .mask(
                    Rectangle()
                        .overlay(
                            Rectangle()
                                .frame(width: cropRect.width, height: cropRect.height)
                                .position(x: cropRect.midX, y: cropRect.midY)
                                .blendMode(.destinationOut)
                        )
                )
                .allowsHitTesting(false)

            // Crop frame border
            Rectangle()
                .stroke(Color.white, lineWidth: 2)
                .frame(width: cropRect.width, height: cropRect.height)
                .position(x: cropRect.midX, y: cropRect.midY)
                .allowsHitTesting(false)

            // Corner handles with smooth dragging
            cornerHandle(.topLeft)
            cornerHandle(.topRight)
            cornerHandle(.bottomLeft)
            cornerHandle(.bottomRight)
        }
    }

    private enum Corner {
        case topLeft, topRight, bottomLeft, bottomRight
    }

    private func cornerHandle(_ corner: Corner) -> some View {
        let position: CGPoint
        switch corner {
        case .topLeft: position = CGPoint(x: cropRect.minX, y: cropRect.minY)
        case .topRight: position = CGPoint(x: cropRect.maxX, y: cropRect.minY)
        case .bottomLeft: position = CGPoint(x: cropRect.minX, y: cropRect.maxY)
        case .bottomRight: position = CGPoint(x: cropRect.maxX, y: cropRect.maxY)
        }

        return Circle()
            .fill(Color.white)
            .frame(width: handleSize, height: handleSize)
            .shadow(color: .black.opacity(0.3), radius: 2, x: 0, y: 1)
            .position(x: position.x, y: position.y)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        updateCropRect(for: corner, with: value.location)
                    }
            )
    }

    private func updateCropRect(for corner: Corner, with location: CGPoint) {
        // Clamp to image bounds
        let clampedX = max(imageBounds.minX, min(location.x, imageBounds.maxX))
        let clampedY = max(imageBounds.minY, min(location.y, imageBounds.maxY))

        var newRect = cropRect

        switch corner {
        case .topLeft:
            let newX = min(clampedX, cropRect.maxX - minCropSize)
            let newY = min(clampedY, cropRect.maxY - minCropSize)
            newRect = CGRect(
                x: newX,
                y: newY,
                width: cropRect.maxX - newX,
                height: cropRect.maxY - newY
            )

        case .topRight:
            let newMaxX = max(clampedX, cropRect.minX + minCropSize)
            let newY = min(clampedY, cropRect.maxY - minCropSize)
            newRect = CGRect(
                x: cropRect.minX,
                y: newY,
                width: newMaxX - cropRect.minX,
                height: cropRect.maxY - newY
            )

        case .bottomLeft:
            let newX = min(clampedX, cropRect.maxX - minCropSize)
            let newMaxY = max(clampedY, cropRect.minY + minCropSize)
            newRect = CGRect(
                x: newX,
                y: cropRect.minY,
                width: cropRect.maxX - newX,
                height: newMaxY - cropRect.minY
            )

        case .bottomRight:
            let newMaxX = max(clampedX, cropRect.minX + minCropSize)
            let newMaxY = max(clampedY, cropRect.minY + minCropSize)
            newRect = CGRect(
                x: cropRect.minX,
                y: cropRect.minY,
                width: newMaxX - cropRect.minX,
                height: newMaxY - cropRect.minY
            )
        }

        // Ensure crop rect stays within image bounds
        cropRect = newRect.intersection(imageBounds)
    }
}

#Preview {
    ImageCropView(
        image: UIImage(systemName: "photo")!,
        onCrop: { _ in },
        onCancel: {}
    )
}
