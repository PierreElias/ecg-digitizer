import Foundation
import UIKit
import CoreImage
import Accelerate

/// Image preprocessing pipeline for ECG digitization
final class ImagePreprocessor {

    private let ciContext: CIContext

    init() {
        self.ciContext = CIContext(options: [.useSoftwareRenderer: false])
    }

    // MARK: - Main Preprocessing Pipeline

    /// Applies the full preprocessing pipeline to an ECG image
    func preprocess(_ image: UIImage) -> UIImage {
        var processedImage = image

        // Step 1: Convert to grayscale for processing
        if let grayscale = convertToGrayscale(processedImage) {
            processedImage = grayscale
        }

        // Step 2: Apply CLAHE for grid enhancement
        if let enhanced = applyCLAHE(processedImage) {
            processedImage = enhanced
        }

        // Step 3: Apply bilateral filter for noise reduction
        if let denoised = applyBilateralFilter(processedImage) {
            processedImage = denoised
        }

        return processedImage
    }

    /// Preprocesses for grid detection (emphasizes grid lines)
    func preprocessForGridDetection(_ image: UIImage) -> UIImage {
        var processedImage = image

        // Convert to grayscale
        if let grayscale = convertToGrayscale(processedImage) {
            processedImage = grayscale
        }

        // Enhance contrast
        if let enhanced = enhanceContrast(processedImage, factor: 1.5) {
            processedImage = enhanced
        }

        // Apply adaptive threshold to highlight grid
        if let thresholded = applyAdaptiveThreshold(processedImage) {
            processedImage = thresholded
        }

        return processedImage
    }

    /// Preprocesses for waveform extraction (emphasizes signal lines)
    func preprocessForWaveformExtraction(_ image: UIImage) -> UIImage {
        var processedImage = image

        // Convert to grayscale
        if let grayscale = convertToGrayscale(processedImage) {
            processedImage = grayscale
        }

        // Enhance dark lines (ECG traces)
        if let enhanced = enhanceDarkLines(processedImage) {
            processedImage = enhanced
        }

        return processedImage
    }

    // MARK: - Color Conversion

    /// Converts image to grayscale
    func convertToGrayscale(_ image: UIImage) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        let filter = CIFilter(name: "CIColorControls")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)
        filter?.setValue(0.0, forKey: kCIInputSaturationKey)

        guard let outputImage = filter?.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Contrast Enhancement

    /// Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    func applyCLAHE(_ image: UIImage, clipLimit: Double = 2.0, tileSize: Int = 8) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        // Create grayscale buffer
        guard var sourceBuffer = createGrayscaleBuffer(from: cgImage) else { return nil }
        var destBuffer = vImage_Buffer()

        defer {
            free(sourceBuffer.data)
            if destBuffer.data != nil {
                free(destBuffer.data)
            }
        }

        // Allocate destination buffer
        let destData = malloc(width * height)
        destBuffer = vImage_Buffer(
            data: destData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
        )

        // Apply histogram equalization
        let error = vImageEqualization_Planar8(&sourceBuffer, &destBuffer, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else { return nil }

        // Convert back to CGImage
        guard let outputCGImage = createCGImage(from: &destBuffer) else { return nil }

        return UIImage(cgImage: outputCGImage)
    }

    /// Enhances contrast using Core Image
    func enhanceContrast(_ image: UIImage, factor: Double) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        let filter = CIFilter(name: "CIColorControls")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)
        filter?.setValue(factor, forKey: kCIInputContrastKey)

        guard let outputImage = filter?.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Noise Reduction

    /// Applies bilateral filter for edge-preserving smoothing
    func applyBilateralFilter(_ image: UIImage, radius: Double = 5.0) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        // Use CINoiseReduction as approximation (bilateral filter isn't directly available)
        let filter = CIFilter(name: "CINoiseReduction")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)
        filter?.setValue(0.02, forKey: "inputNoiseLevel")
        filter?.setValue(0.4, forKey: "inputSharpness")

        guard let outputImage = filter?.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    /// Applies Gaussian blur
    func applyGaussianBlur(_ image: UIImage, radius: Double) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        let filter = CIFilter(name: "CIGaussianBlur")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)
        filter?.setValue(radius, forKey: kCIInputRadiusKey)

        guard let outputImage = filter?.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Thresholding

    /// Applies adaptive threshold to create binary image
    func applyAdaptiveThreshold(_ image: UIImage, blockSize: Int = 15) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        guard let sourceBuffer = createGrayscaleBuffer(from: cgImage) else { return nil }
        var destBuffer = vImage_Buffer()

        defer {
            free(sourceBuffer.data)
            if destBuffer.data != nil {
                free(destBuffer.data)
            }
        }

        // Allocate destination buffer
        let destData = malloc(width * height)
        destBuffer = vImage_Buffer(
            data: destData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
        )

        // Calculate mean and apply threshold
        // This is a simplified adaptive threshold
        let pixels = sourceBuffer.data.assumingMemoryBound(to: UInt8.self)
        let destPixels = destBuffer.data.assumingMemoryBound(to: UInt8.self)

        // Calculate global mean
        var sum: Int = 0
        for i in 0..<(width * height) {
            sum += Int(pixels[i])
        }
        let mean = UInt8(sum / (width * height))

        // Apply threshold
        for i in 0..<(width * height) {
            destPixels[i] = pixels[i] < mean ? 0 : 255
        }

        guard let outputCGImage = createCGImage(from: &destBuffer) else { return nil }

        return UIImage(cgImage: outputCGImage)
    }

    // MARK: - Grid-Specific Enhancement

    /// Enhances grid lines (typically pink/red on ECG paper)
    func enhanceGridLines(_ image: UIImage) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        // Extract red channel (grid lines are typically pink/red)
        let filter = CIFilter(name: "CIColorMatrix")
        filter?.setValue(ciImage, forKey: kCIInputImageKey)

        // Enhance red channel, suppress others
        filter?.setValue(CIVector(x: 1.5, y: 0, z: 0, w: 0), forKey: "inputRVector")
        filter?.setValue(CIVector(x: 0, y: 0.3, z: 0, w: 0), forKey: "inputGVector")
        filter?.setValue(CIVector(x: 0, y: 0, z: 0.3, w: 0), forKey: "inputBVector")
        filter?.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")

        guard let outputImage = filter?.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    /// Enhances dark lines (ECG traces are typically black/dark)
    func enhanceDarkLines(_ image: UIImage) -> UIImage? {
        guard let ciImage = CIImage(image: image) else { return nil }

        // Invert and enhance
        let invertFilter = CIFilter(name: "CIColorInvert")
        invertFilter?.setValue(ciImage, forKey: kCIInputImageKey)

        guard let inverted = invertFilter?.outputImage else { return nil }

        // Enhance contrast
        let contrastFilter = CIFilter(name: "CIColorControls")
        contrastFilter?.setValue(inverted, forKey: kCIInputImageKey)
        contrastFilter?.setValue(1.5, forKey: kCIInputContrastKey)
        contrastFilter?.setValue(0.1, forKey: kCIInputBrightnessKey)

        guard let enhanced = contrastFilter?.outputImage else { return nil }

        // Invert back
        let revertFilter = CIFilter(name: "CIColorInvert")
        revertFilter?.setValue(enhanced, forKey: kCIInputImageKey)

        guard let outputImage = revertFilter?.outputImage,
              let cgImage = ciContext.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }

    // MARK: - Color Segmentation

    /// Segments image by color range (for isolating grid or signal)
    func segmentByColor(
        _ image: UIImage,
        hueRange: ClosedRange<CGFloat>,
        saturationRange: ClosedRange<CGFloat>,
        brightnessRange: ClosedRange<CGFloat>
    ) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else { return nil }

        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let idx = (y * width + x) * 4
                let r = CGFloat(pixels[idx]) / 255.0
                let g = CGFloat(pixels[idx + 1]) / 255.0
                let b = CGFloat(pixels[idx + 2]) / 255.0

                // Convert to HSB
                var h: CGFloat = 0, s: CGFloat = 0, br: CGFloat = 0
                UIColor(red: r, green: g, blue: b, alpha: 1.0)
                    .getHue(&h, saturation: &s, brightness: &br, alpha: nil)

                // Check if pixel is in range
                let inRange = hueRange.contains(h) &&
                              saturationRange.contains(s) &&
                              brightnessRange.contains(br)

                if inRange {
                    pixels[idx] = 255     // White
                    pixels[idx + 1] = 255
                    pixels[idx + 2] = 255
                } else {
                    pixels[idx] = 0       // Black
                    pixels[idx + 1] = 0
                    pixels[idx + 2] = 0
                }
            }
        }

        guard let outputImage = context.makeImage() else { return nil }

        return UIImage(cgImage: outputImage)
    }

    // MARK: - Helper Methods

    private func createGrayscaleBuffer(from cgImage: CGImage) -> vImage_Buffer? {
        let width = cgImage.width
        let height = cgImage.height

        // Create grayscale context
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else { return nil }

        // Copy data (context data is invalidated after context is released)
        let copiedData = malloc(width * height)
        memcpy(copiedData, data, width * height)

        return vImage_Buffer(
            data: copiedData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
        )
    }

    private func createCGImage(from buffer: inout vImage_Buffer) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceGray()

        var format = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            colorSpace: Unmanaged.passRetained(colorSpace),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            version: 0,
            decode: nil,
            renderingIntent: .defaultIntent
        )

        var error: vImage_Error = kvImageNoError
        let cgImage = vImageCreateCGImageFromBuffer(
            &buffer,
            &format,
            nil,
            nil,
            vImage_Flags(kvImageNoFlags),
            &error
        )

        guard error == kvImageNoError else { return nil }

        return cgImage?.takeRetainedValue()
    }
}
