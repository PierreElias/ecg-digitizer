import Foundation
import UIKit
import PDFKit

/// Exports ECG recordings to PDF format with clinical-quality visualization
final class PDFExporter {

    // MARK: - Configuration

    struct Config {
        static let pageSize = CGSize(width: 612, height: 792)  // Letter size in points (8.5" x 11")
        static let margin: CGFloat = 36  // 0.5 inch margins
        static let gridLineWidth: CGFloat = 0.25
        static let waveformLineWidth: CGFloat = 1.0
        static let minorGridColor = UIColor(red: 1.0, green: 0.8, blue: 0.8, alpha: 1.0)
        static let majorGridColor = UIColor(red: 1.0, green: 0.6, blue: 0.6, alpha: 1.0)
        static let waveformColor = UIColor.black
        static let titleFont = UIFont.boldSystemFont(ofSize: 14)
        static let bodyFont = UIFont.systemFont(ofSize: 10)
        static let smallFont = UIFont.systemFont(ofSize: 8)
    }

    // MARK: - Export Methods

    /// Exports an ECG recording to PDF data
    /// - Parameter recording: The ECG recording to export
    /// - Returns: PDF data
    func export(_ recording: ECGRecording) -> Data {
        let renderer = UIGraphicsPDFRenderer(bounds: CGRect(origin: .zero, size: Config.pageSize))

        return renderer.pdfData { context in
            // Page 1: Title and metadata
            context.beginPage()
            drawTitlePage(recording: recording, context: context)

            // Page 2: ECG waveforms
            context.beginPage()
            drawWaveformPage(recording: recording, context: context)

            // Page 3: Original image (if available)
            if recording.originalImageData != nil {
                context.beginPage()
                drawOriginalImagePage(recording: recording, context: context)
            }
        }
    }

    /// Exports an ECG recording to a temporary PDF file
    /// - Parameter recording: The ECG recording to export
    /// - Returns: URL of the temporary PDF file
    func exportToFile(_ recording: ECGRecording) throws -> URL {
        let pdfData = export(recording)

        let fileName = "\(recording.reportName.replacingOccurrences(of: " ", with: "_")).pdf"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        try pdfData.write(to: tempURL)

        return tempURL
    }

    // MARK: - Page Drawing

    private func drawTitlePage(recording: ECGRecording, context: UIGraphicsPDFRendererContext) {
        let pageRect = CGRect(origin: .zero, size: Config.pageSize)
        let contentRect = pageRect.insetBy(dx: Config.margin, dy: Config.margin)

        var yOffset = contentRect.minY

        // Title
        let title = "ECG Digitization Report"
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 24),
            .foregroundColor: UIColor.black
        ]
        let titleSize = title.size(withAttributes: titleAttributes)
        title.draw(at: CGPoint(x: (pageRect.width - titleSize.width) / 2, y: yOffset), withAttributes: titleAttributes)
        yOffset += titleSize.height + 20

        // Subtitle with date
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        dateFormatter.timeStyle = .short
        let subtitle = "Generated: \(dateFormatter.string(from: recording.timestamp))"
        let subtitleAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12),
            .foregroundColor: UIColor.darkGray
        ]
        let subtitleSize = subtitle.size(withAttributes: subtitleAttributes)
        subtitle.draw(at: CGPoint(x: (pageRect.width - subtitleSize.width) / 2, y: yOffset), withAttributes: subtitleAttributes)
        yOffset += subtitleSize.height + 40

        // Separator line
        context.cgContext.setStrokeColor(UIColor.lightGray.cgColor)
        context.cgContext.setLineWidth(1)
        context.cgContext.move(to: CGPoint(x: contentRect.minX, y: yOffset))
        context.cgContext.addLine(to: CGPoint(x: contentRect.maxX, y: yOffset))
        context.cgContext.strokePath()
        yOffset += 20

        // Metadata section
        let sectionTitle = "Recording Information"
        let sectionTitleAttributes: [NSAttributedString.Key: Any] = [
            .font: Config.titleFont,
            .foregroundColor: UIColor.black
        ]
        sectionTitle.draw(at: CGPoint(x: contentRect.minX, y: yOffset), withAttributes: sectionTitleAttributes)
        yOffset += 25

        // Metadata items
        let metadataItems: [(String, String)] = [
            ("Recording ID", recording.id.uuidString),
            ("Layout", recording.layout.displayName),
            ("Paper Speed", recording.parameters.paperSpeed.displayName),
            ("Voltage Gain", recording.parameters.voltageGain.displayName),
            ("Lead Count", "\(recording.leads.count)"),
            ("Sampling Rate", "\(Int(recording.leads.first?.samplingRate ?? 500)) Hz"),
            ("Duration", String(format: "%.1f seconds", recording.leads.first?.duration ?? 0)),
            ("Validation Status", recording.validationStatus.displayName)
        ]

        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: Config.bodyFont,
            .foregroundColor: UIColor.darkGray
        ]
        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 10),
            .foregroundColor: UIColor.black
        ]

        for (label, value) in metadataItems {
            let labelText = "\(label):"
            labelText.draw(at: CGPoint(x: contentRect.minX + 20, y: yOffset), withAttributes: labelAttributes)
            value.draw(at: CGPoint(x: contentRect.minX + 150, y: yOffset), withAttributes: valueAttributes)
            yOffset += 18
        }

        yOffset += 30

        // Grid calibration section (if available)
        if let calibration = recording.gridCalibration {
            let calibTitle = "Grid Calibration"
            calibTitle.draw(at: CGPoint(x: contentRect.minX, y: yOffset), withAttributes: sectionTitleAttributes)
            yOffset += 25

            let calibItems: [(String, String)] = [
                ("Horizontal", String(format: "%.4f mm/pixel", calibration.mmPerPixelHorizontal)),
                ("Vertical", String(format: "%.4f mm/pixel", calibration.mmPerPixelVertical)),
                ("Grid Quality", String(format: "%.1f%%", calibration.qualityScore * 100))
            ]

            for (label, value) in calibItems {
                let labelText = "\(label):"
                labelText.draw(at: CGPoint(x: contentRect.minX + 20, y: yOffset), withAttributes: labelAttributes)
                value.draw(at: CGPoint(x: contentRect.minX + 150, y: yOffset), withAttributes: valueAttributes)
                yOffset += 18
            }
        }

        // Footer
        drawFooter(context: context, pageNumber: 1)
    }

    private func drawWaveformPage(recording: ECGRecording, context: UIGraphicsPDFRendererContext) {
        let pageRect = CGRect(origin: .zero, size: Config.pageSize)
        let contentRect = pageRect.insetBy(dx: Config.margin, dy: Config.margin)

        // Header
        let title = "Digitized ECG Waveforms"
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: Config.titleFont,
            .foregroundColor: UIColor.black
        ]
        title.draw(at: CGPoint(x: contentRect.minX, y: contentRect.minY), withAttributes: titleAttributes)

        // ECG grid area
        let gridRect = CGRect(
            x: contentRect.minX,
            y: contentRect.minY + 30,
            width: contentRect.width,
            height: contentRect.height - 60
        )

        // Draw grid
        drawECGGrid(in: gridRect, context: context.cgContext)

        // Draw waveforms based on layout
        drawWaveforms(recording: recording, in: gridRect, context: context.cgContext)

        // Add calibration marks and labels
        drawCalibrationInfo(recording: recording, in: gridRect, context: context)

        // Footer
        drawFooter(context: context, pageNumber: 2)
    }

    private func drawOriginalImagePage(recording: ECGRecording, context: UIGraphicsPDFRendererContext) {
        let pageRect = CGRect(origin: .zero, size: Config.pageSize)
        let contentRect = pageRect.insetBy(dx: Config.margin, dy: Config.margin)

        // Header
        let title = "Original Captured Image"
        let titleAttributes: [NSAttributedString.Key: Any] = [
            .font: Config.titleFont,
            .foregroundColor: UIColor.black
        ]
        title.draw(at: CGPoint(x: contentRect.minX, y: contentRect.minY), withAttributes: titleAttributes)

        // Draw image
        if let imageData = recording.originalImageData,
           let image = UIImage(data: imageData) {
            let imageRect = CGRect(
                x: contentRect.minX,
                y: contentRect.minY + 30,
                width: contentRect.width,
                height: contentRect.height - 60
            )

            // Calculate aspect-fit rect
            let aspectRect = aspectFitRect(for: image.size, in: imageRect)
            image.draw(in: aspectRect)

            // Border around image
            context.cgContext.setStrokeColor(UIColor.lightGray.cgColor)
            context.cgContext.setLineWidth(0.5)
            context.cgContext.stroke(aspectRect)
        }

        // Footer
        drawFooter(context: context, pageNumber: 3)
    }

    // MARK: - Grid Drawing

    private func drawECGGrid(in rect: CGRect, context: CGContext) {
        // Calculate grid spacing (assume 25mm/s and 10mm/mV standard)
        let mmPerPoint: CGFloat = 0.35  // Approximately 1mm = 2.83 points
        let smallSquare = mmPerPoint * 2.83  // 1mm squares
        let largeSquare = smallSquare * 5    // 5mm squares

        context.saveGState()

        // Draw minor grid lines (1mm)
        context.setStrokeColor(Config.minorGridColor.cgColor)
        context.setLineWidth(Config.gridLineWidth)

        var x = rect.minX
        while x <= rect.maxX {
            context.move(to: CGPoint(x: x, y: rect.minY))
            context.addLine(to: CGPoint(x: x, y: rect.maxY))
            x += smallSquare
        }

        var y = rect.minY
        while y <= rect.maxY {
            context.move(to: CGPoint(x: rect.minX, y: y))
            context.addLine(to: CGPoint(x: rect.maxX, y: y))
            y += smallSquare
        }
        context.strokePath()

        // Draw major grid lines (5mm)
        context.setStrokeColor(Config.majorGridColor.cgColor)
        context.setLineWidth(Config.gridLineWidth * 2)

        x = rect.minX
        while x <= rect.maxX {
            context.move(to: CGPoint(x: x, y: rect.minY))
            context.addLine(to: CGPoint(x: x, y: rect.maxY))
            x += largeSquare
        }

        y = rect.minY
        while y <= rect.maxY {
            context.move(to: CGPoint(x: rect.minX, y: y))
            context.addLine(to: CGPoint(x: rect.maxX, y: y))
            y += largeSquare
        }
        context.strokePath()

        context.restoreGState()
    }

    private func drawWaveforms(recording: ECGRecording, in rect: CGRect, context: CGContext) {
        let layout = recording.layout
        let leads = recording.leads

        guard !leads.isEmpty else { return }

        let rows = layout.rows + layout.rhythmLeads
        let columns = layout.columns

        let cellWidth = rect.width / CGFloat(columns)
        let cellHeight = rect.height / CGFloat(rows)

        context.saveGState()
        context.setStrokeColor(Config.waveformColor.cgColor)
        context.setLineWidth(Config.waveformLineWidth)

        let leadOrder = layout.standardLeadOrder

        for (index, leadType) in leadOrder.enumerated() {
            guard let lead = leads.first(where: { $0.type == leadType }) else { continue }

            let row = index / columns
            let col = index % columns

            let cellRect = CGRect(
                x: rect.minX + CGFloat(col) * cellWidth,
                y: rect.minY + CGFloat(row) * cellHeight,
                width: cellWidth,
                height: cellHeight
            )

            drawSingleLead(lead, in: cellRect, context: context)

            // Draw lead label
            let labelAttributes: [NSAttributedString.Key: Any] = [
                .font: Config.smallFont,
                .foregroundColor: UIColor.black
            ]
            lead.type.rawValue.draw(
                at: CGPoint(x: cellRect.minX + 5, y: cellRect.minY + 2),
                withAttributes: labelAttributes
            )
        }

        context.restoreGState()
    }

    private func drawSingleLead(_ lead: ECGLead, in rect: CGRect, context: CGContext) {
        guard !lead.samples.isEmpty else { return }

        let samples = lead.samples
        let xScale = rect.width / CGFloat(samples.count)
        let yCenter = rect.midY

        // Scale voltage to pixels (10mm/mV standard, ~2.83 points/mm)
        let pointsPerMm: CGFloat = 2.83
        let yScale = pointsPerMm * 10  // 10mm per mV

        context.move(to: CGPoint(
            x: rect.minX,
            y: yCenter - CGFloat(samples[0]) * yScale
        ))

        for (index, voltage) in samples.enumerated() {
            let x = rect.minX + CGFloat(index) * xScale
            let y = yCenter - CGFloat(voltage) * yScale

            // Clamp to cell bounds
            let clampedY = max(rect.minY + 5, min(rect.maxY - 5, y))
            context.addLine(to: CGPoint(x: x, y: clampedY))
        }

        context.strokePath()
    }

    private func drawCalibrationInfo(recording: ECGRecording, in rect: CGRect, context: UIGraphicsPDFRendererContext) {
        let infoText = "\(recording.parameters.paperSpeed.displayName) | \(recording.parameters.voltageGain.displayName)"
        let attributes: [NSAttributedString.Key: Any] = [
            .font: Config.smallFont,
            .foregroundColor: UIColor.darkGray
        ]
        infoText.draw(
            at: CGPoint(x: rect.minX, y: rect.maxY + 5),
            withAttributes: attributes
        )
    }

    private func drawFooter(context: UIGraphicsPDFRendererContext, pageNumber: Int) {
        let pageRect = CGRect(origin: .zero, size: Config.pageSize)

        let footerText = "ECG Digitizer | Page \(pageNumber)"
        let attributes: [NSAttributedString.Key: Any] = [
            .font: Config.smallFont,
            .foregroundColor: UIColor.gray
        ]
        let textSize = footerText.size(withAttributes: attributes)
        footerText.draw(
            at: CGPoint(x: (pageRect.width - textSize.width) / 2, y: pageRect.height - Config.margin + 10),
            withAttributes: attributes
        )
    }

    // MARK: - Helpers

    private func aspectFitRect(for imageSize: CGSize, in containerRect: CGRect) -> CGRect {
        let widthRatio = containerRect.width / imageSize.width
        let heightRatio = containerRect.height / imageSize.height
        let scale = min(widthRatio, heightRatio)

        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale

        return CGRect(
            x: containerRect.midX - scaledWidth / 2,
            y: containerRect.midY - scaledHeight / 2,
            width: scaledWidth,
            height: scaledHeight
        )
    }
}
