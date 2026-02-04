import SwiftUI

/// Custom view for rendering ECG waveforms with grid
struct ECGWaveformView: View {
    let leads: [ECGLead]
    let layout: ECGLayout
    let parameters: ProcessingParameters

    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background grid
                ECGGridView(
                    size: geometry.size,
                    parameters: parameters
                )

                // Waveforms
                ECGWaveformsOverlay(
                    leads: leads,
                    layout: layout,
                    size: geometry.size,
                    parameters: parameters
                )
            }
            .scaleEffect(scale)
            .offset(offset)
            .gesture(magnificationGesture)
            .gesture(dragGesture)
        }
        .background(Color.white)
        .clipped()
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                scale = max(1.0, min(4.0, value))
            }
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                if scale > 1.0 {
                    offset = value.translation
                }
            }
            .onEnded { _ in
                if scale <= 1.0 {
                    offset = .zero
                }
            }
    }
}

// MARK: - ECG Grid View

struct ECGGridView: View {
    let size: CGSize
    let parameters: ProcessingParameters

    // Grid configuration
    private let smallSquareSize: CGFloat = 4  // 1mm = 4 points on screen
    private let largeSquareSize: CGFloat = 20  // 5mm = 20 points

    var body: some View {
        Canvas { context, size in
            // Draw small grid (1mm squares) - light pink
            drawGrid(
                context: context,
                size: size,
                spacing: smallSquareSize,
                color: Color.pink.opacity(0.3),
                lineWidth: 0.5
            )

            // Draw large grid (5mm squares) - darker pink
            drawGrid(
                context: context,
                size: size,
                spacing: largeSquareSize,
                color: Color.pink.opacity(0.6),
                lineWidth: 1.0
            )
        }
    }

    private func drawGrid(
        context: GraphicsContext,
        size: CGSize,
        spacing: CGFloat,
        color: Color,
        lineWidth: CGFloat
    ) {
        var path = Path()

        // Vertical lines
        var x: CGFloat = 0
        while x <= size.width {
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: size.height))
            x += spacing
        }

        // Horizontal lines
        var y: CGFloat = 0
        while y <= size.height {
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: size.width, y: y))
            y += spacing
        }

        context.stroke(path, with: .color(color), lineWidth: lineWidth)
    }
}

// MARK: - ECG Waveforms Overlay

struct ECGWaveformsOverlay: View {
    let leads: [ECGLead]
    let layout: ECGLayout
    let size: CGSize
    let parameters: ProcessingParameters

    var body: some View {
        Canvas { context, canvasSize in
            let regions = calculateLeadRegions(canvasSize: canvasSize)

            for (index, (leadType, region)) in regions.enumerated() {
                if let lead = leads.first(where: { $0.type == leadType }) {
                    // Draw lead label
                    drawLeadLabel(context: context, leadType: leadType, region: region)

                    // Draw waveform - pass column index for sample slicing
                    let column = index % layout.columns
                    let isRhythmLead = leadType.isRhythmLead
                    drawWaveform(context: context, lead: lead, region: region, column: column, isRhythmLead: isRhythmLead)
                }
            }

            // Draw calibration info
            drawCalibrationInfo(context: context, size: canvasSize)
        }
    }

    private func calculateLeadRegions(canvasSize: CGSize) -> [(LeadType, CGRect)] {
        var regions: [(LeadType, CGRect)] = []

        let rows = layout.rows + layout.rhythmLeads
        let columns = layout.columns

        let cellWidth = canvasSize.width / CGFloat(columns)
        let cellHeight = canvasSize.height / CGFloat(rows)

        let leadOrder = layout.standardLeadOrder

        for (index, leadType) in leadOrder.enumerated() {
            let row = index / columns
            let col = index % columns

            let region = CGRect(
                x: CGFloat(col) * cellWidth,
                y: CGFloat(row) * cellHeight,
                width: cellWidth,
                height: cellHeight
            )

            regions.append((leadType, region))
        }

        // Add rhythm leads
        for i in 0..<layout.rhythmLeads {
            let rhythmType: LeadType = [.R1, .R2, .R3][i]
            let rhythmRow = layout.rows + i

            let region = CGRect(
                x: 0,
                y: CGFloat(rhythmRow) * cellHeight,
                width: canvasSize.width,
                height: cellHeight
            )

            regions.append((rhythmType, region))
        }

        return regions
    }

    private func drawLeadLabel(
        context: GraphicsContext,
        leadType: LeadType,
        region: CGRect
    ) {
        let text = Text(leadType.displayName)
            .font(.system(size: 10, weight: .medium))
            .foregroundColor(.black)

        context.draw(
            text,
            at: CGPoint(x: region.minX + 5, y: region.minY + 12)
        )
    }

    private func drawWaveform(
        context: GraphicsContext,
        lead: ECGLead,
        region: CGRect,
        column: Int = 0,
        isRhythmLead: Bool = false
    ) {
        guard !lead.samples.isEmpty else { return }

        var path = Path()

        // For multi-column layouts (like 3x4), each lead only has meaningful data
        // in its respective time window (column). We need to slice the samples.
        // For rhythm leads, use all samples.
        let samples: ArraySlice<Double>
        let columns = layout.columns

        if columns > 1 && !isRhythmLead {
            // Slice samples based on column position
            // Each column represents 1/columns of the total samples
            let samplesPerColumn = lead.samples.count / columns
            let startIndex = column * samplesPerColumn
            let endIndex = min(startIndex + samplesPerColumn, lead.samples.count)
            samples = lead.samples[startIndex..<endIndex]
        } else {
            // For single-column layouts or rhythm leads, use all samples
            samples = lead.samples[0..<lead.samples.count]
        }

        guard !samples.isEmpty else { return }

        let xScale = region.width / CGFloat(samples.count)
        let yScale = region.height / 4.0  // ±2mV range
        let yCenter = region.midY

        for (index, voltage) in samples.enumerated() {
            let x = region.minX + CGFloat(index) * xScale
            let y = yCenter - CGFloat(voltage) * yScale  // Invert Y (positive up)

            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }

        context.stroke(path, with: .color(.black), lineWidth: 1.5)
    }

    private func drawCalibrationInfo(context: GraphicsContext, size: CGSize) {
        let text = Text("\(parameters.paperSpeed.displayName) | \(parameters.voltageGain.displayName)")
            .font(.system(size: 8))
            .foregroundColor(.gray)

        context.draw(
            text,
            at: CGPoint(x: size.width - 60, y: size.height - 10)
        )
    }
}

// MARK: - Single Lead Waveform View

struct SingleLeadWaveformView: View {
    let lead: ECGLead
    let showGrid: Bool

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                if showGrid {
                    ECGGridView(
                        size: geometry.size,
                        parameters: .standard
                    )
                }

                Canvas { context, size in
                    drawWaveform(context: context, size: size)
                }
            }
        }
        .background(Color.white)
    }

    private func drawWaveform(context: GraphicsContext, size: CGSize) {
        guard !lead.samples.isEmpty else { return }

        var path = Path()

        let xScale = size.width / CGFloat(lead.samples.count)
        let yScale = size.height / 4.0
        let yCenter = size.height / 2

        for (index, voltage) in lead.samples.enumerated() {
            let x = CGFloat(index) * xScale
            let y = yCenter - CGFloat(voltage) * yScale

            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }

        context.stroke(path, with: .color(.black), lineWidth: 1.5)

        // Draw lead label
        let label = Text(lead.type.displayName)
            .font(.system(size: 12, weight: .bold))
            .foregroundColor(.black)

        context.draw(label, at: CGPoint(x: 15, y: 15))
    }
}

// MARK: - Calibration Mark View

struct CalibrationMarkView: View {
    let parameters: ProcessingParameters

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                // Draw 1mV calibration square
                let squareHeight: CGFloat = 40  // 1mV at standard 10mm/mV
                let squareWidth: CGFloat = 8    // 0.2s at standard 25mm/s

                var path = Path()
                let startX: CGFloat = 10
                let startY = size.height / 2 + squareHeight / 2

                // Draw calibration pulse
                path.move(to: CGPoint(x: startX, y: startY))
                path.addLine(to: CGPoint(x: startX, y: startY - squareHeight))
                path.addLine(to: CGPoint(x: startX + squareWidth, y: startY - squareHeight))
                path.addLine(to: CGPoint(x: startX + squareWidth, y: startY))

                context.stroke(path, with: .color(.black), lineWidth: 2)

                // Labels
                let voltageLabel = Text("1 mV")
                    .font(.system(size: 8))
                    .foregroundColor(.gray)

                context.draw(
                    voltageLabel,
                    at: CGPoint(x: startX + squareWidth + 15, y: size.height / 2)
                )
            }
        }
        .frame(width: 60, height: 60)
    }
}

#Preview {
    let sampleLead = ECGLead(
        type: .II,
        samples: (0..<500).map { i in
            let t = Double(i) / 500.0 * 2.0 * .pi * 3  // 3 cycles
            return sin(t) * 0.5 + sin(t * 3) * 0.2  // ECG-like waveform
        },
        samplingRate: 500
    )

    return VStack {
        SingleLeadWaveformView(lead: sampleLead, showGrid: true)
            .frame(height: 100)
            .padding()

        CalibrationMarkView(parameters: .standard)
    }
}
