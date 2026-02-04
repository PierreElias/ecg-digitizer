import Foundation

/// Exports ECG recordings to HL7 v2.x aECG format
/// Following HL7 annotated ECG (aECG) specification for 12-lead ECG data
final class HL7Exporter {

    // MARK: - Configuration

    struct Config {
        static let fieldSeparator = "|"
        static let componentSeparator = "^"
        static let subcomponentSeparator = "&"
        static let repetitionSeparator = "~"
        static let escapeCharacter = "\\"
        static let encodingCharacters = "^~\\&"
        static let sendingApplication = "ECGDigitizer"
        static let sendingFacility = "ECGDigitizerApp"
        static let messageType = "ORU^R01"
        static let processingId = "P"  // Production
        static let versionId = "2.5.1"
    }

    // MARK: - Export Methods

    /// Exports an ECG recording to HL7 v2.x format
    /// - Parameter recording: The ECG recording to export
    /// - Returns: HL7 formatted message string
    func export(_ recording: ECGRecording) -> String {
        var segments: [String] = []

        // MSH - Message Header
        segments.append(generateMSH(recording: recording))

        // PID - Patient Identification (placeholder - would be populated from patient data)
        segments.append(generatePID())

        // OBR - Observation Request
        segments.append(generateOBR(recording: recording))

        // OBX segments for each lead
        var obxSequence = 1
        for lead in recording.leads {
            segments.append(generateOBX(
                sequence: obxSequence,
                lead: lead,
                recording: recording
            ))
            obxSequence += 1
        }

        // OBX for metadata
        segments.append(generateMetadataOBX(sequence: obxSequence, recording: recording))

        return segments.joined(separator: "\r")
    }

    /// Exports an ECG recording to a temporary HL7 file
    /// - Parameter recording: The ECG recording to export
    /// - Returns: URL of the temporary HL7 file
    func exportToFile(_ recording: ECGRecording) throws -> URL {
        let hl7Content = export(recording)

        let fileName = "\(recording.reportName.replacingOccurrences(of: " ", with: "_")).hl7"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        try hl7Content.write(to: tempURL, atomically: true, encoding: .utf8)

        return tempURL
    }

    // MARK: - Segment Generation

    /// MSH - Message Header Segment
    private func generateMSH(recording: ECGRecording) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMddHHmmss"
        let timestamp = dateFormatter.string(from: Date())

        let messageControlId = recording.id.uuidString.prefix(20)

        let fields: [String] = [
            "MSH",
            Config.encodingCharacters,
            Config.sendingApplication,
            Config.sendingFacility,
            "",  // Receiving application
            "",  // Receiving facility
            timestamp,
            "",  // Security
            Config.messageType,
            String(messageControlId),
            Config.processingId,
            Config.versionId
        ]

        return fields.joined(separator: Config.fieldSeparator)
    }

    /// PID - Patient Identification Segment (placeholder)
    private func generatePID() -> String {
        // In a production app, this would be populated from patient data
        let fields: [String] = [
            "PID",
            "1",  // Set ID
            "",   // Patient ID (external)
            "",   // Patient ID (internal)
            "",   // Alternate Patient ID
            "Unknown^Patient",  // Patient name
            "",   // Mother's maiden name
            "",   // Date of birth
            "",   // Sex
            "",   // Patient alias
            "",   // Race
            "",   // Patient address
            "",   // County code
            "",   // Phone (home)
            "",   // Phone (business)
            "",   // Primary language
            "",   // Marital status
            "",   // Religion
            ""    // Patient account number
        ]

        return fields.joined(separator: Config.fieldSeparator)
    }

    /// OBR - Observation Request Segment
    private func generateOBR(recording: ECGRecording) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMddHHmmss"
        let timestamp = dateFormatter.string(from: recording.timestamp)

        let fields: [String] = [
            "OBR",
            "1",  // Set ID
            recording.id.uuidString,  // Placer order number
            recording.id.uuidString,  // Filler order number
            "93000^Electrocardiogram^CPT",  // Universal service ID
            "",   // Priority
            timestamp,  // Requested date/time
            timestamp,  // Observation date/time
            "",   // Observation end date/time
            "",   // Collection volume
            "",   // Collector identifier
            "",   // Specimen action code
            "",   // Danger code
            "",   // Relevant clinical info
            "",   // Specimen received date/time
            "",   // Specimen source
            "",   // Ordering provider
            "",   // Order callback phone number
            "",   // Placer field 1
            "",   // Placer field 2
            "",   // Filler field 1
            "",   // Filler field 2
            "",   // Results rpt/status change
            "",   // Charge to practice
            "",   // Diagnostic service section ID
            "F",  // Result status (Final)
            "",   // Parent result
            "",   // Quantity/timing
            "",   // Result copies to
            "",   // Parent
            "",   // Transportation mode
            "",   // Reason for study
            "",   // Principal result interpreter
            "",   // Assistant result interpreter
            ""    // Technician
        ]

        return fields.joined(separator: Config.fieldSeparator)
    }

    /// OBX - Observation Segment for ECG lead data
    private func generateOBX(sequence: Int, lead: ECGLead, recording: ECGRecording) -> String {
        // Encode waveform data as base64
        let waveformData = encodeWaveformData(lead.samples)

        let loincCode = getLoincCode(for: lead.type)

        let fields: [String] = [
            "OBX",
            String(sequence),  // Set ID
            "ED",  // Value type (Encapsulated Data)
            "\(loincCode)^\(lead.type.rawValue) Lead^LN",  // Observation identifier
            "",    // Observation sub-ID
            "^application^octet-stream^Base64^\(waveformData)",  // Observation value
            "mV",  // Units
            "",    // Reference range
            "",    // Abnormal flags
            "",    // Probability
            "",    // Nature of abnormal test
            "F",   // Observation result status (Final)
            "",    // Effective date of reference range
            "",    // User defined access checks
            "",    // Date/time of observation
            "",    // Producer's ID
            "",    // Responsible observer
            "",    // Observation method
            "",    // Equipment instance identifier
            "",    // Date/time of analysis
            "",    // Reserved for harmonization
            "",    // Reserved for harmonization
            "",    // Performing organization name
            ""     // Performing organization address
        ]

        return fields.joined(separator: Config.fieldSeparator)
    }

    /// OBX - Observation Segment for metadata
    private func generateMetadataOBX(sequence: Int, recording: ECGRecording) -> String {
        let metadataJson = generateMetadataJson(recording: recording)
        let encodedMetadata = Data(metadataJson.utf8).base64EncodedString()

        let fields: [String] = [
            "OBX",
            String(sequence),
            "ED",
            "METADATA^ECG Metadata^L",
            "",
            "^application^json^Base64^\(encodedMetadata)",
            "",
            "",
            "",
            "",
            "",
            "F",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]

        return fields.joined(separator: Config.fieldSeparator)
    }

    // MARK: - Helper Methods

    /// Encodes waveform samples as base64
    private func encodeWaveformData(_ samples: [Double]) -> String {
        // Convert doubles to scaled integers (microvolts) for efficient storage
        let microvolts = samples.map { Int32($0 * 1000) }

        // Pack into data
        var data = Data()
        for value in microvolts {
            var v = value.littleEndian
            data.append(Data(bytes: &v, count: MemoryLayout<Int32>.size))
        }

        return data.base64EncodedString()
    }

    /// Gets LOINC code for a lead type
    private func getLoincCode(for leadType: LeadType) -> String {
        switch leadType {
        case .I: return "11544-4"
        case .II: return "11545-1"
        case .III: return "11546-9"
        case .aVR: return "11547-7"
        case .aVL: return "11548-5"
        case .aVF: return "11549-3"
        case .V1: return "11550-1"
        case .V2: return "11551-9"
        case .V3: return "11552-7"
        case .V4: return "11553-5"
        case .V5: return "11554-3"
        case .V6: return "11555-0"
        case .R1: return "11545-1"  // Typically Lead II rhythm
        case .R2: return "11553-5"  // Typically V4 rhythm
        case .R3: return "11555-0"  // Typically V6 rhythm
        }
    }

    /// Generates JSON metadata
    private func generateMetadataJson(recording: ECGRecording) -> String {
        let metadata: [String: Any] = [
            "recordingId": recording.id.uuidString,
            "timestamp": ISO8601DateFormatter().string(from: recording.timestamp),
            "layout": recording.layout.rawValue,
            "paperSpeed": [
                "value": recording.parameters.paperSpeed.mmPerSecond,
                "unit": "mm/s"
            ],
            "voltageGain": [
                "value": recording.parameters.voltageGain.mmPerMillivolt,
                "unit": "mm/mV"
            ],
            "samplingRate": recording.leads.first?.samplingRate ?? 500,
            "leadCount": recording.leads.count,
            "validationStatus": recording.validationStatus.rawValue,
            "applicationVersion": "1.0.0",
            "applicationName": Config.sendingApplication
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: metadata, options: []),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            return jsonString
        }

        return "{}"
    }
}

// MARK: - HL7 XML Export (Alternative Format)

extension HL7Exporter {

    /// Exports ECG recording to HL7 aECG XML format
    /// - Parameter recording: The ECG recording to export
    /// - Returns: XML formatted string following HL7 aECG schema
    func exportXML(_ recording: ECGRecording) -> String {
        let dateFormatter = ISO8601DateFormatter()
        let timestamp = dateFormatter.string(from: recording.timestamp)

        var xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <AnnotatedECG xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            <id root="\(recording.id.uuidString)"/>
            <code code="93000" codeSystem="2.16.840.1.113883.6.12" displayName="Electrocardiogram"/>
            <effectiveTime value="\(timestamp.replacingOccurrences(of: "-", with: "").replacingOccurrences(of: ":", with: "").prefix(14))"/>
            <component>
                <series>
                    <code code="REPRESENTATIVE_BEAT" codeSystem="2.16.840.1.113883.5.1063"/>
                    <effectiveTime>
                        <low value="\(timestamp.replacingOccurrences(of: "-", with: "").replacingOccurrences(of: ":", with: "").prefix(14))"/>
                    </effectiveTime>

        """

        // Add each lead as a sequence set
        for lead in recording.leads {
            xml += generateLeadSequenceSetXML(lead: lead)
        }

        xml += """
                </series>
            </component>
            <component>
                <annotationSet>
                    <component>
                        <annotation>
                            <code code="MDC_ECG_HEART_RATE" codeSystem="2.16.840.1.113883.6.24"/>
                            <value value="\(recording.estimatedHeartRate ?? 0)" unit="/min"/>
                        </annotation>
                    </component>
                </annotationSet>
            </component>
        </AnnotatedECG>
        """

        return xml
    }

    private func generateLeadSequenceSetXML(lead: ECGLead) -> String {
        let samplingInterval = 1000.0 / lead.samplingRate  // ms per sample

        // Convert samples to space-separated string
        let samplesString = lead.samples.map { String(format: "%.4f", $0) }.joined(separator: " ")

        return """
                    <sequenceSet>
                        <component>
                            <sequence>
                                <code code="MDC_ECG_LEAD_\(lead.type.rawValue)" codeSystem="2.16.840.1.113883.6.24"/>
                                <value>
                                    <origin value="0" unit="mV"/>
                                    <scale value="\(String(format: "%.6f", samplingInterval))" unit="ms"/>
                                    <digits>\(samplesString)</digits>
                                </value>
                            </sequence>
                        </component>
                    </sequenceSet>

        """
    }

    /// Exports to XML file
    func exportXMLToFile(_ recording: ECGRecording) throws -> URL {
        let xmlContent = exportXML(recording)

        let fileName = "\(recording.reportName.replacingOccurrences(of: " ", with: "_")).xml"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        try xmlContent.write(to: tempURL, atomically: true, encoding: .utf8)

        return tempURL
    }
}
