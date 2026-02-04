import Foundation

/// Client for the PathwAI EchoNext API
/// Sends ECG waveforms for AI analysis
class PathwAIClient {

    // MARK: - Singleton

    static let shared = PathwAIClient()

    // MARK: - Configuration

    /// Base URL for the PathwAI API
    private let baseURL = URL(string: "https://api-uat.pathwaihealth.com")!

    /// Target sample count for API (pads with zeros if shorter)
    private let targetSampleCount = 5000

    /// Session for API requests
    private let session: URLSession

    /// Cached access token
    private var accessToken: String?
    private var tokenExpiry: Date?

    // MARK: - Initialization

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 120
        self.session = URLSession(configuration: config)
    }

    // MARK: - Public Methods

    /// Analyze ECG using PathwAI EchoNext API
    /// - Parameters:
    ///   - recording: The ECG recording with extracted leads
    ///   - patientAge: Patient age
    ///   - patientAgeUnits: Age units (years, months, etc.)
    ///   - patientSex: Patient sex (M or F)
    ///   - clientAssertion: JWT assertion for authentication
    /// - Returns: Analysis response
    func analyzeECG(
        recording: ECGRecording,
        patientAge: Int,
        patientAgeUnits: PatientAgeUnits,
        patientSex: PatientSex,
        clientAssertion: String
    ) async throws -> EchoNextResponse {

        // Authenticate if needed
        if accessToken == nil || (tokenExpiry ?? Date.distantPast) < Date() {
            try await authenticate(clientAssertion: clientAssertion)
        }

        guard let token = accessToken else {
            throw PathwAIError.authenticationFailed("No access token")
        }

        // Build request
        let url = baseURL.appendingPathComponent("v1/echonext-reports")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Build request body
        let requestBody = try buildRequestBody(
            recording: recording,
            patientAge: patientAge,
            patientAgeUnits: patientAgeUnits,
            patientSex: patientSex
        )

        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        // Send request
        let (data, response) = try await session.data(for: request)

        // Check response
        if let httpResponse = response as? HTTPURLResponse {
            if httpResponse.statusCode != 200 {
                let errorBody = String(data: data, encoding: .utf8) ?? "Unknown error"
                throw PathwAIError.apiError(statusCode: httpResponse.statusCode, message: errorBody)
            }
        }

        // Decode response
        let decoder = JSONDecoder()
        return try decoder.decode(EchoNextResponse.self, from: data)
    }

    // MARK: - Authentication

    private func authenticate(clientAssertion: String) async throws {
        let url = baseURL.appendingPathComponent("v1/auth/token")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: String] = [
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": clientAssertion,
            "scope": "reports:identified"
        ]

        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)

        if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
            throw PathwAIError.authenticationFailed("HTTP \(httpResponse.statusCode)")
        }

        struct TokenResponse: Decodable {
            let access_token: String
            let expires_in: Int
        }

        let tokenResponse = try JSONDecoder().decode(TokenResponse.self, from: data)
        self.accessToken = tokenResponse.access_token
        self.tokenExpiry = Date().addingTimeInterval(TimeInterval(tokenResponse.expires_in - 60))
    }

    // MARK: - Request Building

    private func buildRequestBody(
        recording: ECGRecording,
        patientAge: Int,
        patientAgeUnits: PatientAgeUnits,
        patientSex: PatientSex
    ) throws -> [String: Any] {

        // Get leads and convert to base64
        let leads = recording.leads

        // Find required leads (I, II, V1-V6)
        guard let leadI = leads.first(where: { $0.type == .I }),
              let leadII = leads.first(where: { $0.type == .II }),
              let leadV1 = leads.first(where: { $0.type == .V1 }),
              let leadV2 = leads.first(where: { $0.type == .V2 }),
              let leadV3 = leads.first(where: { $0.type == .V3 }),
              let leadV4 = leads.first(where: { $0.type == .V4 }),
              let leadV5 = leads.first(where: { $0.type == .V5 }),
              let leadV6 = leads.first(where: { $0.type == .V6 }) else {
            throw PathwAIError.missingLeads("Required leads: I, II, V1-V6")
        }

        // Convert each lead to base64 (padded to 5000 samples)
        // Use correct lead indices for 3x4 layout masking:
        //   Lead I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
        let sampleRate = Int(leadI.samplingRate)

        return [
            "patient_age": patientAge,
            "patient_age_units": patientAgeUnits.rawValue,
            "patient_sex": patientSex.rawValue,
            "lead_i": encodeLeadToBase64(leadI, leadIndex: 0),    // Quarter 0
            "lead_ii": encodeLeadToBase64(leadII, leadIndex: 1),  // Rhythm strip (full)
            "lead_v1": encodeLeadToBase64(leadV1, leadIndex: 6),  // Quarter 2
            "lead_v2": encodeLeadToBase64(leadV2, leadIndex: 7),  // Quarter 2
            "lead_v3": encodeLeadToBase64(leadV3, leadIndex: 8),  // Quarter 2
            "lead_v4": encodeLeadToBase64(leadV4, leadIndex: 9),  // Quarter 3
            "lead_v5": encodeLeadToBase64(leadV5, leadIndex: 10), // Quarter 3
            "lead_v6": encodeLeadToBase64(leadV6, leadIndex: 11), // Quarter 3
            "ecg_manufacturer": "ECGDigitizer",
            "sample_rate": sampleRate,
            "sample_count": targetSampleCount,
            "lead_amplitude_per_bit": 0.001  // 1 µV resolution
        ]
    }

    // MARK: - Lead Encoding

    /// Encode ECG lead samples to base64 with 3x4 layout masking
    ///
    /// Standard 12-lead ECG layout (3 rows x 4 columns):
    ///     Column 0    Column 1    Column 2    Column 3
    ///     (0-1250)   (1250-2500) (2500-3750) (3750-5000)
    /// Row 0:  I         aVR         V1          V4
    /// Row 1:  II        aVL         V2          V5
    /// Row 2:  III       aVF         V3          V6
    ///
    /// Each lead only has data in its respective quarter, except Lead II
    /// which is the rhythm strip spanning the full 5000 samples.
    ///
    /// - Parameters:
    ///   - lead: The ECG lead to encode
    ///   - leadIndex: Index of the lead (0-11) for masking
    /// - Returns: Base64 encoded string
    private func encodeLeadToBase64(_ lead: ECGLead, leadIndex: Int) -> String {
        var samples = lead.samples

        // Pad with zeros to reach target sample count
        if samples.count < targetSampleCount {
            samples.append(contentsOf: Array(repeating: 0.0, count: targetSampleCount - samples.count))
        } else if samples.count > targetSampleCount {
            samples = Array(samples.prefix(targetSampleCount))
        }

        // Apply 3x4 layout mask
        // Each quarter is 1250 samples (5000 / 4)
        let steps = targetSampleCount / 4  // 1250
        let quarter = leadIndex / 3  // 0, 1, 2, or 3
        let quarterStart = quarter * steps
        let quarterEnd = (quarter + 1) * steps

        // Lead II (index 1) is the rhythm strip - don't mask it
        let isRhythmStrip = (leadIndex == 1)

        if !isRhythmStrip {
            // Zero out samples outside this lead's quarter
            for i in 0..<targetSampleCount {
                if i < quarterStart || i >= quarterEnd {
                    samples[i] = 0.0
                }
            }
        }

        // Convert Double samples to Float32 binary data
        var floatSamples = samples.map { Float($0) }

        // Convert to binary data
        let data = Data(bytes: &floatSamples, count: floatSamples.count * MemoryLayout<Float>.size)

        // Base64 encode
        return data.base64EncodedString()
    }

    /// Encode lead without masking (for backward compatibility)
    private func encodeLeadToBase64(_ lead: ECGLead) -> String {
        return encodeLeadToBase64(lead, leadIndex: 0)
    }
}

// MARK: - Supporting Types

enum PatientAgeUnits: String, Codable {
    case hours
    case days
    case weeks
    case months
    case years
}

enum PatientSex: String, Codable {
    case male = "M"
    case female = "F"
}

// MARK: - Response Types

struct EchoNextResponse: Decodable {
    let success: Bool?
    let analysis: AnalysisInfo?
    let patient: PatientInfo?
    let ecg: ECGInfo?
    let report_html: String?
    let error: String?

    struct AnalysisInfo: Decodable {
        let status: String  // "complete", "incomplete"
        let shd: String  // "detected", "not_detected", "unclassifiable"
        let sublabels: [String: String]?
        let metadata: ProcessingMetadata?
    }

    struct ProcessingMetadata: Decodable {
        let echonext_version: String?
        let error_code: String?
        let error_description: String?
    }

    struct PatientInfo: Decodable {
        let age: Int?
        let age_units: String?
        let sex: String?
    }

    struct ECGInfo: Decodable {
        let manufacturer: String?
        let sample_rate: Int?
        let sample_count: Int?
    }
}

// MARK: - Errors

enum PathwAIError: Error, LocalizedError {
    case authenticationFailed(String)
    case missingLeads(String)
    case apiError(statusCode: Int, message: String)
    case encodingError(String)

    var errorDescription: String? {
        switch self {
        case .authenticationFailed(let message):
            return "Authentication failed: \(message)"
        case .missingLeads(let message):
            return "Missing required leads: \(message)"
        case .apiError(let statusCode, let message):
            return "API error (\(statusCode)): \(message)"
        case .encodingError(let message):
            return "Encoding error: \(message)"
        }
    }
}

// MARK: - ECGLead Extension

extension ECGLead {
    /// Convert lead samples to base64-encoded Float32 binary data
    /// - Parameter targetLength: Target number of samples (pads with zeros if shorter)
    /// - Returns: Base64 encoded string
    func toBase64(targetLength: Int = 5000) -> String {
        var paddedSamples = samples

        // Pad with zeros to reach target length
        if paddedSamples.count < targetLength {
            paddedSamples.append(contentsOf: Array(repeating: 0.0, count: targetLength - paddedSamples.count))
        } else if paddedSamples.count > targetLength {
            paddedSamples = Array(paddedSamples.prefix(targetLength))
        }

        // Convert to Float32
        var floatSamples = paddedSamples.map { Float($0) }

        // Convert to binary data
        let data = Data(bytes: &floatSamples, count: floatSamples.count * MemoryLayout<Float>.size)

        return data.base64EncodedString()
    }
}
