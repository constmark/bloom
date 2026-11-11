import Foundation

struct BloomReadiness: Decodable {
    static let currentSchemaVersion: UInt32 = 3
    static let uiProtocolVersion: UInt32 = 3

    let schemaVersion: UInt32
    let object: String
    let protocolVersion: UInt32
    let minimumUIProtocolVersion: UInt32
    let maximumUIProtocolVersion: UInt32
    let serverVersion: String
    let status: String
    let progress: Int
    let model: String
    let loading: Bool
    let loadError: String?
    let inputModalities: [String]
    let modelTasks: [String]
    let contextWindow: UInt64?
    let inFlightRequests: UInt64
    let availablePermits: UInt64
    let memoryPressureHigh: Bool
    let ramUtilization: Double

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case object
        case protocolVersion = "protocol_version"
        case minimumUIProtocolVersion = "minimum_ui_protocol_version"
        case maximumUIProtocolVersion = "maximum_ui_protocol_version"
        case serverVersion = "server_version"
        case status
        case progress
        case model
        case loading
        case loadError = "load_error"
        case inputModalities = "input_modalities"
        case modelTasks = "model_tasks"
        case contextWindow = "context_window"
        case inFlightRequests = "in_flight_requests"
        case availablePermits = "available_permits"
        case memoryPressureHigh = "memory_pressure_high"
        case ramUtilization = "ram_utilization"
    }

    enum ValidationError: LocalizedError {
        case invalid(String)

        var errorDescription: String? {
            switch self {
            case let .invalid(message):
                return message
            }
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(UInt32.self, forKey: .schemaVersion)
        object = try container.decode(String.self, forKey: .object)
        protocolVersion = try container.decode(UInt32.self, forKey: .protocolVersion)
        minimumUIProtocolVersion = try container.decode(UInt32.self, forKey: .minimumUIProtocolVersion)
        maximumUIProtocolVersion = try container.decode(UInt32.self, forKey: .maximumUIProtocolVersion)
        serverVersion = try container.decode(String.self, forKey: .serverVersion)
        status = try container.decode(String.self, forKey: .status)
        progress = try container.decode(Int.self, forKey: .progress)
        model = try container.decode(String.self, forKey: .model)
        loading = try container.decode(Bool.self, forKey: .loading)
        loadError = try container.decode(String?.self, forKey: .loadError)
        inputModalities = try container.decode([String].self, forKey: .inputModalities)
        modelTasks = try container.decode([String].self, forKey: .modelTasks)
        contextWindow = try container.decode(UInt64?.self, forKey: .contextWindow)
        inFlightRequests = try container.decode(UInt64.self, forKey: .inFlightRequests)
        availablePermits = try container.decode(UInt64.self, forKey: .availablePermits)
        memoryPressureHigh = try container.decode(Bool.self, forKey: .memoryPressureHigh)
        ramUtilization = try container.decode(Double.self, forKey: .ramUtilization)
    }

    /// Validate the complete schema-v3 readiness contract used by the native client.
    ///
    /// The endpoint may advertise a newer server protocol while retaining this UI
    /// protocol. A range that excludes either the advertised server protocol or
    /// this UI protocol is incompatible and must never be treated as readiness.
    func validate() throws {
        guard schemaVersion == Self.currentSchemaVersion, object == "bloom.readiness" else {
            throw ValidationError.invalid("unsupported readiness document identity")
        }
        guard protocolVersion > 0,
              minimumUIProtocolVersion > 0,
              maximumUIProtocolVersion > 0,
              minimumUIProtocolVersion <= maximumUIProtocolVersion,
              minimumUIProtocolVersion <= protocolVersion,
              protocolVersion <= maximumUIProtocolVersion
        else {
            throw ValidationError.invalid("readiness protocol compatibility range is invalid")
        }
        guard minimumUIProtocolVersion <= Self.uiProtocolVersion,
              Self.uiProtocolVersion <= maximumUIProtocolVersion
        else {
            throw ValidationError.invalid(
                "UI protocol \(Self.uiProtocolVersion) is outside the server's supported range"
            )
        }
        guard Self.isBoundedText(serverVersion, maximumCount: 64, trim: true, noWhitespace: true) else {
            throw ValidationError.invalid("readiness server version is missing or invalid")
        }
        guard status == "ready" || status == "not_ready" else {
            throw ValidationError.invalid("readiness status is invalid")
        }
        guard (0 ... 100).contains(progress) else {
            throw ValidationError.invalid("readiness progress is invalid")
        }
        guard Self.isBoundedText(model, maximumCount: 256, trim: true) else {
            throw ValidationError.invalid("readiness model identity is missing or invalid")
        }
        if let loadError,
           !Self.isBoundedText(loadError, maximumCount: 512) {
            throw ValidationError.invalid("readiness load error is invalid")
        }
        guard inputModalities.count <= 16,
              Set(inputModalities).count == inputModalities.count,
              inputModalities.allSatisfy({ Self.isBoundedText($0, maximumCount: 64) })
        else {
            throw ValidationError.invalid("readiness input modalities are invalid")
        }
        guard [[], ["generation"], ["embedding"], ["embedding", "rerank"], ["rerank"], ["tensor"]].contains(modelTasks)
        else {
            throw ValidationError.invalid("readiness model tasks are invalid")
        }
        if contextWindow == .some(0) {
            throw ValidationError.invalid("readiness context window must be positive when present")
        }
        guard ramUtilization.isFinite, (0.0 ... 1.0).contains(ramUtilization) else {
            throw ValidationError.invalid("readiness RAM utilization is invalid")
        }
        if status == "ready"
            && (model == "not loaded"
                || progress != 100
                || loading
                || loadError != nil
                || modelTasks.isEmpty
                || contextWindow == nil
                || availablePermits == 0
                || memoryPressureHigh) {
            throw ValidationError.invalid("readiness ready state is internally inconsistent")
        }
    }

    var isCompatible: Bool {
        do {
            try validate()
            return true
        } catch {
            return false
        }
    }

    var supportsChat: Bool {
        isCompatible
            && status == "ready"
            && modelTasks.contains("generation")
    }

    private static func isBoundedText(
        _ value: String,
        maximumCount: Int,
        trim: Bool = false,
        noWhitespace: Bool = false
    ) -> Bool {
        guard !value.isEmpty, value.count <= maximumCount else { return false }
        if trim && value.trimmingCharacters(in: .whitespacesAndNewlines) != value {
            return false
        }
        if noWhitespace && value.contains(where: { $0.isWhitespace }) {
            return false
        }
        return !value.unicodeScalars.contains { scalar in
            (scalar.value <= 0x1F) || (0x7F ... 0x9F).contains(scalar.value)
        }
    }
}

struct APIChatMessage: Codable {
    let role: String
    let content: String
}

struct ChatCompletionRequest: Encodable {
    let model: String
    let messages: [APIChatMessage]
    let maxTokens: Int
    let temperature: Double
    let topP: Double
    let stream: Bool
    let streamOptions: StreamOptions

    struct StreamOptions: Encodable {
        let includeUsage: Bool

        enum CodingKeys: String, CodingKey {
            case includeUsage = "include_usage"
        }
    }

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case stream
        case streamOptions = "stream_options"
    }
}

struct ChatCompletionChunk: Decodable {
    let id: String?
    let model: String?
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Decodable {
        let index: Int?
        let delta: Delta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Decodable {
        let content: String?
    }

    struct Usage: Decodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

/// One bounded text choice, matching the request made by the native client.
/// A transport EOF or a bare [DONE] must never turn incomplete output into success.
struct ChatStreamValidator {
    static let maximumResponseBytes = 64 * 1024 * 1024
    static let maximumOutputBytes = 16 * 1024 * 1024
    let expectedModel: String
    private var responseBytes = 0
    private var outputBytes = 0
    private var requestID: String?
    private var model: String?
    private var finished = false
    private var receivedUsage = false

    init(expectedModel: String) {
        self.expectedModel = expectedModel
    }

    mutating func observeByte() throws {
        guard responseBytes < Self.maximumResponseBytes else {
            throw BloomClientError.responseLimitExceeded
        }
        responseBytes += 1
    }

    mutating func observe(_ chunk: ChatCompletionChunk) throws {
        if let id = chunk.id {
            guard !id.isEmpty, id.utf8.count <= 128,
                  id.utf8.allSatisfy({ (65 ... 90).contains($0) || (97 ... 122).contains($0)
                      || (48 ... 57).contains($0) || $0 == 45 || $0 == 95 }),
                  requestID == nil || requestID == id else { throw BloomClientError.malformedStream }
            requestID = id
        }
        if let value = chunk.model {
            guard !value.isEmpty, value.count <= 256,
                  value.trimmingCharacters(in: .whitespacesAndNewlines) == value,
                  !value.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
                  expectedModel == "default" || value == expectedModel,
                  model == nil || model == value else { throw BloomClientError.malformedStream }
            model = value
        }
        if let usage = chunk.usage {
            let (sum, overflow) = usage.promptTokens.addingReportingOverflow(usage.completionTokens)
            guard !receivedUsage, usage.promptTokens >= 0, usage.completionTokens >= 0,
                  !overflow, sum == usage.totalTokens else { throw BloomClientError.malformedStream }
            receivedUsage = true
        }
        if chunk.choices.isEmpty {
            guard finished, chunk.usage != nil else { throw BloomClientError.malformedStream }
            return
        }
        guard !finished, chunk.choices.count == 1, let choice = chunk.choices.first,
              choice.index == 0 else { throw BloomClientError.malformedStream }
        let nextBytes = choice.delta.content?.utf8.count ?? 0
        guard nextBytes <= Self.maximumOutputBytes - outputBytes else {
            throw BloomClientError.responseLimitExceeded
        }
        outputBytes += nextBytes
        if let reason = choice.finishReason {
            guard ["stop", "length", "content_filter"].contains(reason) else {
                throw BloomClientError.malformedStream
            }
            finished = true
        }
    }

    func finish() throws {
        guard finished, requestID != nil, model != nil else { throw BloomClientError.malformedStream }
    }
}

struct APIErrorEnvelope: Decodable {
    let error: APIError

    struct APIError: Decodable {
        let message: String
    }
}

enum SSEEvent: Equatable {
    case data(Data)
    case done
}

struct SSEParser {
    // Bound the raw event, including ignored fields and comments. Parsing bytes
    // directly avoids AsyncBytes.lines retaining an unbounded unfinished line.
    static let maximumEventBytes = 1024 * 1024
    private var line = Data()
    private var payload = Data()
    private var hasData = false
    private var eventBytes = 0
    private var skipLF = false
    private var firstLine = true

    mutating func append(_ byte: UInt8) throws -> SSEEvent? {
        if skipLF {
            skipLF = false
            if byte == 0x0A { return nil }
        }
        eventBytes += 1
        guard eventBytes <= Self.maximumEventBytes else {
            throw BloomClientError.streamLimitExceeded
        }
        if byte == 0x0D || byte == 0x0A {
            skipLF = byte == 0x0D
            return try finishLine()
        }
        line.append(byte)
        return nil
    }

    private mutating func finishLine() throws -> SSEEvent? {
        defer { line.removeAll(keepingCapacity: true) }
        if firstLine {
            firstLine = false
            if line.starts(with: [0xEF, 0xBB, 0xBF]) { line.removeFirst(3) }
        }
        guard String(data: line, encoding: .utf8) != nil else {
            throw BloomClientError.malformedStream
        }
        if line.isEmpty {
            eventBytes = 0
            guard hasData else { return nil }
            let data = payload
            payload = Data()
            hasData = false
            return data == Data("[DONE]".utf8) ? .done : .data(data)
        }
        guard line == Data("data".utf8) || line.starts(with: Data("data:".utf8)) else {
            return nil
        }
        var value = line.dropFirst(min(5, line.count))
        if value.first == 0x20 { value = value.dropFirst() }
        if hasData { payload.append(0x0A) }
        payload.append(contentsOf: value)
        hasData = true
        return nil
    }

}

enum BloomClientError: LocalizedError {
    case invalidServerURL
    case invalidAPIKey
    case incompatibleServer
    case serverNotReady(String)
    case invalidHTTPResponse
    case http(Int, String)
    case streamError(String)
    case malformedStream
    case streamEndedWithoutDone
    case streamLimitExceeded
    case responseLimitExceeded
    case requestLimitExceeded

    var errorDescription: String? {
        switch self {
        case .invalidServerURL:
            return "Enter a valid HTTP or HTTPS Bloom server URL."
        case .invalidAPIKey:
            return "The API key must contain at most 4096 visible ASCII characters."
        case .incompatibleServer:
            return "The endpoint is not a compatible Bloom server."
        case let .serverNotReady(message):
            return message
        case .invalidHTTPResponse:
            return "The server returned an invalid HTTP response."
        case let .http(status, message):
            return "Server error \(status): \(message)"
        case let .streamError(message):
            return "Generation failed: \(message)"
        case .malformedStream:
            return "The server returned malformed streaming data."
        case .streamEndedWithoutDone:
            return "The streaming response ended without a completion marker."
        case .streamLimitExceeded:
            return "The server returned a streaming event larger than the 1 MiB limit."
        case .responseLimitExceeded:
            return "The response exceeds the native client's stream or output limit."
        case .requestLimitExceeded:
            return "The conversation exceeds the native client's 1 MiB request limit. Start a new conversation."
        }
    }
}
