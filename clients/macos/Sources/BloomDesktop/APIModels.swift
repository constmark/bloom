import Foundation

struct BloomReadiness: Decodable {
    let schemaVersion: Int
    let object: String
    let status: String
    let progress: Int
    let model: String?
    let modelTasks: [String]
    let contextWindow: Int?
    let loadError: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case object
        case status
        case progress
        case model
        case modelTasks = "model_tasks"
        case contextWindow = "context_window"
        case loadError = "load_error"
    }

    var supportsChat: Bool {
        schemaVersion == 3
            && object == "bloom.readiness"
            && status == "ready"
            && model != nil
            && modelTasks.contains("generation")
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
    let model: String?
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Decodable {
        let delta: Delta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
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

enum SSEParser {
    static func parse(line: String) -> SSEEvent? {
        let normalized = line.hasSuffix("\r") ? String(line.dropLast()) : line
        guard !normalized.isEmpty, !normalized.hasPrefix(":"), normalized.hasPrefix("data:") else {
            return nil
        }

        var payload = String(normalized.dropFirst("data:".count))
        if payload.first == " " {
            payload.removeFirst()
        }
        if payload == "[DONE]" {
            return .done
        }
        guard let data = payload.data(using: .utf8) else {
            return nil
        }
        return .data(data)
    }
}

enum BloomClientError: LocalizedError {
    case invalidServerURL
    case incompatibleServer
    case serverNotReady(String)
    case invalidHTTPResponse
    case http(Int, String)
    case malformedStream
    case streamEndedWithoutDone

    var errorDescription: String? {
        switch self {
        case .invalidServerURL:
            return "Enter a valid HTTP or HTTPS Bloom server URL."
        case .incompatibleServer:
            return "The endpoint is not a compatible Bloom server."
        case let .serverNotReady(message):
            return message
        case .invalidHTTPResponse:
            return "The server returned an invalid HTTP response."
        case let .http(status, message):
            return "Server error \(status): \(message)"
        case .malformedStream:
            return "The server returned malformed streaming data."
        case .streamEndedWithoutDone:
            return "The streaming response ended without a completion marker."
        }
    }
}
