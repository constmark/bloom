import Foundation

protocol BloomAPITransport {
    @MainActor func readiness() async throws -> BloomReadiness
    @MainActor func streamChat(
        model: String,
        messages: [APIChatMessage],
        onChunk: (ChatCompletionChunk) -> Void
    ) async throws
}

struct BloomAPIClient: BloomAPITransport {
    private static let maxReadinessResponseBytes = 64 * 1024
    private static let redirectPolicy = RejectRedirects()
    private static let defaultSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForResource = 300
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        return URLSession(configuration: configuration)
    }()

    let serverURL: URL
    let apiKey: String
    private let session: URLSession

    init(serverURL: String, apiKey: String, session: URLSession = Self.defaultSession) throws {
        let trimmed = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            let url = URL(string: trimmed),
            trimmed.utf8.count <= 2048,
            let scheme = url.scheme?.lowercased(),
            scheme == "http" || scheme == "https",
            let host = url.host, !host.isEmpty,
            url.user == nil,
            url.password == nil,
            url.query == nil,
            url.fragment == nil
        else {
            throw BloomClientError.invalidServerURL
        }
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedKey.utf8.count <= 4096,
              trimmedKey.utf8.allSatisfy({ (0x21 ... 0x7E).contains($0) }) else {
            throw BloomClientError.invalidAPIKey
        }
        self.serverURL = url
        self.apiKey = trimmedKey
        self.session = session
    }

    @MainActor
    func readiness() async throws -> BloomReadiness {
        var request = URLRequest(url: endpoint("ready"))
        request.timeoutInterval = 10
        request.cachePolicy = .reloadIgnoringLocalCacheData
        applyHeaders(to: &request)

        let (bytes, response) = try await session.bytes(for: request, delegate: Self.redirectPolicy)
        defer { bytes.task.cancel() }
        guard let http = response as? HTTPURLResponse else {
            throw BloomClientError.invalidHTTPResponse
        }
        guard (200 ... 599).contains(http.statusCode) else {
            throw BloomClientError.invalidHTTPResponse
        }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard data.count < Self.maxReadinessResponseBytes else {
                throw BloomClientError.incompatibleServer
            }
            data.append(byte)
        }
        guard http.statusCode == 200 || http.statusCode == 503 else {
            throw BloomClientError.http(http.statusCode, Self.errorMessage(from: data))
        }

        let readiness: BloomReadiness
        do {
            readiness = try JSONDecoder().decode(BloomReadiness.self, from: data)
            try readiness.validate()
        } catch {
            // A reachable endpoint with an incomplete or incompatible readiness
            // document is not a Bloom server for this client.
            throw BloomClientError.incompatibleServer
        }
        // /ready is deliberately public. Verify the supplied credential before
        // showing Ready, including when the catalog has no model loaded yet.
        try await verifyAPIAccess()
        return readiness
    }

    @MainActor
    private func verifyAPIAccess() async throws {
        var request = URLRequest(url: endpoint("v1/models"))
        request.timeoutInterval = 10
        request.cachePolicy = .reloadIgnoringLocalCacheData
        applyHeaders(to: &request)
        let (bytes, response) = try await session.bytes(for: request, delegate: Self.redirectPolicy)
        defer { bytes.task.cancel() }
        guard let http = response as? HTTPURLResponse else { throw BloomClientError.invalidHTTPResponse }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard data.count < Self.maxReadinessResponseBytes else { throw BloomClientError.incompatibleServer }
            data.append(byte)
        }
        guard http.statusCode == 200 else {
            throw BloomClientError.http(http.statusCode, Self.errorMessage(from: data))
        }
        struct Models: Decodable {
            let object: String
            let data: [Model]
            struct Model: Decodable { let id: String; let object: String }
        }
        guard let models = try? JSONDecoder().decode(Models.self, from: data),
              models.object == "list", models.data.count <= 4096,
              models.data.allSatisfy({ $0.object == "model" && !$0.id.isEmpty && $0.id.utf8.count <= 1024 })
        else { throw BloomClientError.incompatibleServer }
    }

    @MainActor
    func streamChat(
        model: String,
        messages: [APIChatMessage],
        onChunk: (ChatCompletionChunk) -> Void
    ) async throws {
        var request = URLRequest(url: endpoint("v1/chat/completions"))
        request.httpMethod = "POST"
        request.timeoutInterval = 300
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        applyHeaders(to: &request)
        request.httpBody = try JSONEncoder().encode(
            ChatCompletionRequest(
                model: model,
                messages: messages,
                maxTokens: 256,
                temperature: 0.7,
                topP: 0.9,
                stream: true,
                streamOptions: .init(includeUsage: true)
            )
        )
        guard messages.count <= 2048, request.httpBody!.count <= 1024 * 1024 else {
            throw BloomClientError.requestLimitExceeded
        }

        let (bytes, response) = try await session.bytes(for: request, delegate: Self.redirectPolicy)
        defer { bytes.task.cancel() }
        guard let http = response as? HTTPURLResponse else {
            throw BloomClientError.invalidHTTPResponse
        }
        guard http.statusCode == 200 else {
            var errorData = Data()
            for try await byte in bytes.prefix(64 * 1024) {
                errorData.append(byte)
            }
            throw BloomClientError.http(http.statusCode, Self.errorMessage(from: errorData))
        }
        guard http.mimeType?.lowercased() == "text/event-stream" else {
            throw BloomClientError.malformedStream
        }

        var receivedDone = false
        var parser = SSEParser()
        var validator = ChatStreamValidator(expectedModel: model)
        for try await byte in bytes {
            try Task.checkCancellation()
            try validator.observeByte()
            guard let event = try parser.append(byte) else {
                continue
            }
            switch event {
            case .done:
                try validator.finish()
                receivedDone = true
            case let .data(data):
                if let envelope = try? JSONDecoder().decode(APIErrorEnvelope.self, from: data) {
                    throw BloomClientError.streamError(String(envelope.error.message.prefix(4096)))
                }
                guard let chunk = try? JSONDecoder().decode(ChatCompletionChunk.self, from: data) else {
                    throw BloomClientError.malformedStream
                }
                try validator.observe(chunk)
                onChunk(chunk)
            }
            if receivedDone {
                break
            }
        }
        guard receivedDone else {
            throw BloomClientError.streamEndedWithoutDone
        }
    }

    private func endpoint(_ path: String) -> URL {
        serverURL.appendingPathComponent(path)
    }

    private func applyHeaders(to request: inout URLRequest) {
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedKey.isEmpty {
            request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")
        }
    }

    private static func errorMessage(from data: Data) -> String {
        if let envelope = try? JSONDecoder().decode(APIErrorEnvelope.self, from: data) {
            return String(envelope.error.message.prefix(4096))
        }
        if let text = String(data: data, encoding: .utf8), !text.isEmpty {
            return String(text.prefix(512))
        }
        return "No error details were returned."
    }
}

// Redirecting a configured endpoint can forward private prompts or credentials
// to a different service. Require users to configure the final endpoint instead.
private final class RejectRedirects: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}
