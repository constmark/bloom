import Foundation

struct BloomAPIClient {
    let serverURL: URL
    let apiKey: String

    init(serverURL: String, apiKey: String) throws {
        let trimmed = serverURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            let url = URL(string: trimmed),
            let scheme = url.scheme?.lowercased(),
            scheme == "http" || scheme == "https",
            url.host != nil
        else {
            throw BloomClientError.invalidServerURL
        }
        self.serverURL = url
        self.apiKey = apiKey
    }

    @MainActor
    func readiness() async throws -> BloomReadiness {
        var request = URLRequest(url: endpoint("ready"))
        request.timeoutInterval = 10
        request.cachePolicy = .reloadIgnoringLocalCacheData
        applyHeaders(to: &request)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw BloomClientError.invalidHTTPResponse
        }
        guard (200 ... 599).contains(http.statusCode) else {
            throw BloomClientError.invalidHTTPResponse
        }
        guard http.statusCode == 200 || http.statusCode == 503 else {
            throw BloomClientError.http(http.statusCode, Self.errorMessage(from: data))
        }

        let readiness = try JSONDecoder().decode(BloomReadiness.self, from: data)
        guard readiness.schemaVersion == 3, readiness.object == "bloom.readiness" else {
            throw BloomClientError.incompatibleServer
        }
        return readiness
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

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
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

        var receivedDone = false
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard let event = SSEParser.parse(line: line) else {
                continue
            }
            switch event {
            case .done:
                receivedDone = true
            case let .data(data):
                if let envelope = try? JSONDecoder().decode(APIErrorEnvelope.self, from: data) {
                    throw BloomClientError.http(200, envelope.error.message)
                }
                guard let chunk = try? JSONDecoder().decode(ChatCompletionChunk.self, from: data) else {
                    throw BloomClientError.malformedStream
                }
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
            return envelope.error.message
        }
        if let text = String(data: data, encoding: .utf8), !text.isEmpty {
            return String(text.prefix(512))
        }
        return "No error details were returned."
    }
}
