import Foundation

private func expectEqual<T: Equatable>(_ actual: T, _ expected: T, _ message: String = "Values differ") {
    precondition(actual == expected, message)
}

private func expectTrue(_ condition: Bool, _ message: String = "Expected true") {
    precondition(condition, message)
}

private func expectFalse(_ condition: Bool, _ message: String = "Expected false") {
    precondition(!condition, message)
}

private func expectNil<T>(_ value: T?, _ message: String = "Expected nil") {
    precondition(value == nil, message)
}

private func expectThrows<T>(_ expression: @autoclosure () throws -> T, check: (Error) -> Void) {
    do {
        _ = try expression()
        fatalError("Expected an error")
    } catch {
        check(error)
    }
}

@MainActor
private final class AsyncSignal {
    private let name: String
    private var signalled = false
    private var pending: CheckedContinuation<Void, Error>?
    private var timeout: Task<Void, Never>?

    init(_ name: String) { self.name = name }

    func signal() {
        signalled = true
        timeout?.cancel()
        pending?.resume()
        pending = nil
    }

    func wait() async throws {
        if signalled { return }
        try await withCheckedThrowingContinuation { pending = $0
            timeout = Task {
                do { try await Task.sleep(nanoseconds: 2_000_000_000) } catch { return }
                pending?.resume(throwing: NSError(
                    domain: "BloomDesktopChecks", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Timed out: \(name)"]
                ))
                pending = nil
            }
        }
    }
}

@MainActor
func runRegressionChecks() async throws {
    let parser = SSEParserChecks()
    try parser.testDispatchesAtBlankLineAndCombinesMultilineData()
    try parser.testUnterminatedDataAndCompletionAreNotDispatched()
    try parser.testSupportsAllLineEndingsAndEmptyData()
    try parser.testUTF8IsPreservedAcrossIndividualBytes()
    parser.testRejectsMalformedUTF8()
    parser.testBoundsUnterminatedLinesAndAggregateEvents()
    try parser.testEventLimitResetsAfterDispatch()
    let lifecycle = ChatViewModelChecks()
    try await lifecycle.testStoppedGenerationCannotMutateReplacement(lateFailure: false)
    try await lifecycle.testStoppedGenerationCannotMutateReplacement(lateFailure: true)
    try await lifecycle.testConnectionEditsRequireNewReadinessAndIgnoreStaleResult()
    try await lifecycle.testCancelledConnectionCanBeRetried()
    try await lifecycle.testGenerationUsesConnectedClientSnapshot()
    try await checkHTTPTransport()
}

private struct SSEParserChecks {
    private func parse(_ input: [UInt8]) throws -> [SSEEvent] {
        var parser = SSEParser()
        return try input.compactMap { try parser.append($0) }
    }

    func testDispatchesAtBlankLineAndCombinesMultilineData() throws {
        let input = "\u{FEFF}: keepalive\r\nevent: message\r\ndata: {\r\ndata: \"choices\": []}\r\n\r\ndata: [DONE]\r\n\r\n"
        expectEqual(try parse(Array(input.utf8)), [
            .data(Data("{\n\"choices\": []}".utf8)), .done,
        ])
    }

    func testUnterminatedDataAndCompletionAreNotDispatched() throws {
        expectEqual(try parse(Array("data: [DONE]\n".utf8)), [])
        expectEqual(try parse(Array("data: {\"choices\":[]}".utf8)), [])
    }

    func testSupportsAllLineEndingsAndEmptyData() throws {
        for newline in ["\n", "\r", "\r\n"] {
            expectEqual(try parse(Array("data\(newline)\(newline)data: [DONE]\(newline)\(newline)".utf8)), [
                .data(Data()), .done,
            ])
        }
    }

    func testUTF8IsPreservedAcrossIndividualBytes() throws {
        expectEqual(try parse(Array("data: 中文 🌸\n\n".utf8)), [
            .data(Data("中文 🌸".utf8)),
        ])
    }

    func testRejectsMalformedUTF8() {
        expectThrows(try parse(Array("data: ".utf8) + [0xFF, 0x0A, 0x0A])) {
            guard case BloomClientError.malformedStream = $0 else {
                fatalError("Unexpected error: \($0)")
            }
        }
    }

    func testBoundsUnterminatedLinesAndAggregateEvents() {
        for input in [
            Array(repeating: UInt8(0x61), count: SSEParser.maximumEventBytes + 1),
            Array(String(repeating: "data: x\n", count: SSEParser.maximumEventBytes / 8 + 1).utf8),
            Array(String(repeating: ": x\n", count: SSEParser.maximumEventBytes / 4 + 1).utf8),
        ] {
            expectThrows(try parse(input)) {
                guard case BloomClientError.streamLimitExceeded = $0 else {
                    fatalError("Unexpected error: \($0)")
                }
            }
        }
    }

    func testEventLimitResetsAfterDispatch() throws {
        let event = "data: " + String(repeating: "a", count: 600_000) + "\n\n"
        expectEqual(try parse(Array((event + event).utf8)).count, 2)
    }
}


private let readinessFixture = """
{"schema_version":3,"object":"bloom.readiness","protocol_version":3,
"minimum_ui_protocol_version":3,"maximum_ui_protocol_version":3,
"server_version":"0.1.0","status":"ready","progress":100,"model":"tiny",
"loading":false,"load_error":null,"input_modalities":["text"],
"model_tasks":["generation"],"context_window":1024,"in_flight_requests":0,
"available_permits":4,"memory_pressure_high":false,"ram_utilization":0.1}
"""

@MainActor
private final class ControlledClient: BloomAPITransport {
    var delayReadiness = false
    var readinessContinuation: CheckedContinuation<BloomReadiness, Error>?
    var readinessStarted: (() -> Void)?
    var streamStarted: (() -> Void)?
    var streamFinished: (() -> Void)?
    var pendingStreams: [CheckedContinuation<[ChatCompletionChunk], Error>] = []
    var requests: [[APIChatMessage]] = []

    static func ready() throws -> BloomReadiness {
        try JSONDecoder().decode(BloomReadiness.self, from: Data(readinessFixture.utf8))
    }

    func readiness() async throws -> BloomReadiness {
        if delayReadiness {
            return try await withCheckedThrowingContinuation {
                readinessContinuation = $0
                readinessStarted?()
            }
        }
        return try Self.ready()
    }

    func streamChat(model: String, messages: [APIChatMessage], onChunk: (ChatCompletionChunk) -> Void) async throws {
        requests.append(messages)
        // Intentionally ignores cancellation to reproduce a late transport callback.
        defer { streamFinished?() }
        let chunks: [ChatCompletionChunk] = try await withCheckedThrowingContinuation {
            pendingStreams.append($0)
            streamStarted?()
        }
        for chunk in chunks { onChunk(chunk) }
    }
}

@MainActor
private struct ChatViewModelChecks {
    private let suite = "BloomDesktopChecks.\(UUID().uuidString)"

    @MainActor
    func testStoppedGenerationCannotMutateReplacement(lateFailure: Bool) async throws {
        let client = ControlledClient()
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let model = ChatViewModel(defaults: defaults) { _, _ in client }
        await model.connect()
        let firstStarted = AsyncSignal("first stream started")
        client.streamStarted = { firstStarted.signal() }
        model.draft = "first"
        model.send()
        try await firstStarted.wait()
        model.stop()
        expectEqual(model.messages.map(\.content), ["first"])

        let secondStarted = AsyncSignal("second stream started")
        client.streamStarted = { secondStarted.signal() }
        model.draft = "second"
        model.send()
        try await secondStarted.wait()
        expectEqual(client.requests[1].map(\.content), ["first", "second"])

        let firstFinished = AsyncSignal("cancelled stream finished late")
        client.streamFinished = { firstFinished.signal() }
        let staleChunk = try JSONDecoder().decode(ChatCompletionChunk.self, from: Data(
            #"{"choices":[{"delta":{"content":"stale"}}]}"#.utf8
        ))
        if lateFailure {
            client.pendingStreams[0].resume(throwing: URLError(.networkConnectionLost))
        } else {
            client.pendingStreams[0].resume(returning: [staleChunk])
        }
        try await firstFinished.wait()
        expectTrue(model.isGenerating, "Old cleanup must not unlock the replacement generation")
        expectNil(model.metrics, "Old success must not overwrite replacement metrics")
        expectNil(model.errorMessage)
        expectFalse(model.messages.contains(where: { $0.content.contains("stale") }))

        let secondFinished = AsyncSignal("second stream cancelled")
        client.streamFinished = { secondFinished.signal() }
        model.stop()
        client.pendingStreams[1].resume(throwing: URLError(.cancelled))
        try await secondFinished.wait()
        expectFalse(model.isGenerating)
        expectNil(model.errorMessage, "A user cancellation should not become an error banner")
    }

    @MainActor
    func testConnectionEditsRequireNewReadinessAndIgnoreStaleResult() async throws {
        let client = ControlledClient()
        client.delayReadiness = true
        let started = AsyncSignal("readiness started")
        client.readinessStarted = { started.signal() }
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let model = ChatViewModel(defaults: defaults) { _, _ in client }
        let connecting = Task { await model.connect() }
        try await started.wait()
        model.serverURL = "http://localhost:4567"
        client.readinessContinuation?.resume(returning: try ControlledClient.ready())
        await connecting.value
        expectEqual(model.connectionState, .disconnected)
        expectNil(model.modelName)
        model.draft = "private prompt"
        expectFalse(model.canSend)
        model.send()
        expectTrue(model.messages.isEmpty)

        client.delayReadiness = false
        await model.connect()
        expectTrue(model.canSend)
        model.apiKey = "new credential"
        expectEqual(model.connectionState, .disconnected)
        expectFalse(model.canSend)
    }

    @MainActor
    func testGenerationUsesConnectedClientSnapshot() async throws {
        let client = ControlledClient()
        var constructions = 0
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let model = ChatViewModel(defaults: defaults) { _, _ in
            constructions += 1
            return client
        }
        await model.connect()
        let started = AsyncSignal("stream started")
        client.streamStarted = { started.signal() }
        model.draft = "hello"
        model.send()
        model.serverURL = "https://different.invalid"
        try await started.wait()
        expectEqual(constructions, 1, "Sending must reuse the client whose readiness was verified")
        model.stop()
        let finished = AsyncSignal("stream finished")
        client.streamFinished = { finished.signal() }
        client.pendingStreams[0].resume(returning: [])
        try await finished.wait()
    }

    func testCancelledConnectionCanBeRetried() async throws {
        let client = ControlledClient()
        client.delayReadiness = true
        let started = AsyncSignal("cancellable readiness started")
        client.readinessStarted = { started.signal() }
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let model = ChatViewModel(defaults: defaults) { _, _ in client }
        let connecting = Task { await model.connect() }
        try await started.wait()
        connecting.cancel()
        client.readinessContinuation?.resume(returning: try ControlledClient.ready())
        await connecting.value
        expectEqual(model.connectionState, .disconnected)
        expectNil(model.errorMessage)
        client.delayReadiness = false
        await model.connect()
        expectEqual(model.connectionState, .ready)
    }
}

// Exercise the real URLSession byte reader without a network service. The
// oversized fixture deliberately never reaches EOF; the client must reject it
// while reading instead of waiting to check the size after body completion.
private final class FixtureProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let scenario = request.url!.pathComponents[1]
        let contentType = scenario == "wrong-type" ? "application/json" :
            (request.url!.lastPathComponent == "ready" ? "application/json" : "text/event-stream; charset=utf-8")
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": contentType]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body: Data
        switch scenario {
        case "oversized": body = Data(repeating: 0x20, count: 64 * 1024 + 1)
        case "ready": body = Data(readinessFixture.utf8)
        case "multiline":
            body = Data("data: {\r\ndata: \"choices\":[{\"delta\":{\"content\":\"中文\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n".utf8)
        case "truncated": body = Data("data: [DONE]\n".utf8)
        default: body = Data("data: [DONE]\n\n".utf8)
        }
        client?.urlProtocol(self, didLoad: body)
        if scenario != "oversized" { client?.urlProtocolDidFinishLoading(self) }
    }

    override func stopLoading() {}
}

@MainActor
private func checkHTTPTransport() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [FixtureProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    func client(_ scenario: String) throws -> BloomAPIClient {
        try BloomAPIClient(serverURL: "http://bloom-checks.invalid/\(scenario)", apiKey: "", session: session)
    }

    let ready = try await client("ready").readiness()
    expectTrue(ready.supportsChat)
    var contents: [String] = []
    try await client("multiline").streamChat(model: "tiny", messages: []) {
        contents.append($0.choices.first?.delta.content ?? "")
    }
    expectEqual(contents, ["中文"])

    do {
        try await client("wrong-type").streamChat(model: "tiny", messages: []) { _ in
            fatalError("Wrong media type must never dispatch a chunk")
        }
        fatalError("Wrong media type must be rejected")
    } catch BloomClientError.malformedStream {}

    do {
        try await client("truncated").streamChat(model: "tiny", messages: []) { _ in }
        fatalError("An unfinished completion event must fail")
    } catch BloomClientError.streamEndedWithoutDone {}

    let rejected = AsyncSignal("oversized readiness rejected before EOF")
    let boundedRead = Task {
        do {
            _ = try await client("oversized").readiness()
            fatalError("Oversized readiness must be rejected")
        } catch BloomClientError.incompatibleServer {
            rejected.signal()
        } catch {
            fatalError("Unexpected readiness failure: \(error)")
        }
    }
    defer { boundedRead.cancel() }
    try await rejected.wait()
    await boundedRead.value
}
