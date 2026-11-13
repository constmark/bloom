import Foundation

private func catalogFixture(_ phase: String = "ready", active: Bool = true) throws -> DesktopCatalog {
    try JSONDecoder().decode(DesktopCatalog.self, from: catalogBytes(phase, active: active))
}

private func catalogBytes(_ phase: String = "ready", active: Bool = true) throws -> Data {
    try JSONSerialization.data(withJSONObject: [
        "schema_version": 1, "object": "bloom.model_catalog", "root": "/server/models", "root_exists": true,
        "data": [["id": "encoder.gguf", "name": "Encoder", "kind": "file", "format": "gguf",
                  "size_bytes": 123, "size_complete": true, "active": active]],
        "active_model": active ? ["id": "tiny", "catalog_id": "encoder.gguf", "source": "catalog"] as Any : NSNull(),
        "load": ["phase": phase, "progress": phase == "ready" ? 100 : 12,
                 "requested_model": phase == "loading" ? "encoder.gguf" as Any : NSNull(),
                 "error": phase == "error" ? "fixture load failure" as Any : NSNull()]
    ])
}

private func embeddingFixture(model: String = "tiny", index: Int = 0) throws -> DesktopEmbeddingResult {
    try JSONDecoder().decode(DesktopEmbeddingResult.self, from: Data("""
    {"object":"list","model":"\(model)","data":[{"object":"embedding","index":\(index),"embedding":[0.6,0.8]}],"usage":{"prompt_tokens":3,"total_tokens":3}}
    """.utf8))
}

@MainActor
private func toolsWait(_ condition: () -> Bool) async throws {
    let end = Date().addingTimeInterval(3)
    while !condition() {
        guard Date() < end else { throw NSError(domain: "DesktopToolsChecks timeout", code: 1) }
        try await Task.sleep(nanoseconds: 1_000_000)
    }
}

@MainActor
private final class ToolsClient: BloomAPITransport {
    var catalogs: [DesktopCatalog] = []
    var loadedIDs: [String] = []
    var unloaded = false
    var delayCatalog = false
    var catalogContinuation: CheckedContinuation<DesktopCatalog, Error>?
    var embeddingContinuations: [CheckedContinuation<DesktopEmbeddingResult, Error>] = []
    var embeddingRequests: [(String, [String], Int?)] = []
    var streamContinuation: CheckedContinuation<Void, Never>?
    var encoderReady = false

    func readiness() async throws -> BloomReadiness {
        let tasks = encoderReady ? "embedding" : "generation"
        return try JSONDecoder().decode(BloomReadiness.self, from: Data("""
        {"schema_version":3,"object":"bloom.readiness","protocol_version":3,
        "minimum_ui_protocol_version":3,"maximum_ui_protocol_version":3,"server_version":"0.1.0",
        "status":"ready","progress":100,"model":"tiny","loading":false,"load_error":null,
        "input_modalities":["text"],"model_tasks":["\(tasks)"],"context_window":1024,
        "in_flight_requests":0,"available_permits":4,"memory_pressure_high":false,"ram_utilization":0.1}
        """.utf8))
    }
    func streamChat(model: String, messages: [APIChatMessage], onChunk: (ChatCompletionChunk) -> Void) async throws {
        await withCheckedContinuation { streamContinuation = $0 }
    }
    func modelCatalog() async throws -> DesktopCatalog {
        if delayCatalog { return try await withCheckedThrowingContinuation { catalogContinuation = $0 } }
        return catalogs.count > 1 ? catalogs.removeFirst() : catalogs[0]
    }
    func loadCatalogModel(_ id: String) async throws { loadedIDs.append(id); encoderReady = true }
    func unloadCatalogModel() async throws { unloaded = true }
    func embeddings(model: String, inputs: [String], dimensions: Int?) async throws -> DesktopEmbeddingResult {
        embeddingRequests.append((model, inputs, dimensions))
        return try await withCheckedThrowingContinuation { embeddingContinuations.append($0) }
    }
}

@MainActor
func runDesktopToolsChecks() async throws {
    let fixture = try catalogFixture()
    try fixture.validate()
    for id in ["../secret", "a/b", "a\\b", ".", "..", " name", "x\n", ""] {
        precondition(!DesktopCatalog.validID(id))
    }
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    for filename in ["model-catalog.json", "model-catalog-empty.json"] {
        let data = try Data(contentsOf: root.appendingPathComponent("examples/\(filename)"))
        try JSONDecoder().decode(DesktopCatalog.self, from: data).validate()
    }
    var malformed = try JSONSerialization.jsonObject(with: catalogBytes()) as! [String: Any]
    malformed.removeValue(forKey: "active_model")
    do {
        _ = try JSONDecoder().decode(DesktopCatalog.self, from: JSONSerialization.data(withJSONObject: malformed))
        fatalError("Missing catalog state must not decode as empty")
    } catch is DecodingError {}
    do { try embeddingFixture(index: 1).validate(model: "tiny", count: 1, dimensions: 2); fatalError("Bad embedding index") }
    catch BloomClientError.serverNotReady(_) {}
    do { try embeddingFixture().validate(model: "different", count: 1, dimensions: nil); fatalError("Model mismatch") }
    catch BloomClientError.serverNotReady(_) {}
    do { try embeddingFixture().validate(model: "tiny", count: 1, dimensions: 3); fatalError("Width mismatch") }
    catch BloomClientError.serverNotReady(_) {}
    let nonfinite = DesktopEmbeddingResult(object: "list", model: "tiny", data: [
        .init(object: "embedding", index: 0, embedding: [.infinity])
    ], usage: .init(prompt_tokens: nil, total_tokens: nil))
    do { try nonfinite.validate(model: "tiny", count: 1, dimensions: nil); fatalError("Nonfinite vector") }
    catch BloomClientError.serverNotReady(_) {}

    let client = ToolsClient()
    client.catalogs = [fixture]
    var keys: [String] = []
    let tools = DesktopToolsViewModel(pollNanoseconds: 1_000_000) { _, key in keys.append(key); return client }
    tools.configure(serverURL: "http://localhost:3000", apiKey: "inference-secret", operatorKey: "operator-secret")
    tools.refreshCatalog()
    try await toolsWait { !tools.isManaging }
    precondition(tools.catalog?.data.count == 1 && keys.last == "operator-secret")
    var willChange = 0
    var didChange = 0
    tools.onRuntimeWillChange = { willChange += 1 }
    tools.onRuntimeDidChange = { didChange += 1 }
    client.catalogs = [try catalogFixture("loading", active: false), fixture]
    tools.loadModel("encoder.gguf")
    try await toolsWait { !tools.isManaging }
    precondition(client.loadedIDs == ["encoder.gguf"] && willChange == 1 && didChange == 1)
    precondition(tools.catalog?.load.phase == "ready" && tools.managementError == nil)
    client.catalogs = [try catalogFixture("error", active: false)]
    tools.loadModel("encoder.gguf")
    try await toolsWait { !tools.isManaging }
    precondition(tools.managementError == "fixture load failure")
    client.catalogs = [fixture]
    tools.loadModel("encoder.gguf")
    try await toolsWait { !tools.isManaging }
    precondition(tools.managementError == nil && client.loadedIDs.count == 3)
    client.catalogs = [try catalogFixture("idle", active: false)]
    tools.unloadModel()
    try await toolsWait { !tools.isManaging }
    precondition(client.unloaded && tools.catalog?.active_model == nil)

    client.delayCatalog = true
    tools.refreshCatalog()
    try await toolsWait { client.catalogContinuation != nil }
    tools.stopMonitoring()
    tools.configure(serverURL: "http://changed.invalid", apiKey: "new-key", operatorKey: "")
    client.catalogContinuation?.resume(returning: fixture)
    client.catalogContinuation = nil
    try await Task.sleep(nanoseconds: 5_000_000)
    precondition(tools.catalog == nil && !tools.isManaging && tools.managementError == nil)
    client.delayCatalog = false
    client.catalogs = [fixture]
    tools.refreshCatalog()
    try await toolsWait { !tools.isManaging }
    precondition(keys.last == "new-key", "Empty operator key should use inference key")

    tools.embeddingInput = "hello"
    tools.embeddingDimensions = "2"
    tools.generateEmbeddings(model: "tiny")
    try await toolsWait { client.embeddingContinuations.count == 1 }
    precondition(keys.last == "new-key")
    tools.cancelEmbedding()
    tools.embeddingInput = "replacement"
    tools.generateEmbeddings(model: "tiny")
    try await toolsWait { client.embeddingContinuations.count == 2 }
    client.embeddingContinuations[0].resume(returning: try embeddingFixture())
    try await Task.sleep(nanoseconds: 5_000_000)
    precondition(tools.isEmbedding && tools.embeddingResult == nil)
    client.embeddingContinuations[1].resume(returning: try embeddingFixture())
    try await toolsWait { !tools.isEmbedding }
    precondition(tools.embeddingInputs == ["replacement"] && tools.embeddingResult?.data[0].embedding == [0.6, 0.8])
    let exported = try tools.embeddingResult!.jsonData()
    let exportedText = String(decoding: exported, as: UTF8.self)
    precondition(!exportedText.contains("new-key") && !exportedText.contains("operator-secret"))
    tools.embeddingDimensions = "invalid"
    tools.generateEmbeddings(model: "tiny")
    precondition(!tools.isEmbedding && tools.embeddingError != nil && client.embeddingRequests.count == 2)

    // Model mutation stops chat and invalidates readiness before any network
    // mutation; after polling it reconnects and enables only the new task.
    let suite = "BloomDesktopToolsChecks.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    let chatClient = ToolsClient()
    chatClient.catalogs = [fixture]
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("BloomDesktopToolsChecks-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: temporary) }
    let chat = ChatViewModel(defaults: defaults, storage: FileConversationStore(url: temporary.appendingPathComponent("history.json"))) { _, _ in chatClient }
    chat.tools.refreshCatalog()
    try await toolsWait { !chat.tools.isManaging }
    await chat.connect()
    chat.draft = "old request"
    chat.send()
    try await toolsWait { chatClient.streamContinuation != nil }
    chat.tools.loadModel("encoder.gguf")
    precondition(!chat.isGenerating && chat.connectionState == .disconnected)
    try await toolsWait { !chat.tools.isManaging }
    precondition(chat.modelTasks == ["embedding"] && chat.connectionState == .ready && !chat.canSend)
    chatClient.streamContinuation?.resume()

    try await checkToolsTransport()
    print("Bloom Desktop model-management and embedding checks passed")
}

private final class ToolsRequestLog {
    private let lock = NSLock()
    private var values: [URLRequest] = []
    func append(_ request: URLRequest) { lock.lock(); defer { lock.unlock() }; values.append(request) }
    func last() -> URLRequest? { lock.lock(); defer { lock.unlock() }; return values.last }
}
private let toolsRequests = ToolsRequestLog()

private final class ToolsFixtureProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        toolsRequests.append(request)
        let path = request.url!.path
        let unauthorized = path.contains("unauthorized")
        let oversizedError = path.contains("oversized-error")
        let status = unauthorized ? 403 : (oversizedError ? 500 : (path.hasSuffix("switch") ? 202 : 200))
        let type = path.contains("wrong-type") ? "text/html" : "application/json"
        var headers = ["Content-Type": type]
        if path.contains("oversized") && !oversizedError { headers["Content-Length"] = String(16 * 1024 * 1024 + 1) }
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: headers)!, cacheStoragePolicy: .notAllowed)
        let body: Data
        if unauthorized { body = Data(#"{"error":{"message":"operator required"}}"#.utf8) }
        else if oversizedError { body = Data(repeating: 32, count: 64 * 1024 + 1) }
        else if path.contains("oversized") { body = Data(repeating: 32, count: 16 * 1024 * 1024 + 1) }
        else if path.hasSuffix("switch") { body = Data(#"{"object":"bloom.model_load","accepted":true,"model":"encoder.gguf","sequence":1}"#.utf8) }
        else if path.hasSuffix("unload") { body = Data(#"{"object":"bloom.model_unload","unloaded":true}"#.utf8) }
        else if path.hasSuffix("embeddings") { body = try! embeddingFixture().jsonData() }
        else { body = try! catalogBytes() }
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@MainActor
private func checkToolsTransport() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [ToolsFixtureProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let client = try BloomAPIClient(serverURL: "http://tools.invalid", apiKey: "operator-secret", session: session)
    _ = try await client.modelCatalog()
    precondition(toolsRequests.last()?.value(forHTTPHeaderField: "Authorization") == "Bearer operator-secret")
    try await client.loadCatalogModel("encoder.gguf")
    precondition(toolsRequests.last()?.url?.path == "/v1/model-management/switch" && toolsRequests.last()?.httpMethod == "POST")
    try await client.unloadCatalogModel()
    precondition(toolsRequests.last()?.url?.path == "/v1/model-management/unload")
    let result = try await client.embeddings(model: "tiny", inputs: ["hello"], dimensions: 2)
    precondition(result.data[0].embedding == [0.6, 0.8])
    // One grapheme can contain many Unicode scalars; match the server's
    // character limit rather than accepting an unbounded combining sequence.
    let combining = "a" + String(repeating: "\u{0301}", count: 262_144)
    do { _ = try await client.embeddings(model: "tiny", inputs: [combining], dimensions: nil); fatalError("Scalar limit bypass") }
    catch BloomClientError.serverNotReady(_) {}
    for scenario in ["unauthorized", "wrong-type", "oversized", "oversized-error"] {
        do {
            _ = try await BloomAPIClient(serverURL: "http://tools.invalid/\(scenario)", apiKey: "key", session: session).modelCatalog()
            fatalError("Invalid tools response accepted: \(scenario)")
        } catch BloomClientError.http(let code, _) where scenario == "unauthorized" && code == 403 {
        } catch BloomClientError.invalidHTTPResponse where scenario == "wrong-type" {
        } catch BloomClientError.responseLimitExceeded where scenario.hasPrefix("oversized") {}
    }
    do { try await client.loadCatalogModel("../secret"); fatalError("Invalid selector accepted") }
    catch BloomClientError.serverNotReady(_) {}
}
