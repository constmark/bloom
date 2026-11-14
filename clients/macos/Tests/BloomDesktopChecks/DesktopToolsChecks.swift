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

private func rerankBytes() -> Data {
    Data(#"{"id":"rerank-1","object":"rerank","model":"tiny","results":[{"index":1,"relevance_score":0.8,"document":{"text":"second"}},{"index":0,"relevance_score":0.2,"document":{"text":"first"}}],"usage":{"prompt_tokens":9,"total_tokens":9}}"#.utf8)
}

private func rerankFixture() throws -> DesktopRerankResult {
    try JSONDecoder().decode(DesktopRerankResult.self, from: rerankBytes())
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
    var rerankContinuations: [CheckedContinuation<DesktopRerankResult, Error>] = []
    var rerankRequests: [(String, String, [String], Int?)] = []
    var streamContinuation: CheckedContinuation<Void, Never>?
    var encoderReady = false
    var rerankOnlyReady = false

    func readiness() async throws -> BloomReadiness {
        let tasks = rerankOnlyReady ? "rerank" : (encoderReady ? "embedding" : "generation")
        return try JSONDecoder().decode(BloomReadiness.self, from: Data("""
        {"schema_version":3,"object":"bloom.readiness","protocol_version":3,
        "minimum_ui_protocol_version":3,"maximum_ui_protocol_version":3,"server_version":"0.1.0",
        "status":"ready","progress":100,"model":"tiny","loading":false,"load_error":null,
        "input_modalities":["text"],"model_tasks":["\(tasks)"],"context_window":1024,
        "in_flight_requests":0,"available_permits":4,"memory_pressure_high":false,"ram_utilization":0.1}
        """.utf8))
    }
    func streamChat(
        model: String, messages: [APIChatMessage], maxTokens: Int, onChunk: (ChatCompletionChunk) -> Void
    ) async throws {
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
    func rerank(model: String, query: String, documents: [String], topN: Int?) async throws -> DesktopRerankResult {
        rerankRequests.append((model, query, documents, topN))
        return try await withCheckedThrowingContinuation { rerankContinuations.append($0) }
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
    ], usage: .init(prompt_tokens: 0, total_tokens: 0))
    do { try nonfinite.validate(model: "tiny", count: 1, dimensions: nil); fatalError("Nonfinite vector") }
    catch BloomClientError.serverNotReady(_) {}
    for vector in [[0.0, 0.0], [1.0, 1.0], [Double.greatestFiniteMagnitude, 0.0]] {
        let invalid = DesktopEmbeddingResult(object: "list", model: "tiny", data: [
            .init(object: "embedding", index: 0, embedding: vector)
        ], usage: .init(prompt_tokens: 3, total_tokens: 3))
        do { try invalid.validate(model: "tiny", count: 1, dimensions: nil); fatalError("Unnormalized vector") }
        catch BloomClientError.serverNotReady(_) {}
    }
    let badUsage = DesktopEmbeddingResult(object: "list", model: "tiny", data: try embeddingFixture().data,
                                         usage: .init(prompt_tokens: 3, total_tokens: 4))
    do { try badUsage.validate(model: "tiny", count: 1, dimensions: nil); fatalError("Bad embedding usage") }
    catch BloomClientError.serverNotReady(_) {}
    let missingUsage = Data(#"{"object":"list","model":"tiny","data":[{"object":"embedding","index":0,"embedding":[0.6,0.8]}],"usage":{}}"#.utf8)
    do { _ = try JSONDecoder().decode(DesktopEmbeddingResult.self, from: missingUsage); fatalError("Missing embedding usage") }
    catch is DecodingError {}

    try checkRerankValidation()

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

    tools.rerankQuery = "find relevant text"
    tools.rerankDocuments = " first\n\n second "
    tools.rerank(model: "tiny")
    try await toolsWait { client.rerankContinuations.count == 1 }
    precondition(keys.last == "new-key" && client.rerankRequests[0].2 == ["first", "second"])
    tools.cancelReranking()
    tools.rerankQuery = "replacement query"
    tools.rerank(model: "tiny")
    try await toolsWait { client.rerankContinuations.count == 2 }
    client.rerankContinuations[0].resume(returning: try rerankFixture())
    try await Task.sleep(nanoseconds: 5_000_000)
    precondition(tools.isReranking && tools.rerankResult == nil)
    client.rerankContinuations[1].resume(returning: try rerankFixture())
    try await toolsWait { !tools.isReranking }
    precondition(tools.rerankResultQuery == "replacement query" && tools.rerankResult?.results.first?.index == 1)
    tools.rerankQuery = "edited after completion"
    let rerankExport = try tools.rerankResult!.jsonData(query: tools.rerankResultQuery)
    let exportedObject = try JSONSerialization.jsonObject(with: rerankExport) as! [String: Any]
    precondition(exportedObject["query"] as? String == "replacement query")
    precondition(exportedObject["object"] as? String == "bloom.rerank_result")
    precondition((exportedObject["results"] as? [[String: Any]])?.first?["document"] as? String == "second")
    precondition(!String(decoding: rerankExport, as: UTF8.self).contains("new-key"))
    tools.rerankTopN = "3"
    tools.rerank(model: "tiny")
    precondition(!tools.isReranking && tools.rerankError != nil && client.rerankRequests.count == 2)
    tools.rerankTopN = "invalid"
    tools.rerank(model: "tiny")
    precondition(!tools.isReranking && client.rerankRequests.count == 2)
    tools.rerankTopN = ""
    tools.rerank(model: "tiny")
    try await toolsWait { client.rerankContinuations.count == 3 }
    tools.configure(serverURL: "http://next.invalid", apiKey: "another-key", operatorKey: "")
    client.rerankContinuations[2].resume(returning: try rerankFixture())
    try await Task.sleep(nanoseconds: 5_000_000)
    precondition(!tools.isReranking && tools.rerankResult == nil && tools.embeddingResult == nil)

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

    // A remote operator can change the model. Reconnecting must clear old
    // results and cancel pending encoder work before admitting the new runtime.
    chat.tools.embeddingInput = "hello"
    chat.tools.generateEmbeddings(model: "tiny")
    try await toolsWait { chatClient.embeddingContinuations.count == 1 }
    chatClient.embeddingContinuations[0].resume(returning: try embeddingFixture())
    try await toolsWait { !chat.tools.isEmbedding }
    precondition(chat.tools.embeddingResult != nil)
    chat.tools.rerankQuery = "query"
    chat.tools.rerankDocuments = "first\nsecond"
    chat.tools.rerank(model: "tiny")
    try await toolsWait { chatClient.rerankContinuations.count == 1 }
    chatClient.rerankOnlyReady = true
    await chat.connect()
    precondition(chat.tools.embeddingResult == nil && !chat.tools.isReranking)
    precondition(chat.connectionState == .ready && chat.modelTasks == ["rerank"] && !chat.canSend)
    chatClient.rerankContinuations[0].resume(returning: try rerankFixture())
    try await Task.sleep(nanoseconds: 5_000_000)
    precondition(chat.tools.rerankResult == nil)

    try await checkToolsTransport()
    print("Bloom Desktop model-management, embedding and reranking checks passed")
}

private func checkRerankValidation() throws {
    try rerankFixture().validate(model: "tiny", documents: ["first", "second"], topN: nil)
    for scenario in ["model", "id", "count", "duplicate", "order", "tie", "document", "usage"] {
        var payload = try JSONSerialization.jsonObject(with: rerankBytes()) as! [String: Any]
        var rows = payload["results"] as! [[String: Any]]
        switch scenario {
        case "model": payload["model"] = "other"
        case "id": payload["id"] = "bad id"
        case "count": rows.removeLast()
        case "duplicate": rows[1]["index"] = 1; rows[1]["document"] = ["text": "second"]
        case "order": rows.reverse()
        case "tie": rows[1]["relevance_score"] = 0.8
        case "document": rows[0]["document"] = ["text": "unsubmitted"]
        case "usage": payload["usage"] = ["prompt_tokens": 9, "total_tokens": 10]
        default: fatalError("Unknown scenario")
        }
        payload["results"] = rows
        let decoded = try JSONDecoder().decode(DesktopRerankResult.self, from: JSONSerialization.data(withJSONObject: payload))
        do { try decoded.validate(model: "tiny", documents: ["first", "second"], topN: nil); fatalError("Invalid rerank accepted: \(scenario)") }
        catch BloomClientError.serverNotReady(_) {}
    }
    let fixture = try rerankFixture()
    let cross = DesktopRerankResult(id: fixture.id, object: fixture.object, model: fixture.model,
        results: [.init(index: 1, relevance_score: 9.25, document: .init(text: "second")),
                  .init(index: 0, relevance_score: -4.5, document: .init(text: "first"))], usage: fixture.usage)
    try cross.validate(model: "tiny", documents: ["first", "second"], topN: nil)
    let exportedCross = try JSONSerialization.jsonObject(with: cross.jsonData(query: "query")) as! [String: Any]
    let exportedScores = (exportedCross["results"] as! [[String: Any]]).map { $0["relevance_score"] as! Double }
    precondition(exportedScores == [9.25, -4.5])
    let top = DesktopRerankResult(id: fixture.id, object: fixture.object, model: fixture.model,
                                results: [fixture.results[0]], usage: fixture.usage)
    try top.validate(model: "tiny", documents: ["first", "second"], topN: 1)
    let nonfinite = DesktopRerankResult(id: fixture.id, object: fixture.object, model: fixture.model,
                                      results: [.init(index: 0, relevance_score: .infinity, document: .init(text: "first"))], usage: fixture.usage)
    do { try nonfinite.validate(model: "tiny", documents: ["first"], topN: nil); fatalError("Nonfinite reranking score") }
    catch BloomClientError.serverNotReady(_) {}
    for documents in [[], [""], Array(repeating: "x", count: 257), [String(repeating: "a", count: 262_145)], Array(repeating: String(repeating: "a", count: 262_144), count: 3)] {
        do { try DesktopRerankResult.validateInput(model: "tiny", query: "query", documents: documents, topN: nil); fatalError("Bad rerank input") }
        catch BloomClientError.serverNotReady(_) {}
    }
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
        var recorded = request
        if recorded.httpBody == nil, let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var bytes = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                guard count > 0 else { break }
                bytes.append(contentsOf: buffer.prefix(count))
            }
            recorded.httpBody = bytes
        }
        toolsRequests.append(recorded)
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
        else if path.hasSuffix("rerank") { body = rerankBytes() }
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
    let ranked = try await client.rerank(model: "tiny", query: "query", documents: ["first", "second"], topN: 2)
    precondition(ranked.results.first?.index == 1)
    precondition(toolsRequests.last()?.url?.path == "/v1/rerank" && toolsRequests.last()?.httpMethod == "POST")
    precondition(toolsRequests.last()?.value(forHTTPHeaderField: "Authorization") == "Bearer operator-secret")
    let sent = try JSONSerialization.jsonObject(with: toolsRequests.last()!.httpBody!) as! [String: Any]
    precondition(sent["return_documents"] as? Bool == true && sent["top_n"] as? Int == 2)
    precondition(sent["query"] as? String == "query" && sent["documents"] as? [String] == ["first", "second"])
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
