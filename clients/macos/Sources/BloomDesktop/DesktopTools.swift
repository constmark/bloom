import Foundation
import Combine

struct DesktopCatalog: Decodable {
    struct Entry: Decodable, Identifiable {
        let id: String
        let name: String
        let kind: String
        let format: String
        let size_bytes: UInt64
        let size_complete: Bool
        let active: Bool
    }
    struct Active: Decodable {
        let id: String
        let catalog_id: String?
        let source: String
        enum CodingKeys: String, CodingKey { case id, catalog_id, source }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            catalog_id = try c.decode(String?.self, forKey: .catalog_id)
            source = try c.decode(String.self, forKey: .source)
        }
    }
    struct Load: Decodable {
        let phase: String
        let progress: Int
        let requested_model: String?
        let error: String?
        enum CodingKeys: String, CodingKey { case phase, progress, requested_model, error }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            phase = try c.decode(String.self, forKey: .phase)
            progress = try c.decode(Int.self, forKey: .progress)
            requested_model = try c.decode(String?.self, forKey: .requested_model)
            error = try c.decode(String?.self, forKey: .error)
        }
    }
    let schema_version: Int
    let object: String
    let root: String
    let root_exists: Bool
    let data: [Entry]
    let active_model: Active?
    let load: Load
    enum CodingKeys: String, CodingKey { case schema_version, object, root, root_exists, data, active_model, load }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schema_version = try c.decode(Int.self, forKey: .schema_version)
        object = try c.decode(String.self, forKey: .object)
        root = try c.decode(String.self, forKey: .root)
        root_exists = try c.decode(Bool.self, forKey: .root_exists)
        data = try c.decode([Entry].self, forKey: .data)
        active_model = try c.decode(Active?.self, forKey: .active_model)
        load = try c.decode(Load.self, forKey: .load)
    }

    static func validID(_ id: String) -> Bool {
        !id.isEmpty && id.utf8.count <= 255 && id != "." && id != ".."
            && !id.contains("/") && !id.contains("\\")
            && !id.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
            && id.trimmingCharacters(in: .whitespacesAndNewlines) == id
    }

    func validate() throws {
        guard schema_version == 1, object == "bloom.model_catalog", root.utf8.count <= 16_384,
              data.count <= 4096, (root_exists || data.isEmpty), Set(data.map(\.id)).count == data.count,
              data.filter(\.active).count <= 1,
              ["idle", "loading", "ready", "error"].contains(load.phase),
              (0...100).contains(load.progress), (load.error?.utf8.count ?? 0) <= 4096,
              (load.requested_model?.utf8.count ?? 0) <= 1024,
              load.phase != "loading" || (load.requested_model != nil && load.error == nil),
              load.phase != "error" || load.error?.isEmpty == false,
              load.phase != "ready" || (active_model != nil && load.progress == 100 && load.error == nil) else {
            throw BloomClientError.serverNotReady("The server returned an invalid model catalog.")
        }
        for item in data {
            guard Self.validID(item.id), !item.name.isEmpty, item.name.utf8.count <= 1024,
                  ["file", "directory"].contains(item.kind),
                  ["gguf", "onnx", "coreml", "tensorrt", "bloom", "transformers", "openvino"].contains(item.format)
            else { throw BloomClientError.incompatibleServer }
        }
        if let active = active_model {
            guard !active.id.isEmpty, active.id.utf8.count <= 1024,
                  ["catalog", "external"].contains(active.source) else { throw BloomClientError.incompatibleServer }
            if active.source == "catalog" {
                guard let id = active.catalog_id, Self.validID(id),
                      data.contains(where: { $0.id == id && $0.active }) else { throw BloomClientError.incompatibleServer }
            } else if active.catalog_id != nil { throw BloomClientError.incompatibleServer }
        } else if data.contains(where: \.active) || load.phase == "ready" {
            throw BloomClientError.incompatibleServer
        }
    }
}

struct DesktopEmbeddingResult: Codable {
    struct Vector: Codable, Identifiable {
        var id: Int { index }
        let object: String
        let index: Int
        let embedding: [Double]
    }
    struct Usage: Codable {
        let prompt_tokens: UInt64?
        let total_tokens: UInt64?
    }
    let object: String
    let model: String
    let data: [Vector]
    let usage: Usage

    func validate(model expected: String, count: Int, dimensions: Int?) throws {
        guard object == "list", model == expected, data.count == count,
              let width = data.first?.embedding.count, (1...16_384).contains(width),
              dimensions == nil || width == dimensions,
              data.count * width <= 1_048_576,
              data.enumerated().allSatisfy({ index, row in
                  row.object == "embedding" && row.index == index && row.embedding.count == width
                      && row.embedding.allSatisfy(\.isFinite)
              }) else { throw BloomClientError.serverNotReady("The server returned invalid embedding vectors.") }
    }

    func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }
}

@MainActor
final class DesktopToolsViewModel: ObservableObject {
    @Published private(set) var catalog: DesktopCatalog?
    @Published private(set) var isManaging = false
    @Published private(set) var managementError: String?
    @Published private(set) var managementNotice: String?
    @Published var embeddingInput = ""
    @Published var embeddingDimensions = ""
    @Published private(set) var isEmbedding = false
    @Published private(set) var embeddingError: String?
    @Published private(set) var embeddingResult: DesktopEmbeddingResult?
    @Published private(set) var embeddingInputs: [String] = []

    var onRuntimeWillChange: (() -> Void)?
    var onRuntimeDidChange: (() async -> Void)?
    private let makeClient: (String, String) throws -> any BloomAPITransport
    private var serverURL = ""
    private var apiKey = ""
    private var operatorKey = ""
    private var epoch = UUID()
    private var managementID: UUID?
    private var embeddingID: UUID?
    private var managementTask: Task<Void, Never>?
    private var embeddingTask: Task<Void, Never>?
    private let pollNanoseconds: UInt64

    init(pollNanoseconds: UInt64 = 750_000_000,
         makeClient: @escaping (String, String) throws -> any BloomAPITransport) {
        self.pollNanoseconds = pollNanoseconds
        self.makeClient = makeClient
    }

    func configure(serverURL: String, apiKey: String, operatorKey: String) {
        guard self.serverURL != serverURL || self.apiKey != apiKey || self.operatorKey != operatorKey else { return }
        epoch = UUID()
        stopMonitoring()
        cancelEmbedding()
        self.serverURL = serverURL
        self.apiKey = apiKey
        self.operatorKey = operatorKey
        catalog = nil
        managementError = nil
        managementNotice = nil
        embeddingError = nil
        embeddingResult = nil
        embeddingInputs = []
    }

    func refreshCatalog() { manage(loadID: nil, unload: false) }
    func loadModel(_ id: String) {
        guard catalog?.data.contains(where: { $0.id == id }) == true else { return }
        manage(loadID: id, unload: false)
    }
    func unloadModel() {
        guard catalog?.active_model != nil else { return }
        manage(loadID: nil, unload: true)
    }

    func stopMonitoring() {
        managementID = nil
        managementTask?.cancel()
        managementTask = nil
        isManaging = false
        managementNotice = "Monitoring stopped. A load already accepted by the server continues; refresh to check it."
    }

    private func manage(loadID: String?, unload: Bool) {
        guard !isManaging else { return }
        let operation = UUID()
        let currentEpoch = epoch
        managementID = operation
        isManaging = true
        managementError = nil
        managementNotice = nil
        let mutating = loadID != nil || unload
        if mutating {
            cancelEmbedding()
            embeddingResult = nil
            onRuntimeWillChange?()
        }
        let key = operatorKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? apiKey : operatorKey
        let requestedURL = serverURL
        managementTask = Task { [weak self] in
            guard let self else { return }
            defer {
                if managementID == operation { isManaging = false; managementID = nil; managementTask = nil }
            }
            do {
                let client = try makeClient(requestedURL, key)
                try Task.checkCancellation()
                if let loadID { try await client.loadCatalogModel(loadID) }
                if unload { try await client.unloadCatalogModel() }
                // Poll only the documented catalog lifecycle; there is no
                // server-side load-cancellation endpoint.
                for _ in 0..<400 {
                    try Task.checkCancellation()
                    let snapshot = try await client.modelCatalog()
                    guard epoch == currentEpoch, managementID == operation else { return }
                    try snapshot.validate()
                    catalog = snapshot
                    if snapshot.load.phase != "loading" {
                        if snapshot.load.phase == "error" {
                            managementError = snapshot.load.error ?? "Model loading failed."
                        } else if let loadID, snapshot.active_model?.catalog_id != loadID {
                            managementError = "The requested model is not active. Refresh and retry loading."
                        } else if unload, snapshot.active_model != nil {
                            managementError = "The model is still active. Refresh and retry unloading."
                        } else { managementNotice = mutating ? "Model state updated." : nil }
                        await onRuntimeDidChange?()
                        return
                    }
                    try await Task.sleep(nanoseconds: pollNanoseconds)
                }
                throw BloomClientError.serverNotReady("Model loading is still in progress. Refresh to continue monitoring.")
            } catch {
                guard epoch == currentEpoch, managementID == operation else { return }
                if !(error is CancellationError) && !Task.isCancelled {
                    managementError = error.localizedDescription
                }
            }
        }
    }

    func generateEmbeddings(model: String) {
        guard !isEmbedding, !isManaging else { return }
        let inputs = embeddingInput.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        let rawDimensions = embeddingDimensions.trimmingCharacters(in: .whitespacesAndNewlines)
        let dimensions = rawDimensions.isEmpty ? nil : Int(rawDimensions)
        guard (1...256).contains(inputs.count), inputs.allSatisfy({ $0.unicodeScalars.count <= 262_144 }),
              inputs.reduce(0, { $0 + $1.utf8.count }) <= 768 * 1024,
              rawDimensions.isEmpty || dimensions.map({ (1...16_384).contains($0) }) == true else {
            embeddingError = "Enter 1–256 nonblank lines (at most 768 KiB total) and dimensions from 1–16384, or leave dimensions empty."
            return
        }
        let operation = UUID()
        let currentEpoch = epoch
        let requestedURL = serverURL
        let key = apiKey
        embeddingID = operation
        isEmbedding = true
        embeddingError = nil
        embeddingResult = nil
        embeddingTask = Task { [weak self] in
            guard let self else { return }
            defer {
                if embeddingID == operation { isEmbedding = false; embeddingID = nil; embeddingTask = nil }
            }
            do {
                let client = try makeClient(requestedURL, key)
                let result = try await client.embeddings(model: model, inputs: inputs, dimensions: dimensions)
                guard epoch == currentEpoch, embeddingID == operation else { return }
                try Task.checkCancellation()
                try result.validate(model: model, count: inputs.count, dimensions: dimensions)
                embeddingInputs = inputs
                embeddingResult = result
            } catch {
                guard epoch == currentEpoch, embeddingID == operation else { return }
                if !(error is CancellationError) && !Task.isCancelled { embeddingError = error.localizedDescription }
            }
        }
    }

    func cancelEmbedding() {
        embeddingID = nil
        embeddingTask?.cancel()
        embeddingTask = nil
        isEmbedding = false
    }
}
