import Foundation
import Combine

struct DesktopMessage: Identifiable, Equatable {
    enum Role: String {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    var content: String

    init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }
}

struct GenerationMetrics: Equatable {
    let promptTokens: Int?
    let completionTokens: Int?
    let ttftMilliseconds: Double?
    let elapsedMilliseconds: Double
}

@MainActor
final class ChatViewModel: ObservableObject {
    enum ConnectionState: Equatable {
        case disconnected
        case connecting
        case loading(Int)
        case ready
        case failed

        var label: String {
            switch self {
            case .disconnected: return "Disconnected"
            case .connecting: return "Connecting"
            case let .loading(progress): return "Loading \(progress)%"
            case .ready: return "Ready"
            case .failed: return "Unavailable"
            }
        }
    }

    @Published var serverURL: String {
        didSet { if serverURL != oldValue { invalidateConnection() } }
    }
    @Published var apiKey = "" {
        didSet { if apiKey != oldValue { invalidateConnection() } }
    }
    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var modelName: String?
    @Published private(set) var contextWindow: UInt64?
    @Published private(set) var messages: [DesktopMessage] = []
    @Published var draft = ""
    @Published private(set) var isGenerating = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var metrics: GenerationMetrics?

    private var generationTask: Task<Void, Never>?
    private var activeGenerationID: UUID?
    private var connectionID = UUID()
    private var connectedClient: (any BloomAPITransport)?
    private let makeClient: (String, String) throws -> any BloomAPITransport
    private let defaults: UserDefaults
    private var automationStarted = false

    init(defaults: UserDefaults = .standard, makeClient: @escaping (String, String) throws -> any BloomAPITransport = {
        try BloomAPIClient(serverURL: $0, apiKey: $1)
    }) {
        self.defaults = defaults
        serverURL = defaults.string(forKey: "bloom.serverURL") ?? "http://127.0.0.1:3000"
        self.makeClient = makeClient
    }

    var canSend: Bool {
        connectionState == .ready
            && !isGenerating
            && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func bootstrap() async {
        await connect()
        guard
            !automationStarted,
            connectionState == .ready,
            let prompt = ProcessInfo.processInfo.environment["BLOOM_DESKTOP_AUTOMATION_PROMPT"],
            !prompt.isEmpty
        else {
            return
        }
        automationStarted = true
        draft = prompt
        send()
    }

    func connect() async {
        guard !isGenerating else { return }
        let attemptID = UUID()
        connectionID = attemptID
        connectedClient = nil
        let requestedURL = serverURL
        connectionState = .connecting
        errorMessage = nil
        modelName = nil
        contextWindow = nil

        do {
            let client = try makeClient(requestedURL, apiKey)
            let readiness = try await client.readiness()
            guard connectionID == attemptID else { return }
            try Task.checkCancellation()
            defaults.set(requestedURL, forKey: "bloom.serverURL")
            modelName = readiness.model
            contextWindow = readiness.contextWindow

            if readiness.supportsChat {
                connectedClient = client
                connectionState = .ready
            } else if readiness.loading {
                connectionState = .loading(readiness.progress)
                errorMessage = "The model is still loading. Connect again when it reaches 100%."
            } else {
                connectionState = .failed
                errorMessage = readiness.loadError
                    ?? "The server is reachable, but no generation model is ready."
            }
        } catch {
            guard connectionID == attemptID else { return }
            if Task.isCancelled || error is CancellationError {
                invalidateConnection()
                return
            }
            connectionState = .failed
            errorMessage = Self.describe(error)
        }
    }

    func send() {
        let prompt = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, !prompt.isEmpty, let modelName, let client = connectedClient else { return }

        errorMessage = nil
        metrics = nil
        draft = ""
        messages.append(DesktopMessage(role: .user, content: prompt))
        let requestMessages = messages.map {
            APIChatMessage(role: $0.role.rawValue, content: $0.content)
        }
        let assistantID = UUID()
        messages.append(DesktopMessage(id: assistantID, role: .assistant, content: ""))
        activeGenerationID = assistantID
        isGenerating = true

        generationTask = Task { [weak self] in
            guard let self else { return }
            await self.performGeneration(
                client: client,
                model: modelName,
                requestMessages: requestMessages,
                assistantID: assistantID
            )
        }
    }

    func stop() {
        let assistantID = activeGenerationID
        activeGenerationID = nil
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        if let assistantID { removeEmptyAssistant(assistantID) }
    }

    func clearConversation() {
        guard !isGenerating else { return }
        messages.removeAll()
        metrics = nil
        errorMessage = nil
    }

    private func performGeneration(
        client: any BloomAPITransport,
        model: String,
        requestMessages: [APIChatMessage],
        assistantID: UUID
    ) async {
        let started = Date()
        var firstTokenAt: Date?
        var usage: ChatCompletionChunk.Usage?

        do {
            try Task.checkCancellation()
            try await client.streamChat(model: model, messages: requestMessages) { [weak self] chunk in
                guard let self, self.activeGenerationID == assistantID, !Task.isCancelled else { return }
                usage = chunk.usage ?? usage
                let content = chunk.choices.first?.delta.content ?? ""
                guard !content.isEmpty else { return }
                if firstTokenAt == nil {
                    firstTokenAt = Date()
                }
                if let index = self.messages.firstIndex(where: { $0.id == assistantID }) {
                    self.messages[index].content.append(content)
                }
            }

            guard activeGenerationID == assistantID, !Task.isCancelled else { return }
            let finished = Date()
            metrics = GenerationMetrics(
                promptTokens: usage?.promptTokens,
                completionTokens: usage?.completionTokens,
                ttftMilliseconds: firstTokenAt.map { $0.timeIntervalSince(started) * 1_000 },
                elapsedMilliseconds: finished.timeIntervalSince(started) * 1_000
            )
        } catch {
            guard activeGenerationID == assistantID else { return }
            if !Task.isCancelled && !(error is CancellationError) {
                errorMessage = Self.describe(error)
            }
            removeEmptyAssistant(assistantID)
        }

        guard activeGenerationID == assistantID else { return }
        activeGenerationID = nil
        isGenerating = false
        generationTask = nil
    }

    private func removeEmptyAssistant(_ id: UUID) {
        if let index = messages.firstIndex(where: { $0.id == id }), messages[index].content.isEmpty {
            messages.remove(at: index)
        }
    }

    private func invalidateConnection() {
        connectionID = UUID()
        connectedClient = nil
        modelName = nil
        contextWindow = nil
        connectionState = .disconnected
    }

    private static func describe(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }
}
