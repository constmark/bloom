import Foundation

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

    @Published var serverURL = UserDefaults.standard.string(forKey: "bloom.serverURL")
        ?? "http://127.0.0.1:3000"
    @Published var apiKey = ""
    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var modelName: String?
    @Published private(set) var contextWindow: Int?
    @Published private(set) var messages: [DesktopMessage] = []
    @Published var draft = ""
    @Published private(set) var isGenerating = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var metrics: GenerationMetrics?

    private var generationTask: Task<Void, Never>?
    private var automationStarted = false

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
        connectionState = .connecting
        errorMessage = nil
        modelName = nil
        contextWindow = nil

        do {
            let client = try BloomAPIClient(serverURL: serverURL, apiKey: apiKey)
            let readiness = try await client.readiness()
            UserDefaults.standard.set(serverURL, forKey: "bloom.serverURL")
            modelName = readiness.model
            contextWindow = readiness.contextWindow

            if readiness.supportsChat {
                connectionState = .ready
            } else if readiness.status == "loading" {
                connectionState = .loading(readiness.progress)
                errorMessage = "The model is still loading. Connect again when it reaches 100%."
            } else {
                connectionState = .failed
                errorMessage = readiness.loadError
                    ?? "The server is reachable, but no generation model is ready."
            }
        } catch {
            connectionState = .failed
            errorMessage = Self.describe(error)
        }
    }

    func send() {
        let prompt = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend, !prompt.isEmpty, let modelName else { return }

        errorMessage = nil
        metrics = nil
        draft = ""
        messages.append(DesktopMessage(role: .user, content: prompt))
        let requestMessages = messages.map {
            APIChatMessage(role: $0.role.rawValue, content: $0.content)
        }
        let assistantID = UUID()
        messages.append(DesktopMessage(id: assistantID, role: .assistant, content: ""))
        isGenerating = true

        generationTask = Task { [weak self] in
            guard let self else { return }
            await self.performGeneration(
                model: modelName,
                requestMessages: requestMessages,
                assistantID: assistantID
            )
        }
    }

    func stop() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    func clearConversation() {
        guard !isGenerating else { return }
        messages.removeAll()
        metrics = nil
        errorMessage = nil
    }

    private func performGeneration(
        model: String,
        requestMessages: [APIChatMessage],
        assistantID: UUID
    ) async {
        let started = Date()
        var firstTokenAt: Date?
        var usage: ChatCompletionChunk.Usage?

        do {
            let client = try BloomAPIClient(serverURL: serverURL, apiKey: apiKey)
            try await client.streamChat(model: model, messages: requestMessages) { [weak self] chunk in
                guard let self else { return }
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

            let finished = Date()
            metrics = GenerationMetrics(
                promptTokens: usage?.promptTokens,
                completionTokens: usage?.completionTokens,
                ttftMilliseconds: firstTokenAt.map { $0.timeIntervalSince(started) * 1_000 },
                elapsedMilliseconds: finished.timeIntervalSince(started) * 1_000
            )
        } catch is CancellationError {
            if let index = messages.firstIndex(where: { $0.id == assistantID }),
               messages[index].content.isEmpty
            {
                messages.remove(at: index)
            }
        } catch {
            errorMessage = Self.describe(error)
            if let index = messages.firstIndex(where: { $0.id == assistantID }),
               messages[index].content.isEmpty
            {
                messages.remove(at: index)
            }
        }

        isGenerating = false
        generationTask = nil
    }

    private static func describe(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }
}
