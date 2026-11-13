import Foundation
import Combine

struct DesktopMessage: Identifiable, Equatable, Codable {
    enum Role: String, Codable {
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
    @Published var operatorKey = "" {
        didSet { configureTools() }
    }
    let tools: DesktopToolsViewModel
    @Published private(set) var modelTasks: [String] = []
    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var modelName: String?
    @Published private(set) var contextWindow: UInt64?
    @Published private(set) var messages: [DesktopMessage] = [] {
        didSet { conversationChanged() }
    }
    @Published var draft = "" {
        didSet { conversationChanged() }
    }
    @Published private(set) var conversations: [DesktopConversation] = []
    @Published private(set) var selectedConversationID: UUID?
    @Published private(set) var storageError: String?
    @Published private(set) var storageNeedsRecovery = false
    @Published private(set) var isGenerating = false
    @Published private(set) var errorMessage: String?
    @Published private(set) var metrics: GenerationMetrics?

    private var generationTask: Task<Void, Never>?
    private var activeGenerationID: UUID?
    private var connectionID = UUID()
    private var connectedClient: (any BloomAPITransport)?
    private let makeClient: (String, String) throws -> any BloomAPITransport
    private let defaults: UserDefaults
    private let storage: any ConversationPersistence
    private var restoringConversation = false
    private var checkpointTask: Task<Void, Never>?
    private var automationStarted = false

    init(defaults: UserDefaults = .standard, storage: any ConversationPersistence = FileConversationStore.standard,
         makeClient: @escaping (String, String) throws -> any BloomAPITransport = {
        try BloomAPIClient(serverURL: $0, apiKey: $1)
    }) {
        self.defaults = defaults
        self.storage = storage
        serverURL = defaults.string(forKey: "bloom.serverURL") ?? "http://127.0.0.1:3000"
        self.makeClient = makeClient
        tools = DesktopToolsViewModel(makeClient: makeClient)
        restoreHistory()
        configureTools()
        tools.onRuntimeWillChange = { [weak self] in
            self?.stop()
            self?.clearConnection()
        }
        tools.onRuntimeDidChange = { [weak self] in await self?.connect() }
    }

    var canSend: Bool {
        connectionState == .ready
            && modelTasks.contains("generation")
            && !isGenerating
            && !storageNeedsRecovery
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
            modelTasks = readiness.modelTasks

            if readiness.status == "ready" {
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
        guard canSend, !prompt.isEmpty, let modelName, let client = connectedClient,
              let conversationID = selectedConversationID else { return }
        guard prompt.utf8.count <= ConversationArchive.maxTextBytes,
              messages.count + 2 <= ConversationArchive.maxMessages else {
            errorMessage = "This conversation has reached its message or text limit. Start a new conversation."
            return
        }

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
        checkpoint()

        generationTask = Task { [weak self] in
            guard let self else { return }
            await self.performGeneration(
                client: client,
                model: modelName,
                requestMessages: requestMessages,
                assistantID: assistantID,
                conversationID: conversationID
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
        checkpoint()
    }

    func clearConversation() {
        guard !isGenerating, !storageNeedsRecovery else { return }
        messages.removeAll()
        metrics = nil
        errorMessage = nil
        checkpoint()
    }

    var conversationTitle: String {
        conversations.first(where: { $0.id == selectedConversationID })?.title ?? "Conversation history"
    }

    func newConversation() {
        guard !storageNeedsRecovery else { return }
        guard conversations.count < ConversationArchive.maxConversations else {
            storageError = "The 128-conversation limit has been reached. Delete an older conversation first."
            return
        }
        stop()
        let conversation = DesktopConversation()
        conversations.insert(conversation, at: 0)
        display(conversation)
        checkpoint()
    }

    func selectConversation(_ id: UUID) {
        guard !storageNeedsRecovery, id != selectedConversationID,
              let conversation = conversations.first(where: { $0.id == id }) else { return }
        stop()
        display(conversation)
        checkpoint()
    }

    func deleteConversation(_ id: UUID) {
        guard !storageNeedsRecovery, conversations.contains(where: { $0.id == id }) else { return }
        if selectedConversationID == id { stop() }
        conversations.removeAll(where: { $0.id == id })
        if conversations.isEmpty { conversations = [DesktopConversation()] }
        if selectedConversationID == id { display(conversations[0]) }
        checkpoint()
    }

    /// Flush immediately at lifecycle boundaries, or retry a failed save.
    func checkpoint() {
        checkpointTask?.cancel()
        checkpointTask = nil
        guard !storageNeedsRecovery, let selectedConversationID else { return }
        do {
            try storage.save(ConversationArchive(selectedID: selectedConversationID, conversations: conversations))
            storageError = nil
        } catch {
            storageError = "History could not be saved. Your current conversations remain in memory. \(Self.describe(error))"
        }
    }

    func retryStorage() {
        if storageNeedsRecovery { restoreHistory() } else { checkpoint() }
    }

    func preserveDamagedHistoryAndReset() {
        guard storageNeedsRecovery else { return }
        do {
            try storage.quarantine()
            storageNeedsRecovery = false
            let archive = ConversationArchive.empty()
            conversations = archive.conversations
            display(conversations[0])
            checkpoint()
        } catch { storageError = Self.describe(error) }
    }

    private func restoreHistory() {
        do {
            let archive = try storage.load() ?? .empty()
            try archive.validate()
            conversations = archive.conversations
            storageNeedsRecovery = false
            storageError = nil
            display(conversations.first(where: { $0.id == archive.selectedID })!)
        } catch {
            storageNeedsRecovery = true
            storageError = "History could not be loaded. The original file is preserved; writing is paused. \(Self.describe(error))"
        }
    }

    private func display(_ conversation: DesktopConversation) {
        restoringConversation = true
        selectedConversationID = conversation.id
        messages = conversation.messages
        draft = conversation.draft
        metrics = nil
        errorMessage = nil
        restoringConversation = false
    }

    private func conversationChanged() {
        guard !restoringConversation, !storageNeedsRecovery,
              let index = conversations.firstIndex(where: { $0.id == selectedConversationID }) else { return }
        conversations[index].messages = messages.filter { $0.role != .assistant || !$0.content.isEmpty }
        conversations[index].draft = draft
        conversations[index].updatedAt = Date()
        // Throttle checkpoints rather than debouncing: long streams still
        // preserve partial output every 500 ms, without writing for every token.
        guard checkpointTask == nil else { return }
        checkpointTask = Task { [weak self] in
            do { try await Task.sleep(nanoseconds: 500_000_000) } catch { return }
            self?.checkpoint()
        }
    }

    private func performGeneration(
        client: any BloomAPITransport,
        model: String,
        requestMessages: [APIChatMessage],
        assistantID: UUID,
        conversationID: UUID
    ) async {
        let started = Date()
        var firstTokenAt: Date?
        var usage: ChatCompletionChunk.Usage?

        do {
            try Task.checkCancellation()
            try await client.streamChat(model: model, messages: requestMessages) { [weak self] chunk in
                guard let self, self.activeGenerationID == assistantID,
                      self.selectedConversationID == conversationID, !Task.isCancelled else { return }
                usage = chunk.usage ?? usage
                let content = chunk.choices.first?.delta.content ?? ""
                guard !content.isEmpty else { return }
                if firstTokenAt == nil {
                    firstTokenAt = Date()
                }
                if let index = self.messages.firstIndex(where: { $0.id == assistantID }) {
                    guard self.messages[index].content.utf8.count + content.utf8.count <= ConversationArchive.maxTextBytes else {
                        self.errorMessage = "The response reached the 1 MiB message limit and was stopped."
                        self.stop()
                        return
                    }
                    self.messages[index].content.append(content)
                }
            }

            guard activeGenerationID == assistantID, selectedConversationID == conversationID,
                  !Task.isCancelled else { return }
            let finished = Date()
            metrics = GenerationMetrics(
                promptTokens: usage?.promptTokens,
                completionTokens: usage?.completionTokens,
                ttftMilliseconds: firstTokenAt.map { $0.timeIntervalSince(started) * 1_000 },
                elapsedMilliseconds: finished.timeIntervalSince(started) * 1_000
            )
        } catch {
            guard activeGenerationID == assistantID, selectedConversationID == conversationID else { return }
            if !Task.isCancelled && !(error is CancellationError) {
                errorMessage = Self.describe(error)
            }
            removeEmptyAssistant(assistantID)
        }

        guard activeGenerationID == assistantID, selectedConversationID == conversationID else { return }
        activeGenerationID = nil
        isGenerating = false
        generationTask = nil
        checkpoint()
    }

    private func removeEmptyAssistant(_ id: UUID) {
        if let index = messages.firstIndex(where: { $0.id == id }), messages[index].content.isEmpty {
            messages.remove(at: index)
        }
    }

    private func invalidateConnection() {
        configureTools()
        clearConnection()
    }

    private func configureTools() {
        tools.configure(serverURL: serverURL, apiKey: apiKey, operatorKey: operatorKey)
    }

    private func clearConnection() {
        connectionID = UUID()
        connectedClient = nil
        modelName = nil
        contextWindow = nil
        modelTasks = []
        connectionState = .disconnected
    }

    private static func describe(_ error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }
}
