import SwiftUI

@main
struct BloomDesktopApp: App {
    @StateObject private var viewModel = ChatViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .frame(minWidth: 780, minHeight: 600)
        }
        .defaultSize(width: 1_020, height: 720)
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Conversation") {
                    viewModel.newConversation()
                }
                .keyboardShortcut("n", modifiers: .command)
                .disabled(viewModel.storageNeedsRecovery)
            }
        }
    }
}

private struct ContentView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        HStack(spacing: 0) {
            Sidebar(viewModel: viewModel)
                .frame(width: 270)
            Divider()
            ChatWorkspace(viewModel: viewModel)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .task {
            await viewModel.bootstrap()
        }
        .onChange(of: scenePhase) { phase in
            if phase != .active { viewModel.checkpoint() }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            viewModel.stop()
            viewModel.checkpoint()
        }
    }
}

private struct Sidebar: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var pendingDeletion: DesktopConversation?
    @State private var showsModels = false
    @State private var showsEmbeddings = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [.indigo, .purple],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                    Image(systemName: "sparkles")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .frame(width: 42, height: 42)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Bloom")
                        .font(.title2.weight(.semibold))
                    Text("Native Metal client")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                Label("Connection", systemImage: "point.3.connected.trianglepath.dotted")
                    .font(.headline)

                TextField("Server URL", text: $viewModel.serverURL)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Bloom server URL")

                SecureField("API key (optional)", text: $viewModel.apiKey)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Bloom API key")

                Button {
                    Task { await viewModel.connect() }
                } label: {
                    HStack {
                        if viewModel.connectionState == .connecting {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "arrow.clockwise")
                        }
                        Text("Connect")
                        Spacer()
                    }
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isGenerating || viewModel.connectionState == .connecting)
            }

            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 9, height: 9)
                    Text(viewModel.connectionState.label)
                        .font(.headline)
                }

                if let modelName = viewModel.modelName {
                    LabeledContent("Model") {
                        Text(modelName)
                            .lineLimit(2)
                            .multilineTextAlignment(.trailing)
                            .textSelection(.enabled)
                    }
                    .font(.caption)
                }
                if let contextWindow = viewModel.contextWindow {
                    LabeledContent("Context", value: "\(contextWindow) tokens")
                        .font(.caption)
                }
                HStack {
                    Button("Models…") { showsModels = true }
                    Button("Embeddings…") { showsEmbeddings = true }
                }
                if !viewModel.modelTasks.isEmpty {
                    Text(viewModel.modelTasks.joined(separator: ", ")).font(.caption2).foregroundStyle(.secondary)
                }
            }
            .padding(14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Conversations").font(.headline)
                    Spacer()
                    Button {
                        viewModel.newConversation()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                    .help("New conversation (⌘N)")
                    .accessibilityLabel("New conversation")
                    .disabled(viewModel.storageNeedsRecovery)
                }
                ScrollView {
                    LazyVStack(spacing: 5) {
                        ForEach(viewModel.conversations) { conversation in
                            Button {
                                viewModel.selectConversation(conversation.id)
                            } label: {
                                HStack(alignment: .top, spacing: 8) {
                                    Image(systemName: "bubble.left")
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(conversation.title).lineLimit(2)
                                        Text(conversation.updatedAt, style: .date)
                                            .font(.caption2).foregroundStyle(.secondary)
                                    }
                                    Spacer(minLength: 0)
                                    if conversation.id == viewModel.selectedConversationID {
                                        Image(systemName: "checkmark").font(.caption)
                                    }
                                }
                                .padding(9)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(conversation.id == viewModel.selectedConversationID
                                    ? Color.accentColor.opacity(0.15) : Color.clear,
                                    in: RoundedRectangle(cornerRadius: 8))
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Open \(conversation.title)")
                            .contextMenu {
                                Button("Delete conversation", role: .destructive) {
                                    pendingDeletion = conversation
                                }
                            }
                        }
                    }
                }
                .frame(minHeight: 80)
                if let conversation = viewModel.conversations.first(where: { $0.id == viewModel.selectedConversationID }) {
                    Button("Delete conversation", systemImage: "trash") {
                        pendingDeletion = conversation
                    }
                    .font(.caption)
                    .disabled(viewModel.storageNeedsRecovery)
                }
            }

            Text("History is saved on this Mac. API keys stay in memory. Switching conversations stops the current response.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .background(Color(nsColor: .underPageBackgroundColor))
        .sheet(isPresented: $showsModels) {
            DesktopModelsView(tools: viewModel.tools, chat: viewModel)
        }
        .sheet(isPresented: $showsEmbeddings) {
            DesktopEmbeddingsView(tools: viewModel.tools, chat: viewModel)
        }
        .alert(item: $pendingDeletion) { conversation in
            Alert(title: Text("Delete \(conversation.title)?"),
                  message: Text("This removes the conversation from this Mac. This cannot be undone."),
                  primaryButton: .destructive(Text("Delete")) {
                      viewModel.deleteConversation(conversation.id)
                  }, secondaryButton: .cancel())
        }
    }

    private var statusColor: Color {
        switch viewModel.connectionState {
        case .ready: return .green
        case .connecting, .loading: return .orange
        case .disconnected: return .secondary
        case .failed: return .red
        }
    }
}

private struct ChatWorkspace: View {
    @ObservedObject var viewModel: ChatViewModel
    @FocusState private var composerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(viewModel.conversationTitle)
                        .font(.title2.weight(.semibold))
                        .lineLimit(1)
                    Text(viewModel.modelName ?? "Connect to a Bloom server to begin")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Clear", systemImage: "trash") {
                    viewModel.clearConversation()
                }
                .disabled(viewModel.messages.isEmpty || viewModel.isGenerating)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 17)

            Divider()

            if let storageError = viewModel.storageError {
                VStack(alignment: .leading, spacing: 8) {
                    Label(storageError, systemImage: "externaldrive.badge.exclamationmark")
                        .font(.callout).textSelection(.enabled)
                    HStack {
                        Button(viewModel.storageNeedsRecovery ? "Retry loading history" : "Retry saving") {
                            viewModel.retryStorage()
                        }
                        if viewModel.storageNeedsRecovery {
                            Button("Back up damaged history and start fresh") {
                                viewModel.preserveDamagedHistoryAndReset()
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(Color.orange.opacity(0.12))
            }

            if viewModel.messages.isEmpty {
                EmptyConversation(state: viewModel.connectionState)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                MessageList(messages: viewModel.messages, isGenerating: viewModel.isGenerating)
            }

            if let errorMessage = viewModel.errorMessage {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(errorMessage)
                        .font(.callout)
                        .textSelection(.enabled)
                    Spacer()
                }
                .padding(12)
                .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal, 24)
            }

            if let metrics = viewModel.metrics {
                MetricsBar(metrics: metrics)
                    .padding(.horizontal, 24)
                    .padding(.top, 8)
            }

            Composer(viewModel: viewModel, focused: $composerFocused)
                .disabled(viewModel.storageNeedsRecovery)
                .padding(20)
        }
        .onChange(of: viewModel.connectionState) { state in
            if state == .ready {
                composerFocused = true
            }
        }
    }
}

private struct EmptyConversation: View {
    let state: ChatViewModel.ConnectionState

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: state == .ready ? "bubble.left.and.bubble.right.fill" : "bolt.horizontal.circle")
                .font(.system(size: 42, weight: .light))
                .foregroundStyle(.indigo)
            Text(state == .ready ? "Ready for a local conversation" : "Connect to the local Bloom server")
                .font(.title3.weight(.semibold))
            Text(state == .ready
                 ? "Responses stream directly from the native Metal backend."
                 : "The default endpoint is http://127.0.0.1:3000.")
                .foregroundStyle(.secondary)
        }
        .multilineTextAlignment(.center)
        .padding(40)
    }
}

private struct MessageList: View {
    let messages: [DesktopMessage]
    let isGenerating: Bool

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 14) {
                    ForEach(messages) { message in
                        MessageBubble(
                            message: message,
                            showsProgress: isGenerating
                                && message.role == .assistant
                                && message.id == messages.last?.id
                                && message.content.isEmpty
                        )
                        .id(message.id)
                    }
                }
                .padding(24)
            }
            .onChange(of: messages) { updated in
                guard let last = updated.last else { return }
                withAnimation(.easeOut(duration: 0.18)) {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }
}

private struct MessageBubble: View {
    let message: DesktopMessage
    let showsProgress: Bool

    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user {
                Spacer(minLength: 90)
            }

            VStack(alignment: .leading, spacing: 7) {
                Text(message.role == .user ? "You" : "Bloom")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                if showsProgress {
                    ProgressView()
                        .controlSize(.small)
                        .padding(.vertical, 4)
                } else {
                    Text(message.content)
                        .font(.body)
                        .textSelection(.enabled)
                }
            }
            .padding(14)
            .background(
                message.role == .user ? Color.accentColor.opacity(0.14) : Color(nsColor: .controlBackgroundColor),
                in: RoundedRectangle(cornerRadius: 14, style: .continuous)
            )

            if message.role == .assistant {
                Spacer(minLength: 90)
            }
        }
        .frame(maxWidth: .infinity)
    }
}

private struct MetricsBar: View {
    let metrics: GenerationMetrics

    var body: some View {
        HStack(spacing: 16) {
            Label(milliseconds(metrics.elapsedMilliseconds), systemImage: "clock")
            if let ttft = metrics.ttftMilliseconds {
                Text("TTFT \(milliseconds(ttft))")
            }
            if let prompt = metrics.promptTokens, let completion = metrics.completionTokens {
                Text("\(prompt) prompt · \(completion) output")
            }
            Spacer()
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
    }

    private func milliseconds(_ value: Double) -> String {
        value >= 1_000 ? String(format: "%.2f s", value / 1_000) : String(format: "%.0f ms", value)
    }
}

private struct Composer: View {
    @ObservedObject var viewModel: ChatViewModel
    let focused: FocusState<Bool>.Binding

    var body: some View {
        HStack(alignment: .bottom, spacing: 12) {
            ZStack(alignment: .topLeading) {
                if viewModel.draft.isEmpty {
                    Text("Message the local model…")
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 8)
                }
                TextEditor(text: $viewModel.draft)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .focused(focused)
                    .accessibilityLabel("Message")
            }
            .frame(minHeight: 58, maxHeight: 110)
            .padding(6)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.secondary.opacity(0.25))
            }

            if viewModel.isGenerating {
                Button("Stop", systemImage: "stop.fill") {
                    viewModel.stop()
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            } else {
                Button("Send", systemImage: "arrow.up") {
                    viewModel.send()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.canSend)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
    }
}
