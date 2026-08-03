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
                    viewModel.clearConversation()
                }
                .keyboardShortcut("n", modifiers: .command)
                .disabled(viewModel.isGenerating)
            }
        }
    }
}

private struct ContentView: View {
    @ObservedObject var viewModel: ChatViewModel

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
    }
}

private struct Sidebar: View {
    @ObservedObject var viewModel: ChatViewModel

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
            }
            .padding(14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))

            Spacer()

            Text("Messages go directly to the configured local Bloom API. No browser or embedded web UI is used.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .background(Color(nsColor: .underPageBackgroundColor))
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
                    Text("Local conversation")
                        .font(.title2.weight(.semibold))
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
