import SwiftUI
import UniformTypeIdentifiers

struct DesktopModelsView: View {
    @ObservedObject var tools: DesktopToolsViewModel
    @ObservedObject var chat: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Local models").font(.title2.bold())
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            SecureField("Operator API key (optional; otherwise uses API key)", text: $chat.operatorKey)
                .textFieldStyle(.roundedBorder)
            Text("Models are discovered in the server's configured directory. Loading or unloading stops your active response. Files are kept on disk.")
                .font(.callout).foregroundStyle(.secondary)
            HStack {
                Button("Refresh") { tools.refreshCatalog() }.disabled(tools.isManaging)
                if tools.isManaging {
                    ProgressView().controlSize(.small)
                    Button("Stop monitoring") { tools.stopMonitoring() }
                }
                Spacer()
                Button("Unload active model") { tools.unloadModel() }
                    .disabled(tools.isManaging || tools.catalog?.active_model == nil || tools.catalog?.load.phase == "loading")
            }
            if let catalog = tools.catalog {
                Text(catalog.root).font(.caption).textSelection(.enabled)
                if !catalog.root_exists {
                    Text("This server directory does not exist yet. Add supported model files on the server, then refresh.")
                        .foregroundStyle(.secondary)
                }
                HStack {
                    Text("State: \(catalog.load.phase)")
                    if catalog.load.phase == "loading" {
                        ProgressView(value: Double(catalog.load.progress), total: 100)
                        Text("\(catalog.load.progress)%")
                    }
                    if let active = catalog.active_model {
                        Text("Active: \(active.id)").lineLimit(1)
                    }
                }
                if let requested = catalog.load.requested_model, catalog.load.phase == "loading" {
                    Text("Loading \(requested). Stopping monitoring does not cancel the server load.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if catalog.data.isEmpty {
                    Text("No supported local models found. Place a model in the server directory above and refresh.")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(catalog.data) { model in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(model.name).font(.headline)
                                Text("\(model.id) · \(model.format) · \(ByteCountFormatter.string(fromByteCount: Int64(clamping: model.size_bytes), countStyle: .file))\(model.size_complete ? "" : " (partial estimate)")")
                                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                            }
                            Spacer()
                            if model.active { Label("Active", systemImage: "checkmark.circle.fill").foregroundStyle(.green) }
                            Button(model.active ? "Load / retry" : "Load") { tools.loadModel(model.id) }
                                .disabled(tools.isManaging || catalog.load.phase == "loading")
                        }
                        .padding(.vertical, 4)
                    }
                }
            } else {
                Text("Refresh to discover models with your operator credentials.")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            if let error = tools.managementError {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).textSelection(.enabled)
            }
            if let notice = tools.managementNotice { Text(notice).font(.caption).foregroundStyle(.secondary) }
        }
        .padding(22)
        .frame(minWidth: 640, minHeight: 500)
        .task { if tools.catalog == nil && !tools.isManaging { tools.refreshCatalog() } }
    }
}

struct DesktopEmbeddingsView: View {
    @ObservedObject var tools: DesktopToolsViewModel
    @ObservedObject var chat: ChatViewModel
    @State private var exportError: String?
    @Environment(\.dismiss) private var dismiss

    private var available: Bool {
        chat.connectionState == .ready && chat.modelTasks.contains("embedding") && !tools.isManaging && !tools.isReranking
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Sentence embeddings").font(.title2.bold())
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Text(chat.modelName ?? "No active model").font(.headline)
            if !available {
                Text("Load an embedding model from Local models, then Connect. Chat-only and rerank-only models cannot create sentence vectors.")
                    .foregroundStyle(.secondary)
            }
            Text("One input per line · 1–256 inputs · at most 768 KiB total").font(.caption)
            TextEditor(text: $tools.embeddingInput)
                .font(.body).frame(minHeight: 120, maxHeight: 180)
                .border(Color.secondary.opacity(0.3))
                .disabled(tools.isEmbedding)
                .accessibilityLabel("Embedding input texts")
            HStack {
                Text("Dimensions")
                TextField("Native width", text: $tools.embeddingDimensions)
                    .frame(width: 140).textFieldStyle(.roundedBorder).disabled(tools.isEmbedding)
                Spacer()
                if tools.isEmbedding {
                    ProgressView().controlSize(.small)
                    Button("Cancel") { tools.cancelEmbedding() }
                } else {
                    Button("Generate vectors") {
                        if let model = chat.modelName { tools.generateEmbeddings(model: model) }
                    }
                    .disabled(!available)
                }
            }
            if let result = tools.embeddingResult {
                HStack {
                    Text("\(result.data.count) vectors · \(result.data.first?.embedding.count ?? 0) dimensions · tokens: \(result.usage.prompt_tokens)")
                        .font(.caption)
                    Spacer()
                    Button("Copy JSON") {
                        do {
                            let text = String(decoding: try result.jsonData(), as: UTF8.self)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(text, forType: .string)
                            exportError = nil
                        } catch { exportError = error.localizedDescription }
                    }
                    Button("Save JSON…") { export(result) }
                }
                List(result.data) { row in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(row.index + 1). \(tools.embeddingInputs.indices.contains(row.index) ? tools.embeddingInputs[row.index] : "")")
                            .lineLimit(2)
                        Text(row.embedding.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", ") + (row.embedding.count > 8 ? ", …" : ""))
                            .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    }
                }
            } else { Spacer() }
            if let error = tools.embeddingError ?? exportError {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).textSelection(.enabled)
            }
            Text("Inputs, vectors and credentials stay out of conversation history. Cancel interrupts the client request; the server releases inference resources when its current native operation returns.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(22)
        .frame(minWidth: 700, minHeight: 610)
    }

    private func export(_ result: DesktopEmbeddingResult) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "bloom-embeddings.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try result.jsonData().write(to: url, options: .atomic); exportError = nil }
        catch { exportError = error.localizedDescription }
    }
}

struct DesktopRerankView: View {
    @ObservedObject var tools: DesktopToolsViewModel
    @ObservedObject var chat: ChatViewModel
    @State private var exportError: String?
    @Environment(\.dismiss) private var dismiss

    private var available: Bool {
        chat.connectionState == .ready && chat.modelTasks.contains("rerank")
            && !tools.isManaging && !tools.isEmbedding
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Document reranking").font(.title2.bold())
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Text(chat.modelName ?? "No active model").font(.headline)
            if !available {
                Text("Load a model that supports reranking from Local models, then Connect.")
                    .foregroundStyle(.secondary)
            }
            TextField("Query", text: $tools.rerankQuery)
                .textFieldStyle(.roundedBorder).disabled(tools.isReranking)
                .accessibilityLabel("Reranking query")
            Text("One document per line · 1–256 documents · query and documents at most 768 KiB total")
                .font(.caption)
            TextEditor(text: $tools.rerankDocuments)
                .font(.body).frame(minHeight: 120, maxHeight: 180)
                .border(Color.secondary.opacity(0.3)).disabled(tools.isReranking)
                .accessibilityLabel("Reranking documents")
            HStack {
                Text("Results")
                TextField("All documents", text: $tools.rerankTopN)
                    .frame(width: 140).textFieldStyle(.roundedBorder).disabled(tools.isReranking)
                Spacer()
                if tools.isReranking {
                    ProgressView().controlSize(.small)
                    Button("Cancel") { tools.cancelReranking() }
                } else {
                    Button("Rank documents") {
                        if let model = chat.modelName { tools.rerank(model: model) }
                    }.disabled(!available)
                }
            }
            if let result = tools.rerankResult {
                HStack {
                    Text("\(result.results.count) results · tokens: \(result.usage.prompt_tokens)").font(.caption)
                    Spacer()
                    Button("Copy JSON") {
                        do {
                            let text = String(decoding: try result.jsonData(query: tools.rerankResultQuery), as: UTF8.self)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(text, forType: .string)
                            exportError = nil
                        } catch { exportError = error.localizedDescription }
                    }
                    Button("Save JSON…") { export(result) }
                }
                Text("Query: \(tools.rerankResultQuery)").font(.caption).lineLimit(2)
                List(result.results) { row in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Document \(row.index + 1) · score \(String(format: "%.6f", row.relevance_score))")
                            .font(.headline)
                        Text(row.document.text).lineLimit(4).textSelection(.enabled)
                    }
                }
            } else { Spacer() }
            if let error = tools.rerankError ?? exportError {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).textSelection(.enabled)
            }
            Text("Query, documents and scores stay out of conversation history. JSON exports include the original query and ranked documents. Cancel stops the client request.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(22).frame(minWidth: 700, minHeight: 610)
    }

    private func export(_ result: DesktopRerankResult) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "bloom-rerank.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try result.jsonData(query: tools.rerankResultQuery).write(to: url, options: .atomic); exportError = nil }
        catch { exportError = error.localizedDescription }
    }
}
