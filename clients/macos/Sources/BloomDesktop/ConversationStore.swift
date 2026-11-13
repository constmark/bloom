import Foundation
import Darwin

struct DesktopConversation: Identifiable, Codable, Equatable {
    var id = UUID()
    var createdAt = Date()
    var updatedAt = Date()
    var messages: [DesktopMessage] = []
    var draft = ""

    var title: String {
        guard let prompt = messages.first(where: { $0.role == .user })?.content else {
            return "New conversation"
        }
        return String(prompt.split(whereSeparator: \.isWhitespace).joined(separator: " ").prefix(64))
    }
}

struct ConversationArchive: Codable, Equatable {
    static let currentVersion = 1
    static let maxBytes = 16 * 1024 * 1024
    static let maxConversations = 128
    static let maxMessages = 2048
    static let maxTextBytes = 1024 * 1024

    var version = currentVersion
    var selectedID: UUID
    var conversations: [DesktopConversation]

    static func empty() -> Self {
        let conversation = DesktopConversation()
        return Self(selectedID: conversation.id, conversations: [conversation])
    }

    func validate() throws {
        guard version == Self.currentVersion else {
            throw ConversationStorageError.invalid("Unsupported conversation archive version \(version).")
        }
        guard !conversations.isEmpty, conversations.count <= Self.maxConversations,
              Set(conversations.map(\.id)).count == conversations.count,
              conversations.contains(where: { $0.id == selectedID }) else {
            throw ConversationStorageError.invalid("Invalid conversation selection, count, or duplicate IDs.")
        }
        var textBytes = 0
        for conversation in conversations {
            guard conversation.createdAt.timeIntervalSince1970.isFinite,
                  conversation.updatedAt.timeIntervalSince1970.isFinite,
                  conversation.messages.count <= Self.maxMessages,
                  Set(conversation.messages.map(\.id)).count == conversation.messages.count,
                  conversation.draft.utf8.count <= Self.maxTextBytes,
                  conversation.messages.allSatisfy({ $0.content.utf8.count <= Self.maxTextBytes }) else {
                throw ConversationStorageError.invalid("Conversation messages exceed storage limits or contain duplicate IDs.")
            }
            // Check the aggregate before JSONEncoder allocates its output.
            // Per-message limits alone permit hundreds of GiB in an archive.
            let draftBytes = conversation.draft.utf8.count
            guard draftBytes <= Self.maxBytes - textBytes else {
                throw ConversationStorageError.invalid("Conversation text exceeds the 16 MiB archive limit.")
            }
            textBytes += draftBytes
            for message in conversation.messages {
                let count = message.content.utf8.count
                guard count <= Self.maxBytes - textBytes else {
                    throw ConversationStorageError.invalid("Conversation text exceeds the 16 MiB archive limit.")
                }
                textBytes += count
            }
        }
    }
}

enum ConversationStorageError: LocalizedError {
    case invalid(String)
    case io(String)

    var errorDescription: String? {
        switch self {
        case let .invalid(message), let .io(message): return message
        }
    }
}

protocol ConversationPersistence {
    func load() throws -> ConversationArchive?
    func save(_ archive: ConversationArchive) throws
    /// Preserve the unreadable archive before the user starts a fresh store.
    func quarantine() throws
}

/// One bounded, versioned document contains only conversation text and IDs.
/// Connection credentials and runtime state are deliberately absent.
struct FileConversationStore: ConversationPersistence {
    let url: URL

    static var standard: Self {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return Self(url: base.appendingPathComponent("Bloom", isDirectory: true)
            .appendingPathComponent("conversations.json"))
    }

    func load() throws -> ConversationArchive? {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        if descriptor < 0 {
            if errno == ENOENT { return nil }
            throw ioError("Could not open conversation history")
        }
        defer { Darwin.close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw ioError("Could not inspect conversation history") }
        guard (info.st_mode & S_IFMT) == S_IFREG else {
            throw ConversationStorageError.invalid("Conversation history must be a regular file.")
        }
        guard info.st_size <= ConversationArchive.maxBytes else {
            throw ConversationStorageError.invalid("Conversation history exceeds the 16 MiB limit.")
        }
        var bytes = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = Darwin.read(descriptor, &buffer, min(buffer.count, ConversationArchive.maxBytes + 1 - bytes.count))
            if count < 0 {
                if errno == EINTR { continue }
                throw ioError("Could not read conversation history")
            }
            if count == 0 { break }
            bytes.append(contentsOf: buffer.prefix(count))
            guard bytes.count <= ConversationArchive.maxBytes else {
                throw ConversationStorageError.invalid("Conversation history exceeds the 16 MiB limit.")
            }
        }
        let archive: ConversationArchive
        do { archive = try JSONDecoder().decode(ConversationArchive.self, from: bytes) }
        catch { throw ConversationStorageError.invalid("Conversation history is malformed. The original file has been preserved.") }
        try archive.validate()
        return archive
    }

    func save(_ archive: ConversationArchive) throws {
        try archive.validate()
        let bytes = try JSONEncoder().encode(archive)
        guard bytes.count <= ConversationArchive.maxBytes else {
            throw ConversationStorageError.invalid("Conversation history exceeds the 16 MiB limit. Delete an older conversation to save.")
        }
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let directoryValues = try directory.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
        guard directoryValues.isDirectory == true, directoryValues.isSymbolicLink != true else {
            throw ConversationStorageError.invalid("Conversation storage directory must not be a symbolic link.")
        }
        let temporary = directory.appendingPathComponent(".conversations-\(UUID().uuidString).tmp")
        let descriptor = Darwin.open(temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw ioError("Could not create conversation checkpoint") }
        defer {
            Darwin.close(descriptor)
            try? FileManager.default.removeItem(at: temporary)
        }
        try bytes.withUnsafeBytes { raw in
            var offset = 0
            while offset < raw.count {
                let count = Darwin.write(descriptor, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw ioError("Could not write conversation checkpoint") }
                offset += count
            }
        }
        guard fsync(descriptor) == 0 else { throw ioError("Could not flush conversation checkpoint") }
        guard Darwin.rename(temporary.path, url.path) == 0 else { throw ioError("Could not replace conversation history") }
        // Persist the directory entry too when supported by the filesystem.
        let directoryDescriptor = Darwin.open(directory.path, O_RDONLY)
        if directoryDescriptor >= 0 {
            _ = fsync(directoryDescriptor)
            Darwin.close(directoryDescriptor)
        }
    }

    func quarantine() throws {
        let backup = url.deletingLastPathComponent()
            .appendingPathComponent("conversations-recovery-\(UUID().uuidString).json")
        guard Darwin.rename(url.path, backup.path) == 0 else {
            if errno == ENOENT { return }
            throw ioError("Could not preserve the damaged conversation history")
        }
    }

    private func ioError(_ action: String) -> ConversationStorageError {
        .io("\(action): \(String(cString: strerror(errno)))")
    }
}
