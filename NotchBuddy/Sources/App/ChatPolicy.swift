import Foundation

// MARK: - Chat policy

/// Chat's conversation policy: owns the `[BrainMessage]` history (no
/// ChatSession — rollback stays with the caller, chat never trims), assembles
/// each turn's canonical request, and consumes the engine's responses. The
/// provider switches live here, not in the engine: Anthropic goes through
/// AnthropicClient with the server-side web_search tool; every other provider
/// speaks the OpenAI chat-completions dialect, streaming when it helps.
@MainActor
final class ChatPolicy {
    static let shared = ChatPolicy()

    /// Multi-turn history in canonical form. Assistant turns carry
    /// rawBlocksJSON so everything the API returned (citations,
    /// encrypted_index, tool_use_id) replays verbatim on the next request.
    private(set) var history: [BrainMessage] = []

    func reset() {
        history = []
    }

    /// Chosen in Settings; falls back to the default when the field is left empty.
    private var model: String {
        let m = AppState.shared.claudeModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return m.isEmpty ? AppState.defaultClaudeModel : m
    }

    private var apiKey: String? { KeychainStore.shared.get(ChatProvider.anthropic.keychainKey) }

    /// Resolved once: NSFullUserName() is a system call, and the name cannot change under us
    /// while the app runs.
    private let systemPrompt = ChatPolicy.makeSystemPrompt()

    /// Greets the user by their macOS first name when there is one worth using, and stays
    /// neutral otherwise — same wording as the Windows build.
    private nonisolated static func makeSystemPrompt() -> String {
        let opening = if let firstName = resolveUserFirstName() {
            "You are Mochi, \(firstName)'s personal AI assistant embedded in the notch of their Mac."
        } else {
            "You are Mochi, a personal AI assistant embedded in the notch of the user's Mac."
        }
        return """
        \(opening) \
        You have web search access and can help with absolutely anything — research, coding, finding places, recommendations, tasks, questions. \
        Respond in the user's language. Be thorough and complete — use as much detail as the task requires. \
        Use light Markdown when it helps: short paragraphs, bullet lists, **bold**, `inline code` and fenced code blocks. Avoid tables and big headings: the chat window is small.
        """
    }

    /// Chat's web search, in the engine's declaration form — serializes to
    /// the flat provider JSON: {type, name, max_uses}.
    private let webSearchTool = ToolDeclaration(
        name: "web_search",
        description: "",
        inputSchema: ["max_uses": .number(5)],
        serverType: "web_search_20250305")

    // MARK: - Chat (multi-turn, natural text + web search)

    func chat(query: String, context: PromptContext?, state: AppState) async {
        if DemoEngine.shared.isActive {
            state.stateOverride = .thinking
            await DemoEngine.shared.streamChatResponse(for: query)
            state.stateOverride = nil
            return
        }
        guard state.chatProvider == .anthropic else {
            await chatOpenAICompatible(query: query, context: context, state: state)
            return
        }
        guard let key = apiKey, !key.isEmpty else {
            await showError("API key missing. Open settings.", state: state)
            return
        }

        // Build user content for this turn
        var userBlocks: [BrainMessage.Block] = []

        // Add file/window context on first message only
        if history.isEmpty, let context = context {
            switch context {
            case .window(let app, let title, let url):
                var text = "Context — App: \(app), Window: \(title)"
                if let url = url { text += ", URL: \(url)" }
                userBlocks.append(.text(text))
            case .file(let name, let fileURL):
                if let fileURL = fileURL, let block = readFileAsBlock(url: fileURL) {
                    userBlocks.append(block)
                }
                userBlocks.append(.text("File: \(name)"))
            }
        }
        userBlocks.append(.text(query))

        history.append(BrainMessage(role: .user, blocks: userBlocks))

        let client = AnthropicClient(endpoint: ModelEndpoint(
            baseURL: "https://api.anthropic.com/v1",
            dialect: .anthropicMessages,
            apiKey: key,
            model: model))
        let request = CompletionRequest(
            system: systemPrompt,
            messages: history,
            tools: [webSearchTool],
            maxTokens: 4096,
            betas: ["web-search-2025-03-05"],
            timeout: 45)

        do {
            let response = try await client.complete(request)
            await handleChatResult(response, state: state)
        } catch {
            history.removeLast()
            await showError(Self.chatErrorMessage(for: error), state: state)
        }
    }

    // MARK: - OpenAI-compatible chat (Google Gemini / OpenAI / Ollama / LM Studio)

    func chatOpenAICompatible(query: String, context: PromptContext?, state: AppState) async {
        let provider = state.chatProvider
        guard provider != .anthropic else { return }

        // Full prefix including /v1 — normaliseURL strips it for storage,
        // LocalChat and the endpoints expect it back.
        let baseURL: String
        if provider == .ollama {
            baseURL = LocalChat.normaliseURL(state.ollamaServerURL) + "/v1"
        } else if provider == .lmstudio {
            baseURL = LocalChat.normaliseURL(state.lmstudioServerURL) + "/v1"
        } else if provider == .custom {
            baseURL = state.customChatBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            switch provider {
            case .google:    baseURL = "https://generativelanguage.googleapis.com/v1beta/openai"
            case .openai:    baseURL = "https://api.openai.com/v1"
            case .deepseek:  baseURL = "https://api.deepseek.com/v1"
            case .qwen:      baseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            case .moonshot:  baseURL = "https://api.moonshot.cn/v1"
            case .anthropic, .ollama, .lmstudio, .custom: baseURL = ""
            }
        }

        guard !baseURL.isEmpty else {
            if provider.isLocal {
                let name = provider == .ollama ? "Ollama" : "LM Studio"
                await showError("Connect \(name) in Settings → Chat first.", state: state)
            } else if provider == .custom {
                await showError("Set a Custom endpoint in Settings → Chat first.", state: state)
            }
            return
        }
        guard let url = URL(string: "\(baseURL)/chat/completions") else { return }
        if provider == .custom, state.activeChatModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            await showError("Pick a model for the Custom endpoint in Settings → Chat.", state: state)
            return
        }

        // Auth: local servers send nothing; custom endpoints authenticate
        // optionally (empty key = no header); the others require a key.
        let apiKey: String?
        if provider == .custom {
            let key = KeychainStore.shared.get(provider.keychainKey) ?? ""
            apiKey = key.isEmpty ? nil : key
        } else if provider.isLocal {
            apiKey = nil
        } else {
            guard let key = KeychainStore.shared.get(provider.keychainKey), !key.isEmpty else {
                await showError("\(provider.displayName) API key missing. Configure it in Settings.", state: state)
                return
            }
            apiKey = key
        }

        // Build messages. Known degradation (pinned, replaced by tool-based
        // reading later): each history turn survives only through its first
        // text block — images, documents and later text blocks are dropped.
        var msgs: [[String: Any]] = [["role": "system", "content": systemPrompt]]
        for m in history {
            let firstText: String?
            if case .text(let text)? = m.blocks.first(where: { if case .text = $0 { return true } else { return false } }) {
                firstText = text
            } else {
                firstText = nil
            }
            if let firstText {
                msgs.append(["role": m.role.rawValue, "content": firstText])
            } else {
                // Tool-only assistant turn (no text at all): kept as an empty
                // turn — the old path sent the raw Anthropic block array to
                // the OpenAI endpoint, which no server accepts.
                msgs.append(["role": m.role.rawValue, "content": ""])
            }
        }
        var userText = query
        if history.isEmpty, let ctx = context {
            switch ctx {
            case .window(let app, let title, let url):
                var prefix = "Context — App: \(app), Window: \(title)"
                if let u = url { prefix += ", URL: \(u)" }
                userText = prefix + "\n\n" + query
            case .file(let name, let fileURL):
                if provider.isLocal, let fileURL = fileURL {
                    let ext = fileURL.pathExtension.lowercased()
                    let binaryExts = ["pdf", "jpg", "jpeg", "png", "gif", "webp"]
                    if !binaryExts.contains(ext),
                       let text = try? String(contentsOf: fileURL, encoding: .utf8), !text.isEmpty {
                        let truncated = text.count > 24_000
                            ? String(text.prefix(24_000)) + "\n[truncated]"
                            : text
                        userText = "File: \(name)\n\n\(truncated)\n\n" + query
                    } else {
                        userText = "File: \(name)\n\n" + query
                    }
                } else {
                    userText = "File: \(name)\n\n" + query
                }
            }
        }
        msgs.append(["role": "user", "content": userText])
        history.append(BrainMessage(role: .user, blocks: [.text(userText)]))

        let useStream = provider.isLocal || provider == .deepseek || provider == .qwen || provider == .moonshot || provider == .custom
        var body: [String: Any] = [
            "model": state.activeChatModel,
            "max_tokens": 4096,
            "messages": msgs,
        ]
        if useStream { body["stream"] = true }

        var req = URLRequest(url: url, timeoutInterval: useStream ? 120 : 30)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey {
            req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        if useStream {
            // Add placeholder (hidden until first token via ChatBubble empty-content guard)
            let placeholder = ChatMessage(role: .assistant, content: "")
            let msgId = placeholder.id
            state.chatHistory.append(placeholder)
            state.stateOverride = .thinking
            let modelCopy = state.activeChatModel
            let streamBody: [String: Any] = [
                "model": modelCopy,
                "messages": msgs,
                "stream": true,
                "max_tokens": 4096,
            ]
            let encodedBody = (try? JSONSerialization.data(withJSONObject: streamBody)) ?? Data()
            do {
                let final = try await LocalChat.streamChat(
                    baseURL: baseURL,
                    encodedBody: encodedBody,
                    model: modelCopy,
                    apiKey: apiKey
                ) { [state, msgId] visible in
                    if !visible.isEmpty, state.stateOverride == .thinking {
                        state.stateOverride = nil   // hide typing dots on first visible text
                    }
                    if let idx = state.chatHistory.firstIndex(where: { $0.id == msgId }) {
                        state.chatHistory[idx].content = visible
                    }
                }
                history.append(BrainMessage(role: .assistant, blocks: [.text(final)]))
                if let idx = state.chatHistory.firstIndex(where: { $0.id == msgId }) {
                    state.chatHistory[idx].content = final
                }
                state.stateOverride = nil
                state.view = .prompt
                NotificationCenter.default.post(name: .triggerEmote, object: BotEmote.happy)
            } catch let e as LocalChatError {
                history.removeLast()
                state.chatHistory.removeAll { $0.id == msgId }
                state.stateOverride = nil
                let msg: String
                switch e {
                case .serverUnreachable:
                    if provider == .ollama {
                        msg = "Ollama isn't running. Open it, then ask again."
                    } else if provider == .lmstudio {
                        msg = "Start the local server in LM Studio, then ask again."
                    } else {
                        msg = "\(provider.displayName) is unreachable. Check your connection and API key."
                    }
                case .modelNotFound(let m):
                    msg = "\(m) isn't installed. Pick another model above the chat box."
                case .serverError(let s):
                    msg = s
                }
                await showError(msg, state: state)
            } catch {
                history.removeLast()
                state.chatHistory.removeAll { $0.id == msgId }
                state.stateOverride = nil
                await showError(error.localizedDescription, state: state)
            }
        } else {
            // Non-streaming (Google, OpenAI)
            do {
                let (data, response) = try await URLSession.shared.data(for: req)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let err = (json["error"] as? [String: Any])?["message"] as? String {
                        throw NSError(domain: "ChatAPI", code: 0, userInfo: [NSLocalizedDescriptionKey: err])
                    }
                    throw NSError(domain: "ChatAPI", code: 0, userInfo: [NSLocalizedDescriptionKey: "HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)"])
                }
                guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let choices = json["choices"] as? [[String: Any]],
                      let message = choices.first?["message"] as? [String: Any],
                      let content = message["content"] as? String else {
                    throw NSError(domain: "ChatAPI", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unexpected response format"])
                }
                let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
                history.append(BrainMessage(role: .assistant, blocks: [.text(trimmed)]))
                state.chatHistory.append(ChatMessage(role: .assistant, content: trimmed))
                state.stateOverride = nil
                state.view = .prompt
                NotificationCenter.default.post(name: .triggerEmote, object: BotEmote.happy)
            } catch {
                history.removeLast()
                await showError(error.localizedDescription, state: state)
            }
        }
    }

    // MARK: - Chat result handler

    private func handleChatResult(_ response: CompletionResponse, state: AppState) async {
        // Store full content (includes tool_use/tool_result blocks) for correct multi-turn context
        let content = Self.contentArray(from: response)
        history.append(AnthropicConverter.assistantMessage(fromContentJSON: content))

        guard let text = claudeResponseText(fromContent: content) else {
            await showError("No response text.", state: state)
            return
        }

        // Add to display history
        state.chatHistory.append(ChatMessage(role: .assistant, content: text))

        state.stateOverride = nil
        state.view = .prompt
        NotificationCenter.default.post(name: .triggerEmote, object: BotEmote.happy)
    }

    private func showError(_ message: String, state: AppState) async {
        state.stateOverride = .error
        state.noteMessage = message
        state.view = .note
    }

    /// The engine's errors, shown with the exact messages the old inline
    /// chat path produced.
    private static func chatErrorMessage(for error: Error) -> String {
        switch error {
        case BrainError.modelNotFound:
            // Reads the Settings field (untrimmed), like the old callAPI did.
            return "Model not found: \(AppState.shared.claudeModel). Pick another one in Settings."
        case BrainError.http(_, let body):
            return body
        case BrainError.network(let message):
            return message
        case BrainError.emptyResponse:
            return "Unexpected API response."
        case BrainError.cancelled:
            return "cancelled"
        default:
            return error.localizedDescription
        }
    }

    /// The response's content array back in raw form — what the conversation
    /// storage holds (rawBlocksJSON is the response's content verbatim).
    private static func contentArray(from response: CompletionResponse) -> [[String: Any]] {
        if let raw = response.rawBlocksJSON,
           let data = raw.data(using: .utf8),
           let content = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            return content
        }
        return response.blocks.compactMap(AnthropicConverter.blockJSON)
    }

    // MARK: - File content block builder

    private func readFileAsBlock(url: URL) -> BrainMessage.Block? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        let ext = url.pathExtension.lowercased()
        let base64 = data.base64EncodedString()

        if ext == "pdf" {
            return .document(mediaType: "application/pdf", base64: base64)
        } else if ["jpg", "jpeg"].contains(ext) {
            return .image(mediaType: "image/jpeg", base64: base64)
        } else if ext == "png" {
            return .image(mediaType: "image/png", base64: base64)
        } else if ext == "gif" {
            return .image(mediaType: "image/gif", base64: base64)
        } else if ext == "webp" {
            return .image(mediaType: "image/webp", base64: base64)
        } else {
            // Text/code — inline as text if <= 200 KB
            guard data.count <= 200_000,
                  let text = String(data: data, encoding: .utf8) else { return nil }
            return .text("File contents:\n\(text)")
        }
    }
}
