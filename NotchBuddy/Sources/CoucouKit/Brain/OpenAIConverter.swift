import Foundation

// Pure canonical ↔ OpenAI chat-completions JSON conversion. No I/O, no state.
// Known degradations pinned by goldens, replaced wholesale by tool-based file
// reading later: only the FIRST text block of each message survives; images
// and documents are dropped; server tool blocks have no equivalent.

enum OpenAIConverter {

    // MARK: Request

    /// The POST /chat/completions body. `model` comes from the endpoint.
    /// The client adds "stream": true for the streaming path.
    static func requestBody(_ request: CompletionRequest, model: String) -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "max_tokens": request.maxTokens,
            "messages": messages(system: request.system, messages: request.messages),
        ]
        let functions = request.tools.compactMap(toolJSON)
        if !functions.isEmpty { body["tools"] = functions }
        if let temperature = request.temperature { body["temperature"] = temperature }
        return body
    }

    /// Canonical messages as OpenAI chat messages. The system prompt leads;
    /// tool results become their own role:"tool" messages after the user turn.
    static func messages(system: String, messages: [BrainMessage]) -> [[String: Any]] {
        var out: [[String: Any]] = [["role": "system", "content": system]]
        for message in messages {
            let texts = message.blocks.compactMap { block -> String? in
                if case .text(let text) = block { return text }
                return nil
            }
            switch message.role {
            case .assistant:
                var json: [String: Any] = ["role": "assistant"]
                // Known degradation (pinned, later replaced by tool-based
                // reading): only the first text block of the turn survives.
                json["content"] = texts.first ?? ""
                let calls = message.blocks.compactMap { block -> [String: Any]? in
                    guard case .toolUse(let id, let name, let inputJSON) = block else { return nil }
                    return ["id": id, "type": "function",
                            "function": ["name": name, "arguments": inputJSON]]
                }
                if !calls.isEmpty { json["tool_calls"] = calls }
                out.append(json)
            case .user:
                if let text = texts.first {
                    out.append(["role": "user", "content": text])
                }
                for block in message.blocks {
                    guard case .toolResult(let id, let content) = block else { continue }
                    out.append(["role": "tool", "tool_call_id": id, "content": content])
                }
            }
        }
        return out
    }

    /// A client tool declaration in the function-calling form. Server tools
    /// (serverType set) have no OpenAI equivalent and are skipped.
    static func toolJSON(_ declaration: ToolDeclaration) -> [String: Any]? {
        guard declaration.serverType == nil else { return nil }
        return ["type": "function",
                "function": ["name": declaration.name,
                             "description": declaration.description,
                             "parameters": declaration.inputSchema.mapValues(\.jsonAny)]]
    }

    // MARK: Response

    /// A full non-streaming chat completion.
    static func response(fromJSON json: [String: Any]) -> CompletionResponse? {
        guard let choice = (json["choices"] as? [[String: Any]])?.first else { return nil }
        let message = choice["message"] as? [String: Any] ?? [:]
        var blocks: [BrainMessage.Block] = []
        if let content = message["content"] as? String {
            let filtered = filterThinkingBlocks(content)
            if !filtered.isEmpty { blocks.append(.text(filtered)) }
        }
        for call in message["tool_calls"] as? [[String: Any]] ?? [] {
            guard let id = call["id"] as? String,
                  let function = call["function"] as? [String: Any],
                  let name = function["name"] as? String else { continue }
            // arguments is already the wire JSON string — kept verbatim.
            blocks.append(.toolUse(id: id, name: name,
                                   inputJSON: function["arguments"] as? String ?? "{}"))
        }
        guard !blocks.isEmpty else { return nil }
        return CompletionResponse(
            blocks: blocks,
            stopReason: stopReason(choice["finish_reason"] as? String),
            usage: usage(fromJSON: json["usage"]),
            model: json["model"] as? String,
            rawBlocksJSON: nil)
    }

    static func stopReason(_ raw: String?) -> StopReason {
        switch raw {
        case "stop": return .endTurn
        case "tool_calls": return .toolUse
        default: return .other
        }
    }

    private static func usage(fromJSON json: Any?) -> TokenUsage? {
        guard let json = json as? [String: Any],
              let prompt = json["prompt_tokens"] as? Int,
              let completion = json["completion_tokens"] as? Int else { return nil }
        return TokenUsage(inputTokens: prompt, outputTokens: completion)
    }

    // MARK: SSE

    /// What one `data:` line carries.
    struct SSEEvent: Equatable, Sendable {
        var text: String?
        /// Tool-call fragments, accumulated by index while streaming.
        var toolCall: (index: Int, id: String?, name: String?, arguments: String?)?
        var finishReason: String?

        static func == (lhs: SSEEvent, rhs: SSEEvent) -> Bool {
            lhs.text == rhs.text
                && lhs.toolCall?.index == rhs.toolCall?.index
                && lhs.toolCall?.id == rhs.toolCall?.id
                && lhs.toolCall?.name == rhs.toolCall?.name
                && lhs.toolCall?.arguments == rhs.toolCall?.arguments
                && lhs.finishReason == rhs.finishReason
        }
    }

    /// Parses one OpenAI-SSE event line. nil for non-data lines, [DONE],
    /// missing or null content, and reasoning-only deltas.
    static func parseSSEEvent(_ line: String) -> SSEEvent? {
        guard line.hasPrefix("data: ") else { return nil }
        let payload = String(line.dropFirst(6))
        guard payload != "[DONE]" else { return nil }
        guard let data = payload.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choice = (json["choices"] as? [[String: Any]])?.first else { return nil }

        var event = SSEEvent(text: nil, toolCall: nil, finishReason: nil)
        if let delta = choice["delta"] as? [String: Any] {
            if let content = delta["content"] as? String { event.text = content }
            if let call = (delta["tool_calls"] as? [[String: Any]])?.first {
                let function = call["function"] as? [String: Any] ?? [:]
                event.toolCall = (index: call["index"] as? Int ?? 0,
                                  id: call["id"] as? String,
                                  name: function["name"] as? String,
                                  arguments: function["arguments"] as? String)
            }
        }
        event.finishReason = choice["finish_reason"] as? String
        // Nil when nothing usable: reasoning-only deltas, keep-alives.
        if event.text == nil && event.toolCall == nil && event.finishReason == nil { return nil }
        return event
    }

    /// The text delta of one SSE line — the classic `delta.content` parse.
    static func parseSSEDelta(_ line: String) -> String? {
        parseSSEEvent(line)?.text
    }

    // MARK: Think-block filtering (reasoning models like DeepSeek-R1)

    /// Removes completed `<think>` blocks.
    static func filterThinkingBlocks(_ text: String) -> String {
        let pattern = "<think>[\\s\\S]*?</think>"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// For streaming: also hides content inside an open (unclosed) `<think>`
    /// block that hasn't received its closing tag yet.
    static func progressiveFilter(_ text: String) -> String {
        let cleaned = filterThinkingBlocks(text)
        if let range = cleaned.range(of: "<think>") {
            return String(cleaned[..<range.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return cleaned
    }
}
