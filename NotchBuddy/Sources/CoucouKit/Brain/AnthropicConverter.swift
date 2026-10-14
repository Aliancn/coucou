import Foundation

// Pure canonical ↔ Anthropic Messages JSON conversion. No I/O, no state.
// Round-trip fidelity red line: everything the API returns but the typed
// blocks can't hold (encrypted_content, citations, tool_use_id, future
// fields) survives through BrainMessage.rawBlocksJSON, replayed verbatim by
// contentJSON(_:). Normalization never happens here — only in the tests'
// .shape comparison.

enum AnthropicConverter {

    // MARK: Request

    /// The POST /messages body. `model` comes from the endpoint, never the
    /// request, so the two cannot diverge.
    static func requestBody(_ request: CompletionRequest, model: String) -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "max_tokens": request.maxTokens,
            "system": request.system,
            "messages": request.messages.map(messageJSON),
        ]
        if !request.tools.isEmpty { body["tools"] = request.tools.map(toolJSON) }
        if let temperature = request.temperature { body["temperature"] = temperature }
        return body
    }

    /// One message as Anthropic expects it. Assistant messages that came from
    /// a response replay rawBlocksJSON verbatim; only locally built messages
    /// are rebuilt from blocks.
    static func messageJSON(_ message: BrainMessage) -> [String: Any] {
        var json: [String: Any] = ["role": message.role.rawValue]
        if message.role == .assistant,
           let raw = message.rawBlocksJSON,
           let data = raw.data(using: .utf8),
           let content = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            json["content"] = content
        } else {
            json["content"] = message.blocks.compactMap(blockJSON)
        }
        return json
    }

    /// One canonical block as a content-block JSON object, or nil when the
    /// block has no Anthropic form (user-built thinking etc.).
    static func blockJSON(_ block: BrainMessage.Block) -> [String: Any]? {
        switch block {
        case .text(let text):
            return ["type": "text", "text": text]
        case .image(let mediaType, let base64):
            return ["type": "image",
                    "source": ["type": "base64", "media_type": mediaType, "data": base64]]
        case .document(let mediaType, let base64):
            return ["type": "document",
                    "source": ["type": "base64", "media_type": mediaType, "data": base64]]
        case .toolUse(let id, let name, let inputJSON):
            return ["type": "tool_use", "id": id, "name": name, "input": decodedJSONObject(inputJSON)]
        case .toolResult(let id, let content):
            return ["type": "tool_result", "tool_use_id": id, "content": content]
        case .serverToolUse(let id, let name, let inputJSON):
            return ["type": "server_tool_use", "id": id, "name": name, "input": decodedJSONObject(inputJSON)]
        case .webSearchToolResult(let id, let rawJSON):
            guard let block = decodedJSONObject(rawJSON) as? [String: Any] else { return nil }
            var json = block
            json["tool_use_id"] = id
            return json
        }
    }

    /// A tool declaration. Server tools (serverType set) take the provider's
    /// flat form — max_uses and user_location ride in inputSchema's remaining
    /// keys; client tools take the classic name/description/input_schema form.
    static func toolJSON(_ declaration: ToolDeclaration) -> [String: Any] {
        var schema = declaration.inputSchema.mapValues(\.jsonAny)
        if let serverType = declaration.serverType {
            schema.removeValue(forKey: "type")
            var json: [String: Any] = ["type": serverType, "name": declaration.name]
            for (key, value) in schema { json[key] = value }
            return json
        }
        return ["name": declaration.name,
                "description": declaration.description,
                "input_schema": schema]
    }

    // MARK: Response

    /// A full Messages API response. rawBlocksJSON keeps the content array
    /// verbatim for the next round-trip.
    static func response(fromJSON json: [String: Any]) -> CompletionResponse? {
        guard let content = json["content"] as? [[String: Any]] else { return nil }
        return CompletionResponse(
            blocks: content.compactMap(block(fromJSON:)),
            stopReason: stopReason(json["stop_reason"] as? String),
            usage: usage(fromJSON: json["usage"]),
            model: json["model"] as? String,
            rawBlocksJSON: canonicalJSONString(from: content)
        )
    }

    /// The assistant message a response turns into — typed blocks for the
    /// caller, rawBlocksJSON for the next request.
    static func assistantMessage(fromContentJSON content: [[String: Any]]) -> BrainMessage {
        BrainMessage(role: .assistant,
                     blocks: content.compactMap(block(fromJSON:)),
                     rawBlocksJSON: canonicalJSONString(from: content))
    }

    /// One response content block, or nil for what the canonical model has no
    /// use for (thinking, redacted_thinking, future types). Dropped here does
    /// not mean lost: rawBlocksJSON still carries it back.
    static func block(fromJSON json: [String: Any]) -> BrainMessage.Block? {
        switch json["type"] as? String {
        case "text":
            guard let text = json["text"] as? String else { return nil }
            return .text(text)
        case "tool_use":
            guard let id = json["id"] as? String, let name = json["name"] as? String else { return nil }
            return .toolUse(id: id, name: name, inputJSON: canonicalJSONString(from: json["input"] ?? [:]) ?? "{}")
        case "server_tool_use":
            guard let id = json["id"] as? String, let name = json["name"] as? String else { return nil }
            return .serverToolUse(id: id, name: name, inputJSON: canonicalJSONString(from: json["input"] ?? [:]) ?? "{}")
        case "web_search_tool_result":
            guard let id = json["tool_use_id"] as? String else { return nil }
            return .webSearchToolResult(id: id, rawJSON: canonicalJSONString(from: json) ?? "{}")
        default:
            return nil
        }
    }

    static func stopReason(_ raw: String?) -> StopReason {
        switch raw {
        case "end_turn": return .endTurn
        case "tool_use": return .toolUse
        case "pause_turn": return .pauseTurn
        default: return .other
        }
    }

    private static func usage(fromJSON json: Any?) -> TokenUsage? {
        guard let json = json as? [String: Any],
              let input = json["input_tokens"] as? Int,
              let output = json["output_tokens"] as? Int else { return nil }
        return TokenUsage(inputTokens: input, outputTokens: output)
    }

    // MARK: JSON helpers

    private static func decodedJSONObject(_ string: String) -> Any {
        guard let data = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) else { return [:] }
        return json
    }
}
