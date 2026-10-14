import Foundation

// The pluggable LLM engine: canonical request/response value types shared by
// every dialect client. The App layer owns policies (chat, voice) and resolves
// credentials into ModelEndpoint; CoucouKit never touches the Keychain.

/// Which wire format an endpoint speaks. All dialect differences live in the
/// converters and clients; everything above them is canonical.
enum Dialect: String, Codable {
    case anthropicMessages
    case openaiChat
}

/// A fully resolved endpoint. Keychain resolution happens in the App layer;
/// `apiKey == nil` sends no Authorization header (local models, optional auth).
struct ModelEndpoint: Equatable, Sendable {
    /// Full prefix including the required `/v1`, but not `/chat/completions`
    /// or `/messages` — those belong to the dialect.
    var baseURL: String
    var dialect: Dialect
    var apiKey: String?
    var model: String
}

/// A JSON value for tool input schemas, decodable without boilerplate.
enum JSONValue: Equatable, Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case object([String: JSONValue])
    case array([JSONValue])
}

/// How a tool is advertised to the model.
struct ToolDeclaration: Equatable, Sendable {
    var name: String
    var description: String
    var inputSchema: [String: JSONValue]
    /// Anthropic server-side tool type (e.g. "web_search_20250305"). When set,
    /// the Anthropic converter serializes the flat provider form — the other
    /// keys ride top-level (max_uses, user_location go in inputSchema) and
    /// description is not sent. The OpenAI converter skips such declarations
    /// (no equivalent).
    var serverType: String? = nil
}

/// One turn in a conversation. Anthropic's block format is the canonical
/// base: the OpenAI path already treats Anthropic JSON as the source of truth
/// for history, and server-side tool round-trips (encrypted content,
/// citations) must survive unchanged.
struct BrainMessage: Equatable, Sendable {
    enum Role: String, Sendable {
        case user, assistant
    }

    enum Block: Equatable, Sendable {
        case text(String)
        case image(mediaType: String, base64: String)
        case document(mediaType: String, base64: String)
        /// A client-side tool call the App layer must execute.
        case toolUse(id: String, name: String, inputJSON: String)
        case toolResult(id: String, content: String)
        /// An Anthropic server-side tool use, kept verbatim for round-trips.
        case serverToolUse(id: String, name: String, inputJSON: String)
        /// A web-search tool result, passed back raw — never parsed here.
        case webSearchToolResult(id: String, rawJSON: String)
    }

    let role: Role
    var blocks: [Block]
    /// Verbatim wire form of an assistant message returned by a provider
    /// (its content array, canonically serialized). Round-trip fidelity:
    /// encrypted_content, citations, tool_use_id and future fields survive
    /// untouched because the Anthropic serializer replays this instead of
    /// rebuilding from blocks. Blocks stay the typed view; neither is mutated
    /// once parsed. nil on messages built locally.
    var rawBlocksJSON: String? = nil

    init(role: Role, blocks: [Block], rawBlocksJSON: String? = nil) {
        self.role = role
        self.blocks = blocks
        self.rawBlocksJSON = rawBlocksJSON
    }
}

enum StopReason: Equatable, Sendable {
    case endTurn
    case toolUse
    case pauseTurn
    case other
}

struct TokenUsage: Equatable, Sendable {
    var inputTokens: Int
    var outputTokens: Int
}

struct CompletionRequest: Sendable {
    var system: String
    var messages: [BrainMessage]
    var tools: [ToolDeclaration] = []
    var maxTokens: Int = 4096
    var temperature: Double? = nil
    /// Sent as the anthropic-beta header (e.g. web-search-2025-03-05).
    var betas: [String] = []
    /// Per call shape: 45/30/120/300s today.
    var timeout: TimeInterval
}

struct CompletionResponse: Sendable {
    var blocks: [BrainMessage.Block]
    var stopReason: StopReason
    var usage: TokenUsage?
    var model: String?
    /// The response's content array as received, canonically serialized.
    /// The tool-use loop copies it into the assistant message's
    /// rawBlocksJSON so the next round-trip replays it verbatim.
    var rawBlocksJSON: String? = nil
}

/// A tool call extracted from a response block, handed to a ClientTool.
struct ToolCall: Equatable, Sendable {
    var id: String
    var name: String
    var inputJSON: String
}

/// What a ClientTool returns. `acted` carries the "already done, don't do it
/// again" signal that drives the voice stack's "C'est fait." feedback.
struct ToolOutcome: Sendable {
    var text: String
    var acted: Bool
}

enum BrainError: Error, Equatable, Sendable {
    case network(String)
    case http(status: Int, body: String)
    case modelNotFound(provider: String, model: String)
    case cancelled
    case emptyResponse
}

/// What a streaming client emits: filtered text deltas, then each complete
/// turn. AsyncThrowingStream supports cooperative cancellation natively.
enum BrainEvent: Sendable {
    /// A text delta with thinking blocks already filtered out.
    case delta(String)
    /// One complete round-trip response.
    case turn(CompletionResponse)
}

// MARK: - JSON bridging

extension JSONValue {
    /// The plain Foundation value, ready for JSONSerialization.
    var jsonAny: Any {
        switch self {
        case .string(let s): return s
        case .number(let n): return n
        case .bool(let b): return b
        case .null: return NSNull()
        case .object(let o): return o.mapValues(\.jsonAny)
        case .array(let a): return a.map(\.jsonAny)
        }
    }
}

/// Deterministic JSON text (sorted keys, no whitespace) — the canonical form
/// stored in inputJSON / rawBlocksJSON and compared byte-for-byte in goldens.
func canonicalJSONString(from value: Any) -> String? {
    guard JSONSerialization.isValidJSONObject(value),
          let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else {
        return nil
    }
    return String(data: data, encoding: .utf8)
}
