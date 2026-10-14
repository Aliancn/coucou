import XCTest

// The chat path's byte-parity red line. The old inline path built its body
// from the stored conversation JSON verbatim; ChatPolicy now stores a
// [BrainMessage] history (user turns built as blocks, assistant turns via
// assistantMessage(fromContentJSON:) so rawBlocksJSON replays verbatim).
// Both forms of the same conversation must produce the same canonical body,
// byte for byte.

final class BrainChatParityTests: XCTestCase {

    private func fixtureJSON(_ name: String, file: StaticString = #filePath, line: UInt = #line) throws -> [String: Any] {
        let base = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: base, withExtension: ext),
                                "fixture \(name) not found", file: file, line: line)
        return try XCTUnwrap(try? JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
                             "fixture \(name) is not a JSON object", file: file, line: line)
    }

    /// A conversation as the App layer stores it: a first user turn with a
    /// dropped file (image + document blocks from readFileAsBlock, plus the
    /// "File:" text), an assistant turn replayed from the web-search fixture
    /// (citations, encrypted_index, server_tool_use — all verbatim), a plain
    /// text reply, and another assistant turn from the text fixture.
    private func storedConversation() throws -> [[String: Any]] {
        var conversation: [[String: Any]] = []
        let firstUser: [[String: Any]] = [
            ["type": "document",
             "source": ["type": "base64", "media_type": "application/pdf", "data": "cGRmZGF0YQ=="]],
            ["type": "image",
             "source": ["type": "base64", "media_type": "image/png", "data": "aW1hZ2U="]],
            ["type": "text", "text": "File: report.pdf"],
            ["type": "text", "text": "Que dis ce document ?"],
        ]
        conversation.append(["role": "user", "content": firstUser])
        let webSearch = try fixtureJSON("anthropic-web-search-normal.json")
        conversation.append(["role": "assistant",
                             "content": try XCTUnwrap(webSearch["content"] as? [[String: Any]])])
        conversation.append(["role": "user", "content": [["type": "text", "text": "Merci !"]]])
        let text = try fixtureJSON("anthropic-text.json")
        conversation.append(["role": "assistant",
                             "content": try XCTUnwrap(text["content"] as? [[String: Any]])])
        return conversation
    }

    /// The same conversation as ChatPolicy's [BrainMessage] history: user
    /// turns as blocks, assistant turns through assistantMessage(fromContentJSON:)
    /// — exactly how chat() and handleChatResult() append.
    private func chatPolicyHistory(_ conversation: [[String: Any]]) -> [BrainMessage] {
        conversation.map { message in
            let role: BrainMessage.Role = (message["role"] as? String) == "assistant" ? .assistant : .user
            let content = (message["content"] as? [[String: Any]]) ?? []
            if role == .assistant {
                return AnthropicConverter.assistantMessage(fromContentJSON: content)
            }
            return BrainMessage(role: role, blocks: content.compactMap(AnthropicConverter.block(fromJSON:)))
        }
    }

    /// The body the old inline chat path built: the stored conversation
    /// verbatim, with the flat web-search tool.
    private func oldBody(_ conversation: [[String: Any]]) -> [String: Any] {
        [
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "tools": [["type": "web_search_20250305", "name": "web_search", "max_uses": 5]],
            "system": "Tu es Mochi.",
            "messages": conversation,
        ]
    }

    /// The body ChatPolicy builds through the engine.
    private func newBody(_ conversation: [[String: Any]]) -> [String: Any] {
        let request = CompletionRequest(
            system: "Tu es Mochi.",
            messages: chatPolicyHistory(conversation),
            tools: [ToolDeclaration(name: "web_search", description: "",
                                    inputSchema: ["max_uses": .number(5)],
                                    serverType: "web_search_20250305")],
            maxTokens: 4096,
            betas: ["web-search-2025-03-05"],
            timeout: 45)
        return AnthropicConverter.requestBody(request, model: "claude-sonnet-4-5")
    }

    /// The red line: same conversation in, same canonical bytes out. The
    /// assistant turns replay through rawBlocksJSON (citations,
    /// encrypted_index, tool_use_id untouched), the user turns rebuild
    /// image/document/text blocks in the exact stored form.
    func testChatBodyByteParity() throws {
        let conversation = try storedConversation()
        let old = try XCTUnwrap(canonicalJSONString(from: oldBody(conversation)))
        let new = try XCTUnwrap(canonicalJSONString(from: newBody(conversation)))
        XCTAssertEqual(old, new)
    }

    /// Every message of the conversation survives — nothing dropped, order
    /// kept, roles kept.
    func testHistoryKeepsEveryTurn() throws {
        let messages = chatPolicyHistory(try storedConversation())
        XCTAssertEqual(messages.map(\.role), [.user, .assistant, .user, .assistant])
        // First user turn: document + image + two texts.
        XCTAssertEqual(messages[0].blocks.count, 4)
        // Web-search assistant turn: text + server_tool_use + result + 2 cited texts.
        XCTAssertEqual(messages[1].blocks.count, 5)
        XCTAssertEqual(messages[2].blocks, [.text("Merci !")])
    }
}
