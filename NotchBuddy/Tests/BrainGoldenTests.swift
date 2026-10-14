import XCTest

// Golden tests, two tiers:
// - byte tier (every category — frozen input + pure functions are
//   deterministic): the converter's canonical serialization compared
//   byte-for-byte with a .golden file. P2's acceptance is "old path and new
//   path eat the same fixture and produce the same .golden".
// - shape tier (web-search only — re-recording against the live API drifts
//   in text and ids): FixtureNormalizer's structural fingerprint against a
//   .shape file.
// Missing goldens are recorded in place, next to this file's fixtures dir,
// and compared from the next build on.

final class BrainGoldenTests: XCTestCase {

    // MARK: Fixtures

    private func fixtureData(_ name: String, file: StaticString = #filePath, line: UInt = #line) throws -> Data {
        let base = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        guard let url = Bundle(for: Self.self).url(forResource: base, withExtension: ext) else {
            XCTFail("fixture \(name) not found in the test bundle", file: file, line: line)
            throw CocoaError(.fileNoSuchFile)
        }
        return try Data(contentsOf: url)
    }

    /// Records a missing golden into the source fixtures dir (next to this
    /// test file), or compares the converter output byte-for-byte with the
    /// bundled one. Recording is deterministic — the input is frozen — so
    /// re-recording writes the same bytes.
    private func assertGolden(_ value: String, _ name: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let base = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        if let url = Bundle(for: Self.self).url(forResource: base, withExtension: ext) {
            let expected = try String(contentsOf: url, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertEqual(expected, value, "golden \(name) mismatch", file: file, line: line)
            return
        }
        let dir = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        try (value + "\n").data(using: .utf8)!.write(to: dir.appendingPathComponent(name))
        print("RECORDED \(name) (\(value.utf8.count) bytes)")
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try XCTUnwrap(try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      "fixture is not a JSON object")
    }

    // MARK: Byte tier — response round-trips

    /// Parse each recorded response, replay it as an assistant message
    /// (rawBlocksJSON), serialize it back: the wire content must survive
    /// byte-for-byte (in canonical key order). This is the round-trip
    /// fidelity red line: encrypted_index, citations, tool_use_id and the
    /// error-object variant all live here.
    func testAnthropicResponseRoundTrips() throws {
        let fixtures = [
            "anthropic-text",
            "anthropic-tool-use",
            "anthropic-web-search-normal",
            "anthropic-web-search-pause",
            "anthropic-web-search-error",
        ]
        for fixture in fixtures {
            let json = try json(try fixtureData("\(fixture).json"))
            let content = try XCTUnwrap(json["content"] as? [[String: Any]],
                                        "\(fixture): no content array")
            let message = AnthropicConverter.assistantMessage(fromContentJSON: content)
            let wire = AnthropicConverter.messageJSON(message)
            let canonical = try XCTUnwrap(canonicalJSONString(from: wire["content"]!))
            try assertGolden(canonical, "\(fixture).roundtrip.golden")
        }
    }

    // MARK: Byte tier — request goldens (7 block kinds × 2 dialects)

    /// One canonical request exercising every block kind in both dialects:
    /// the OpenAI side shows its pinned degradations (first text block only,
    /// images and documents dropped, server tool blocks skipped,
    /// toolResult → role:"tool") and the tool declaration forms (client:
    /// name/description/input_schema; server: flat max_uses + user_location).
    func testRequestGoldens() throws {
        let messages: [BrainMessage] = [
            BrainMessage(role: .user, blocks: [
                .text("Voici le contexte."),
                .image(mediaType: "image/png", base64: "aW1hZ2U="),
                .document(mediaType: "application/pdf", base64: "cGRm"),
            ]),
            BrainMessage(role: .assistant, blocks: [
                .text("Je regarde l'image."),
                .toolUse(id: "toolu_fixture_01", name: "get_weather",
                         inputJSON: #"{"city":"Paris"}"#),
            ]),
            BrainMessage(role: .user, blocks: [
                .toolResult(id: "toolu_fixture_01", content: "18°C, pluie fine"),
            ]),
            BrainMessage(role: .assistant, blocks: [
                .serverToolUse(id: "srvtoolu_fixture_01", name: "web_search",
                               inputJSON: #"{"query":"weather Paris"}"#),
                .webSearchToolResult(id: "srvtoolu_fixture_01",
                                     rawJSON: #"{"content":[],"tool_use_id":"srvtoolu_fixture_01","type":"web_search_tool_result"}"#),
                .text("Il pleut."),
            ]),
        ]
        let request = CompletionRequest(
            system: "Tu es Coucou.",
            messages: messages,
            tools: [
                ToolDeclaration(name: "get_weather", description: "Météo d'une ville",
                                inputSchema: [
                                    "type": .string("object"),
                                    "properties": .object(["city": .object(["type": .string("string")])]),
                                    "required": .array([.string("city")]),
                                ]),
                ToolDeclaration(name: "web_search", description: "Web search",
                                inputSchema: ["type": .string("web_search_20250305"),
                                              "max_uses": .number(3),
                                              "user_location": .object(["type": .string("approximate"),
                                                                        "timezone": .string("Europe/Paris")])],
                                serverType: "web_search_20250305"),
            ],
            temperature: 0.7,
            timeout: 45)

        let anthropic = AnthropicConverter.requestBody(request, model: "claude-sonnet-4-5")
        try assertGolden(try XCTUnwrap(canonicalJSONString(from: anthropic)), "request-anthropic.golden")

        let openai = OpenAIConverter.requestBody(request, model: "sgl/glm-5.3-flash")
        try assertGolden(try XCTUnwrap(canonicalJSONString(from: openai)), "request-openai.golden")
    }

    // MARK: Shape tier — web-search structural fingerprints

    func testWebSearchShapes() throws {
        let fixtures = [
            "anthropic-web-search-normal",
            "anthropic-web-search-pause",
            "anthropic-web-search-error",
        ]
        for fixture in fixtures {
            let json = try json(try fixtureData("\(fixture).json"))
            let shape = try XCTUnwrap(FixtureNormalizer.shape(ofResponseJSON: json),
                                      "\(fixture): shape failed")
            try assertGolden(shape, "\(fixture).shape")
        }
    }

    // MARK: SSE replay

    /// The recorded real stream: reasoning_content deltas (content null) are
    /// ignored, empty-string content survives, the tail chunk carries
    /// finish_reason "stop", and nothing from reasoning leaks into deltas.
    func testSSEReplayRecordedStream() throws {
        let lines = try String(data: fixtureData("openai-sse-stream.txt"), encoding: .utf8)!
            .components(separatedBy: .newlines)
        var deltas: [String] = []
        var finishReasons: [String] = []
        for line in lines {
            guard let event = OpenAIConverter.parseSSEEvent(line) else { continue }
            if let text = event.text { deltas.append(text) }
            if let finish = event.finishReason { finishReasons.append(finish) }
        }
        let joined = deltas.joined()
        XCTAssertFalse(joined.isEmpty)
        XCTAssertEqual(finishReasons, ["stop"])
        // No think tags in this stream: the joined deltas are the final text.
        XCTAssertEqual(OpenAIConverter.filterThinkingBlocks(joined), joined)
        XCTAssertFalse(joined.contains("user asks"), "reasoning_content leaked into deltas")
    }

    /// The fake-local event set (think block split across two chunks,
    /// reasoning-only delta, [DONE]): progressiveFilter hides the open think
    /// block while it streams; the final text is the expected answer.
    func testSSEReplayFakeLocalThinkBlocks() throws {
        let expected = """
        ## Answer

        Here is the result:

        - **item 1**
        - item 2

        ```python
        print('hello')
        ```
        """
        let lines = try String(data: fixtureData("openai-sse-fake-local.txt"), encoding: .utf8)!
            .components(separatedBy: .newlines)
        var accumulated = ""
        var visibleWhileOpen = ""
        var sawThink = false
        var capturedWhileOpen = false
        var eventCount = 0
        for line in lines {
            guard let event = OpenAIConverter.parseSSEEvent(line) else { continue }
            eventCount += 1
            guard let text = event.text else { continue }
            accumulated += text
            if text.contains("<think>") { sawThink = true }
            // Capture the visible text right after the first think chunk —
            // while the block is still open, nothing of it may show.
            if sawThink && !capturedWhileOpen {
                visibleWhileOpen = OpenAIConverter.progressiveFilter(accumulated)
                capturedWhileOpen = true
            }
        }
        // 7 events with content: the reasoning-only delta and [DONE] parse to nil.
        XCTAssertEqual(eventCount, 7)
        XCTAssertEqual(visibleWhileOpen, "")
        XCTAssertEqual(OpenAIConverter.filterThinkingBlocks(accumulated), expected)
        XCTAssertEqual(OpenAIConverter.progressiveFilter(accumulated), expected)
    }

    /// OAI tool-call deltas merge by index into the assembled call.
    func testSSEToolCallAccumulation() {
        let lines = [
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_\",\"arguments\":\"\"}}]}}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]}}]}",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
            "data: [DONE]",
        ]
        var accumulated = ""
        var toolCalls: [Int: [String: String]] = [:]
        var finish: String?
        for line in lines {
            guard let event = OpenAIConverter.parseSSEEvent(line) else { continue }
            if let text = event.text { accumulated += text }
            if let call = event.toolCall {
                var parts = toolCalls[call.index] ?? [:]
                if let id = call.id { parts["id"] = id }
                if let name = call.name { parts["name"] = (parts["name"] ?? "") + name }
                if let arguments = call.arguments { parts["arguments"] = (parts["arguments"] ?? "") + arguments }
                toolCalls[call.index] = parts
            }
            if let f = event.finishReason { finish = f }
        }
        XCTAssertEqual(accumulated, "")
        XCTAssertEqual(toolCalls[0]?["id"], "call_1")
        XCTAssertEqual(toolCalls[0]?["name"], "get_weather")
        XCTAssertEqual(toolCalls[0]?["arguments"], #"{"city":"Paris"}"#)
        XCTAssertEqual(finish, "tool_calls")
    }
}
