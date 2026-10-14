import XCTest

// Table-driven tool-use loop tests, each case lifted line-by-line from the
// voice brain's hand-rolled implementation so P1 pins today's semantics and
// P3b's rewiring is guarded by the same batch. Mocks are plain non-isolated
// types — the payoff of the actor-neutral ClientTool design.

// MARK: - Mocks

/// Serves a scripted sequence of results, recording every request it got.
final class MockClient: LLMClient, @unchecked Sendable {
    let endpoint = ModelEndpoint(baseURL: "https://mock.example/v1",
                                 dialect: .anthropicMessages, apiKey: nil, model: "mock-model")

    private let lock = NSLock()
    private var script: [Result<CompletionResponse, Error>] = []
    private(set) var requests: [CompletionRequest] = []

    init(script: [CompletionResponse] = [], error: (any Error)? = nil) {
        self.script = error.map { [.failure($0)] } ?? script.map { .success($0) }
    }

    init(results: [Result<CompletionResponse, Error>]) {
        self.script = results
    }

    func complete(_ request: CompletionRequest) async throws -> CompletionResponse {
        try lock.withLock {
            requests.append(request)
            guard !script.isEmpty else { throw BrainError.emptyResponse }
            return try script.removeFirst().get()
        }
    }

    func stream(_ request: CompletionRequest) -> AsyncThrowingStream<BrainEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: BrainError.network("not implemented")) }
    }

    func listModels() async throws -> [ModelListing] { [] }

    var callCount: Int { lock.withLock { requests.count } }
}

/// A client tool with scriptable behaviour.
struct MockTool: ClientTool {
    let declaration = ToolDeclaration(name: "mock", description: "Mock tool", inputSchema: [:])
    var acted = false
    var result = "mock done"
    var throwsOnCall = false

    func execute(call: ToolCall) async throws -> ToolOutcome {
        if throwsOnCall { throw BrainError.network("boom") }
        return ToolOutcome(text: result, acted: acted)
    }
}

// MARK: - Tests

final class BrainLoopTests: XCTestCase {

    private func response(_ blocks: [BrainMessage.Block], _ stop: StopReason,
                          rawJSON: String? = nil) -> CompletionResponse {
        CompletionResponse(blocks: blocks, stopReason: stop, usage: nil, model: "mock-model",
                           rawBlocksJSON: rawJSON)
    }

    private func request(_ messages: [BrainMessage]) -> CompletionRequest {
        CompletionRequest(system: "s", messages: messages, timeout: 5)
    }

    private let question = BrainMessage(role: .user, blocks: [.text("Mets la musique.")])

    // MARK: pause_turn

    /// pause_turn resumes without consuming a round: a long web search passes
    /// through without ever spending the tool budget. (The original burned an
    /// iteration on it.)
    func testPauseTurnDoesNotConsumeARound() async throws {
        let pause = response([.serverToolUse(id: "s1", name: "web_search",
                                             inputJSON: #"{"query":"x"}"#)], .pauseTurn)
        let client = MockClient(script: [pause, pause, response([.text("Voilà.")], .endTurn)])
        let result = try await toolUseLoop(request([question]), client: client, maxRounds: 2)
        XCTAssertEqual(client.callCount, 3)
        XCTAssertEqual(result.response.blocks, [.text("Voilà.")])
        XCTAssertEqual(result.messages.count, 4)   // question + 2 pause turns + final
        XCTAssertFalse(result.acted)
    }

    /// But a pathological always-pausing server still terminates: calls are
    /// bounded at maxRounds * 2 even when no round is ever consumed, and the
    /// dangling last pause turn is dropped on exit (only the last — the
    /// original repair drops just one turn too).
    func testAlwaysPausingServerTerminates() async throws {
        let pause = response([.serverToolUse(id: "s1", name: "web_search", inputJSON: "{}")],
                             .pauseTurn)
        let client = MockClient(script: Array(repeating: pause, count: 99))
        let result = try await toolUseLoop(request([question]), client: client, maxRounds: 2)
        XCTAssertEqual(client.callCount, 4)   // maxRounds * 2
        XCTAssertEqual(result.messages.count, 4)   // question + 3 pauses, last dropped
    }

    // MARK: Round budget

    /// The model keeps calling tools: the budget is spent, the last response
    /// is pure tools (the caller sees an empty spoken text), and the history
    /// ends with the tool results — legal for the next request.
    func testMaxRoundsSpentOnPureToolTurns() async throws {
        let toolTurn = response([.toolUse(id: "t1", name: "mock", inputJSON: "{}")], .toolUse)
        let client = MockClient(script: [toolTurn, toolTurn])
        let result = try await toolUseLoop(request([question]), client: client, clientTools: [MockTool()],
                                           maxRounds: 2)
        XCTAssertEqual(client.callCount, 2)
        XCTAssertEqual(result.response.stopReason, .toolUse)
        XCTAssertFalse(result.response.blocks.contains { if case .text = $0 { true } else { false } })
        XCTAssertEqual(result.messages.count, 5)   // question + 2 × (assistant, results)
        guard case .toolResult(let id, let content) = result.messages.last?.blocks.first else {
            return XCTFail("history must end with tool results")
        }
        XCTAssertEqual(id, "t1")
        XCTAssertEqual(content, "mock done")
    }

    // MARK: Server tools

    /// A server tool use never gets a tool_result: the web search runs on
    /// Anthropic's side, the result comes back as blocks of the same turn.
    func testServerToolUseGetsNoToolResult() async throws {
        let turn = response([
            .serverToolUse(id: "s1", name: "web_search", inputJSON: #"{"query":"x"}"#),
            .webSearchToolResult(id: "s1", rawJSON: #"{"content":[],"tool_use_id":"s1","type":"web_search_tool_result"}"#),
            .text("Trouvé."),
        ], .endTurn)
        let client = MockClient(script: [turn])
        let result = try await toolUseLoop(request([question]), client: client)
        XCTAssertEqual(client.callCount, 1)
        XCTAssertEqual(result.messages.count, 2)   // question + assistant turn, nothing appended
    }

    /// A web_search_tool_result counts as "answered" when repairing: an
    /// assistant turn whose server tool use came back with its result stays
    /// in the history (contrast: testDanglingServerToolUseIsDropped).
    func testWebSearchResultCountsAsAnsweredInRepair() async throws {
        let turn = response([
            .serverToolUse(id: "s1", name: "web_search", inputJSON: #"{"query":"x"}"#),
            .webSearchToolResult(id: "s1", rawJSON: #"{"content":[],"tool_use_id":"s1","type":"web_search_tool_result"}"#),
        ], .endTurn)
        let client = MockClient(script: [turn])
        let result = try await toolUseLoop(request([question]), client: client)
        XCTAssertEqual(result.messages.count, 2)   // question + answered assistant turn kept
    }

    // MARK: History repair

    /// An assistant turn whose server tool use never got its search result is
    /// dropped on exit: the next request would be refused with it.
    func testDanglingServerToolUseIsDropped() async throws {
        let turn = response([.serverToolUse(id: "s1", name: "web_search", inputJSON: "{}")],
                            .endTurn)
        let client = MockClient(script: [turn])
        let result = try await toolUseLoop(request([question]), client: client)
        XCTAssertEqual(result.messages, [question])   // dangling assistant turn dropped
    }

    /// A tool_use block with stop_reason end_turn (the call was cut by
    /// max_tokens, or the model gave up mid-call): the dangling assistant
    /// turn is dropped — the next request would be refused with it.
    func testDanglingClientToolUseIsDropped() async throws {
        let turn = response([.toolUse(id: "t1", name: "mock", inputJSON: "{}")], .endTurn)
        let client = MockClient(script: [turn])
        let result = try await toolUseLoop(request([question]), client: client)
        XCTAssertEqual(result.messages, [question])
    }

    /// Trimming does NOT live in the loop: the 24-message cap is the voice
    /// policy's choice; the loop never cuts history.
    func testLoopNeverTrims() async throws {
        let longHistory = (0..<30).map { i in
            BrainMessage(role: i.isMultiple(of: 2) ? .user : .assistant, blocks: [.text("m\(i)")])
        }
        let client = MockClient(script: [response([.text("ok")], .endTurn)])
        let result = try await toolUseLoop(request(longHistory), client: client)
        XCTAssertEqual(result.messages.count, 31)
    }

    // MARK: Tools

    /// A tool that changed something sets the acted flag — the "C'est fait."
    /// signal the voice reply needs.
    func testActedPropagates() async throws {
        let toolTurn = response([.toolUse(id: "t1", name: "mock", inputJSON: "{}")], .toolUse)
        let client = MockClient(script: [toolTurn, response([.text("C'est fait.")], .endTurn)])
        let tool = MockTool(acted: true)
        let result = try await toolUseLoop(request([question]), client: client, clientTools: [tool])
        XCTAssertTrue(result.acted)
        XCTAssertEqual(result.response.blocks, [.text("C'est fait.")])
    }

    /// An unknown tool (the model hallucinated a name): the result says so,
    /// nothing acted, and the loop carries on with the next round.
    func testUnknownTool() async throws {
        let toolTurn = response([.toolUse(id: "t1", name: "nope", inputJSON: "{}")], .toolUse)
        let client = MockClient(script: [toolTurn, response([.text("Je ne peux pas.")], .endTurn)])
        let result = try await toolUseLoop(request([question]), client: client,
                                           clientTools: [MockTool()])
        XCTAssertFalse(result.acted)
        XCTAssertEqual(result.messages[2].blocks,
                       [.toolResult(id: "t1", content: "unknown tool nope")])
        XCTAssertEqual(result.response.blocks, [.text("Je ne peux pas.")])
    }

    /// A throwing tool must not break the history: the model gets the error
    /// as its result and can recover on the next round.
    func testThrowingToolRecovers() async throws {
        let toolTurn = response([.toolUse(id: "t1", name: "mock", inputJSON: "{}")], .toolUse)
        let client = MockClient(script: [toolTurn, response([.text("Raté, je réessaie.")], .endTurn)])
        let tool = MockTool(throwsOnCall: true)
        let result = try await toolUseLoop(request([question]), client: client, clientTools: [tool])
        guard case .toolResult(_, let content) = result.messages[2].blocks.first else {
            return XCTFail("expected a tool result")
        }
        XCTAssertTrue(content.hasPrefix("error:"))
        XCTAssertEqual(result.messages.count, 4)
    }

    /// The loop returns the full extended history — assistant turns with
    /// their raw wire form, tool results in between — safe to feed back for
    /// the voice conversation's next turn.
    func testExtendedHistoryIsReturned() async throws {
        let toolTurn = response([.toolUse(id: "t1", name: "mock", inputJSON: #"{"a":1}"#)],
                                .toolUse, rawJSON: #"[]"#)
        let client = MockClient(script: [toolTurn, response([.text("Fait.")], .endTurn)])
        let result = try await toolUseLoop(request([question]), client: client, clientTools: [MockTool()])
        XCTAssertEqual(result.messages.count, 4)
        XCTAssertEqual(result.messages[1].role, .assistant)
        XCTAssertEqual(result.messages[1].rawBlocksJSON, #"[]"#)
        XCTAssertEqual(result.messages[2].blocks, [.toolResult(id: "t1", content: "mock done")])
        XCTAssertEqual(result.messages[3].blocks, [.text("Fait.")])
    }

    /// Client tool declarations join the request automatically; server tool
    /// declarations travel in request.tools — the loop must not lose either.
    func testToolDeclarationsAreMergedIntoTheRequest() async throws {
        let serverTool = ToolDeclaration(name: "web_search", description: "",
                                         inputSchema: [:], serverType: "web_search_20250305")
        var req = request([question])
        req.tools = [serverTool]
        let client = MockClient(script: [response([.text("ok")], .endTurn)])
        _ = try await toolUseLoop(req, client: client, clientTools: [MockTool()])
        let names = client.requests.first?.tools.map(\.name) ?? []
        XCTAssertEqual(Set(names), ["web_search", "mock"])
    }
}
