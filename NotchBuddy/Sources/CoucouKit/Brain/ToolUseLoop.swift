import Foundation

// The generic tool-use loop, semantics lifted from the voice brain's
// hand-rolled implementation:
// - client toolUse blocks execute one by one, each appending a toolResult to
//   a new user message; serverToolUse blocks never get a tool_result
// - pauseTurn resumes without consuming a round (long web-search middle
//   states), bounded so a pathological always-pausing server still ends
// - history trimming does NOT live here (the 24-message cap is a voice-policy
//   choice; chat never trims)
// - on exit the history is legal for the next request: a dangling assistant
//   tool-use turn (no result after it) is dropped
// The loop only calls complete(); intermediate rounds never emit deltas.

/// Runs the request through the client, executing client tools as the model
/// calls them, until a turn without tool use or the round budget is spent.
/// Server-tool declarations travel in `request.tools`; client tools are
/// passed here so their declarations join the request automatically and
/// their outcomes surface in the result.
///
/// - Returns: the last response, the extended history (assistant turns and
///   tool results included, safe to feed the next request), and whether any
///   tool changed something (the "C'est fait." signal).
func toolUseLoop(
    _ request: CompletionRequest,
    client: any LLMClient,
    clientTools: [any ClientTool] = [],
    maxRounds: Int = 5
) async throws -> (response: CompletionResponse, messages: [BrainMessage], acted: Bool) {
    var history = request.messages
    var req = request
    req.tools += clientTools.map(\.declaration)
    var toolsByName: [String: any ClientTool] = [:]
    for tool in clientTools { toolsByName[tool.declaration.name] = tool }

    var consumedRounds = 0
    var totalCalls = 0
    // pause_turn rounds are free but bounded (at most maxRounds of them), so
    // an always-pausing server cannot loop forever.
    let maxCalls = maxRounds * 2
    var acted = false
    var lastResponse: CompletionResponse?

    while consumedRounds < maxRounds && totalCalls < maxCalls {
        totalCalls += 1
        req.messages = history
        let response: CompletionResponse
        do {
            response = try await client.complete(req)
        } catch {
            repairHistory(&history)
            throw error
        }
        history.append(BrainMessage(role: .assistant, blocks: response.blocks,
                                    rawBlocksJSON: response.rawBlocksJSON))
        lastResponse = response
        // Long web search: go on with the same round budget.
        if response.stopReason == .pauseTurn { continue }

        let calls = response.blocks.compactMap { block -> ToolCall? in
            guard case .toolUse(let id, let name, let inputJSON) = block else { return nil }
            return ToolCall(id: id, name: name, inputJSON: inputJSON)
        }
        guard response.stopReason == .toolUse, !calls.isEmpty else { break }
        consumedRounds += 1

        var results: [BrainMessage.Block] = []
        for call in calls {
            let outcome: ToolOutcome
            if let tool = toolsByName[call.name] {
                do {
                    outcome = try await tool.execute(call: call)
                } catch {
                    // A failing tool must not break the history: the model
                    // gets the error as its result and can recover.
                    outcome = ToolOutcome(text: "error: \(error.localizedDescription)", acted: false)
                }
            } else {
                outcome = ToolOutcome(text: "unknown tool \(call.name)", acted: false)
            }
            if outcome.acted { acted = true }
            results.append(.toolResult(id: call.id, content: outcome.text))
        }
        history.append(BrainMessage(role: .user, blocks: results))
    }

    repairHistory(&history)
    guard let response = lastResponse else { throw BrainError.emptyResponse }
    return (response, history, acted)
}

/// The API refuses a history whose last assistant turn calls a tool with no
/// result after it (cut by max_tokens, a failure, or a pause on the last
/// round): drop that turn. A web_search_tool_result counts as the answer for
/// a server tool use.
private func repairHistory(_ history: inout [BrainMessage]) {
    guard let last = history.last, last.role == .assistant else { return }
    var hasClientCall = false
    var hasServerCall = false
    var hasSearchResult = false
    for block in last.blocks {
        switch block {
        case .toolUse: hasClientCall = true
        case .serverToolUse: hasServerCall = true
        case .webSearchToolResult: hasSearchResult = true
        default: break
        }
    }
    if hasClientCall || (hasServerCall && !hasSearchResult) {
        history.removeLast()
    }
}
