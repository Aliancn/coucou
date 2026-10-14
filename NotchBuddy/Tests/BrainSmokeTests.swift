import XCTest

// Spike A smoke test: proves the BrainTests target builds, runs and sees the
// Brain sources (they compile straight into this module — no @testable import
// of the app). Replaced by the real suites once the engine lands.

final class BrainSmokeTests: XCTestCase {
    func testDialectRawValues() {
        XCTAssertEqual(Dialect.anthropicMessages.rawValue, "anthropicMessages")
        XCTAssertEqual(Dialect.openaiChat.rawValue, "openaiChat")
    }

    func testClaudeResponseTextJoinsBlocksWithoutSeparator() {
        let content: [[String: Any]] = [
            ["type": "text", "text": "Voici"],
            ["type": "web_search_tool_result", "content": []],
            ["type": "text", "text": " la réponse."],
        ]
        XCTAssertEqual(claudeResponseText(fromContent: content), "Voici la réponse.")
    }
}
