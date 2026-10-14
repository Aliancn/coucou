import XCTest

// Spike B: a @MainActor tool class conforming to ClientTool the way the App
// layer will (nonisolated let declaration — SE-0434). If this stops compiling
// under Swift 6 strict concurrency, the plan falls back to option (a): make
// the loop and the engine @MainActor wholesale.

@MainActor
final class SpikeMainActorTool: ClientTool {
    nonisolated let declaration = ToolDeclaration(
        name: "spike",
        description: "Spike tool",
        inputSchema: ["type": .string("object")]
    )

    func execute(call: ToolCall) async throws -> ToolOutcome {
        ToolOutcome(text: "ran \(call.name)", acted: true)
    }
}

final class BrainToolSpikeTests: XCTestCase {
    func testMainActorToolConformsToClientTool() async throws {
        let tool: any ClientTool = SpikeMainActorTool()
        XCTAssertEqual(tool.declaration.name, "spike")
        let outcome = try await tool.execute(call: ToolCall(id: "1", name: "spike", inputJSON: "{}"))
        XCTAssertTrue(outcome.acted)
        XCTAssertEqual(outcome.text, "ran spike")
    }
}
