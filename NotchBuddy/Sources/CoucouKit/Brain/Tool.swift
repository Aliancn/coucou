import Foundation

// Tool protocols. The engine stays actor-neutral: concrete client tools are
// @MainActor classes in the App layer (Swift 6 lets an actor-isolated async
// method witness a nonisolated async protocol requirement; call sites hop
// implicitly), so the loop can run anywhere while tools touch UI frameworks.
// `declaration` must be implemented as `nonisolated let` (SE-0434) so the
// requirement witnesses synchronously.

/// A tool executed by the provider itself (e.g. Anthropic web search).
/// The engine only declares it; results come back as
/// webSearchToolResult blocks kept verbatim for the next round-trip.
protocol ServerTool: Sendable {
    var declaration: ToolDeclaration { get }
}

/// A tool the app executes locally when the model calls it.
/// Concrete tools never return nil — failure is a thrown BrainError.
protocol ClientTool: Sendable {
    var declaration: ToolDeclaration { get }
    func execute(call: ToolCall) async throws -> ToolOutcome
}
