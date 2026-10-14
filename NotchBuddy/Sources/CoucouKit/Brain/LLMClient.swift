import Foundation

// The one primitive every dialect implements. Policies build canonical
// CompletionRequests and consume BrainEvents; nothing above this protocol
// knows which wire format the endpoint speaks.

/// A fully resolved LLM endpoint speaking exactly one dialect.
protocol LLMClient: Sendable {
    var endpoint: ModelEndpoint { get }

    /// One non-streaming round-trip. The tool-use loop and every
    /// tool-carrying call go through here.
    func complete(_ request: CompletionRequest) async throws -> CompletionResponse

    /// Streaming text. Deltas are raw (the policy accumulates and throttles);
    /// the final .turn carries the filtered response. Anthropic streams in
    /// P4 — its client currently degrades to one complete() call.
    func stream(_ request: CompletionRequest) -> AsyncThrowingStream<BrainEvent, Error>

    /// Model ids as the provider lists them. Filtering heuristics (embed,
    /// rerank…) live in the App layer's catalog, not here.
    func listModels() async throws -> [String]
}
