import Foundation

// The OpenAI chat-completions dialect, streaming over SSE. Deltas are emitted
// raw — the policy accumulates and throttles; thinking is filtered here
// (filterThinkingBlocks) for the final turn. apiKey == nil sends no
// Authorization header (Ollama and friends accept anonymous requests).

struct OpenAIClient: LLMClient {
    let endpoint: ModelEndpoint

    func complete(_ request: CompletionRequest) async throws -> CompletionResponse {
        let response = try await roundTrip(request, stream: false)
        guard let parsed = OpenAIConverter.response(fromJSON: response) else {
            throw BrainError.emptyResponse
        }
        return parsed
    }

    func stream(_ request: CompletionRequest) -> AsyncThrowingStream<BrainEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let json = try await self.roundTrip(request, stream: true) { delta in
                        continuation.yield(.delta(delta))
                    }
                    guard let parsed = OpenAIConverter.response(fromJSON: json) else {
                        throw BrainError.emptyResponse
                    }
                    continuation.yield(.turn(parsed))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func listModels() async throws -> [ModelListing] {
        guard let url = URL(string: endpoint.baseURL + "/models") else {
            throw BrainError.network("bad base URL: \(endpoint.baseURL)")
        }
        var req = URLRequest(url: url, timeoutInterval: 10)
        if let apiKey = endpoint.apiKey { req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        let data = try await send(req, model: endpoint.model)
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["data"] as? [[String: Any]] else { throw BrainError.emptyResponse }
        return items.compactMap { item in
            guard let id = item["id"] as? String else { return nil }
            return ModelListing(id: id, created: item["created"] as? Int)
        }
    }

    // MARK: Plumbing

    /// One POST /chat/completions. Streaming calls `onDelta` with each raw
    /// text delta and returns the parsed JSON of the assembled final
    /// response (content, tool_calls, finish_reason merged from the deltas).
    private func roundTrip(_ request: CompletionRequest, stream: Bool,
                           onDelta: ((String) -> Void)? = nil) async throws -> [String: Any] {
        var body = OpenAIConverter.requestBody(request, model: endpoint.model)
        if stream { body["stream"] = true }
        guard let url = URL(string: endpoint.baseURL + "/chat/completions") else {
            throw BrainError.network("bad base URL: \(endpoint.baseURL)")
        }
        var req = URLRequest(url: url, timeoutInterval: request.timeout)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "content-type")
        if let apiKey = endpoint.apiKey { req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        guard stream else {
            let data = try await send(req, model: endpoint.model)
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw BrainError.emptyResponse
            }
            return json
        }

        // SSE: accumulate text deltas, tool-call fragments and the finish
        // reason, then hand back one response-shaped JSON object.
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await URLSession.shared.bytes(for: req)
        } catch let error as URLError where error.code == .cancelled {
            throw BrainError.cancelled
        } catch {
            throw BrainError.network("Cannot reach \(endpoint.baseURL). Is the server running?")
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status == 404 { throw BrainError.modelNotFound(provider: providerName, model: endpoint.model) }
        guard status == 200 else {
            var raw = ""
            do { for try await byte in bytes { raw.append(Character(UnicodeScalar(byte))); if raw.count > 4096 { break } } } catch {}
            if let data = raw.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let message = (json["error"] as? [String: Any])?["message"] as? String {
                throw BrainError.http(status: status, body: message)
            }
            throw BrainError.http(status: status, body: "HTTP \(status)")
        }

        var accumulated = ""
        var toolCalls: [Int: [String: String]] = [:]   // index → [id, name, arguments]
        var finishReason: String?
        do {
            for try await line in bytes.lines {
                guard let event = OpenAIConverter.parseSSEEvent(line) else { continue }
                if let text = event.text { accumulated += text; onDelta?(text) }
                if let call = event.toolCall {
                    var parts = toolCalls[call.index] ?? [:]
                    if let id = call.id { parts["id"] = id }
                    if let name = call.name { parts["name"] = (parts["name"] ?? "") + name }
                    if let arguments = call.arguments { parts["arguments"] = (parts["arguments"] ?? "") + arguments }
                    toolCalls[call.index] = parts
                }
                if let finish = event.finishReason { finishReason = finish }
            }
        } catch let error as URLError where error.code == .cancelled {
            throw BrainError.cancelled
        } catch {
            throw BrainError.network("Cannot reach \(endpoint.baseURL). Is the server running?")
        }

        var message: [String: Any] = ["content": OpenAIConverter.filterThinkingBlocks(accumulated)]
        if !toolCalls.isEmpty {
            let calls = toolCalls
                .sorted(by: { $0.key < $1.key })
                .compactMap { entry -> [String: Any]? in
                    let parts = entry.value
                    guard let id = parts["id"], let name = parts["name"] else { return nil }
                    return ["id": id, "type": "function",
                            "function": ["name": name, "arguments": parts["arguments"] ?? "{}"]]
                }
            if !calls.isEmpty { message["tool_calls"] = calls }
        }
        return ["choices": [["message": message,
                             "finish_reason": finishReason ?? "stop"] as [String: Any]]]
    }

    private func send(_ req: URLRequest, model: String) async throws -> Data {
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: req)
        } catch let error as URLError where error.code == .cancelled {
            throw BrainError.cancelled
        } catch {
            throw BrainError.network("Cannot reach \(endpoint.baseURL). Is the server running?")
        }
        guard let http = response as? HTTPURLResponse else { throw BrainError.network("no response") }
        guard http.statusCode == 200 else {
            if http.statusCode == 404 {
                throw BrainError.modelNotFound(provider: providerName, model: model)
            }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let message = (json["error"] as? [String: Any])?["message"] as? String {
                throw BrainError.http(status: http.statusCode, body: message)
            }
            throw BrainError.http(status: http.statusCode,
                                  body: String(data: data, encoding: .utf8) ?? "unknown error")
        }
        return data
    }

    private var providerName: String {
        URL(string: endpoint.baseURL)?.host() ?? endpoint.baseURL
    }
}
