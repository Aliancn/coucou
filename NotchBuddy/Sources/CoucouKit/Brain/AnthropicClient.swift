import Foundation

// The Messages API dialect, non-streaming for now (SSE arrives later; the
// streaming contract already exists through LLMClient.stream). Credentials
// arrive fully resolved in ModelEndpoint — this file never touches the
// Keychain.

struct AnthropicClient: LLMClient {
    let endpoint: ModelEndpoint

    private let version = "2023-06-01"

    func complete(_ request: CompletionRequest) async throws -> CompletionResponse {
        let url = try requestURL(path: "/messages")
        var req = URLRequest(url: url, timeoutInterval: request.timeout)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "content-type")
        req.setValue(version, forHTTPHeaderField: "anthropic-version")
        if let apiKey = endpoint.apiKey { req.setValue(apiKey, forHTTPHeaderField: "x-api-key") }
        if !request.betas.isEmpty {
            req.setValue(request.betas.joined(separator: ","), forHTTPHeaderField: "anthropic-beta")
        }
        req.httpBody = try JSONSerialization.data(
            withJSONObject: AnthropicConverter.requestBody(request, model: endpoint.model))

        let data = try await send(req)
        return try parseResponse(data)
    }

    /// Streaming is P4 (SSE + input_json_delta accumulation). Until then the
    /// stream degrades to one complete round-trip — deltas are simply absent.
    func stream(_ request: CompletionRequest) -> AsyncThrowingStream<BrainEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    continuation.yield(.turn(try await self.complete(request)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func listModels() async throws -> [String] {
        let url = try requestURL(path: "/models?limit=100")
        var req = URLRequest(url: url, timeoutInterval: 10)
        req.setValue(version, forHTTPHeaderField: "anthropic-version")
        if let apiKey = endpoint.apiKey { req.setValue(apiKey, forHTTPHeaderField: "x-api-key") }
        let data = try await send(req)
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["data"] as? [[String: Any]] else { throw BrainError.emptyResponse }
        return items.compactMap { $0["id"] as? String }
    }

    // MARK: Plumbing

    private func requestURL(path: String) throws -> URL {
        guard let url = URL(string: endpoint.baseURL + path) else {
            throw BrainError.network("bad base URL: \(endpoint.baseURL)")
        }
        return url
    }

    private func send(_ req: URLRequest) async throws -> Data {
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: req)
        } catch let error as URLError where error.code == .cancelled {
            throw BrainError.cancelled
        } catch {
            throw BrainError.network(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else { throw BrainError.network("no response") }
        guard http.statusCode == 200 else { throw httpError(status: http.statusCode, body: data) }
        return data
    }

    /// The Anthropic error format: {"type":"error","error":{"type":"…","message":"…"}}.
    private func httpError(status: Int, body: Data) -> BrainError {
        if let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
           let err = json["error"] as? [String: Any],
           let type = err["type"] as? String {
            if type == "not_found_error" {
                return .modelNotFound(provider: providerName, model: endpoint.model)
            }
            if let message = err["message"] as? String {
                return .http(status: status, body: message)
            }
        }
        return .http(status: status, body: String(data: body, encoding: .utf8) ?? "unknown error")
    }

    private func parseResponse(_ data: Data) throws -> CompletionResponse {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let response = AnthropicConverter.response(fromJSON: json) else {
            throw BrainError.emptyResponse
        }
        return response
    }

    private var providerName: String {
        URL(string: endpoint.baseURL)?.host() ?? endpoint.baseURL
    }
}
