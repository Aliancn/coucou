import Foundation

// MARK: - Claude API

@MainActor
final class ClaudeService {
    static let shared = ClaudeService()

    private let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    private let anthropicVersion = "2023-06-01"

    /// Chosen in Settings; falls back to the default when the field is left empty.
    private var model: String {
        let m = AppState.shared.claudeModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return m.isEmpty ? AppState.defaultClaudeModel : m
    }

    var apiKey: String? { KeychainStore.shared.get("anthropic-api-key") }

    #if !APPSTORE
    // MARK: - Voice brain turn (Claude with Coucou's tools)

    /// Fast model for the voice brain; the Settings model when it is unavailable.
    private static let voiceModel = "claude-haiku-4-5"

    /// One Messages API call for the voice brain: returns the assistant content blocks and
    /// the stop reason. The fast model first, the Settings model if the key can't use it.
    func voiceTurn(system: String, messages: [[String: Any]], tools: [[String: Any]])
        async -> (content: [[String: Any]], stopReason: String)? {
        guard let key = apiKey, !key.isEmpty else { return nil }
        func body(_ model: String) -> [String: Any] {
            ["model": model, "max_tokens": 2048, "system": system, "tools": tools, "messages": messages]
        }
        let data: Data
        do {
            data = try await callAPI(body: body(Self.voiceModel), key: key, beta: "web-search-2025-03-05")
        } catch {
            guard let retry = try? await callAPI(body: body(model), key: key, beta: "web-search-2025-03-05") else {
                appendAppLog("nb.log", "[Voice] Claude turn failed: \(error.localizedDescription)")
                return nil
            }
            data = retry
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let content = json["content"] as? [[String: Any]] else { return nil }
        return (content, json["stop_reason"] as? String ?? "end_turn")
    }

    #endif

    // MARK: - API call

    private func callAPI(body: [String: Any], key: String, beta: String? = nil) async throws -> Data {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue(key, forHTTPHeaderField: "x-api-key")
        request.setValue(anthropicVersion, forHTTPHeaderField: "anthropic-version")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        if let beta { request.setValue(beta, forHTTPHeaderField: "anthropic-beta") }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 45

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            // Parse Anthropic error format: {"type":"error","error":{"type":"…","message":"…"}}
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let err = json["error"] as? [String: Any],
               let errType = err["type"] as? String,
               let errMsg = err["message"] as? String {
                if errType == "not_found_error" {
                    let id = AppState.shared.claudeModel
                    throw NSError(domain: "Claude", code: 0,
                        userInfo: [NSLocalizedDescriptionKey:
                            "Model not found: \(id). Pick another one in Settings."])
                }
                throw NSError(domain: "Claude", code: 0,
                    userInfo: [NSLocalizedDescriptionKey: errMsg])
            }
            let msg = String(data: data, encoding: .utf8) ?? "unknown error"
            throw NSError(domain: "Claude", code: 0, userInfo: [NSLocalizedDescriptionKey: msg])
        }
        return data
    }
}
