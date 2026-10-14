#if !APPSTORE
import Foundation

// MARK: - Search policy (Exa)

/// Web search through Exa (api.exa.ai): the client-side search for providers
/// without a server-side one (everything but Anthropic's web_search tool).
/// The key lives in the Keychain as exa-api-key. Network only to Exa, only
/// when the key is set — nil results mean "answer without search", never a
/// spoken failure.
enum SearchPolicy {

    struct Result {
        var title: String
        var url: String
        var text: String
    }

    static var isConfigured: Bool {
        !(KeychainStore.shared.get("exa-api-key") ?? "").isEmpty
    }

    /// Fresh web results for `query`, or nil when Exa isn't configured or
    /// the call fails — the caller then answers without search.
    static func search(_ query: String, maxResults: Int = 3) async -> [Result]? {
        guard let key = KeychainStore.shared.get("exa-api-key"), !key.isEmpty else { return nil }
        guard let url = URL(string: "https://api.exa.ai/search") else { return nil }
        var req = URLRequest(url: url, timeoutInterval: 10)
        req.httpMethod = "POST"
        req.setValue(key, forHTTPHeaderField: "x-api-key")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: [
            "query": query,
            "numResults": maxResults,
            "contents": ["text": ["maxCharacters": 1000]],
        ])
        guard let (data, response) = try? await URLSession.shared.data(for: req),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let results = json["results"] as? [[String: Any]] else { return nil }
        return results.compactMap { result in
            guard let url = result["url"] as? String else { return nil }
            return Result(title: result["title"] as? String ?? "",
                          url: url,
                          text: result["text"] as? String ?? "")
        }
    }

    /// The results as context text for the answering model.
    static func contextText(_ results: [Result], query: String) -> String {
        let entries = results.map { result in
            var entry = "— \(result.title) (\(result.url))"
            if !result.text.isEmpty { entry += "\n\(result.text)" }
            return entry
        }
        return "Fresh web results for \"\(query)\":\n" + entries.joined(separator: "\n\n")
    }
}
#endif
