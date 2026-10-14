import Foundation

// MARK: - Claude API

@MainActor
final class ClaudeService {
    static let shared = ClaudeService()

    private let endpoint = URL(string: "https://api.anthropic.com/v1/messages")!
    private let anthropicVersion = "2023-06-01"

    // MARK: - Model list

    /// Fetches available models from the Anthropic API in the order the API returns them
    /// (newest first). Returns an empty array on any error — callers fall back to a static list.
    static func fetchModels(apiKey: String) async -> [(id: String, label: String)] {
        guard let url = URL(string: "https://api.anthropic.com/v1/models?limit=100") else { return [] }
        var req = URLRequest(url: url, timeoutInterval: 10)
        req.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        req.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        guard let (data, response) = try? await URLSession.shared.data(for: req),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["data"] as? [[String: Any]] else { return [] }
        return items.compactMap { item in
            guard let id = item["id"] as? String,
                  let name = item["display_name"] as? String else { return nil }
            return (id: id, label: name)
        }
    }

    /// Fetches Gemini models via the OpenAI-compatible endpoint.
    /// Strips the "models/" prefix that the API sometimes returns and filters non-chat models.
    static func fetchGoogleModels(apiKey: String) async -> [(id: String, label: String)] {
        guard let url = URL(string: "https://generativelanguage.googleapis.com/v1beta/openai/models") else { return [] }
        var req = URLRequest(url: url, timeoutInterval: 10)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        guard let (data, response) = try? await URLSession.shared.data(for: req),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["data"] as? [[String: Any]] else { return [] }
        let excluded = ["embed", "imagen", "veo", "aqa", "tts", "audio", "live"]
        return items.compactMap { item in
            guard let raw = item["id"] as? String else { return nil }
            let id = raw.hasPrefix("models/") ? String(raw.dropFirst(7)) : raw
            let lower = id.lowercased()
            guard !excluded.contains(where: { lower.contains($0) }) else { return nil }
            return (id: id, label: id)
        }
    }

    /// Fetches chat models from the OpenAI API, sorted newest-first by creation date.
    /// Excludes non-chat model families.
    static func fetchOpenAIModels(apiKey: String) async -> [(id: String, label: String)] {
        guard let url = URL(string: "https://api.openai.com/v1/models") else { return [] }
        var req = URLRequest(url: url, timeoutInterval: 10)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        guard let (data, response) = try? await URLSession.shared.data(for: req),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["data"] as? [[String: Any]] else { return [] }
        let excluded = ["embed", "tts", "whisper", "dall-e", "audio", "realtime", "moderat",
                        "codex", "computer-use", "transcribe", "image", "sora",
                        "babbage", "davinci", "instruct"]
        return items
            .compactMap { item -> (id: String, created: Int)? in
                guard let id = item["id"] as? String else { return nil }
                let lower = id.lowercased()
                guard !excluded.contains(where: { lower.contains($0) }) else { return nil }
                return (id: id, created: item["created"] as? Int ?? 0)
            }
            .sorted { $0.created > $1.created }
            .map { (id: $0.id, label: $0.id) }
    }

    /// Fetches models from any OpenAI-compatible `/v1/models` endpoint.
    static func fetchOpenAICompatibleModels(baseURL: String, apiKey: String) async -> [(id: String, label: String)] {
        guard let url = URL(string: "\(baseURL)/models") else { return [] }
        var req = URLRequest(url: url, timeoutInterval: 10)
        req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        guard let (data, response) = try? await URLSession.shared.data(for: req),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["data"] as? [[String: Any]] else { return [] }
        return items.compactMap { item in
            guard let id = item["id"] as? String else { return nil }
            return (id: id, label: id)
        }
    }

    /// Chosen in Settings; falls back to the default when the field is left empty.
    private var model: String {
        let m = AppState.shared.claudeModel.trimmingCharacters(in: .whitespacesAndNewlines)
        return m.isEmpty ? AppState.defaultClaudeModel : m
    }

    var apiKey: String? { KeychainStore.shared.get("anthropic-api-key") }

    #if !APPSTORE
    // MARK: - Voice brain turn (Claude with Coucou's tools)

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

    // MARK: - Voice answer (Coucou's voice, web search)

    /// Fast model for spoken answers; the chat model from Settings when it is unavailable.
    private static let voiceModel = "claude-haiku-4-5"

    /// A short answer Coucou can say aloud, with web search, in the answer language.
    /// `history` holds the previous exchanges of this voice conversation (memory only).
    /// nil on any failure: no key, network, quota.
    func voiceAnswer(_ question: String, history: [VoiceWebTurn], french: Bool) async -> String? {
        guard let key = apiKey, !key.isEmpty else { return nil }
        let today = Date().formatted(.dateTime.weekday(.wide).day().month(.wide).year()
            .locale(Locale(identifier: french ? "fr_FR" : "en_US")))
        let system = french
            ? """
              Tu es Coucou, un assistant vocal dans le notch du Mac. Nous sommes le \(today). \
              Ta réponse est lue à voix haute : 2 à 4 phrases courtes, en français, sans markdown, sans liste, \
              sans lien, sans citer tes sources. Cherche sur le web dès que la réponse peut avoir changé \
              (actualité, résultats, prix, horaires, météo, personnes). Donne directement les faits utiles. \
              S'il y a clairement plus à raconter, termine par une seule courte question pour proposer la suite.
              """
            : """
              You are Coucou, a voice assistant in the Mac's notch. Today is \(today). \
              Your answer is read aloud: 2 to 4 short sentences in English, no markdown, no lists, no links, \
              never cite sources. Search the web whenever the answer may have changed (news, scores, prices, \
              opening hours, weather, people). Give the useful facts straight away. \
              When there is clearly more worth telling, end with one short question offering to go on.
              """
        var messages: [[String: Any]] = []
        for turn in history {
            messages.append(["role": "user", "content": turn.question])
            messages.append(["role": "assistant", "content": turn.answer])
        }
        messages.append(["role": "user", "content": question])

        var search: [String: Any] = ["type": "web_search_20250305", "name": "web_search", "max_uses": 3]
        var location: [String: Any] = ["type": "approximate", "timezone": TimeZone.current.identifier]
        if let country = Locale.current.region?.identifier, country.count == 2 { location["country"] = country }
        search["user_location"] = location

        func body(_ model: String) -> [String: Any] {
            ["model": model, "max_tokens": 700, "system": system, "tools": [search], "messages": messages]
        }
        let data: Data
        do {
            data = try await callAPI(body: body(Self.voiceModel), key: key, beta: "web-search-2025-03-05")
        } catch {
            // The fast model is not available to this key: the Settings model instead.
            guard let retry = try? await callAPI(body: body(model), key: key, beta: "web-search-2025-03-05") else {
                appendAppLog("nb.log", "[Voice] web answer failed")
                return nil
            }
            data = retry
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let content = json["content"] as? [[String: Any]],
              let text = claudeResponseText(fromContent: content) else { return nil }
        let spoken = VoiceQuery.spokenText(text)
        return spoken.isEmpty ? nil : spoken
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
