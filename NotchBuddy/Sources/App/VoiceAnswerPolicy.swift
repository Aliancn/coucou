#if !APPSTORE
import Foundation

// MARK: - Voice answer policy

/// A short answer Coucou can say aloud after "OK Coucou", in the answer
/// language, with fresh web results when the provider can search.
/// Anthropic keeps its server-side web_search tool (fast model first, the
/// Settings model as fallback — the old voiceAnswer path verbatim); every
/// other configured provider answers through the OpenAI dialect, with Exa's
/// results riding along when its key is set, and a system prompt that
/// degrades honestly when nothing can search.
@MainActor
enum VoiceAnswerPolicy {

    /// Fast model for spoken answers; the Settings model when it is unavailable.
    private static let fastModel = "claude-haiku-4-5"

    /// Any provider the user configured for chat — the active one first, so
    /// the answer runs on whatever the user already set up. nil when nothing
    /// is configured: the gate for the voice answer card.
    static var firstAvailableEndpoint: ModelEndpoint? {
        let state = AppState.shared
        let order = [state.chatProvider] + ChatProvider.allCases.filter { $0 != state.chatProvider }
        for provider in order {
            if case .resolved(let endpoint) = ProviderCatalog.chatEndpoint(for: provider, in: state) {
                return endpoint
            }
        }
        return nil
    }

    /// The answer card is available as soon as one chat provider is usable.
    static var isAvailable: Bool { firstAvailableEndpoint != nil }

    /// A short answer to `question`, with web search, in the answer language.
    /// `history` holds the previous exchanges of this voice conversation
    /// (memory only). nil on any failure: no provider, network, quota.
    static func answer(_ question: String, history: [VoiceWebTurn], french: Bool) async -> String? {
        guard let endpoint = firstAvailableEndpoint else { return nil }
        let today = Date().formatted(.dateTime.weekday(.wide).day().month(.wide).year()
            .locale(Locale(identifier: french ? "fr_FR" : "en_US")))
        switch endpoint.dialect {
        case .anthropicMessages:
            return await anthropicAnswer(question, history: history, french: french,
                                          today: today, endpoint: endpoint)
        case .openaiChat:
            return await openAIAnswer(question, history: history, french: french,
                                      today: today, endpoint: endpoint)
        }
    }

    // MARK: - Anthropic (server-side web search)

    /// The old voiceAnswer path, engine-shaped: the web_search server tool
    /// with the user's approximate location, the fast model first, the
    /// Settings model when the fast one is not available to this key.
    private static func anthropicAnswer(_ question: String, history: [VoiceWebTurn],
                                        french: Bool, today: String,
                                        endpoint: ModelEndpoint) async -> String? {
        let system = systemPrompt(french: french, today: today, canSearch: true)
        var messages: [BrainMessage] = []
        for turn in history {
            messages.append(BrainMessage(role: .user, blocks: [.text(turn.question)]))
            messages.append(BrainMessage(role: .assistant, blocks: [.text(turn.answer)]))
        }
        messages.append(BrainMessage(role: .user, blocks: [.text(question)]))

        var searchSchema: [String: JSONValue] = ["max_uses": .number(3)]
        var location: [String: JSONValue] = ["type": .string("approximate"),
                                             "timezone": .string(TimeZone.current.identifier)]
        if let country = Locale.current.region?.identifier, country.count == 2 {
            location["country"] = .string(country)
        }
        searchSchema["user_location"] = .object(location)
        let search = ToolDeclaration(name: "web_search", description: "",
                                     inputSchema: searchSchema,
                                     serverType: "web_search_20250305")

        func request() -> CompletionRequest {
            CompletionRequest(system: system, messages: messages, tools: [search],
                              maxTokens: 700, betas: ["web-search-2025-03-05"], timeout: 45)
        }

        var fast = endpoint
        fast.model = fastModel
        do {
            return try await spoken(AnthropicClient(endpoint: fast).complete(request()))
        } catch {
            // The fast model is not available to this key: the Settings model instead.
            do {
                return try await spoken(AnthropicClient(endpoint: endpoint).complete(request()))
            } catch {
                appendAppLog("nb.log", "[Voice] web answer failed")
                return nil
            }
        }
    }

    // MARK: - OpenAI dialect (Exa search or no search)

    /// Providers without a server-side search tool: Exa's fresh results ride
    /// along in the question when its key is set; without it the system
    /// prompt says so instead of pretending.
    private static func openAIAnswer(_ question: String, history: [VoiceWebTurn],
                                     french: Bool, today: String,
                                     endpoint: ModelEndpoint) async -> String? {
        let results = await SearchPolicy.search(question)
        let canSearch = results != nil
        let system = systemPrompt(french: french, today: today, canSearch: canSearch)
        var messages: [BrainMessage] = []
        for turn in history {
            messages.append(BrainMessage(role: .user, blocks: [.text(turn.question)]))
            messages.append(BrainMessage(role: .assistant, blocks: [.text(turn.answer)]))
        }
        var asked = question
        if let results {
            asked = SearchPolicy.contextText(results, query: question) + "\n\n" + question
        }
        messages.append(BrainMessage(role: .user, blocks: [.text(asked)]))
        let request = CompletionRequest(system: system, messages: messages,
                                        maxTokens: 700, timeout: 45)
        do {
            return try await spoken(OpenAIClient(endpoint: endpoint).complete(request))
        } catch {
            appendAppLog("nb.log", "[Voice] web answer failed")
            return nil
        }
    }

    // MARK: - Shared

    /// 2 to 4 spoken sentences, no markdown, no lists, no links. The search
    /// sentence is dropped — replaced by an honest fallback — when the
    /// provider cannot search at all.
    private static func systemPrompt(french: Bool, today: String, canSearch: Bool) -> String {
        if french {
            let search = canSearch
                ? "Cherche sur le web dès que la réponse peut avoir changé (actualité, résultats, prix, horaires, météo, personnes)."
                : "Tu n'as pas accès au web : réponds avec ce que tu sais, et dis-le quand la réponse a pu changer."
            return """
              Tu es Coucou, un assistant vocal dans le notch du Mac. Nous sommes le \(today). \
              Ta réponse est lue à voix haute : 2 à 4 phrases courtes, en français, sans markdown, sans liste, \
              sans lien, sans citer tes sources. \(search) Donne directement les faits utiles. \
              S'il y a clairement plus à raconter, termine par une seule courte question pour proposer la suite.
              """
        }
        let search = canSearch
            ? "Search the web whenever the answer may have changed (news, scores, prices, opening hours, weather, people)."
            : "You have no web access: answer from what you know, and say so when the answer may have changed."
        return """
          You are Coucou, a voice assistant in the Mac's notch. Today is \(today). \
          Your answer is read aloud: 2 to 4 short sentences in English, no markdown, no lists, no links, \
          never cite sources. \(search) Give the useful facts straight away. \
          When there is clearly more worth telling, end with one short question offering to go on.
          """
    }

    /// The response's spoken text: the text blocks joined (web-search
    /// responses interleave them with tool blocks), trimmed for speech.
    private static func spoken(_ response: CompletionResponse) -> String? {
        let content: [[String: Any]]
        if let raw = response.rawBlocksJSON,
           let data = raw.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            content = parsed
        } else {
            content = response.blocks.compactMap(AnthropicConverter.blockJSON)
        }
        guard let text = claudeResponseText(fromContent: content) else { return nil }
        let spokenText = VoiceQuery.spokenText(text)
        return spokenText.isEmpty ? nil : spokenText
    }
}
#endif
