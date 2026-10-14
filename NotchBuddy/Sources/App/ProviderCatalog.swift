import Foundation

// MARK: - Provider catalog

/// Every provider's chat plumbing in one place: which model field of AppState
/// stores its choice, where its model list comes from, and how that list is
/// filtered. ChatProvider keeps the identity (pill IDs, display names,
/// accents — stable contract values); this catalog is the single switch a new
/// provider's wiring goes into.
///
/// Keychain reads happen here (App layer) — the engine's clients never see
/// the Keychain, only resolved ModelEndpoints.
@MainActor
enum ProviderCatalog {

    /// A model list entry as the in-chat picker and Settings show it.
    typealias ModelEntry = (id: String, label: String)

    // MARK: - Model storage

    /// The model the user picked for `provider` (its Settings field).
    static func chatModel(for provider: ChatProvider, in state: AppState) -> String {
        switch provider {
        case .anthropic: return state.claudeModel
        case .google:    return state.googleChatModel
        case .openai:    return state.openAIChatModel
        case .ollama:    return state.ollamaChatModel
        case .lmstudio:  return state.lmstudioChatModel
        case .deepseek:  return state.deepseekChatModel
        case .qwen:      return state.qwenChatModel
        case .moonshot:  return state.moonshotChatModel
        case .custom:    return state.customChatModel
        }
    }

    /// Writes the model choice for `provider` back to its Settings field.
    static func setChatModel(_ id: String, for provider: ChatProvider, in state: AppState) {
        switch provider {
        case .anthropic: state.claudeModel = id
        case .google:    state.googleChatModel = id
        case .openai:    state.openAIChatModel = id
        case .ollama:    state.ollamaChatModel = id
        case .lmstudio:  state.lmstudioChatModel = id
        case .deepseek:  state.deepseekChatModel = id
        case .qwen:      state.qwenChatModel = id
        case .moonshot:  state.moonshotChatModel = id
        case .custom:    state.customChatModel = id
        }
    }

    /// The model a provider prefers when the stored choice isn't in its list:
    /// Anthropic the newest Sonnet, Google a Flash, OpenAI a mini; everyone
    /// else the first model the API returns.
    static func preferredModel(for provider: ChatProvider, in models: [ModelEntry]) -> String? {
        let preferred: String?
        switch provider {
        case .anthropic: preferred = "sonnet"
        case .google:    preferred = "flash"
        case .openai:    preferred = "mini"
        default:         preferred = nil
        }
        if let preferred, let match = models.first(where: { $0.id.contains(preferred) }) {
            return match.id
        }
        return models.first?.id
    }

    // MARK: - Model list fetch

    /// Fetches `provider`'s model list the way its API expects (the engine's
    /// listModels + this catalog's per-provider filtering), then applies the
    /// provider's picker entries. Local providers distinguish "unreachable"
    /// so the caller can name the server in its error.
    static func fetchModels(for provider: ChatProvider, in state: AppState)
        async -> Result<[ModelEntry], LocalChatError> {
        // Local servers: LocalChat's fetch (5 s timeout, local filter list).
        if provider.isLocal {
            let baseURL = localBaseURL(for: provider, in: state)
            return await LocalChat.fetchModelsResult(baseURL: baseURL + "/v1")
        }
        let endpoint = listEndpoint(for: provider, in: state)
        let listings: [ModelListing]
        do {
            listings = try await client(for: endpoint).listModels()
        } catch {
            return .failure(.serverUnreachable(endpoint.baseURL))
        }
        return .success(filter(listings, for: provider))
    }

    /// The normalised (no /v1, no trailing slash) server URL of a local
    /// provider, as Settings stores it.
    static func localBaseURL(for provider: ChatProvider, in state: AppState) -> String {
        LocalChat.normaliseURL(provider == .ollama ? state.ollamaServerURL : state.lmstudioServerURL)
    }

    /// The endpoint a provider's model list is fetched from. The API key is
    /// optional for custom endpoints (local gateways often authenticate
    /// nothing) and read here — absent keys fail as unreachable upstream,
    /// the caller's own gate names the missing key first.
    private static func listEndpoint(for provider: ChatProvider, in state: AppState) -> ModelEndpoint {
        let baseURL: String
        switch provider {
        case .anthropic: baseURL = "https://api.anthropic.com/v1"
        case .google:    baseURL = "https://generativelanguage.googleapis.com/v1beta/openai"
        case .openai:    baseURL = "https://api.openai.com/v1"
        case .deepseek:  baseURL = "https://api.deepseek.com/v1"
        case .qwen:      baseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        case .moonshot:  baseURL = "https://api.moonshot.cn/v1"
        case .ollama, .lmstudio, .custom:
            baseURL = state.customChatBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let apiKey = KeychainStore.shared.get(provider.keychainKey)
        return ModelEndpoint(baseURL: baseURL,
                             dialect: provider == .anthropic ? .anthropicMessages : .openaiChat,
                             apiKey: (apiKey?.isEmpty == false) ? apiKey : nil,
                             model: "")
    }

    private static func client(for endpoint: ModelEndpoint) -> any LLMClient {
        switch endpoint.dialect {
        case .anthropicMessages: return AnthropicClient(endpoint: endpoint)
        case .openaiChat:        return OpenAIClient(endpoint: endpoint)
        }
    }

    /// The provider-specific filters the old per-API fetchers applied, kept
    /// verbatim: Google strips the "models/" prefix and drops non-chat
    /// families, OpenAI drops non-chat families and sorts newest-first,
    /// Anthropic keeps its display names, everyone else passes ids through.
    private static func filter(_ listings: [ModelListing], for provider: ChatProvider) -> [ModelEntry] {
        switch provider {
        case .anthropic:
            return listings.map { (id: $0.id, label: $0.label ?? $0.id) }
        case .google:
            let excluded = ["embed", "imagen", "veo", "aqa", "tts", "audio", "live"]
            return listings.compactMap { listing in
                let id = listing.id.hasPrefix("models/") ? String(listing.id.dropFirst(7)) : listing.id
                let lower = id.lowercased()
                guard !excluded.contains(where: { lower.contains($0) }) else { return nil }
                return (id: id, label: id)
            }
        case .openai:
            let excluded = ["embed", "tts", "whisper", "dall-e", "audio", "realtime", "moderat",
                            "codex", "computer-use", "transcribe", "image", "sora",
                            "babbage", "davinci", "instruct"]
            return listings
                .filter { listing in
                    let lower = listing.id.lowercased()
                    return !excluded.contains(where: { lower.contains($0) })
                }
                .sorted { ($0.created ?? 0) > ($1.created ?? 0) }
                .map { (id: $0.id, label: $0.id) }
        default:
            return listings.map { (id: $0.id, label: $0.id) }
        }
    }
}
