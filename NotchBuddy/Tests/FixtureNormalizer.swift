import Foundation

// The .shape normalizer — the ONLY place normalization is allowed to exist.
// Byte-exact goldens freeze converter output for recorded inputs; shapes are
// for web-search fixtures, where re-recording against the live API drifts in
// text and ids but the structure must stay. A shape keeps:
//   - the block type sequence
//   - tool names
//   - id prefix fingerprints (srvtoolu_, toolu_…) and id↔result pairing
//   - citation counts, error codes
// and drops: all text content, full ids, encrypted_* fields, urls.

enum FixtureNormalizer {

    /// The structural fingerprint of an Anthropic response, as stable
    /// canonical JSON. Compare with the recorded `.shape` file.
    static func shape(ofResponseJSON json: [String: Any]) -> String? {
        guard let content = json["content"] as? [[String: Any]] else { return nil }
        let blocks = content.map { shape(ofBlockJSON: $0) }
        let out: [String: Any] = [
            "stop_reason": json["stop_reason"] as? String ?? "",
            "blocks": blocks,
        ]
        return canonicalJSONString(from: out)
    }

    /// The fingerprint of one content block.
    static func shape(ofBlockJSON json: [String: Any]) -> [String: Any] {
        let type = json["type"] as? String ?? ""
        switch type {
        case "tool_use", "server_tool_use":
            return [
                "type": type,
                "id": idFingerprint(json["id"]),
                "name": json["name"] ?? "",
            ]
        case "web_search_tool_result":
            var out: [String: Any] = [
                "type": type,
                "tool_use_id": idFingerprint(json["tool_use_id"]),
            ]
            if let content = json["content"] as? [[String: Any]] {
                // Array of results: count only — titles, urls, encrypted_index dropped.
                out["result_count"] = content.count
            } else if let content = json["content"] as? [String: Any],
                      let errorCode = content["error_code"] as? String {
                // Single error object — the shape that is easy to get wrong.
                out["error"] = ["type": content["type"] ?? "", "error_code": errorCode]
            }
            return out
        case "text":
            var out: [String: Any] = ["type": type]
            if let citations = json["citations"] as? [[String: Any]] {
                out["citation_count"] = citations.count
            }
            return out
        default:
            return ["type": type]
        }
    }

    /// The id's prefix up to the first digit (srvtoolu_, toolu_…) — enough to
    /// check kind and pairing without freezing generated ids.
    private static func idFingerprint(_ id: Any?) -> String {
        guard let id = id as? String else { return "" }
        let prefix = id.prefix { !$0.isNumber }
        return prefix.isEmpty ? "<opaque>" : String(prefix)
    }
}
