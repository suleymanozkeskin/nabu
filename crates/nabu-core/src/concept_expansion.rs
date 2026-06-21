//! Optional, opt-in concept/synonym query expansion for lexical search.
//!
//! Lexical FTS matches only the literal query tokens, so a concept query
//! ("bug") never retrieves a document that records the same concept under a
//! different word ("error", "failure"). This module expands each query term
//! with a curated set of synonyms drawn from a static, alphabetically-grouped
//! map. Expansion is:
//!
//! - **Opt-in** — driven by `SearchOptions.expand_concepts`; the default
//!   lexical and hybrid behavior is byte-for-byte unchanged when the flag is
//!   off.
//! - **Additive / OR-combined** — synonyms join the term list, they never
//!   replace it. Downstream `quoted_fts_terms` already OR-joins terms, so the
//!   FTS match set only ever grows and bm25 keeps literal-term documents ranked
//!   above synonym-only ones (the original term still scores on more columns).
//! - **Order-preserving and deduplicated** — the original terms keep their
//!   position at the front; synonyms are appended in a stable order with
//!   case-insensitive dedup so a term and its own synonym set never double-count.
//!
//! This is a recall aid for the lexical path, not a replacement for the
//! embedding model: it has no notion of context and only relates words present
//! in the curated map.

/// Curated, bidirectional concept clusters. Every word in a cluster expands to
/// every other word in that cluster. Clusters are intentionally conservative
/// (engineering-domain near-synonyms only) so expansion improves recall without
/// dragging in unrelated senses. All entries are lowercase ASCII; matching is
/// case-insensitive.
const CONCEPT_CLUSTERS: &[&[&str]] = &[
    &["bug", "error", "failure", "fault", "defect"],
    &["perf", "performance", "latency", "throughput", "speed"],
    &["crash", "panic", "abort", "segfault"],
    &["config", "configuration", "settings", "setup"],
    &["auth", "authentication", "login", "signin"],
    &["delete", "remove", "purge", "erase"],
    &["fix", "patch", "repair", "resolve"],
    &["docs", "documentation", "readme"],
    &["test", "tests", "spec", "specs"],
    &["dependency", "dependencies", "deps", "package", "packages"],
];

/// Expand a list of literal query terms with curated concept synonyms.
///
/// The original `terms` are preserved at the front of the result in their
/// original order and casing. For every original term whose lowercase form
/// belongs to a concept cluster, the other cluster members are appended (once
/// each, case-insensitively deduplicated against everything already present).
///
/// When no term maps to a cluster the result equals the input (modulo the
/// case-insensitive dedup that already held within a normal query), so callers
/// can apply this unconditionally without changing behavior for non-concept
/// queries.
pub(crate) fn expand_query_terms(terms: &[String]) -> Vec<String> {
    let mut expanded: Vec<String> = Vec::with_capacity(terms.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut push_unique = |value: &str, sink: &mut Vec<String>| {
        let key = value.to_ascii_lowercase();
        if seen.insert(key) {
            sink.push(value.to_string());
        }
    };

    for term in terms {
        push_unique(term, &mut expanded);
    }

    for term in terms {
        let lowered = term.to_ascii_lowercase();
        for cluster in CONCEPT_CLUSTERS {
            if cluster.contains(&lowered.as_str()) {
                for synonym in *cluster {
                    push_unique(synonym, &mut expanded);
                }
            }
        }
    }

    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn original_terms_lead_and_synonyms_follow() {
        let expanded = expand_query_terms(&owned(&["bug"]));
        assert_eq!(expanded[0], "bug", "original term keeps front position");
        for synonym in ["error", "failure", "fault", "defect"] {
            assert!(
                expanded.iter().any(|term| term == synonym),
                "missing synonym {synonym}"
            );
        }
    }

    #[test]
    fn non_concept_terms_are_unchanged() {
        let input = owned(&["nabu", "raw_offset", "harness"]);
        assert_eq!(expand_query_terms(&input), input);
    }

    #[test]
    fn expansion_is_case_insensitive_and_deduplicated() {
        // "Bug" and "ERROR" are in the same cluster; expanding the pair must not
        // duplicate either, and the original casing is preserved up front.
        let expanded = expand_query_terms(&owned(&["Bug", "ERROR"]));
        assert_eq!(expanded[0], "Bug");
        assert_eq!(expanded[1], "ERROR");
        let error_count = expanded
            .iter()
            .filter(|term| term.eq_ignore_ascii_case("error"))
            .count();
        assert_eq!(error_count, 1, "error must not be duplicated");
        let bug_count = expanded
            .iter()
            .filter(|term| term.eq_ignore_ascii_case("bug"))
            .count();
        assert_eq!(bug_count, 1, "bug must not be duplicated");
    }

    #[test]
    fn multiple_clusters_each_expand() {
        let expanded = expand_query_terms(&owned(&["perf", "crash"]));
        assert!(expanded.iter().any(|term| term == "latency"));
        assert!(expanded.iter().any(|term| term == "panic"));
    }
}
