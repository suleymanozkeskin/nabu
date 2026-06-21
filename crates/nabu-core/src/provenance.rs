//! Provenance reference extraction: pulls GitHub PR references and commit SHAs
//! out of rendered event text so they can be indexed and queried.
//!
//! This is intentionally precision-first. The transcripts we index are dense
//! with hex-looking tokens (UUIDs, hashes, base16 blobs, timestamps), so the
//! SHA matcher requires word boundaries and rejects tokens that are decimal-only
//! or that sit inside a longer hex run. PR references are normalized to a single
//! canonical form so that `#54`, `org/repo#54`, and a pull URL for 54 all index
//! and query identically.
//!
//! The extractor is the source of truth for the `event_refs` table written in
//! `index.rs`; it does not touch git or the network. Resolving "is it merged"
//! is a separate, query-time concern handled by `search::corroborate`.

use regex::Regex;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// The kind of provenance reference extracted from event text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum RefKind {
    /// A GitHub pull-request reference (`#54`, `org/repo#54`, or a pull URL).
    Pr,
    /// A git commit SHA (7-40 lowercase hex characters).
    Commit,
}

impl RefKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Pr => "pr",
            Self::Commit => "commit",
        }
    }
}

/// A single extracted reference, normalized to its canonical stored form.
///
/// `value` is the form that lands in `event_refs.ref_value` and the form a
/// `ref=` search filter compares against:
/// - PR refs normalize to `#<number>` (the `org/repo` qualifier and any URL
///   wrapper are dropped; the issue/PR number is the join key across sessions).
/// - Commit SHAs normalize to their lowercase hex string as written.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ProvenanceRef {
    pub(crate) kind: RefKind,
    pub(crate) value: String,
}

/// Extract every distinct provenance reference from `text`.
///
/// Results are deduplicated on `(kind, value)` and returned in sorted order so
/// indexing is deterministic regardless of where a ref appears in the text.
pub(crate) fn extract_refs(text: &str) -> Vec<ProvenanceRef> {
    let mut found: BTreeSet<ProvenanceRef> = BTreeSet::new();

    // PR references from GitHub pull URLs, e.g.
    //   https://github.com/org/repo/pull/54
    //   github.com/org/repo/pull/54#issuecomment-123
    // The host and path qualifier are dropped; only the PR number is kept.
    for captures in pr_url_regex().captures_iter(text) {
        push_pr(&mut found, &captures["num"]);
    }

    // Qualified and bare hash PR references, e.g. `org/repo#54`, `#54`, `PR #54`.
    // A leading `/` (a path segment like `pull/54`) or alphanumeric character
    // disqualifies the `#` so we don't re-capture URL tails or `color: #54abcd`.
    for captures in pr_hash_regex().captures_iter(text) {
        push_pr(&mut found, &captures["num"]);
    }

    // Commit SHAs: 7-40 hex chars bounded by non-hex-word characters, rejecting
    // decimal-only tokens (years, counts) and `0x`-prefixed literals.
    for captures in commit_regex().captures_iter(text) {
        let sha = &captures["sha"];
        if is_plausible_commit_sha(sha) {
            found.insert(ProvenanceRef {
                kind: RefKind::Commit,
                value: sha.to_ascii_lowercase(),
            });
        }
    }

    found.into_iter().collect()
}

fn push_pr(found: &mut BTreeSet<ProvenanceRef>, number: &str) {
    // Drop a leading zero-padded form; `#0054` and `#54` are the same PR.
    let normalized = number.trim_start_matches('0');
    let normalized = if normalized.is_empty() {
        "0"
    } else {
        normalized
    };
    found.insert(ProvenanceRef {
        kind: RefKind::Pr,
        value: format!("#{normalized}"),
    });
}

/// Reject hex tokens that are almost certainly not commit SHAs.
fn is_plausible_commit_sha(sha: &str) -> bool {
    // Decimal-only tokens are line numbers, counts, years, ports - not SHAs.
    if sha.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    // A SHA needs at least one hex letter (a-f) to be distinguishable from a
    // long decimal id; the all-digit guard above already covers the empty case.
    sha.bytes().any(|byte| byte.is_ascii_alphabetic())
}

fn pr_url_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        // github.com/<owner>/<repo>/pull/<num> (or /pull/<num>/files, #comment...).
        // Owner/repo segments are GitHub-name shaped; the number is captured.
        Regex::new(
            r"(?i)\bgithub\.com/[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/[A-Za-z0-9._-]+/pull/(?P<num>[0-9]{1,9})\b",
        )
        .expect("valid pr url regex")
    })
}

fn pr_hash_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        // Optional `org/repo` qualifier, optional `PR ` prefix, then `#<num>`.
        // `(?:^|[^0-9A-Za-z/#])` rejects a `#` glued to a word char or a path
        // segment, so URL tails (`pull/54`) and hex colors (`#54abcd`) miss.
        Regex::new(
            r"(?i)(?:^|[^0-9A-Za-z/#])(?:[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+)?(?:PR\s*)?#(?P<num>[0-9]{1,9})(?:[^0-9A-Za-z]|$)",
        )
        .expect("valid pr hash regex")
    })
}

fn commit_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        // 7-40 hex chars with non-hex boundaries on both sides. The trailing
        // `(?![0-9a-fx])` (case-insensitive) rejects longer hex runs and the
        // `0x` of hex literals so a 64-char blob or `0xdeadbeef` does not match.
        Regex::new(r"(?i)(?:^|[^0-9a-fx])(?P<sha>[0-9a-f]{7,40})(?:[^0-9a-fx]|$)")
            .expect("valid commit regex")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(text: &str, kind: RefKind) -> Vec<String> {
        extract_refs(text)
            .into_iter()
            .filter(|reference| reference.kind == kind)
            .map(|reference| reference.value)
            .collect()
    }

    #[test]
    fn extracts_bare_hash_pr() {
        assert_eq!(values("fixed in #54", RefKind::Pr), vec!["#54".to_string()]);
    }

    #[test]
    fn extracts_pr_prefixed_reference() {
        assert_eq!(
            values("landed via PR #1234 today", RefKind::Pr),
            vec!["#1234".to_string()]
        );
    }

    #[test]
    fn extracts_qualified_org_repo_reference() {
        assert_eq!(
            values("see suleymanozkeskin/nabu#73 for context", RefKind::Pr),
            vec!["#73".to_string()]
        );
    }

    #[test]
    fn extracts_pull_url_reference() {
        assert_eq!(
            values(
                "merged https://github.com/suleymanozkeskin/nabu/pull/60 cleanly",
                RefKind::Pr,
            ),
            vec!["#60".to_string()]
        );
    }

    #[test]
    fn extracts_pull_url_with_trailing_path_and_anchor() {
        assert_eq!(
            values(
                "review github.com/org/repo/pull/75/files#diff-abc here",
                RefKind::Pr,
            ),
            vec!["#75".to_string()]
        );
    }

    #[test]
    fn pr_forms_normalize_to_one_value() {
        // Bare, qualified, and URL forms of the same PR collapse to `#54`.
        let refs = values(
            "#54 and org/repo#54 and https://github.com/org/repo/pull/54",
            RefKind::Pr,
        );
        assert_eq!(refs, vec!["#54".to_string()]);
    }

    #[test]
    fn zero_padded_pr_normalizes() {
        assert_eq!(values("#0054", RefKind::Pr), vec!["#54".to_string()]);
    }

    #[test]
    fn multiple_distinct_prs_sorted() {
        let refs = values("touches #59, #54, and #73", RefKind::Pr);
        assert_eq!(
            refs,
            vec!["#54".to_string(), "#59".to_string(), "#73".to_string()]
        );
    }

    #[test]
    fn hex_color_is_not_a_pr() {
        // `#54abcd` is a color literal, not `#54`.
        assert!(values("background: #54abcd;", RefKind::Pr).is_empty());
    }

    #[test]
    fn extracts_short_commit_sha() {
        assert_eq!(
            values("reverted in 1e0a357 earlier", RefKind::Commit),
            vec!["1e0a357".to_string()]
        );
    }

    #[test]
    fn extracts_full_commit_sha() {
        let sha = "100a8704bf3c2d1e5a6f7b8c9d0e1f2a3b4c5d6e";
        assert_eq!(
            values(&format!("at commit {sha}"), RefKind::Commit),
            vec![sha.to_string()]
        );
    }

    #[test]
    fn commit_sha_is_lowercased() {
        assert_eq!(
            values("commit 1E0A357 there", RefKind::Commit),
            vec!["1e0a357".to_string()]
        );
    }

    #[test]
    fn decimal_only_token_is_not_a_commit() {
        // 7+ digits that happen to be in hex range but are decimal-only.
        assert!(values("error code 1234567 occurred", RefKind::Commit).is_empty());
        assert!(values("port 8080443 open", RefKind::Commit).is_empty());
    }

    #[test]
    fn hex_literal_prefix_is_not_a_commit() {
        // `0xdeadbeef` is a literal, the `0x` boundary must reject it.
        assert!(values("mask 0xdeadbeef applied", RefKind::Commit).is_empty());
    }

    #[test]
    fn oversized_hex_blob_is_not_a_commit() {
        // 64-char hex (e.g. a sha256) exceeds the 40-char commit ceiling and the
        // boundary rejects it rather than capturing a 40-char prefix.
        let blob = "a".repeat(64);
        assert!(values(&format!("digest {blob} stored"), RefKind::Commit).is_empty());
    }

    #[test]
    fn too_short_hex_is_not_a_commit() {
        // 6 hex chars is below the 7-char floor.
        assert!(values("ref abc123 only", RefKind::Commit).is_empty());
    }

    #[test]
    fn uuid_segments_are_documented_boundary() {
        // A canonical UUID is hyphen-bounded 8/4/4/4/12 groups. Runs of 7+ hex
        // are candidates: the 8-char `550e8400` block is hex of SHA length and
        // letter-bearing, so it extracts (an unavoidable, accepted false
        // positive). The 12-char node `446655440000` is decimal-only, so the
        // plausibility guard rejects it; the 4-char middle groups are too short.
        // This documents the precision boundary: a hex-letter-bearing 7-40 run
        // is indistinguishable from an abbreviated SHA by text alone.
        let refs = values(
            "id 550e8400-e29b-41d4-a716-446655440000 here",
            RefKind::Commit,
        );
        assert_eq!(refs, vec!["550e8400".to_string()]);
    }

    #[test]
    fn mixed_text_extracts_both_kinds() {
        let refs = extract_refs("fix #54 shipped in 1e0a357");
        assert_eq!(
            refs,
            vec![
                ProvenanceRef {
                    kind: RefKind::Pr,
                    value: "#54".to_string(),
                },
                ProvenanceRef {
                    kind: RefKind::Commit,
                    value: "1e0a357".to_string(),
                },
            ]
        );
    }

    #[test]
    fn empty_text_yields_nothing() {
        assert!(extract_refs("").is_empty());
        assert!(extract_refs("no refs in this plain sentence at all").is_empty());
    }
}
