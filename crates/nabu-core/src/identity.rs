//! Event identity: dedupe-key derivation (frozen salt), content hashing, and
//! session-id sanitization.

use crate::{
    identity_payload, normalize_identity_text, search_document_for_event, CanonicalType,
    DedupeParts, Result,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn dedupe_key(parts: DedupeParts<'_>) -> Result<String> {
    let mut hasher = Sha256::new();
    // Internal hash domain separator — intentionally NOT renamed with the product.
    // Changing this string re-keys every event and would duplicate/orphan existing
    // stores on reindex. Bump the version only on a deliberate identity change.
    hasher.update(b"harness-raven-dedupe-v2\0");
    hash_part(&mut hasher, parts.tool.as_str());
    hash_part(&mut hasher, parts.session_id);
    hash_part(&mut hasher, parts.canonical_type.as_str());

    if let Some(source_event_id) = parts.source_event_id {
        hash_part(&mut hasher, "native-id");
        hash_part(&mut hasher, source_event_id);
    } else {
        hash_part(&mut hasher, "content");
        hash_part(
            &mut hasher,
            &identity_content_hash(parts.canonical_type, parts.payload)?,
        );
        if let Some(sequence) = parts.sequence {
            hash_part(&mut hasher, "sequence");
            hash_part(&mut hasher, &sequence.to_string());
        } else {
            hash_part(&mut hasher, "unsequenced");
        }
    }

    Ok(format!("sha256:{}", hex::encode(hasher.finalize())))
}

fn hash_part(hasher: &mut Sha256, value: &str) {
    hasher.update(value.as_bytes());
    hasher.update([0]);
}

pub fn sanitize_session_id(session_id: &str) -> String {
    let sanitized: String = session_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "_".to_string()
    } else {
        sanitized
    }
}

pub(crate) fn hash_line(line: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(line.as_bytes());
    hex::encode(hasher.finalize())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn identity_content_hash(canonical_type: CanonicalType, payload: &Value) -> Result<String> {
    let document = search_document_for_event(canonical_type, payload);
    let identity_text = normalize_identity_text(&document.identity_text());
    let bytes = if identity_text.is_empty() {
        serde_json::to_vec(&identity_payload(payload))?
    } else {
        identity_text.into_bytes()
    };
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tool;
    use proptest::prelude::*;
    use serde_json::json;

    #[test]
    fn session_id_sanitization_is_stable_and_filesystem_safe() {
        let unsafe_id = "thread/../with spaces:and:unicode-ç";
        let first = sanitize_session_id(unsafe_id);
        let second = sanitize_session_id(unsafe_id);

        assert_eq!(first, second);
        assert_eq!(first, "thread_.._with_spaces_and_unicode-_");
        assert!(first
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')));
    }

    #[test]
    fn dedupe_key_generation_is_stable_across_source_and_time_metadata() {
        let payload_a = json!({
            "hook_event_name": "UserPromptSubmit",
            "captured_at": "2026-06-17T12:00:59Z",
            "prompt": "identity ignores observation metadata"
        });
        let payload_b = json!({
            "hook_event_name": "UserPromptSubmit",
            "captured_at": "2026-06-17T12:01:01Z",
            "prompt": "identity ignores observation metadata"
        });
        let first = dedupe_key(DedupeParts {
            tool: Tool::Codex,
            session_id: "session",
            canonical_type: CanonicalType::UserMessage,
            source_event_id: None,
            sequence: None,
            payload: &payload_a,
        })
        .unwrap();
        let second = dedupe_key(DedupeParts {
            tool: Tool::Codex,
            session_id: "session",
            canonical_type: CanonicalType::UserMessage,
            source_event_id: None,
            sequence: None,
            payload: &payload_b,
        })
        .unwrap();
        let third = dedupe_key(DedupeParts {
            tool: Tool::Codex,
            session_id: "session",
            canonical_type: CanonicalType::ToolResult,
            source_event_id: None,
            sequence: None,
            payload: &payload_a,
        })
        .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, third);
        assert!(first.starts_with("sha256:"));
    }

    proptest! {
        #[test]
        fn dedupe_key_property_ignores_observation_metadata(
            prompt in "[ -~]{1,256}",
            first_second in 0u8..60,
            second_second in 0u8..60,
            route in "(hook|backfill|event_stream)"
        ) {
            let payload_a = json!({
                "hook_event_name": "UserPromptSubmit",
                "captured_at": format!("2026-06-17T12:00:{first_second:02}Z"),
                "source": route,
                "session_id": "volatile-a",
                "cwd": "/tmp/a",
                "prompt": prompt
            });
            let payload_b = json!({
                "hook_event_name": "UserPromptSubmit",
                "captured_at": format!("2026-06-17T12:01:{second_second:02}Z"),
                "source": "different-route",
                "session_id": "volatile-b",
                "cwd": "/tmp/b",
                "prompt": prompt
            });

            let first = dedupe_key(DedupeParts {
                tool: Tool::Codex,
                session_id: "stable-session",
                canonical_type: CanonicalType::UserMessage,
                source_event_id: None,
                sequence: None,
                payload: &payload_a,
            }).unwrap();
            let second = dedupe_key(DedupeParts {
                tool: Tool::Codex,
                session_id: "stable-session",
                canonical_type: CanonicalType::UserMessage,
                source_event_id: None,
                sequence: None,
                payload: &payload_b,
            }).unwrap();

            prop_assert_eq!(first, second);
        }
    }
}
