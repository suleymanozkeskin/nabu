//! Secret redaction for exported and displayed text: regex rule set, env-var
//! assignment masking, and sensitive-key detection.

use regex::{Captures, Regex};
use serde_json::Value;
use std::sync::OnceLock;

pub fn redact_export_text(text: &str) -> String {
    redact_text(text)
}

pub fn redact_export_json(value: Value) -> Value {
    redact_json_value(value)
}

pub(crate) fn redact_json_value(value: Value) -> Value {
    match value {
        Value::String(text) => Value::String(redact_text(&text)),
        Value::Array(values) => Value::Array(values.into_iter().map(redact_json_value).collect()),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| {
                    if is_sensitive_key(&key) {
                        (key, Value::String("[REDACTED:ENV_VALUE]".to_string()))
                    } else {
                        (key, redact_json_value(value))
                    }
                })
                .collect(),
        ),
        other => other,
    }
}

pub(crate) fn redact_text(text: &str) -> String {
    let mut redacted = text.to_string();
    for (regex, replacement) in redaction_regex_rules() {
        redacted = regex.replace_all(&redacted, *replacement).into_owned();
    }
    env_assignment_redaction_regex()
        .replace_all(&redacted, |captures: &Captures<'_>| {
            format!("{}[REDACTED:ENV_VALUE]", &captures[1])
        })
        .into_owned()
}

fn redaction_regex_rules() -> &'static [(Regex, &'static str)] {
    static RULES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    RULES
        .get_or_init(|| {
            [
                (
                    r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                    "[REDACTED:PRIVATE_KEY]",
                ),
                (
                    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}",
                    "Bearer [REDACTED:BEARER_TOKEN]",
                ),
                (r"\bsk-[A-Za-z0-9_-]{20,}\b", "[REDACTED:API_KEY]"),
                (r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b", "[REDACTED:API_KEY]"),
                (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "[REDACTED:API_KEY]"),
                (r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b", "[REDACTED:API_KEY]"),
                (r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED:API_KEY]"),
            ]
            .into_iter()
            .map(|(pattern, replacement)| {
                (
                    Regex::new(pattern).expect("valid redaction regex"),
                    replacement,
                )
            })
            .collect()
        })
        .as_slice()
}

fn env_assignment_redaction_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r##"(?im)^([A-Z0-9_]*(API|TOKEN|SECRET|KEY|PASSWORD)[A-Z0-9_]*\s*=\s*)(['"]?)[^\s'"#]{8,}(['"]?)"##,
        )
        .expect("valid env redaction regex")
    })
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    normalized.contains("api_key")
        || normalized.contains("apikey")
        || normalized.contains("access_token")
        || normalized.contains("auth_token")
        || normalized.contains("bearer")
        || normalized.contains("secret")
        || normalized.contains("password")
        || normalized.contains("private_key")
        || normalized.ends_with("_key")
        || normalized.ends_with("token")
}
