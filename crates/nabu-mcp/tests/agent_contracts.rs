use nabu_mcp::core::{index_once, ingest_hook_event, init_home, Tool};
use nabu_mcp::serve_with_io;
use serde_json::{json, Value};
use std::io::Cursor;
use tempfile::tempdir;

#[test]
fn server_advertises_locked_tools_resources_and_prompts() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
            json!({"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}),
            json!({"jsonrpc":"2.0","id":4,"method":"prompts/list","params":{}}),
        ],
    );

    let joined = output
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    for expected in [
        "search_history",
        "list_sessions",
        "get_session",
        "export_session",
        "get_event",
        "history_doctor",
        "recall_answer",
        "nabu://sessions",
        "nabu://schema/tools",
        "recall_project_history",
        "prepare_handoff_summary",
    ] {
        assert!(joined.contains(expected), "{expected}");
    }
}

#[test]
fn tool_errors_are_structured_and_recoverable() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_history",
                "arguments": {
                    "query": "fixture",
                    "limit": 0
                }
            }
        })],
    );

    let result = &output[0]["result"];
    assert_eq!(result["isError"], true);
    assert_eq!(result["structuredContent"]["ok"], false);
    assert_eq!(
        result["structuredContent"]["error"]["code"],
        "VALIDATION_ERROR"
    );
    assert!(result["structuredContent"]["error"]["code"].is_string());
    assert!(result["structuredContent"]["error"]["message"].is_string());
    assert_eq!(result["structuredContent"]["error"]["recoverable"], true);
    assert!(result["structuredContent"]["error"]["hint"].is_string());
    assert!(result["structuredContent"]["error"]["details"].is_object());
}

#[test]
fn protocol_errors_return_jsonrpc_error_frames_and_do_not_stop_workers() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let output = run_mcp_lines(
        &home,
        vec![
            json!({"jsonrpc":"2.0","id":"bad-request","params":{}}).to_string(),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}).to_string(),
        ],
    );

    let protocol_error = output
        .iter()
        .find(|response| response["id"] == "bad-request")
        .unwrap();
    assert_eq!(protocol_error["error"]["code"], -32600);
    assert_eq!(protocol_error["error"]["message"], "invalid request");
    assert_eq!(
        protocol_error["error"]["data"]["message"],
        "json-rpc method is required"
    );

    let tools_response = output.iter().find(|response| response["id"] == 2).unwrap();
    assert!(tools_response["result"]["tools"].as_array().unwrap().len() >= 7);
}

#[test]
fn search_history_description_teaches_citation_first_payload_opt_in_loop() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}})],
    );
    let tools = output[0]["result"]["tools"].as_array().unwrap();
    let search = tools
        .iter()
        .find(|tool| tool["name"] == "search_history")
        .unwrap();
    let description = search["description"].as_str().unwrap();
    assert!(description.contains("citation-first"));
    assert!(description.contains("payload=null by default"));
    assert!(description.contains("include_payload=true"));
    assert!(description.contains("get_session around_raw_line"));
    assert!(description.contains("corroborate=true"));
    assert!(description.contains("read-only git"));
    assert!(description.contains("needs_network"));

    let schema = &search["inputSchema"]["properties"];
    for key in [
        "offset",
        "include_payload",
        "include_deltas",
        "dedupe",
        "max_snippet_chars",
        "mode",
        "corroborate",
        "expand_concepts",
    ] {
        assert!(schema.get(key).is_some(), "{key}");
    }
}

#[test]
fn corroborate_parameter_annotates_existing_mcp_read_surfaces() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let cwd = temp.path().join("not-a-repo");
    std::fs::create_dir_all(&cwd).unwrap();
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "mcp-corroboration-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "mcp-corroboration-1",
            "cwd": cwd,
            "project_root": cwd,
            "prompt": "mcp corroboration marker commit deadbee file src/lib.rs PR #99"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let search_default = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_history",
                "arguments": {
                    "query": "mcp corroboration marker",
                    "limit": 1
                }
            }
        })],
    );
    let default_hit = &search_default[0]["result"]["structuredContent"]["results"][0];
    assert!(default_hit.get("corroboration").is_none());
    let raw_line = default_hit["raw_line"].as_i64().unwrap();

    let search_corroborated = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_history",
                "arguments": {
                    "query": "mcp corroboration marker",
                    "limit": 1,
                    "corroborate": true
                }
            }
        })],
    );
    let corroboration =
        &search_corroborated[0]["result"]["structuredContent"]["results"][0]["corroboration"];
    assert_eq!(corroboration["repo"], Value::Null);
    assert!(corroboration["refs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|reference| {
            reference["kind"] == "commit"
                && reference["ref"] == "deadbee"
                && reference["status"] == "unresolved"
                && reference["reason"] == "no_repo"
        }));
    assert!(corroboration["refs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|reference| {
            reference["kind"] == "pr"
                && reference["ref"] == "#99"
                && reference["status"] == "unresolved"
                && reference["reason"] == "needs_network"
        }));

    let session_default = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_session",
                "arguments": {
                    "tool": "claude",
                    "session_id": "mcp-corroboration-session",
                    "limit_events": 1
                }
            }
        })],
    );
    assert!(
        session_default[0]["result"]["structuredContent"]["events"][0]
            .get("corroboration")
            .is_none()
    );

    let session_corroborated = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_session",
                "arguments": {
                    "tool": "claude",
                    "session_id": "mcp-corroboration-session",
                    "limit_events": 1,
                    "corroborate": true
                }
            }
        })],
    );
    assert!(
        session_corroborated[0]["result"]["structuredContent"]["events"][0]
            .get("corroboration")
            .is_some()
    );

    let event_corroborated = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "get_event",
                "arguments": {
                    "tool": "claude",
                    "session_id": "mcp-corroboration-session",
                    "raw_line": raw_line,
                    "corroborate": true
                }
            }
        })],
    );
    assert!(event_corroborated[0]["result"]["structuredContent"]
        .get("corroboration")
        .is_some());

    let recall_corroborated = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "recall_answer",
                "arguments": {
                    "query": "mcp corroboration marker",
                    "limit": 1,
                    "corroborate": true
                }
            }
        })],
    );
    let recall_hit = &recall_corroborated[0]["result"]["structuredContent"]["hits"][0];
    assert!(recall_hit.get("corroboration").is_some());
    assert!(recall_hit["context"].as_array().unwrap()[0]
        .get("corroboration")
        .is_some());
}

#[test]
fn recall_answer_returns_cited_context_without_generated_answer() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "recall-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "recall-user-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "recall answer context marker about auth migrations"
        }),
    )
    .unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "recall-session",
            "hook_event_name": "Stop",
            "message_id": "recall-assistant-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "response": "assistant context for recall answer marker"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "recall_answer",
                "arguments": {
                    "query": "recall answer auth migrations",
                    "limit": 2,
                    "before": 1,
                    "after": 1
                }
            }
        })],
    );

    let result = &output[0]["result"];
    assert_eq!(result["isError"], false);
    assert!(result["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("no answer text was generated"));
    let structured = &result["structuredContent"];
    assert_eq!(structured["mode_applied"], "lexical");
    let hits = structured["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0]["raw_line"].is_i64());
    assert!(hits[0]["context"].as_array().unwrap().iter().any(|event| {
        event["raw_file"].is_string()
            && event["raw_line"].is_i64()
            && event["session_id"] == "recall-session"
    }));
}

#[test]
fn search_history_oversize_response_returns_truncated_prefix_with_continuation() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    for index in 0..50 {
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "cap-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": format!("cap-{index}"),
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": format!("largecap marker {index} {}", "payload ".repeat(12_000))
            }),
        )
        .unwrap();
    }
    index_once(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_history",
                "arguments": {
                    "query": "largecap marker",
                    "limit": 50,
                    "include_payload": true
                }
            }
        })],
    );

    let structured = &output[0]["result"]["structuredContent"];
    assert_eq!(structured["truncated"], true);
    assert!(structured["returned"].as_u64().unwrap() > 0);
    assert!(structured["continuation"]["next_offset"].as_u64().unwrap() > 0);
    assert!(serde_json::to_vec(structured).unwrap().len() <= 256 * 1024);
    assert!(output[0]["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("truncated"));
}

#[test]
fn get_session_oversize_response_keeps_cited_prefix() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "large-session-page",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "large-session-page-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": format!("session cap marker {}", "payload ".repeat(80_000))
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_session",
                "arguments": {
                    "tool": "claude",
                    "session_id": "large-session-page",
                    "limit_events": 10
                }
            }
        })],
    );

    let structured = &output[0]["result"]["structuredContent"];
    assert_eq!(structured["mcp_truncated"], true);
    assert_eq!(structured["events"].as_array().unwrap().len(), 1);
    assert_eq!(structured["events"][0]["raw_line"], 1);
    assert_eq!(structured["events"][0]["text_truncated"], true);
    assert!(structured["continuation"]["next_after_raw_line"].is_i64());
    assert!(serde_json::to_vec(structured).unwrap().len() <= 256 * 1024);
}

#[test]
fn get_event_oversize_response_preserves_raw_citation() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "large-event",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "large-event-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": format!("event cap marker {}", "payload ".repeat(80_000))
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_event",
                "arguments": {
                    "tool": "claude",
                    "session_id": "large-event",
                    "raw_line": 1
                }
            }
        })],
    );

    let structured = &output[0]["result"]["structuredContent"];
    assert_eq!(structured["mcp_truncated"], true);
    assert_eq!(structured["raw_line"], 1);
    assert!(structured["raw_file"]
        .as_str()
        .unwrap()
        .contains("large-event"));
    assert_eq!(structured["envelope"]["payload"]["truncated"], true);
    assert!(structured["searchable_text"]
        .as_str()
        .unwrap()
        .contains("event cap marker"));
    assert!(
        structured["searchable_text_truncated_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        structured["mcp_original_size_bytes"].as_u64().unwrap()
            > serde_json::to_vec(structured).unwrap().len() as u64
    );
    assert!(serde_json::to_vec(structured).unwrap().len() <= 256 * 1024);
}

#[test]
fn export_session_markdown_oversize_response_keeps_content_prefix() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "large-markdown-export",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "large-markdown-export-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": format!("markdown cap marker {}", "payload ".repeat(80_000))
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "export_session",
                "arguments": {
                    "tool": "claude",
                    "session_id": "large-markdown-export",
                    "format": "markdown"
                }
            }
        })],
    );

    let structured = &output[0]["result"]["structuredContent"];
    assert_eq!(structured["mcp_truncated"], true);
    assert_eq!(structured["format"], "markdown");
    assert!(structured["content"]
        .as_str()
        .unwrap()
        .contains("markdown cap marker"));
    assert!(structured["content_truncated_bytes"].as_u64().unwrap() > 0);
    assert!(serde_json::to_vec(structured).unwrap().len() <= 256 * 1024);
}

#[test]
fn numeric_pointer_minimums_are_enforced() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_event",
                "arguments": {
                    "tool": "claude",
                    "session_id": "fixture-session",
                    "raw_line": 0
                }
            }
        })],
    );

    let result = &output[0]["result"];
    assert_eq!(result["isError"], true);
    assert_eq!(
        result["structuredContent"]["error"]["code"],
        "VALIDATION_ERROR"
    );
    assert!(result["structuredContent"]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("raw_line must be at least 1"));
}

#[test]
fn numeric_pagination_bounds_and_types_are_enforced() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search_history",
                    "arguments": {
                        "query": "needle",
                        "limit": 51
                    }
                }
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "search_history",
                    "arguments": {
                        "query": "needle",
                        "max_snippet_chars": 12.5
                    }
                }
            }),
        ],
    );

    let first_response = output.iter().find(|message| message["id"] == 1).unwrap();
    let first = &first_response["result"]["structuredContent"]["error"];
    assert_eq!(first["code"], "VALIDATION_ERROR");
    assert!(first["message"]
        .as_str()
        .unwrap()
        .contains("limit must be between 1 and 50"));

    let second_response = output.iter().find(|message| message["id"] == 2).unwrap();
    let second = &second_response["result"]["structuredContent"]["error"];
    assert_eq!(second["code"], "VALIDATION_ERROR");
    assert!(second["message"]
        .as_str()
        .unwrap()
        .contains("max_snippet_chars must be an integer"));
}

#[test]
fn export_session_redacts_agent_facing_content() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "fixture-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "secret-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456\nAuthorization: Bearer abcdefghijklmnopqrstuvwxyz123456\n-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "export_session",
                "arguments": {
                    "tool": "claude",
                    "session_id": "fixture-session",
                    "format": "markdown",
                    "redact": true
                }
            }
        })],
    );

    let content = output[0]["result"]["structuredContent"]["content"]
        .as_str()
        .unwrap();
    assert!(content.contains("[REDACTED:ENV_VALUE]"));
    assert!(content.contains("Bearer [REDACTED:BEARER_TOKEN]"));
    assert!(content.contains("[REDACTED:PRIVATE_KEY]"));
    assert!(!content.contains("sk-abcdefghijklmnopqrstuvwxyz123456"));
    assert!(!content.contains("abcdefghijklmnopqrstuvwxyz123456"));
    assert!(!content.contains("-----BEGIN PRIVATE KEY-----"));
}

#[test]
fn export_session_jsonl_can_exceed_general_response_cap() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let prompt = "oversized jsonl export marker ".repeat(12_000);
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "large-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "large-jsonl-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": prompt
        }),
    )
    .unwrap();

    let output = run_mcp(
        &home,
        vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "export_session",
                "arguments": {
                    "tool": "claude",
                    "session_id": "large-session",
                    "format": "jsonl",
                    "redact": false
                }
            }
        })],
    );

    let structured = &output[0]["result"]["structuredContent"];
    assert_eq!(structured["format"], "jsonl");
    assert!(structured["content"].as_str().unwrap().len() > 256 * 1024);
    assert!(structured["content"]
        .as_str()
        .unwrap()
        .contains("oversized jsonl export marker"));
    assert!(structured.get("truncated").is_none());
}

#[test]
fn concurrent_requests_all_receive_exactly_one_response() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    // Many in-flight requests at once: the server handles them on a worker pool
    // and may answer out of order, but every `id` must come back exactly once.
    let count = 50;
    let messages = (1..=count)
        .map(|id| {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "tools/call",
                "params": {
                    "name": "list_sessions",
                    "arguments": {}
                }
            })
        })
        .collect::<Vec<_>>();

    let output = run_mcp(&home, messages);

    let mut ids = output
        .iter()
        .map(|response| response["id"].as_i64().expect("response carries an id"))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, (1..=count).collect::<Vec<_>>());
}

fn run_mcp(home: &std::path::Path, messages: Vec<Value>) -> Vec<Value> {
    run_mcp_lines(
        home,
        messages
            .into_iter()
            .map(|message| message.to_string())
            .collect(),
    )
}

fn run_mcp_lines(home: &std::path::Path, messages: Vec<String>) -> Vec<Value> {
    let input = messages.join("\n") + "\n";
    let mut output = Vec::new();
    serve_with_io(home.to_path_buf(), Cursor::new(input), &mut output).unwrap();
    String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
