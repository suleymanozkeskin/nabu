use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use tempfile::tempdir;

fn run_codex_hook(home: &Path, payload: &Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_nabu"))
        .arg("--home")
        .arg(home)
        .args(["ingest", "hook", "--tool", "codex"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start nabu hook ingest");

    child
        .stdin
        .take()
        .expect("hook stdin")
        .write_all(payload.to_string().as_bytes())
        .expect("write hook payload");
    child.wait_with_output().expect("wait for hook ingest")
}

fn assert_success_without_output(output: Output) {
    assert!(
        output.status.success(),
        "hook failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"");
    assert_eq!(output.stderr, b"");
}

#[test]
fn codex_prompt_and_stop_hook_successes_are_silent() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let prompt = json!({
        "session_id": "silent-hook-session",
        "hook_event_name": "UserPromptSubmit",
        "prompt": "keep capture output out of chat"
    });
    let stop = json!({
        "session_id": "silent-hook-session",
        "hook_event_name": "Stop",
        "stop_hook_active": false,
        "last_assistant_message": "done"
    });

    assert_success_without_output(run_codex_hook(&home, &prompt));
    assert_success_without_output(run_codex_hook(&home, &prompt));
    assert_success_without_output(run_codex_hook(&home, &stop));
}

const THREAD_ID: &str = "01a0d482-236c-7731-9447-cfcaa7c28964";
/// Upper bound for the detached reconcile child to finish in a test.
const BACKGROUND_RECONCILE_WAIT: std::time::Duration = std::time::Duration::from_secs(20);
const BACKGROUND_RECONCILE_POLL: std::time::Duration = std::time::Duration::from_millis(50);

fn write_rollout(dir: &Path, reply: &str) -> std::path::PathBuf {
    let path = dir.join(format!("rollout-2026-09-24T19-41-33-{THREAD_ID}.jsonl"));
    let lines = [
        json!({"type": "session_meta", "payload": {"id": THREAD_ID, "session_id": THREAD_ID}}),
        json!({"type": "response_item", "payload": {"type": "message", "role": "assistant",
               "content": [{"type": "output_text", "text": reply}]}}),
    ];
    let content: String = lines.iter().map(|line| format!("{line}\n")).collect();
    std::fs::write(&path, content).unwrap();
    path
}

fn captured_canonical_types(home: &Path) -> Vec<String> {
    let raw = home
        .join("raw/codex")
        .join(format!("codex_{THREAD_ID}.jsonl"));
    std::fs::read_to_string(raw)
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap()["canonical_type"].to_string())
        .collect()
}

#[test]
fn codex_stop_hook_reconciles_the_rollout_in_the_background() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let rollout = write_rollout(temp.path(), "background reconcile reply");
    let stop = json!({
        "session_id": THREAD_ID,
        "hook_event_name": "Stop",
        "stop_hook_active": false,
        "transcript_path": rollout,
        "last_assistant_message": "background reconcile reply"
    });

    assert_success_without_output(run_codex_hook(&home, &stop));

    let deadline = std::time::Instant::now() + BACKGROUND_RECONCILE_WAIT;
    let assistant = "\"assistant.message\"".to_string();
    while !captured_canonical_types(&home).contains(&assistant) {
        assert!(
            std::time::Instant::now() < deadline,
            "rollout reply not captured: {:?}",
            captured_canonical_types(&home)
        );
        std::thread::sleep(BACKGROUND_RECONCILE_POLL);
    }
}

#[test]
fn ingest_codex_rollout_imports_new_lines() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let rollout = write_rollout(temp.path(), "manual reconcile reply");
    let output = Command::new(env!("CARGO_BIN_EXE_nabu"))
        .arg("--home")
        .arg(&home)
        .args([
            "ingest",
            "codex-rollout",
            "--thread-id",
            THREAD_ID,
            "--path",
        ])
        .arg(&rollout)
        .output()
        .expect("run codex-rollout ingest");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("imported 2 new events"));
    assert_eq!(
        captured_canonical_types(&home),
        vec!["\"session.started\"", "\"assistant.message\""]
    );
}

#[test]
fn ingest_codex_rollout_refuses_another_thread() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let rollout = write_rollout(temp.path(), "wrong thread reply");
    let output = Command::new(env!("CARGO_BIN_EXE_nabu"))
        .arg("--home")
        .arg(&home)
        .args([
            "ingest",
            "codex-rollout",
            "--thread-id",
            "01a0aa82-77d8-7b71-8102-3359a040fdbd",
            "--path",
        ])
        .arg(&rollout)
        .output()
        .expect("run codex-rollout ingest");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("belongs to thread"));
    assert!(captured_canonical_types(&home).is_empty());
}
