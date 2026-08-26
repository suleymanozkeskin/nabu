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
