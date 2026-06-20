//! Unit and integration tests for nabu-core.
//!
//! Swept verbatim from the lib.rs `mod tests` block during the module split;
//! kept as a sibling test module so lib.rs stays a thin facade.

use super::*;
#[cfg(feature = "semantic")]
use rusqlite::params;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use tempfile::tempdir;

const SEMANTIC_RETRIEVAL_FIXTURE_JSON: &str =
    include_str!("../../../fixtures/semantic/retrieval.json");

#[derive(Debug, Deserialize)]
struct SemanticRetrievalFixture {
    schema_version: u32,
    tool: Tool,
    session_id: String,
    cwd: String,
    project_root: String,
    events: Vec<SemanticRetrievalEvent>,
    queries: Vec<SemanticRetrievalQuery>,
}

#[derive(Debug, Deserialize)]
struct SemanticRetrievalEvent {
    event_id: String,
    role: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct SemanticRetrievalQuery {
    query: String,
    relevant_event_ids: Vec<String>,
}

#[cfg(feature = "semantic")]
struct FakeEmbedder {
    batch_size: usize,
    intra_threads: usize,
    fail_on_call: Option<usize>,
    calls: std::cell::Cell<usize>,
}

#[cfg(feature = "semantic")]
impl FakeEmbedder {
    fn new(batch_size: usize, intra_threads: usize, fail_on_call: Option<usize>) -> Self {
        Self {
            batch_size,
            intra_threads,
            fail_on_call,
            calls: std::cell::Cell::new(0),
        }
    }
}

#[cfg(feature = "semantic")]
impl Embedder for FakeEmbedder {
    fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>> {
        let call = self.calls.get().saturating_add(1);
        self.calls.set(call);
        if self.fail_on_call == Some(call) {
            return Err(Error::SemanticUnavailable(format!(
                "fake embed failure on call {call}"
            )));
        }
        Ok(documents
            .iter()
            .map(|document| {
                let mut vector = vec![0.0; SEMANTIC_VECTOR_DIMENSIONS];
                vector[0] = document.len().max(1) as f32;
                vector
            })
            .collect())
    }

    fn embed_query(&self, _query: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; SEMANTIC_VECTOR_DIMENSIONS];
        vector[0] = 1.0;
        Ok(vector)
    }

    fn document_batch_size(&self) -> usize {
        self.batch_size
    }

    fn intra_threads(&self) -> usize {
        self.intra_threads
    }
}

#[test]
fn envelope_validation_rejects_invalid_enum_values() {
    for (field, value) in [
        ("tool", "bad-tool"),
        ("source", "bad-source"),
        ("canonical_type", "bad.type"),
    ] {
        let mut envelope = valid_envelope_json();
        envelope[field] = json!(value);

        let result = serde_json::from_value::<EventEnvelope>(envelope);
        assert!(result.is_err(), "{field} should reject {value}");
    }
}

#[test]
fn envelope_validation_rejects_mismatched_filename_session_id() {
    let mut envelope: EventEnvelope = serde_json::from_value(valid_envelope_json()).unwrap();
    envelope.filename_session_id = "wrong".to_string();

    assert!(envelope.validate().is_err());
}

fn purge_opts(keep_model: bool, keep_config: bool, dry_run: bool) -> PurgeAllOptions {
    PurgeAllOptions {
        keep_model,
        keep_config,
        dry_run,
    }
}

#[test]
fn purge_all_dry_run_removes_nothing() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    fs::write(home.join("raw/claude/x.jsonl"), b"{}\n").unwrap();

    let report = purge_all(&home, purge_opts(false, false, true)).unwrap();

    assert!(report.dry_run);
    assert!(home.join("raw").is_dir(), "dry run must not delete raw");
    assert!(home.join("index").is_dir());
    assert!(home.join("config.toml").is_file());
    assert!(report.authoritative_in_scope);
    assert_eq!(report.bytes_reclaimed, 0);
    assert!(report.bytes_in_scope > 0);
    let raw = report.artifacts.iter().find(|a| a.name == "raw").unwrap();
    assert_eq!(raw.action, PurgeAction::WouldRemove);
    assert_eq!(raw.tier, PurgeTier::Authoritative);
}

#[test]
fn purge_all_removes_known_artifacts_but_keeps_home_and_foreign_files() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    fs::write(home.join("NOTES.txt"), b"keep me").unwrap();

    let report = purge_all(&home, purge_opts(false, false, false)).unwrap();

    assert!(home.is_dir(), "home directory itself must remain");
    for gone in crate::purge::PURGE_KNOWN_ENTRIES {
        assert!(!home.join(gone).exists(), "{gone} should be removed");
    }
    assert!(
        home.join("NOTES.txt").is_file(),
        "foreign files must be left untouched"
    );
    assert!(report
        .unknown_entries
        .iter()
        .any(|p| p.ends_with("NOTES.txt")));
    assert!(report.authoritative_in_scope);
    assert_eq!(
        report
            .artifacts
            .iter()
            .find(|a| a.name == "raw")
            .unwrap()
            .action,
        PurgeAction::Removed
    );
    assert!(report.bytes_reclaimed > 0);
}

#[test]
fn purge_all_keep_model_and_config_preserves_them() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    fs::write(home.join("models/model.bin"), b"weights").unwrap();

    let report = purge_all(&home, purge_opts(true, true, false)).unwrap();

    assert!(home.join("models").is_dir(), "models kept");
    assert!(home.join("models/model.bin").is_file());
    assert!(home.join("config.toml").is_file(), "config kept");
    assert!(!home.join("raw").exists());
    assert!(!home.join("index").exists());
    assert_eq!(
        report
            .artifacts
            .iter()
            .find(|a| a.name == "models")
            .unwrap()
            .action,
        PurgeAction::Preserved
    );
    assert_eq!(
        report
            .artifacts
            .iter()
            .find(|a| a.name == "config.toml")
            .unwrap()
            .action,
        PurgeAction::Preserved
    );
}

#[test]
fn purge_all_refuses_non_store_directory() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("not-a-store");
    fs::create_dir_all(&home).unwrap();
    fs::write(home.join("random.txt"), b"x").unwrap();

    let err = purge_all(&home, purge_opts(false, false, true)).unwrap_err();
    assert!(matches!(err, Error::Validation(_)));
    assert!(
        home.join("random.txt").is_file(),
        "nothing removed on refusal"
    );
}

#[test]
fn purge_all_missing_home_is_idempotent_noop() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("never-created");

    let report = purge_all(&home, purge_opts(false, false, false)).unwrap();
    assert!(report.artifacts.is_empty());
    assert_eq!(report.bytes_reclaimed, 0);
    assert!(!report.authoritative_in_scope);
}

#[cfg(unix)]
#[test]
fn purge_all_removes_model_symlink_not_its_target() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let external = temp.path().join("external-models");
    fs::create_dir_all(&external).unwrap();
    fs::write(external.join("weights.bin"), b"important").unwrap();
    fs::remove_dir_all(home.join("models")).unwrap();
    std::os::unix::fs::symlink(&external, home.join("models")).unwrap();

    let report = purge_all(&home, purge_opts(false, false, false)).unwrap();

    assert!(!home.join("models").exists(), "symlink unlinked");
    assert!(external.is_dir(), "symlink target preserved");
    assert!(
        external.join("weights.bin").is_file(),
        "target contents preserved"
    );
    assert_eq!(
        report
            .artifacts
            .iter()
            .find(|a| a.name == "models")
            .unwrap()
            .action,
        PurgeAction::Removed
    );
}

#[test]
fn init_home_creates_required_layout_and_valid_sqlite_database() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");

    let report = init_home(&home).unwrap();

    for relative in [
        ".",
        "raw",
        "raw/codex",
        "raw/claude",
        "raw/opencode",
        "spool",
        "spool/dedupe",
        "checkpoints",
        "blobs/sha256",
        "models",
        "logs",
        "backups",
    ] {
        assert!(home.join(relative).is_dir(), "{relative} should exist");
        assert_private_dir_mode(&home.join(relative));
    }
    assert_private_file_mode(&home.join("config.toml"));
    assert!(report.db_path.is_file());
    assert_private_file_mode(&report.db_path);
    if report.db_path.with_file_name("harness.db-wal").exists() {
        assert_private_file_mode(&report.db_path.with_file_name("harness.db-wal"));
    }
    if report.db_path.with_file_name("harness.db-shm").exists() {
        assert_private_file_mode(&report.db_path.with_file_name("harness.db-shm"));
    }

    let conn = Connection::open(&report.db_path).unwrap();
    let integrity: String = conn
        .query_row("PRAGMA integrity_check;", [], |row| row.get(0))
        .unwrap();
    let user_version: i64 = conn
        .query_row("PRAGMA user_version;", [], |row| row.get(0))
        .unwrap();

    assert_eq!(integrity, "ok");
    assert_eq!(user_version, 1);
    assert_eq!(opencode_server_url(&home).unwrap(), None);
}

#[test]
fn open_index_rebuilds_current_shaped_but_empty_fts_table() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "fts-recovery-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "fts-recovery-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "interrupted fts rebuild recovery marker"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let db_path = home.join("index").join("harness.db");
    {
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch("DROP TABLE IF EXISTS events_fts;")
            .unwrap();
        conn.execute_batch(crate::db::EVENTS_FTS_SCHEMA).unwrap();
    }

    open_index(&db_path).unwrap();

    let results = search_history(&home, "interrupted fts rebuild recovery", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, "fts-recovery-session");
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_feature_loads_sqlite_vec_with_bundled_rusqlite() {
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(crate::db::sqlite_vec_auto_extension()));
    }

    let conn = Connection::open_in_memory().unwrap();
    let version: String = conn
        .query_row("select vec_version()", [], |row| row.get(0))
        .unwrap();
    assert!(version.starts_with('v'), "{version}");

    conn.execute(
        "create virtual table vectors using vec0(embedding float[4])",
        [],
    )
    .unwrap();
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_feature_initializes_derived_vector_schema() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();

    assert!(table_exists(&conn, &db_path, "vector_units").unwrap());
    assert!(table_exists(&conn, &db_path, "vector_unit_texts").unwrap());
    assert!(table_exists(&conn, &db_path, "vector_unit_embeddings").unwrap());
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
        0
    );
    let version: String = conn
        .query_row("select vec_version()", [], |row| row.get(0))
        .unwrap();
    assert!(version.starts_with('v'), "{version}");

    let footprint = storage_footprint(&home);
    assert_eq!(footprint.vectors_bytes, 0);
    assert!(footprint.index_bytes > 0);
}

#[test]
fn embeddinggemma_prompt_prefixes_are_pinned() {
    assert_eq!(
        query_embedding_input("  auth bug  "),
        "task: search result | query: auth bug"
    );
    assert_eq!(
        document_embedding_input("  fixed login timeout  "),
        "title: none | text: fixed login timeout"
    );
    assert_ne!(
        query_embedding_input("same text"),
        document_embedding_input("same text")
    );
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_vectors_persist_in_sqlite_vec() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();
    ensure_semantic_vector_schema(&conn, &db_path).unwrap();

    let mut vector = vec![0.0_f32; SEMANTIC_VECTOR_DIMENSIONS];
    vector[0] = 1.0;
    let vector_blob = vector_to_blob(&vector).unwrap();
    conn.execute(
        "INSERT INTO vector_unit_embeddings(unit_id, embedding) VALUES (?1, ?2)",
        params![1_i64, vector_blob.clone()],
    )
    .unwrap();

    let unit_id: i64 = conn
        .query_row(
            "SELECT unit_id FROM vector_unit_embeddings
                 WHERE embedding MATCH ?1 AND k = 1
                 ORDER BY distance",
            params![vector_blob],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(unit_id, 1);
    assert_eq!(storage_footprint(&home).vectors_bytes, 1024);
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_index_materializes_units_without_model_or_payload_duplication() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "semantic-units",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "semantic-units-1",
            "prompt": "remember the fuzzy auth regression fix",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture"
        }),
    )
    .unwrap();

    index_once(&home).unwrap();
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();

    assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 1);
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
        1
    );
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
        0
    );
    let payload_json: Option<String> = conn
        .query_row("SELECT payload_json FROM events LIMIT 1", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(payload_json.is_none());
    assert!(!embedding_model_status(&home).semantic_available);
}

#[test]
fn opencode_hook_resolves_session_id_from_native_event_shapes() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    // Message/part/tool/etc. events carry top-level `sessionID`; live plugin
    // payloads may nest the object under `info`/`part`.
    let by_session_id = ingest_hook_event(
        &home,
        Tool::Opencode,
        json!({
            "hook_event_name": "message.updated",
            "id": "msg_abc",
            "sessionID": "ses_top_level",
            "role": "assistant"
        }),
    )
    .unwrap();
    assert!(by_session_id.appended);
    assert!(by_session_id
        .raw_file
        .to_string_lossy()
        .contains("ses_top_level"));

    let nested_part = ingest_hook_event(
        &home,
        Tool::Opencode,
        json!({
            "hook_event_name": "message.part.updated",
            "part": { "id": "prt_1", "sessionID": "ses_nested_part", "type": "text" }
        }),
    )
    .unwrap();
    assert!(nested_part
        .raw_file
        .to_string_lossy()
        .contains("ses_nested_part"));

    // `session.*` events have no `sessionID`; the session id is `id`.
    let session_created = ingest_hook_event(
        &home,
        Tool::Opencode,
        json!({
            "hook_event_name": "session.created",
            "id": "ses_from_id",
            "directory": "/tmp/project"
        }),
    )
    .unwrap();
    assert!(session_created
        .raw_file
        .to_string_lossy()
        .contains("ses_from_id"));
}

#[test]
fn opencode_hook_does_not_mistake_message_id_for_session_id() {
    // A non-session event with `id` but no `sessionID` must NOT fall back to
    // `id` (that would be the message id, not the session id).
    let payload = json!({
        "hook_event_name": "message.updated",
        "id": "msg_no_session"
    });
    let result = opencode_hook_session_id(&payload, "message.updated");
    assert!(matches!(result, Err(Error::Validation(_))));
}

#[test]
fn opencode_hook_rejects_event_without_resolvable_session_id() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let result = ingest_hook_event(
        &home,
        Tool::Opencode,
        json!({ "hook_event_name": "file.edited", "filename": "src/lib.rs" }),
    );
    assert!(matches!(result, Err(Error::Validation(_))));
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_index_no_embed_skips_fake_model_and_leaves_vectors_empty() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    write_fake_semantic_model_files(&home);
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "semantic-no-embed",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "semantic-no-embed-1",
            "prompt": "deferred semantic fake model marker",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture"
        }),
    )
    .unwrap();

    let report = index_once_with_options(&home, IndexOptions { embed: false }).unwrap();
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();

    assert_eq!(report.indexed_events, 1);
    assert_eq!(
        search_history(&home, "deferred semantic fake model", 10)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 1);
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
        0
    );
    assert!(!embedding_model_status(&home).semantic_available);
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_index_discloses_unembedded_count_before_model_load() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    write_fake_semantic_model_files(&home);
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "semantic-plan",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "semantic-plan-1",
            "prompt": "semantic plan progress marker",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture"
        }),
    )
    .unwrap();

    let mut progress = Vec::new();
    let result = index_once_with_options_and_progress(&home, IndexOptions::default(), |event| {
        progress.push(event)
    });

    assert!(
        result.is_err(),
        "fake model files should make model loading fail after the plan is emitted"
    );
    assert_eq!(progress.first().unwrap().phase, "embedding_plan");
    assert_eq!(progress.first().unwrap().status, "ready");
    assert_eq!(progress.first().unwrap().total_units, 1);
    assert!(progress
        .iter()
        .any(|event| event.phase == "loading_model" && event.status == "started"));
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_collect_uses_compact_unit_texts_and_backfills_legacy_rows() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "semantic-texts",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "semantic-texts-1",
            "prompt": "compact vector unit text marker",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
        1
    );
    conn.execute("DELETE FROM vector_unit_texts", []).unwrap();
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
        0
    );

    let units = collect_unembedded_units(&conn, &db_path).unwrap();
    assert_eq!(units.len(), 1);
    assert!(units[0].text.contains("compact vector unit text marker"));
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_texts").unwrap(),
        1
    );
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_index_does_not_load_fake_model_when_no_units_need_embedding() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    write_fake_semantic_model_files(&home);

    let mut progress = Vec::new();
    let embedded =
        embed_index_if_available_with_progress(&home, |event| progress.push(event)).unwrap();

    assert_eq!(embedded, 0);
    assert!(progress.is_empty());
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_model_download_is_noop_when_cache_is_complete() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    write_fake_semantic_model_files(&home);

    let mut progress = Vec::new();
    let report = download_embedding_model_with_progress(&home, SEMANTIC_MODEL_ID, |event| {
        progress.push(event)
    })
    .unwrap();

    assert!(progress.is_empty());
    assert_eq!(report.downloaded_files, 0);
    assert_eq!(report.total_files, SEMANTIC_MODEL_REMOTE_FILES.len());
    assert!(report.on_disk_bytes > 0);
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_embedding_batches_by_length_and_streams_progress() {
    let mut units = vec![
        UnembeddedUnit {
            unit_id: 2,
            text: "x ".repeat(200),
            estimated_tokens: 200,
        },
        UnembeddedUnit {
            unit_id: 3,
            text: "tiny".to_string(),
            estimated_tokens: estimated_embedding_token_count("tiny"),
        },
        UnembeddedUnit {
            unit_id: 1,
            text: "short unit".to_string(),
            estimated_tokens: estimated_embedding_token_count("short unit"),
        },
    ];
    bucket_unembedded_units(&mut units);
    assert_eq!(
        units.iter().map(|unit| unit.unit_id).collect::<Vec<_>>(),
        vec![1, 3, 2]
    );

    let progress = embedding_index_progress(
        "embedding",
        "running",
        50,
        100,
        Instant::now() - StdDuration::from_secs(2),
        &FakeEmbedder::new(64, 8, None),
        2048,
    );
    assert_eq!(progress.batch_size, 64);
    assert_eq!(progress.write_chunk_size, 2048);
    assert_eq!(progress.intra_threads, 8);
    assert!(progress.units_per_second > 0.0);
    assert!(progress.eta_seconds.is_some());
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_embedding_writes_commit_in_resumable_chunks() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    for index in 0..3 {
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "semantic-resume",
                "hook_event_name": "UserPromptSubmit",
                "message_id": format!("semantic-resume-{index}"),
                "sequence": index,
                "prompt": format!("semantic resumable embedding unit {index}"),
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture"
            }),
        )
        .unwrap();
    }
    index_once(&home).unwrap();

    let db_path = home.join("index").join("harness.db");
    let mut conn = open_index(&db_path).unwrap();
    assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 3);

    let failing = FakeEmbedder::new(1, 4, Some(3));
    let mut failed_progress = Vec::new();
    let result = embed_unembedded_units_with_config(
        &mut conn,
        &db_path,
        &failing,
        EmbeddingWriteConfig {
            write_chunk_size: 2,
        },
        |event| failed_progress.push(event),
    );
    assert!(result.is_err());
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
        2
    );
    assert_eq!(failed_progress.first().unwrap().status, "started");

    let succeeding = FakeEmbedder::new(1, 4, None);
    let mut resumed_progress = Vec::new();
    let embedded = embed_unembedded_units_with_config(
        &mut conn,
        &db_path,
        &succeeding,
        EmbeddingWriteConfig {
            write_chunk_size: 2,
        },
        |event| resumed_progress.push(event),
    )
    .unwrap();
    assert_eq!(embedded, 1);
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
        3
    );
    assert_eq!(resumed_progress.first().unwrap().status, "started");
    assert_eq!(resumed_progress.last().unwrap().status, "completed");
    assert_eq!(resumed_progress.last().unwrap().embedded_units, 1);
    assert_eq!(resumed_progress.last().unwrap().total_units, 1);
}

#[cfg(feature = "semantic")]
#[test]
#[ignore = "semantic acceptance requires a local embedding model cache"]
fn semantic_acceptance_no_embed_defers_vectors_until_later_default_index() {
    let model_home = required_semantic_test_model_home();
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    attach_semantic_model_cache(&home, &model_home);
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "semantic-deferred-real-model",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "semantic-deferred-real-model-1",
            "prompt": "defer semantic embedding until the later default index pass",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture"
        }),
    )
    .unwrap();

    let first = index_once_with_options(&home, IndexOptions { embed: false }).unwrap();
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();
    assert_eq!(first.indexed_events, 1);
    assert_eq!(table_count(&conn, &db_path, "vector_units").unwrap(), 1);
    assert_eq!(
        table_count(&conn, &db_path, "vector_unit_embeddings").unwrap(),
        0
    );
    assert!(!embedding_model_status(&home).semantic_available);

    let mut progress = Vec::new();
    let second = index_once_with_options_and_progress(&home, IndexOptions::default(), |event| {
        progress.push(event)
    })
    .unwrap();
    assert_eq!(second.indexed_events, 0);
    assert!(
        table_count(&conn, &db_path, "vector_unit_embeddings").unwrap() > 0,
        "default index should embed units deferred by --no-embed"
    );
    assert!(embedding_model_status(&home).semantic_available);
    assert_eq!(progress.first().unwrap().phase, "embedding_plan");
    assert_eq!(progress.first().unwrap().total_units, 1);
}

#[cfg(feature = "semantic")]
#[test]
#[ignore = "semantic acceptance requires a local embedding model cache"]
fn semantic_acceptance_hybrid_beats_lexical_on_labeled_retrieval_fixture() {
    let model_home = required_semantic_test_model_home();
    let fixture = semantic_retrieval_fixture();
    assert!(!fixture.events.is_empty());
    assert!(!fixture.queries.is_empty());

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    attach_semantic_model_cache(&home, &model_home);
    seed_semantic_retrieval_fixture(&home, &fixture);
    index_once(&home).unwrap();
    assert!(embedding_model_status(&home).semantic_available);

    let k = 3usize;
    let first_results = hybrid_result_ids_by_query(&home, &fixture, k);
    let first_vectors = vector_snapshot(&home);
    assert!(!first_vectors.is_empty());

    let mut strict_wins = 0usize;
    let mut aggregate_lexical_precision = 0.0;
    let mut aggregate_hybrid_precision = 0.0;
    let mut aggregate_lexical_recall = 0.0;
    let mut aggregate_hybrid_recall = 0.0;
    for query in &fixture.queries {
        let relevant = query
            .relevant_event_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let lexical = result_event_ids(
            &home,
            search_history_page(
                &home,
                &query.query,
                SearchOptions {
                    mode: SearchMode::Lexical,
                    limit: k,
                    dedupe: false,
                    ..SearchOptions::default()
                },
            )
            .unwrap()
            .results,
        );
        let hybrid = result_event_ids(
            &home,
            search_history_page(
                &home,
                &query.query,
                SearchOptions {
                    mode: SearchMode::Hybrid,
                    limit: k,
                    dedupe: false,
                    ..SearchOptions::default()
                },
            )
            .unwrap()
            .results,
        );
        let lexical_precision = precision_at_k(&lexical, &relevant, k);
        let hybrid_precision = precision_at_k(&hybrid, &relevant, k);
        let lexical_recall = recall_at_k(&lexical, &relevant, k);
        let hybrid_recall = recall_at_k(&hybrid, &relevant, k);
        aggregate_lexical_precision += lexical_precision;
        aggregate_hybrid_precision += hybrid_precision;
        aggregate_lexical_recall += lexical_recall;
        aggregate_hybrid_recall += hybrid_recall;

        eprintln!(
                "semantic fixture query={:?} lexical_ids={:?} hybrid_ids={:?} precision@{} lexical={:.3} hybrid={:.3} recall@{} lexical={:.3} hybrid={:.3}",
                query.query,
                lexical,
                hybrid,
                k,
                lexical_precision,
                hybrid_precision,
                k,
                lexical_recall,
                hybrid_recall
            );
        assert!(
            hybrid_precision >= lexical_precision,
            "hybrid precision regressed for query {:?}: lexical {:?} ({:.3}), hybrid {:?} ({:.3})",
            query.query,
            lexical,
            lexical_precision,
            hybrid,
            hybrid_precision
        );
        assert!(
            hybrid_recall >= lexical_recall,
            "hybrid recall regressed for query {:?}: lexical {:?} ({:.3}), hybrid {:?} ({:.3})",
            query.query,
            lexical,
            lexical_recall,
            hybrid,
            hybrid_recall
        );
        if hybrid_precision > lexical_precision || hybrid_recall > lexical_recall {
            strict_wins += 1;
        }
    }
    let query_count = fixture.queries.len() as f64;
    eprintln!(
            "semantic fixture aggregate precision@{} lexical={:.3} hybrid={:.3} recall@{} lexical={:.3} hybrid={:.3} strict_wins={}/{}",
            k,
            aggregate_lexical_precision / query_count,
            aggregate_hybrid_precision / query_count,
            k,
            aggregate_lexical_recall / query_count,
            aggregate_hybrid_recall / query_count,
            strict_wins,
            fixture.queries.len()
        );
    assert!(
            strict_wins > 0,
            "hybrid tied lexical on every labeled semantic query; this does not prove the M5 retrieval-quality win"
        );

    remove_index_database(&home);
    index_once(&home).unwrap();
    let second_vectors = vector_snapshot(&home);
    let second_results = hybrid_result_ids_by_query(&home, &fixture, k);

    assert_eq!(first_vectors, second_vectors);
    assert_eq!(first_results, second_results);
}

fn semantic_retrieval_fixture() -> SemanticRetrievalFixture {
    serde_json::from_str(SEMANTIC_RETRIEVAL_FIXTURE_JSON)
        .expect("semantic retrieval fixture must be valid JSON")
}

#[cfg(feature = "semantic")]
fn write_fake_semantic_model_files(home: &Path) {
    let model_root = semantic_model_cache_path(home);
    for (_, local) in SEMANTIC_MODEL_REMOTE_FILES {
        let path = model_root.join(local);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"not a real model").unwrap();
    }
}

#[cfg(feature = "semantic")]
fn required_semantic_test_model_home() -> PathBuf {
    semantic_test_model_home().expect(
        "semantic acceptance tests require NABU_SEMANTIC_MODEL_DIR or \
             NABU_SEMANTIC_TEST_HOME to point at a downloaded embeddinggemma-300m-q4 cache",
    )
}

#[cfg(feature = "semantic")]
fn semantic_test_model_home() -> Option<PathBuf> {
    if let Ok(model_dir) = std::env::var("NABU_SEMANTIC_MODEL_DIR") {
        let model_dir = PathBuf::from(model_dir);
        if semantic_model_files_present_at(&model_dir) {
            return Some(model_dir);
        }
    }

    let mut candidates = Vec::new();
    if let Ok(home) = std::env::var("NABU_SEMANTIC_TEST_HOME") {
        candidates.push(PathBuf::from(home));
    }
    if let Ok(home) = std::env::var("NABU_HOME") {
        candidates.push(PathBuf::from(home));
    }
    if let Ok(home) = resolve_home(None) {
        candidates.push(home);
    }

    candidates.into_iter().find_map(|home| {
        let cache_path = semantic_model_cache_path(&home);
        semantic_model_files_present_at(&cache_path).then_some(cache_path)
    })
}

#[cfg(feature = "semantic")]
fn semantic_model_files_present_at(cache_path: &Path) -> bool {
    SEMANTIC_MODEL_REMOTE_FILES
        .iter()
        .all(|(_, local)| cache_path.join(local).is_file())
}

#[cfg(feature = "semantic")]
fn attach_semantic_model_cache(home: &Path, source_cache_path: &Path) {
    let model_root = home.join("models");
    fs::create_dir_all(&model_root).unwrap();
    let target = semantic_model_cache_path(home);
    if target.exists() {
        return;
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(source_cache_path, &target).unwrap();
    }
    #[cfg(not(unix))]
    {
        let _ = source_cache_path;
        panic!("semantic model cache symlink test requires a Unix platform");
    }
}

#[cfg(feature = "semantic")]
fn seed_semantic_retrieval_fixture(home: &Path, fixture: &SemanticRetrievalFixture) {
    for (index, event) in fixture.events.iter().enumerate() {
        let mut payload = json!({
            "session_id": fixture.session_id,
            "message_id": event.event_id,
            "cwd": fixture.cwd,
            "project_root": fixture.project_root,
        });
        match event.role.as_str() {
            "user" => {
                payload["hook_event_name"] = json!("UserPromptSubmit");
                payload["prompt"] = json!(event.text);
            }
            "assistant" => {
                payload["hook_event_name"] = json!("MessageDisplay");
                payload["text"] = json!(event.text);
                payload["index"] = json!(index as i64);
                payload["final"] = json!(true);
            }
            role => panic!("unsupported semantic fixture role: {role}"),
        }
        ingest_hook_event(home, Tool::Claude, payload).unwrap();
    }
}

#[cfg(feature = "semantic")]
fn result_event_ids(home: &Path, results: Vec<SearchResult>) -> Vec<String> {
    results
        .iter()
        .map(|result| result_event_id(home, result))
        .collect()
}

#[cfg(feature = "semantic")]
fn result_event_id(home: &Path, result: &SearchResult) -> String {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();
    conn.query_row(
        "SELECT COALESCE(message_id, source_event_id, CAST(id AS TEXT))
             FROM events
             WHERE tool = ?1
               AND session_id = ?2
               AND raw_file = ?3
               AND raw_line = ?4
             ORDER BY id
             LIMIT 1",
        params![
            result.tool.as_str(),
            &result.session_id,
            &result.raw_file,
            result.raw_line,
        ],
        |row| row.get(0),
    )
    .unwrap()
}

#[cfg(feature = "semantic")]
fn precision_at_k(ids: &[String], relevant: &BTreeSet<String>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    relevant_hits_at_k(ids, relevant, k) as f64 / k as f64
}

#[cfg(feature = "semantic")]
fn recall_at_k(ids: &[String], relevant: &BTreeSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    relevant_hits_at_k(ids, relevant, k) as f64 / relevant.len() as f64
}

#[cfg(feature = "semantic")]
fn relevant_hits_at_k(ids: &[String], relevant: &BTreeSet<String>, k: usize) -> usize {
    ids.iter()
        .take(k)
        .filter(|event_id| relevant.contains(*event_id))
        .count()
}

#[cfg(feature = "semantic")]
fn hybrid_result_ids_by_query(
    home: &Path,
    fixture: &SemanticRetrievalFixture,
    k: usize,
) -> Vec<Vec<String>> {
    fixture
        .queries
        .iter()
        .map(|query| {
            result_event_ids(
                home,
                search_history_page(
                    home,
                    &query.query,
                    SearchOptions {
                        mode: SearchMode::Hybrid,
                        limit: k,
                        dedupe: false,
                        ..SearchOptions::default()
                    },
                )
                .unwrap()
                .results,
            )
        })
        .collect()
}

#[cfg(feature = "semantic")]
fn vector_snapshot(home: &Path) -> Vec<(i64, Vec<u8>)> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();
    let mut statement = conn
        .prepare("SELECT unit_id, embedding FROM vector_unit_embeddings ORDER BY unit_id")
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .unwrap();
    rows.map(|row| row.unwrap()).collect()
}

#[cfg(feature = "semantic")]
fn remove_index_database(home: &Path) {
    let db_path = home.join("index").join("harness.db");
    for path in [
        db_path.clone(),
        db_path.with_file_name("harness.db-wal"),
        db_path.with_file_name("harness.db-shm"),
    ] {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("failed to remove {}: {error}", path.display()),
        }
    }
}

#[test]
fn opencode_server_url_reads_config_toml_key() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    fs::write(
        home.join("config.toml"),
        "schema_version = 1\n\n[opencode]\nserver_url = \"http://127.0.0.1:4096\"\n",
    )
    .unwrap();

    assert_eq!(
        opencode_server_url(&home).unwrap(),
        Some("http://127.0.0.1:4096".to_string())
    );
}

#[cfg(unix)]
fn assert_private_dir_mode(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let mode = fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700, "{} should be 0700", path.display());
}

#[cfg(not(unix))]
fn assert_private_dir_mode(_path: &Path) {}

#[cfg(unix)]
fn assert_private_file_mode(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let mode = fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "{} should be 0600", path.display());
}

#[cfg(not(unix))]
fn assert_private_file_mode(_path: &Path) {}

#[test]
fn redaction_rules_match_contract_and_preserve_safe_text() {
    let fixture = include_str!("../../../fixtures/redaction/secrets.txt");
    let redacted = redact_export_text(fixture);

    for expected in [
        "[REDACTED:PRIVATE_KEY]",
        "Bearer [REDACTED:BEARER_TOKEN]",
        "[REDACTED:API_KEY]",
        "DATABASE_PASSWORD=[REDACTED:ENV_VALUE]",
    ] {
        assert!(redacted.contains(expected), "{expected}");
    }
    for secret in [
        "private-key-material",
        "abcdefghijklmnopqrstuvwxyz123456",
        "supersecretvalue",
        "AKIA1234567890ABCDEF",
    ] {
        assert!(!redacted.contains(secret), "{secret}");
    }
    assert!(redacted.contains("redaction fixture marker keeps safe surrounding text"));
    assert!(redacted.contains("redaction fixture marker keeps trailing safe text"));
}

#[test]
fn oversized_payloads_are_spilled_and_indexed_from_blob() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let prompt =
        "oversized payload fixture marker ".repeat((MAX_INLINE_ENVELOPE_BYTES / 32) + 1024);

    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "fixture-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "oversized-payload-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": prompt
        }),
    )
    .unwrap();

    let raw_path = canonical_raw_path(&home, Tool::Claude, "fixture-session");
    let raw = fs::read_to_string(&raw_path).unwrap();
    let envelope: EventEnvelope = serde_json::from_str(raw.trim_end()).unwrap();
    let payload_ref = envelope.payload_ref.as_deref().unwrap();
    assert!(payload_ref.starts_with("sha256:"));
    assert!(envelope.payload.is_null());
    let hash = payload_ref.trim_start_matches("sha256:");
    assert!(home
        .join("blobs")
        .join("sha256")
        .join(format!("{hash}.json"))
        .is_file());
    assert!(raw.len() < MAX_INLINE_ENVELOPE_BYTES);

    index_once(&home).unwrap();
    let page = search_history_page(
        &home,
        "oversized payload fixture marker",
        SearchOptions {
            limit: 1,
            include_payload: true,
            dedupe: false,
            max_snippet_chars: 80,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(page.results.len(), 1);
    assert_eq!(page.results[0].session_id, "fixture-session");
    assert!(page.results[0]
        .payload
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap()
        .contains("oversized payload fixture marker"));
}

#[test]
fn markdown_export_includes_sensitivity_warning() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "warning-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "warning-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "markdown warning marker"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let full = export_session_markdown_with_options(&home, Tool::Claude, "warning-session", false)
        .unwrap();
    let redacted =
        export_session_markdown_with_options(&home, Tool::Claude, "warning-session", true).unwrap();

    assert!(full.contains("Sensitivity: this export is not redacted."));
    assert!(redacted.contains("Sensitivity: redacted export."));
}

#[test]
fn raw_append_and_index_dedupe_same_native_event_across_sources() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let payload = json!({
        "session_id": "fixture-session",
        "hook_event_name": "UserPromptSubmit",
        "event_id": "same-native-event-1",
        "cwd": "/tmp/nabu-fixture",
        "project_root": "/tmp/nabu-fixture",
        "prompt": "cross source dedupe fixture marker"
    });

    let first = ingest_hook_event(&home, Tool::Codex, payload.clone()).unwrap();
    let backfill_event = envelope_from_backfill_payload(
        Tool::Codex,
        Path::new("/tmp/codex.jsonl"),
        0,
        payload,
        &BackfillParseContext::default(),
    )
    .unwrap();
    let second = append_prepared_event(&home, backfill_event).unwrap();

    let raw =
        fs::read_to_string(canonical_raw_path(&home, Tool::Codex, "fixture-session")).unwrap();
    assert!(first.appended);
    assert!(!second.appended);
    assert_eq!(raw.lines().count(), 1);
    index_once(&home).unwrap();
    let results = search_history(&home, "cross source dedupe fixture marker", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, "fixture-session");
}

#[test]
fn raw_append_dedupes_unsequenced_event_across_observation_time_and_route() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let hook_payload = json!({
        "session_id": "fixture-session",
        "hook_event_name": "UserPromptSubmit",
        "captured_at": "2026-06-17T12:00:59Z",
        "cwd": "/tmp/nabu-fixture",
        "project_root": "/tmp/nabu-fixture",
        "prompt": "unsequenced duplicate marker"
    });
    let backfill_payload = json!({
        "session_id": "fixture-session",
        "hook_event_name": "UserPromptSubmit",
        "captured_at": "2026-06-17T12:01:01Z",
        "cwd": "/tmp/nabu-fixture",
        "project_root": "/tmp/nabu-fixture",
        "prompt": "unsequenced duplicate marker"
    });

    let first = ingest_hook_event(&home, Tool::Claude, hook_payload).unwrap();
    let mut event = envelope_from_backfill_payload(
        Tool::Claude,
        Path::new("/tmp/claude-transcript.jsonl"),
        42,
        backfill_payload,
        &BackfillParseContext::default(),
    )
    .unwrap();
    event.sequence = None;
    let second = append_prepared_event(&home, event).unwrap();

    let raw =
        fs::read_to_string(canonical_raw_path(&home, Tool::Claude, "fixture-session")).unwrap();
    assert!(first.appended);
    assert!(!second.appended);
    assert_eq!(raw.lines().count(), 1);

    index_once(&home).unwrap();
    let results = search_history(&home, "unsequenced duplicate marker", 10).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn dedupe_sidecar_covers_large_session_and_self_heals() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let session_id = "large-sidecar-session";
    let seed_events = (0..10_000)
        .map(|index| {
            envelope_from_backfill_payload(
                Tool::Claude,
                Path::new("/tmp/large-sidecar.jsonl"),
                index as u64,
                json!({
                    "session_id": session_id,
                    "hook_event_name": "UserPromptSubmit",
                    "message_id": format!("large-sidecar-{index}"),
                    "sequence": index as i64,
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "prompt": format!("large sidecar marker {index}")
                }),
                &BackfillParseContext::default(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    append_prepared_events(&home, seed_events).unwrap();

    let raw_path = canonical_raw_path(&home, Tool::Claude, session_id);
    let sidecar = DedupeSidecarFiles::for_raw_file(&home, &raw_path);
    assert_eq!(raw_line_count(&raw_path), 10_000);
    assert_eq!(dedupe_sidecar_entry_count(&sidecar), 10_000);

    let duplicate = ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "message_id": "large-sidecar-1234",
            "sequence": 1234,
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "large sidecar marker 1234"
        }),
    )
    .unwrap();
    assert!(!duplicate.appended);
    assert_eq!(duplicate.raw_offset, raw_offset_for_line(&raw_path, 1234));
    assert_eq!(raw_line_count(&raw_path), 10_000);

    fs::remove_dir_all(&sidecar.buckets_dir).unwrap();
    let duplicate_after_delete = ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "message_id": "large-sidecar-4321",
            "sequence": 4321,
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "large sidecar marker 4321"
        }),
    )
    .unwrap();
    assert!(!duplicate_after_delete.appended);
    assert_eq!(raw_line_count(&raw_path), 10_000);
    assert_eq!(dedupe_sidecar_entry_count(&sidecar), 10_000);

    let corrupt_payload = json!({
        "session_id": session_id,
        "hook_event_name": "UserPromptSubmit",
        "message_id": "large-sidecar-9876",
        "sequence": 9876,
        "cwd": "/tmp/nabu-fixture",
        "project_root": "/tmp/nabu-fixture",
        "prompt": "large sidecar marker 9876"
    });
    let corrupt_event = envelope_from_backfill_payload(
        Tool::Claude,
        Path::new("/tmp/large-sidecar.jsonl"),
        9876,
        corrupt_payload.clone(),
        &BackfillParseContext::default(),
    )
    .unwrap();
    let corrupt_key = dedupe_key(DedupeParts {
        tool: corrupt_event.tool,
        session_id: &corrupt_event.session_id,
        canonical_type: corrupt_event.canonical_type,
        source_event_id: corrupt_event.source_event_id.as_deref(),
        sequence: corrupt_event.sequence,
        payload: &corrupt_event.payload,
    })
    .unwrap();
    let corrupt_bucket = crate::ingest::dedupe_bucket_index(&corrupt_key).unwrap();
    fs::write(sidecar.bucket_path(corrupt_bucket), b"sha256:truncated").unwrap();
    let duplicate_after_corruption =
        ingest_hook_event(&home, Tool::Claude, corrupt_payload).unwrap();
    assert!(!duplicate_after_corruption.appended);
    assert_eq!(raw_line_count(&raw_path), 10_000);
    assert_eq!(dedupe_sidecar_entry_count(&sidecar), 10_000);

    index_once(&home).unwrap();
    // Under OR semantics the shared "large sidecar marker" terms match every
    // seeded event; the unique "1234" term is rare, so bm25 ranks that event
    // first. Assert the targeted event is the top hit rather than the sole one.
    let results = search_history(&home, "large sidecar marker 1234", 10).unwrap();
    assert!(results[0].snippet.contains("large sidecar marker 1234"));
}

#[test]
fn concurrent_appends_keep_raw_and_sidecar_consistent() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let session_id = "concurrent-sidecar-session";
    let mut handles = Vec::new();

    for index in 0..64 {
        let home = home.clone();
        handles.push(std::thread::spawn(move || {
            ingest_hook_event(
                &home,
                Tool::Codex,
                json!({
                    "session_id": session_id,
                    "hook_event_name": "UserPromptSubmit",
                    "message_id": format!("concurrent-sidecar-{index}"),
                    "sequence": index as i64,
                    "cwd": "/tmp/nabu-fixture",
                    "project_root": "/tmp/nabu-fixture",
                    "prompt": format!("concurrent sidecar marker {index}")
                }),
            )
            .unwrap()
        }));
    }

    for handle in handles {
        assert!(handle.join().unwrap().appended);
    }

    let raw_path = canonical_raw_path(&home, Tool::Codex, session_id);
    let sidecar = DedupeSidecarFiles::for_raw_file(&home, &raw_path);
    assert_eq!(raw_line_count(&raw_path), 64);
    assert_eq!(dedupe_sidecar_entry_count(&sidecar), 64);
}

#[test]
fn native_order_preserves_identical_content_and_unordered_still_collapses() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    for sequence in [1, 2] {
        let report = ingest_hook_event(
            &home,
            Tool::Codex,
            json!({
                "session_id": "ordered-identical-session",
                "hook_event_name": "UserPromptSubmit",
                "sequence": sequence,
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": "identical ordered content marker"
            }),
        )
        .unwrap();
        assert!(report.appended);
    }
    assert_eq!(
        raw_line_count(&canonical_raw_path(
            &home,
            Tool::Codex,
            "ordered-identical-session"
        )),
        2
    );

    let first = ingest_hook_event(
        &home,
        Tool::Codex,
        json!({
            "session_id": "unordered-identical-session",
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "identical unordered content marker"
        }),
    )
    .unwrap();
    let second = ingest_hook_event(
        &home,
        Tool::Codex,
        json!({
            "session_id": "unordered-identical-session",
            "hook_event_name": "UserPromptSubmit",
            "captured_at": "2099-01-01T00:00:00Z",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "identical unordered content marker"
        }),
    )
    .unwrap();
    assert!(first.appended);
    assert!(!second.appended);
    assert_eq!(
        raw_line_count(&canonical_raw_path(
            &home,
            Tool::Codex,
            "unordered-identical-session"
        )),
        1
    );
}

#[test]
fn source_specific_ordering_fields_are_mapped_to_sequence() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    for part_index in [7, 8] {
        let report = ingest_hook_event(
            &home,
            Tool::Opencode,
            json!({
                "session_id": "opencode-part-order-session",
                "hook_event_name": "message.part.updated",
                "message_id": "shared-opencode-message",
                "part": {
                    "index": part_index,
                    "text": "same opencode part text marker"
                },
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "delta": "same opencode part text marker"
            }),
        )
        .unwrap();
        assert!(report.appended);
    }

    for item_index in [3, 4] {
        let report = ingest_hook_event(
            &home,
            Tool::Codex,
            json!({
                "session_id": "codex-item-order-session",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "item_index": item_index,
                    "content": [{"type": "output_text", "text": "same codex item text marker"}]
                },
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture"
            }),
        )
        .unwrap();
        assert!(report.appended);
    }

    let first = envelope_from_backfill_payload(
        Tool::Claude,
        Path::new("/tmp/transcript.jsonl"),
        10,
        json!({
            "session_id": "backfill-offset-order-session",
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "same backfill offset text marker"
        }),
        &BackfillParseContext::default(),
    )
    .unwrap();
    let second = envelope_from_backfill_payload(
        Tool::Claude,
        Path::new("/tmp/transcript.jsonl"),
        20,
        json!({
            "session_id": "backfill-offset-order-session",
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "same backfill offset text marker"
        }),
        &BackfillParseContext::default(),
    )
    .unwrap();
    assert!(append_prepared_event(&home, first).unwrap().appended);
    assert!(append_prepared_event(&home, second).unwrap().appended);

    assert_eq!(
        raw_line_count(&canonical_raw_path(
            &home,
            Tool::Opencode,
            "opencode-part-order-session"
        )),
        2
    );
    assert_eq!(
        raw_line_count(&canonical_raw_path(
            &home,
            Tool::Codex,
            "codex-item-order-session"
        )),
        2
    );
    assert_eq!(
        raw_line_count(&canonical_raw_path(
            &home,
            Tool::Claude,
            "backfill-offset-order-session"
        )),
        2
    );
}

#[test]
fn codex_native_transcript_backfill_derives_session_id_from_metadata() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("codex-sessions");
    init_home(&home).unwrap();
    fs::create_dir_all(&source).unwrap();

    let session_id = "019a4b44-cc3b-7c51-8944-a7d7ebb9e6fe";
    fs::write(
            source.join(format!("rollout-2025-11-03T20-49-51-{session_id}.jsonl")),
            format!(
                "{{\"timestamp\":\"2025-11-03T19:49:51.304Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session_id}\",\"cwd\":\"/tmp/native-codex\"}}}}\n\
                 {{\"timestamp\":\"2025-11-03T19:50:01.966Z\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"native codex backfill marker\"}}]}}}}\n"
            ),
        )
        .unwrap();

    let report = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
    assert_eq!(report.source_files, 1);
    assert_eq!(report.appended_events, 2);
    assert_eq!(report.checkpoint_files, 1);

    index_once(&home).unwrap();
    let results = search_history(&home, "native codex backfill marker", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool, Tool::Codex);
    assert_eq!(results[0].session_id, session_id);
    assert_eq!(results[0].canonical_type, "user.message");
}

#[cfg(unix)]
#[test]
fn backfill_skips_source_file_that_vanishes_before_read() {
    // A session file discovered during the scan can be deleted/rotated by the
    // live tool before backfill reads it (os error 2). One vanished file must
    // not abort the whole backfill. A dangling symlink reproduces a candidate
    // that passes discovery but fails with NotFound on read.
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("codex-sessions");
    init_home(&home).unwrap();
    fs::create_dir_all(&source).unwrap();

    let session_id = "019a4f57-3d5f-7f52-96cc-cb2e1eacb7a9";
    fs::write(
            source.join(format!("rollout-2025-11-04T15-48-28-{session_id}.jsonl")),
            "{\"timestamp\":\"2025-11-04T14:48:28.000Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"surviving codex marker\"}]}}\n",
        )
        .unwrap();
    // Discovered by extension, but reading it yields NotFound.
    std::os::unix::fs::symlink(
        temp.path().join("does-not-exist.jsonl"),
        source.join("rollout-2025-11-04T16-00-00-vanished.jsonl"),
    )
    .unwrap();

    // Dry run (the wizard's "Scanning past sessions…") must not fail.
    let dry = backfill_dry_run(&home, Some(Tool::Codex), &source, None).unwrap();
    assert_eq!(dry.source_files, 1);

    // The real backfill must skip the vanished file and import the valid one.
    let report = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
    assert_eq!(report.source_files, 1);
    assert_eq!(report.appended_events, 1);

    index_once(&home).unwrap();
    let results = search_history(&home, "surviving codex marker", 10).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn codex_native_transcript_backfill_derives_session_id_from_filename() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("codex-sessions");
    init_home(&home).unwrap();
    fs::create_dir_all(&source).unwrap();

    let session_id = "019a4f57-3d5f-7f52-96cc-cb2e1eacb7a9";
    fs::write(
            source.join(format!("rollout-2025-11-04T15-48-28-{session_id}.jsonl")),
            "{\"timestamp\":\"2025-11-04T14:48:28.000Z\",\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"filename derived codex marker\"}]}}\n",
        )
        .unwrap();

    let report = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
    assert_eq!(report.appended_events, 1);

    index_once(&home).unwrap();
    let results = search_history(&home, "filename derived codex marker", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, session_id);
}

#[test]
fn claude_native_backfill_ignores_project_sidecars() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("claude-projects");
    init_home(&home).unwrap();
    fs::create_dir_all(source.join("session/subagents")).unwrap();

    let session_id = "15a53bfd-c382-488d-b890-687021285e49";
    fs::write(
            source.join(format!("{session_id}.jsonl")),
            format!(
                "{{\"session_id\":\"{session_id}\",\"cwd\":\"/tmp/native-claude\",\"project_root\":\"/tmp/native-claude\",\"type\":\"claude.transcript.user\",\"canonical_type\":\"user.message\",\"event_id\":\"claude-native-1\",\"message\":\"native claude marker\"}}\n"
            ),
        )
        .unwrap();
    fs::write(source.join("sessions-index.json"), "{\"sessions\":[]}").unwrap();
    fs::write(
        source.join("session/subagents/agent-a676598cc8f883f73.meta.json"),
        "{\"agent_id\":\"agent-a676598cc8f883f73\"}",
    )
    .unwrap();
    fs::write(
        source.join("session/subagents/skill-injections.jsonl"),
        "{\"kind\":\"plugin-config\",\"type\":\"skill-injection\"}\n",
    )
    .unwrap();

    let report = backfill_since(&home, Some(Tool::Claude), &source, None).unwrap();
    assert_eq!(report.source_files, 1);
    assert_eq!(report.appended_events, 1);
    assert_eq!(report.checkpoint_files, 1);

    index_once(&home).unwrap();
    let results = search_history(&home, "native claude marker", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, session_id);
}

#[test]
fn sanitized_real_native_fixtures_import_defensively_for_all_tools() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    init_home(&home).unwrap();

    let claude_report = backfill_since(
        &home,
        Some(Tool::Claude),
        &repo.join("fixtures/native/claude/projects"),
        None,
    )
    .unwrap();
    let codex_report = backfill_since(
        &home,
        Some(Tool::Codex),
        &repo.join("fixtures/native/codex/sessions"),
        None,
    )
    .unwrap();
    let opencode_report = backfill_since(
        &home,
        Some(Tool::Opencode),
        &repo.join("fixtures/native/opencode"),
        None,
    )
    .unwrap();

    assert_eq!(claude_report.source_files, 1);
    assert_eq!(claude_report.appended_events, 5);
    assert_eq!(codex_report.source_files, 1);
    assert_eq!(codex_report.appended_events, 4);
    assert_eq!(opencode_report.source_files, 5);
    assert_eq!(opencode_report.appended_events, 8);
    assert_eq!(checkpoint_row_count(&home), 7);

    index_once(&home).unwrap();
    assert_eq!(
        search_history(&home, "sanitized native claude user marker", 10).unwrap()[0].session_id,
        "11111111-1111-4111-8111-111111111111"
    );
    assert_eq!(
        search_history(&home, "sanitized native codex assistant marker", 10).unwrap()[0].session_id,
        "22222222-2222-4222-8222-222222222222"
    );
    assert_eq!(
        search_history(&home, "sanitized native opencode assistant marker", 10).unwrap()[0]
            .session_id,
        "33333333-3333-4333-8333-333333333333"
    );

    let claude_raw =
        canonical_raw_path(&home, Tool::Claude, "11111111-1111-4111-8111-111111111111");
    let envelopes = raw_envelopes(&claude_raw);
    let parse_error = envelopes
        .iter()
        .find(|event| event.canonical_type == CanonicalType::Error)
        .expect("malformed native line should import as error");
    assert_eq!(
        parse_error.payload.get("type").and_then(Value::as_str),
        Some("parse_error")
    );
    assert!(parse_error.payload.get("raw_line").is_some());
}

#[test]
fn opencode_native_fixture_maps_m8_types_worktree_and_metadata_session() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = repo.join("fixtures/native/opencode");
    let session_id = "33333333-3333-4333-8333-333333333333";
    init_home(&home).unwrap();

    let first = backfill_since(&home, Some(Tool::Opencode), &fixture, None).unwrap();
    let second = backfill_since(&home, Some(Tool::Opencode), &fixture, None).unwrap();

    assert_eq!(first.source_files, 5);
    assert_eq!(first.appended_events, 8);
    assert_eq!(second.appended_events, 0);

    let raw_path = canonical_raw_path(&home, Tool::Opencode, session_id);
    let envelopes = raw_envelopes(&raw_path);
    assert_eq!(envelopes.len(), 8);
    let error_count = envelopes
        .iter()
        .filter(|event| event.canonical_type == CanonicalType::Error)
        .count();
    assert_eq!(error_count, 0);

    assert!(envelopes.iter().any(|event| {
        event.source_event_type == "reasoning"
            && event.canonical_type == CanonicalType::AssistantDelta
    }));
    assert!(envelopes.iter().any(|event| {
        event.source_event_type == "step-start"
            && event.canonical_type == CanonicalType::AssistantDelta
    }));
    assert!(envelopes.iter().any(|event| {
        event.source_event_type == "step-finish"
            && event.canonical_type == CanonicalType::AssistantDelta
    }));
    assert!(envelopes.iter().any(|event| {
        event.source_event_type == "patch" && event.canonical_type == CanonicalType::FileChanged
    }));
    assert!(envelopes.iter().any(|event| {
        event.source_event_type == "session.created"
            && event.canonical_type == CanonicalType::SessionStarted
    }));
    assert!(envelopes
        .iter()
        .all(|event| event.project_root.is_some() && event.cwd.is_some()));
    assert!(envelopes.iter().any(|event| {
        event.source_event_type == "reasoning"
            && event.project_root.as_deref() == Some("/Users/example/opencode-project")
            && event.cwd.as_deref() == Some("/Users/example/opencode-project")
    }));
    assert!(!canonical_raw_path(&home, Tool::Opencode, "project_meta").exists());
}

#[test]
fn backfill_uses_sqlite_checkpoints_and_incremental_rerun_is_noop() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("codex-sessions");
    let session_id = "44444444-4444-4444-8444-444444444444";
    init_home(&home).unwrap();
    fs::create_dir_all(&source).unwrap();
    fs::write(
            source.join(format!("rollout-2026-06-18T10-00-00-{session_id}.jsonl")),
            format!(
                "{{\"timestamp\":\"2026-06-18T10:00:00.000Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session_id}\",\"cwd\":\"/tmp/native-codex\"}}}}\n\
                 {{\"timestamp\":\"2026-06-18T10:00:01.000Z\",\"type\":\"response_item\",\"payload\":{{\"id\":\"checkpoint-user-1\",\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"checkpoint native marker\"}}]}}}}\n"
            ),
        )
        .unwrap();

    let first = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
    let second = backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();

    assert_eq!(first.appended_events, 2);
    assert_eq!(first.checkpoint_files, 1);
    assert_eq!(second.appended_events, 0);
    assert_eq!(checkpoint_row_count(&home), 1);
    assert_eq!(checkpoint_sidecar_count(&home), 0);
    assert_eq!(
        raw_line_count(&canonical_raw_path(&home, Tool::Codex, session_id)),
        2
    );
}

#[test]
fn raw_index_checkpoints_skip_unchanged_canonical_files_and_refresh_on_append() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let session_id = "raw-index-checkpoint-session";
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "message_id": "raw-index-checkpoint-1",
            "prompt": "raw checkpoint first marker"
        }),
    )
    .unwrap();

    let raw_path = canonical_raw_path(&home, Tool::Claude, session_id);
    let first = index_once(&home).unwrap();
    let second = index_once(&home).unwrap();
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();
    let source_meta = source_file_metadata(&raw_path).unwrap();

    assert_eq!(first.indexed_events, 1);
    assert_eq!(second.indexed_events, 0);
    assert!(raw_index_checkpoint_is_current(
        &conn,
        &db_path,
        Tool::Claude,
        &raw_path,
        &source_meta
    )
    .unwrap());
    let raw_checkpoint_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM checkpoints WHERE source_kind = 'raw_jsonl'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(raw_checkpoint_count, 1);

    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "message_id": "raw-index-checkpoint-2",
            "prompt": "raw checkpoint second marker"
        }),
    )
    .unwrap();
    let changed_meta = source_file_metadata(&raw_path).unwrap();
    assert!(!raw_index_checkpoint_is_current(
        &conn,
        &db_path,
        Tool::Claude,
        &raw_path,
        &changed_meta
    )
    .unwrap());

    let third = index_once(&home).unwrap();
    assert_eq!(third.indexed_events, 1);
}

#[test]
fn fts_schema_migration_rebuilds_without_reindexing_raw_files() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "fts-migration-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "fts-migration-1",
            "prompt": "streaming fts rebuild marker"
        }),
    )
    .unwrap();

    assert_eq!(index_once(&home).unwrap().indexed_events, 1);
    let db_path = home.join("index").join("harness.db");
    Connection::open(&db_path)
        .unwrap()
        .execute_batch(
            "DROP TABLE IF EXISTS events_fts;
                 CREATE VIRTUAL TABLE events_fts USING fts5(searchable_text);",
        )
        .unwrap();

    assert_eq!(index_once(&home).unwrap().indexed_events, 0);
    let results = search_history(&home, "streaming fts rebuild marker", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, "fts-migration-session");
}

#[test]
fn discontinuities_emit_once_for_truncation_rotation_and_deletion() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let root = temp.path().join("claude-projects");
    init_home(&home).unwrap();
    fs::create_dir_all(&root).unwrap();

    let truncated_session = "55555555-5555-4555-8555-555555555555";
    let truncated = root.join(format!("{truncated_session}.jsonl"));
    fs::write(
        &truncated,
        claude_user_line(truncated_session, "truncate-1", "truncation original one")
            + &claude_user_line(truncated_session, "truncate-2", "truncation original two"),
    )
    .unwrap();
    backfill_since(&home, Some(Tool::Claude), &root, None).unwrap();
    fs::write(
        &truncated,
        claude_user_line(truncated_session, "truncate-1", "truncation original one"),
    )
    .unwrap();
    assert_eq!(
        backfill_since(&home, Some(Tool::Claude), &root, None)
            .unwrap()
            .discontinuities,
        1
    );
    assert_eq!(
        backfill_since(&home, Some(Tool::Claude), &root, None)
            .unwrap()
            .discontinuities,
        0
    );
    assert_eq!(
        discontinuity_count(&home, Tool::Claude, truncated_session, "source.truncated"),
        1
    );

    let rotated_session = "66666666-6666-4666-8666-666666666666";
    let rotated = root.join(format!("{rotated_session}.jsonl"));
    fs::write(
        &rotated,
        claude_user_line(rotated_session, "rotate-1", "rotation original marker"),
    )
    .unwrap();
    backfill_since(&home, Some(Tool::Claude), &root, None).unwrap();
    fs::remove_file(&rotated).unwrap();
    fs::write(
        &rotated,
        claude_user_line(
            rotated_session,
            "rotate-2",
            "rotation replacement marker with enough bytes to avoid truncation precedence",
        ),
    )
    .unwrap();
    assert_eq!(
        backfill_since(&home, Some(Tool::Claude), &root, None)
            .unwrap()
            .discontinuities,
        1
    );
    assert_eq!(
        backfill_since(&home, Some(Tool::Claude), &root, None)
            .unwrap()
            .discontinuities,
        0
    );
    assert_eq!(
        discontinuity_count(&home, Tool::Claude, rotated_session, "source.rotated"),
        1
    );

    let deleted_session = "77777777-7777-4777-8777-777777777777";
    let deleted = root.join(format!("{deleted_session}.jsonl"));
    fs::write(
        &deleted,
        claude_user_line(deleted_session, "delete-1", "deletion original marker"),
    )
    .unwrap();
    backfill_since(&home, Some(Tool::Claude), &root, None).unwrap();
    fs::remove_file(&deleted).unwrap();
    assert_eq!(
        backfill_since(&home, Some(Tool::Claude), &root, None)
            .unwrap()
            .discontinuities,
        1
    );
    assert_eq!(
        backfill_since(&home, Some(Tool::Claude), &root, None)
            .unwrap()
            .discontinuities,
        0
    );
    assert_eq!(
        discontinuity_count(&home, Tool::Claude, deleted_session, "source.deleted"),
        1
    );
}

#[test]
fn dry_run_reports_missing_events_and_writes_nothing() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("codex-sessions");
    let session_id = "88888888-8888-4888-8888-888888888888";
    init_home(&home).unwrap();
    fs::create_dir_all(&source).unwrap();
    fs::write(
            source.join(format!("rollout-2026-06-18T11-00-00-{session_id}.jsonl")),
            "{\"timestamp\":\"2026-06-18T11:00:00.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"dry-run-shared\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"dry run shared marker\"}]}}\n\
                 {\"timestamp\":\"2026-06-18T11:00:01.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"dry-run-gap\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"dry run gap marker\"}]}}\n",
        )
        .unwrap();
    ingest_hook_event(
        &home,
        Tool::Codex,
        json!({
            "session_id": session_id,
            "type": "response_item",
            "payload": {
                "id": "dry-run-shared",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "dry run shared marker"}]
            }
        }),
    )
    .unwrap();

    let before_raw_lines = raw_line_count(&canonical_raw_path(&home, Tool::Codex, session_id));
    let report = backfill_dry_run(&home, Some(Tool::Codex), &source, None).unwrap();
    let after_raw_lines = raw_line_count(&canonical_raw_path(&home, Tool::Codex, session_id));

    assert_eq!(report.source_files, 1);
    assert_eq!(report.on_disk_events, 2);
    assert_eq!(report.captured_events, 1);
    assert_eq!(report.missing_events, 1);
    assert_eq!(report.partial_sessions, 1);
    assert_eq!(report.sessions[0].would_import.len(), 1);
    assert_eq!(
        report.sessions[0].would_import[0].canonical_type,
        "assistant.message"
    );
    assert_eq!(before_raw_lines, after_raw_lines);
    assert_eq!(checkpoint_row_count(&home), 0);
}

#[test]
fn partial_live_capture_then_backfill_reconciles_for_each_tool() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    assert_reconciles_claude(&home, temp.path());
    assert_reconciles_codex(&home, temp.path());
    assert_reconciles_opencode(&home, temp.path());
}

#[test]
fn doctor_reports_compact_coverage_summary() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("claude-projects");
    let session_id = "99999999-9999-4999-8999-999999999999";
    init_home(&home).unwrap();
    fs::create_dir_all(&source).unwrap();
    fs::write(
        source.join(format!("{session_id}.jsonl")),
        claude_user_line(session_id, "doctor-coverage-1", "doctor coverage marker"),
    )
    .unwrap();
    backfill_since(&home, Some(Tool::Claude), &source, None).unwrap();
    index_once(&home).unwrap();

    let report = doctor_with_options(&home, false);

    assert_eq!(report.coverage.checkpointed_sources, 2);
    assert_eq!(report.coverage.captured_sessions, 1);
    assert_eq!(report.coverage.captured_events, 1);
    assert!(report.storage_footprint.raw_bytes > 0);
    assert!(report.storage_footprint.index_bytes > 0);
    assert_eq!(report.storage_footprint.vectors_bytes, 0);
    assert_eq!(report.storage_footprint.models_bytes, 0);
    assert_eq!(
        report.storage_footprint.canonical_total,
        report
            .storage_footprint
            .raw_bytes
            .saturating_add(report.storage_footprint.blobs_bytes)
    );
    assert_eq!(
        report.storage_footprint.derived_total,
        report
            .storage_footprint
            .index_bytes
            .saturating_add(report.storage_footprint.spool_bytes)
            .saturating_add(report.storage_footprint.models_bytes)
    );
    assert!(report.storage_footprint.total_bytes >= report.storage_footprint.raw_bytes);
}

#[cfg(not(feature = "semantic"))]
#[test]
fn default_build_reports_no_semantic_model_without_touching_network() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let status = embedding_model_status(&home);

    assert!(!status.feature_enabled);
    assert_eq!(status.model_id, "embeddinggemma-300m-q4");
    assert_eq!(status.expected_dimensions, 256);
    assert!(!status.model_present);
    assert!(!status.semantic_available);
    assert!(status.message.contains("semantic feature is disabled"));
}

#[test]
fn embedding_model_disclosure_reports_terms_and_measured_local_footprint() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let cache_path = semantic_model_cache_path(&home);
    fs::create_dir_all(&cache_path).unwrap();
    fs::write(cache_path.join("partial-file.bin"), b"partial model bytes").unwrap();

    let disclosure = embedding_model_disclosure(&home, SEMANTIC_MODEL_ID).unwrap();

    assert_eq!(disclosure.model_id, SEMANTIC_MODEL_ID);
    assert_eq!(disclosure.repository, SEMANTIC_MODEL_REPO);
    assert_eq!(disclosure.total_files, SEMANTIC_MODEL_REMOTE_FILES.len());
    assert!(disclosure.current_on_disk_bytes >= "partial model bytes".len() as u64);
    assert!(!disclosure.model_present);
    assert!(disclosure.license_summary.contains("Gemma Terms of Use"));
    assert!(disclosure.cache_path.ends_with(SEMANTIC_MODEL_ID));
}

#[test]
fn semantic_retrieval_fixture_is_labeled_without_requiring_model() {
    let fixture = semantic_retrieval_fixture();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.tool, Tool::Claude);
    assert!(!fixture.session_id.trim().is_empty());
    assert!(!fixture.cwd.trim().is_empty());
    assert!(!fixture.project_root.trim().is_empty());
    assert!(!fixture.events.is_empty());
    assert!(!fixture.queries.is_empty());

    let event_ids = fixture
        .events
        .iter()
        .map(|event| {
            assert!(!event.event_id.trim().is_empty());
            assert!(matches!(event.role.as_str(), "user" | "assistant"));
            assert!(!event.text.trim().is_empty());
            event.event_id.clone()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(event_ids.len(), fixture.events.len());

    for query in &fixture.queries {
        assert!(!query.query.trim().is_empty());
        assert!(!query.relevant_event_ids.is_empty());
        for event_id in &query.relevant_event_ids {
            assert!(
                event_ids.contains(event_id),
                "query {:?} references unknown event id {event_id}",
                query.query
            );
        }
    }
}

#[test]
fn corroboration_extracts_and_resolves_refs_read_only_against_local_git() {
    let extraction_refs = extract_corroboration_candidates(
            "commit abcdef1 landed on branch feature/corroborate, touched src/lib.rs, and referenced PR #42.",
        )
        .into_iter()
        .map(|candidate| (candidate.kind.as_str().to_string(), candidate.reference))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        extraction_refs,
        BTreeSet::from([
            ("branch".to_string(), "feature/corroborate".to_string()),
            ("commit".to_string(), "abcdef1".to_string()),
            ("file".to_string(), "src/lib.rs".to_string()),
            ("pr".to_string(), "#42".to_string()),
        ])
    );

    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = temp.path().join("repo");
    fs::create_dir_all(repo.join("src")).unwrap();
    run_git_setup(temp.path(), &["init", repo.to_str().unwrap()]);
    run_git(&repo, &["config", "user.email", "nabu@example.invalid"]);
    run_git(&repo, &["config", "user.name", "Nabu Test"]);
    fs::write(repo.join("src/lib.rs"), "pub fn corroborated() {}\n").unwrap();
    run_git(&repo, &["add", "src/lib.rs"]);
    run_git(&repo, &["commit", "-m", "initial corroboration fixture"]);
    run_git(&repo, &["branch", "feature/corroborate"]);
    fs::create_dir_all(repo.join("notes")).unwrap();
    fs::write(
        repo.join("notes/trace.txt"),
        "untracked corroboration note\n",
    )
    .unwrap();
    let commit = run_git(&repo, &["rev-parse", "HEAD"]);
    let commit_prefix = &commit[..12];
    let missing_commit = "ffffffffffffffffffffffffffffffffffffffff";
    let before_snapshot = git_snapshot(&repo);

    init_home(&home).unwrap();
    let text = format!(
            "corroboration marker commit {commit_prefix} and missing commit {missing_commit}; branch feature/corroborate and branch missing/branch; files src/lib.rs notes/trace.txt src/missing.txt; PR #123."
        );
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "corroboration-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "corroboration-message",
            "cwd": repo,
            "project_root": repo,
            "prompt": text,
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    clear_git_invocations();
    let default_page = search_history_page(
        &home,
        "corroboration marker",
        SearchOptions {
            limit: 1,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(default_page.returned, 1);
    assert!(default_page.results[0].corroboration.is_none());
    assert!(captured_git_invocations().is_empty());

    let page = search_history_page(
        &home,
        "corroboration marker",
        SearchOptions {
            limit: 1,
            corroborate: true,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    let corroboration = page.results[0].corroboration.as_ref().unwrap();
    let canonical_repo = fs::canonicalize(&repo).unwrap();
    assert_eq!(
        corroboration.repo.as_deref(),
        Some(canonical_repo.to_str().unwrap())
    );
    assert_ref_status(corroboration, "commit", commit_prefix, "present", None);
    assert_ref_status(corroboration, "commit", missing_commit, "missing", None);
    assert_ref_status(
        corroboration,
        "branch",
        "feature/corroborate",
        "present",
        None,
    );
    assert_ref_status(corroboration, "branch", "missing/branch", "missing", None);
    assert_ref_status(corroboration, "file", "src/lib.rs", "present", None);
    assert_ref_status(corroboration, "file", "notes/trace.txt", "untracked", None);
    assert_ref_status(corroboration, "file", "src/missing.txt", "missing", None);
    assert_ref_status(
        corroboration,
        "pr",
        "#123",
        "unresolved",
        Some("needs_network"),
    );
    assert_eq!(git_snapshot(&repo), before_snapshot);
    assert_no_network_git_commands(&captured_git_invocations());

    let no_repo = temp.path().join("no-repo");
    fs::create_dir_all(&no_repo).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "no-repo-corroboration-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "no-repo-corroboration-message",
            "cwd": no_repo,
            "project_root": no_repo,
            "prompt": "no repo marker commit deadbee file src/lib.rs PR #7",
        }),
    )
    .unwrap();
    index_once(&home).unwrap();
    let no_repo_page = search_history_page(
        &home,
        "no repo marker",
        SearchOptions {
            limit: 1,
            corroborate: true,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    let no_repo_corroboration = no_repo_page.results[0].corroboration.as_ref().unwrap();
    assert_eq!(no_repo_corroboration.repo, None);
    assert_ref_status(
        no_repo_corroboration,
        "commit",
        "deadbee",
        "unresolved",
        Some("no_repo"),
    );
    assert_ref_status(
        no_repo_corroboration,
        "file",
        "src/lib.rs",
        "unresolved",
        Some("no_repo"),
    );
    assert_ref_status(
        no_repo_corroboration,
        "pr",
        "#7",
        "unresolved",
        Some("needs_network"),
    );
}

fn run_git_setup(cwd: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .env("PAGER", "cat")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git setup failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn run_git(repo: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .env("PAGER", "cat")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git command failed: git -C {} {}\n{}\n{}",
        repo.display(),
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

#[derive(Debug, PartialEq, Eq)]
struct GitSnapshot {
    head: String,
    refs: String,
    index: String,
    status: String,
}

fn git_snapshot(repo: &Path) -> GitSnapshot {
    GitSnapshot {
        head: run_git(repo, &["rev-parse", "HEAD"]),
        refs: run_git(repo, &["for-each-ref", "--format=%(refname):%(objectname)"]),
        index: run_git(repo, &["ls-files", "-s"]),
        status: run_git(repo, &["status", "--porcelain=v1", "-z"]),
    }
}

fn assert_ref_status(
    corroboration: &Corroboration,
    kind: &str,
    reference: &str,
    status: &str,
    reason: Option<&str>,
) {
    let found = corroboration
        .refs
        .iter()
        .find(|candidate| candidate.kind == kind && candidate.reference == reference)
        .unwrap_or_else(|| panic!("missing corroborated ref {kind} {reference}"));
    assert_eq!(found.status, status);
    assert_eq!(found.reason.as_deref(), reason);
}

fn clear_git_invocations() {
    git_invocations().lock().unwrap().clear();
}

fn captured_git_invocations() -> Vec<Vec<String>> {
    git_invocations().lock().unwrap().clone()
}

fn assert_no_network_git_commands(commands: &[Vec<String>]) {
    assert!(
        !commands.is_empty(),
        "corroboration should have used local git read commands"
    );
    for command in commands {
        let Some(operation) = command.first().map(String::as_str) else {
            continue;
        };
        assert!(
            matches!(operation, "rev-parse" | "cat-file" | "log" | "ls-files"),
            "unexpected git operation in corroboration path: {command:?}"
        );
        assert!(
            !matches!(operation, "fetch" | "pull" | "ls-remote"),
            "network-capable git command must not run: {command:?}"
        );
    }
}

#[test]
fn date_or_duration_filters_and_purge_before_use_normalized_thresholds() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    for (session_id, message_id, captured_at, prompt) in [
        (
            "old-session",
            "old-message",
            "2020-01-01T00:00:00Z",
            "datefilter old marker",
        ),
        (
            "new-session",
            "new-message",
            "2099-01-01T00:00:00Z",
            "datefilter new marker",
        ),
    ] {
        let event = envelope_from_backfill_payload(
            Tool::Claude,
            Path::new("/tmp/datefilter.jsonl"),
            0,
            json!({
                "session_id": session_id,
                "hook_event_name": "UserPromptSubmit",
                "message_id": message_id,
                "captured_at": captured_at,
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": prompt
            }),
            &BackfillParseContext::default(),
        )
        .unwrap();
        append_prepared_event(&home, event).unwrap();
    }

    index_once(&home).unwrap();

    let recent = search_history_filtered(
        &home,
        "datefilter",
        SearchOptions {
            since: Some("1d".to_string()),
            limit: 10,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].session_id, "new-session");

    let sessions = list_sessions(&home, Some(Tool::Claude), None, Some("1d"), 10).unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].session_id, "new-session");

    let report = purge_before(&home, "2021-01-01").unwrap();
    assert_eq!(report.indexed_events_removed, 1);
    // Purge dropped the old event; only the new one survives. Under OR
    // semantics the shared "datefilter"/"marker" terms match the survivor, so
    // assert on the surviving session rather than an AND-coupled zero count.
    let surviving = search_history(&home, "datefilter marker", 10).unwrap();
    assert_eq!(surviving.len(), 1);
    assert_eq!(surviving[0].session_id, "new-session");
}

// P0 bug #1: a more-specific multi-word query must not collapse recall to
// zero. Controlled by varying only term count with no session filter.
#[test]
fn search_multi_word_query_keeps_recall_with_or_semantics() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    // Target matches four of the long query's six terms; the distractor
    // matches only two. Neither matches all six.
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "recall-target",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "recall-target-1",
            "prompt": "nabu reduce memory usage while profiling the index"
        }),
    )
    .unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "recall-distractor",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "recall-distractor-1",
            "prompt": "nabu memory note"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    // Two terms: both events surface.
    let short = search_history(&home, "nabu memory", 10).unwrap();
    assert_eq!(short.len(), 2);

    // Six terms, more specific. AND semantics required every term in one event
    // and returned zero; OR keeps recall and bm25 ranks the event satisfying
    // more terms first.
    let long = search_history(
        &home,
        "nabu reduce memory usage performance optimization",
        10,
    )
    .unwrap();
    assert!(
        !long.is_empty(),
        "more-specific query must not collapse recall to zero"
    );
    assert_eq!(long[0].session_id, "recall-target");
}

#[test]
fn search_filters_apply_session_type_file_and_command() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "filter-session",
            "hook_event_name": "PreToolUse",
            "message_id": "command-filter-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "tool_name": "bash",
            "command": "cargo test --workspace",
            "input": "command filter marker"
        }),
    )
    .unwrap();
    ingest_hook_event(
        &home,
        Tool::Opencode,
        json!({
            "session_id": "file-session",
            "event": "file.edited",
            "id": "file-filter-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "path": "/tmp/nabu-fixture/src/auth.rs",
            "diff": "file filter marker"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let command_results = search_history_filtered(
        &home,
        "command filter marker",
        SearchOptions {
            session_id: Some("filter-session".to_string()),
            canonical_type: Some("tool.call".to_string()),
            command: Some("cargo test".to_string()),
            limit: 10,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(command_results.len(), 1);
    assert_eq!(command_results[0].canonical_type, "tool.call");

    let wrong_command = search_history_filtered(
        &home,
        "command filter marker",
        SearchOptions {
            command: Some("npm install".to_string()),
            limit: 10,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert!(wrong_command.is_empty());

    let file_results = search_history_filtered(
        &home,
        "file filter marker",
        SearchOptions {
            file: Some("src/auth.rs".to_string()),
            canonical_type: Some("file.changed".to_string()),
            limit: 10,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(file_results.len(), 1);
    assert_eq!(file_results[0].session_id, "file-session");
}

#[test]
fn search_defaults_are_citation_first_and_full_payload_is_opt_in() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "citation-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": "citation-1",
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": format!("{} needle-centered citation marker {}", "prefix ".repeat(80), "suffix ".repeat(80))
            }),
        )
        .unwrap();
    index_once(&home).unwrap();
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();
    let payload_json: Option<String> = conn
        .query_row("SELECT payload_json FROM events LIMIT 1", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(payload_json.is_none());

    let default_page = search_history_page(
        &home,
        "needle-centered citation marker",
        SearchOptions {
            max_snippet_chars: 48,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(default_page.returned, 1);
    assert_eq!(default_page.max_snippet_chars_applied, 48);
    assert!(default_page.results[0].payload.is_null());
    assert!(default_page.results[0].score > 0.0);
    assert!(default_page.results[0].snippet.contains("needle-centered"));
    assert!(default_page.results[0].snippet.chars().count() <= 48);

    let full_page = search_history_page(
        &home,
        "needle-centered citation marker",
        SearchOptions {
            include_payload: true,
            max_snippet_chars: 5_000,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(full_page.max_snippet_chars_applied, 1_000);
    assert!(full_page.results[0].payload.get("prompt").is_some());
}

#[test]
fn payload_hydration_uses_raw_offset_and_falls_back_to_line_scan() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    for line in 1..=4 {
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "offset-payload-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": format!("offset-payload-{line}"),
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": format!("offset payload marker {line}")
            }),
        )
        .unwrap();
    }
    index_once(&home).unwrap();

    let raw_path = canonical_raw_path(&home, Tool::Claude, "offset-payload-session");
    let raw_file = raw_path.display().to_string();
    let offset = raw_offset_for_line(&raw_path, 3) as i64;
    let scanned = raw_envelope_for_line_scan(&raw_path, 4).unwrap();
    let sought = raw_envelope_for_pointer(&raw_file, 4, Some(offset)).unwrap();
    let fallback = raw_envelope_for_pointer(&raw_file, 4, Some(offset + 1)).unwrap();

    assert_eq!(sought, scanned);
    assert_eq!(fallback, scanned);
    assert_eq!(
        payload_for_raw_pointer(&raw_file, 4, Some(offset))
            .unwrap()
            .get("prompt"),
        Some(&json!("offset payload marker 4"))
    );
}

#[test]
fn search_payload_hydration_uses_grouped_raw_offsets() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    for line in 1..=3 {
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "grouped-payload-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": format!("grouped-payload-{line}"),
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": format!("grouped payload shared marker {line}")
            }),
        )
        .unwrap();
    }
    index_once(&home).unwrap();

    let page = search_history_page(
        &home,
        "grouped payload shared marker",
        SearchOptions {
            include_payload: true,
            limit: 3,
            dedupe: false,
            ..SearchOptions::default()
        },
    )
    .unwrap();

    assert_eq!(page.returned, 3);
    let prompts = page
        .results
        .iter()
        .map(|result| {
            result
                .payload
                .get("prompt")
                .and_then(Value::as_str)
                .unwrap()
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(prompts.len(), 3);
    assert!(prompts.contains("grouped payload shared marker 1"));
    assert!(prompts.contains("grouped payload shared marker 2"));
    assert!(prompts.contains("grouped payload shared marker 3"));
}

#[test]
fn search_auto_falls_back_to_lexical_and_forced_hybrid_errors_without_semantic_backend() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "mode-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "mode-1",
            "prompt": "search mode lexical fallback marker"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let auto_page = search_history_page(
        &home,
        "search mode lexical fallback marker",
        SearchOptions {
            mode: SearchMode::Auto,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(auto_page.mode_requested, SearchMode::Auto);
    assert_eq!(auto_page.mode_applied, SearchMode::Lexical);
    assert!(!auto_page.semantic_available);
    assert_eq!(auto_page.returned, 1);

    let lexical_page = search_history_page(
        &home,
        "search mode lexical fallback marker",
        SearchOptions {
            mode: SearchMode::Lexical,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(lexical_page.mode_applied, SearchMode::Lexical);
    assert_eq!(lexical_page.returned, 1);

    let error = search_history_page(
        &home,
        "search mode lexical fallback marker",
        SearchOptions {
            mode: SearchMode::Hybrid,
            ..SearchOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, Error::SemanticUnavailable(_)));
}

#[test]
fn embedding_units_are_structured_and_exclude_tool_output_noise() {
    let payload = json!({
        "tool_name": "shell",
        "command": "cargo test --workspace",
        "status": "failed",
        "stdout": "very long compiler output that should remain lexical-only",
        "stderr": "more noisy output that should not become a vector unit"
    });
    let document = search_document_for_event(CanonicalType::ToolResult, &payload);

    let units = embedding_units_for_document(&document);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].kind, EmbeddingUnitKind::ToolIntent);
    assert!(units[0].text.contains("cargo test --workspace"));
    assert!(!units[0].text.contains("compiler output"));
    assert_eq!(units[0].text_hash, sha256_hex(units[0].text.as_bytes()));
}

#[test]
fn search_and_session_exclude_deltas_by_default_and_restore_on_opt_in() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "delta-session",
            "hook_event_name": "MessageDisplay",
            "message_id": "delta-message",
            "index": 0,
            "final": false,
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "delta": "delta-only fixture marker"
        }),
    )
    .unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "delta-session",
            "hook_event_name": "MessageDisplay",
            "message_id": "final-message",
            "index": 1,
            "final": true,
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "message": "final fixture marker"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    // Default search excludes deltas. Under OR semantics the shared
    // "fixture"/"marker" terms also surface the non-delta final message, so
    // assert the delta is absent rather than an AND-coupled empty result.
    let default_search = search_history(&home, "delta-only fixture marker", 10).unwrap();
    assert!(default_search
        .iter()
        .all(|result| result.canonical_type != "assistant.delta"));
    let delta_search = search_history_page(
        &home,
        "delta-only fixture marker",
        SearchOptions {
            include_deltas: true,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(delta_search.results[0].canonical_type, "assistant.delta");

    let default_session = get_session_page(
        &home,
        Tool::Claude,
        "delta-session",
        SessionOptions::default(),
    )
    .unwrap();
    assert!(default_session
        .events
        .iter()
        .all(|event| event.canonical_type != "assistant.delta"));

    let full_session = get_session_page(
        &home,
        Tool::Claude,
        "delta-session",
        SessionOptions {
            include_deltas: true,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    assert_eq!(full_session.events[0].canonical_type, "assistant.delta");
    assert!(
        export_session_markdown_with_options(&home, Tool::Claude, "delta-session", false)
            .unwrap()
            .contains("delta-only fixture marker")
    );
}

#[test]
fn session_context_window_clamps_and_wins_over_after_raw_line() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    for line in 1..=5 {
        ingest_hook_event(
            &home,
            Tool::Claude,
            json!({
                "session_id": "window-session",
                "hook_event_name": "UserPromptSubmit",
                "message_id": format!("window-{line}"),
                "cwd": "/tmp/nabu-fixture",
                "project_root": "/tmp/nabu-fixture",
                "prompt": format!("window marker line {line}")
            }),
        )
        .unwrap();
    }
    index_once(&home).unwrap();

    let window = get_session_page(
        &home,
        Tool::Claude,
        "window-session",
        SessionOptions {
            around_raw_line: Some(3),
            after_raw_line: Some(4),
            before: 1,
            after: 1,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    assert_eq!(window.mode, "window");
    assert_eq!(
        window
            .events
            .iter()
            .map(|event| event.raw_line)
            .collect::<Vec<_>>(),
        vec![2, 3, 4]
    );

    let clamped = get_session_page(
        &home,
        Tool::Claude,
        "window-session",
        SessionOptions {
            around_raw_line: Some(1),
            before: 10,
            after: 0,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    assert_eq!(clamped.events[0].raw_line, 1);
}

#[test]
fn search_dedupes_twins_only_at_retrieval_layer() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let source = temp.path().join("codex-sessions");
    init_home(&home).unwrap();
    fs::create_dir_all(&source).unwrap();

    let session_id = "019b0000-0000-7000-8000-000000000001";
    fs::write(
            source.join(format!("rollout-2026-06-18T00-00-00-{session_id}.jsonl")),
            format!(
                "{{\"timestamp\":\"2026-06-18T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session_id}\",\"cwd\":\"/tmp/native-codex\"}}}}\n\
                 {{\"timestamp\":\"2026-06-18T00:00:01Z\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"twinned codex answer marker\"}}]}}}}\n\
                 {{\"timestamp\":\"2026-06-18T00:00:01Z\",\"type\":\"event_msg\",\"payload\":{{\"type\":\"agent_message\",\"message\":\"twinned codex answer marker\"}}}}\n"
            ),
        )
        .unwrap();
    backfill_since(&home, Some(Tool::Codex), &source, None).unwrap();
    index_once(&home).unwrap();

    let deduped = search_history_page(
        &home,
        "twinned codex answer marker",
        SearchOptions::default(),
    )
    .unwrap();
    assert_eq!(deduped.results.len(), 1);
    assert_eq!(deduped.results[0].also_at.len(), 1);

    let not_deduped = search_history_page(
        &home,
        "twinned codex answer marker",
        SearchOptions {
            dedupe: false,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(not_deduped.results.len(), 2);

    let session =
        get_session_page(&home, Tool::Codex, session_id, SessionOptions::default()).unwrap();
    assert_eq!(
        session
            .events
            .iter()
            .filter(|event| event.text.contains("twinned codex answer marker"))
            .count(),
        2
    );
}

#[test]
fn doctor_fast_and_deep_report_their_integrity_scope() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "doctor-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "doctor-1",
            "prompt": "doctor marker"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let fast = doctor_with_options(&home, false);
    assert_eq!(fast.level, "fast");
    assert_eq!(fast.integrity, "structural");
    assert!(fast.index.ok);
    assert!(fast.index.message.contains("core tables"));
    assert!(fast.stats.is_none());
    assert!(fast.latest_captured_events["claude"].is_some());

    let deep = doctor_with_options(&home, true);
    assert_eq!(deep.level, "deep");
    assert_eq!(deep.integrity, "full");
    assert!(deep.index.message.contains("integrity_check"));
    assert_eq!(deep.stats.unwrap().events, 1);

    let db_path = home.join("index").join("harness.db");
    let conn = Connection::open(&db_path).unwrap();
    let plan = conn
            .query_row(
                "EXPLAIN QUERY PLAN
                 SELECT tool, session_id, canonical_type, captured_at, searchable_text, raw_file, raw_line, raw_offset
                 FROM events
                 WHERE tool = 'claude'
                 ORDER BY captured_at DESC, id DESC
                 LIMIT 1",
                [],
                |row| row.get::<_, String>(3),
            )
            .unwrap();
    assert!(plan.contains("idx_events_tool_captured"), "{plan}");
}

#[test]
fn schema_helpers_reject_non_identifier_names() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path).unwrap();

    let count_error = table_count(&conn, &db_path, "events;DROP TABLE events").unwrap_err();
    assert!(matches!(count_error, Error::Validation(_)));

    let column_error =
        crate::db::ensure_table_column(&conn, &db_path, "checkpoints", "bad-column", "TEXT")
            .unwrap_err();
    assert!(matches!(column_error, Error::Validation(_)));
}

#[cfg(unix)]
#[test]
fn directory_size_does_not_follow_symlink_cycles() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path().join("root");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("payload.txt"), b"12345").unwrap();
    symlink(&root, root.join("cycle")).unwrap();

    assert_eq!(directory_size(&root).unwrap(), 5);
}

#[test]
fn set_opencode_server_url_round_trips_and_preserves_other_settings() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let config = home.join("config.toml");

    // A config with an unrelated key and a comment that must survive edits.
    fs::write(
        &config,
        "schema_version = 1\n# keep me\n\n[opencode]\n# server_url = \"http://127.0.0.1:4096\"\n",
    )
    .unwrap();

    // Set activates the commented seed and is readable through the reader.
    // Read via the config parser directly so ambient env vars can't shadow it.
    set_opencode_server_url(&home, Some("http://localhost:9999")).unwrap();
    assert_eq!(
        crate::config::read_opencode_server_url_from_config(&config)
            .unwrap()
            .as_deref(),
        Some("http://localhost:9999")
    );
    let after_set = fs::read_to_string(&config).unwrap();
    assert!(after_set.contains("schema_version = 1"));
    assert!(after_set.contains("# keep me"));
    assert!(after_set.contains("server_url = \"http://localhost:9999\""));

    // Idempotent: setting the same value does not rewrite the file.
    set_opencode_server_url(&home, Some("http://localhost:9999")).unwrap();
    assert_eq!(fs::read_to_string(&config).unwrap(), after_set);

    // Clear removes the active line but keeps the rest.
    set_opencode_server_url(&home, None).unwrap();
    assert_eq!(
        crate::config::read_opencode_server_url_from_config(&config).unwrap(),
        None
    );
    let after_clear = fs::read_to_string(&config).unwrap();
    assert!(after_clear.contains("schema_version = 1"));
    assert!(after_clear.contains("# keep me"));
    assert!(!after_clear.contains("server_url = \"http://localhost:9999\""));
}

#[test]
fn set_opencode_server_url_appends_section_when_absent() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    // No config.toml yet — writer must create it and append the section.
    set_opencode_server_url(&home, Some("http://127.0.0.1:4096")).unwrap();
    assert_eq!(
        crate::config::read_opencode_server_url_from_config(&home.join("config.toml"))
            .unwrap()
            .as_deref(),
        Some("http://127.0.0.1:4096")
    );
}

#[test]
fn set_opencode_server_url_rejects_invalid_values_without_rewriting_config() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();
    let config = home.join("config.toml");
    let before = fs::read_to_string(&config).unwrap();

    for invalid in [
        "https://127.0.0.1:4096",
        "http://",
        "http://127.0.0.1:not-a-port",
        "http://127.0.0.1:4096\nserver_url = \"http://evil\"",
        "http://127.0.0.1:4096\\broken",
        " http://127.0.0.1:4096",
    ] {
        let error = set_opencode_server_url(&home, Some(invalid)).unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{invalid}: {error}");
        assert_eq!(fs::read_to_string(&config).unwrap(), before);
    }
}

#[test]
fn search_treats_hyphenated_queries_as_plain_text() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    ingest_hook_event(
        &home,
        Tool::Claude,
        json!({
            "session_id": "hyphen-search-session",
            "hook_event_name": "UserPromptSubmit",
            "message_id": "hyphen-search-1",
            "cwd": "/tmp/nabu-fixture",
            "project_root": "/tmp/nabu-fixture",
            "prompt": "nabu project setup goals and tasks"
        }),
    )
    .unwrap();
    index_once(&home).unwrap();

    let results = search_history(&home, "nabu project setup goals and tasks", 10).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, "hyphen-search-session");
}

#[test]
fn search_rejects_queries_without_searchable_text() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    init_home(&home).unwrap();

    let error = search_history(&home, "-- : ()", 10).unwrap_err();

    assert!(
        matches!(error, Error::Validation(message) if message == "query must contain searchable text")
    );
}

#[test]
fn codex_exec_json_ingest_preserves_delta_order_and_usage_metadata() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    init_home(&home).unwrap();

    let report = ingest_file(
        &home,
        Tool::Codex,
        Source::ExecJson,
        &repo.join("fixtures/codex/exec-json.jsonl"),
    )
    .unwrap();

    assert_eq!(report.appended_events, 5);
    let raw_path = canonical_raw_path(&home, Tool::Codex, "codex-exec-stream-session");
    let envelopes = raw_envelopes(&raw_path);
    assert!(envelopes
        .iter()
        .all(|event| event.source == Source::ExecJson));
    let deltas = envelopes
        .iter()
        .filter(|event| event.canonical_type == CanonicalType::AssistantDelta)
        .collect::<Vec<_>>();
    assert_eq!(
        deltas
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
    assert!(deltas[0]
        .payload
        .get("delta")
        .and_then(Value::as_str)
        .unwrap()
        .ends_with("delta one"));
    assert!(deltas[1]
        .payload
        .get("delta")
        .and_then(Value::as_str)
        .unwrap()
        .ends_with("delta two"));

    index_once(&home).unwrap();
    let usage_results = search_history_page(
        &home,
        "total_tokens 42",
        SearchOptions {
            tool: Some(Tool::Codex),
            limit: 10,
            ..SearchOptions::default()
        },
    )
    .unwrap();
    assert_eq!(usage_results.results.len(), 1);
    assert_eq!(usage_results.results[0].canonical_type, "session.ended");
    let export =
        export_session_jsonl_with_options(&home, Tool::Codex, "codex-exec-stream-session", false)
            .unwrap();
    assert!(export.contains("\"total_tokens\":42"));
}

#[test]
fn codex_app_server_ingest_preserves_jsonrpc_payloads_and_delta_order() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    init_home(&home).unwrap();

    let report = ingest_file(
        &home,
        Tool::Codex,
        Source::AppServer,
        &repo.join("fixtures/codex/app-server-notifications.jsonl"),
    )
    .unwrap();

    assert_eq!(report.appended_events, 6);
    let raw_path = canonical_raw_path(&home, Tool::Codex, "codex-app-server-session");
    let envelopes = raw_envelopes(&raw_path);
    assert!(envelopes
        .iter()
        .all(|event| event.source == Source::AppServer));
    assert_eq!(envelopes[0].source_event_type, "thread/started");
    assert!(envelopes[0].payload.get("jsonrpc").is_some());
    assert!(envelopes
        .iter()
        .any(|event| event.canonical_type == CanonicalType::ToolCall));
    let deltas = envelopes
        .iter()
        .filter(|event| event.canonical_type == CanonicalType::AssistantDelta)
        .collect::<Vec<_>>();
    assert_eq!(
        deltas
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
    assert!(deltas.iter().all(|event| event
        .source_event_id
        .as_deref()
        .unwrap()
        .contains(":delta")));
}

#[test]
fn codex_streaming_and_hook_identity_dedupe_same_event_but_keep_deltas() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let stream = temp.path().join("codex-stream.jsonl");
    let session_id = "codex-stream-identity-session";
    init_home(&home).unwrap();

    ingest_hook_event(
        &home,
        Tool::Codex,
        json!({
            "session_id": session_id,
            "type": "response_item",
            "payload": {
                "id": "codex-stream-shared-item",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "codex stream shared identity marker"}]
            }
        }),
    )
    .unwrap();
    fs::write(
            &stream,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&json!({
                    "timestamp": "2026-06-18T10:00:00Z",
                    "type": "item.completed",
                    "thread_id": session_id,
                    "item": {
                        "id": "codex-stream-shared-item",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "codex stream shared identity marker"}]
                    }
                }))
                .unwrap(),
                serde_json::to_string(&json!({
                    "timestamp": "2026-06-18T10:00:01Z",
                    "type": "item/agentMessage/delta",
                    "thread_id": session_id,
                    "turn_id": "codex-stream-turn",
                    "message_id": "codex-stream-delta-message",
                    "sequence": 0,
                    "delta": "codex stream granularity marker"
                }))
                .unwrap()
            ),
        )
        .unwrap();

    let report = ingest_file(&home, Tool::Codex, Source::ExecJson, &stream).unwrap();

    assert_eq!(report.appended_events, 1);
    let envelopes = raw_envelopes(&canonical_raw_path(&home, Tool::Codex, session_id));
    assert_eq!(
        envelopes
            .iter()
            .filter(|event| event.source_event_id.as_deref() == Some("codex-stream-shared-item"))
            .count(),
        1
    );
    assert!(envelopes
        .iter()
        .any(|event| event.canonical_type == CanonicalType::AssistantDelta));
}

#[test]
fn opencode_server_messages_reconcile_gaps_without_spool_copy() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let session_id = "opencode-server-reconcile-session";
    init_home(&home).unwrap();
    ingest_hook_event(
        &home,
        Tool::Opencode,
        json!({
            "session_id": session_id,
            "hook_event_name": "message.updated",
            "id": "opencode-server-shared-message",
            "role": "assistant",
            "text": "opencode server shared marker"
        }),
    )
    .unwrap();

    let report = ingest_opencode_server_messages(
        &home,
        session_id,
        json!([
            {
                "id": "opencode-server-shared-message",
                "sessionID": session_id,
                "role": "assistant",
                "text": "opencode server shared marker"
            },
            {
                "id": "opencode-server-gap-message",
                "sessionID": session_id,
                "role": "assistant",
                "worktree": "/Users/example/opencode-server-worktree",
                "parts": [
                    {
                        "id": "opencode-server-gap-part",
                        "type": "text",
                        "text": "opencode server recovered part marker"
                    }
                ]
            }
        ]),
    )
    .unwrap();

    assert_eq!(report.appended_events, 1);
    assert!(!home.join("spool/opencode-api").exists());
    let envelopes = raw_envelopes(&canonical_raw_path(&home, Tool::Opencode, session_id));
    assert_eq!(
        envelopes
            .iter()
            .filter(
                |event| event.source_event_id.as_deref() == Some("opencode-server-shared-message")
            )
            .count(),
        1
    );
    let gap = envelopes
        .iter()
        .find(|event| {
            event.source_event_type == "message.part.updated"
                && event.payload.pointer("/part/text").and_then(Value::as_str)
                    == Some("opencode server recovered part marker")
        })
        .unwrap();
    assert_eq!(
        gap.project_root.as_deref(),
        Some("/Users/example/opencode-server-worktree")
    );
    assert_eq!(
        gap.cwd.as_deref(),
        Some("/Users/example/opencode-server-worktree")
    );
    assert!(envelopes.iter().any(|event| {
        event.source_event_type == "message.part.updated"
            && event.payload.pointer("/part/text").and_then(Value::as_str)
                == Some("opencode server recovered part marker")
    }));
}

fn raw_line_count(path: &Path) -> usize {
    fs::read_to_string(path).unwrap().lines().count()
}

fn dedupe_sidecar_entry_count(sidecar: &DedupeSidecarFiles) -> usize {
    fs::read_dir(&sidecar.buckets_dir)
        .unwrap()
        .map(|entry| raw_line_count(&entry.unwrap().path()))
        .sum()
}

fn raw_envelopes(path: &Path) -> Vec<EventEnvelope> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn raw_offset_for_line(path: &Path, zero_based_line: usize) -> u64 {
    let content = fs::read_to_string(path).unwrap();
    content
        .lines()
        .take(zero_based_line)
        .map(|line| line.len() as u64 + 1)
        .sum()
}

fn checkpoint_row_count(home: &Path) -> i64 {
    let db_path = home.join("index").join("harness.db");
    let conn = Connection::open(db_path).unwrap();
    conn.query_row("SELECT COUNT(*) FROM checkpoints", [], |row| row.get(0))
        .unwrap()
}

fn checkpoint_sidecar_count(home: &Path) -> usize {
    let dir = home.join("checkpoints");
    fs::read_dir(dir)
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .ok()
                .map(|entry| entry.path().is_file())
                .unwrap_or(false)
        })
        .count()
}

fn discontinuity_count(home: &Path, tool: Tool, session_id: &str, reason: &str) -> usize {
    raw_envelopes(&canonical_raw_path(home, tool, session_id))
        .into_iter()
        .filter(|event| {
            event.canonical_type == CanonicalType::SourceDiscontinuity
                && event.payload.get("reason").and_then(Value::as_str) == Some(reason)
        })
        .count()
}

fn claude_user_line(session_id: &str, uuid: &str, text: &str) -> String {
    serde_json::to_string(&json!({
        "type": "user",
        "sessionId": session_id,
        "uuid": uuid,
        "timestamp": "2026-06-18T12:00:00.000Z",
        "cwd": "/tmp/native-claude",
        "message": {
            "role": "user",
            "content": text
        }
    }))
    .unwrap()
        + "\n"
}

fn assert_reconciles_claude(home: &Path, temp_root: &Path) {
    let source = temp_root.join("reconcile-claude");
    fs::create_dir_all(&source).unwrap();
    let session_id = "aaaaaaa1-aaaa-4aaa-8aaa-aaaaaaaaaaa1";
    ingest_hook_event(
        home,
        Tool::Claude,
        json!({
            "session_id": session_id,
            "hook_event_name": "UserPromptSubmit",
            "event_id": "claude-reconcile-shared",
            "prompt": "claude reconcile shared marker"
        }),
    )
    .unwrap();
    ingest_hook_event(
        home,
        Tool::Claude,
        json!({
            "session_id": session_id,
            "hook_event_name": "MessageDisplay",
            "message_id": "claude-reconcile-delta",
            "index": 0,
            "final": false,
            "delta": "claude reconcile granularity marker"
        }),
    )
    .unwrap();
    fs::write(
        source.join(format!("{session_id}.jsonl")),
        format!(
            "{}{}{}",
            claude_user_line(
                session_id,
                "claude-reconcile-shared",
                "claude reconcile shared marker"
            ),
            serde_json::to_string(&json!({
                "type": "assistant",
                "sessionId": session_id,
                "timestamp": "2026-06-18T12:00:01.000Z",
                "message": {
                    "id": "claude-reconcile-gap",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "claude reconcile gap marker"}]
                }
            }))
            .unwrap()
                + "\n",
            serde_json::to_string(&json!({
                "type": "assistant",
                "sessionId": session_id,
                "timestamp": "2026-06-18T12:00:02.000Z",
                "message": {
                    "id": "claude-reconcile-final",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "claude reconcile granularity marker"}]
                }
            }))
            .unwrap()
                + "\n"
        ),
    )
    .unwrap();

    let report = backfill_since(home, Some(Tool::Claude), &source, None).unwrap();
    assert_eq!(report.appended_events, 2);
    let envelopes = raw_envelopes(&canonical_raw_path(home, Tool::Claude, session_id));
    assert_eq!(
        envelopes
            .iter()
            .filter(|event| event.source_event_id.as_deref() == Some("claude-reconcile-shared"))
            .count(),
        1
    );
    assert!(envelopes.iter().any(|event| {
        event.canonical_type == CanonicalType::AssistantMessage
            && event.source_event_id.as_deref() == Some("claude-reconcile-gap")
    }));
    assert!(envelopes
        .iter()
        .any(|event| event.canonical_type == CanonicalType::AssistantDelta));
    assert!(envelopes.iter().any(
        |event| event.canonical_type == CanonicalType::AssistantMessage
            && event.source_event_id.as_deref() == Some("claude-reconcile-final")
    ));
}

fn assert_reconciles_codex(home: &Path, temp_root: &Path) {
    let source = temp_root.join("reconcile-codex");
    fs::create_dir_all(&source).unwrap();
    let session_id = "bbbbbbb2-bbbb-4bbb-8bbb-bbbbbbbbbbb2";
    ingest_hook_event(
        home,
        Tool::Codex,
        json!({
            "session_id": session_id,
            "type": "response_item",
            "payload": {
                "id": "codex-reconcile-shared",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "codex reconcile shared marker"}]
            }
        }),
    )
    .unwrap();
    fs::write(
            source.join(format!("rollout-2026-06-18T12-00-00-{session_id}.jsonl")),
            "{\"timestamp\":\"2026-06-18T12:00:00.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"codex-reconcile-shared\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"codex reconcile shared marker\"}]}}\n\
                 {\"timestamp\":\"2026-06-18T12:00:01.000Z\",\"type\":\"response_item\",\"payload\":{\"id\":\"codex-reconcile-gap\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"codex reconcile gap marker\"}]}}\n",
        )
        .unwrap();

    let report = backfill_since(home, Some(Tool::Codex), &source, None).unwrap();
    assert_eq!(report.appended_events, 1);
    let envelopes = raw_envelopes(&canonical_raw_path(home, Tool::Codex, session_id));
    assert_eq!(
        envelopes
            .iter()
            .filter(|event| event.source_event_id.as_deref() == Some("codex-reconcile-shared"))
            .count(),
        1
    );
    assert!(envelopes
        .iter()
        .any(|event| event.source_event_id.as_deref() == Some("codex-reconcile-gap")));
}

fn assert_reconciles_opencode(home: &Path, temp_root: &Path) {
    let root = temp_root.join("reconcile-opencode");
    let session_id = "ccccccc3-cccc-4ccc-8ccc-ccccccccccc3";
    let message_dir = root.join("storage/message").join(session_id);
    fs::create_dir_all(&message_dir).unwrap();
    ingest_hook_event(
        home,
        Tool::Opencode,
        json!({
            "session_id": session_id,
            "event": "message.updated",
            "id": "opencode-reconcile-shared",
            "text": "opencode reconcile shared marker"
        }),
    )
    .unwrap();
    fs::write(
        message_dir.join("opencode-reconcile-shared.json"),
        serde_json::to_string_pretty(&json!({
            "id": "opencode-reconcile-shared",
            "sessionID": session_id,
            "role": "assistant",
            "text": "opencode reconcile shared marker"
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        message_dir.join("opencode-reconcile-gap.json"),
        serde_json::to_string_pretty(&json!({
            "id": "opencode-reconcile-gap",
            "sessionID": session_id,
            "role": "assistant",
            "text": "opencode reconcile gap marker"
        }))
        .unwrap(),
    )
    .unwrap();

    let report = backfill_since(home, Some(Tool::Opencode), &root, None).unwrap();
    assert_eq!(report.appended_events, 1);
    let envelopes = raw_envelopes(&canonical_raw_path(home, Tool::Opencode, session_id));
    assert_eq!(
        envelopes
            .iter()
            .filter(|event| event.source_event_id.as_deref() == Some("opencode-reconcile-shared"))
            .count(),
        1
    );
    assert!(envelopes
        .iter()
        .any(|event| event.source_event_id.as_deref() == Some("opencode-reconcile-gap")));
}

fn valid_envelope_json() -> Value {
    json!({
        "schema_version": 1,
        "captured_at": "2026-06-17T12:00:00Z",
        "tool": "codex",
        "tool_version": null,
        "session_id": "session/one",
        "filename_session_id": "session_one",
        "turn_id": null,
        "message_id": null,
        "project_root": null,
        "cwd": "/tmp/nabu-fixture",
        "source": "hook",
        "source_event_type": "UserPromptSubmit",
        "canonical_type": "user.message",
        "source_event_id": null,
        "dedupe_key": "sha256:abc",
        "sequence": null,
        "raw_file": null,
        "raw_offset": null,
        "payload": {}
    })
}
