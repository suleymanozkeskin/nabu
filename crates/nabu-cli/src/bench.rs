//! Hidden benchmark subcommands for ingest and search latency.
//!
//! `run_ingest_bench` and `run_search_bench` back the `nabu bench` commands:
//! they seed a throwaway home, replay events or queries, and report p95/p99
//! latency against fixed gates. `run_large_session_ingest_bench` exercises the
//! dedupe path against a pre-seeded large session. All helpers (seeding,
//! payload construction, percentile, JSONL loading) are private.

use nabu_core::{
    canonical_raw_path, dedupe_key, ingest_hook_event, init_home, sanitize_session_id,
    search_history_page, CanonicalType, DedupeParts, Error, EventEnvelope, SearchOptions, Source,
    Tool, SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(crate) fn run_ingest_bench(
    events_path: &PathBuf,
    seed_events: usize,
    iterations: usize,
) -> nabu_core::Result<Value> {
    let events = load_bench_events(events_path)?;
    if events.is_empty() {
        return Err(Error::Validation(
            "benchmark events file must contain at least one JSONL event".to_string(),
        ));
    }
    if seed_events > 0 && seed_events < 10_000 {
        return Err(Error::Validation(
            "--seed-events must be 0 or at least 10000".to_string(),
        ));
    }

    let home = std::env::temp_dir().join(format!(
        "nabu-bench-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    init_home(&home)?;

    let iterations = iterations.clamp(1, 10_000);
    if seed_events > 0 {
        return run_large_session_ingest_bench(
            &home,
            events_path,
            &events,
            seed_events,
            iterations,
        );
    }

    let mut durations = Vec::with_capacity(iterations);
    for index in 0..iterations {
        let (tool, mut payload) = events[index % events.len()].clone();
        payload["message_id"] = json!(format!("bench-message-{index}"));
        payload["sequence"] = json!(index as i64);
        let started = Instant::now();
        ingest_hook_event(&home, tool, payload)?;
        durations.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    durations.sort_by(|left, right| left.total_cmp(right));
    let p95 = percentile(&durations, 0.95);
    let p99 = percentile(&durations, 0.99);

    Ok(json!({
        "iterations": iterations,
        "p95_ms": p95,
        "p99_ms": p99,
        "p95_under_50_ms": p95 < 50.0,
        "p99_under_250_ms": p99 < 250.0,
        "events": events_path,
        "home": home
    }))
}

fn run_large_session_ingest_bench(
    home: &Path,
    events_path: &PathBuf,
    events: &[(Tool, Value)],
    seed_events: usize,
    iterations: usize,
) -> nabu_core::Result<Value> {
    let tool = events[0].0;
    let session_id = "bench-large-session";
    seed_large_bench_session(home, tool, session_id, seed_events)?;

    let warm_payload = bench_user_payload(
        session_id,
        "bench-seed-message-0",
        0,
        "bench large-session seed marker 0",
    );
    let warm_report = ingest_hook_event(home, tool, warm_payload)?;
    if warm_report.appended {
        return Err(Error::Validation(
            "large-session bench warmup unexpectedly appended duplicate seed event".to_string(),
        ));
    }

    let mut new_durations = Vec::with_capacity(iterations);
    let mut duplicate_durations = Vec::with_capacity(iterations);
    for index in 0..iterations {
        let sequence = seed_events.saturating_add(index);
        let new_payload = bench_user_payload(
            session_id,
            &format!("bench-measured-message-{index}"),
            sequence,
            &format!("bench large-session measured marker {index}"),
        );
        let started = Instant::now();
        let new_report = ingest_hook_event(home, tool, new_payload)?;
        new_durations.push(started.elapsed().as_secs_f64() * 1000.0);
        if !new_report.appended {
            return Err(Error::Validation(
                "large-session bench measured new event was deduped".to_string(),
            ));
        }

        let duplicate_index = index % seed_events;
        let duplicate_payload = bench_user_payload(
            session_id,
            &format!("bench-seed-message-{duplicate_index}"),
            duplicate_index,
            &format!("bench large-session seed marker {duplicate_index}"),
        );
        let started = Instant::now();
        let duplicate_report = ingest_hook_event(home, tool, duplicate_payload)?;
        duplicate_durations.push(started.elapsed().as_secs_f64() * 1000.0);
        if duplicate_report.appended {
            return Err(Error::Validation(
                "large-session bench measured duplicate event appended".to_string(),
            ));
        }
    }

    new_durations.sort_by(|left, right| left.total_cmp(right));
    duplicate_durations.sort_by(|left, right| left.total_cmp(right));
    let new_p95 = percentile(&new_durations, 0.95);
    let new_p99 = percentile(&new_durations, 0.99);
    let duplicate_p95 = percentile(&duplicate_durations, 0.95);
    let duplicate_p99 = percentile(&duplicate_durations, 0.99);

    Ok(json!({
        "mode": "large_session",
        "iterations": iterations,
        "seed_events": seed_events,
        "events": events_path,
        "home": home,
        "new_event": {
            "p95_ms": new_p95,
            "p99_ms": new_p99,
            "p95_under_50_ms": new_p95 < 50.0,
            "p99_under_250_ms": new_p99 < 250.0
        },
        "duplicate_event": {
            "p95_ms": duplicate_p95,
            "p99_ms": duplicate_p99,
            "p95_under_50_ms": duplicate_p95 < 50.0,
            "p99_under_250_ms": duplicate_p99 < 250.0
        }
    }))
}

fn seed_large_bench_session(
    home: &Path,
    tool: Tool,
    session_id: &str,
    seed_events: usize,
) -> nabu_core::Result<()> {
    let raw_path = canonical_raw_path(home, tool, session_id);
    if let Some(parent) = raw_path.parent() {
        fs::create_dir_all(parent).map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&raw_path)
        .map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?;
    let mut raw_offset = file
        .metadata()
        .map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?
        .len();
    for index in 0..seed_events {
        let payload = bench_user_payload(
            session_id,
            &format!("bench-seed-message-{index}"),
            index,
            &format!("bench large-session seed marker {index}"),
        );
        let source_event_id = format!("bench-seed-message-{index}");
        let dedupe_key = dedupe_key(DedupeParts {
            tool,
            session_id,
            canonical_type: CanonicalType::UserMessage,
            source_event_id: Some(&source_event_id),
            sequence: Some(index as i64),
            payload: &payload,
        })?;
        let envelope = EventEnvelope {
            schema_version: SCHEMA_VERSION,
            captured_at: "2026-06-18T00:00:00Z".to_string(),
            tool,
            tool_version: None,
            session_id: session_id.to_string(),
            filename_session_id: sanitize_session_id(session_id),
            turn_id: None,
            message_id: Some(source_event_id.clone()),
            project_root: Some("/tmp/nabu-bench".to_string()),
            cwd: Some("/tmp/nabu-bench".to_string()),
            source: Source::Hook,
            source_event_type: "UserPromptSubmit".to_string(),
            canonical_type: CanonicalType::UserMessage,
            source_event_id: Some(source_event_id),
            dedupe_key,
            sequence: Some(index as i64),
            raw_file: Some(raw_path.display().to_string()),
            raw_offset: Some(raw_offset as i64),
            payload,
            payload_ref: None,
        };
        envelope.validate()?;
        let line = serde_json::to_vec(&envelope)?;
        file.write_all(&line).map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?;
        file.write_all(b"\n").map_err(|source| Error::Io {
            path: raw_path.clone(),
            source,
        })?;
        raw_offset += line.len() as u64 + 1;
    }
    Ok(())
}

fn bench_user_payload(session_id: &str, message_id: &str, sequence: usize, prompt: &str) -> Value {
    json!({
        "session_id": session_id,
        "hook_event_name": "UserPromptSubmit",
        "message_id": message_id,
        "sequence": sequence as i64,
        "cwd": "/tmp/nabu-bench",
        "project_root": "/tmp/nabu-bench",
        "prompt": prompt
    })
}

pub(crate) fn run_search_bench(
    home: &Path,
    query: &str,
    iterations: usize,
    limit: usize,
) -> nabu_core::Result<Value> {
    if query.trim().is_empty() {
        return Err(Error::Validation(
            "benchmark query must not be empty".to_string(),
        ));
    }
    let iterations = iterations.clamp(1, 10_000);
    let limit = limit.clamp(1, 50);
    let mut durations = Vec::with_capacity(iterations);
    let mut result_count = 0usize;
    for _ in 0..iterations {
        let started = Instant::now();
        let page = search_history_page(
            home,
            query,
            SearchOptions {
                limit,
                ..SearchOptions::default()
            },
        )?;
        durations.push(started.elapsed().as_secs_f64() * 1000.0);
        result_count = page.returned;
    }
    durations.sort_by(|left, right| left.total_cmp(right));
    let p95 = percentile(&durations, 0.95);
    let p99 = percentile(&durations, 0.99);

    Ok(json!({
        "iterations": iterations,
        "query": query,
        "limit": limit,
        "result_count": result_count,
        "p95_ms": p95,
        "p99_ms": p99,
        "p95_under_200_ms": p95 < 200.0,
        "home": home
    }))
}

fn load_bench_events(events_path: &PathBuf) -> nabu_core::Result<Vec<(Tool, Value)>> {
    let file = File::open(events_path).map_err(|source| Error::Io {
        path: events_path.clone(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|source| Error::Io {
            path: events_path.clone(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let payload: Value = serde_json::from_str(&line)?;
        let tool = payload
            .get("tool")
            .and_then(Value::as_str)
            .map(Tool::from_str)
            .transpose()?
            .unwrap_or(Tool::Claude);
        events.push((tool, payload));
    }

    Ok(events)
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    let index = ((sorted_values.len() as f64 - 1.0) * percentile).ceil() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}
