//! Ingest pipeline: home initialization; hook, file, and server event
//! ingestion; append-with-lock; the dedupe sidecar; and payload spilling.

use crate::{
    append_prepared_events, canonical_raw_path, canonical_type_for_payload, chmod,
    create_config_if_missing, create_dir_0700, dedupe_key, harness_home_for_raw_file,
    hook_event_name, i64_pointer, initialize_database, lock_path_for_raw_file,
    message_id_for_payload, opencode_hook_session_id, opencode_server_events_from_payload,
    parse_ingest_file_source, required_string, sanitize_session_id, string_pointer, AppendReport,
    DedupeParts, Error, EventEnvelope, FileIngestReport, InitReport, Result, Source, Tool,
    MAX_INLINE_ENVELOPE_BYTES, SCHEMA_VERSION,
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

pub fn init_home(home: &Path) -> Result<InitReport> {
    let dirs = [
        home.to_path_buf(),
        home.join("raw"),
        home.join("raw").join("codex"),
        home.join("raw").join("claude"),
        home.join("raw").join("opencode"),
        home.join("spool"),
        home.join("spool").join("dedupe"),
        home.join("index"),
        home.join("models"),
        home.join("checkpoints"),
        home.join("blobs"),
        home.join("blobs").join("sha256"),
        home.join("logs"),
        home.join("backups"),
    ];

    for dir in dirs {
        create_dir_0700(&dir)?;
    }

    let config_path = home.join("config.toml");
    create_config_if_missing(&config_path)?;

    let db_path = home.join("index").join("harness.db");
    initialize_database(&db_path)?;

    Ok(InitReport {
        home: home.to_path_buf(),
        db_path,
    })
}

pub fn ingest_hook_event(home: &Path, tool: Tool, payload: Value) -> Result<AppendReport> {
    let source_event_type = hook_event_name(&payload)?.to_string();
    // OpenCode plugin events do not carry a top-level `session_id`; resolve from
    // the tool's own event shapes. Claude/Codex hooks emit `session_id` directly.
    let session_id = match tool {
        Tool::Opencode => opencode_hook_session_id(&payload, &source_event_type)?,
        _ => required_string(&payload, "session_id")?.to_string(),
    };
    let filename_session_id = sanitize_session_id(&session_id);
    let canonical_type = canonical_type_for_payload(tool, &source_event_type, &payload);
    let sequence = sequence_for_payload(tool, &source_event_type, &payload, None);
    let source_event_id = source_event_id_for_payload(tool, &source_event_type, &payload, sequence);
    let raw_file = canonical_raw_path(home, tool, &session_id);

    if let Some(parent) = raw_file.parent() {
        create_dir_0700(parent)?;
    }

    let lock_path = lock_path_for_raw_file(&raw_file);
    let lock_file = OpenOptions::new()
        .create(true)
        // Lock sentinel: content is never written, so do not truncate.
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|source| Error::Io {
            path: lock_path.clone(),
            source,
        })?;
    chmod(&lock_path, 0o600)?;
    lock_file.lock_exclusive().map_err(|source| Error::Io {
        path: lock_path.clone(),
        source,
    })?;

    let append_result = append_envelope_locked(
        home,
        &raw_file,
        EventEnvelope {
            schema_version: SCHEMA_VERSION,
            captured_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            tool,
            tool_version: payload
                .get("tool_version")
                .and_then(Value::as_str)
                .map(str::to_string),
            session_id,
            filename_session_id,
            turn_id: payload
                .get("turn_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            message_id: payload
                .get("message_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            project_root: payload
                .get("project_root")
                .and_then(Value::as_str)
                .map(str::to_string),
            cwd: payload
                .get("cwd")
                .and_then(Value::as_str)
                .map(str::to_string),
            source: Source::Hook,
            source_event_type,
            canonical_type,
            source_event_id,
            dedupe_key: String::new(),
            sequence,
            raw_file: None,
            raw_offset: None,
            payload,
            payload_ref: None,
        },
    );

    let unlock_result = FileExt::unlock(&lock_file).map_err(|source| Error::Io {
        path: lock_path,
        source,
    });

    match (append_result, unlock_result) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

pub fn ingest_file(
    home: &Path,
    tool: Tool,
    source: Source,
    path: &Path,
) -> Result<FileIngestReport> {
    let parsed = parse_ingest_file_source(tool, source, path)?;
    let mut events = parsed.events;
    for event in &mut events {
        event.source = source;
    }
    let appended_events = append_prepared_events(home, events)?
        .into_iter()
        .filter(|report| report.appended)
        .count();
    Ok(FileIngestReport { appended_events })
}

pub fn ingest_opencode_server_messages(
    home: &Path,
    session_id: &str,
    payload: Value,
) -> Result<FileIngestReport> {
    let events = opencode_server_events_from_payload(session_id, payload)?;
    let appended_events = append_prepared_events(home, events)?
        .into_iter()
        .filter(|report| report.appended)
        .count();
    Ok(FileIngestReport { appended_events })
}

pub(crate) fn append_envelope_locked(
    home: &Path,
    raw_file: &Path,
    envelope: EventEnvelope,
) -> Result<AppendReport> {
    let mut reports = append_envelopes_locked(home, raw_file, vec![envelope])?;
    reports
        .pop()
        .ok_or_else(|| Error::Validation("append produced no report".to_string()))
}

pub(crate) fn append_envelopes_locked(
    home: &Path,
    raw_file: &Path,
    events: Vec<EventEnvelope>,
) -> Result<Vec<AppendReport>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(raw_file)
        .map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;
    chmod(raw_file, 0o600)?;

    let mut keyed_events = Vec::with_capacity(events.len());
    let mut lookup_keys = HashSet::with_capacity(events.len());
    for mut envelope in events {
        envelope.dedupe_key = dedupe_key(DedupeParts {
            tool: envelope.tool,
            session_id: &envelope.session_id,
            canonical_type: envelope.canonical_type,
            source_event_id: envelope.source_event_id.as_deref(),
            sequence: envelope.sequence,
            payload: &envelope.payload,
        })?;
        lookup_keys.insert(envelope.dedupe_key.clone());
        keyed_events.push(envelope);
    }

    let mut dedupe_state = append_dedupe_state(home, raw_file, &lookup_keys)?;
    let mut raw_offset = file
        .metadata()
        .map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?
        .len();
    let mut reports = Vec::with_capacity(keyed_events.len());

    for mut envelope in keyed_events {
        if let Some(existing) = dedupe_state.existing(&envelope.dedupe_key) {
            reports.push(AppendReport {
                raw_file: raw_file.to_path_buf(),
                raw_offset: existing.raw_offset,
                session_id: envelope.session_id,
                dedupe_key: envelope.dedupe_key,
                appended: false,
            });
            continue;
        }

        let event_raw_offset = raw_offset;
        envelope.raw_file = Some(raw_file.display().to_string());
        envelope.raw_offset = Some(event_raw_offset as i64);
        spill_payload_if_needed(home, &mut envelope)?;
        envelope.validate()?;

        let line = serde_json::to_vec(&envelope)?;
        file.write_all(&line).map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;
        file.write_all(b"\n").map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;

        let line_len = line.len() as u64 + 1;
        dedupe_state.record_appended(
            envelope.dedupe_key.clone(),
            ExistingRawEvent {
                raw_offset: event_raw_offset,
            },
            line_len,
        );
        raw_offset += line_len;

        reports.push(AppendReport {
            raw_file: raw_file.to_path_buf(),
            raw_offset: event_raw_offset,
            session_id: envelope.session_id,
            dedupe_key: envelope.dedupe_key,
            appended: true,
        });
    }

    dedupe_state.flush();

    Ok(reports)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExistingRawEvent {
    raw_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawDedupeSnapshot {
    pub(crate) events: HashMap<String, ExistingRawEvent>,
    ordered: Vec<(String, u64)>,
    raw_len: u64,
}

impl RawDedupeSnapshot {
    fn empty(raw_len: u64) -> Self {
        Self {
            events: HashMap::new(),
            ordered: Vec::new(),
            raw_len,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AppendDedupeState {
    events: HashMap<String, ExistingRawEvent>,
    pending: Vec<(String, u64)>,
    raw_len: u64,
    key_count: usize,
    bucket_lengths: Vec<u64>,
    sidecar: Option<DedupeSidecarFiles>,
}

impl AppendDedupeState {
    fn from_sidecar(
        events: HashMap<String, ExistingRawEvent>,
        raw_len: u64,
        key_count: usize,
        bucket_lengths: Vec<u64>,
        sidecar: DedupeSidecarFiles,
    ) -> Self {
        Self {
            events,
            pending: Vec::new(),
            raw_len,
            key_count,
            bucket_lengths,
            sidecar: Some(sidecar),
        }
    }

    fn from_snapshot(
        snapshot: RawDedupeSnapshot,
        sidecar: Option<DedupeSidecarFiles>,
        bucket_lengths: Vec<u64>,
        lookup_keys: &HashSet<String>,
    ) -> Self {
        let key_count = snapshot.ordered.len();
        let events = snapshot
            .events
            .into_iter()
            .filter(|(dedupe_key, _)| lookup_keys.contains(dedupe_key))
            .collect();
        Self {
            events,
            pending: Vec::new(),
            raw_len: snapshot.raw_len,
            key_count,
            bucket_lengths,
            sidecar,
        }
    }

    fn existing(&self, dedupe_key: &str) -> Option<&ExistingRawEvent> {
        self.events.get(dedupe_key)
    }

    fn record_appended(
        &mut self,
        dedupe_key: String,
        existing: ExistingRawEvent,
        raw_line_len: u64,
    ) {
        self.pending.push((dedupe_key.clone(), existing.raw_offset));
        self.events.entry(dedupe_key).or_insert(existing);
        self.raw_len = self.raw_len.saturating_add(raw_line_len);
        self.key_count = self.key_count.saturating_add(1);
    }

    fn flush(&mut self) {
        let Some(sidecar) = self.sidecar.as_ref() else {
            return;
        };
        if self.pending.is_empty() {
            return;
        }
        match append_dedupe_sidecar(sidecar, self) {
            Ok(bucket_lengths) => {
                self.bucket_lengths = bucket_lengths;
                if let Err(error) = write_dedupe_sidecar_meta(
                    sidecar,
                    self.raw_len,
                    self.key_count,
                    &self.bucket_lengths,
                ) {
                    eprintln!(
                        "nabu: dedupe sidecar metadata update failed at {}: {}; future appends will rebuild or fall back to raw",
                        sidecar.meta.display(),
                        error
                    );
                    self.sidecar = None;
                    return;
                }
            }
            Err(error) => {
                eprintln!(
                    "nabu: dedupe sidecar update failed at {}: {}; future appends will rebuild or fall back to raw",
                    sidecar.meta.display(),
                    error
                );
                self.sidecar = None;
                return;
            }
        }
        self.pending.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DedupeSidecarFiles {
    pub(crate) buckets_dir: PathBuf,
    meta: PathBuf,
    legacy_keys: PathBuf,
    legacy_offsets: PathBuf,
}

impl DedupeSidecarFiles {
    pub(crate) fn for_raw_file(home: &Path, raw_file: &Path) -> Self {
        let base = raw_file
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("raw");
        let dir = home.join("spool").join("dedupe");
        Self {
            buckets_dir: dir.join(format!("{base}.buckets")),
            meta: dir.join(format!("{base}.meta.json")),
            legacy_keys: dir.join(format!("{base}.keys")),
            legacy_offsets: dir.join(format!("{base}.offsets")),
        }
    }

    pub(crate) fn bucket_path(&self, bucket: usize) -> PathBuf {
        self.buckets_dir.join(format!("{bucket:02x}.dedupe"))
    }

    fn file_paths(&self) -> [&Path; 3] {
        [&self.meta, &self.legacy_keys, &self.legacy_offsets]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DedupeSidecarMeta {
    schema_version: u32,
    raw_len: u64,
    key_count: usize,
    bucket_count: usize,
    bucket_lengths: Vec<u64>,
}

const DEDUPE_SIDECAR_SCHEMA_VERSION: u32 = 2;
const DEDUPE_BUCKET_COUNT: usize = 256;

fn append_dedupe_state(
    home: &Path,
    raw_file: &Path,
    lookup_keys: &HashSet<String>,
) -> Result<AppendDedupeState> {
    let sidecar = DedupeSidecarFiles::for_raw_file(home, raw_file);
    match load_append_dedupe_sidecar(raw_file, &sidecar, lookup_keys) {
        Ok(Some(state)) => Ok(state),
        Ok(None) => rebuild_dedupe_state(raw_file, sidecar, lookup_keys),
        Err(error) => {
            eprintln!(
                "nabu: dedupe sidecar read failed at {}: {}; falling back to raw-derived check",
                sidecar.meta.display(),
                error
            );
            Ok(AppendDedupeState::from_snapshot(
                read_raw_dedupe_snapshot(raw_file)?,
                None,
                zero_bucket_lengths(),
                lookup_keys,
            ))
        }
    }
}

fn rebuild_dedupe_state(
    raw_file: &Path,
    sidecar: DedupeSidecarFiles,
    lookup_keys: &HashSet<String>,
) -> Result<AppendDedupeState> {
    let snapshot = read_raw_dedupe_snapshot(raw_file)?;
    match write_full_dedupe_sidecar(&sidecar, &snapshot) {
        Ok(bucket_lengths) => Ok(AppendDedupeState::from_snapshot(
            snapshot,
            Some(sidecar),
            bucket_lengths,
            lookup_keys,
        )),
        Err(error) => {
            eprintln!(
                "nabu: dedupe sidecar rebuild failed at {}: {}; falling back to raw-derived check",
                sidecar.meta.display(),
                error
            );
            Ok(AppendDedupeState::from_snapshot(
                snapshot,
                None,
                zero_bucket_lengths(),
                lookup_keys,
            ))
        }
    }
}

fn load_append_dedupe_sidecar(
    raw_file: &Path,
    sidecar: &DedupeSidecarFiles,
    lookup_keys: &HashSet<String>,
) -> Result<Option<AppendDedupeState>> {
    let Some((meta, raw_len)) = read_dedupe_sidecar_meta(raw_file, sidecar)? else {
        return Ok(None);
    };
    let mut events = HashMap::new();
    let mut buckets = BTreeMap::<usize, HashSet<String>>::new();
    for dedupe_key in lookup_keys {
        let Some(bucket) = dedupe_bucket_index(dedupe_key) else {
            return Ok(None);
        };
        buckets
            .entry(bucket)
            .or_default()
            .insert(dedupe_key.clone());
    }

    for (bucket, needed) in buckets {
        if !load_dedupe_bucket(
            sidecar,
            bucket,
            meta.bucket_lengths[bucket],
            Some(&needed),
            &mut events,
        )? {
            return Ok(None);
        }
    }

    Ok(Some(AppendDedupeState::from_sidecar(
        events,
        raw_len,
        meta.key_count,
        meta.bucket_lengths,
        sidecar.clone(),
    )))
}

pub(crate) fn load_full_dedupe_sidecar_events(
    raw_file: &Path,
    sidecar: &DedupeSidecarFiles,
) -> Result<Option<HashMap<String, ExistingRawEvent>>> {
    let Some((meta, _)) = read_dedupe_sidecar_meta(raw_file, sidecar)? else {
        return Ok(None);
    };
    let mut events = HashMap::new();
    for bucket in 0..DEDUPE_BUCKET_COUNT {
        if !load_dedupe_bucket(
            sidecar,
            bucket,
            meta.bucket_lengths[bucket],
            None,
            &mut events,
        )? {
            return Ok(None);
        }
    }
    if events.len() > meta.key_count {
        return Ok(None);
    }
    Ok(Some(events))
}

fn read_dedupe_sidecar_meta(
    raw_file: &Path,
    sidecar: &DedupeSidecarFiles,
) -> Result<Option<(DedupeSidecarMeta, u64)>> {
    let raw_len = match fs::metadata(raw_file) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(source) => {
            return Err(Error::Io {
                path: raw_file.to_path_buf(),
                source,
            })
        }
    };
    let meta_bytes = match fs::read(&sidecar.meta) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(Error::Io {
                path: sidecar.meta.clone(),
                source,
            })
        }
    };
    let Ok(meta) = serde_json::from_slice::<DedupeSidecarMeta>(&meta_bytes) else {
        return Ok(None);
    };
    if meta.schema_version != DEDUPE_SIDECAR_SCHEMA_VERSION
        || meta.raw_len != raw_len
        || meta.bucket_count != DEDUPE_BUCKET_COUNT
        || meta.bucket_lengths.len() != DEDUPE_BUCKET_COUNT
    {
        return Ok(None);
    }

    Ok(Some((meta, raw_len)))
}

fn load_dedupe_bucket(
    sidecar: &DedupeSidecarFiles,
    bucket: usize,
    expected_len: u64,
    needed: Option<&HashSet<String>>,
    events: &mut HashMap<String, ExistingRawEvent>,
) -> Result<bool> {
    let path = sidecar.bucket_path(bucket);
    match fs::metadata(&path) {
        Ok(metadata) if metadata.len() == expected_len => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && expected_len == 0 => {
            return Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(source) => return Err(Error::Io { path, source }),
    }
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(source) => return Err(Error::Io { path, source }),
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        let Some((dedupe_key, raw_offset)) = parse_dedupe_bucket_entry(line.trim_end()) else {
            return Ok(false);
        };
        if dedupe_bucket_index(dedupe_key) != Some(bucket) {
            return Ok(false);
        }
        if needed.map(|keys| keys.contains(dedupe_key)).unwrap_or(true) {
            events
                .entry(dedupe_key.to_string())
                .or_insert(ExistingRawEvent { raw_offset });
        }
    }
    Ok(true)
}

fn parse_dedupe_bucket_entry(line: &str) -> Option<(&str, u64)> {
    let (dedupe_key, raw_offset) = line.split_once('\t')?;
    if !valid_dedupe_key(dedupe_key) {
        return None;
    }
    Some((dedupe_key, raw_offset.parse::<u64>().ok()?))
}

fn valid_dedupe_key(value: &str) -> bool {
    value.len() == "sha256:".len() + 64
        && value.starts_with("sha256:")
        && value["sha256:".len()..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn read_raw_dedupe_snapshot(raw_file: &Path) -> Result<RawDedupeSnapshot> {
    if !raw_file.exists() {
        return Ok(RawDedupeSnapshot::empty(0));
    }

    let file = File::open(raw_file).map_err(|source| Error::Io {
        path: raw_file.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut raw_offset = 0u64;
    let mut events = HashMap::new();
    let mut ordered = Vec::new();

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line).map_err(|source| Error::Io {
            path: raw_file.to_path_buf(),
            source,
        })?;
        if bytes == 0 {
            break;
        }
        let parsed: EventEnvelope = serde_json::from_str(line.trim_end())?;
        if !parsed.dedupe_key.is_empty() {
            let event_raw_offset = parsed.raw_offset.unwrap_or(raw_offset as i64).max(0) as u64;
            ordered.push((parsed.dedupe_key.clone(), event_raw_offset));
            events.entry(parsed.dedupe_key).or_insert(ExistingRawEvent {
                raw_offset: event_raw_offset,
            });
        }
        raw_offset += bytes as u64;
    }

    Ok(RawDedupeSnapshot {
        events,
        ordered,
        raw_len: raw_offset,
    })
}

fn write_full_dedupe_sidecar(
    sidecar: &DedupeSidecarFiles,
    snapshot: &RawDedupeSnapshot,
) -> Result<Vec<u64>> {
    let Some(parent) = sidecar.meta.parent() else {
        return Err(Error::Validation(
            "dedupe sidecar has no parent".to_string(),
        ));
    };
    create_dir_0700(parent)?;
    match fs::remove_dir_all(&sidecar.buckets_dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(Error::Io {
                path: sidecar.buckets_dir.clone(),
                source,
            })
        }
    }
    create_dir_0700(&sidecar.buckets_dir)?;

    let mut bucket_lengths = zero_bucket_lengths();
    let mut bucket_files = (0..DEDUPE_BUCKET_COUNT)
        .map(|_| None)
        .collect::<Vec<Option<File>>>();
    for (dedupe_key, raw_offset) in &snapshot.ordered {
        let Some(bucket) = dedupe_bucket_index(dedupe_key) else {
            return Err(Error::Validation(format!(
                "invalid dedupe key in raw snapshot: {dedupe_key}"
            )));
        };
        if bucket_files[bucket].is_none() {
            let path = sidecar.bucket_path(bucket);
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
            chmod(&path, 0o600)?;
            bucket_files[bucket] = Some(file);
        }
        let entry = dedupe_bucket_entry(dedupe_key, *raw_offset);
        if let Some(file) = bucket_files[bucket].as_mut() {
            file.write_all(entry.as_bytes())
                .map_err(|source| Error::Io {
                    path: sidecar.bucket_path(bucket),
                    source,
                })?;
        }
        bucket_lengths[bucket] = bucket_lengths[bucket].saturating_add(entry.len() as u64);
    }
    drop(bucket_files);

    write_dedupe_sidecar_meta(
        sidecar,
        snapshot.raw_len,
        snapshot.ordered.len(),
        &bucket_lengths,
    )?;
    Ok(bucket_lengths)
}

fn append_dedupe_sidecar(
    sidecar: &DedupeSidecarFiles,
    state: &AppendDedupeState,
) -> Result<Vec<u64>> {
    let Some(parent) = sidecar.meta.parent() else {
        return Err(Error::Validation(
            "dedupe sidecar has no parent".to_string(),
        ));
    };
    create_dir_0700(parent)?;
    create_dir_0700(&sidecar.buckets_dir)?;

    let mut bucket_lengths = state.bucket_lengths.clone();
    if bucket_lengths.len() != DEDUPE_BUCKET_COUNT {
        return Err(Error::Validation(
            "dedupe sidecar bucket metadata is invalid".to_string(),
        ));
    }
    let mut pending_by_bucket = BTreeMap::<usize, Vec<&(String, u64)>>::new();
    for entry in &state.pending {
        let Some(bucket) = dedupe_bucket_index(&entry.0) else {
            return Err(Error::Validation(format!(
                "invalid pending dedupe key: {}",
                entry.0
            )));
        };
        pending_by_bucket.entry(bucket).or_default().push(entry);
    }

    for (bucket, entries) in pending_by_bucket {
        let path = sidecar.bucket_path(bucket);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
        for (dedupe_key, raw_offset) in entries {
            let entry = dedupe_bucket_entry(dedupe_key, *raw_offset);
            file.write_all(entry.as_bytes())
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
            bucket_lengths[bucket] = bucket_lengths[bucket].saturating_add(entry.len() as u64);
        }
        chmod(&path, 0o600)?;
    }
    Ok(bucket_lengths)
}

fn write_dedupe_sidecar_meta(
    sidecar: &DedupeSidecarFiles,
    raw_len: u64,
    key_count: usize,
    bucket_lengths: &[u64],
) -> Result<()> {
    let meta = DedupeSidecarMeta {
        schema_version: DEDUPE_SIDECAR_SCHEMA_VERSION,
        raw_len,
        key_count,
        bucket_count: DEDUPE_BUCKET_COUNT,
        bucket_lengths: bucket_lengths.to_vec(),
    };
    let bytes = serde_json::to_vec_pretty(&meta)?;
    fs::write(&sidecar.meta, bytes).map_err(|source| Error::Io {
        path: sidecar.meta.clone(),
        source,
    })?;
    chmod(&sidecar.meta, 0o600)
}

pub(crate) fn dedupe_bucket_index(dedupe_key: &str) -> Option<usize> {
    if !valid_dedupe_key(dedupe_key) {
        return None;
    }
    usize::from_str_radix(&dedupe_key["sha256:".len().."sha256:".len() + 2], 16).ok()
}

fn dedupe_bucket_entry(dedupe_key: &str, raw_offset: u64) -> String {
    format!("{dedupe_key}\t{raw_offset}\n")
}

fn zero_bucket_lengths() -> Vec<u64> {
    vec![0; DEDUPE_BUCKET_COUNT]
}

pub(crate) fn remove_dedupe_sidecar_for_raw_file(raw_file: &Path) -> Result<()> {
    let home = harness_home_for_raw_file(raw_file);
    let sidecar = DedupeSidecarFiles::for_raw_file(&home, raw_file);
    for path in sidecar.file_paths() {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(Error::Io {
                    path: path.to_path_buf(),
                    source,
                })
            }
        }
    }
    match fs::remove_dir_all(&sidecar.buckets_dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(Error::Io {
                path: sidecar.buckets_dir,
                source,
            })
        }
    }
    Ok(())
}

fn spill_payload_if_needed(home: &Path, envelope: &mut EventEnvelope) -> Result<()> {
    if serde_json::to_vec(envelope)?.len() <= MAX_INLINE_ENVELOPE_BYTES {
        return Ok(());
    }

    let payload_bytes = serde_json::to_vec(&envelope.payload)?;
    let mut hasher = Sha256::new();
    hasher.update(&payload_bytes);
    let hash = hex::encode(hasher.finalize());
    let blob_dir = home.join("blobs").join("sha256");
    create_dir_0700(&blob_dir)?;
    let blob_path = blob_dir.join(format!("{hash}.json"));
    if !blob_path.exists() {
        fs::write(&blob_path, &payload_bytes).map_err(|source| Error::Io {
            path: blob_path.clone(),
            source,
        })?;
        chmod(&blob_path, 0o600)?;
    }
    envelope.payload = Value::Null;
    envelope.payload_ref = Some(format!("sha256:{hash}"));
    Ok(())
}

pub(crate) fn source_event_id_for_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
    sequence: Option<i64>,
) -> Option<String> {
    if source_event_type == "MessageDisplay" {
        if let Some(message_id) = payload.get("message_id").and_then(Value::as_str) {
            let index = payload
                .get("index")
                .and_then(Value::as_i64)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let final_state = payload
                .get("final")
                .and_then(Value::as_bool)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "false".to_string());
            return Some(format!("{message_id}:{index}:{final_state}"));
        }
    }
    if source_event_type == "item/agentMessage/delta" {
        if let Some(message_id) = message_id_for_payload(payload) {
            let sequence = sequence
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            return Some(format!("{message_id}:{sequence}:delta"));
        }
    }
    if tool == Tool::Opencode
        && matches!(
            source_event_type,
            "message.part.updated" | "message.part.removed"
        )
    {
        for key in ["event_id", "id", "part_id"] {
            if let Some(value) = payload.get(key).and_then(Value::as_str) {
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
        for pointer in [
            "/payload/event_id",
            "/payload/id",
            "/payload/part_id",
            "/part/id",
            "/payload/part/id",
        ] {
            if let Some(value) = string_pointer(payload, pointer) {
                return Some(value);
            }
        }
        return None;
    }

    for key in ["event_id", "message_id", "turn_id", "id"] {
        if let Some(value) = payload.get(key).and_then(Value::as_str) {
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    for pointer in [
        "/payload/event_id",
        "/payload/message_id",
        "/payload/turn_id",
        "/payload/call_id",
        "/payload/id",
        "/payload/item/id",
        "/payload/item/message_id",
        "/payload/item/messageId",
        "/params/event_id",
        "/params/message_id",
        "/params/turn_id",
        "/params/call_id",
        "/params/id",
        "/params/item/id",
        "/params/item/message_id",
        "/params/item/messageId",
        "/item/id",
        "/item/message_id",
        "/item/messageId",
        "/message/id",
        "/messageID",
        "/uuid",
        "/attachment/toolUseID",
        "/toolUseID",
    ] {
        if let Some(value) = string_pointer(payload, pointer) {
            return Some(value);
        }
    }
    None
}

pub(crate) fn sequence_for_payload(
    tool: Tool,
    source_event_type: &str,
    payload: &Value,
    backfill_offset: Option<u64>,
) -> Option<i64> {
    for pointer in [
        "/sequence",
        "/index",
        "/ordinal",
        "/order",
        "/payload/sequence",
        "/payload/index",
        "/payload/ordinal",
        "/payload/order",
        "/params/sequence",
        "/params/index",
        "/params/ordinal",
        "/params/order",
    ] {
        if let Some(sequence) = i64_pointer(payload, pointer) {
            return Some(sequence);
        }
    }

    if tool == Tool::Codex {
        for pointer in [
            "/item_index",
            "/item_ordinal",
            "/turn_index",
            "/turn_ordinal",
            "/response_index",
            "/output_index",
            "/payload/item_index",
            "/payload/item_ordinal",
            "/payload/turn_index",
            "/payload/turn_ordinal",
            "/payload/response_index",
            "/payload/output_index",
            "/payload/item/index",
            "/payload/item/ordinal",
            "/payload/turn/index",
            "/payload/turn/ordinal",
            "/params/item_index",
            "/params/item_ordinal",
            "/params/turn_index",
            "/params/turn_ordinal",
            "/params/response_index",
            "/params/output_index",
            "/params/item/index",
            "/params/item/ordinal",
            "/params/turn/index",
            "/params/turn/ordinal",
            "/item/index",
            "/item/ordinal",
            "/turn/index",
            "/turn/ordinal",
        ] {
            if let Some(sequence) = i64_pointer(payload, pointer) {
                return Some(sequence);
            }
        }
    }

    if tool == Tool::Opencode
        && matches!(
            source_event_type,
            "message.part.updated" | "message.part.removed"
        )
    {
        for pointer in [
            "/part_index",
            "/part_sequence",
            "/payload/part_index",
            "/payload/part_sequence",
            "/part/index",
            "/part/sequence",
            "/payload/part/index",
            "/payload/part/sequence",
            "/params/part_index",
            "/params/part_sequence",
            "/params/part/index",
            "/params/part/sequence",
        ] {
            if let Some(sequence) = i64_pointer(payload, pointer) {
                return Some(sequence);
            }
        }
    }

    backfill_offset.and_then(|offset| i64::try_from(offset).ok())
}

pub(crate) fn resolved_payload_for_envelope(
    raw_file: &Path,
    envelope: &EventEnvelope,
) -> Result<Value> {
    let Some(payload_ref) = envelope.payload_ref.as_deref() else {
        return Ok(envelope.payload.clone());
    };
    let Some(hash) = payload_ref.strip_prefix("sha256:") else {
        return Err(Error::Validation(format!(
            "unsupported payload_ref: {payload_ref}"
        )));
    };
    let blob_path = harness_home_for_raw_file(raw_file)
        .join("blobs")
        .join("sha256")
        .join(format!("{hash}.json"));
    let content = fs::read_to_string(&blob_path).map_err(|source| Error::Io {
        path: blob_path,
        source,
    })?;
    Ok(serde_json::from_str(&content)?)
}
