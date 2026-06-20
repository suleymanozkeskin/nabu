//! Feature-gated semantic implementation: embedding-model management, the
//! fastembed embedder, vector index read/write, the embedding pipeline, and
//! CPU core-count detection. Both the real `#[cfg(feature = "semantic")]` path
//! and the `#[cfg(not(feature = "semantic"))]` stubs live here so the facade
//! re-exports the same names in both builds (hard constraint 4).

use crate::*;
#[cfg(feature = "semantic")]
use rusqlite::params;
use rusqlite::Connection;
#[cfg(feature = "semantic")]
use serde_json::Value;
#[cfg(all(feature = "semantic", target_os = "linux"))]
use std::collections::HashSet;

pub(crate) const SEMANTIC_MODEL_ID: &str = "embeddinggemma-300m-q4";
pub(crate) const SEMANTIC_MODEL_REPO: &str = "onnx-community/embeddinggemma-300m-ONNX";
pub(crate) const SEMANTIC_VECTOR_DIMENSIONS: usize = 256;
pub(crate) const SEMANTIC_MODEL_REMOTE_FILES: &[(&str, &str)] = &[
    ("onnx/model_q4.onnx", "onnx/model_q4.onnx"),
    ("onnx/model_q4.onnx_data", "onnx/model_q4.onnx_data"),
    ("tokenizer.json", "tokenizer.json"),
    ("config.json", "config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
];
#[cfg(any(feature = "semantic", test))]
const EMBEDDING_GEMMA_QUERY_PREFIX: &str = "task: search result | query: ";
#[cfg(any(feature = "semantic", test))]
const EMBEDDING_GEMMA_DOCUMENT_PREFIX: &str = "title: none | text: ";
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_MAX_LENGTH: usize = 2048;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_BATCH_SIZE: usize = 64;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_WRITE_CHUNK_SIZE: usize = 2048;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_COLLECT_BATCH_SIZE: usize = 4096;
#[cfg(feature = "semantic")]
const SEMANTIC_EMBED_PROGRESS_INTERVAL: StdDuration = StdDuration::from_secs(2);

#[cfg(feature = "semantic")]
struct FastembedEmbedder {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    batch_size: usize,
    intra_threads: usize,
}

#[cfg(feature = "semantic")]
impl Embedder for FastembedEmbedder {
    fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>> {
        let prompted = documents
            .iter()
            .map(|document| document_embedding_input(document))
            .collect::<Vec<_>>();
        let mut model = self.model.lock().map_err(|_| {
            Error::SemanticUnavailable("embedding model lock is poisoned".to_string())
        })?;
        let vectors = model
            .embed(&prompted, Some(self.batch_size))
            .map_err(|source| {
                Error::SemanticUnavailable(format!("document embedding failed: {source}"))
            })?;
        vectors
            .into_iter()
            .map(truncate_and_normalize_embedding)
            .collect()
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let prompted = [query_embedding_input(query)];
        let mut model = self.model.lock().map_err(|_| {
            Error::SemanticUnavailable("embedding model lock is poisoned".to_string())
        })?;
        let mut vectors = model.embed(&prompted, Some(1)).map_err(|source| {
            Error::SemanticUnavailable(format!("query embedding failed: {source}"))
        })?;
        let vector = vectors.pop().ok_or_else(|| {
            Error::SemanticUnavailable("query embedding returned no vector".to_string())
        })?;
        truncate_and_normalize_embedding(vector)
    }

    fn document_batch_size(&self) -> usize {
        self.batch_size
    }

    fn intra_threads(&self) -> usize {
        self.intra_threads
    }
}

#[cfg(any(feature = "semantic", test))]
pub(crate) fn document_embedding_input(document: &str) -> String {
    format!("{EMBEDDING_GEMMA_DOCUMENT_PREFIX}{}", document.trim())
}

#[cfg(any(feature = "semantic", test))]
pub(crate) fn query_embedding_input(query: &str) -> String {
    format!("{EMBEDDING_GEMMA_QUERY_PREFIX}{}", query.trim())
}

#[cfg(feature = "semantic")]
fn truncate_and_normalize_embedding(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    if vector.len() < SEMANTIC_VECTOR_DIMENSIONS {
        return Err(Error::SemanticUnavailable(format!(
            "embedding returned {} dimensions, expected at least {}",
            vector.len(),
            SEMANTIC_VECTOR_DIMENSIONS
        )));
    }
    vector.truncate(SEMANTIC_VECTOR_DIMENSIONS);
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    Ok(vector)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileSignature {
    len: u64,
    modified_nanos: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SemanticStatusSignature {
    model_files: Option<Vec<FileSignature>>,
    index_files: Vec<Option<FileSignature>>,
}

#[derive(Debug, Clone)]
struct SemanticStatusCacheEntry {
    signature: SemanticStatusSignature,
    status: EmbeddingModelStatus,
}

#[cfg(feature = "semantic")]
struct CachedLocalEmbedder {
    model_files: Vec<FileSignature>,
    embedder: Arc<FastembedEmbedder>,
}

pub fn embedding_model_status(home: &Path) -> EmbeddingModelStatus {
    let signature = semantic_status_signature(home);
    let cache_key = home.to_path_buf();
    if let Ok(cache) = semantic_status_cache().lock() {
        if let Some(entry) = cache.get(&cache_key) {
            if entry.signature == signature {
                return entry.status.clone();
            }
        }
    }

    let status = embedding_model_status_uncached(home, &signature);
    if let Ok(mut cache) = semantic_status_cache().lock() {
        cache.insert(
            cache_key,
            SemanticStatusCacheEntry {
                signature,
                status: status.clone(),
            },
        );
    }
    status
}

fn embedding_model_status_uncached(
    home: &Path,
    signature: &SemanticStatusSignature,
) -> EmbeddingModelStatus {
    let cache_path = semantic_model_cache_path(home);
    let feature_enabled = cfg!(feature = "semantic");
    let model_present = signature.model_files.is_some();
    let vector_rows = if feature_enabled && model_present {
        semantic_vector_row_count(home).unwrap_or(0)
    } else {
        0
    };
    let semantic_available = feature_enabled && model_present && vector_rows > 0;
    let message = if !feature_enabled {
        "semantic feature is disabled in this build".to_string()
    } else if !model_present {
        "semantic feature is enabled, but the local model is not installed".to_string()
    } else if vector_rows == 0 {
        "semantic feature is enabled and the local model is installed, but the vector index has no embeddings; run nabu index --once".to_string()
    } else {
        "semantic search is available".to_string()
    };

    EmbeddingModelStatus {
        feature_enabled,
        model_id: SEMANTIC_MODEL_ID.to_string(),
        model_present,
        semantic_available,
        cache_path: cache_path.display().to_string(),
        expected_dimensions: SEMANTIC_VECTOR_DIMENSIONS,
        message,
    }
}

fn semantic_status_cache() -> &'static Mutex<HashMap<PathBuf, SemanticStatusCacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, SemanticStatusCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn prune_embedding_cache(home: &Path) -> Result<StorageFootprint> {
    let model_root = home.join("models");
    if model_root.exists() {
        fs::remove_dir_all(&model_root).map_err(|source| Error::Io {
            path: model_root.clone(),
            source,
        })?;
    }
    create_dir_0700(&model_root)?;
    Ok(storage_footprint(home))
}

pub fn embedding_model_disclosure(home: &Path, model: &str) -> Result<EmbeddingModelDisclosure> {
    if model != SEMANTIC_MODEL_ID {
        return Err(Error::Validation(format!(
            "unsupported embedding model: {model}"
        )));
    }
    let cache_path = semantic_model_cache_path(home);
    let current_on_disk_bytes = directory_size(&cache_path).unwrap_or(0);
    Ok(EmbeddingModelDisclosure {
        model_id: SEMANTIC_MODEL_ID.to_string(),
        repository: SEMANTIC_MODEL_REPO.to_string(),
        cache_path: cache_path.display().to_string(),
        total_files: SEMANTIC_MODEL_REMOTE_FILES.len(),
        current_on_disk_bytes,
        model_present: semantic_model_files_present(home),
        license_summary: gemma_terms_summary().to_string(),
    })
}

fn gemma_terms_summary() -> &'static str {
    "Gemma Terms of Use: open-weight license permitting responsible commercial use, fine-tuning, and redistribution; no per-token fees."
}

pub fn download_embedding_model(home: &Path, model: &str) -> Result<EmbeddingDownloadReport> {
    download_embedding_model_with_progress(home, model, |_| {})
}

#[cfg(feature = "semantic")]
pub fn download_embedding_model_with_progress<F>(
    home: &Path,
    model: &str,
    mut progress: F,
) -> Result<EmbeddingDownloadReport>
where
    F: FnMut(EmbeddingDownloadProgress),
{
    if model != SEMANTIC_MODEL_ID {
        return Err(Error::Validation(format!(
            "unsupported embedding model: {model}"
        )));
    }

    init_home(home)?;
    let cache_path = semantic_model_cache_path(home);
    create_dir_0700(&cache_path)?;
    if semantic_model_files_present(home) {
        return Ok(EmbeddingDownloadReport {
            model_id: SEMANTIC_MODEL_ID.to_string(),
            cache_path: cache_path.display().to_string(),
            downloaded_files: 0,
            total_files: SEMANTIC_MODEL_REMOTE_FILES.len(),
            downloaded_bytes: 0,
            on_disk_bytes: directory_size(&cache_path).unwrap_or(0),
            license_summary: gemma_terms_summary().to_string(),
        });
    }
    let transient_cache = cache_path.join(".hf-download-cache");
    if transient_cache.exists() {
        fs::remove_dir_all(&transient_cache).map_err(|source| Error::Io {
            path: transient_cache.clone(),
            source,
        })?;
    }
    create_dir_0700(&transient_cache)?;

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(transient_cache.clone())
        .with_progress(false)
        .build()
        .map_err(|source| Error::SemanticUnavailable(format!("model download failed: {source}")))?;
    let repo = api.model(SEMANTIC_MODEL_REPO.to_string());
    let total_files = SEMANTIC_MODEL_REMOTE_FILES.len();
    let mut downloaded_files = 0usize;
    let mut downloaded_bytes = 0u64;

    for (remote, local) in SEMANTIC_MODEL_REMOTE_FILES {
        progress(EmbeddingDownloadProgress {
            model_id: SEMANTIC_MODEL_ID.to_string(),
            file: (*remote).to_string(),
            downloaded_files,
            total_files,
            phase: "downloading".to_string(),
        });
        let source_path = repo.get(remote).map_err(|source| {
            Error::SemanticUnavailable(format!("model download failed for {remote}: {source}"))
        })?;
        let source_path = fs::canonicalize(&source_path).unwrap_or(source_path);
        let target_path = cache_path.join(local);
        if let Some(parent) = target_path.parent() {
            create_dir_0700(parent)?;
        }
        fs::copy(&source_path, &target_path).map_err(|source| Error::Io {
            path: target_path.clone(),
            source,
        })?;
        downloaded_bytes = downloaded_bytes.saturating_add(
            fs::metadata(&target_path)
                .map_err(|source| Error::Io {
                    path: target_path.clone(),
                    source,
                })?
                .len(),
        );
        chmod(&target_path, 0o600)?;
        downloaded_files += 1;
        progress(EmbeddingDownloadProgress {
            model_id: SEMANTIC_MODEL_ID.to_string(),
            file: (*remote).to_string(),
            downloaded_files,
            total_files,
            phase: "stored".to_string(),
        });
    }

    fs::remove_dir_all(&transient_cache).map_err(|source| Error::Io {
        path: transient_cache,
        source,
    })?;

    Ok(EmbeddingDownloadReport {
        model_id: SEMANTIC_MODEL_ID.to_string(),
        cache_path: cache_path.display().to_string(),
        downloaded_files,
        total_files,
        downloaded_bytes,
        on_disk_bytes: directory_size(&cache_path).unwrap_or(downloaded_bytes),
        license_summary: gemma_terms_summary().to_string(),
    })
}

#[cfg(not(feature = "semantic"))]
pub fn download_embedding_model_with_progress<F>(
    _home: &Path,
    _model: &str,
    _progress: F,
) -> Result<EmbeddingDownloadReport>
where
    F: FnMut(EmbeddingDownloadProgress),
{
    Err(Error::SemanticUnavailable(
        "semantic backend is not available in this build; rebuild with --features semantic to enable explicit model download".to_string(),
    ))
}

pub(crate) fn semantic_search_available(home: &Path) -> bool {
    if !cfg!(feature = "semantic") {
        return false;
    }
    embedding_model_status(home).semantic_available
}

pub(crate) fn semantic_model_cache_path(home: &Path) -> PathBuf {
    home.join("models").join(SEMANTIC_MODEL_ID)
}

fn semantic_model_files_present(home: &Path) -> bool {
    semantic_model_file_signatures(home).is_some()
}

fn semantic_status_signature(home: &Path) -> SemanticStatusSignature {
    let model_files = semantic_model_file_signatures(home);
    let index_files = if cfg!(feature = "semantic") && model_files.is_some() {
        semantic_index_file_signatures(home)
    } else {
        Vec::new()
    };
    SemanticStatusSignature {
        model_files,
        index_files,
    }
}

fn semantic_index_file_signatures(home: &Path) -> Vec<Option<FileSignature>> {
    let db_path = home.join("index").join("harness.db");
    vec![
        file_signature(&db_path),
        file_signature(&db_path.with_file_name("harness.db-wal")),
        file_signature(&db_path.with_file_name("harness.db-shm")),
    ]
}

fn semantic_model_file_signatures(home: &Path) -> Option<Vec<FileSignature>> {
    let cache_path = semantic_model_cache_path(home);
    let mut signatures = Vec::with_capacity(SEMANTIC_MODEL_REMOTE_FILES.len());
    for (_, local) in SEMANTIC_MODEL_REMOTE_FILES {
        let path = cache_path.join(local);
        if !path.is_file() {
            return None;
        }
        signatures.push(file_signature(&path)?);
    }
    Some(signatures)
}

fn file_signature(path: &Path) -> Option<FileSignature> {
    let metadata = fs::metadata(path).ok()?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    Some(FileSignature {
        len: metadata.len(),
        modified_nanos,
    })
}

fn semantic_vector_row_count(home: &Path) -> Result<i64> {
    let db_path = home.join("index").join("harness.db");
    if !db_path.exists() {
        return Ok(0);
    }
    let conn = open_index(&db_path)?;
    if !table_exists(&conn, &db_path, "vector_unit_embeddings")? {
        return Ok(0);
    }
    table_count(&conn, &db_path, "vector_unit_embeddings")
}

#[cfg(feature = "semantic")]
fn semantic_intra_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .max(1);
    let requested = std::env::var("NABU_SEMANTIC_INTRA_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .or_else(platform_physical_core_count)
        .unwrap_or(available);
    requested.clamp(1, available)
}

#[cfg(all(feature = "semantic", target_os = "macos"))]
fn platform_physical_core_count() -> Option<usize> {
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.physicalcpu".as_ptr(),
            (&mut value as *mut libc::c_int).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value as usize)
}

#[cfg(all(feature = "semantic", target_os = "linux"))]
fn platform_physical_core_count() -> Option<usize> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    parse_linux_physical_core_count(&cpuinfo)
}

#[cfg(all(
    feature = "semantic",
    not(any(target_os = "linux", target_os = "macos"))
))]
fn platform_physical_core_count() -> Option<usize> {
    None
}

#[cfg(all(feature = "semantic", target_os = "linux"))]
fn parse_linux_physical_core_count(cpuinfo: &str) -> Option<usize> {
    let mut physical_cores = HashSet::new();
    let mut processors = 0usize;
    let mut physical_id: Option<String> = None;
    let mut core_id: Option<String> = None;

    for line in cpuinfo.lines().chain(std::iter::once("")) {
        let line = line.trim();
        if line.is_empty() {
            if let (Some(package), Some(core)) = (physical_id.take(), core_id.take()) {
                physical_cores.insert((package, core));
            } else {
                physical_id = None;
                core_id = None;
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "processor" => processors = processors.saturating_add(1),
            "physical id" => physical_id = Some(value.trim().to_string()),
            "core id" => core_id = Some(value.trim().to_string()),
            _ => {}
        }
    }

    if !physical_cores.is_empty() {
        Some(physical_cores.len())
    } else if processors > 0 {
        Some(processors)
    } else {
        None
    }
}

#[cfg(feature = "semantic")]
fn load_local_embedder(home: &Path) -> Result<Option<Arc<FastembedEmbedder>>> {
    let Some(model_files) = semantic_model_file_signatures(home) else {
        return Ok(None);
    };
    let cache_key = semantic_model_cache_path(home);
    if let Ok(cache) = local_embedder_cache().lock() {
        if let Some(entry) = cache.get(&cache_key) {
            if entry.model_files == model_files {
                return Ok(Some(Arc::clone(&entry.embedder)));
            }
        }
    }

    let embedder = Arc::new(load_local_embedder_uncached(home)?);
    if let Ok(mut cache) = local_embedder_cache().lock() {
        cache.insert(
            cache_key,
            CachedLocalEmbedder {
                model_files,
                embedder: Arc::clone(&embedder),
            },
        );
    }
    Ok(Some(embedder))
}

#[cfg(feature = "semantic")]
fn local_embedder_cache() -> &'static Mutex<HashMap<PathBuf, CachedLocalEmbedder>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, CachedLocalEmbedder>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "semantic")]
fn load_local_embedder_uncached(home: &Path) -> Result<FastembedEmbedder> {
    let intra_threads = semantic_intra_threads();
    let cache_path = semantic_model_cache_path(home);
    let tokenizer_files = fastembed::TokenizerFiles {
        tokenizer_file: read_model_file(&cache_path, "tokenizer.json")?,
        config_file: read_model_file(&cache_path, "config.json")?,
        special_tokens_map_file: read_model_file(&cache_path, "special_tokens_map.json")?,
        tokenizer_config_file: read_model_file(&cache_path, "tokenizer_config.json")?,
    };
    let mut model = fastembed::UserDefinedEmbeddingModel::new(
        read_model_file(&cache_path, "onnx/model_q4.onnx")?,
        tokenizer_files,
    )
    .with_external_initializer(
        "model_q4.onnx_data".to_string(),
        read_model_file(&cache_path, "onnx/model_q4.onnx_data")?,
    )
    .with_pooling(fastembed::Pooling::Mean)
    .with_quantization(fastembed::QuantizationMode::None);
    model.output_key = Some(fastembed::OutputKey::ByName("sentence_embedding"));

    let text_embedding = fastembed::TextEmbedding::try_new_from_user_defined(
        model,
        fastembed::InitOptionsUserDefined::new()
            .with_max_length(SEMANTIC_EMBED_MAX_LENGTH)
            .with_intra_threads(intra_threads),
    )
    .map_err(|source| {
        Error::SemanticUnavailable(format!("failed to load local embedding model: {source}"))
    })?;

    Ok(FastembedEmbedder {
        model: std::sync::Mutex::new(text_embedding),
        batch_size: SEMANTIC_EMBED_BATCH_SIZE,
        intra_threads,
    })
}

#[cfg(feature = "semantic")]
fn read_model_file(cache_path: &Path, local: &str) -> Result<Vec<u8>> {
    let path = cache_path.join(local);
    fs::read(&path).map_err(|source| Error::Io { path, source })
}

#[cfg(feature = "semantic")]
pub(crate) fn vector_search_results(
    home: &Path,
    query: &str,
    options: &SearchOptions,
    fetch_limit: usize,
    query_terms: &[String],
    max_snippet_chars: usize,
) -> Result<Vec<RankedSearchResult>> {
    let Some(embedder) = load_local_embedder(home)? else {
        return Err(Error::SemanticUnavailable(
            "local embedding model is not installed".to_string(),
        ));
    };
    let query_vector = embedder.embed_query(query)?;
    let query_blob = vector_to_blob(&query_vector)?;
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    ensure_semantic_vector_schema(&conn, &db_path)?;

    let ctx = VectorQueryContext {
        conn: &conn,
        db_path: &db_path,
        query_blob: &query_blob,
        options,
        query_terms,
        max_snippet_chars,
    };
    let max_vector_k = max_vector_search_k(fetch_limit);
    let mut vector_k = initial_vector_search_k(fetch_limit, options).min(max_vector_k);
    loop {
        let row_limit = vector_search_row_limit(fetch_limit, vector_k);
        let results = vector_search_results_for_k(&ctx, vector_k, row_limit)?;
        let unique = unique_ranked_results_by_event(results);
        if unique.len() >= fetch_limit || vector_k >= max_vector_k {
            return Ok(unique);
        }
        let next_vector_k = vector_k.saturating_mul(2).min(max_vector_k);
        if next_vector_k == vector_k {
            return Ok(unique);
        }
        vector_k = next_vector_k;
    }
}

#[cfg(feature = "semantic")]
/// Loop-invariant inputs to a vector search; only `vector_k`/`row_limit` vary
/// across the adaptive-fetch retries, so the rest travel together as context.
#[cfg(feature = "semantic")]
#[derive(Clone, Copy)]
struct VectorQueryContext<'a> {
    conn: &'a Connection,
    db_path: &'a Path,
    query_blob: &'a [u8],
    options: &'a SearchOptions,
    query_terms: &'a [String],
    max_snippet_chars: usize,
}

#[cfg(feature = "semantic")]
fn vector_search_results_for_k(
    ctx: &VectorQueryContext,
    vector_k: usize,
    row_limit: usize,
) -> Result<Vec<RankedSearchResult>> {
    let VectorQueryContext {
        conn,
        db_path,
        query_blob,
        options,
        query_terms,
        max_snippet_chars,
    } = *ctx;
    let mut sql = String::from(
        "SELECT
           e.id,
           e.tool,
           e.session_id,
           e.canonical_type,
           e.captured_at,
           ve.distance,
           e.searchable_text,
           e.raw_file,
           e.raw_line,
           e.raw_offset,
           e.compaction_state,
           e.cwd,
           e.project_root
         FROM vector_unit_embeddings ve
         JOIN vector_units vu ON vu.id = ve.unit_id
         JOIN events e ON e.id = vu.event_id
         WHERE ve.embedding MATCH ? AND ve.k = ?",
    );
    let mut params = vec![
        SqlValue::Blob(query_blob.to_vec()),
        SqlValue::Integer(vector_k as i64),
    ];

    if let Some(tool) = options.tool {
        sql.push_str(" AND e.tool = ?");
        params.push(SqlValue::Text(tool.as_str().to_string()));
    }
    if let Some(session_id) = options.session_id.as_deref() {
        sql.push_str(" AND e.session_id = ?");
        params.push(SqlValue::Text(session_id.to_string()));
    }
    if let Some(cwd) = options.cwd.as_deref() {
        sql.push_str(" AND e.cwd = ?");
        params.push(SqlValue::Text(cwd.to_string()));
    }
    if let Some(since) = options.since.as_deref() {
        sql.push_str(" AND e.captured_at >= ?");
        params.push(SqlValue::Text(normalize_date_or_duration(since, "since")?));
    }
    if let Some(canonical_type) = options.canonical_type.as_deref() {
        sql.push_str(" AND e.canonical_type = ?");
        params.push(SqlValue::Text(canonical_type.to_string()));
    }
    if !options.include_deltas {
        sql.push_str(" AND e.canonical_type != 'assistant.delta'");
    }
    if let Some(file) = options.file.as_deref() {
        sql.push_str(
            " AND EXISTS (
                SELECT 1
                FROM event_files ef
                JOIN files f ON f.id = ef.file_id
                WHERE ef.event_id = e.id
                  AND (f.path = ? OR f.path LIKE ?)
              )",
        );
        params.push(SqlValue::Text(file.to_string()));
        params.push(SqlValue::Text(format!("%{file}%")));
    }
    if let Some(command) = options.command.as_deref() {
        sql.push_str(
            " AND EXISTS (
                SELECT 1
                FROM tool_events te
                WHERE te.event_id = e.id
                  AND te.command LIKE ?
              )",
        );
        params.push(SqlValue::Text(format!("%{command}%")));
    }
    sql.push_str(" ORDER BY ve.distance LIMIT ?");
    params.push(SqlValue::Integer(row_limit as i64));

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(1)?;
            let searchable_text = row.get::<_, String>(6).unwrap_or_default();
            let distance = row.get::<_, f64>(5)?;
            Ok(RankedSearchResult {
                event_id: row.get(0)?,
                result: SearchResult {
                    tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                    session_id: row.get(2)?,
                    canonical_type: row.get(3)?,
                    timestamp: row.get(4)?,
                    score: 1.0 / (1.0 + distance),
                    snippet: match_centered_snippet(
                        None,
                        searchable_text.clone(),
                        query_terms,
                        max_snippet_chars,
                    ),
                    raw_file: row.get(7)?,
                    raw_line: row.get(8)?,
                    raw_offset: row.get(9)?,
                    compaction_state: row.get(10)?,
                    payload: Value::Null,
                    also_at: Vec::new(),
                    corroboration: None,
                    retrieval_key: sha256_hex(searchable_text.as_bytes()),
                    corroboration_text: searchable_text,
                    cwd: row.get(11)?,
                    project_root: row.get(12)?,
                },
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row.map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?);
    }
    Ok(results)
}

#[cfg(not(feature = "semantic"))]
pub(crate) fn vector_search_results(
    _home: &Path,
    _query: &str,
    _options: &SearchOptions,
    _fetch_limit: usize,
    _query_terms: &[String],
    _max_snippet_chars: usize,
) -> Result<Vec<RankedSearchResult>> {
    Err(Error::SemanticUnavailable(
        "semantic backend is not available in this build; rebuild with --features semantic"
            .to_string(),
    ))
}

#[cfg(feature = "semantic")]
fn max_vector_search_k(fetch_limit: usize) -> usize {
    fetch_limit
        .clamp(1, MAX_SEARCH_LIMIT * 20)
        .saturating_mul(4)
        .max(1)
}

#[cfg(feature = "semantic")]
fn initial_vector_search_k(fetch_limit: usize, options: &SearchOptions) -> usize {
    let multiplier = if vector_search_filter_count(options) == 0 {
        2
    } else {
        4
    };
    fetch_limit
        .clamp(1, MAX_SEARCH_LIMIT * 20)
        .saturating_mul(multiplier)
        .max(1)
}

#[cfg(feature = "semantic")]
fn vector_search_row_limit(fetch_limit: usize, vector_k: usize) -> usize {
    let vector_k = vector_k.max(1);
    fetch_limit.saturating_mul(2).max(1).min(vector_k)
}

#[cfg(feature = "semantic")]
fn vector_search_filter_count(options: &SearchOptions) -> usize {
    [
        options.tool.is_some(),
        options.session_id.is_some(),
        options.cwd.is_some(),
        options.since.is_some(),
        options.canonical_type.is_some(),
        options.file.is_some(),
        options.command.is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count()
}

#[cfg(feature = "semantic")]
pub(crate) fn insert_vector_unit_rows(
    conn: &Connection,
    path: &Path,
    event_id: i64,
    envelope: &EventEnvelope,
    raw_line: i64,
    raw_offset: i64,
    search_document: &SearchDocument,
) -> Result<()> {
    let created_at = OffsetDateTime::now_utc().format(&Rfc3339)?;
    for unit in embedding_units_for_document(search_document) {
        insert_vector_unit_text(conn, path, &unit, &created_at)?;
        conn.execute(
            "INSERT OR IGNORE INTO vector_units(
               event_id,
               tool,
               session_id,
               unit_kind,
               unit_index,
               text_hash,
               raw_file,
               raw_line,
               raw_offset,
               created_at
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                event_id,
                envelope.tool.as_str(),
                &envelope.session_id,
                unit.kind.as_str(),
                unit.unit_index as i64,
                &unit.text_hash,
                path.display().to_string(),
                raw_line,
                raw_offset,
                &created_at,
            ],
        )
        .map_err(|source| Error::Sqlite {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn insert_vector_unit_text(
    conn: &Connection,
    path: &Path,
    unit: &EmbeddingUnit,
    created_at: &str,
) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO vector_unit_texts(text_hash, text, created_at)
         VALUES (?1, ?2, ?3)",
        params![&unit.text_hash, &unit.text, created_at],
    )
    .map_err(|source| Error::Sqlite {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(not(feature = "semantic"))]
pub(crate) fn insert_vector_unit_rows(
    _conn: &Connection,
    _path: &Path,
    _event_id: i64,
    _envelope: &EventEnvelope,
    _raw_line: i64,
    _raw_offset: i64,
    _search_document: &SearchDocument,
) -> Result<()> {
    Ok(())
}

#[cfg(feature = "semantic")]
pub(crate) fn embed_index_if_available_with_progress<F>(
    home: &Path,
    mut progress: F,
) -> Result<usize>
where
    F: FnMut(EmbeddingIndexProgress),
{
    if !semantic_model_files_present(home) {
        return Ok(0);
    }
    let db_path = home.join("index").join("harness.db");
    let mut conn = open_index(&db_path)?;
    ensure_semantic_vector_schema(&conn, &db_path)?;
    sync_vector_units(&conn, &db_path)?;
    let total_units = count_unembedded_units(&conn, &db_path)?;
    if total_units == 0 {
        return Ok(0);
    }

    let planned_threads = semantic_intra_threads();
    progress(embedding_index_plan_progress(
        total_units,
        SEMANTIC_EMBED_BATCH_SIZE,
        SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        planned_threads,
    ));
    progress(embedding_index_phase_progress(
        "loading_model",
        "started",
        SEMANTIC_EMBED_BATCH_SIZE,
        SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        planned_threads,
    ));
    let Some(embedder) = load_local_embedder(home)? else {
        return Ok(0);
    };
    progress(embedding_index_phase_progress(
        "loading_model",
        "completed",
        embedder.document_batch_size(),
        SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        embedder.intra_threads(),
    ));
    embed_unembedded_units_paged_with_config(
        &mut conn,
        &db_path,
        &*embedder,
        total_units,
        EmbeddingWriteConfig::default(),
        progress,
    )
}

#[cfg(not(feature = "semantic"))]
pub(crate) fn embed_index_if_available_with_progress<F>(_home: &Path, _progress: F) -> Result<usize>
where
    F: FnMut(EmbeddingIndexProgress),
{
    Ok(0)
}

#[cfg(feature = "semantic")]
fn sync_vector_units(conn: &Connection, db_path: &Path) -> Result<usize> {
    let mut statement = conn
        .prepare(
            "SELECT id, payload_json, tool, session_id, canonical_type, raw_file, raw_line, raw_offset
             FROM events
             WHERE NOT EXISTS (
               SELECT 1 FROM vector_units vu WHERE vu.event_id = events.id
             )
             ORDER BY id",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, Option<i64>>(7)?,
            ))
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;

    let mut inserted = 0usize;
    for row in rows {
        let (
            event_id,
            payload_json,
            tool,
            session_id,
            canonical_type,
            raw_file,
            raw_line,
            raw_offset,
        ) = row.map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let canonical_type = CanonicalType::from_str(&canonical_type)?;
        let payload = match payload_json.as_deref() {
            Some(payload_json) => serde_json::from_str(payload_json)?,
            None => payload_for_raw_pointer(&raw_file, raw_line, raw_offset)?,
        };
        let document = search_document_for_event(canonical_type, &payload);
        let created_at = OffsetDateTime::now_utc().format(&Rfc3339)?;
        for unit in embedding_units_for_document(&document) {
            insert_vector_unit_text(conn, db_path, &unit, &created_at)?;
            let changed = conn
                .execute(
                    "INSERT OR IGNORE INTO vector_units(
                       event_id,
                       tool,
                       session_id,
                       unit_kind,
                       unit_index,
                       text_hash,
                       raw_file,
                       raw_line,
                       raw_offset,
                       created_at
                     )
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        event_id,
                        &tool,
                        &session_id,
                        unit.kind.as_str(),
                        unit.unit_index as i64,
                        &unit.text_hash,
                        &raw_file,
                        raw_line,
                        raw_offset,
                        &created_at,
                    ],
                )
                .map_err(|source| Error::Sqlite {
                    path: db_path.to_path_buf(),
                    source,
                })?;
            inserted = inserted.saturating_add(changed);
        }
    }
    Ok(inserted)
}

#[cfg(feature = "semantic")]
fn embed_unembedded_units_paged_with_config(
    conn: &mut Connection,
    db_path: &Path,
    embedder: &dyn Embedder,
    total_units: usize,
    config: EmbeddingWriteConfig,
    mut progress: impl FnMut(EmbeddingIndexProgress),
) -> Result<usize> {
    let mut embedded = 0usize;
    let mut pending_writes = Vec::with_capacity(embedder.document_batch_size());
    let started = Instant::now();
    let mut last_emit = started;
    let mut after_unit_id = 0i64;

    progress(embedding_index_progress(
        "embedding",
        "started",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));

    loop {
        let mut page = collect_unembedded_units_page(
            conn,
            db_path,
            after_unit_id,
            SEMANTIC_EMBED_COLLECT_BATCH_SIZE,
        )?;
        if page.rows_seen == 0 {
            break;
        }
        after_unit_id = page.last_unit_id;
        bucket_unembedded_units(&mut page.units);

        for batch in page.units.chunks(embedder.document_batch_size()) {
            let texts = batch
                .iter()
                .map(|unit| unit.text.clone())
                .collect::<Vec<_>>();
            let vectors = embedder.embed_documents(&texts)?;
            for (unit, vector) in batch.iter().zip(vectors) {
                pending_writes.push((unit.unit_id, vector));
                embedded += 1;
                if pending_writes.len() >= config.write_chunk_size {
                    flush_embedding_writes(conn, db_path, &pending_writes)?;
                    pending_writes.clear();
                }
            }
            if embedded < total_units && last_emit.elapsed() >= SEMANTIC_EMBED_PROGRESS_INTERVAL {
                progress(embedding_index_progress(
                    "embedding",
                    "running",
                    embedded,
                    total_units,
                    started,
                    embedder,
                    config.write_chunk_size,
                ));
                last_emit = Instant::now();
            }
        }
    }

    if !pending_writes.is_empty() {
        flush_embedding_writes(conn, db_path, &pending_writes)?;
    }
    progress(embedding_index_progress(
        "embedding",
        "completed",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));
    Ok(embedded)
}

#[cfg(feature = "semantic")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn embed_unembedded_units_with_config(
    conn: &mut Connection,
    db_path: &Path,
    embedder: &dyn Embedder,
    config: EmbeddingWriteConfig,
    progress: impl FnMut(EmbeddingIndexProgress),
) -> Result<usize> {
    let units = collect_unembedded_units(conn, db_path)?;
    embed_collected_unembedded_units_with_config(conn, db_path, embedder, units, config, progress)
}

#[cfg(feature = "semantic")]
fn embed_collected_unembedded_units_with_config(
    conn: &mut Connection,
    db_path: &Path,
    embedder: &dyn Embedder,
    mut units: Vec<UnembeddedUnit>,
    config: EmbeddingWriteConfig,
    mut progress: impl FnMut(EmbeddingIndexProgress),
) -> Result<usize> {
    bucket_unembedded_units(&mut units);
    let total_units = units.len();
    let mut embedded = 0usize;
    let mut pending_writes = Vec::with_capacity(embedder.document_batch_size());
    let started = Instant::now();
    let mut last_emit = started;
    progress(embedding_index_progress(
        "embedding",
        "started",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));
    for batch in units.chunks(embedder.document_batch_size()) {
        let texts = batch
            .iter()
            .map(|unit| unit.text.clone())
            .collect::<Vec<_>>();
        let vectors = embedder.embed_documents(&texts)?;
        for (unit, vector) in batch.iter().zip(vectors) {
            pending_writes.push((unit.unit_id, vector));
            embedded += 1;
            if pending_writes.len() >= config.write_chunk_size {
                flush_embedding_writes(conn, db_path, &pending_writes)?;
                pending_writes.clear();
            }
        }
        if embedded < total_units && last_emit.elapsed() >= SEMANTIC_EMBED_PROGRESS_INTERVAL {
            progress(embedding_index_progress(
                "embedding",
                "running",
                embedded,
                total_units,
                started,
                embedder,
                config.write_chunk_size,
            ));
            last_emit = Instant::now();
        }
    }
    if !pending_writes.is_empty() {
        flush_embedding_writes(conn, db_path, &pending_writes)?;
    }
    progress(embedding_index_progress(
        "embedding",
        "completed",
        embedded,
        total_units,
        started,
        embedder,
        config.write_chunk_size,
    ));
    Ok(embedded)
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct EmbeddingWriteConfig {
    pub(crate) write_chunk_size: usize,
}

#[cfg(feature = "semantic")]
impl Default for EmbeddingWriteConfig {
    fn default() -> Self {
        Self {
            write_chunk_size: SEMANTIC_EMBED_WRITE_CHUNK_SIZE,
        }
    }
}

#[cfg(feature = "semantic")]
fn flush_embedding_writes(
    conn: &mut Connection,
    db_path: &Path,
    rows: &[(i64, Vec<f32>)],
) -> Result<()> {
    let tx = conn.transaction().map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })?;
    {
        let mut statement = tx
            .prepare(
                "INSERT OR REPLACE INTO vector_unit_embeddings(unit_id, embedding)
                 VALUES (?1, ?2)",
            )
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;
        for (unit_id, vector) in rows {
            statement
                .execute(params![unit_id, vector_to_blob(vector)?])
                .map_err(|source| Error::Sqlite {
                    path: db_path.to_path_buf(),
                    source,
                })?;
        }
    }
    tx.commit().map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(feature = "semantic")]
pub(crate) fn embedding_index_progress(
    phase: &str,
    status: &str,
    embedded_units: usize,
    total_units: usize,
    started: Instant,
    embedder: &dyn Embedder,
    write_chunk_size: usize,
) -> EmbeddingIndexProgress {
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let units_per_second = if embedded_units == 0 || elapsed_seconds <= f64::EPSILON {
        0.0
    } else {
        embedded_units as f64 / elapsed_seconds
    };
    let eta_seconds = if units_per_second > 0.0 && embedded_units < total_units {
        Some(((total_units - embedded_units) as f64 / units_per_second).ceil() as u64)
    } else {
        None
    };
    EmbeddingIndexProgress {
        phase: phase.to_string(),
        status: status.to_string(),
        embedded_units,
        total_units,
        units_per_second,
        eta_seconds,
        batch_size: embedder.document_batch_size(),
        write_chunk_size,
        intra_threads: embedder.intra_threads(),
    }
}

#[cfg(feature = "semantic")]
fn embedding_index_phase_progress(
    phase: &str,
    status: &str,
    batch_size: usize,
    write_chunk_size: usize,
    intra_threads: usize,
) -> EmbeddingIndexProgress {
    EmbeddingIndexProgress {
        phase: phase.to_string(),
        status: status.to_string(),
        embedded_units: 0,
        total_units: 0,
        units_per_second: 0.0,
        eta_seconds: None,
        batch_size,
        write_chunk_size,
        intra_threads,
    }
}

#[cfg(feature = "semantic")]
fn embedding_index_plan_progress(
    total_units: usize,
    batch_size: usize,
    write_chunk_size: usize,
    intra_threads: usize,
) -> EmbeddingIndexProgress {
    EmbeddingIndexProgress {
        phase: "embedding_plan".to_string(),
        status: "ready".to_string(),
        embedded_units: 0,
        total_units,
        units_per_second: 0.0,
        eta_seconds: None,
        batch_size,
        write_chunk_size,
        intra_threads,
    }
}

#[cfg(feature = "semantic")]
#[derive(Debug, Clone)]
pub(crate) struct UnembeddedUnit {
    pub(crate) unit_id: i64,
    pub(crate) text: String,
    pub(crate) estimated_tokens: usize,
}

#[cfg(feature = "semantic")]
struct UnembeddedUnitPage {
    units: Vec<UnembeddedUnit>,
    last_unit_id: i64,
    rows_seen: usize,
}

#[cfg(feature = "semantic")]
pub(crate) fn bucket_unembedded_units(units: &mut [UnembeddedUnit]) {
    units.sort_by_key(|unit| (embedding_length_bucket(unit.estimated_tokens), unit.unit_id));
}

#[cfg(feature = "semantic")]
fn embedding_length_bucket(tokens: usize) -> usize {
    match tokens {
        0..=64 => 64,
        65..=128 => 128,
        129..=256 => 256,
        257..=512 => 512,
        513..=1024 => 1024,
        _ => SEMANTIC_EMBED_MAX_LENGTH,
    }
}

#[cfg(feature = "semantic")]
pub(crate) fn estimated_embedding_token_count(text: &str) -> usize {
    let by_words = text.split_whitespace().count();
    let by_chars = text.chars().count().div_ceil(4);
    by_words.max(by_chars).min(SEMANTIC_EMBED_MAX_LENGTH)
}

#[cfg(feature = "semantic")]
fn count_unembedded_units(conn: &Connection, db_path: &Path) -> Result<usize> {
    let count = conn
        .query_row(
            "SELECT COUNT(*)
             FROM vector_units vu
             LEFT JOIN vector_unit_embeddings ve ON ve.unit_id = vu.id
             WHERE ve.unit_id IS NULL",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    Ok(count.max(0) as usize)
}

#[cfg(feature = "semantic")]
pub(crate) fn collect_unembedded_units(
    conn: &Connection,
    db_path: &Path,
) -> Result<Vec<UnembeddedUnit>> {
    let mut units = Vec::new();
    let mut after_unit_id = 0i64;
    loop {
        let page = collect_unembedded_units_page(
            conn,
            db_path,
            after_unit_id,
            SEMANTIC_EMBED_COLLECT_BATCH_SIZE,
        )?;
        if page.rows_seen == 0 {
            break;
        }
        after_unit_id = page.last_unit_id;
        units.extend(page.units);
    }
    Ok(units)
}

#[cfg(feature = "semantic")]
fn collect_unembedded_units_page(
    conn: &Connection,
    db_path: &Path,
    after_unit_id: i64,
    limit: usize,
) -> Result<UnembeddedUnitPage> {
    let mut statement = conn
        .prepare(
            "SELECT
               vu.id,
               vu.unit_kind,
               vu.unit_index,
               vu.text_hash,
               vut.text,
               e.canonical_type,
               e.payload_json,
               e.raw_file,
               e.raw_line,
               e.raw_offset
             FROM vector_units vu
             JOIN events e ON e.id = vu.event_id
             LEFT JOIN vector_unit_texts vut ON vut.text_hash = vu.text_hash
             LEFT JOIN vector_unit_embeddings ve ON ve.unit_id = vu.id
             WHERE ve.unit_id IS NULL
               AND vu.id > ?1
             ORDER BY vu.id
             LIMIT ?2",
        )
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
    let mut rows = statement
        .query(params![
            after_unit_id,
            limit.clamp(1, SEMANTIC_EMBED_COLLECT_BATCH_SIZE) as i64
        ])
        .map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;

    let mut units = Vec::new();
    let mut rows_seen = 0usize;
    let mut last_unit_id = after_unit_id;
    while let Some(row) = rows.next().map_err(|source| Error::Sqlite {
        path: db_path.to_path_buf(),
        source,
    })? {
        rows_seen += 1;
        let unit_id = row.get::<_, i64>(0).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        last_unit_id = unit_id;
        let unit_kind = row.get::<_, String>(1).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let unit_index = row.get::<_, i64>(2).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let text_hash = row.get::<_, String>(3).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let stored_text = row
            .get::<_, Option<String>>(4)
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;
        let canonical_type = row.get::<_, String>(5).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let payload_json = row
            .get::<_, Option<String>>(6)
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;
        let raw_file = row.get::<_, String>(7).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let raw_line = row.get::<_, i64>(8).map_err(|source| Error::Sqlite {
            path: db_path.to_path_buf(),
            source,
        })?;
        let raw_offset = row
            .get::<_, Option<i64>>(9)
            .map_err(|source| Error::Sqlite {
                path: db_path.to_path_buf(),
                source,
            })?;

        if let Some(text) = stored_text {
            units.push(UnembeddedUnit {
                unit_id,
                estimated_tokens: estimated_embedding_token_count(&text),
                text,
            });
            continue;
        }

        let canonical_type = CanonicalType::from_str(&canonical_type)?;
        let payload = match payload_json.as_deref() {
            Some(payload_json) => serde_json::from_str(payload_json)?,
            None => payload_for_raw_pointer(&raw_file, raw_line, raw_offset)?,
        };
        let unit_kind = EmbeddingUnitKind::from_str(&unit_kind)?;
        let unit_index = usize::try_from(unit_index)
            .map_err(|_| Error::Validation(format!("negative vector unit index: {unit_index}")))?;
        let document = search_document_for_event(canonical_type, &payload);
        if let Some(unit) = embedding_units_for_document(&document)
            .into_iter()
            .find(|unit| {
                unit.kind == unit_kind
                    && unit.unit_index == unit_index
                    && unit.text_hash == text_hash
            })
        {
            let created_at = OffsetDateTime::now_utc().format(&Rfc3339)?;
            insert_vector_unit_text(conn, db_path, &unit, &created_at)?;
            units.push(UnembeddedUnit {
                unit_id,
                estimated_tokens: estimated_embedding_token_count(&unit.text),
                text: unit.text,
            });
        }
    }
    Ok(UnembeddedUnitPage {
        units,
        last_unit_id,
        rows_seen,
    })
}

#[cfg(feature = "semantic")]
pub(crate) fn vector_to_blob(vector: &[f32]) -> Result<Vec<u8>> {
    if vector.len() != SEMANTIC_VECTOR_DIMENSIONS {
        return Err(Error::SemanticUnavailable(format!(
            "vector has {} dimensions, expected {}",
            vector.len(),
            SEMANTIC_VECTOR_DIMENSIONS
        )));
    }
    let mut blob = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    Ok(blob)
}
