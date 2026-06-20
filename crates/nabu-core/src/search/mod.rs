//! History search: lexical FTS, hybrid/auto dispatch, reciprocal-rank fusion,
//! snippet/term helpers, and result hydration. The git corroboration engine is
//! the `corroborate` submodule; the feature-gated vector-read path stays in
//! lib.rs (the semantic module, final phase) and is reached via `crate::`.

pub(crate) mod corroborate;

use crate::{
    normalize_date_or_duration, open_index, open_raw_offset_reader, raw_envelope_for_line_scan,
    read_raw_envelope_at_offset, resolved_payload_for_envelope, semantic_search_available,
    sha256_hex, vector_search_results, Error, RankedSearchResult, Result, SearchContinuation,
    SearchMode, SearchOptions, SearchPage, SearchResult, Tool, MAX_SEARCH_LIMIT,
    MAX_SEARCH_SNIPPET_CHARS,
};
pub(crate) use corroborate::corroborate_text;
use rusqlite::params_from_iter;
use rusqlite::types::Value as SqlValue;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub fn search_history(home: &Path, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    search_history_filtered(
        home,
        query,
        SearchOptions {
            limit,
            ..SearchOptions::default()
        },
    )
}

pub fn search_history_filtered(
    home: &Path,
    query: &str,
    options: SearchOptions,
) -> Result<Vec<SearchResult>> {
    Ok(search_history_page(home, query, options)?.results)
}

pub fn search_history_page(home: &Path, query: &str, options: SearchOptions) -> Result<SearchPage> {
    if query.trim().is_empty() {
        return Err(Error::Validation("query must not be empty".to_string()));
    }
    let mode_requested = options.mode;
    let semantic_available = semantic_search_available(home);
    let mut mode_applied = match mode_requested {
        SearchMode::Auto if semantic_available => SearchMode::Hybrid,
        SearchMode::Auto => SearchMode::Lexical,
        SearchMode::Lexical => SearchMode::Lexical,
        SearchMode::Hybrid if semantic_available => SearchMode::Hybrid,
        SearchMode::Hybrid => {
            return Err(Error::SemanticUnavailable(
                "local embedding model and vector index are not available; run lexical mode or install the semantic model explicitly".to_string(),
            ))
        }
    };
    if mode_applied == SearchMode::Hybrid {
        match search_history_hybrid_page(home, query, options.clone(), semantic_available) {
            Ok(page) => return Ok(page),
            Err(Error::SemanticUnavailable(_)) if mode_requested == SearchMode::Auto => {
                mode_applied = SearchMode::Lexical;
            }
            Err(error) => return Err(error),
        }
    }
    let query_terms = searchable_terms(query)?;
    let fts_query = quoted_fts_terms(&query_terms);
    let limit = options.limit.clamp(1, MAX_SEARCH_LIMIT);
    let offset = options.offset;
    let max_snippet_chars = options.max_snippet_chars.clamp(1, MAX_SEARCH_SNIPPET_CHARS);
    let raw_fetch_limit = search_overfetch_limit(offset, limit);
    let (mut results, has_more_raw_rows) = lexical_search_ranked_results(
        home,
        &options,
        &query_terms,
        &fts_query,
        raw_fetch_limit,
        max_snippet_chars,
    )?;
    if options.dedupe {
        results = dedupe_ranked_search_results(results)?;
    }

    let total_estimated = if has_more_raw_rows {
        None
    } else {
        Some(results.len())
    };
    let has_more_logical_rows = results.len() > offset.saturating_add(limit);
    let mut page_results = results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|ranked| ranked.result)
        .collect::<Vec<_>>();
    if options.include_payload {
        hydrate_search_result_payloads(&mut page_results)?;
    }
    if options.corroborate {
        annotate_search_results_with_corroboration(&mut page_results);
    }
    let returned = page_results.len();
    let continuation = if returned > 0 && (has_more_raw_rows || has_more_logical_rows) {
        Some(SearchContinuation {
            next_offset: offset.saturating_add(returned),
        })
    } else {
        None
    };

    Ok(SearchPage {
        results: page_results,
        truncated: continuation.is_some(),
        returned,
        total_estimated,
        continuation,
        mode_requested,
        mode_applied,
        semantic_available,
        limit_applied: limit,
        offset_applied: offset,
        max_snippet_chars_applied: max_snippet_chars,
        include_payload: options.include_payload,
        include_deltas: options.include_deltas,
        dedupe: options.dedupe,
    })
}

fn lexical_search_ranked_results(
    home: &Path,
    options: &SearchOptions,
    query_terms: &[String],
    fts_query: &str,
    fetch_limit: usize,
    max_snippet_chars: usize,
) -> Result<(Vec<RankedSearchResult>, bool)> {
    let db_path = home.join("index").join("harness.db");
    let conn = open_index(&db_path)?;
    let mut sql = String::from(
        "SELECT
           e.id,
           e.tool,
           e.session_id,
           e.canonical_type,
           e.captured_at,
           -bm25(events_fts, 8.0, 6.0, 4.0, 1.0, 0.5) AS score,
           NULL AS snippet,
           e.searchable_text,
           e.raw_file,
           e.raw_line,
           e.raw_offset,
           e.compaction_state,
           e.cwd,
           e.project_root
         FROM events_fts
         JOIN events e ON e.id = events_fts.rowid
         WHERE events_fts MATCH ?",
    );
    let mut params = vec![SqlValue::Text(fts_query.to_string())];

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
    sql.push_str(
        " ORDER BY bm25(events_fts, 8.0, 6.0, 4.0, 1.0, 0.5), e.captured_at DESC, e.raw_line ASC
          LIMIT ?",
    );
    params.push(SqlValue::Integer(fetch_limit.saturating_add(1) as i64));

    let mut statement = conn.prepare(&sql).map_err(|source| Error::Sqlite {
        path: db_path.clone(),
        source,
    })?;
    let rows = statement
        .query_map(params_from_iter(params), |row| {
            let tool_text: String = row.get(1)?;
            let searchable_text = row.get::<_, String>(7).unwrap_or_default();
            Ok(RankedSearchResult {
                event_id: row.get(0)?,
                result: SearchResult {
                    tool: Tool::from_str(&tool_text).map_err(|_| rusqlite::Error::InvalidQuery)?,
                    session_id: row.get(2)?,
                    canonical_type: row.get(3)?,
                    timestamp: row.get(4)?,
                    score: row.get(5)?,
                    snippet: match_centered_snippet(
                        row.get::<_, Option<String>>(6)?,
                        searchable_text.clone(),
                        query_terms,
                        max_snippet_chars,
                    ),
                    raw_file: row.get(8)?,
                    raw_line: row.get(9)?,
                    raw_offset: row.get(10)?,
                    compaction_state: row.get(11)?,
                    payload: Value::Null,
                    also_at: Vec::new(),
                    corroboration: None,
                    retrieval_key: sha256_hex(searchable_text.as_bytes()),
                    corroboration_text: searchable_text,
                    cwd: row.get(12)?,
                    project_root: row.get(13)?,
                },
            })
        })
        .map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row.map_err(|source| Error::Sqlite {
            path: db_path.clone(),
            source,
        })?);
    }
    let has_more_raw_rows = results.len() > fetch_limit;
    if has_more_raw_rows {
        results.truncate(fetch_limit);
    }
    Ok((results, has_more_raw_rows))
}

fn search_history_hybrid_page(
    home: &Path,
    query: &str,
    options: SearchOptions,
    semantic_available: bool,
) -> Result<SearchPage> {
    let query_terms = searchable_terms(query)?;
    let limit = options.limit.clamp(1, MAX_SEARCH_LIMIT);
    let offset = options.offset;
    let max_snippet_chars = options.max_snippet_chars.clamp(1, MAX_SEARCH_SNIPPET_CHARS);
    let raw_fetch_limit = search_overfetch_limit(offset, limit);
    let fts_query = quoted_fts_terms(&query_terms);

    let (lexical_results, _) = lexical_search_ranked_results(
        home,
        &options,
        &query_terms,
        &fts_query,
        raw_fetch_limit,
        max_snippet_chars,
    )?;
    let vector_results = vector_search_results(
        home,
        query,
        &options,
        raw_fetch_limit,
        &query_terms,
        max_snippet_chars,
    )?;
    let mut results = reciprocal_rank_fuse(lexical_results, vector_results);

    if options.dedupe {
        results = dedupe_ranked_search_results(results)?;
    }
    let total_estimated = Some(results.len());
    let has_more_logical_rows = results.len() > offset.saturating_add(limit);
    let mut page_results = results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|ranked| ranked.result)
        .collect::<Vec<_>>();
    if options.include_payload {
        hydrate_search_result_payloads(&mut page_results)?;
    }
    let returned = page_results.len();
    let continuation = if returned > 0 && has_more_logical_rows {
        Some(SearchContinuation {
            next_offset: offset.saturating_add(returned),
        })
    } else {
        None
    };

    Ok(SearchPage {
        results: page_results,
        truncated: continuation.is_some(),
        returned,
        total_estimated,
        continuation,
        mode_requested: options.mode,
        mode_applied: SearchMode::Hybrid,
        semantic_available,
        limit_applied: limit,
        offset_applied: offset,
        max_snippet_chars_applied: max_snippet_chars,
        include_payload: options.include_payload,
        include_deltas: options.include_deltas,
        dedupe: options.dedupe,
    })
}

fn reciprocal_rank_fuse(
    lexical_results: Vec<RankedSearchResult>,
    vector_results: Vec<RankedSearchResult>,
) -> Vec<RankedSearchResult> {
    const RRF_K: f64 = 60.0;
    let lexical_results = unique_ranked_results_by_event(lexical_results);
    let vector_results = unique_ranked_results_by_event(vector_results);
    let mut fused: HashMap<i64, (RankedSearchResult, f64)> = HashMap::new();

    for (rank, result) in lexical_results.into_iter().enumerate() {
        let key = result.event_id;
        let entry = fused.entry(key).or_insert((result, 0.0));
        entry.1 += 1.0 / (RRF_K + rank as f64 + 1.0);
    }
    for (rank, result) in vector_results.into_iter().enumerate() {
        let key = result.event_id;
        let entry = fused.entry(key).or_insert((result, 0.0));
        entry.1 += 1.0 / (RRF_K + rank as f64 + 1.0);
    }

    let mut results = fused
        .into_values()
        .map(|(mut result, score)| {
            result.result.score = score;
            result
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .result
            .score
            .total_cmp(&left.result.score)
            .then_with(|| right.result.timestamp.cmp(&left.result.timestamp))
            .then_with(|| left.result.raw_line.cmp(&right.result.raw_line))
    });
    results
}

pub(crate) fn unique_ranked_results_by_event(
    results: Vec<RankedSearchResult>,
) -> Vec<RankedSearchResult> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for result in results {
        if seen.insert(result.event_id) {
            unique.push(result);
        }
    }
    unique
}

fn annotate_search_results_with_corroboration(results: &mut [SearchResult]) {
    for result in results {
        result.corroboration = Some(corroborate_text(
            result.cwd.as_deref(),
            result.project_root.as_deref(),
            &result.corroboration_text,
        ));
    }
}

fn search_overfetch_limit(offset: usize, limit: usize) -> usize {
    let requested_window = offset.saturating_add(limit);
    let extra = requested_window.min(500).max(limit);
    requested_window.saturating_add(extra)
}

fn bounded_snippet(snippet: String, max_chars: usize) -> String {
    truncate_chars(snippet.trim().to_string(), max_chars)
}

pub(crate) fn match_centered_snippet(
    sqlite_snippet: Option<String>,
    searchable_text: String,
    query_terms: &[String],
    max_chars: usize,
) -> String {
    if let Some(snippet) = sqlite_snippet.filter(|snippet| !snippet.trim().is_empty()) {
        return bounded_snippet(snippet, max_chars);
    }
    if searchable_text.chars().count() <= max_chars {
        return searchable_text.trim().to_string();
    }
    let lower_text = searchable_text.to_lowercase();
    let first_match = query_terms
        .iter()
        .filter_map(|term| lower_text.find(&term.to_lowercase()))
        .min()
        .unwrap_or(0);
    let half_window = max_chars.saturating_div(2);
    let mut start = first_match.saturating_sub(half_window);
    while start > 0 && !searchable_text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = start.saturating_add(max_chars).min(searchable_text.len());
    while end > start && !searchable_text.is_char_boundary(end) {
        end -= 1;
    }
    searchable_text[start..end].trim().to_string()
}

fn truncate_chars(mut value: String, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value;
    }
    let mut cutoff = 0usize;
    for (count, (index, character)) in value.char_indices().enumerate() {
        if count == max_chars {
            break;
        }
        cutoff = index + character.len_utf8();
    }
    value.truncate(cutoff);
    value
}

fn dedupe_ranked_search_results(
    results: Vec<RankedSearchResult>,
) -> Result<Vec<RankedSearchResult>> {
    let mut seen: HashMap<(String, String, String), usize> = HashMap::new();
    let mut deduped: Vec<RankedSearchResult> = Vec::new();
    for result in results {
        let key = retrieval_twin_key(&result.result);
        if let Some(existing) = seen.get(&key).copied() {
            deduped[existing]
                .result
                .also_at
                .push(result.result.raw_line);
        } else {
            seen.insert(key, deduped.len());
            deduped.push(result);
        }
    }
    Ok(deduped)
}

fn retrieval_twin_key(result: &SearchResult) -> (String, String, String) {
    (
        result.session_id.clone(),
        result.canonical_type.clone(),
        result.retrieval_key.clone(),
    )
}

fn searchable_terms(query: &str) -> Result<Vec<String>> {
    let mut terms = Vec::new();
    let mut current = String::new();

    for character in query.chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character);
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        terms.push(current);
    }

    if terms.is_empty() {
        return Err(Error::Validation(
            "query must contain searchable text".to_string(),
        ));
    }

    Ok(terms)
}

fn quoted_fts_terms(terms: &[String]) -> String {
    terms
        .iter()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" AND ")
}

fn hydrate_search_result_payloads(results: &mut [SearchResult]) -> Result<()> {
    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for (index, result) in results.iter().enumerate() {
        grouped
            .entry(result.raw_file.clone())
            .or_default()
            .push(index);
    }

    for (raw_file, mut indexes) in grouped {
        indexes.sort_by_key(|index| {
            (
                results[*index].raw_offset.unwrap_or(i64::MAX),
                results[*index].raw_line,
            )
        });
        let raw_path = PathBuf::from(&raw_file);
        let mut offset_reader = None;
        for index in indexes {
            let raw_line = results[index].raw_line;
            let raw_offset = results[index].raw_offset;
            let envelope = if let Some(raw_offset) = raw_offset {
                if offset_reader.is_none() {
                    offset_reader = Some(open_raw_offset_reader(&raw_path)?);
                }
                match read_raw_envelope_at_offset(
                    &raw_path,
                    offset_reader.as_mut().expect("offset reader initialized"),
                    raw_offset,
                )? {
                    Some(envelope) => envelope,
                    None => raw_envelope_for_line_scan(&raw_path, raw_line)?,
                }
            } else {
                raw_envelope_for_line_scan(&raw_path, raw_line)?
            };
            results[index].payload = resolved_payload_for_envelope(&raw_path, &envelope)?;
        }
    }
    Ok(())
}
