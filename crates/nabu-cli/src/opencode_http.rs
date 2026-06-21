//! Minimal HTTP client for the OpenCode server backfill path.
//!
//! OpenCode can expose session transcripts over a local HTTP API. To reconcile
//! those without pulling in a full HTTP stack, this module speaks just enough
//! HTTP/1.1 over a raw `TcpStream` to GET a session's messages and parse the
//! JSON body. Only `fetch_opencode_session_messages` is public; URL parsing,
//! request construction, and response parsing stay private.

use nabu_core::Error;
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;

pub(crate) fn fetch_opencode_session_messages(
    server_url: &str,
    session_id: &str,
) -> nabu_core::Result<Value> {
    let (host, port, base_path) = parse_http_url(server_url)?;
    let mut stream = TcpStream::connect((host, port)).map_err(|source| Error::Io {
        path: PathBuf::from(server_url),
        source,
    })?;
    let mut request = String::with_capacity(
        "GET  HTTP/1.1\r\nHost: \r\nAccept: application/json\r\nConnection: close\r\n\r\n".len()
            + opencode_session_messages_path_len(base_path, session_id)
            + host.len(),
    );
    request.push_str("GET ");
    push_opencode_session_messages_path(&mut request, base_path, session_id);
    request.push_str(" HTTP/1.1\r\nHost: ");
    request.push_str(host);
    request.push_str("\r\nAccept: application/json\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|source| Error::Io {
            path: PathBuf::from(server_url),
            source,
        })?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|source| Error::Io {
            path: PathBuf::from(server_url),
            source,
        })?;
    parse_http_json_response(server_url, &response)
}

fn parse_http_url(url: &str) -> nabu_core::Result<(&str, u16, &str)> {
    let Some(rest) = url.strip_prefix("http://") else {
        return Err(Error::Validation(
            "OpenCode server URL must use http:// for local reconciliation".to_string(),
        ));
    };
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = authority
        .rsplit_once(':')
        .and_then(|(host, port)| Some((host, port.parse::<u16>().ok()?)))
        .unwrap_or((authority, 80));
    if host.is_empty() {
        return Err(Error::Validation(
            "OpenCode server URL host must not be empty".to_string(),
        ));
    }
    Ok((host, port, path))
}

fn opencode_session_messages_path_len(base: &str, session_id: &str) -> usize {
    let base = base.trim_matches('/');
    let suffix_len = "/session/".len() + session_id.len() + "/message".len();
    if base.is_empty() {
        suffix_len
    } else {
        1 + base.len() + suffix_len
    }
}

fn push_opencode_session_messages_path(request: &mut String, base: &str, session_id: &str) {
    let base = base.trim_matches('/');
    if !base.is_empty() {
        request.push('/');
        request.push_str(base);
    }
    request.push_str("/session/");
    request.push_str(session_id);
    request.push_str("/message");
}

fn parse_http_json_response(server_url: &str, response: &[u8]) -> nabu_core::Result<Value> {
    let Some(split) = response.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err(Error::Validation(
            "OpenCode server returned an invalid HTTP response".to_string(),
        ));
    };
    let headers = std::str::from_utf8(&response[..split]).map_err(|_| {
        Error::Validation("OpenCode server returned non-UTF8 HTTP headers".to_string())
    })?;
    let status = headers.lines().next().unwrap_or_default();
    if !status.contains(" 200 ") {
        return Err(Error::Validation(format!(
            "OpenCode server request failed: {status}"
        )));
    }
    let body = &response[split + 4..];
    serde_json::from_slice(body).map_err(|source| {
        Error::Validation(format!(
            "OpenCode server returned invalid JSON from {server_url}: {source}"
        ))
    })
}
