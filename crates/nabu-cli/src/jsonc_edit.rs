//! Layout-preserving JSONC editing for the OpenCode config file.
//!
//! OpenCode's `opencode.json` is JSON-with-comments. To register/unregister the
//! `nabu` MCP server without reformatting the user's file or dropping their
//! comments, this module hand-rolls a tokenizer that locates the `mcp` object
//! and splices a property in/out while preserving surrounding bytes, indentation,
//! newline style, and BOM. The public surface is intentionally narrow: the
//! caller only needs `rewrite_opencode_mcp_text_preserving_layout` (the
//! layout-preserving rewrite, returning `None` when the file shape requires a
//! full reserialize) and `opencode_mcp_text_entry_installed` (text-based
//! detection). All tokenizer/scanner primitives stay private to this module.

use crate::McpConfigAction;
use nabu_core::Error;

pub(crate) fn rewrite_opencode_mcp_text_preserving_layout(
    content: &str,
    action: McpConfigAction,
) -> nabu_core::Result<Option<String>> {
    let Some(root_start) = json_root_object_start(content) else {
        return Ok(None);
    };
    if find_matching_json_delimiter(content, root_start).is_none() {
        return Ok(None);
    }
    let Some(mcp_count) = json_object_property_count(content, root_start, "mcp") else {
        return Ok(None);
    };
    if mcp_count > 1 {
        return Err(Error::Validation(
            "OpenCode config has duplicate top-level mcp keys".to_string(),
        ));
    }

    match action {
        McpConfigAction::Install => {
            if opencode_mcp_text_entry_installed(content) {
                return Ok(Some(content.to_string()));
            }
            let rewritten = if let Some(mcp) = find_json_object_property(content, root_start, "mcp")
            {
                if content.as_bytes().get(mcp.value_start).copied() != Some(b'{') {
                    return Ok(None);
                }
                let Some(nabu_count) = json_object_property_count(content, mcp.value_start, "nabu")
                else {
                    return Ok(None);
                };
                if nabu_count > 1 {
                    return Err(Error::Validation(
                        "OpenCode config has duplicate mcp.nabu keys".to_string(),
                    ));
                }
                insert_json_object_property(content, mcp.value_start, OPENCODE_NABU_MCP_PROPERTY)?
            } else {
                insert_json_object_property(content, root_start, OPENCODE_MCP_PROPERTY)?
            };
            Ok(Some(rewritten))
        }
        McpConfigAction::Uninstall => {
            let Some((mcp, nabu)) = find_opencode_nabu_text_entry(content) else {
                return Ok(Some(content.to_string()));
            };
            let Some(nabu_count) = json_object_property_count(content, mcp.value_start, "nabu")
            else {
                return Ok(None);
            };
            if nabu_count > 1 {
                return Err(Error::Validation(
                    "OpenCode config has duplicate mcp.nabu keys".to_string(),
                ));
            }
            let mut rewritten = remove_json_object_property(content, nabu);
            if let Some(root_start) = json_root_object_start(&rewritten) {
                if let Some(mcp) = find_json_object_property(&rewritten, root_start, "mcp") {
                    if rewritten.as_bytes().get(mcp.value_start).copied() == Some(b'{')
                        && json_object_is_whitespace_empty(&rewritten, mcp.value_start)
                    {
                        rewritten = remove_json_object_property(&rewritten, mcp);
                    }
                }
            }
            Ok(Some(rewritten))
        }
    }
}

pub(crate) fn opencode_mcp_text_entry_installed(content: &str) -> bool {
    find_opencode_nabu_text_entry(content).is_some()
}

fn find_opencode_nabu_text_entry(content: &str) -> Option<(JsonPropertyRange, JsonPropertyRange)> {
    let root_start = json_root_object_start(content)?;
    let mcp = find_json_object_property(content, root_start, "mcp")?;
    if content.as_bytes().get(mcp.value_start).copied() != Some(b'{') {
        return None;
    }
    let nabu = find_json_object_property(content, mcp.value_start, "nabu")?;
    Some((mcp, nabu))
}

#[derive(Clone, Copy)]
struct JsonPropertyRange {
    property_start: usize,
    value_start: usize,
    value_end: usize,
}

fn json_root_object_start(content: &str) -> Option<usize> {
    let index = skip_json_ws_and_comments(content, 0);
    (content.as_bytes().get(index).copied() == Some(b'{')).then_some(index)
}

fn find_json_object_property(
    content: &str,
    object_start: usize,
    key: &str,
) -> Option<JsonPropertyRange> {
    if content.as_bytes().get(object_start).copied() != Some(b'{') {
        return None;
    }
    let object_end = find_matching_json_delimiter(content, object_start)?;
    let mut index = object_start + 1;
    while index < object_end {
        index = skip_json_ws_and_comments(content, index);
        if index >= object_end || content.as_bytes().get(index).copied() == Some(b'}') {
            break;
        }
        let property_start = index;
        let (property_key, after_key) = parse_json_string_key(content, index)?;
        index = skip_json_ws_and_comments(content, after_key);
        if content.as_bytes().get(index).copied() != Some(b':') {
            return None;
        }
        let value_start = skip_json_ws_and_comments(content, index + 1);
        let value_end = scan_json_value_end(content, value_start)?;
        if property_key == key {
            return Some(JsonPropertyRange {
                property_start,
                value_start,
                value_end,
            });
        }
        index = skip_json_ws_and_comments(content, value_end);
        if content.as_bytes().get(index).copied() == Some(b',') {
            index += 1;
        }
    }
    None
}

fn json_object_property_count(content: &str, object_start: usize, key: &str) -> Option<usize> {
    if content.as_bytes().get(object_start).copied() != Some(b'{') {
        return None;
    }
    let object_end = find_matching_json_delimiter(content, object_start)?;
    let mut index = object_start + 1;
    let mut count = 0usize;
    while index < object_end {
        index = skip_json_ws_and_comments(content, index);
        if index >= object_end || content.as_bytes().get(index).copied() == Some(b'}') {
            break;
        }
        let (property_key, after_key) = parse_json_string_key(content, index)?;
        index = skip_json_ws_and_comments(content, after_key);
        if content.as_bytes().get(index).copied() != Some(b':') {
            return None;
        }
        let value_start = skip_json_ws_and_comments(content, index + 1);
        let value_end = scan_json_value_end(content, value_start)?;
        if property_key == key {
            count += 1;
        }
        index = skip_json_ws_and_comments(content, value_end);
        if content.as_bytes().get(index).copied() == Some(b',') {
            index += 1;
        }
    }
    Some(count)
}

fn insert_json_object_property(
    content: &str,
    object_start: usize,
    property: &str,
) -> nabu_core::Result<String> {
    let Some(object_end) = find_matching_json_delimiter(content, object_start) else {
        return Err(Error::Validation(
            "OpenCode config must be a JSON/JSONC object".to_string(),
        ));
    };
    let newline = detect_newline(content);
    let close_indent = line_indent_before(content, object_end);
    let property_indent = format!("{close_indent}  ");
    let property = indent_json_property(property, &property_indent, newline);
    let mut output = String::with_capacity(content.len() + property.len() + 4);

    if let Some(last_property) = last_json_object_property(content, object_start) {
        let after_value = skip_json_ws_and_comments(content, last_property.value_end);
        let has_trailing_comma =
            after_value < object_end && content.as_bytes().get(after_value).copied() == Some(b',');
        output.push_str(&content[..last_property.value_end]);
        if !has_trailing_comma {
            output.push(',');
        }
        output.push_str(content[last_property.value_end..object_end].trim_end());
    } else {
        output.push_str(content[..object_end].trim_end());
    }
    output.push_str(newline);
    output.push_str(&property);
    output.push_str(newline);
    output.push_str(&close_indent);
    output.push_str(&content[object_end..]);
    Ok(output)
}

fn remove_json_object_property(content: &str, property: JsonPropertyRange) -> String {
    let bytes = content.as_bytes();
    let after_inline_ws = skip_inline_ws(content, property.value_end);
    let has_trailing_comma = bytes.get(after_inline_ws).copied() == Some(b',');
    let end = removable_property_end(content, property.value_end);
    let line_start = removable_line_start(content, property.property_start);
    let start = if has_trailing_comma {
        line_start
    } else if let Some(previous_comma) = previous_non_ws_byte(content, property.property_start)
        .filter(|(_, byte)| *byte == b',')
        .map(|(index, _)| index)
    {
        previous_comma
    } else {
        line_start
    };
    let mut output = String::with_capacity(content.len().saturating_sub(end - start));
    output.push_str(&content[..start]);
    output.push_str(&content[end..]);
    output
}

fn last_json_object_property(content: &str, object_start: usize) -> Option<JsonPropertyRange> {
    if content.as_bytes().get(object_start).copied() != Some(b'{') {
        return None;
    }
    let object_end = find_matching_json_delimiter(content, object_start)?;
    let mut index = object_start + 1;
    let mut last = None;
    while index < object_end {
        index = skip_json_ws_and_comments(content, index);
        if index >= object_end || content.as_bytes().get(index).copied() == Some(b'}') {
            break;
        }
        let property_start = index;
        let (_, after_key) = parse_json_string_key(content, index)?;
        index = skip_json_ws_and_comments(content, after_key);
        if content.as_bytes().get(index).copied() != Some(b':') {
            return None;
        }
        let value_start = skip_json_ws_and_comments(content, index + 1);
        let value_end = scan_json_value_end(content, value_start)?;
        last = Some(JsonPropertyRange {
            property_start,
            value_start,
            value_end,
        });
        index = skip_json_ws_and_comments(content, value_end);
        if content.as_bytes().get(index).copied() == Some(b',') {
            index += 1;
        }
    }
    last
}

fn json_object_is_whitespace_empty(content: &str, object_start: usize) -> bool {
    find_matching_json_delimiter(content, object_start)
        .map(|object_end| content[object_start + 1..object_end].trim().is_empty())
        .unwrap_or(false)
}

fn skip_json_ws_and_comments(content: &str, mut index: usize) -> usize {
    let bytes = content.as_bytes();
    if index == 0 && bytes.starts_with(b"\xEF\xBB\xBF") {
        index = 3;
    }
    loop {
        while bytes
            .get(index)
            .is_some_and(|byte| matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
        {
            index += 1;
        }
        if bytes.get(index).copied() == Some(b'/') && bytes.get(index + 1).copied() == Some(b'/') {
            index += 2;
            while bytes
                .get(index)
                .is_some_and(|byte| !matches!(byte, b'\n' | b'\r'))
            {
                index += 1;
            }
            continue;
        }
        if bytes.get(index).copied() == Some(b'/') && bytes.get(index + 1).copied() == Some(b'*') {
            index += 2;
            while index + 1 < bytes.len()
                && !(bytes[index] == b'*' && bytes.get(index + 1).copied() == Some(b'/'))
            {
                index += 1;
            }
            index = (index + 2).min(bytes.len());
            continue;
        }
        return index;
    }
}

fn skip_inline_ws(content: &str, mut index: usize) -> usize {
    let bytes = content.as_bytes();
    while bytes
        .get(index)
        .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
    {
        index += 1;
    }
    index
}

fn removable_property_end(content: &str, value_end: usize) -> usize {
    let bytes = content.as_bytes();
    let mut end = skip_inline_ws(content, value_end);
    if bytes.get(end).copied() == Some(b',') {
        end += 1;
        end = skip_inline_ws(content, end);
    }
    if bytes.get(end).copied() == Some(b'/') && bytes.get(end + 1).copied() == Some(b'/') {
        end += 2;
        while bytes
            .get(end)
            .is_some_and(|byte| !matches!(byte, b'\n' | b'\r'))
        {
            end += 1;
        }
    } else if bytes.get(end).copied() == Some(b'/') && bytes.get(end + 1).copied() == Some(b'*') {
        end += 2;
        while end + 1 < bytes.len() && !(bytes[end] == b'*' && bytes[end + 1] == b'/') {
            end += 1;
        }
        end = (end + 2).min(bytes.len());
        end = skip_inline_ws(content, end);
    }
    if content[end..].starts_with("\r\n") {
        end + 2
    } else if content[end..].starts_with('\n') {
        end + 1
    } else {
        end
    }
}

fn parse_json_string_key(content: &str, index: usize) -> Option<(String, usize)> {
    if content.as_bytes().get(index).copied() != Some(b'"') {
        return None;
    }
    let end = skip_json_string(content, index)?;
    let raw = &content[index + 1..end - 1];
    if raw.contains('\\') {
        serde_json::from_str(&content[index..end])
            .ok()
            .map(|key| (key, end))
    } else {
        Some((raw.to_string(), end))
    }
}

fn skip_json_string(content: &str, mut index: usize) -> Option<usize> {
    let bytes = content.as_bytes();
    if bytes.get(index).copied() != Some(b'"') {
        return None;
    }
    index += 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index = index.saturating_add(2),
            b'"' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}

fn scan_json_value_end(content: &str, index: usize) -> Option<usize> {
    match content.as_bytes().get(index).copied()? {
        b'{' | b'[' => find_matching_json_delimiter(content, index).map(|end| end + 1),
        b'"' => skip_json_string(content, index),
        _ => {
            let bytes = content.as_bytes();
            let mut end = index;
            while end < bytes.len() && !matches!(bytes[end], b',' | b'}' | b']') {
                if bytes[end] == b'/'
                    && matches!(bytes.get(end + 1).copied(), Some(b'/') | Some(b'*'))
                {
                    break;
                }
                end += 1;
            }
            let mut trimmed_end = end;
            while trimmed_end > index
                && matches!(bytes[trimmed_end - 1], b' ' | b'\n' | b'\r' | b'\t')
            {
                trimmed_end -= 1;
            }
            Some(trimmed_end)
        }
    }
}

fn find_matching_json_delimiter(content: &str, open_index: usize) -> Option<usize> {
    let bytes = content.as_bytes();
    let mut stack = vec![match bytes.get(open_index).copied()? {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    }];
    let mut index = open_index + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => index = skip_json_string(content, index)?,
            b'/' if bytes.get(index + 1).copied() == Some(b'/') => {
                index += 2;
                while bytes
                    .get(index)
                    .is_some_and(|byte| !matches!(byte, b'\n' | b'\r'))
                {
                    index += 1;
                }
            }
            b'/' if bytes.get(index + 1).copied() == Some(b'*') => {
                index += 2;
                while index + 1 < bytes.len() && !(bytes[index] == b'*' && bytes[index + 1] == b'/')
                {
                    index += 1;
                }
                index = (index + 2).min(bytes.len());
            }
            b'{' => {
                stack.push(b'}');
                index += 1;
            }
            b'[' => {
                stack.push(b']');
                index += 1;
            }
            b'}' | b']' => {
                if stack.pop()? != bytes[index] {
                    return None;
                }
                if stack.is_empty() {
                    return Some(index);
                }
                index += 1;
            }
            _ => index += 1,
        }
    }
    None
}

fn line_indent_before(content: &str, index: usize) -> String {
    let line_start = content[..index]
        .rfind('\n')
        .map(|offset| offset + 1)
        .unwrap_or(0);
    content[line_start..index]
        .chars()
        .take_while(|character| matches!(character, ' ' | '\t'))
        .collect()
}

fn detect_newline(content: &str) -> &'static str {
    if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

fn indent_json_property(property: &str, indent: &str, newline: &str) -> String {
    let mut output = String::new();
    for (index, line) in property.lines().enumerate() {
        if index > 0 {
            output.push_str(newline);
        }
        output.push_str(indent);
        output.push_str(line);
    }
    output
}

fn removable_line_start(content: &str, property_start: usize) -> usize {
    let line_start = content[..property_start]
        .rfind('\n')
        .map(|offset| offset + 1)
        .unwrap_or(0);
    if content[line_start..property_start].trim().is_empty() {
        line_start
    } else {
        property_start
    }
}

fn previous_non_ws_byte(content: &str, before: usize) -> Option<(usize, u8)> {
    let bytes = content.as_bytes();
    let mut index = before;
    while index > 0 {
        index -= 1;
        if !matches!(bytes[index], b' ' | b'\n' | b'\r' | b'\t') {
            return Some((index, bytes[index]));
        }
    }
    None
}

const OPENCODE_MCP_PROPERTY: &str = r#""mcp": {
  "nabu": {
    "type": "local",
    "command": [
      "nabu",
      "mcp",
      "serve",
      "--transport",
      "stdio"
    ],
    "enabled": true
  }
}"#;

#[cfg(test)]
pub(crate) fn jsonc_to_json_value(content: &str) -> serde_json::Value {
    let bytes = content.as_bytes();
    let mut index = if bytes.starts_with(b"\xEF\xBB\xBF") {
        3
    } else {
        0
    };
    let mut stripped = String::with_capacity(content.len());
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let end = skip_json_string(content, index).expect("valid JSON string");
                stripped.push_str(&content[index..end]);
                index = end;
            }
            b'/' if bytes.get(index + 1).copied() == Some(b'/') => {
                index += 2;
                while bytes
                    .get(index)
                    .is_some_and(|byte| !matches!(byte, b'\n' | b'\r'))
                {
                    index += 1;
                }
            }
            b'/' if bytes.get(index + 1).copied() == Some(b'*') => {
                index += 2;
                while index + 1 < bytes.len() && !(bytes[index] == b'*' && bytes[index + 1] == b'/')
                {
                    index += 1;
                }
                index = (index + 2).min(bytes.len());
            }
            _ => {
                let character = content[index..].chars().next().expect("valid UTF-8");
                stripped.push(character);
                index += character.len_utf8();
            }
        }
    }
    serde_json::from_str(&stripped).expect("rewritten JSONC parses after comment stripping")
}

const OPENCODE_NABU_MCP_PROPERTY: &str = r#""nabu": {
  "type": "local",
  "command": [
    "nabu",
    "mcp",
    "serve",
    "--transport",
    "stdio"
  ],
  "enabled": true
}"#;
