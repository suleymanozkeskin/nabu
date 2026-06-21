# Security Policy

## Supported Versions

nabu is pre-1.0. Security fixes are made on `main` and shipped in the next
published release.

## Reporting a Vulnerability

Please report security issues privately by opening a GitHub security advisory on
the repository. Avoid posting secrets, transcripts, or proof-of-concept payloads
in public issues.

Include:

- The affected command or MCP tool.
- Whether semantic search was built with `--features semantic`.
- A minimal redacted input fixture when possible.
- The expected and observed behavior.

## Threat Model

nabu stores local coding-agent history. Raw captures may contain source code,
credentials, prompts, tool output, file paths, and other sensitive data. The
default design keeps capture, indexing, search, export, and MCP reads local to
the machine and does not send transcript data to a hosted service.

Primary risks:

- Accidental disclosure through exports, MCP responses, terminal output, or
  backups.
- Parser bugs while ingesting untrusted local JSON/JSONL emitted by external
  tools.
- Dependency vulnerabilities in bundled/native components such as SQLite,
  sqlite-vec, ONNX Runtime, TLS, and JSON parsers.
- Local file permission mistakes that expose raw history or backups to other
  users on the same machine.

Non-goals:

- nabu does not sandbox the coding agents it observes.
- nabu does not guarantee that redaction catches every secret.
- nabu does not make local model downloads anonymous; semantic model acquisition
  contacts the configured Hugging Face endpoint when explicitly requested.

Recommended handling:

- Treat `~/.nabu/raw`, `~/.nabu/blobs`, exports, and backups as sensitive.
- Use `--redact` for agent-facing or shareable exports.
- Run `nabu purge` when history should be removed.
- Keep CI supply-chain gates passing before release.
