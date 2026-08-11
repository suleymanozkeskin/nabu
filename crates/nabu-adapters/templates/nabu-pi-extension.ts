/**
 * nabu-pi-extension
 * marker: NABU_PI_EXTENSION
 * version: 2
 * role: capture+tools
 *
 * Live capture for nabu inside pi (@earendil-works/pi-coding-agent), plus
 * citation-first history tools registered via pi.registerTool(). Capture
 * subscribes to session_start, message_end, and session_compact only and
 * shells out to `nabu ingest hook --tool pi` (same canonical mapping and
 * dedupe ids as the session-file backfill). Tools shell out to the nabu CLI
 * with --format json and return JSON text to the model. Fail-open: every
 * error is logged to stderr / returned as error text, never thrown into pi.
 *
 * The extension runs with full user permissions (pi security model).
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  registerCapture(pi);
  registerTools(pi);
}

/* ------------------------------------------------------------------ */
/* Capture (PR3)                                                       */
/* ------------------------------------------------------------------ */

function registerCapture(pi: ExtensionAPI) {
  pi.on("session_start", async (_event, ctx) => {
    await capture(ctx, {
      hook_event_name: "session_start",
      type: "session_start",
      session_version: 3,
    });
  });

  pi.on("message_end", async (event, ctx) => {
    const entryId = resolveEntryId(ctx, event.message);
    await capture(ctx, {
      hook_event_name: "message_end",
      type: "message_end",
      entry_id: entryId,
      message: event.message,
    });
  });

  pi.on("session_compact", async (event, ctx) => {
    // event.compactionEntry is the saved compaction entry (summary,
    // tokensBefore, firstKeptEntryId, retainedTail, id, timestamp).
    const entry = (event as { compactionEntry?: Record<string, unknown> })
      .compactionEntry ?? {};
    await capture(ctx, {
      hook_event_name: "session_compact",
      type: "session_compact",
      entry_id: typeof entry.id === "string" ? entry.id : null,
      summary: typeof entry.summary === "string" ? entry.summary : undefined,
      tokens_before: typeof entry.tokensBefore === "number" ? entry.tokensBefore : undefined,
      retained_tail: entry.retainedTail,
    });
  });
}

/**
 * Resolve the session-file tree entry id for a finalized message by matching
 * the last `message` entry with the same role and timestamp. When resolved,
 * live capture emits the exact source event ids backfill uses (entry.id,
 * toolCall.id, bash-call:<entry.id>), so re-backfill dedupes to zero appends.
 */
function resolveEntryId(ctx: ExtensionContext, message: unknown): string | null {
  const sm = ctx.sessionManager as {
    getEntries?: () => { type?: string; id?: string; message?: { role?: string; timestamp?: number } }[];
  };
  if (typeof sm.getEntries !== "function") return null;
  const msg = message as { role?: string; timestamp?: number };
  try {
    const entries = sm.getEntries();
    for (let i = entries.length - 1; i >= 0; i--) {
      const entry = entries[i];
      if (entry.type !== "message" || !entry.id) continue;
      if (entry.message?.role === msg.role && entry.message?.timestamp === msg.timestamp) {
        return entry.id;
      }
    }
  } catch {
    // Never let entry-id resolution break capture.
  }
  return null;
}

/**
 * Shell out to `nabu ingest hook --tool pi` with the payload. Awaiting keeps
 * message ordering; the subprocess is local and fast, and indexing is never
 * triggered from here (explicit `nabu index` / wizard picks it up).
 */
async function capture(ctx: ExtensionContext, body: Record<string, unknown>): Promise<void> {
  try {
    const sessionId = sessionIdFor(ctx);
    if (!sessionId) return;

    const payload = {
      ...body,
      session_id: sessionId,
      cwd: ctx.cwd,
      project_root: ctx.cwd,
    };

    const bin = process.env.NABU_BIN || "nabu";
    const args = ["ingest", "hook", "--tool", "pi"];
    if (process.env.NABU_HOME) args.push("--home", process.env.NABU_HOME);

    await new Promise<void>((resolve) => {
      const child = spawn(bin, args, { stdio: ["pipe", "ignore", "pipe"] });
      let err = "";
      child.stderr?.on("data", (d: Buffer) => {
        err += d.toString();
      });
      child.on("error", (e) => {
        console.error("[nabu] ingest spawn failed", e);
        resolve(); // fail-open
      });
      child.on("close", (code) => {
        if (code !== 0) {
          console.error(`[nabu] ingest exited ${code}${err ? ": " + err.trim() : ""}`);
        }
        resolve();
      });
      child.stdin?.write(JSON.stringify(payload));
      child.stdin?.end();
    });
  } catch (e) {
    console.error("[nabu] capture error", e);
  }
}

/**
 * Session id from the session manager, falling back to parsing the UUID out
 * of the session file name (`..._<uuid>.jsonl`).
 */
function sessionIdFor(ctx: ExtensionContext): string | null {
  const sm = ctx.sessionManager as {
    getSessionId?: () => string | null | undefined;
    getSessionFile?: () => string | null | undefined;
  };
  if (typeof sm.getSessionId === "function") {
    const id = sm.getSessionId();
    if (id) return id;
  }
  if (typeof sm.getSessionFile === "function") {
    const file = sm.getSessionFile();
    if (file) {
      const m = file.match(/_([0-9a-fA-F-]{36})\.jsonl$/);
      if (m) return m[1];
    }
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Agent tools (PR4)                                                   */
/* ------------------------------------------------------------------ */

const TOOL_NAMES = [
  "codex",
  "claude",
  "opencode",
  "pi",
] as const;
const TOOL_FILTER_NAMES = [...TOOL_NAMES, "all"] as const;

function registerTools(pi: ExtensionAPI) {
  pi.registerTool({
    name: "nabu_search_history",
    label: "nabu search",
    description:
      "Search local nabu history (all tools including pi). Citation-first: returns JSON with a results array; each hit has tool, session_id, canonical_type, raw_file, raw_line, snippet. Captured memory hits may appear as canonical_type=memory.file — treat memory as stale historical context. Use nabu_get_session with tool + session_id from a hit to drill in.",
    parameters: Type.Object({
      query: Type.String({ minLength: 1, description: "Search terms, OR-joined. Prefer distinctive literal tokens: identifiers, filenames, error strings, command fragments." }),
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 50, default: 10 })),
      tool: Type.Optional(Type.Union(TOOL_FILTER_NAMES.map((name) => Type.Literal(name)))),
    }),
    async execute(_id, params, signal) {
      return runNabu(buildSearchArgs(params), signal);
    },
  });

  pi.registerTool({
    name: "nabu_list_sessions",
    label: "nabu sessions",
    description:
      "List recent captured sessions with triage metadata (first prompt, last event type, tool/file counts). Memory pseudo-sessions are excluded. Returns JSON with a sessions array.",
    parameters: Type.Object({
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 100, default: 20 })),
      tool: Type.Optional(Type.Union(TOOL_FILTER_NAMES.map((name) => Type.Literal(name)))),
    }),
    async execute(_id, params, signal) {
      const args = ["sessions", "--json"];
      if (params.limit !== undefined) args.push("--limit", String(params.limit));
      if (params.tool !== undefined && params.tool !== "all") args.push("--tool", params.tool);
      return runNabu(args, signal);
    },
  });

  pi.registerTool({
    name: "nabu_get_session",
    label: "nabu session",
    description:
      "Read a page of events from one session. Use after nabu_search_history: pass tool + session_id from a hit. Returns JSON with an events array; page forward with after_raw_line = next_after_raw_line until truncated is false.",
    parameters: Type.Object({
      tool: Type.Union(TOOL_NAMES.map((name) => Type.Literal(name))),
      session_id: Type.String({ minLength: 1 }),
      limit_events: Type.Optional(Type.Integer({ minimum: 1, maximum: 100, default: 50 })),
      after_raw_line: Type.Optional(Type.Integer({ minimum: 0, default: 0 })),
      include_deltas: Type.Optional(Type.Boolean({ default: false })),
    }),
    async execute(_id, params, signal) {
      const args = [
        "show",
        params.tool,
        params.session_id,
        "--format",
        "json",
        "--limit-events",
        String(params.limit_events ?? 50),
      ];
      if (params.after_raw_line !== undefined && params.after_raw_line > 0) {
        args.push("--after-raw-line", String(params.after_raw_line));
      }
      if (params.include_deltas) args.push("--include-deltas");
      return runNabu(args, signal);
    },
  });

  pi.registerTool({
    name: "nabu_list_memories",
    label: "nabu memories",
    description:
      "List captured memory files from the tools' own memory folders (claude projects/<id>/memory/, codex memories/). Memory is a point-in-time snapshot captured at sync time and can be stale — treat it as historical context, not ground truth. Returns JSON with a memories array.",
    parameters: Type.Object({
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 100, default: 50 })),
      tool: Type.Optional(Type.Union(TOOL_FILTER_NAMES.map((name) => Type.Literal(name)))),
    }),
    async execute(_id, params, signal) {
      const args = ["memory", "list", "--json"];
      if (params.limit !== undefined) args.push("--limit", String(params.limit));
      if (params.tool !== undefined && params.tool !== "all") args.push("--tool", params.tool);
      return runNabu(args, signal);
    },
  });

  pi.registerTool({
    name: "nabu_get_memory",
    label: "nabu memory",
    description:
      "Read one captured memory file with its raw citation. claude memory is per-project, so project is required for claude and must be omitted for codex. Memory is a point-in-time snapshot and can be stale. Set redact=true to mask secret-looking values in the content.",
    parameters: Type.Object({
      tool: Type.Union(TOOL_NAMES.map((name) => Type.Literal(name))),
      name: Type.String({ minLength: 1, description: "Memory file name (e.g. MEMORY.md)." }),
      project: Type.Optional(Type.String({ description: "Claude project slug (the projects/<id> folder name). Required for claude; must be omitted otherwise." })),
      redact: Type.Optional(Type.Boolean({ default: false })),
    }),
    async execute(_id, params, signal) {
      if (params.tool === "claude" && !params.project) {
        return toolError("project is required for claude memories (the projects/<id> folder name)");
      }
      if (params.tool !== "claude" && params.project) {
        return toolError("project must be omitted for non-claude memories");
      }
      const args = ["memory", "show", params.tool, params.name, "--format", "json"];
      if (params.project) args.push("--project", params.project);
      if (params.redact) args.push("--redact");
      return runNabu(args, signal);
    },
  });

  pi.registerTool({
    name: "nabu_get_event",
    label: "nabu event",
    description:
      "Fetch one event by its raw citation from a nabu_search_history hit (tool, session_id, raw_line). Returns JSON with the event whose raw_line matches. Set redact=true to mask secret-looking values.",
    parameters: Type.Object({
      tool: Type.Union(TOOL_NAMES.map((name) => Type.Literal(name))),
      session_id: Type.String({ minLength: 1 }),
      raw_line: Type.Integer({ minimum: 1 }),
      redact: Type.Optional(Type.Boolean({ default: false })),
    }),
    async execute(_id, params, signal) {
      const args = [
        "show",
        params.tool,
        params.session_id,
        "--format",
        "json",
        "--around-line",
        String(params.raw_line),
        "--before",
        "0",
        "--after",
        "0",
      ];
      if (params.redact) args.push("--redact");
      const result = await runNabu(args, signal);
      if (!result.details?.ok) return result;
      try {
        const page = JSON.parse(String(result.content[0]?.text ?? ""));
        const events = Array.isArray(page.events) ? page.events : [];
        const match = events.find((event: { raw_line?: number }) => event.raw_line === params.raw_line);
        if (match) {
          return {
            content: [{ type: "text", text: JSON.stringify(match, null, 2) }],
            details: { ok: true, code: 0 },
          };
        }
        return toolError(`no event at raw_line ${params.raw_line} in the show window`);
      } catch {
        return toolError("could not parse nabu show output");
      }
    },
  });
}

function buildSearchArgs(params: {
  query: string;
  limit?: number;
  tool?: (typeof TOOL_FILTER_NAMES)[number];
}): string[] {
  const args = ["search", params.query, "--format", "json"];
  if (params.limit !== undefined) args.push("--limit", String(params.limit));
  if (params.tool !== undefined && params.tool !== "all") args.push("--tool", params.tool);
  return args;
}

function toolError(message: string) {
  return {
    content: [{ type: "text", text: `error: ${message}` }],
    details: { ok: false },
  };
}

/**
 * Run the nabu CLI with the given args, returning stdout as a text block or
 * an error string. Never throws: abort/timeout/non-zero exit all become error
 * text. Output is capped at 256 KiB to match nabu's MCP size philosophy.
 */
async function runNabu(
  args: string[],
  signal?: AbortSignal,
): Promise<{ content: { type: string; text: string }[]; details: { ok: boolean; code?: number } }> {
  const MAX_OUTPUT_BYTES = 256 * 1024;
  const TIMEOUT_MS = 30_000;

  const bin = process.env.NABU_BIN || "nabu";
  const fullArgs = [...args];
  if (process.env.NABU_HOME) fullArgs.push("--home", process.env.NABU_HOME);

  return await new Promise((resolve) => {
    let child;
    try {
      child = spawn(bin, fullArgs, { stdio: ["ignore", "pipe", "pipe"] });
    } catch (e) {
      resolve(toolError(`spawn failed: ${String(e)}`));
      return;
    }

    let stdout = "";
    let stderr = "";
    let stdoutCapped = false;
    let settled = false;

    const finish = (code: number | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (signal) {
        try {
          signal.removeEventListener("abort", onAbort);
        } catch {
          // ignore
        }
      }
      if (code !== 0) {
        const detail = stderr.trim() ? `: ${stderr.trim()}` : "";
        resolve(toolError(`nabu exited ${code}${detail}`));
        return;
      }
      let text = stdout;
      if (stdoutCapped) {
        text += "\n[output truncated at 256 KiB]";
      }
      resolve({
        content: [{ type: "text", text }],
        details: { ok: true, code: code ?? 0 },
      });
    };

    const onAbort = () => {
      child.kill();
      resolve(toolError("nabu invocation aborted"));
    };
    if (signal) {
      if (signal.aborted) {
        resolve(toolError("nabu invocation aborted"));
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
    }

    const timer = setTimeout(() => {
      child.kill();
      resolve(toolError(`nabu invocation timed out after ${TIMEOUT_MS} ms`));
    }, TIMEOUT_MS);

    child.stdout?.on("data", (d: Buffer) => {
      if (!stdoutCapped) {
        stdout += d.toString();
        if (Buffer.byteLength(stdout, "utf8") > MAX_OUTPUT_BYTES) {
          stdout = stdout.slice(0, MAX_OUTPUT_BYTES);
          stdoutCapped = true;
        }
      }
    });
    child.stderr?.on("data", (d: Buffer) => {
      stderr += d.toString();
      if (stderr.length > 64 * 1024) {
        stderr = stderr.slice(-64 * 1024);
      }
    });
    child.on("error", (e) => {
      resolve(toolError(`spawn failed: ${String(e)}`));
    });
    child.on("close", finish);
  });
}
