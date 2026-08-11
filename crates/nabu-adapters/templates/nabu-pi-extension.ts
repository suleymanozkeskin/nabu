/**
 * nabu-pi-extension
 * marker: NABU_PI_EXTENSION
 * version: 1
 * role: capture
 *
 * Live capture for nabu inside pi (@earendil-works/pi-coding-agent).
 * Subscribes to session_start, message_end, and session_compact only, and
 * shells out to `nabu ingest hook --tool pi` with the same canonical mapping
 * and dedupe ids as the session-file backfill. Fail-open: every error is
 * logged to stderr and never thrown into pi.
 *
 * The extension runs with full user permissions (pi security model).
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";

export default function (pi: ExtensionAPI) {
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
