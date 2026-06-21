# Demo beat — surviving compaction

A self-contained ~12 s beat for the demo reel and landing loop. It can stand
alone or slot in front of the existing "recover a past fix over MCP" sequence.
Source tooling lives in the demo repo; this is the shot list and the exact,
verified commands to drive it.

## The one idea

Compaction empties the live context window. nabu still has the exact turn —
verbatim, on local disk, with a citation. Show the loss, then the recovery.

## Format

- Terminal: 100×30, the landing's palette (Catppuccin Mocha base, lapis-and-gold
  accents matching `site/design-system/tokens/colors.css`).
- Font: the site mono; cursor visible; type at ~30 ms/char so commands read.
- Capture: drive a real PTY and record frames (asciinema or PTY+pyte), then
  render to GIF. Use real `nabu` output — do not mock. Hold the payoff frame
  ~2 s longer than the rest.
- No narration; one-line captions per shot, gold on transparent.

## Shot list

| # | Time | On screen | Caption |
| --- | --- | --- | --- |
| 1 | 0.0–2.0 s | A Claude Code prompt: `Canary token: PLUM-VELVET-3391-XYLOPHONE-QUASAR. Secret number 8472.` submitted; agent replies `ack`. | "every turn is captured as it happens" |
| 2 | 2.0–4.5 s | `/compact` → `Compacted (ctrl+o to see full summary)`, then the two hook lines: `PreCompact [nabu ingest hook …] skipped duplicate …` / `PostCompact [nabu ingest hook …] appended …`. | "compaction frees the window" |
| 3 | 4.5–6.0 s | Scroll the live context to show the verbatim canary is gone — only the summary remains. | "the context forgets" |
| 4 | 6.0–10.0 s | Run `nabu search "XYLOPHONE QUASAR canary" --tool claude --session <session>`; the hit prints the full verbatim turn with its `…/claude_<session>.jsonl:2` citation. | "nabu doesn't" |
| 5 | 10.0–12.0 s | Hold on the cited hit; underline the `tool:session:raw_line` coordinate. | "verbatim. cited. on local disk." |

## Exact commands (verified)

```shell
# shot 1 capture is live in the agent; confirm it indexed:
nabu index --once
nabu search "XYLOPHONE QUASAR canary" --tool claude

# shot 2 is the agent's /compact; the hook lines are nabu's PreCompact/PostCompact output.

# shot 4 payoff:
nabu search "XYLOPHONE QUASAR canary" --tool claude --session <session>
nabu show claude <session> --around-line 2 --before 0 --after 0
```

## Notes

- The literal canary string is a demo prop; keep it stable across takes so the
  before/after frames line up.
- Keep the claim exactly as the product behaves: the turn is *retrievable
  verbatim with a citation after compaction* — not "you never lose context."
- `skipped duplicate` in shot 2 is correct and worth leaving visible: it shows
  nabu recording the boundary once, not double-writing already-captured turns.
