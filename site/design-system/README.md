# nabu design system — vendored foundation

Imported from the **nabu Design System** Claude Design project
(`b0c66dfe-b08a-4fd7-95b4-c504e0e3b9c7`). This directory is the source of truth
for nabu's visual identity in code: link `styles.css` and reference the tokens.

## Palette — lapis & gold

Mesopotamian: deep **lapis lazuli** ground, a single **gold** accent, warm
**sand/parchment** text (never cold white), **carnelian** for danger and
**verdigris** for success.

| Role | Token | Hex |
|------|-------|-----|
| Background | `--bg-base` / `--lapis-900` | `#0e1322` |
| Sunken | `--bg-sunken` / `--lapis-950` | `#0b0f1a` |
| Body text | `--text-body` / `--sand-200` | `#d2c4a6` |
| Muted text | `--text-muted` / `--sand-400` | `#8f856a` |
| Accent | `--accent` / `--gold-500` | `#d9a23f` |
| Success | `--success` / `--verdigris-400` | `#6bb0a0` |
| Danger | `--danger` / `--carnelian-400` | `#d96b54` |
| Warning | `--warning` / `--gold-400` | `#e8b44a` |
| Info | `--info` / `--lapis-300` | `#7e95c6` |

Type is mono-forward: **IBM Plex Mono** (UI/terminal), **IBM Plex Sans**
(long-form prose), **Noto Sans Cuneiform** (ceremonial glyphs only).

## Contents

```
styles.css              global entry — @imports the tokens
tokens/
  colors.css            lapis / gold / clay / sand / carnelian / verdigris ramps + semantic aliases
  typography.css        families, type scale, weights, line-heights, tracking
  spacing.css           4px grid, control heights, container widths
  effects.css           radii, borders, shadows, glows, motion
  fonts.css             Google Fonts CDN @import (self-host for production)
assets/logo/
  nabu-logo.svg             wedge wordmark on dark lapis
  nabu-logo-transparent.svg wedge wordmark, transparent
```

The wordmark is the word `nabu` built from cuneiform **wedges** (gold
bronze→gold→pale-gold gradient) with the divine name `𒀭𒀝` beneath. Same SVG
drives the demo title card (`nabu-internal-docs/demo-source/demo/`).

## Where these are used

- **Demo pipeline** — `palette.py` in the demo-source repo mirrors these hexes;
  the title card rasterizes `nabu-logo.svg`.
- **CLI** — `crates/nabu-cli/src/theme.rs` emits the gold/verdigris/carnelian
  accents as 24-bit truecolor.
- **Docs site** — these tokens are the intended global stylesheet for the
  planned Astro Starlight site (see `../README.md`).

## Not vendored

`_ds_bundle.js` (compiled React component bundle), the React component sources
(`components/**`), and the interactive `ui_kits/nabu-web` landing/console live
in the Design project. Pull them with the `DesignSync` MCP tool if the Starlight
site needs the rendered components rather than just the tokens.
