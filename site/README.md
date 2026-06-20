# site

Home for nabu's web presence — the planned **Astro Starlight** site for GitHub
Pages (presentation landing + docs).

## Status

- `design-system/` — **imported.** The nabu design-system foundation (tokens,
  global stylesheet, logo). Usable today as the site's global CSS.
- Starlight scaffold — **not yet built.** Needs `package.json`,
  `astro.config.mjs`, the `@astrojs/starlight` integration, content collections
  for the docs in `../docs/`, a landing page (port of the design system's
  `ui_kits/nabu-web` landing), and a GitHub Pages deploy workflow.

## Intended shape

- **Landing** — a custom Astro page using `design-system/styles.css`, built from
  the wedge wordmark + lapis/gold identity. The demo GIF (`../demo/demo-full.gif`)
  is the hero.
- **Docs** — Starlight pages sourced from the existing specs in `../docs/`
  (capture guarantees, usage, MCP tools, event envelope).
