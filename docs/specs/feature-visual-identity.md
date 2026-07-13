# Spec — Visual identity pass ("sexy pass"): paper & ink

**Status:** 🎨 **design-locked** (grilled 2026-07-13 — see Grill ledger below; 11 forks, none parked).
**Phased:** **V1** design tokens + fonts + icons → **V2** layout rhythm + header/wordmark + empty
states + ~70ch reading measure → **V3** Tauri app icon + branding + polish audit. **Stop-early after
V1 is allowed** (each phase ships independently; V1 is a complete, coherent deliverable on its own).
**Owner of execution:** Claude Code (Svelte/TS + CSS only — no backend, no `src/` change).
**Roadmap:** `docs/ui-checklist.md` §3 "Visual identity pass" row; ROADMAP Phase-8 (open UI track).
**Pattern reference:** thin-shell rule (`apps/` render, no logic; root `CLAUDE.md`). Every color the
app renders already flows through the CSS custom properties in `apps/desktop/src/app.css` (verified:
**zero hardcoded hex in any `.svelte` component**), so a token retheme recolors the whole app without
touching component styles. Local-first / offline / on-proxy constraints (`.claude/KNOWN_ISSUES.md`)
forbid any runtime CDN — **all font assets are bundled locally**.

## Requirement (the why)

The desktop app is functionally complete for Phase 8's shipped tracks (chat, citations, settings
sandbox, provider switch, conversation history, library browser, A/B compare) but reads as a
developer tool: the default system-sans everywhere, a generic blue accent, and emoji glyphs standing
in for iconography. The user asked for a full **visual identity** — not a token re-tint, but a
deliberate look that matches what the product *is*: a scholarly, local-first research assistant that
gives grounded answers over the user's own library. The grill (below) widened the ask from a
token-retheme to a phased visual-identity program and locked the direction to **paper & ink**.

## Grill ledger (2026-07-13)

All 11 forks resolved, none parked. (Recorded verbatim from the grill session; the baton entry of
2026-07-13 carries the same ledger.)

| # | Branch | Resolution | Deciding reason |
|---|--------|-----------|-----------------|
| 1 | Ambition: token re-tint vs. full visual identity | **Full visual identity.** | User widened it past the token-retheme recommendation — they want a real identity, phased so it ships safely. |
| 2 | Direction / mood | **Paper & ink** — warm ivory light / warm charcoal dark, scholarly. | Matches the product: grounded research over the user's own documents, not a chat toy. A warm neutral reads as "paper", not "app chrome". |
| 3 | Typefaces | **Spectral (serif) + Inter (sans)**, local woff2, **no CDN**. | Spectral is a screen-legible scholarly serif with true italics; Inter is the reference UI sans. Local woff2 respects the offline/proxy/local-first constraint. |
| 4 | Serif reach — where does the serif apply? | **Reading surfaces**: answers, library chunks, source excerpts, content headings. **Chrome stays sans** (buttons, labels, sidebar, settings form). Reopens to **headings-only** if the full reach feels heavy in V1. | The serif earns its keep on prose the user *reads*; on dense chrome it costs legibility. A clean seam: serif = content, sans = controls. |
| 5 | Accent color | **Deep indigo.** | A single call-to-action signal that reads as ink-adjacent (not a generic SaaS blue) and does **not** collide with the warn (amber) / ok (green) semantic pair. |
| 6 | Icons | **Lucide inline SVGs** replace **all chrome emoji glyphs**. | One coherent, stroke-consistent icon set instead of emoji that render differently per-OS. Inline SVG = no icon-font, no CDN, `currentColor`-themable. |
| 7 | Elevation | **2 shadow tokens** (`--shadow-1` resting card, `--shadow-2` raised drawer/panel). | Two steps is enough to express the drawer-over-content depth already in the app; more is noise. |
| 8 | Motion | **Extend existing patterns** (the Settings/SourcePanel fly+fade, reduced-motion collapse). | The app already has a consistent, reduced-motion-aware transition vocabulary; V1 reuses it, invents nothing. |
| 9 | Layout scope | **V2, not V1:** header/wordmark, empty states, spacing/type scale, ~70ch reading measure. **Shell topology is OUT entirely** (`sidebar │ main │ drawer` stays as verified). | Don't re-risk verified layout/behavior in a look pass. V1 is skin-deep-safe (tokens/fonts/icons); structural rhythm is a separate, later phase. |
| 10 | Product name | **`doc_assistant` kept.** Display-name rename declined. | Out of scope for a look pass; renaming touches packaging, docs, and identity beyond CSS. |
| 11 | Phasing | **V1 tokens+fonts+icons → V2 layout+wordmark+empty-states+measure → V3 app-icon+branding+audit.** Stop-early after V1 allowed. | Ships the safe, high-signal skin first (fully reversible, no behavior change); defers layout rhythm and asset/branding work to their own phases. |

**Asserted (not asked — one defensible answer, no live trade-off):** keep the existing
`data-theme`-attribute-over-media-query theme mechanism (U1) unchanged; keep the warn/ok semantic
token pair (only re-tinted to the warm palette); backend-embedded content emoji (the `🧪`/`🖥`/`🔎`/`📄`
glyphs produced inside answer/provenance **markdown** by `src/doc_assistant/chat_controller.py` /
`commands.py`) are **content, not chrome** — replacing them would put icon concerns in the library
layer (violates the thin-shell rule) and is deferred out of the icon pass.

---

## Direction — paper & ink

A low-chroma neutral field (white with a whisper of ivory in light, warm charcoal in dark) with a single deep-indigo accent
for affordance, warm amber/green retained for the warn/ok semantic pair, a scholarly serif on prose
and a clean sans on controls. Nothing loud; the documents are the subject.

### Token contract (V1)

The V1 palette **re-keys the existing token set** in `apps/desktop/src/app.css` — same variable names
(so every component recolors for free), warmed values, plus new font-stack and shadow tokens. The four
existing blocks are all updated in lockstep: `:root` (light default) · `:root[data-theme='dark']` ·
`:root[data-theme='light']` · `@media (prefers-color-scheme: dark) :root:not([data-theme])`. Manual
theme still wins over the OS media query (U1 mechanism unchanged).

**Light (paper — white / ivory).** *Amended 2026-07-13 (user feedback on the live V1): the first cut's
warm ivory `#f7f3ea` read too beige — pulled to a white page with a whisper-ivory surface layer. White-
forward, still faintly warm (not clinical), indigo accent + ink text unchanged.*

| Token | Value | Role |
|-------|-------|------|
| `--bg` | `#ffffff` | Page — white |
| `--surface` | `#f8f7f3` | Raised surface (cards, header) — whisper ivory |
| `--surface-2` | `#efece5` | Deeper surface (user bubble, code, chips-neutral) |
| `--border` | `#e5e1d8` | Hairlines — soft warm grey |
| `--fg` | `#23201b` | Ink — near-black |
| `--fg-2` | `#6b6559` | Muted ink — secondary text |
| `--accent` | `#4a3fa6` | Deep indigo — the one affordance signal |
| `--accent-fg` | `#ffffff` | Text on accent |
| `--warn-fg` `--warn-bg` `--warn-border` | `#8a5300` / `#fbeecb` / `#e6c987` | Warn (amber) |
| `--ok-fg` `--ok-border` | `#2f6b3d` / `#9cc9a6` | OK (green) |

**Dark (warm charcoal / paper-white):**

| Token | Value | Role |
|-------|-------|------|
| `--bg` | `#1b1813` | Warm charcoal (not blue-black) |
| `--surface` | `#23201a` | Raised surface |
| `--surface-2` | `#2e2a22` | Deeper surface |
| `--border` | `#3b362c` | Hairlines |
| `--fg` | `#ece5d6` | Paper-white ink |
| `--fg-2` | `#a79e8b` | Muted |
| `--accent` | `#9a8ff0` | Indigo, lightened for dark-field contrast |
| `--accent-fg` | `#1b1813` | Text on accent |
| `--warn-fg` `--warn-bg` `--warn-border` | `#efc06a` / `#2f2712` / `#5a4a22` | Warn (amber) |
| `--ok-fg` `--ok-border` | `#79c98d` / `#345f3f` | OK (green) |

**Font-stack tokens (both themes, mechanism-independent):**

- `--font-sans: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;`
- `--font-serif: 'Spectral', 'Iowan Old Style', Georgia, Cambria, 'Times New Roman', serif;`

Body/`:root` font is `--font-sans`. Reading surfaces opt into `--font-serif` (see below). The stack
degrades gracefully: until the Spectral/Inter woff2 load, prose renders in Georgia / system-serif and
chrome in system-sans — the serif-vs-sans *distinction* is visible immediately; the branded faces slot
in when the `@font-face` binaries are present.

**Shadow tokens (per-theme — dark uses deeper opacity):**

- `--shadow-1` — resting card. Light: `0 1px 2px rgba(42,38,32,.06), 0 1px 3px rgba(42,38,32,.05)`.
  Dark: `0 1px 2px rgba(0,0,0,.35), 0 1px 3px rgba(0,0,0,.30)`.
- `--shadow-2` — raised drawer/panel. Light: `0 10px 30px rgba(42,38,32,.14)`. Dark: `0 10px 30px rgba(0,0,0,.55)`.

*(Exact hex may be nudged ±a shade during build for contrast; the AA target and the warm/indigo/serif
direction are the lock, not the last digit.)*

### Serif reach (V1)

Apply `--font-serif` to the reading surfaces, chrome stays `--font-sans`:

- **Answers** — `Markdown.svelte` `.md` container (covers `Turn.svelte` + `ReadonlyTurn.svelte`, which render answers through it).
- **Library chunk text** — `LibraryBrowser.svelte` parent/child chunk bodies.
- **Source excerpts** — `SourceCard.svelte` `.excerpt`, `SourcePanel.svelte` chunk detail.
- **Content headings inside those surfaces** — `.md h1/h2/h3`.

Chrome (buttons, form labels, sidebar rows, settings fields, header meta, provenance/claim controls)
stays sans. If the full reach reads heavy in the live V1 preview, fall back to **headings-only** serif
(fork #4's documented reopen) — a one-line scope trim, not a redesign.

### Icons (V1)

New `apps/desktop/src/lib/Icon.svelte` — a single component rendering **Lucide** inline SVGs
(`stroke="currentColor"`, `fill="none"`, `stroke-width="2"`, rounded caps/joins, `aria-hidden` by
default, `size` prop). It replaces every chrome emoji glyph in the `.svelte` templates:

| Glyph | Where | Lucide icon |
|-------|-------|-------------|
| `☰` | `App.svelte` hamburger | `menu` |
| `⬇` | `App.svelte` Export | `download` |
| `⚙` | `App.svelte` Settings | `settings` |
| `←` | `App.svelte` back-to-current | `arrow-left` |
| `↻` | `Sidebar.svelte` New chat | `rotate-ccw` |
| `●` | `Sidebar.svelte` current-chat dot | CSS-drawn dot (glyph removed; not an icon) |
| `✕` | `Settings` / `SourcePanel` / `CompareCard` close | `x` |
| `✓` | `ClaimReview` accept, `Settings` indexed-ok | `check` |
| `⚠` | `ClaimReview` / `SourceCard` / `Turn` / `Settings` warnings | `triangle-alert` |

Buttons that already carry an `aria-label` keep it (the icon is decorative, `aria-hidden`); buttons
with visible text (`⬇ Export`, `↻ New chat`, `✓ Accept`, `← Back`) keep the text as the accessible
label and place the icon before it.

**Out of the icon pass:** the backend-embedded content emoji in streamed answer/provenance markdown
(`🧪`/`🖥`/`🔎`/`📄`) — `src/` content, thin-shell boundary (see grill "asserted").

---

## Font binaries — bundling mechanism (RESOLVED 2026-07-13)

The direction locks **local woff2, no CDN**; the delivery mechanism was resolved with the user at V1
build → **option 1, vendored + committed.** Four latin-subset woff2 (OFL, from the `@fontsource` /
`@fontsource-variable` packages) are vendored under `apps/desktop/src/assets/fonts/`
(`spectral-400`, `spectral-400-italic`, `spectral-600`, `inter-variable`; ~115 KB total) and loaded via
hand-written `@font-face` in `apps/desktop/src/lib/fonts.css` (imported first in `main.ts`, before
`app.css`). Truly offline, committed, zero runtime dependency — the shipped app carries its own fonts.
`font-display: swap` paints immediately in the fallback and upgrades when each face is ready.

*Fetch note (this box): jsDelivr download failed schannel revocation (`CRYPT_E_NO_REVOCATION_CHECK`, the
corporate-proxy quirk) until `curl --ssl-no-revoke` — the same on-proxy TLS workaround pattern the repo
already documents elsewhere.* Rejected: `@fontsource` npm packages (assets would live in gitignored
`node_modules`, not committed — worse fit for a local-first identity than vendoring); deferring (the user
chose to land it now).

---

## Phasing & per-phase DoD

### V1 — tokens + fonts + icons  *(this sprint — SPRINT-016)*

**DoD.** `app.css` re-keyed to the paper & ink palette (all four theme blocks) + font-stack + 2 shadow
tokens; `Icon.svelte` created and every chrome emoji glyph replaced; `--font-serif` applied to the
reading surfaces, chrome on `--font-sans`; Spectral + Inter loaded locally via `@font-face` from vendored
woff2 (resolved mechanism section). `svelte-check` 0 errors; both themes styled and legible (AA body
contrast); manual theme
still overrides the OS media query; no horizontal overflow at mobile width; icons carry correct
`aria-hidden`/label semantics; reduced-motion still respected. Preview-harness-verified live, $0/offline
(chrome + palette + icons on the app shell; serif on a real reading surface — the Library browser is
$0). Byte-level behavior unchanged (a look pass changes no logic, no wire type, no backend).

### V2 — layout rhythm + header/wordmark + empty states + reading measure

Header/wordmark treatment; a coherent spacing + type scale; restyled empty/first-run states; a ~70ch
`max-width` measure on prose. **Shell topology stays put** (fork #9). Separate sprint; separate DoD.

### V3 — Tauri app icon + branding + polish audit

App icon / installer branding assets; a final cross-screen polish audit (both themes, mobile, all
shipped surfaces). Separate sprint.

## Out of scope (whole pass)

- **Shell topology** — `sidebar │ main │ drawer` is verified; not re-risked (fork #9).
- **Product rename** — display name stays `doc_assistant` (fork #10).
- **Backend content emoji** — `🧪`/`🖥`/`🔎`/`📄` in streamed markdown stay (thin-shell boundary).
- **Any behavior / logic / wire-type / locked-setting change** — a look pass touches CSS + templates only.
- **New motion vocabulary** — existing fly/fade + reduced-motion patterns are reused, not extended (fork #8).

## Verification

Frontend-only; verify via `svelte-check` (0 errors) + the preview harness on `desktop-ui` (:1420) — and
`api` (:8001, truststore + `HF_HUB_OFFLINE` per `.claude/launch.json`) for the $0 Library reading-surface
check. No pytest surface (no `src/`/API change). No live paid API call. Screenshots both themes + mobile.
Per-feature gate: `docs/ui-checklist.md` §4.
