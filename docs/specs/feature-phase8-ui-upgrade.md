# Spec — Phase 8 UI/UX upgrade: Settings surface, chat bubble layout, citation side panel

**Status:** 📋 **design-locked for U1/U2/U3/U1b** (grilled 2026-07-10 — see Grill ledger below); **U1c
scoped but NOT designed** (needs its own ADR before it's buildable). Roadmap PRs **U1**/**U1b**/**U1c**
(Settings), **U2** (chat layout), **U3** (citation side panel).
**Owner of execution:** Claude Code (Svelte/TS + the small backend surface U1/U1b need). One PR per
track — they touch disjoint files and ship independently. **Locked build order: U2 → U3 → U1 → U1b →
U1c** (grilled 2026-07-10 — engineering order over the request's listed order: the two fast,
self-contained frontend wins ship first; U1c is gated behind its own ADR regardless of order).
**Pattern reference:** thin-shell rule (`apps/` render, no logic; root `CLAUDE.md`); the drawer/scrim
pattern already shipped in `Settings.svelte` (fly transition, focus trap, Esc-to-close) is reused
verbatim for the new citation panel (U3) rather than inventing a second modal pattern.

**Requirement (the why).** User audit of the running desktop app (2026-07-10, against the real
30,882-chunk corpus) found three concrete gaps against what a modern RAG chat UI should do, plus a
request to round up everything else already logged as UI debt so it can be sequenced together instead
of trickling in one baton entry at a time.

## Grill ledger (2026-07-10)

| # | Branch | Resolution | Deciding reason |
|---|---|---|---|
| 1 | Settings scope: does "all possible options" widen beyond ADR-010's locked 3 knobs? | **Yes, all of it — split into 3 tracks.** U1 stays exactly as ADR-010 locked it (unchanged). **U1b** (new): the two ADR-010 "must revisit" niche knobs (`EPISTEMICS_MARKERS_ENABLED`, `REVIEWER_EVIDENCE_CHARS`) — small ADR-010 amendment, in scope here. **U1c** (new): provider/API-key management — different risk class (secrets, requires rebuild), needs its own ADR, **out of scope of this spec**. | User: "we need all of this... you can separate them if you prefer to keep U1 as it is." Splitting keeps U1 shippable without reopening ADR-010's already-accepted contract, and keeps the higher-risk secrets/provider decision from riding on this grill pass. |
| 2 | Build order: request's listed order (Settings → bubble → panel) vs. engineering order | **Engineering order: U2 → U3 → U1 → U1b → U1c.** | User confirmed explicitly, unprompted by the scope change: ship the two fast, self-contained wins first; Settings (now the biggest track, spanning 3 sub-tracks) is properly sequenced behind them. |
| 3 | Malformed-citation fallback: does "hidden by default" have an exception for bad citations? | **Keep the fallback** — sources render inline when `citation_note_md !== ''`. | User confirmed the spec's proposal: hiding sources is worst exactly when the model cited badly and the reader most needs to check by hand. |
| 4 | User-bubble color | **Neutral `var(--surface-2)`**, not accent-tinted. | User confirmed the spec's proposal: keeps `--accent` as the app's one call-to-action signal; matches the existing low-saturation aesthetic. |
| 5 | Test runner for `theme.ts` / the citation linkifier | **No `vitest` — preview-harness verification only.** | User confirmed: both functions are short and inspectable; a new JS test toolchain (deps, config, an unanswered cpc-gate question) is more ceremony than 2 functions warrant. Revisit if the linkifier grows real edge-case surface. |
| — | Mechanical (asserted, not asked — one defensible answer, no pushback surfaced) | Theme via `localStorage`, not a backend setting · single citation panel, swap-on-click (matches the user's own "like artifacts in Claude" reference) · DOM text-node-walk linkifier · bubble `max-width: min(72%, 640px)`. | Each has exactly one defensible answer already argued in its Decision section below; none is a live trade-off. |

---

## U1 — Settings: expose every honest option + a real (manual) dark mode

**Context.** Two separate gaps, bundled into one track because they both live in `Settings.svelte`:

1. **The RAG sandbox knobs are already fully designed and locked — just not built.**
   `docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md` (accepted 2026-07-09) +
   `docs/specs/feature-rag-sandbox.md` spec exactly "expose all possible options": `TOP_K`,
   `SYNTHESIS_MODE`, `USE_MULTI_QUERY` become live, session-scoped, non-persistent overrides; every
   other knob (`CANDIDATE_K`, retrieval weights, reranker, provider/model, chunk sizes,
   `USE_PARENT_CHILD`) renders **read-only with the specific reason it can't be live** (rebuild /
   re-ingest / measured-inert). U1 **is** that build — see the spec for the full contract
   (`src/doc_assistant/chat_controller.py::RagOverrides`, `pipeline.py` request-scoped multi-query,
   `apps/api/models.py::RagOverrides`, `Settings.svelte` sandbox section). Not re-specced here; U1
   just adopts it as-is.
2. **The current "Engine (read-only)" section under-discloses.** `Settings.svelte:229-242` renders only
   4 of the 8 fields already sitting in the `Settings` TS type (`types.ts:54-70`) — `provider`,
   `model`, `embedding_model`, `top_k`, `candidate_k`, `synthesis_mode` show; **`retrieval_weights`,
   `use_parent_child`, `parent_chunk`, `child_chunk` are fetched and silently dropped.** "All possible
   options" means finishing that disclosure, not just adding the sandbox sliders — a locked knob you
   can't see isn't a documented limit, it's a blind spot.
3. **Dark mode exists but isn't a setting.** `app.css:28-44` already ships a full dark palette, but it's
   wired to `@media (prefers-color-scheme: dark)` only — there is no in-app control, so a user whose OS
   is light-mode-by-policy (or who just prefers dark regardless of OS) has no lever. "Nightly mode" =
   make it a tri-state user choice: **System / Light / Dark**.

### Decision — theme toggle: `data-theme` attribute overrides the media query, persisted client-side

**Options considered.**
1. *CSS-only, OS-driven (status quo).* Zero code, but not a setting — rejected, that's the gap.
2. *Persist theme as a backend setting (`POST /api/settings`).* Rejected: theme is a pure rendering
   preference with zero bearing on retrieval/synthesis quality — routing it through the governed
   settings endpoint (built for `source_dir`, soon the RAG overrides) mixes a cosmetic client
   preference into the one surface ADR-010 just drew a careful non-persistence line around. Also
   makes theme machine-specific data live in `library.db` for no reason.
3. **Client-only, `localStorage` + a `data-theme` attribute on `<html>` (chosen).** Tauri's webview is
   a real browser context — `localStorage` persists across restarts same as any web app. A tri-state
   value (`'system' | 'light' | 'dark'`) is read on boot; `'light'`/`'dark'` set
   `document.documentElement.dataset.theme`, `'system'` clears it (falls through to the existing media
   query). CSS gains two override blocks that win over the media query by attribute-selector specificity
   plus source order:
   ```css
   :root[data-theme='dark']  { /* the existing dark block's declarations, unconditional */ }
   :root[data-theme='light'] { /* the existing light values, unconditional */ }
   @media (prefers-color-scheme: dark) { :root:not([data-theme]) { /* dark values */ } }
   ```
   No flash-of-wrong-theme: the attribute is set synchronously in `main.ts` before Svelte mounts, from
   `localStorage`, before first paint.

**Consequences.** Free (no backend touch), reversible, matches how every other cross-session desktop
preference in this app that isn't retrieval-relevant would be handled if one existed (there isn't a
precedent yet — this is the first purely-cosmetic persisted setting). Correctly does **not** go through
`RagOverrides` (ADR-010's non-persistence wall is specifically for the *retrieval-quality-governed*
knobs; theme was never part of that governance and forcing it through the same non-persistent channel
would just make dark mode forget itself on every restart, which is a worse product than today's
OS-only behaviour).

### Contracts

- **`apps/desktop/src/lib/theme.ts` (new, ~20 lines)** — `type Theme = 'system' | 'light' | 'dark'`;
  `getTheme()/setTheme(t)` wrapping `localStorage.getItem/setItem('theme', …)`; `applyTheme(t)` sets/
  clears `document.documentElement.dataset.theme`. Pure DOM + storage, no framework dependency — callable
  from `main.ts` before `mount()`.
- **`apps/desktop/src/main.ts`** — call `applyTheme(getTheme())` as the first line, before mounting `App`.
- **`apps/desktop/src/app.css`** — restructure the existing dark block per the Decision above (no new
  colors, same two palettes — just re-keyed off `[data-theme]` in addition to the media query).
- **`apps/desktop/src/lib/Settings.svelte`** — new **"Display"** section (place above "Your documents",
  it's the most-reached-for setting): a 3-way segmented control (`System / Light / Dark`), calling
  `setTheme` + `applyTheme` on change; no `busy` state, no backend round-trip, takes effect immediately.
  Extend the existing **"Engine (read-only)"** section to also render `retrieval_weights` (labeled
  *"inert on the shipped top-K by construction (measured)"* per ADR-010 Decision 3, not silently
  omitted), `Parent/child chunk sizes`, `Parent-child retrieval` (on/off) — closing gap 2 above. This
  slots directly above/beside the RAG-sandbox section ADR-010 adds in the same file.

### Guard tests
- `theme.ts`: **no `vitest`** (grill ledger #5) — verify via the preview harness instead:
  `preview_eval` calling `applyTheme('dark')` and reading `document.documentElement.dataset.theme`;
  same for `'system'` clearing it; `getTheme()` round-trips by setting `localStorage` via
  `preview_eval` then reloading.
- Frontend: `svelte-check` clean. Preview-harness verification (this box's known-good path — snapshots +
  synchronous evals, screenshots are flaky here per `.claude/KNOWN_ISSUES.md`): toggle each of the 3
  states, `preview_inspect` the computed `background-color` on `<body>` to confirm it actually flips
  (not just that the attribute is set).
- Manual: restart the dev server after picking Dark — confirm no flash of light theme on reload.

### Definition of done
- Theme is a real tri-state setting, persists across restarts, never touches the backend.
- Every field in the `Settings` TS type renders somewhere in the panel (read-only ones say why).
- ADR-010's sandbox build lands as specced in `feature-rag-sandbox.md` (its own DoD applies unchanged).

---

## U1b — Settings: the two niche knobs ADR-010 deferred (`EPISTEMICS_MARKERS_ENABLED`, `REVIEWER_EVIDENCE_CHARS`)

**Context.** ADR-010's "Must revisit" section named these two explicitly: both are query-time (cheap,
no rebuild) but were left out of v1 as "cosmetic / niche," pending real use of the base sandbox.
Resolved 2026-07-10 (grill pass on this spec): include them now, as a small amendment rather than
reopening ADR-010's core decision — see the amendment note added to
`docs/decisions/ADR-010-rag-sandbox-nonpersistent-overrides.md`.

**Scope, kept intentionally small:**
- `EPISTEMICS_MARKERS_ENABLED` becomes a fourth field on `RagOverrides` (`chat_controller.py`,
  `apps/api/models.py`) — same non-persistent, request-scoped threading as the other three (ADR-010
  Decision 4 applies unchanged: no module-global mutation). Settings gets an on/off switch: "Show
  contested/superseded chips" in the same sandbox section U1 builds.
- `REVIEWER_EVIDENCE_CHARS` becomes a fifth field, an integer input (bounds TBD at build time — read
  the reviewer prompt's current usage to find a sane range before exposing a raw number field; do not
  guess a range here).
- Both get the same provenance-reflects-effective-value treatment as the original three (ADR-010
  Decision 5) and the same isolation guard test (a turn with these set must not leak into the next).

**Depends on:** U1 landing first (extends the same `RagOverrides` dataclass and the same Settings
sandbox section — building U1b before U1 exists would mean threading a 4-field override type through
code U1 hasn't written yet).

**Files owned:** the same files as U1's `feature-rag-sandbox.md` contract list, extended by two fields
each — no new files.

**Definition of done:** both knobs override per-turn exactly like the original three; isolation guard
test covers all five fields, not just the original three; Settings sandbox section shows both with the
same "session only" banner.

---

## U1c — Settings: provider / API-key management (SCOPED, NOT DESIGNED — needs its own ADR)

**Context.** Flagged during the 2026-07-10 grill pass as part of "all possible options," but explicitly
carved out rather than designed here: switching the LLM provider (Anthropic ↔ Ollama) is
construction-time (needs the pipeline rebuilt, per ADR-010's own construction-time/query-time split),
and managing an API key touches secrets storage — both a materially different risk class than anything
else in this spec (which is either pure-frontend or already-governed request-scoped overrides). This
needs its own options-and-trade-offs pass, not a paragraph inside a UI spec.

**Known open questions for that future ADR (not answered here):** where does a user-entered key
persist (`.env`? OS keychain? `library.db`?) and how does that interact with the existing
`load_dotenv(override=True)` precedence (`config.py:14`) already documented as intentional; does a
provider switch require an app restart (rebuild the FastAPI sidecar's pipeline) or can `main.py`'s
lifespan rebuild it live; what happens to an in-flight turn during a provider switch; does the frozen
PyInstaller sidecar's `sys.frozen` OS-trust-store branch (KI-10, `llm.os_trust_http_client()`) change
per-provider.

**Depends on:** its own ADR (`docs/decisions/ADR-011-*` or next available number) — run
`architecture-decision` on this specifically before it's buildable. Not sequenced with a build order
here since it isn't spec'd yet; ROADMAP carries it as `planned, needs ADR`.

---

## U2 — Chat layout: right-aligned user bubble, RAG answer stays full-width

**Context.** Today `Turn.svelte:37-41` renders the user's question as a plain full-width block — a
small uppercase "You" label, then a paragraph, no visual container, flush with the answer below it.
Every mainstream chat UI (ChatGPT, Claude.ai, Gemini) marks the *user* turn as a bounded, right-aligned
bubble and leaves the *assistant* turn as an unbounded, left-aligned block — the asymmetry is what
reads as "chat" rather than "alternating paragraphs." The RAG answer is explicitly **not** getting a
matching bubble: it already renders as a full-width block (`Turn.svelte:43-69`, no wrapping card), and
the requirement is to keep that — sources, claims, provenance, and figures need the room a bubble would
constrain.

### Decision — bubble is a `<div>` styled via CSS only, no new component

This is layout, not a new interaction — doesn't warrant a new component or state. `Turn.svelte`'s `.you`
div gets: `align-self: flex-end` (with `.turn` as a flex column, replacing today's implicit block flow),
`max-width: min(72%, 640px)` so it never spans the full column even in a wide window, `background:
var(--surface-2)`, `border-radius: 14px` with one squared corner (`border-bottom-right-radius: 4px`,
the standard "tail" cue), `padding: 0.55rem 0.85rem`. **Color choice:** neutral `var(--surface-2)`, not
`var(--accent)` — the accent color is already the app's one call-to-action signal (Send button,
primary buttons); tinting every user message the same blue would dilute that and fight the existing
muted aesthetic (`.chip`/`.usage`/`.hint` are all low-saturation). **Confirmed 2026-07-10 (grill
ledger #4)** — neutral, not accent-tinted.

### Contracts
- **`apps/desktop/src/lib/Turn.svelte`** — `.turn` becomes `display: flex; flex-direction: column;`;
  `.you` gains `align-self: flex-end`, `max-width`, bubble background/radius/padding (above); the "You"
  label moves inside the bubble (small, muted, top-left of the bubble) or stays above it right-aligned —
  decide by look, not a hard requirement. `.assistant` is **untouched** — no wrapper, no max-width, no
  background — it already spans the full `.turn` width via normal block flow.
- No change to `App.svelte`, `SourceCard.svelte`, `Provenance.svelte`, or any wire type — this is CSS
  + a template-structure tweak inside one component.

### Guard tests
- `svelte-check` clean.
- Preview harness: `preview_inspect` the `.you` element's computed `max-width` and
  `justify-content`/margins to confirm it's actually right-bounded, not just visually eyeballed;
  `preview_screenshot` (or snapshot if screenshots stay flaky) at desktop width to confirm the answer
  block still spans full width beside a narrow user bubble; resize to `mobile` preset to confirm the
  bubble's `min(72%, 640px)` cap doesn't overflow a narrow viewport.

### Definition of done
- User question renders as a right-aligned, width-capped bubble; RAG answer renders unchanged
  (full-width, no bubble). Both themes (light/dark, from U1) render the bubble with sufficient contrast
  against `--bg`.

---

## U3 — Citation side panel: click `[n]` → slide-over chunk detail, hidden by default

**Context.** The LLM answer text already contains literal inline citation markers matching
`_CITATION_RE = re.compile(r"\[(\d+)\]")` (`src/doc_assistant/synthesis.py:25`) — e.g. "...outperforms
BM25 alone [2]." — which `Markdown.svelte` currently renders as **inert text** (`marked.parse` has no
square-bracket-only syntax, so `[2]` passes through untouched). Separately, `Turn.svelte:48-54` always
renders **every** source as a stacked `SourceCard` grid below the answer, regardless of whether the
reader wants to see it — the exact "chunks appear below the prompt by default" behavior the user wants
removed, in favor of the old Chainlit-style on-demand side panel (and the closest in-app precedent is
`Settings.svelte`'s own slide-over drawer).

### Decision — reuse the Settings drawer pattern; citations become buttons via a text-node walk, not raw-HTML regex

**Options considered for making `[n]` clickable.**
1. *Regex-replace on the raw markdown string before `marked.parse()`.* Simple, but `marked` may
   re-escape or reflow the injected markup depending on surrounding markdown (e.g. inside emphasis),
   and the result depends on `marked`'s escaping rules for injected HTML in source position — fragile.
2. *Regex-replace on the final HTML string.* Same risk in reverse: a naive `\[(\d+)\]` regex over full
   HTML can match inside an attribute value or inside a `<code>` span where a literal `[2]` is a code
   example, not a citation — Markdown.svelte's own comment already flags this file as trusting content
   verbatim (`{@html html}`, no DOMPurify), so "close enough" string regex is how this file already
   operates, but citation-in-code-span is a real enough case (technical corpus, users will paste code)
   to not risk it.
3. **DOM text-node walk after render (chosen).** After `marked.parse()` produces the HTML and it's in
   the DOM (Svelte's `{@html}` + an `$effect` keyed on the rendered content), walk text nodes with
   `document.createTreeWalker(el, NodeFilter.SHOW_TEXT)`, **skipping any node whose ancestor is
   `<code>`/`<pre>`** (so a code example containing `[2]` is left alone), and replace matches of
   `/\[(\d+)\]/g` with a `<button class="citation" data-n="…">[n]</button>`. Slightly more code than a
   regex, but correctly leaves code blocks alone and never touches HTML structure/attributes because it
   only ever mutates text nodes it already holds a reference to.

**Options considered for the panel itself.**
1. *Inline expand (accordion under the clicked citation).* Cheapest, but the user explicitly asked for
   the Chainlit/Claude-artifacts pattern — a fixed side panel — not an inline expand; rejected as not
   what was asked.
2. **Slide-over panel, same mechanics as `Settings.svelte` (chosen).** Reuse: `fly`-in from the right +
   `fade` scrim, `prefers-reduced-motion` → instant swap, Esc-to-close, focus trap on Tab
   (`Settings.svelte:113-136` is copy-adaptable almost verbatim). One panel open at a time, app-wide —
   clicking a different `[n]` (even in a different turn) swaps its content rather than stacking panels,
   matching how the Settings gear behaves today (single drawer, not a stack). Consistent interaction
   vocabulary beats a bespoke citation-panel animation.

**Default-hidden sources.** `Turn.svelte` drops the always-rendered `<div class="sources">` grid
entirely. A source is only ever seen via its citation panel, **except**: if `result.sources.length` and
the answer contains zero recognized `[n]` markers (the audit-flagged malformed/no-citation case,
`chat_controller.py`'s `citation_note_md`), fall back to showing the source list inline — otherwise a
malformed-citation answer would hide its sources with no way to reach them. This mirrors the existing
`citation_note_md` "⚠ Citation check" affordance that already surfaces exactly this failure mode.

### Contracts

- **`apps/desktop/src/lib/SourcePanel.svelte` (new)** — structurally `Settings.svelte`'s scrim+panel
  (`role="dialog"`, `aria-modal`, focus trap, Esc) with `Settings`'s form content replaced by one
  `<SourceCard source={...} />` (existing component, unchanged) plus a small header showing "Source
  [n]" and a close button. No network calls — it's handed a `SourceView` it already has in memory.
- **`apps/desktop/src/lib/Markdown.svelte`** — add an `onCitationClick?: (n: number) => void` prop; the
  text-node-walk citation-linkifier (above) runs in an `$effect` after `{@html html}` renders, attaching
  one delegated `click` listener on the container root (`el.addEventListener('click', …)` reading
  `event.target.closest('.citation')`'s `data-n`) rather than one listener per button. `.citation`
  styling: no default border/background (reads as normal bracketed text), `color: var(--accent)`,
  `cursor: pointer`; an **active** state (`.citation.active` — the clicked one, while its panel is open)
  gets `background: var(--accent); color: var(--accent-fg); border-radius: 3px` — the "colored when
  clicked" behavior asked for. Active state needs the currently-open `n` passed back in as a prop to
  compare against during the walk.
- **`apps/desktop/src/lib/Turn.svelte`** — remove the unconditional `sources` grid; add the
  malformed-citation fallback (above) gated on `!result.sources.every-n-cited` (exact predicate: cross
  `result.citation_note_md !== ''` — the backend already computed this via `audit_citations`, cheaper
  than re-deriving citation coverage client-side). Pass `onCitationClick` through to `Markdown`, sourced
  from a callback prop `Turn` receives from `App`.
- **`apps/desktop/src/App.svelte`** — owns `let activeCitation = $state<{ turnId: number; n: number } |
  null>(null)`; passes `onCitationClick={(n) => (activeCitation = { turnId: t.id, n })}` into each
  `<Turn>`; renders `<SourcePanel>` once at the top level when `activeCitation` is set, resolving the
  `SourceView` by looking up `turns.find(t => t.id === activeCitation.turnId)?.result?.sources.find(s
  => s.n === activeCitation.n)`. Same ownership shape as `showSettings` today — no new state pattern.

### Guard tests
- Citation linkifier: **no `vitest`** (grill ledger #5) — verify via the preview harness: text
  `"See [1] and [12]."` inside a `<code>` block is left alone; the same text outside code becomes two
  `.citation` buttons with `data-n="1"`/`"12"`.
- Preview harness: click a `[n]` in a live turn → `preview_snapshot` confirms the panel opens with the
  right citation text; the clicked button carries `.active`; Esc closes it; clicking a second `[n]`
  swaps the panel content without a second panel appearing; a turn with zero sources rendered inline by
  default (confirm the grid is gone) alongside one triggering the malformed-citation fallback (confirm
  it still shows).

### Definition of done
- No source cards render by default; a source is reachable only via its `[n]` (or the malformed-citation
  fallback).
- `[n]` in the answer is a clickable, accent-colored control; clicking opens a slide-over panel with
  that source's citation/excerpt/markers/figure; the clicked `[n]` visibly highlights while its panel is
  open.
- Code spans containing a bracketed number are never linkified.
- Panel reuses `Settings.svelte`'s a11y mechanics (focus trap, Esc, `aria-modal`) — not a second,
  divergent modal implementation.

---

## Related backlog — already logged elsewhere, not re-specced here

Surfaced while grounding this spec in the current ROADMAP/specs/known-issues, for sequencing alongside
U1–U3 rather than being rediscovered later:

| Item | Where it's tracked | Status |
|---|---|---|
| **RAG sandbox knobs** (the live part of U1) | ADR-010 + `docs/specs/feature-rag-sandbox.md` | design-locked, ready to build — U1 adopts it directly |
| **A/B compare sandbox** (run locked defaults vs. override side by side) | ADR-010 option 4 | recorded north-star, phased *after* the U1/ADR-010 basic sandbox — real cost implication (≈2× per compared turn) |
| **S1 — Selective ingestion backend** (`SourceFile` registry, `--files`/`--dry-run`, `GET/PATCH /api/sources`) | `docs/specs/feature-selective-ingestion.md` | DRAFT, not yet grilled/locked |
| **S2 — Selective ingestion UI** (Tauri sources panel: status chips, select-by-status/type, exclude toggle) | `docs/specs/feature-selective-ingestion.md`, ROADMAP row S2 | planned, blocked on S1 |
| **In-app PDF source viewer** | `docs/specs/pr-m3-tauri-frontend.md` "Out of scope" | deferred "later refinement" at M3, never scheduled since |
| **Styled table rendering** in the answer/source view | `docs/specs/pr-m3-tauri-frontend.md` "Out of scope" | same — deferred, never scheduled |
| **Rich marker UI** — hover a `contested`/`superseded_trend` chip to see the corroborating/contradicting docs, not just the bare chip | `docs/specs/pr-m1-epistemics-markers.md` "Out of scope" (tagged "PR-M3") | deferred at M1, **not actually built** in M3 — currently just a static chip (`SourceCard.svelte:17-21`) |
| **Live-UI smoke test of the marker chips on the real corpus** | `.claude/KNOWN_ISSUES.md` KI-15 (resolved) follow-up | the *backend* label-matching bug (KI-15) is fixed and chips now fire on 3,334 real chunks, but nobody has confirmed the chip renders correctly in the live desktop UI since before the fix — a fast add-on once U3 is in, since the fix makes chips finally common enough to see without hunting |
| **Precise PC (parent-child) re-projection for markers** | `docs/specs/pr-m1-epistemics-markers.md` ADR-1 option 2 | the current containment-based marker mapping is a documented coarse approximation; upgrade needs its own attribution-quality validation, backend work not UI |

None of these are pulled into U1–U3's scope — they're independently sequenceable. The two most
natural next-after items, if this spec's three tracks land well, are **the ADR-010 A/B-compare
north-star** (extends U1 directly) and **S1/S2 selective ingestion** (the other half-built Phase-8 UI
item already in ROADMAP).

---

## Build node

**Depends on:** the shipped Tauri/Svelte shell (PR-M0–M5, all done); ADR-010 + `feature-rag-sandbox.md`
for U1's sandbox half. No new backend model beyond what ADR-010 already contracts; U2/U3 touch only
`apps/desktop/**`.

**Files owned:**
- U1: `apps/desktop/src/lib/theme.ts` (new), `apps/desktop/src/main.ts`, `apps/desktop/src/app.css`,
  `apps/desktop/src/lib/Settings.svelte`, plus everything `feature-rag-sandbox.md` already lists.
- U1b: the same files `feature-rag-sandbox.md` + U1 already touch (`chat_controller.py`,
  `apps/api/models.py`, `Settings.svelte`) — extended, no new files. Depends on U1 landing first.
- U1c: not spec'd — its own ADR decides file ownership.
- U2: `apps/desktop/src/lib/Turn.svelte` (CSS/structure only).
- U3: `apps/desktop/src/lib/SourcePanel.svelte` (new), `apps/desktop/src/lib/Markdown.svelte`,
  `apps/desktop/src/lib/Turn.svelte`, `apps/desktop/src/App.svelte`.

**Status:** design-locked for U1/U1b/U2/U3 (grilled 2026-07-10, ledger above — every branch resolved
or explicitly routed). U1c is scoped but intentionally not designed; it needs its own ADR before any
build session picks it up.

### Definition of done (spec-level)
- U1, U1b, U2, U3 each independently gate-clean: `svelte-check` 0 errors; preview-harness-verified per
  their own DoD above (screenshots flaky on this box — snapshots + synchronous evals per
  `.claude/KNOWN_ISSUES.md`).
- No backend/locked-setting change outside what ADR-010 (+ its 2026-07-10 amendment for U1b) already
  contracts — U2 and U3 are frontend-only diffs.
- `docs/ROADMAP.md` Phase 8 row carries U1/U1b/U1c/U2/U3 sub-rows pointing at this spec; one
  `docs/DEVLOG.md` entry per track when built.

## Out of scope (this spec)
- Everything in the **Related backlog** table above.
- **U1c (provider/API-key management)** — scoped above, but needs its own ADR; not buildable from this
  spec alone.
- A JS unit-test runner (`vitest`) — decided against, grill ledger #5.
- Multi-citation-panel stacking / a "pin panel open while scrolling" mode — v1 is one panel, swap-on-click,
  matching the Settings drawer's existing single-instance behavior.
- Changing the RAG-sandbox *knob set* itself beyond U1b's two additions — everything past that is a
  new ADR-010 amendment, not this spec's call.
