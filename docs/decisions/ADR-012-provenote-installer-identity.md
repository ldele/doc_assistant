<!-- status: active · updated: 2026-07-14 · class: append-only -->

# ADR-012 — Provenote is the installer/product identity; `doc_assistant` stays the code identity

- **Status:** accepted
- **Date:** 2026-07-14
- **Deciders:** user (pickipoc123), Claude Code
- **Context refs:** `docs/specs/feature-visual-identity.md` §V3 (V3a rename, V3b app icon) ·
  ADR-002 (Tauri/FastAPI shell) · flag in §V3b that this surface "may warrant its own ADR".

## Context

The visual-identity pass renamed the product to **Provenote** (V3a, committed `181046c`) and then
regenerated the OS app icon from a designed source (V3b). This deliberately did **not** rename the
code: the Python package `doc_assistant`, the `python -m doc_assistant.*` commands, the npm package
name `doc-assistant-desktop`, and — most consequentially — the Tauri sidecar binary
`binaries/doc-assistant-api` all keep the old name. Meanwhile the Tauri **bundle identifier** changed
(`com.doc-assistant.desktop` → `com.provenote.desktop`) and the installer/window/`productName`/icon
now say Provenote.

A future reader hitting `doc-assistant-api` inside a "Provenote" install will reasonably ask whether
the rename is half-finished. It is not — the split is the decision. This ADR records why, so the
mixed surface reads as intentional, and pins the one-way doors (bundle id, icon pipeline).

## Options

1. **Rename everything to `provenote`** (package, commands, sidecar binary, npm name, bundle id, icon).
   - *Pro:* one name end-to-end; nothing to explain.
   - *Con:* the sidecar binary name is wired into the **M4 frozen-bundle pipeline** (PyInstaller output
     name + Tauri `externalBin` + the sidecar-spawn path); renaming it re-risks a verified freeze for
     zero user-visible gain. Package/command rename breaks every doc, script, and muscle-memory `python
     -m doc_assistant.*` invocation and churns imports across `src/` for a cosmetic win.
2. **Rename only the product/installer surface; keep the code identity `doc_assistant`.** *(chosen)*
   - *Pro:* users and the OS see Provenote (name, window, icon, install id); the code, build pipeline,
     and internal binary stay stable and un-re-risked. Clean, reversible, small blast radius.
   - *Con:* a mixed surface (`doc-assistant-api` under a Provenote app) that needs this ADR to explain.
3. **Defer the rename entirely.** Rejected upstream — the name was locked with the user 2026-07-14.

## Decision

The **user/installer/OS-facing identity is Provenote**; the **code/module/build identity stays
`doc_assistant`**. Concretely:

- **Provenote:** wordmark, `index.html` `<title>`, Tauri window title, `productName`, bundle
  **identifier `com.provenote.desktop`**, README H1, and the **app icon** (`src-tauri/app-icon.png` +
  the regenerated `icons/*` set).
- **`doc_assistant` (unchanged):** the Python package + `python -m doc_assistant.*` commands, the npm
  package name `doc-assistant-desktop`, the sidecar binary `binaries/doc-assistant-api` (+ its
  `externalBin` entry), and the dev/architecture docs.
- **App icon.** A designed **laurel wreath encircling an open book** on a violet rounded tile (open book =
  reading; laurel = scholarship / provenance — on-theme for a research-integrity product). Shares the
  book motif with the in-app header `.mark` but is a richer, gradient "jewel" treatment (an OS icon is
  allowed to be more ornate than the flat UI it launches). Supplied as a designed 1024px master, committed
  as `src-tauri/app-icon.png`; the platform set is regenerated via `npx tauri icon src-tauri/app-icon.png`.
  *(An earlier cut generated a flat white `book-open` from the header glyph via a Pillow script; superseded
  by this designed icon and the script removed — see the V3b DEVLOG entry.)*

## Consequences

- **Easy:** re-skinning the product name/icon later touches only the product surface above; the build
  pipeline and imports are untouched. The icon is reproducible (source + generator both committed).
- **Hard / committed:** the **bundle identifier is a one-way door** — `com.provenote.desktop` is how the
  OS keys installs, settings, and update channels; changing it again would orphan existing installs, so
  it is now fixed. The sidecar binary name is likewise pinned by the freeze pipeline until a deliberate
  M4 change renames it end-to-end.
- **Watch:** the mixed `doc_assistant`/Provenote surface is intentional — do **not** "finish" the
  rename into `src/`, the commands, or the sidecar as a cleanup. If a full code rename is ever wanted,
  it is its own migration (freeze-pipeline + docs + imports), superseding this ADR.
