<!-- status: active · updated: 2026-07-14 · class: disposable -->
<!-- BUILDING 2026-07-14. Sole active contract (SPRINT-018/V3a committed 181046c + archived).
     Design lock: docs/specs/feature-visual-identity.md §V3b + ADR-012 (installer identity). Feature:
     Visual-identity V3b — the Provenote app icon + branding-asset regeneration. A designed laurel-wreath
     + open-book icon (user-supplied 1024px master), regenerate the full src-tauri/icons/* platform set
     via `tauri icon`. NO src/ change, NO wire type, NO locked setting, NO behavior change; icon assets +
     one ADR + docs. NOT browser-preview-verifiable (it is the OS/installer icon) — verified by dimension
     check + a magnified small-size contact sheet. -->

# SPRINT-019 — visual-identity-v3b-icon

- **base:** main
- **DoD:** The Provenote **app icon** is designed + the full platform icon set regenerated from it.
  Specifically:
  - **Design** — a designed **laurel wreath encircling an open book** on a violet rounded tile (open
    book = reading; laurel = scholarship / provenance). Shares the book motif with the in-app header
    `.mark` but is a richer gradient "jewel" treatment.
  - **Source** — the user-supplied 1024×1024 master, committed as `apps/desktop/src-tauri/app-icon.png`.
    (An earlier cut generated a flat `book-open` from the header glyph via a Pillow script; superseded by
    this designed icon and the script removed — see the V3b DEVLOG entry.)
  - **Regenerate** — `npx tauri icon src-tauri/app-icon.png` rewrites the full `src-tauri/icons/*` set
    (32/64/128/128@2x PNGs, `icon.png` 512, `icon.ico` 16→256 frames, `icon.icns` 1024, the Windows
    Store `Square*Logo`/`StoreLogo` set, and android/ios) so the whole set is one coherent brand.
  - **Identity split** — recorded in **ADR-012**: Provenote = installer/product identity (name, window,
    icon, bundle id `com.provenote.desktop`); `doc_assistant` = code identity (package, commands, npm
    name, `doc-assistant-api` sidecar) — unchanged. `tauri.conf.json` `bundle.icon` already points at
    the regenerated files (unchanged paths); no config edit needed.
  - **Verify** — key sizes dimension-checked (32/64/128/256/512/1024 + `.ico` frame set); a magnified
    small-size contact sheet (16/24/32/48/64/128) confirms the mark stays legible down to 16px; **no
    `src/` / API / wire-type / logic / locked-setting / behavior change** (assets + docs only).
- **Out of scope (this sprint):** any `src/` / API / packaging-pipeline change; renaming the sidecar
  binary or the Python/npm package (ADR-012 pins them); a full macOS squircle-safe-area remaster (the
  `tauri icon` default full-bleed rounded tile is shipped, consistent with the prior default set); new
  brand assets beyond the icon (splash, marketing, favicon variants).

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check. -->

## uses
- docs/specs/feature-visual-identity.md
- docs/decisions/ADR-012-provenote-installer-identity.md
- apps/desktop/src-tauri/tauri.conf.json

## affects
- apps/desktop/src-tauri/app-icon.png
- apps/desktop/src-tauri/icons/**
- docs/specs/feature-visual-identity.md
- docs/decisions/ADR-012-provenote-installer-identity.md
<!-- app-icon.png (binary master) + the regenerated icons/* set + the new ADR are affects-only (kept out
     of uses: binaries/generated assets blow the sprint_check line budget, and new files legitimately
     can't pre-exist in uses — tolerated uses⊇affects WARN, SPRINT-016/017/018 precedent). tauri.conf.json
     is uses-only: read to confirm bundle.icon paths, not edited. -->

## contracts
- manual: apps/desktop/src-tauri/icons :: regenerated from app-icon.png; key sizes + .ico frame set dimension-verified; legible to 16px (contact sheet)

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
- docs/ui-checklist.md
