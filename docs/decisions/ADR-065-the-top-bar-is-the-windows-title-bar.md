<!-- status: active · updated: 2026-11-06 (written after the build: `f35cabe`, hover text with ROADMAP 110's merge `5ed8bf5`; choice 7, the settings cog, decided by the user and not built yet) · class: append-only -->

# ADR-065 — The top bar is the window's title bar on Windows

- **Status:** built (Claude Code, 2026-11-06), committed by the user as `f35cabe`, and written
  down afterwards at the user's request (*"Write ADR first"*, then *"Yes, after commit"*). Choices
  1 to 6 are recorded as built and wait for the user's read. Choice 7 is the user's own answer of
  2026-11-06 and is not built yet. The request is the user's, 2026-11-06, on a screenshot of the
  installed app: *"Could we do a better top bar / title bar for provenote window. As you know,
  having the default is not modern (dead space)."*
- **Date:** 2026-11-06
- **Deciders:** user + Claude Code
- **Works with** [ADR-063](ADR-063-a-buttons-look-says-what-it-does.md), the button rules. The
  window controls are a written exception to its four roles (choice 3) and follow its hover-text
  rule. Also works with [ADR-064](ADR-064-the-frontends-folders-follow-the-screens.md):
  `WindowControls.svelte` is part of the frame, in `src/shell/`.
- **In one sentence:** on Windows the window loses its native title bar, and the app's own top
  bar takes its place. Its empty space drags the window, and minimise, maximise/restore and close
  sit at its right end, drawn the way the platform draws them. Other platforms keep their native
  frame.

## To decide — seven choices, each with a recommendation

1. **Which platforms lose the native frame.** *Recommended:* Windows only. A second config file,
   `apps/desktop/src-tauri/tauri.windows.conf.json`, restates the main window with
   `"decorations": false` and `"shadow": true`. Tauri merges it over `tauri.conf.json` on Windows
   only. A merge patch replaces an array whole, so the file has to restate the window, and
   `tests/unit/test_desktop_window_config.py` keeps the two copies the same window. macOS puts
   its window buttons on the left, and drawing the Windows set there would break that platform's
   convention. The release ships for Windows only, so no other platform loses anything.
   `WindowControls.svelte` asks the window whether it is framed (`isDecorated()`) and draws
   nothing when it is, or when there is no Tauri window at all (the Vite preview in a browser).
2. **What drags the window.** *Recommended:* the whole top bar, through Tauri's own drag script.
   `data-tauri-drag-region="deep"` on `.topbar` makes a press anywhere in it drag the window, and
   a double-click maximise or restore it. The script never drags from a button, a link, a field or
   an element with an interactive role (tabs included), so every control in the bar stays
   clickable. The app menu's wrapper carries `data-tauri-drag-region="false"`: its backdrop covers
   the window while the menu is open, and a press on the backdrop must close the menu, not drag.
3. **How the three buttons look: a written exception to ADR-063.** *Recommended:* the platform's
   caption buttons, not one of the app's four roles. Each button is 46 px wide and as tall as the
   bar, flush with the window's top and right edges, so a maximised window's corner pixel is
   Close. The glyphs are 10 px hairlines rather than the app's Lucide icons, and the cursor stays
   an arrow. Close turns red on hover: `--caption-close`, the platform's `#c42b1c`, the same in
   both themes, with white on it at 5.7:1. That red means "this closes the window", not ADR-063's
   *danger*, which is why it is not `--danger`. Close asks nothing first, as the native close
   never did. Keyboard focus shows the app's ring, inset, because the buttons touch the window's
   edge. When the window is inactive, the glyphs grey out, as the platform's do. The rest of
   ADR-063 holds. Each button is named by its `aria-label` (*Minimise*, *Maximise* or *Restore*,
   *Close*) and has hover text through the shared tooltip (choice 6 there), written at the merge
   with ROADMAP 110:

   | Button | Hover text |
   |---|---|
   | Minimise | *Hide the window in the taskbar* |
   | Maximise | *Make the window fill the screen* |
   | Restore (while maximised) | *Return the window to its earlier size* |
   | Close | *Quit Provenote* (closing the window ends the app and its engine, as before) |

   The exception needs no allow-list entry: the buttons use their own class names, and their
   colours are tokens.
4. **What the window may do from the page.** *Recommended:* four permissions in
   `capabilities/default.json`, on top of `core:default` and `dialog:allow-open`:
   `core:window:allow-start-dragging`, `allow-minimize`, `allow-toggle-maximize` and
   `allow-close`. Asking whether the window is maximised or framed, listening for its resize and
   focus events, and the drag script's double-click (`allow-internal-toggle-maximize`) are already
   in `core:default`. These four move, minimise, maximise or restore, and close the window. None
   reads anything. A script injected into the page could at worst close or move the window, and
   the CSP and DOMPurify stop injected script from running at all (`docs/security.md`).
5. **Windows 11's snap layouts on the maximise button.** Hovering the native maximise button opens
   a menu of window layouts. Windows offers it only when the button is part of the native frame
   (its hit test answers "maximise button"), and the drawn button is page content, so hovering it
   shows nothing. *Recommended:* accept the loss. Win+Z still opens the same layouts for the
   window, and the window keeps its native style bits underneath (caption, system menu, thick
   frame, minimise and maximise boxes), so the system's own handling stays. Keeping the hover
   menu would need either the native title bar (the dead row the user asked to remove), or Rust
   code that subclasses the webview's window to answer that hit test for the button's rectangle.
   That code is fragile, it is not in Tauri, and every Tauri upgrade would have to recheck it.
6. **Dialogs and drawers over the bar.** The shared dialog frame (`shell/Dialog.svelte`, ROADMAP
   110 (d)) and the two drawers (Settings, the chat's source panel) cover the whole window,
   including the window controls. *Recommended:* keep it so. A modal dialog on Windows disables
   its owner window, title bar buttons included, so covering them is the platform's own rule. The
   way out of a dialog is its own close or Esc, and Alt+F4, Win+Down and the taskbar still work.
   If raising the controls above every overlay is ever wanted, the shared dialog frame is where
   overlays would learn to keep clear of them. It is one place, not twenty.
7. **Where the settings cog goes.** With the window controls at the right end, the cog sat
   between the wordmark and *Minimise* and read as part of the window's frame. The user,
   2026-11-06: *"The settings cog should be moved to a better place now."* Four places were
   offered: beside the menu, the sidebar's footer, before the wordmark, the status bar. The user
   chose **beside the menu, at the left** (the recommendation). The app's controls then sit
   together at the left, and the right end holds only the wordmark and the window controls. **Not
   built yet:** the move waited for ROADMAP 110's merge into `Topbar.svelte`, which has landed
   (`5ed8bf5`).

## In practice — the screenshot, and the dev window on an invented library

**Before**, from the user's screenshot of the installed 0.7.2: the native title bar (icon,
*Provenote*, minimise/maximise/close) sat in a row of its own above the app's top bar. That row
is 31 px on this display at 96 DPI (caption 23, frame 4, padding 4, from the system's metrics).
Every screen lost it, and the app's top bar below already had empty space at its right end.

**After**, driven on 2026-11-06 in a `tauri dev` window over an invented library (six made-up
documents in `C:\Projects\pv-titlebar`, its API on port 8002, the shell's bundled engine kept
out), never the user's library:

| Check | Seen |
|---|---|
| No native title bar | yes: the top bar is the window's top edge, with a shadow and Windows 11's rounded corners |
| Drag by the bar's empty space | yes: the window moved with the pointer |
| Double-click on the empty space | maximised. The visible frame was exactly the work area (0, 0, 2560, 1392): no overflow past the screen's edges, taskbar uncovered |
| Maximise icon while maximised | the restore glyph |
| The corner pixel of a maximised window | Close, highlighted red |
| *Restore* | back to the earlier size and place |
| Win+Z | the snap layouts for the window |
| Hovering the drawn maximise button | no snap layouts (choice 5) |
| The window's style | caption, system menu, thick frame, minimise and maximise boxes still set (`0x14CF0000`) |
| The Vite preview in a browser | no window controls drawn; drag attributes in place, inert |

**Not seen live**, because the session's screen checks stopped before them: *Minimise* and
*Close* by click, resizing from the edges, keyboard focus on the three buttons, the light theme,
a 640 px window, and a dialog over the bar. They are rows in `docs/release-ux-checklist.md` §6,
to be driven in the installed build before the tag. The code paths behind *Minimise* and *Close*
are held by `apps/desktop/src/lib/core/tauri.test.ts` (each reaches the window, on the window
object; a refused command is swallowed).

## Context

Tauri draws the native frame unless the window is created with `decorations: false`. Without it,
the page has to do three things the frame did: say where the window can be dragged, draw the
buttons, and keep the resize edges. Tauri 2.11 covers the first with its injected drag script and
the third itself: on Windows, `tauri-runtime-wry`'s `undecorated_resizing` answers the edge hit
tests that the webview would otherwise swallow, and the shadow stays when `shadow` is true. So the
app's part is the buttons, the drag attributes and four permissions.

The frontend reaches the window only through `lib/core/tauri.ts`, the one module allowed to touch
`window.__TAURI__` (AD1). Its new functions degrade like the rest: in a browser there is no
window, so no control is drawn and every call is a no-op.

## Options

- **A. The app's own title bar on Windows, the native frame elsewhere** (recommended, built).
- **B. Keep the native title bar.** Nothing to build or maintain, and the dead row stays.
- **C. Frameless everywhere.** One config file, but Windows-style buttons on the right of a macOS
  window, against that platform's convention, for a platform the release does not ship.
- **D. A as built, plus native hit testing for the maximise button** (choice 5). It keeps the snap
  layouts on hover, at the cost of window-procedure code that Tauri does not provide.

## Decision

Option A with choices 1 to 6 as recommended, built on 2026-11-06 (`f35cabe`, committed by the
user). The hover texts came with ROADMAP 110's merge (`5ed8bf5`). Choice 7 is the user's: the
cog moves beside the menu, in a change of its own.

## Consequences

- The window gains a row: 31 px of every screen at 96 DPI, more at higher scaling.
- Closing the window, minimising it and maximising it now go through page code and four
  permissions. A broken drag region or a refused permission shows as a button that does nothing,
  and Alt+F4, Win+arrows and the taskbar stay the way out.
- Snap layouts on hover are gone. Win+Z replaces them.
- A dialog or drawer covers the window controls while it is open (choice 6).
- A new element in the top bar is clickable by default: the drag script skips buttons, links,
  fields and interactive roles. A clickable element with none of those (a bare `div` with a click
  handler) would drag the window instead, and needs a button or `data-tauri-drag-region="false"`.
- A size or minimum changed in `tauri.conf.json` must change in `tauri.windows.conf.json` too.
  The test fails otherwise.

## Enforced by

- `tests/unit/test_desktop_window_config.py`: the Windows file is the main window plus
  `decorations: false` and `shadow: true`, and changes nothing else; other platforms keep their
  frame.
- `tests/unit/test_desktop_security_config.py`: the capability set is exactly the six
  permissions.
- `apps/desktop/src/lib/core/tauri.test.ts`: no controls without a Tauri window or with a framed
  one; the three commands reach the window; refusals are swallowed; events unsubscribe.
- `apps/desktop/src/ui-rules.test.ts` check 2: the three buttons have hover text.

## Confidence

High on choices 1 to 4: they are what the dev window showed, or what the tests pin. Medium on
choice 5's claim that Win+Z and the style bits cover what the hover menu offered: Win+Z was seen,
and dragging to the screen's top edge was not tried. Medium on choice 6: it follows the platform's
modal rule, but the user has not seen a dialog over the new bar. The rows under *Not seen live*
are open until the installed build is driven.
