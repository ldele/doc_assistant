<!-- status: active · updated: 2026-08-12 · class: append-only -->

# ADR-044 — The app tells you a new version exists; it never installs one

- **Status:** accepted (built)
- **Date:** 2026-08-12
- **Deciders:** user (product direction, 2026-08-12), Claude (Claude Code session 2026-08-12)
- **Relates to:** ADR-032 (outbound source verification — still a stub; this ADR ships first and
  sets the transport/privacy precedent it will inherit) · `docs/RELEASE.md` · ADR-011/ADR-034
  (the persisted-user-setting shape reused here)

## Context

Provenote ships as an NSIS installer. Once installed, the app has **no way to tell the user a
newer version exists** — there is no store, no package manager, no channel at all. The user's
framing (2026-08-12): *do it like calibre for now — signal that a new version is available, then
provide a link to the new release, and let the user install it.* Explicitly rejected in the same
breath: **an integrated in-app updater is too ambitious for now**, because several features are
not stable enough to be pushed at users automatically.

That rejection is the load-bearing half. An in-app updater is not merely more code; it is a
different set of promises — it must be able to replace a running binary, roll back a bad write,
verify a signature, and be trusted to do all three unattended. None of that is earned yet.

The constraint that makes even the *notification* a decision rather than a detail: **this app is
local-first and has no outbound network calls today.** Nothing in `src/` opens a socket to the
internet except an LLM provider the user configured. A version check is therefore the **first
outbound network feature to actually ship**, ahead of ADR-032's source verification, which is
still a proposed stub. Whatever discipline is set here becomes the precedent.

## Options

1. **Do nothing.** Users find out about new versions by visiting the repository, or never. —
   *Pros:* the local-first promise stays unqualified; zero new failure surface. *Cons:* in
   practice every install is frozen at the version it shipped with, including the ones carrying a
   known data-loss or wrong-answer bug. Shipping fixes nobody receives is not shipping.
2. **Notify and link (chosen).** An HTTPS GET to the public GitHub Releases API; if the latest tag
   is newer than the running version, show a dismissible banner with a link to the release page.
   No download, no install, no elevation. — *Pros:* the smallest thing that closes the loop;
   auditable in one function; degrades to silence when offline. *Cons:* the user still installs
   manually; a new outbound code path exists and must be honest about failing.
3. **In-app download + install (rejected).** Fetch the installer, verify, run it, restart. —
   *Pros:* the experience users expect from a commercial desktop app. *Cons:* rejected by the user
   for this stage, and rightly: it requires code-signing (there is none), signature verification,
   a rollback story, and enough confidence in the release process to push binaries unattended.
   The current release process has **not yet produced a verified artifact for the tag it cut**
   (v0.5.0 is a source tag; `release_preflight` is RED on `artifact_fresh`). Automating delivery
   on top of a release process that is not yet trustworthy would compound the risk, not manage it.

## Decision

**Option 2, with these constraints treated as part of the decision, not implementation detail.**

- **Notification only.** The app never downloads, writes, executes, or elevates anything as a
  result of a version check. The single terminal action is *open the release page in the browser*.
- **Automatic checking is off by default; a manual check is always available.** The Settings
  toggle governs only the automatic daily check. The **Check now** button performs a check
  whenever pressed, regardless of the toggle — an explicit press is its own consent, and gating it
  behind a preference would make the honest state ("I don't know if I'm current") unreachable for
  a user who declined background traffic.
- **Three states, never two.** `current` · `update_available` · `unknown`. A failed check reports
  **unknown**, never "up to date". Absence of evidence must not display as evidence of absence —
  the same rule ADR-032 sets for verification, and ADR-004/ADR-031 for advisory signals.
- **At most one automatic check per 24 h**, cached in the data home alongside the other persisted
  user settings. The check never runs on the answer path and never blocks a turn.
- **Bounded and fail-safe.** A 5-second timeout; every network, parse, and encoding failure is
  caught and turned into `unknown` with a reason. A version check must not be able to break the
  app it is checking.
- **What leaves the machine is one HTTPS GET** to a public, unauthenticated endpoint:
  `api.github.com/repos/<owner>/<repo>/releases/latest`. No corpus content, no query text, no
  document titles, no install identifier, no telemetry. The request carries a `User-Agent` naming
  the app and its version — required by GitHub's API and the only thing disclosed beyond the
  connection itself. This is stated **in the app**, next to the toggle, not only here.
- **The running version is a real constant** (`doc_assistant.__version__`), gate-checked against
  the other five by `release_preflight` and a unit test. A comparison against a version the app
  guessed would be worse than no comparison.
- **Pre-releases are never offered.** GitHub's `releases/latest` already excludes them; the
  comparison additionally treats a pre-release tag as older than its own release, so a
  `0.6.0-rc1` build is told 0.6.0 is newer and a stable build is never pointed at an rc.

## Consequences

- **Easy:** the persisted-setting shape already exists (`app_settings.py`, ADR-011/ADR-034); the
  transport is stdlib `urllib.request` — no new dependency — and was already spiked against
  Crossref (25/25) for the parked discovery work. The version comparison is pure and fully
  unit-testable without a socket.
- **Hard / committed:** this project now has a network code path in `src/`, permanently. It
  inherits the frozen-build TLS-trust obligation (KI-10) — the same OS-trust-store fix the sidecar
  already carries, now load-bearing for a second caller. Offline behaviour, timeouts and cache
  invalidation become things that must keep working.
- **Release-process coupling.** The banner is only truthful if GitHub Releases are actually cut
  for tags. A pushed tag with no release object reports `unknown` forever — honest, but useless.
  **Cutting a GitHub release is now a step in `docs/RELEASE.md`, not an optional courtesy.**
- **Boundary.** Delivery — downloading, verifying, installing, rolling back — is out of scope and
  needs its own ADR *and* a code-signing decision before it is even designable. Nothing here
  should be read as a step toward it having been started.
