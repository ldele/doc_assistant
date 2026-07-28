# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [SemVer](https://semver.org/) on the `doc_assistant` package, and the desktop app
(Provenote) ships the same number.

The engineering record is finer-grained than this file: per-change entries live in
[`docs/DEVLOG.md`](docs/DEVLOG.md), design decisions in [`docs/decisions.md`](docs/decisions.md).

## [0.3.0] — 2026-07-28

**The first release meant for someone other than its author.** 0.1.x/0.2.x were development
versions; the change that makes this one shippable is that setup no longer requires editing files or
reading the source.

### Added

- **First-run setup in the app** (ADR-034). `Settings → Getting started` shows what the install
  still needs — an answer engine, some documents — with the exact next action for each, and the chat
  screen carries the same checklist until both are done.
- **Bring your own key, in-app.** Paste an Anthropic API key in Settings. It is verified first with a
  free metadata call (no tokens, no charge), refused if the API rejects it, and stored on your
  machine in the data home. `Remove key` forgets it. A key in `.env` still wins and the panel says
  when it does, so the app and the CLI can never send different keys.
- **Honest Ollama detection.** The app probes the local server, lists the models it actually has, and
  distinguishes *not running* from *no models installed* — two states with different fixes. Picking a
  model applies to the next question, with no restart.
- **[`docs/QUICKSTART.md`](docs/QUICKSTART.md)** — the 10-minute first run for a new user, both
  provider paths, with a symptom/cause table.
- `library.count_documents()` — a `COUNT` for the setup view rather than materializing every
  document summary to discard it.

### Fixed

- **A key set at runtime could not have worked.** `pipeline.build_chat_model` resolved the API key
  through an import-time binding, so a key that arrived later was never sent. Every Anthropic call
  site (chat model, one-shot client, figure VLM, CLI cost guard) now resolves per construction.
- **The provider picker claimed Ollama was available on machines that had never installed it**
  (`provider_available("ollama")` was unconditionally true).
- **The empty chat screen named only half the problem** — it said "no documents indexed" whether or
  not a working provider existed.
- Four planning ADRs (030–033) had files but no rows in the decisions index.

### Changed

- The provider/model section in Settings is now the advanced form of the same switch, and no longer
  tells you to add a key to `.env`.
- `.env` is documented as optional for running the app, and required only for CLI/enrichment runs.

### Known limits (unchanged, and stated in the README)

- The in-app key is **plaintext in the data home** — weaker than an OS keychain, which ADR-034
  records as the upgrade path and why it was not taken in this release.
- Validated at ~100 documents; the enrichment layer has corpus-linear paths catalogued with a fix
  plan ([scale review](docs/REVIEW_2026-07-19_scale-robustness.md)).
- Small local models place documents into a taxonomy at 70–87% precision and their self-reported
  confidence carries almost no signal.
- Tested primarily on Windows, plus CI on Linux; macOS (MPS) paths work but are unbenchmarked.
