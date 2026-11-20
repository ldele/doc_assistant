<!-- status: active · updated: 2026-11-20 (proposed; nothing moved) · class: append-only -->

# ADR-067 — The backend's flat modules move into the packages the import graph draws, and the tests follow them

- **Status:** proposed (Claude Code, 2026-11-20). Not accepted until the user has read this file.
  No module has moved and no test has changed. The question is the user's, 2026-11-20: should
  `src/doc_assistant/` and `tests/` be split into more folders? The interface pass (A8) and the
  rest of release A come first. This ADR plans the split for afterwards.
- **Date:** 2026-11-20
- **Deciders:** user + Claude Code
- **Amends** [ADR-023](ADR-023-knowledge-subpackage.md). ADR-023 kept "the whole RAG answer path"
  and `doc_vectors.py` at the top level on purpose. Here both move. ADR-023's clean-break rule
  (no re-exports at the old paths) is kept and extended to every move.
- **Follows** [ADR-064](ADR-064-the-frontends-folders-follow-the-screens.md), the frontend's
  equivalent: a guard test that holds the layout, and moves kept apart from any other change.
- **In one sentence:** 40 of the 44 top-level modules move into eight packages (five new, three
  existing), each placed by what imports it and what it imports; the four that sit below every
  package stay at the top; no old path survives as a re-export; `tests/unit` and
  `tests/integration` mirror the packages; and two guard tests hold the result.

## To decide — eight choices, each with a recommendation

1. **The groups.** *Recommended:* the table in *The placement* below. Five new packages
   (`answer/`, `integrity/`, `providers/`, `services/`, `checks/`), three existing ones that
   take more modules (`ingest/`, `library/`, `read_surface/`), and four modules left at the top.
   The alternative is one `answer/` package holding the integrity layer too (14 modules). It is
   rejected because only one module-level import crosses that line (`provenance` →
   `retrieval_scope`), and the integrity layer already has its own ADRs and docs.
2. **What stays at the top.** *Recommended:* a module stays at the top if `db/` imports it, since
   `db/` is the lowest package and anything it imports must sit below every package (`config`,
   `logging_config`, `app_settings`). A module also stays if it imports nothing, at least three
   groups import it, and no group owns it (`chroma_read`). Every new module lands in a package.
3. **Names.** *Recommended:* a module keeps its file name (`providers/providers.py`,
   `checks/source_checks.py`), and a new package's `__init__.py` holds a docstring and no
   imports. Only the import line changes: `from doc_assistant import pipeline` becomes
   `from doc_assistant.answer import pipeline`, and every `pipeline.X` call stays as written. The
   DEVLOG, the known issues and the ADRs cite files by name, so a kept name stays searchable. A
   renamed module would change its call sites too. A barrel `__init__` is a re-export, which is
   the trap choice 4 is about.
4. **Old import paths.** *Recommended:* a clean break, as ADR-023 did: nothing remains at
   `doc_assistant.pipeline`, and every importer in `src/`, `apps/`, `scripts/` and `tests/` is
   rewritten in the same commit. A re-export at the old path is a separate binding: a
   monkeypatch through it succeeds and the code under test never sees it. *In practice* below
   shows one test fixture where this happens. With no shim, each stale path fails loudly when it
   is used.
5. **`extractors` and the extraction fingerprint.** *Recommended:* first make the fingerprint
   name `extractors` by a fixed identity rather than by its import path, in a commit that leaves
   every digest unchanged, and then move it. Measured below: as written today, the move changes
   the PDF fingerprint, and every PDF would be re-read on the next indexing. The alternative is
   to accept that re-read and let it ride with release B's extraction fixes (B2). It is rejected
   because B2 is not certain to touch PDF extraction, and the cache's own docstring already
   treats a rename that costs a full re-read as a defect.
6. **The `tests/` layout.** *Recommended:* `tests/unit/<package>/` and
   `tests/integration/<package>/` for every package in `src/doc_assistant/`. Tests of `apps/api`
   go under `api/`, and checks on the repository itself (docs, CI pins, release preflight,
   scripts) go under `repo/`. Tests of the four top-level modules stay at the top of their tier.
   Test file names are kept, for the same reason as choice 3.
7. **Duplicate test file names.** *Recommended:* an `__init__.py` in every test folder, the
   convention the suite already follows (eight such files today). pytest's `importlib` mode is
   rejected: three test files import from other test modules (`tests.unit.adapters.test_zotero`,
   `tests.unit.test_extractors_formats`, `tests.conftest`), which that mode does not support, and
   it would change how all 200-odd files load for nothing gained. Duplicate names already exist
   (`test_concept_semantics.py` and `test_keyword_families.py` are in both tiers). They work only
   because each folder is a package.
8. **The guards, the order, and the session.** *Recommended:* the two guard tests in *Enforced
   by* land in the first commit. Then one commit per group, smallest first, the tests after the
   code, and the docs last. All of it in one session of its own, after release A's tag and
   before release B's first session (*Order* below).

## In practice — `test_pipeline_sparse_arm.py`, three ways

`_constructed` (`tests/unit/test_pipeline_sparse_arm.py:89-96`) builds a real `RAGPipeline` with
fakes at its edges. It makes seven string patches: the Chroma path, the parent-child switch, the
embedding loader, the active model's name, the collection name, the `Chroma` class and
`build_chat_model`. It is what `test_construction_holds_no_corpus_in_ram` (ADR-038's memory
property) runs on.

**Today.** `monkeypatch.setattr("doc_assistant.pipeline.Chroma", lambda **_kw: fake)` rebinds
`Chroma` on the module that `RAGPipeline.__init__` reads its globals from, so the constructor
gets the fake.

**After the move, with a re-export left at `doc_assistant/pipeline.py`.** All seven patches still
succeed, but they land on the shim. `RAGPipeline` is defined in `doc_assistant/answer/pipeline.py`
and reads that module's globals, so it gets the real `Chroma` at the real configured path, the
real embedding model, and a chat model built for whatever provider `.env` names, Anthropic by
default (KI-4). Constructing a chat model makes no API call (the
`build_chat_model` docstring), so nothing is billed today. But the test now measures a store it
never set up, and nothing guarantees that it fails. The `chat_controller` split broke 66 tests
the same way (DEVLOG 2026-07-27, `docs/archive/DEVLOG-archive-002.md`). There the barrel was the
re-export, and pytest caught the 66 only because their assertions depended on the patch. A stub
whose job is to keep something from running gives no such signal.

**After the move, clean break (choice 4).** Line 22, `from doc_assistant.pipeline import
RAGPipeline, SparseRetriever`, raises `ModuleNotFoundError` when the file is collected, so the
file fails before any test runs. The rewrite makes line 22 `from doc_assistant.answer.pipeline
import ...` and the seven strings `"doc_assistant.answer.pipeline.Chroma"` and so on. 30 strings
name `pipeline`, in three files.

**A second case: `extractors`, measured 2026-11-20.** `ingest/cache.py` fingerprints each format's
extraction code, and a cached document whose fingerprint has moved is re-extracted. Among the
inputs it hashes are two of `extractors`' module-level values by their `repr()`: `__name__`, and
the module logger, whose repr carries the logger's name. Both contain `doc_assistant.extractors`.
A simulated move (each function's `__module__`, the module's `__name__` and its logger
re-pointed to `doc_assistant.ingest.extractors`, each case in a fresh process) changes **2 of 10
fingerprints: `.pdf` and the whole-module one**. The other eight formats stay the same, because
their code never touches the logger. An unchanged run gives the same ten digests in two
processes. So a plain `git mv` would re-read every PDF in every library on its next indexing.
The cache's docstring puts a whole-library re-extraction at 61 minutes for 97 documents. Choice 5's fix keeps the hashed
strings byte-for-byte as they are today: wherever the module's import path appears in a hashed
value, the fingerprint writes the fixed identity `doc_assistant.extractors` in its place. Today
that replacement changes nothing. After the move it maps the new path back to the old one. The
test is the simulation above: all ten digests are equal before and after.

## Context

- **What is flat, 2026-11-20.** `src/doc_assistant/` has 8 subpackages and 44 modules beside
  `__init__.py`: `adapters`, `chat_controller`, `db`, `eval`, `ingest`, `knowledge`, `library`
  and `read_surface`. The tests sit almost flat: `tests/unit` has 129 test files at its top and 9
  in `adapters/` and `ingest/`, and `tests/integration` has 62 at its top and 13 in `ingest/`.
- **The patch strings.** The tests hold 72 string-form `monkeypatch.setattr`/`delattr` targets,
  in 23 files (a one-line grep finds 57, because 15 calls break across lines). 41 of them, in 12
  files, name a module this ADR moves: `pipeline` 30, `llm` 4, `sparse_index` 2, `query_router`
  2, `metadata_enrich`, `fsutil` and `sources_manifest` 1 each. Another 15 name `config`, which
  stays, and 16 name modules already inside a package. pytest's own resolver resolves all 72
  today.
- **The rule that bites.** `src/doc_assistant/CLAUDE.md`: monkeypatch the module that owns a
  name, never a package that re-exports it. A move changes which module owns the name.
- **How the groups were found.** From the import graph, split into module-level, in-function
  and `TYPE_CHECKING` edges, read with the AST on 2026-11-20. Names and docstrings were
  checked afterwards. Four modules land somewhere other than where their names suggest:
  - `health` reads like the engine's `/api/health`, but it classifies a document's extraction,
    and only `ingest/` imports it.
  - `compare` is the retrieval diff of two source sets, and only `chat_controller` imports it.
  - `app_settings` reads like an app service, but `db/` imports it, so it sits below every
    package.
  - `reingest` is served by `apps/api/routers/library/documents.py` and drives a document the
    way `library/add.py` does, so it goes beside it.
- **Cycles that exist today and that the move neither makes nor fixes.** `sparse_index` imports
  `knowledge/` (the tokeniser), and `knowledge/wiki.py` imports `context` (its token estimate).
  `library/` imports `ingest/` at module level, and `ingest/` imports `stored_paths` and
  `demo_corpus` inside functions. The new packages' empty `__init__` files (choice 3) mean none
  of these can turn into an import-time cycle.
- **Nothing outside reads these paths.** The sidecar build takes `collect_submodules("doc_assistant")`
  (`scripts/doc_assistant_api.spec:56`), which follows new packages. Every `python -m` in the
  docs and scripts names `doc_assistant.ingest` or `doc_assistant.db.migrations`, and neither
  moves. 64 loggers are `structlog.get_logger(__name__)`, so the logger name in a log line
  changes, and nothing filters on it. The one stored identity that depends on a module path is
  the extraction fingerprint (choice 5). The sparse index fingerprint hashes the tokeniser's
  source, not its path.
- **The size of the rewrite.** 174 files (81 tests, 50 in `src/`, 28 scripts, 15 in `apps/`)
  and about 414 import statements, counted with the AST; `scripts/archive/` stays frozen (ADR-023).

## The placement

| Package | Modules | Why here (the graph) |
|---|---|---|
| `answer/` (new) | `pipeline`, `sparse_index`, `prompts`, `context`, `chat_modes`, `retrieval_scope`, `query_router`, `record_lookups`, `commands`, `compare` | what a turn retrieves and sends; `chat_controller/` and `eval/` import them, and they import `providers/` and `library/` |
| `integrity/` (new) | `provenance`, `synthesis`, `reviewer`, `reviewer_aggregate` | the research-integrity layer (Integrity Chunks 1, 2a, 2b, 2c); one module-level edge out to `answer/` (`retrieval_scope`) |
| `providers/` (new) | `providers`, `llm`, `compatible`, `credentials`, `embeddings`, `tracking` | `providers`, `llm` and `compatible` import one another; `credentials` holds the keys they build clients with; `embeddings` builds the other model the app runs; `tracking` counts their tokens |
| `services/` (new) | `conversations`, `export`, `readiness`, `update_check`, `library_home`, `corpus_stats` | what the app keeps or reports apart from an answer; their importers are `apps/` and `chat_controller/`, never the answer path |
| `checks/` (new) | `source_checks`, `document_checks`, `stored_paths` | the checks `docs/security.md` names (S5, S-7); each is imported by three or more groups |
| `ingest/` | `extractors`, `metadata_extractor`, `metadata_enrich`, `health`, `sources_manifest`, `fsutil`, `doc_vectors` | imported only by `ingest/` or by modules that already depend on it (`adapters/`, `library/`) |
| `library/` | `reingest`, `bibtex`, `demo_corpus` | acts on documents (`reingest` beside `add`); `demo_corpus` is folder membership |
| `read_surface/` | `mcp_server` | imports only `read_surface/` |
| top level | `config`, `logging_config`, `app_settings`, `chroma_read` | choice 2 |

`doc_vectors` was kept at the top by ADR-023 as "an input, like `ingest/citations.py`". This
ADR takes that comparison at its word.

## Order

Every move commit holds the `git mv`, every rewritten import and patch string, and the guard's
list of not-yet-placed modules, shorter by that group. It is staged whole, because the commit
hook stashes unstaged tracked files, and a half-staged move would type-check a tree that never
existed.

1. The two guards (*Enforced by*), with the 40 modules still to place listed in the first one.
2. The extraction fingerprint's fixed identity (choice 5), and its test. No digest changes.
3. `read_surface/` ← `mcp_server` (2 files to rewrite).
4. `library/` ← `reingest`, `bibtex`, `demo_corpus` (15 files).
5. `services/` (24 files). 6. `checks/` (19 files). 7. `integrity/` (24 files).
8. `ingest/` ← seven modules (40 files, 3 patch strings).
9. `providers/` (59 files, 4 patch strings).
10. `answer/` (49 files, 34 patch strings).
11. `tests/unit` mirrored, then 12. `tests/integration` mirrored: `git mv`, an `__init__.py` per
    new folder, and the cross-test import lines. A file named for a module goes to that module's
    folder. Any other test goes to the package of the module it tests, judged by its imports
    other than `db` and `config`, which most integration tests import only to set up. The build
    generates this map with a script, and it is read before anything moves.
13. The living docs: `src/doc_assistant/CLAUDE.md`'s layout block, `docs/architecture.md`'s
    module map, and comments that name an old path
    (`apps/desktop/src/screens/settings/ProviderSetup.svelte:7`). Append-only
    records keep their historical paths (ADR-023).

The groups without patch strings go first, so the recipe is proven where a miss can only be an
import error. Per commit: ruff, `uv run --no-sync mypy src`, the two guards, and the tests of the
moved modules and their importers. The full suite runs once, before the push. If it is red, it
is run at each commit to find the move that broke it, and that commit is fixed before anything
is pushed. The packaged build is started once at the end; that new packages reach it through
`collect_submodules` is read in the spec, not yet run.

**The session.** A session of its own, after release A's tag and before release B's first
session (reading real documents). Not during A8: A8, the security review, the ingestion queue
and the release all edit `ingest/`, `library/`, `read_surface/` and `chat_controller/`, and the
security review reads the MCP path that `mcp_server` is part of. A move under any of them
conflicts with every open change. Before release B, because B2 changes extraction and B4 adds row
106's code, which then lands placed. B6's upgrade drive then checks the packaged build.

## Decision

Proposed: the eight choices as recommended, the placement table, the order, and the guards below.

## Consequences

- **Easier:** the package listing states the architecture, with four foundation modules left at
  the top. A task
  on the answer path, the providers or the security checks has a boundary to stay inside. A
  patch string that names a module that no longer exists fails in a millisecond test, not in a
  fixture.
- **Harder:** a one-time rewrite of 174 files. Every open branch that touches `src/` conflicts
  with it, which is why it waits for a gap between releases. Import lines get longer.
- **Commits us to:** new modules land in a package, or the first guard fails. New packages have
  no barrel. Existing barrels (`library/`, `read_surface/`, `chat_controller/`, `ingest/`) stay
  as they are, and the modules moved into them are not added to them.
- **Not decided here:** a layering test (which package may import which). The cycles in
  *Context* would fail it today. Whether existing barrels should go is also left open.
- **Reverses if:** never wholesale. A group that proves wrong is moved again under a later ADR,
  with the same guards.

## Enforced by

Both in `tests/unit/repo/`, first commit of the session:

- **`test_package_layout.py`.** The top of `src/doc_assistant/` holds `__init__.py` and the four
  modules of choice 2, plus, only during the move session, the modules still to place. Any other
  `.py` fails with a message naming the groups and this ADR. The set of packages is a listed set,
  so a new package fails until the list names it. Every folder under `tests/unit` and
  `tests/integration` that holds a test has an `__init__.py`. A `test_<m>.py`, where `<m>` is
  exactly one module inside package P, sits under P's folder, apart from `repo/`.
- **`test_patch_targets.py`.** It parses every test file. Each string first argument of
  `monkeypatch.setattr` or `delattr` (and `mock.patch`, if one appears) that starts with
  `doc_assistant.` or `apps.` must resolve as pytest resolves it: import the longest module
  prefix, then walk the attributes (`doc_assistant.fsutil.os.replace` is such a chain). This
  catches what the suite can miss: a patch in a test skipped on this machine, and one with
  `raising=False` (`tests/unit/test_chat_controller.py:1112`, on `llm`), which passes even when
  its attribute is gone. It does not judge whether the binding is the one the code reads, because
  that cannot be read from the string. `doc_assistant.library.count_documents`
  (`tests/integration/test_api_setup.py:48`) patches a re-export, and it is the right target:
  `apps/api/services.py:140` imports the name from the package at each call. Choice 4 covers the
  rest.

## Confidence

- **Measured, 2026-11-20:** the import graph and its edge kinds; the importer counts; the 72
  patch strings and their resolution; the duplicate test names; the fingerprint change (2 of 10)
  and its stability across processes; that none of the moved names collides with a name the
  target package already exports.
- **Read, not run:** that `collect_submodules` picks up the new packages in the frozen build, and
  that the fixed-identity fix leaves all ten digests equal. It is designed to, and its test is the
  simulation above.
- **Not verified:** the per-test map of steps 11–12, which the build generates and reviews.
  Until then, the counts for `api/` (about 27 files) and `repo/` (about 15) are from file names.

## Links

- [ADR-023](ADR-023-knowledge-subpackage.md): the layout amended here, and the clean-break rule.
- [ADR-064](ADR-064-the-frontends-folders-follow-the-screens.md): the frontend's split and its
  layout test.
- `docs/archive/DEVLOG-archive-002.md`, entry 2026-07-27: the `chat_controller` split and the 66
  tests.
- `src/doc_assistant/ingest/cache.py`, `extraction_fingerprint`: the hashed inputs of choice 5.
