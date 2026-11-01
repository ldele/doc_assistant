"""Extraction cache + content hashing — the bottom layer of the ingest package.

Turns a source file into its cached markdown (extracting + caching on a miss) and
hashes that content. The cached ``.md`` is the source-of-truth the rest of the
pipeline re-reads, so writes go through the atomic helper. Path/extractor config is
read dynamically (``config.X``) so a single seam is monkeypatch-able in tests.
"""

from __future__ import annotations

import dis
import hashlib
import importlib
import inspect
import json
import os
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import CodeType
from typing import Any

import structlog

from doc_assistant import config
from doc_assistant.extractors import IngestRefusedError, extract_to_markdown, ingest_refusal
from doc_assistant.fsutil import atomic_write_text

log = structlog.get_logger(__name__)


#: Where caches for files outside the library folder live, under `config.CACHE_PATH`. A fixed
#: subdirectory so the referenced half of the cache is inspectable and deletable on its own.
_REFERENCED_CACHE_DIR = "referenced"


def get_cache_path(original: Path) -> Path:
    """The cached ``.md`` for a source file, wherever that file lives (ADR-046, AD3b).

    A file **under the library folder** keeps the mirror layout it has always had:
    `data/sources/a/b.pdf` -> `data/cache/a/b.md`. That path is unchanged on purpose — every
    already-extracted document depends on it, and moving it would silently re-extract the whole
    corpus.

    A **referenced** file lives anywhere on disk, so there is no relative path to mirror. Its
    cache is keyed by a digest of its case-normalised absolute path, which gives the three
    properties the mirror layout gave for free: the same file always resolves to the same cache
    entry, two files with the same name in different folders never collide, and the name stays
    filesystem-legal whatever the source path contained. The digest is over the path rather than
    the bytes because this has to resolve *before* the file is read — and cheaply, since
    `registry.scan_root` calls it once per file on every listing.

    The stem is kept alongside the digest so the cache directory is still human-readable when
    someone goes looking for what a document extracted to.
    """
    try:
        relative = original.relative_to(config.DOCS_PATH)
    except ValueError:
        digest = hashlib.sha1(
            os.path.normcase(os.path.abspath(str(original))).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:16]
        return config.CACHE_PATH / _REFERENCED_CACHE_DIR / digest / f"{original.stem}.md"
    return config.CACHE_PATH / relative.with_suffix(".md")


def _fingerprint_path(cached: Path) -> Path:
    """Sibling of the cached ``.md`` holding the extractor identity that produced it.

    A separate file rather than a header inside the ``.md``: that file's bytes ARE the document
    text and are hashed into ``doc_hash``, so anything added to it would change every document's
    identity.
    """
    return cached.with_name(cached.name + ".fp")


# Manual escape hatch. Bump to force a re-extraction that the automatic parts below cannot see —
# e.g. a changed string literal that alters output without changing any bytecode. Rarely needed;
# it exists so "force everyone to re-extract" is always available in one line.
_EXTRACTION_VERSION = 2


def _referenced(code: CodeType) -> tuple[set[str], set[str]]:
    """(global names referenced, modules imported) by a code object and everything nested in it.

    Nested code objects matter: a comprehension or a closure compiles to its own object, so a call
    made only from inside one would otherwise be invisible to the walk.
    """
    names = set(code.co_names)
    imports = {
        i.argval
        for i in dis.get_instructions(code)
        if i.opname == "IMPORT_NAME" and isinstance(i.argval, str)
    }
    for const in code.co_consts:
        if isinstance(const, CodeType):
            sub_names, sub_imports = _referenced(const)
            names |= sub_names
            imports |= sub_imports
    return names, imports


def _extraction_closure(
    blocked: frozenset[str], seeds: tuple[str, ...] = ()
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Walk out from ``extract_to_markdown`` **and** ``seeds``, never entering ``blocked``.

    Returns (``{attribute name: function}``, constant names, top-level module names).

    Walking from the dispatcher is what picks up whatever it does to *every* format — today
    ``strip_image_placeholders`` at the single exit — so a shared post-processing step added later
    cannot silently fall outside a format's identity. Blocking the *other* formats' entry points
    narrows the result back to this one.

    **``seeds`` is not an optimisation, it is a correctness requirement.** Every non-PDF format is
    dispatched through the ``_EXTRACTORS`` dict, which is a runtime lookup: a static walk cannot
    follow it, so `extract_epub` is unreachable from `extract_to_markdown` and an EPUB fix would
    not invalidate a single EPUB cache — KI-40 reintroduced, silently. Seeding the walk with the
    format's own entry function is what closes that hole. Measured, not assumed: before this, the
    `.epub` closure held two functions and neither was `extract_epub`.

    **Keyed and returned by the name the function is bound to on the module, and the *object* is
    carried out rather than the name alone.** Both halves matter, and for the same reason: the
    caller used to hash ``getattr(extractors, name).__code__``, re-resolving a name this walk had
    already resolved. A function whose ``__name__`` differs from its attribute — an alias, or any
    decorator that does not carry ``functools.wraps`` — made that lookup raise, and the caller's
    blanket ``except`` turned it into the whole-module fingerprint: a silent, corpus-wide
    re-extraction (61 min for 97 documents; ~55 h projected at 10,000) triggered by a rename.
    """
    from doc_assistant import extractors

    functions: dict[str, Any] = {}
    constants: set[str] = set()
    modules: set[str] = set()
    stack: list[tuple[str, Any]] = [("extract_to_markdown", extractors.extract_to_markdown)]
    for seed in seeds:
        obj = getattr(extractors, seed, None)
        if inspect.isfunction(obj):
            stack.append((seed, obj))

    while stack:
        attr, fn = stack.pop()
        if attr in functions:
            continue
        functions[attr] = fn
        names, imports = _referenced(fn.__code__)
        modules |= {m.split(".")[0] for m in imports}
        for ref in names:
            obj = getattr(extractors, ref, None)
            if obj is None:
                continue  # an attribute name, a builtin, or a local — not this module's global
            if inspect.isfunction(obj) and obj.__module__ == extractors.__name__:
                if ref not in blocked:
                    stack.append((ref, obj))
            elif inspect.ismodule(obj):
                modules.add(obj.__name__.split(".")[0])
            elif inspect.isclass(obj) or inspect.isbuiltin(obj):
                # An imported symbol (`BeautifulSoup`): its package version is what matters,
                # because the behaviour lives there rather than here.
                origin = getattr(obj, "__module__", None)
                if origin:
                    modules.add(origin.split(".")[0])
            else:
                constants.add(ref)  # a module-level tunable; its VALUE changes output

    return functions, sorted(constants), sorted(modules)


def _stable_repr(obj: object) -> str:
    """A representation that is identical across processes. Addresses are the trap.

    `repr()` of anything holding function or object references embeds a memory address
    (`<function extract_epub at 0x...>`), which changes every run — so hashing it would make the
    fingerprint non-deterministic and no cache entry would ever read fresh again. Measured, not
    theorised: `_EXTRACTORS` is exactly such a value.

    A compiled regex reprs its pattern, which is the part that changes behaviour. A mapping is
    reduced to its sorted keys because that is its structural contribution — any *function* it
    holds is already hashed on its own by the closure walk.
    """
    if isinstance(obj, re.Pattern):
        return f"re({obj.pattern!r},{obj.flags})"
    if isinstance(obj, Mapping):
        return f"keys({sorted(map(str, obj))})"
    text = repr(obj)
    if " at 0x" in text or " object at " in text:
        return f"<unstable {type(obj).__name__}>"
    return text


def _module_version(name: str) -> str:
    """A dependency's version for fingerprinting. Absent and unversioned are distinct answers."""
    try:
        mod = importlib.import_module(name)
    except Exception:
        return "absent"
    version = getattr(mod, "__version__", None) or getattr(mod, "VersionBind", None)
    return str(version) if version else "unversioned"


def _whole_module_fingerprint() -> str:
    """The pre-KI-48 scope: every function in ``extractors``.

    The safe answer whenever per-format scoping cannot be trusted.
    """
    from doc_assistant import extractors

    h = hashlib.sha256()
    h.update(str(_EXTRACTION_VERSION).encode())
    h.update(str(config.PDF_EXTRACTOR).encode())
    h.update(repr(extractors._TEXT_LAYER_KEPT_MIN).encode())
    for name, fn in sorted(inspect.getmembers(extractors, inspect.isfunction)):
        if getattr(fn, "__module__", None) != extractors.__name__:
            continue
        h.update(name.encode())
        h.update(fn.__code__.co_code)
    for module_name in ("pymupdf", "pymupdf4llm"):
        h.update(f"{module_name}={_module_version(module_name)}".encode())
    return h.hexdigest()[:16]


@cache
def extraction_fingerprint(suffix: str | None = None) -> str:
    """Identity of the extraction pipeline **for one format**: bump-free invalidation (KI-40).

    The cached ``.md`` is derived from **(source bytes, extractor code, extractor config,
    extraction dependencies)**, but freshness used to track only the first — so every extraction
    improvement this project shipped was invisible to anyone who had already ingested. KI-14
    (image placeholders), KI-29 (page markers in the embeddings) and the 2026-08-07 text-layer
    fallback all had that hole: the corpus that most needs a fix was the one guaranteed not to
    get it.

    **Scoped per format since KI-48.** Hashing every function in ``extractors`` meant an
    EPUB/HTML-only change invalidated all 97 PDF caches — a whole-corpus re-extraction for a fix
    that provably could not alter a single PDF — because bytecode hashing cannot tell which format
    a change touched. The scope is now the transitive closure of what *this* format's extraction
    actually executes, derived by walking the call graph rather than from a list somebody has to
    remember to update. ``suffix=None`` keeps the whole-module scope.

    What goes in, and why each:

    * **the reachable functions' bytecode** — a logic change must invalidate the cache *without
      anyone remembering to bump a constant*. Bytecode, not source, so it works in the frozen
      build (PyInstaller ships ``.pyc``, and ``inspect.getsource`` would raise), and so comments
      and docstrings — which cannot change output — do not force a re-extraction;
    * **the module-level constants those functions reference** — a global is referenced by *name*
      from a function, so its value never appears in ``co_code``. ``_TEXT_LAYER_KEPT_MIN`` is
      exactly such a knob, and changing it changes output. Collected by the walk, so a new one
      cannot be forgotten the way the old hand-listed single entry could;
    * **the dependency versions reached from those functions** — a PyMuPDF upgrade changes
      extraction output without a line of our code changing, and that is a real cause, not a
      hypothetical;
    * **``config.PDF_EXTRACTOR``**, for PDFs only — it selects which extractor runs at all.

    **Failure is safe by construction.** Any error in the walk falls back to the whole-module
    fingerprint: over-invalidating costs CPU, while under-invalidating silently serves text no
    current version would produce, which is the KI-40 failure this exists to prevent.

    Cached per suffix: it is asked once per document and cannot change mid-run.
    """
    from doc_assistant import extractors

    try:
        # By the name each function is **bound to on the module**, never by its ``__name__``.
        # `blocked` is tested against the attribute names the walk reads out of ``co_names``, so a
        # function whose ``__name__`` differs from its attribute — an alias, or a decorator without
        # `functools.wraps` — would put a string in `blocked` that the walk can never match, and
        # the other formats' entry points would quietly stay inside this format's scope.
        attr_of = {
            id(obj): attr
            for attr, obj in vars(extractors).items()
            if inspect.isfunction(obj) and obj.__module__ == extractors.__name__
        }
        entries = {attr_of.get(id(fn), fn.__name__) for fn in extractors._EXTRACTORS.values()}
        pdf_entry = attr_of.get(
            id(extractors.extract_pdf_pymupdf), extractors.extract_pdf_pymupdf.__name__
        )
        entries.add(pdf_entry)
        if suffix is None:
            required: str | None = None
            blocked: frozenset[str] = frozenset()  # every entry stays reachable
        else:
            entry = extractors._EXTRACTORS.get(suffix)
            required = (
                attr_of.get(id(entry), getattr(entry, "__name__", "")) if entry else pdf_entry
            )
            blocked = frozenset(entries - {required})
        seeds = () if required is None else (required,)
        functions, constants, modules = _extraction_closure(blocked, seeds)
        # There is deliberately no "is `required` reachable?" assertion here. It read as the
        # KI-40 guard and could not fail: `required` is passed in as a *seed*, and every seed is
        # recorded unconditionally — which is the entire reason seeds exist, since the
        # `_EXTRACTORS` dispatch is a runtime lookup no static walk can follow. The invariant it
        # appeared to protect (a format's own extractor is hashed into that format's
        # fingerprint) is real, and is checked behaviourally by
        # `test_a_change_to_a_formats_entry_point_moves_only_that_format` — where it can fail.
    except Exception as e:
        log.warning("extraction_fingerprint_fallback", suffix=suffix, error=str(e))
        return _whole_module_fingerprint()

    h = hashlib.sha256()
    h.update(str(_EXTRACTION_VERSION).encode())
    h.update(b"scope:" + (suffix or "*").encode())
    if suffix is None or suffix == ".pdf":
        h.update(str(config.PDF_EXTRACTOR).encode())
    # The walk already resolved these; hashing the object it handed back rather than re-resolving
    # the name is what keeps a rename from silently costing a whole-corpus re-extraction.
    for name in sorted(functions):
        h.update(name.encode())
        h.update(functions[name].__code__.co_code)
    for name in constants:
        h.update(f"{name}={_stable_repr(getattr(extractors, name))}".encode())
    for name in modules:
        h.update(f"{name}={_module_version(name)}".encode())
    return h.hexdigest()[:16]


def is_cache_fresh(original: Path, cached: Path) -> bool:
    """Whether ``cached`` still reflects both the source bytes **and** the current extractor.

    The extractor half is KI-40: an mtime-only check let an improved extractor sit unused behind
    caches written by the old one. A cache with no recorded fingerprint (written before this
    existed) is **stale** — which is the point: those are precisely the caches holding output no
    current version would produce.
    """
    if not cached.exists():
        return False
    if cached.stat().st_mtime < original.stat().st_mtime:
        return False
    fp = _fingerprint_path(cached)
    try:
        recorded = fp.read_text(encoding="utf-8").strip()
    except OSError:
        return False  # never fingerprinted, or unreadable → re-extract
    return recorded == document_fingerprint(original)


#: The parsed ignore list, by (path, mtime_ns, size) of the file it was read from: a registry scan
#: asks once per document, and reparsing the same file thousands of times buys nothing.
_ignore_list_memo: dict[tuple[str, int, int], list[dict[str, Any]]] = {}


def ignore_entry_is_valid(entry: object) -> bool:
    """Whether an ignore-list entry has the shape every reader of the list relies on."""
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("size", 0), int)
        and isinstance(entry.get("keep", []), list)
        and isinstance(entry.get("spacing", {}), dict)
    )


def _ignore_list() -> list[dict[str, Any]]:
    """The entries of ``config.REJOIN_IGNORE_PATH`` (ADR-056), or none.

    The file is ``{"version": 1, "documents": [{"sha256", "size", "file", "keep": [...]}]}``.
    An unreadable or malformed file is logged and read as empty rather than failing an ingest
    (inform, don't block) — which means its entries stop applying, so the warning says so. An
    entry of the wrong shape is skipped the same way, and the others still apply.
    """
    path = Path(config.REJOIN_IGNORE_PATH)
    try:
        st = path.stat()
    except OSError:
        return []
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _ignore_list_memo:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            documents = list(data.get("documents", []))
        except (OSError, ValueError, AttributeError, TypeError) as e:
            log.warning(
                "rejoin_ignore_unreadable",
                path=str(path),
                error=str(e),
                hint="its entries do not apply until it is fixed; spacing closes everywhere",
            )
            documents = []
        entries = [e for e in documents if ignore_entry_is_valid(e)]
        if len(entries) < len(documents):
            log.warning(
                "rejoin_ignore_entries_skipped",
                path=str(path),
                skipped=len(documents) - len(entries),
                hint="those entries do not apply until they are fixed",
            )
        _ignore_list_memo.clear()
        _ignore_list_memo[key] = entries
    return _ignore_list_memo[key]


def rejoin_overrides(original: Path) -> tuple[frozenset[str], dict[str, str]]:
    """The ignore list's entries for ``original`` (ADR-056): the runs kept as extracted (``"*"``
    keeps them all), and the runs spaced the user's way, as ``{spaced form: chosen form}``.

    Keyed by the PDF's sha256, so an entry follows the file through a move or a rename. The size
    recorded beside each hash is checked first, so a PDF is hashed only when an entry could be
    its own: an empty or absent list costs one ``stat`` and nothing else."""
    if original.suffix.lower() != ".pdf":
        return frozenset(), {}
    entries = _ignore_list()
    if not entries:
        return frozenset(), {}
    try:
        size = original.stat().st_size
    except OSError:
        return frozenset(), {}
    candidates = [e for e in entries if e.get("size") == size]
    if not candidates:
        return frozenset(), {}
    from doc_assistant.sources_manifest import sha256_file

    digest = sha256_file(original)
    keep: set[str] = set()
    spacing: dict[str, str] = {}
    for e in candidates:
        if e.get("sha256") == digest:
            keep.update(str(k) for k in e.get("keep", []))
            spacing.update({str(k): str(v) for k, v in e.get("spacing", {}).items()})
    return frozenset(keep), spacing


def rejoin_ignore_entries(original: Path) -> frozenset[str]:
    """The spaced forms the ignore list keeps for ``original`` (ADR-056); ``"*"`` keeps all."""
    return rejoin_overrides(original)[0]


def document_fingerprint(original: Path) -> str:
    """The fingerprint one document's cache must carry: its format's, plus its ignore-list entries.

    A document with no entries carries exactly ``extraction_fingerprint(suffix)``, so the list
    changes nothing for anyone who has not written one. A document with entries carries a digest
    of them too, so adding, changing or removing an entry re-extracts that document alone."""
    base = extraction_fingerprint(original.suffix.lower())
    keep, spacing = rejoin_overrides(original)
    if not keep and not spacing:
        return base
    lines = sorted(keep) + [f"{k}\t{v}" for k, v in sorted(spacing.items())]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:8]
    return f"{base}+keep:{digest}"


def spacing_record_path(cached: Path) -> Path:
    """Sibling of the cached ``.md`` holding the letter-spacing runs its extraction found (ADR-056
    point 7). A separate file for the same reason as the fingerprint: the ``.md``'s bytes are the
    document's identity."""
    return cached.with_name(cached.name + ".spacing.json")


def read_spacing_record(cached: Path) -> list[dict[str, Any]]:
    """The runs recorded beside ``cached``, or none (no record: nothing was found, or the cache
    predates the record). An unreadable record is logged and read as empty."""
    path = spacing_record_path(cached)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        log.warning("spacing_record_unreadable", path=str(path), error=str(e))
        return []
    runs = data.get("runs", []) if isinstance(data, dict) else []
    return [r for r in runs if isinstance(r, dict)]


_PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")


def locate_spacing_runs(text: str, runs: list[dict[str, Any]]) -> None:
    """Give each recorded run its character span in ``text``, in place (ADR-056 point 8).

    ``start`` and ``end`` are where the run's ``written`` form sits in the cached markdown, so a
    reader can check ``text[start:end] == written`` before acting on a run; both are None for a
    run that is not there (``in_text`` False, or its page holds no such text). A run is looked for
    in its own page's part of the text, and an occurrence another run of that page already claimed
    is passed over, so a running head on every page, or a closed word the body also uses, keeps
    one span per run.

    Here, not in ``extractors``: a span describes the text written, not how it was extracted, so
    changing how spans are found must not re-extract anything (KI-48). A later pass that rewrites
    the cache (the table splices) can move them; the reader's check is what catches that."""
    marks = list(_PAGE_MARKER.finditer(text))
    pages: dict[int, tuple[int, int]] = {}
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pages.setdefault(int(mark.group(1)), (mark.end(), end))
    claimed: set[int] = set()
    for run in runs:
        run["start"] = run["end"] = None
        written = run.get("written")
        section = pages.get(run.get("page", -1))
        if section is None or not run.get("in_text") or not written:
            continue
        lo, hi = section
        at = text.find(written, lo, hi)
        while at >= 0 and at in claimed:
            at = text.find(written, at + 1, hi)
        if at >= 0:
            claimed.add(at)
            run["start"], run["end"] = at, at + len(written)


def _stale_reason(original: Path, cached: Path) -> str:
    """Why a cache entry is being re-extracted — for the log line, not for control flow.

    Worth distinguishing: "the file changed" is expected and self-explanatory, while "the extractor
    changed" means a one-off re-extraction of the whole library and would otherwise look like an
    unexplained slowdown.
    """
    if not cached.exists():
        return "no_cache"
    if cached.stat().st_mtime < original.stat().st_mtime:
        return "source_changed"
    return "extractor_changed"


def write_cache(
    cached: Path,
    text: str,
    *,
    source: Path,
    spacing_record: list[dict[str, Any]] | None = None,
) -> None:
    """Write a cache entry: the markdown **and** the fingerprint of the extractor that made it.

    ``source`` is the file the text was extracted *from*, and it is required rather than optional
    because the fingerprint is scoped per format since KI-48: a caller that guessed wrong — or
    omitted it and got the whole-module scope — would write an entry that `is_cache_fresh` can
    never match, re-extracting that document on every single run. Passing the source makes the
    two sides derive the scope from the same fact.

    A cache entry is that *pair* — a ``.md`` without its ``.fp`` reads as stale (KI-40). This is
    the single place that fact lives, so callers cannot half-write one. Tests that fabricate a
    "already ingested" state use it for the same reason: five separate hand-rolled cache writes are
    exactly how the definition drifted the first time.

    Atomic, because this cached ``.md`` is the source-of-truth the next ingest re-hashes and a
    crash mid-write must not leave a truncated cache that ``is_cache_fresh`` trusts (the same
    hazard the table-splice writers share — see ``fsutil.atomic_write_text``).

    Fingerprint **last**: if that write fails, the entry reads as stale and is re-extracted next
    run — wasteful but correct. The reverse order could vouch for a truncated ``.md``.

    ``spacing_record`` is the letter-spacing runs this text's extraction found (ADR-056 point 7),
    each given its span in ``text`` (:func:`locate_spacing_runs`) and written between the two, so
    a fingerprint never vouches for a record of other text. An empty list removes a previous
    record; ``None`` leaves it alone, for a caller that rewrites the text without extracting it
    again.
    """
    atomic_write_text(cached, text)
    if spacing_record is not None:
        record = spacing_record_path(cached)
        if spacing_record:
            locate_spacing_runs(text, spacing_record)
            atomic_write_text(record, json.dumps({"version": 1, "runs": spacing_record}))
        else:
            record.unlink(missing_ok=True)
    atomic_write_text(_fingerprint_path(cached), document_fingerprint(source))


def load_or_extract(original: Path) -> str:
    """The cached markdown for ``original``, extracting it first when the cache is stale.

    Raises ``IngestRefusedError`` before opening a file that is over the size caps or an archive
    shaped like a zip bomb (docs/security.md S-1). A fresh cache is served without the check: the
    file is not opened. The check sits here rather than in ``extract_to_markdown`` because that
    function's call graph is the extraction fingerprint — a change there re-extracts every
    document (KI-48).
    """
    cached = get_cache_path(original)
    if is_cache_fresh(original, cached):
        return cached.read_text(encoding="utf-8")

    refusal = ingest_refusal(original)
    if refusal is not None:
        raise IngestRefusedError(refusal)
    log.info("extracting", file=original.name, reason=_stale_reason(original, cached))
    keep, spacing = rejoin_overrides(original)
    record: list[dict[str, Any]] = []
    text = extract_to_markdown(
        original,
        pdf_extractor=config.PDF_EXTRACTOR,
        keep_spaced=keep,
        spacing=spacing,
        record=record,
    )
    write_cache(cached, text, source=original, spacing_record=record)
    return text


def doc_hash(text: str) -> str:
    """Content-only hash. Path-independent so documents survive moves/renames."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]
