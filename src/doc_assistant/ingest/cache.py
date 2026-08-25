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
import os
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import CodeType
from typing import Any

import structlog

from doc_assistant import config
from doc_assistant.extractors import extract_to_markdown
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
    blocked: frozenset[str], seeds: tuple[Any, ...] = ()
) -> tuple[list[str], list[str], list[str]]:
    """Walk out from ``extract_to_markdown`` **and** ``seeds``, never entering ``blocked``.

    Returns (function names, constant names, top-level module names), each sorted.

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
    """
    from doc_assistant import extractors

    functions: dict[str, Any] = {}
    constants: set[str] = set()
    modules: set[str] = set()
    stack = [extractors.extract_to_markdown, *seeds]

    while stack:
        fn = stack.pop()
        name = fn.__name__
        if name in functions:
            continue
        functions[name] = fn
        names, imports = _referenced(fn.__code__)
        modules |= {m.split(".")[0] for m in imports}
        for ref in names:
            obj = getattr(extractors, ref, None)
            if obj is None:
                continue  # an attribute name, a builtin, or a local — not this module's global
            if inspect.isfunction(obj) and obj.__module__ == extractors.__name__:
                if ref not in blocked:
                    stack.append(obj)
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

    return sorted(functions), sorted(constants), sorted(modules)


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
        entries = {fn.__name__ for fn in extractors._EXTRACTORS.values()}
        entries.add(extractors.extract_pdf_pymupdf.__name__)
        if suffix is None:
            required: str | None = None
            blocked: frozenset[str] = frozenset()  # every entry stays reachable
        else:
            entry = extractors._EXTRACTORS.get(suffix)
            required = entry.__name__ if entry else extractors.extract_pdf_pymupdf.__name__
            blocked = frozenset(entries - {required})
        seeds = () if required is None else (getattr(extractors, required),)
        functions, constants, modules = _extraction_closure(blocked, seeds)
        if required is not None and required not in functions:
            raise RuntimeError(f"{required} is unreachable from extract_to_markdown")
    except Exception as e:
        log.warning("extraction_fingerprint_fallback", suffix=suffix, error=str(e))
        return _whole_module_fingerprint()

    h = hashlib.sha256()
    h.update(str(_EXTRACTION_VERSION).encode())
    h.update(b"scope:" + (suffix or "*").encode())
    if suffix is None or suffix == ".pdf":
        h.update(str(config.PDF_EXTRACTOR).encode())
    for name in functions:
        h.update(name.encode())
        h.update(getattr(extractors, name).__code__.co_code)
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
    return recorded == extraction_fingerprint(original.suffix.lower())


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


def write_cache(cached: Path, text: str, *, source: Path) -> None:
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
    """
    atomic_write_text(cached, text)
    atomic_write_text(_fingerprint_path(cached), extraction_fingerprint(source.suffix.lower()))


def load_or_extract(original: Path) -> str:
    cached = get_cache_path(original)
    if is_cache_fresh(original, cached):
        return cached.read_text(encoding="utf-8")

    log.info("extracting", file=original.name, reason=_stale_reason(original, cached))
    text = extract_to_markdown(original, pdf_extractor=config.PDF_EXTRACTOR)
    write_cache(cached, text, source=original)
    return text


def doc_hash(text: str) -> str:
    """Content-only hash. Path-independent so documents survive moves/renames."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]
