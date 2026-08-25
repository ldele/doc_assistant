"""Extraction cache + content hashing — the bottom layer of the ingest package.

Turns a source file into its cached markdown (extracting + caching on a miss) and
hashes that content. The cached ``.md`` is the source-of-truth the rest of the
pipeline re-reads, so writes go through the atomic helper. Path/extractor config is
read dynamically (``config.X``) so a single seam is monkeypatch-able in tests.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
from functools import lru_cache
from pathlib import Path

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


@lru_cache(maxsize=1)
def extraction_fingerprint() -> str:
    """Identity of the extraction pipeline: bump-free invalidation of the markdown cache (KI-40).

    The cached ``.md`` is derived from **(source bytes, extractor code, extractor config,
    extraction dependencies)**, but freshness used to track only the first — so every extraction
    improvement this project shipped was invisible to anyone who had already ingested. KI-14
    (image placeholders), KI-29 (page markers in the embeddings) and the 2026-08-07 text-layer
    fallback all had that hole: the corpus that most needs a fix was the one guaranteed not to get
    it.

    Components, and why each:

    * **the extractors' bytecode** — every function defined in ``extractors``, hashed by
      ``co_code``. This is the sparse-index precedent applied here: a logic change must invalidate
      the cache *without anyone remembering to bump a constant*. Bytecode, not source, so it works
      in the frozen build (PyInstaller ships ``.pyc``, and ``inspect.getsource`` would raise), and
      so comments and docstrings — which cannot change output — do not force a re-extraction of
      the whole library;
    * **the tunables bytecode cannot see** — module-level constants are referenced by *name* from a
      function, so their values never appear in ``co_code``. ``_TEXT_LAYER_KEPT_MIN`` is exactly
      such a knob, and changing it changes output;
    * **``config.PDF_EXTRACTOR``** — selects which extractor runs at all;
    * **the extraction dependencies' versions** — a PyMuPDF upgrade changes extraction output
      without a line of our code changing, and that is a real cause, not a hypothetical.

    Cached for the process: it is asked once per document and the answer cannot change mid-run.
    """
    from doc_assistant import extractors

    h = hashlib.sha256()
    h.update(str(_EXTRACTION_VERSION).encode())
    h.update(str(config.PDF_EXTRACTOR).encode())
    h.update(repr(extractors._TEXT_LAYER_KEPT_MIN).encode())

    for name, fn in sorted(inspect.getmembers(extractors, inspect.isfunction)):
        if getattr(fn, "__module__", None) != extractors.__name__:
            continue  # re-exported from elsewhere; not this module's behaviour
        h.update(name.encode())
        h.update(fn.__code__.co_code)

    for module_name in ("pymupdf", "pymupdf4llm"):
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", None) or getattr(mod, "VersionBind", "?")
        except Exception:
            version = "absent"
        h.update(f"{module_name}={version}".encode())

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
    return recorded == extraction_fingerprint()


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


def write_cache(cached: Path, text: str) -> None:
    """Write a cache entry: the markdown **and** the fingerprint of the extractor that made it.

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
    atomic_write_text(_fingerprint_path(cached), extraction_fingerprint())


def load_or_extract(original: Path) -> str:
    cached = get_cache_path(original)
    if is_cache_fresh(original, cached):
        return cached.read_text(encoding="utf-8")

    log.info("extracting", file=original.name, reason=_stale_reason(original, cached))
    text = extract_to_markdown(original, pdf_extractor=config.PDF_EXTRACTOR)
    write_cache(cached, text)
    return text


def doc_hash(text: str) -> str:
    """Content-only hash. Path-independent so documents survive moves/renames."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]
