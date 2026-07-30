"""Persisted snapshot of the BM25 arm — a launch accelerator, never a source of truth (ADR-035).

`RAGPipeline.__init__` otherwise reads **every chunk** out of Chroma and re-tokenises the whole
corpus on every launch. Measured at 97 documents / 33,105 chunks that is 1.058 s of store read plus
0.615 s of index build; it is the only startup component that scales with the corpus, so at the
10,000-document robustness contract it is the difference between a slow launch and an app that
cannot open.

**What is stored, and what deliberately is not.** The payload holds only stdlib types —
`(text, metadata)` tuples, lists of token strings, and (since payload v2) a `(doc_hash,
parent_index) -> parent_text` map. The `Document` / `BM25Okapi` / `BM25Retriever` objects are
reconstructed at load. Pickling the live retriever would be ~0.15 s faster and would make the
on-disk format an implementation detail of a third-party class, where a langchain upgrade could
deserialise into a subtly different object rather than fail outright. It is a trade ADR-035
rejects: this blob cannot hold a foreign class, so that mode does not exist.

**Payload v2 (2026-07-30, KI-32 step 1) stores each parent text ONCE.** Chroma denormalises a
parent's full text into every one of its children (measured 5.5 on the live corpus: 6,045 parents
across 33,105 chunks), so the v1 payload carried that many copies of every parent. Measured effect:
the file went **85.2 MB -> 39.9 MB (2.1x)** and the in-RAM corpus **265 MB -> 195 MB (1.36x)**. The
map is keyed on `(doc_hash, parent_index)` and the per-chunk metadata no longer holds
`parent_text`; `pipeline` re-attaches the text for the handful of parents a turn actually returns.

**Every failure path falls back to the live build.** A missing, stale, corrupt, truncated or
unreadable cache logs and returns ``None``. There is no code path in which a bad cache yields a
*wrong* index rather than a *slower* launch — which is the only property that makes caching a
retrieval input defensible at all.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import pickle  # nosec B403 - local data-home file, fingerprint-verified, stdlib types only
import tempfile
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

#: Bumped when the payload *shape* changes in a way the content fingerprint cannot see.
#: Tokeniser changes do **not** need a bump — they are fingerprinted from source (see
#: `_fingerprint`).
#: v2 (2026-07-30, KI-32 step 1): `parent_text` moved out of per-chunk metadata into a deduplicated
#: `parents` map. A v1 file is stale by fingerprint, so an existing snapshot is rebuilt, not
#: misread.
_CACHE_VERSION = 2

CACHE_FILENAME = "bm25_index.pkl"

#: One entry per parent block: ``(doc_hash, parent_index) -> parent_text``.
ParentTexts = dict[tuple[str, int], str]


def _cache_path(chroma_path: str) -> Path:
    """Sit the snapshot beside the store it mirrors, so the two travel together."""
    return Path(chroma_path).parent / CACHE_FILENAME


def _fingerprint(collection_name: str, chunk_ids: list[str]) -> str:
    """Identify the exact corpus + tokeniser this snapshot was built from.

    Components, and why each is here:

    * **the chunk ids themselves**, sorted — the staleness signal. Any add, removal or replacement
      changes the set, and a `--rebuild` mints fresh UUIDs, so it catches the case a bare *count*
      cannot: an edit that replaced as many chunks as it removed. Reading ids only is cheap (a
      measured 16 ms per 5,000-row page) because it pulls no documents, metadata or embeddings.
      Sorted, so the hash cannot depend on Chroma's page ordering.
    * **collection name** — an embedding-model change points retrieval at a different collection.
    * **the tokeniser's source text** — the index is tokens, so a tokeniser change invalidates it.
      Hashing `inspect.getsource` means that self-invalidates instead of depending on someone
      remembering to bump `_CACHE_VERSION`. It sees `tokenize` itself, not helpers it calls, so it
      is a strong signal rather than a complete one — `_CACHE_VERSION` covers the rest.

    **Rejected: the store file's mtime.** The obvious cheap signal is wrong here — *opening* a
    `chromadb.PersistentClient` rewrites `chroma.sqlite3`'s mtime even for a pure read, so a
    fingerprint built on it invalidates the cache on the very next launch. Measured, not assumed:
    the first implementation did exactly that and never once hit.
    """
    from doc_assistant.knowledge.keywords import tokenize

    digest = hashlib.sha256()
    digest.update(f"{_CACHE_VERSION}\x00{collection_name}\x00{len(chunk_ids)}\x00".encode())
    for chunk_id in sorted(chunk_ids):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")

    try:
        digest.update(inspect.getsource(tokenize).encode("utf-8"))
    except (OSError, TypeError):
        # Frozen builds strip source. Degrade to the version constant rather than failing —
        # a frozen app ships one fixed tokeniser anyway.
        digest.update(b"no-source")

    return digest.hexdigest()


def enabled() -> bool:
    """``DOC_BM25_CACHE=0`` turns the snapshot off without a code change (ADR-035 escape hatch)."""
    return os.getenv("DOC_BM25_CACHE", "1").strip().lower() not in {"0", "false", "no"}


def load(
    chroma_path: str, collection_name: str, chunk_ids: list[str]
) -> tuple[list[tuple[str, dict[str, Any]]], list[list[str]], ParentTexts] | None:
    """Return ``(docs, tokens, parents)`` from the snapshot, or ``None`` to signal "build it live".

    ``None`` is the answer for every unhappy path — disabled, absent, stale, corrupt, or holding a
    payload that fails its own internal consistency check.

    ``parents`` maps ``(doc_hash, parent_index)`` to the parent text that the per-chunk
    metadata no longer carries (payload v2). It is legitimately **empty** in flat
    (non-parent-child) mode, so an empty map is not a reason to refuse the payload.
    """
    if not enabled():
        log.info("bm25_cache_disabled")
        return None

    path = _cache_path(chroma_path)
    if not path.exists():
        return None

    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)  # nosec B301 - see module docstring
    except Exception as e:
        log.warning("bm25_cache_unreadable", error=str(e), path=str(path))
        return None

    if not isinstance(payload, dict):
        log.warning("bm25_cache_malformed", path=str(path))
        return None

    expected = _fingerprint(collection_name, chunk_ids)
    if payload.get("fingerprint") != expected:
        log.info("bm25_cache_stale", hint="corpus or tokenizer changed; rebuilding")
        return None

    docs = payload.get("docs")
    tokens = payload.get("tokens")
    if not isinstance(docs, list) or not isinstance(tokens, list) or len(docs) != len(tokens):
        # A length mismatch here would silently mis-pair every document with another's terms —
        # the same class of failure as KI-31. Refuse the payload instead.
        log.warning(
            "bm25_cache_inconsistent",
            docs=len(docs) if isinstance(docs, list) else None,
            tokens=len(tokens) if isinstance(tokens, list) else None,
        )
        return None

    parents = payload.get("parents")
    if not isinstance(parents, dict):
        # v2 always writes a dict (empty in flat mode). Anything else means a payload this code
        # does not understand, and a missing parent map degrades retrieval silently — a
        # parent-child turn would drop every BM25-only hit rather than expand it. Refuse.
        log.warning("bm25_cache_no_parent_map", type=type(parents).__name__)
        return None

    log.info("bm25_cache_hit", chunks=len(docs), parents=len(parents))
    return docs, tokens, parents


def save(
    chroma_path: str,
    collection_name: str,
    chunk_ids: list[str],
    docs: list[tuple[str, dict[str, Any]]],
    tokens: list[list[str]],
    parents: ParentTexts,
) -> bool:
    """Write the snapshot. Returns whether it landed; a failure is logged, never raised.

    Written to a temp file in the destination directory and then moved, so an interrupted or
    out-of-space write cannot leave a half-file that the next launch has to detect as corrupt.
    """
    if not enabled():
        return False
    if not docs:
        # Nothing to accelerate, and an empty snapshot would only be a stale-file hazard for the
        # first real ingest. The robustness contract's 0-document case is a no-op here.
        return False

    path = _cache_path(chroma_path)
    payload = {
        "fingerprint": _fingerprint(collection_name, chunk_ids),
        "docs": docs,
        "tokens": tokens,
        "parents": parents,
    }
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(payload, fh, protocol=5)
        os.replace(tmp_name, path)
        tmp_name = None
        log.info(
            "bm25_cache_written",
            chunks=len(docs),
            parents=len(parents),
            mb=round(path.stat().st_size / 1e6, 1),
        )
        return True
    except Exception as e:
        log.warning("bm25_cache_write_failed", error=str(e), path=str(path))
        return False
    finally:
        if tmp_name:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
