"""Corpus-size-safe reads over a Chroma collection.

Chroma's SQLite backend binds **one SQL parameter per returned row**, so an unpaged
whole-collection ``get()`` does not merely get slow as the corpus grows — it **fails outright**
with ``too many SQL variables`` once the collection passes SQLite's parameter ceiling (32766 on a
modern build).

This was hit for real on **2026-07-25**: ingesting the transferred corpus took the parent-child
store from ~16k to **33,163** chunks at **97 documents**, and every whole-store read broke at once
— including `pipeline.py`'s BM25 index build, which runs in ``RAGPipeline.__init__``, so the answer
path could not even construct. That is 1% of the way to the 10k-document robustness contract
(`.claude/CONTEXT.md`), and the failure mode is a hard error rather than a slowdown, which is why
paging lives in one shared helper rather than being re-derived at each call site.

Every whole-collection read in `src/` goes through :func:`get_all`. Filtered reads that are bounded
by construction (one document, one hash) may call ``get()`` directly.
"""

from __future__ import annotations

from typing import Any

#: Rows per page. A structural bound on SQL parameters per statement, **not** a corpus-tuned
#: threshold: any value comfortably under SQLite's ceiling behaves identically, and the paging —
#: not the number — is what makes the read safe at any corpus size.
PAGE_SIZE = 5000


def _is_array(value: Any) -> bool:
    """True for a numpy-style array (what chromadb returns for ``embeddings``).

    Duck-typed rather than ``isinstance(value, np.ndarray)`` so this module keeps no numpy
    import of its own: it exists to page a store, not to do numerics.
    """
    return hasattr(value, "shape") and hasattr(value, "__len__")


def get_all(
    # `Any`, deliberately: the two callers are a raw `chromadb.Collection` and LangChain's
    # `Chroma` wrapper. Both expose a `get(where=, include=, limit=, offset=)`, but their
    # signatures differ in the `include` literal type, so no Protocol satisfies both — a
    # structural type here would be a fiction that only type-checks one of the two.
    collection: Any,
    *,
    where: dict[str, Any] | None = None,
    include: list[str] | None = None,
    page_size: int = PAGE_SIZE,
) -> dict[str, Any]:
    """Read a whole collection in bounded pages; same return shape as one big ``get()``.

    ``include`` is passed through unchanged (``["documents", "metadatas"]``, ``["embeddings"]``,
    or ``[]`` for ids only). Pages are concatenated per key, so the caller sees exactly what an
    unpaged call would have returned — ``ids`` always present, the rest as requested. A short page
    ends the walk, so an empty collection costs one query and returns empty lists.
    """
    kwargs: dict[str, Any] = {}
    if where is not None:
        kwargs["where"] = where
    if include is not None:
        kwargs["include"] = include

    out: dict[str, Any] = {}
    offset = 0
    while True:
        page = collection.get(limit=page_size, offset=offset, **kwargs)
        n = len(page.get("ids") or [])
        for key, value in page.items():
            if isinstance(value, list) or _is_array(value):
                # `_is_array` is load-bearing, not defensive: chromadb returns `embeddings` as a
                # **numpy array** and every other key as a plain list. An ndarray fails
                # `isinstance(value, list)`, so without this branch the fall-through below kept
                # only the FIRST page of embeddings while `metadatas` accumulated all of them —
                # a silent truncation to `page_size` rows, invisible because the caller zipped
                # the two with `strict=False`. See KI-31.
                out.setdefault(key, []).extend(value)
            elif key not in out:
                out[key] = value
        if n < page_size:
            # Guarantee the requested keys exist even when the collection is empty.
            out.setdefault("ids", [])
            for key in include or []:
                out.setdefault(key, [])
            return out
        offset += n
