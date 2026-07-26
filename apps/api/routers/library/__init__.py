"""Library router — composed from its three sub-domains.

``/api/library/*`` covers three separate things, and they were one 300-line module until this
split: ``documents`` (the browser: list, chunks, connections, metadata, delete), ``folders``
(ADR-025 F1 organisation) and ``keywords`` (tag families, ADR-015). Each is its own module with
its own ``APIRouter``; this file only composes them so ``main.create_app`` keeps calling
``app.include_router(library.router)`` unchanged.

The three path prefixes (``/documents``, ``/folders``, ``/keyword-families``) are disjoint, so
include order carries no route-matching meaning here — it is the pre-split declaration order,
kept so the diff reads as a move. Order *within* each module is still load-bearing and was
preserved verbatim.
"""

from __future__ import annotations

from fastapi import APIRouter

from apps.api.routers.library import documents, folders, keywords

router = APIRouter()
router.include_router(documents.router)
router.include_router(folders.router)
router.include_router(keywords.router)

__all__ = ["documents", "folders", "keywords", "router"]
