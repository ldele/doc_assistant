"""First-run setup wire models (ADR-034).

Mirrors ``doc_assistant.readiness`` one-for-one, plus the one write body (an API key). The read
side is a *serialization* of the library dataclasses — the shell adds no fields and makes no
judgements of its own (``apps/`` is a thin shell, CONTEXT rule 3).

**No key material ever crosses this boundary outbound.** ``ProviderReadinessModel`` carries
``key_source`` and a last-4 ``key_hint``; there is no field that could hold a key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProviderReadinessModel(BaseModel):
    """Whether one provider can serve the next turn, and what would fix it if not."""

    id: str
    paid: bool
    configured: bool
    reachable: bool | None
    ready: bool
    detail: str
    action: str | None = None
    key_source: Literal["env", "app"] | None = None
    key_hint: str | None = None
    models: list[str] = Field(default_factory=list)


class SetupStepModel(BaseModel):
    """One outstanding (or completed) first-run task."""

    id: str
    title: str
    detail: str
    done: bool
    action: str | None = None


class SetupStateModel(BaseModel):
    """``GET /api/setup`` — the whole first-run picture."""

    providers: list[ProviderReadinessModel]
    active_provider: str
    active_model: str
    active_ready: bool
    chunk_count: int
    document_count: int
    ollama_host: str
    steps: list[SetupStepModel]
    ready: bool


class ApiKeyUpdate(BaseModel):
    """``POST /api/setup/anthropic-key`` — the key to store.

    ``min_length=1`` so an empty body is a 422 rather than a stored blank that would read as
    "configured" while resolving to nothing.
    """

    key: str = Field(min_length=1)


class ApiKeyResult(BaseModel):
    """The outcome of storing a key: whether it verified, and the refreshed setup state.

    ``verification`` is ``"ok"`` (the API accepted the key) or ``"unreachable"`` (stored, but the
    check could not complete — no network, a proxy, a timeout). A rejected key is **not** a result:
    it is a 400, because storing a key the API refuses would leave a broken install looking
    configured.
    """

    stored: bool
    verification: Literal["ok", "unreachable"]
    detail: str
    setup: SetupStateModel
