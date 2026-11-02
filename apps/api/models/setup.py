"""First-run setup wire models (ADR-034, ADR-060).

Mirrors ``doc_assistant.readiness`` one-for-one, plus the write bodies (a key, a server of your
own, the *does not bill me* tick). The read side is a *serialization* of the library dataclasses —
the shell adds no fields and makes no judgements of its own (``apps/`` is a thin shell, CONTEXT
rule 3).

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
    name: str
    group: Literal["local", "cloud", "server"]
    destination: str
    kind: Literal["claude", "ollama", "preset", "server"]
    address: str
    reach: Literal["this_computer", "local_network", "elsewhere"]
    billing: str
    billing_kind: Literal["vendor", "nobody", "unknown", "your_word"]
    can_tick: bool
    takes_key: bool
    key_page: str | None = None
    key_status: Literal["accepted", "refused"] | None = None
    models_fetched_at: str | None = None
    windows: dict[str, int] = Field(default_factory=dict)
    remote: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    sends: list[str] = Field(default_factory=list)


class SetupStepModel(BaseModel):
    """One outstanding (or completed) first-run task: one row of Getting started."""

    id: str
    title: str
    detail: str
    done: bool
    status: str
    action: str | None = None
    offline: bool = False


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
    active_name: str
    active_destination: str
    active_billing: str


class ApiKeyUpdate(BaseModel):
    """``POST /api/setup/providers/{id}/key`` — the key to store.

    ``min_length=1`` so an empty body is a 422 rather than a stored blank that would read as
    "configured" while resolving to nothing.
    """

    key: str = Field(min_length=1)


class ModelListModel(BaseModel):
    """What a server said when asked for its models (ADR-060, choice 8). Held in memory only."""

    status: Literal["ok", "refused", "needs_key", "unreachable", "error"]
    models: list[str] = Field(default_factory=list)
    detail: str = ""
    fetched_at: str = ""
    windows: dict[str, int] = Field(default_factory=dict)


class ApiKeyResult(BaseModel):
    """The outcome of storing a key: whether it verified, and the refreshed setup state.

    ``verification`` is ``"ok"`` (the server accepted the key) or ``"unreachable"`` (stored, but
    the check could not complete — no network, a proxy, a timeout). A rejected key is **not** a
    result: it is a 400, because storing a key the server refuses would leave a broken install
    looking configured.
    """

    stored: bool
    verification: Literal["ok", "unreachable"]
    detail: str
    setup: SetupStateModel


class ServerUpdate(BaseModel):
    """``POST /api/setup/servers`` — *Test connection* on a server of your own.

    ``id`` absent adds a server; present, changes that one. ``key`` absent keeps the stored key
    when the address is unchanged; a changed address always drops it (the key is bound to its
    address).
    """

    id: str | None = Field(default=None, max_length=80)
    name: str = Field(min_length=1, max_length=80)
    address: str = Field(min_length=1, max_length=500)
    key: str | None = Field(default=None, max_length=4096)


class ServerResult(BaseModel):
    """What *Test connection* found, and the setup state with the server in it."""

    id: str
    listing: ModelListModel
    setup: SetupStateModel


class BillingUpdate(BaseModel):
    """``POST /api/setup/providers/{id}/billing`` — the user's word that a server bills nobody."""

    not_billed: bool
