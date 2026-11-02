"""Answer providers beyond Claude and Ollama, and the two facts the screens state about each.

ADR-060 (amended 2026-11-02). Every provider other than Claude and Ollama is spoken to through
one OpenAI-compatible client (:mod:`doc_assistant.compatible`). This module holds what that client
needs to know and what the panel must say:

* **The presets**, one row each in :data:`PRESETS`. No code branches on a preset's id: a new vendor
  is a row. A preset's address is fixed; any other address is a server of your own.
* **Servers of your own**, a name and an address in ``settings.json`` (no secret there; the key is
  bound to the address in ``credentials.json``).
* **Where the text goes** (:func:`route_of`), worked out per model from the address and, when the
  server reports one, the model's upstream host (Ollama's ``remote_host``). Three outcomes: this
  computer, your local network, or *goes to <host>*. A name is never looked up: the sentence is a
  promise, and a name can point anywhere.
* **Who bills** (:class:`Billing`), a fact of its own: a preset's vendor, nobody, Ollama for a
  remote model, or not known until the user ticks *This server does not bill me*. The CLI credit
  guard trips on *billed, or not known* (:attr:`Billing.guard_trips`).
* **The model lists**, fetched on request only and kept in memory with the time they were fetched
  (choice 8). Nothing here touches the network unless a function says it does.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit

import structlog

from doc_assistant import app_settings, config, credentials

log = structlog.get_logger(__name__)

ANTHROPIC_ADDRESS = "https://api.anthropic.com"
"""Where the Anthropic SDK sends a request unless ``ANTHROPIC_BASE_URL`` says otherwise."""

NATIVE_PROVIDERS: tuple[str, ...] = ("anthropic", "ollama")
"""The two providers with adapters of their own (ADR-011, ADR-034)."""


# ============================================================
# The presets
# ============================================================


@dataclass(frozen=True)
class Preset:
    """One vendor's OpenAI-compatible endpoint, as its docs give it."""

    id: str
    name: str
    address: str
    """The base address; ``/models`` and ``/chat/completions`` are joined to it. Fixed."""
    billed_by: str | None
    """Who bills a call, by name; ``None`` when nobody does (a server on this computer)."""
    key_page: str | None
    """Where to get a key; ``None`` when the server takes none."""
    checked: str
    """The date the address was checked against the vendor's docs."""
    key_check: str | None = None
    """A path under the address that checks a key, for a vendor whose model list needs none."""


PRESETS: tuple[Preset, ...] = (
    Preset(
        id="lmstudio",
        name="LM Studio",
        address="http://localhost:1234/v1",
        billed_by=None,
        key_page=None,
        checked="2026-11-01",
    ),
    Preset(
        id="openai",
        name="OpenAI",
        address="https://api.openai.com/v1",
        billed_by="OpenAI",
        key_page="https://platform.openai.com/api-keys",
        checked="2026-11-02",
    ),
    Preset(
        id="gemini",
        name="Google Gemini",
        address="https://generativelanguage.googleapis.com/v1beta/openai",
        billed_by="Google",
        key_page="https://aistudio.google.com/apikey",
        checked="2026-11-02",
    ),
    Preset(
        id="mistral",
        name="Mistral",
        address="https://api.mistral.ai/v1",
        billed_by="Mistral",
        key_page="https://console.mistral.ai/api-keys",
        checked="2026-11-02",
    ),
    Preset(
        id="groq",
        name="Groq",
        address="https://api.groq.com/openai/v1",
        billed_by="Groq",
        key_page="https://console.groq.com/keys",
        checked="2026-11-01",
    ),
    Preset(
        id="openrouter",
        name="OpenRouter",
        address="https://openrouter.ai/api/v1",
        billed_by="OpenRouter",
        key_page="https://openrouter.ai/keys",
        checked="2026-11-02",
        # Its model list answers without a key (checked 2026-11-02), so the list proves nothing
        # about one; `GET /key` refuses a bad key and bills nothing.
        key_check="/key",
    ),
)

_PRESETS_BY_ID = {p.id: p for p in PRESETS}


def preset(provider: str) -> Preset | None:
    return _PRESETS_BY_ID.get(provider.strip().lower())


# ============================================================
# Servers of your own
# ============================================================

SERVER_PREFIX = "server-"
"""Every server of your own has an id with this prefix, so it can never take a preset's."""


@dataclass(frozen=True)
class Server:
    id: str
    name: str
    address: str


def servers() -> tuple[Server, ...]:
    return tuple(Server(**row) for row in app_settings.get_servers())


def server(provider: str) -> Server | None:
    key = provider.strip().lower()
    return next((s for s in servers() if s.id == key), None)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "server"


def check_address(address: str) -> str:
    """``address`` cleaned for storage, or ``ValueError`` saying how to write one."""
    cleaned = address.strip().rstrip("/")
    scheme = urlsplit(cleaned).scheme.lower() if "://" in cleaned else ""
    if scheme not in ("http", "https") or not host_of(cleaned):
        raise ValueError(
            "Write the address with http:// or https://, a host and the path, "
            "e.g. http://localhost:1234/v1."
        )
    return cleaned


def save_server(name: str, address: str, server_id: str | None = None) -> Server:
    """Add a server of your own, or change one. Changing its address removes its key, because a key
    is sent only to the address it was saved with (choice 3)."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Give the server a name.")
    clean_address = check_address(address)
    rows = app_settings.get_servers()
    if server_id is not None:
        row = next((r for r in rows if r["id"] == server_id), None)
        if row is None:
            raise KeyError(server_id)
        if not credentials.same_address(row["address"], clean_address):
            credentials.clear_bound_key(server_id)
        row.update(name=clean_name, address=clean_address)
        saved = Server(server_id, clean_name, clean_address)
    else:
        taken = {r["id"] for r in rows}
        base = SERVER_PREFIX + _slug(clean_name)
        new_id, n = base, 2
        while new_id in taken:
            new_id, n = f"{base}-{n}", n + 1
        rows.append({"id": new_id, "name": clean_name, "address": clean_address})
        saved = Server(new_id, clean_name, clean_address)
    app_settings.set_servers(rows)
    return saved


def remove_server(server_id: str) -> bool:
    """Forget a server of your own, its key and its tick. Returns whether it existed."""
    rows = app_settings.get_servers()
    kept = [r for r in rows if r["id"] != server_id]
    if len(kept) == len(rows):
        return False
    app_settings.set_servers(kept)
    credentials.clear_bound_key(server_id)
    app_settings.set_not_billed(server_id, False)
    forget_list(server_id)
    return True


# ============================================================
# Which providers there are
# ============================================================

Kind = Literal["claude", "ollama", "preset", "server"]


def answer_provider_ids() -> tuple[str, ...]:
    """Every provider the app can answer with: the two native ones, the presets, your servers."""
    return (*NATIVE_PROVIDERS, *(p.id for p in PRESETS), *(s.id for s in servers()))


def kind_of(provider: str) -> Kind | None:
    key = provider.strip().lower()
    if key == "anthropic":
        return "claude"
    if key == "ollama":
        return "ollama"
    if key in _PRESETS_BY_ID:
        return "preset"
    if server(key) is not None:
        return "server"
    return None


def is_compatible(provider: str) -> bool:
    """Whether ``provider`` is answered through the OpenAI-compatible client."""
    return kind_of(provider) in ("preset", "server")


def display_name(provider: str) -> str:
    key = provider.strip().lower()
    if key == "anthropic":
        return "Claude"
    if key == "ollama":
        return "Ollama"
    found = preset(key) or server(key)
    return found.name if found is not None else provider


def address_of(provider: str) -> str:
    """The address ``provider``'s requests go to; ``""`` for one this build does not know."""
    key = provider.strip().lower()
    if key == "anthropic":
        return os.getenv("ANTHROPIC_BASE_URL") or ANTHROPIC_ADDRESS
    if key == "ollama":
        return config.OLLAMA_HOST
    found = preset(key) or server(key)
    return found.address if found is not None else ""


def takes_key(provider: str) -> bool:
    """Whether a key is needed before ``provider`` can answer. A server's key is optional."""
    key = provider.strip().lower()
    if key == "anthropic":
        return True
    found = preset(key)
    return found is not None and found.key_page is not None


@dataclass(frozen=True)
class Endpoint:
    """What one compatible request needs: the address, and the key bound to it, if any."""

    id: str
    name: str
    address: str
    key: str | None = field(default=None, repr=False)
    key_check: str | None = None


def endpoint(provider: str) -> Endpoint:
    """The endpoint for a compatible ``provider``. ``ValueError`` for any other."""
    key = provider.strip().lower()
    found = preset(key)
    if found is not None:
        return Endpoint(
            found.id,
            found.name,
            found.address,
            credentials.get_bound_key(found.id, found.address),
            found.key_check,
        )
    mine = server(key)
    if mine is not None:
        return Endpoint(
            mine.id, mine.name, mine.address, credentials.get_bound_key(mine.id, mine.address)
        )
    raise ValueError(f"'{provider}' is not an OpenAI-compatible provider this build knows")


# ============================================================
# Where the text goes (choice 4, amended)
# ============================================================

Reach = Literal["this_computer", "local_network", "elsewhere"]

_HOSTNAME = re.compile(r"[a-z0-9_]([a-z0-9_-]*[a-z0-9_])?(\.[a-z0-9_]([a-z0-9_-]*[a-z0-9_])?)*\.?")

_SAYS = "Your questions and the passages found for them"


def host_of(address: str) -> str:
    """The host in ``address``, lower case; ``""`` when it cannot be read. A bare ``host:port``
    counts, since ``OLLAMA_HOST`` may be one."""
    text = address.strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text if "://" in text else f"//{text}")
        host = (parts.hostname or "").lower()
        _ = parts.port  # a port that is not a number raises here
    except ValueError:
        return ""
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return host if _HOSTNAME.fullmatch(host) else ""


def reach_of(host: str) -> Reach:
    """Whether ``host`` is this computer, a private address on the local network, or elsewhere.

    Only ``localhost`` and loopback (or unspecified) literals are this computer. A private or
    link-local literal is the local network. Every other name, ``gpu-box`` included, is elsewhere:
    names are never looked up. An empty host (an address that could not be read) is elsewhere too.

    ``*.localhost`` is elsewhere as well, a stricter reading of ADR-060's choice 4. The request
    resolves it through the OS, and Windows does not map it to loopback: ``lab.localhost`` did not
    resolve at all on this machine (2026-11-02), so whatever answers for it is the resolver's
    choice, and the sentence cannot promise it.
    """
    name = host.lower().rstrip(".")
    if name == "localhost":
        return "this_computer"
    try:
        literal = ipaddress.ip_address(name)
    except ValueError:
        return "elsewhere"
    if literal.is_loopback or literal.is_unspecified:
        return "this_computer"
    if literal.is_private or literal.is_link_local:
        return "local_network"
    return "elsewhere"


@dataclass(frozen=True)
class Route:
    """Where one model's text goes."""

    reach: Reach
    host: str
    """The host the text goes to; ``""`` when the address could not be read."""

    @property
    def sentence(self) -> str:
        """The approved sentence for the card, the rows and the forms."""
        if self.reach == "this_computer":
            return f"{_SAYS} stay on this computer."
        if self.reach == "local_network":
            return f"{_SAYS} leave this computer: they go to {self.host}, on your local network."
        if not self.host:
            return f"{_SAYS} leave this computer: the address could not be read."
        return f"{_SAYS} go to {self.host}."

    @property
    def short(self) -> str:
        """The line under an answer."""
        if self.reach == "this_computer":
            return "Stays on this computer"
        if self.host:
            return f"Sent to {self.host}"
        return "Leaves this computer"


def route_of(address: str, upstream: str | None = None) -> Route:
    """Where text sent to ``address`` goes. ``upstream`` is the model's upstream host when the
    server reports one (Ollama's ``remote_host``): the text goes there, whatever the address."""
    host = host_of(upstream) if upstream else host_of(address)
    if upstream and not host:
        return Route("elsewhere", "")
    return Route(reach_of(host), host)


# ============================================================
# Who bills (choice 5, amended)
# ============================================================

BillingKind = Literal["vendor", "nobody", "unknown", "your_word"]


@dataclass(frozen=True)
class Billing:
    kind: BillingKind
    vendor: str = ""

    @property
    def guard_trips(self) -> bool:
        """Billed, or not known: what the CLI credit guard stops on (KI-4)."""
        return self.kind in ("vendor", "unknown")

    @property
    def line(self) -> str:
        """Who bills, as the panel says it."""
        if self.kind == "vendor":
            return f"Billed by {self.vendor}"
        if self.kind == "nobody":
            return "Not billed"
        if self.kind == "your_word":
            return "Not billed (your word)"
        return "Billing not known"

    @property
    def chip(self) -> str:
        """The cost line under an answer, for a provider Provenote has no price for."""
        if self.kind == "vendor":
            return f"Billed by {self.vendor}; Provenote does not know its price."
        if self.kind == "nobody":
            return "No metered cost."
        if self.kind == "your_word":
            return "Not billed (your word)."
        return "Billing not known."


@dataclass(frozen=True)
class Facts:
    """The two facts about one provider and model, kept apart."""

    route: Route
    billing: Billing

    def as_record(self, provider: str) -> dict[str, str]:
        """What an answer's provenance keeps (additive, ADR-060)."""
        return {
            "provider": provider,
            "host": self.route.host,
            "reach": self.route.reach,
            "billing": self.billing.kind,
            "billed_by": self.billing.vendor,
        }


def can_tick(provider: str) -> bool:
    """Whether *This server does not bill me* applies: a server of your own, or an Ollama whose
    host is off this computer (the amendment's table)."""
    kind = kind_of(provider)
    if kind == "server":
        return True
    return kind == "ollama" and reach_of(host_of(config.OLLAMA_HOST)) != "this_computer"


def describe(provider: str, model: str | None = None) -> Facts:
    """Where ``provider``'s text goes and who bills, for ``model`` when given. Local state only.

    An Ollama model's upstream host comes from the last probe (``llm.ollama_remote_host``); a
    model the probe has not seen counts by its address.
    """
    key = provider.strip().lower()
    kind = kind_of(key)
    address = address_of(key)
    route = route_of(address)
    ticked = key in app_settings.get_not_billed()
    if kind == "claude":
        return Facts(route, Billing("vendor", "Anthropic"))
    if kind == "ollama":
        from doc_assistant import llm

        remote = llm.ollama_remote_host(model) if model else None
        if remote:
            return Facts(route_of(address, upstream=remote), Billing("vendor", "Ollama"))
        if route.reach == "this_computer":
            return Facts(route, Billing("nobody"))
        return Facts(route, Billing("your_word" if ticked else "unknown"))
    if kind == "preset":
        found = _PRESETS_BY_ID[key]
        return Facts(
            route, Billing("vendor", found.billed_by) if found.billed_by else Billing("nobody")
        )
    if kind == "server":
        return Facts(route, Billing("your_word" if ticked else "unknown"))
    return Facts(route, Billing("unknown"))


# ============================================================
# What each form says
# ============================================================


def sends() -> tuple[str, ...]:
    """Everything an answer sends to the provider, as each form lists it (the amendment)."""
    items = [
        "your question",
        "up to six earlier messages, answers included, when a follow-up is rewritten",
        "the passages found for it, with their file name and page",
    ]
    if not config.REVIEWER_PROVIDER_PINNED:
        items.append("a flagged answer, for its review")
    return tuple(items)


FORWARDS_NOTE = (
    "Provenote sees only this address; a server that forwards requests sends your text on too."
)

PROMPT_TOKENS = 5000
"""About how long an answer prompt runs on 10 passages (KI-66's measurement)."""


def window_note() -> str:
    """The line a server on this computer or your network carries (choice 7)."""
    return (
        "Provenote cannot set this server's context window. An answer prompt runs to about "
        f"{PROMPT_TOKENS:,} tokens; give the model at least {config.OLLAMA_CONTEXT_TOKENS:,}, or "
        "the start of the prompt, which asks for citations, may be cut without an error."
    )


def prompt_cut_note(reported_prompt_tokens: int, sent_chars: int) -> str | None:
    """A warning when a server reports reading under half the prompt it was sent (choice 7).

    ``sent_chars / 4`` is the rough estimate the cost chip already uses; *below half* is set
    against it and would have caught KI-66's cut (about 5,000 tokens down to 2,050). Nothing is
    said when the server reports no usage.
    """
    if reported_prompt_tokens <= 0 or sent_chars <= 0:
        return None
    estimate = sent_chars // 4
    if reported_prompt_tokens * 2 >= estimate:
        return None
    return (
        f"This server reports reading {reported_prompt_tokens:,} tokens of a prompt of about "
        f"{estimate:,}. It may have cut the start, which asks for citations: give the model a "
        "larger context window."
    )


# ============================================================
# Model lists (choice 8): fetched on request, kept in memory only
# ============================================================

ListStatus = Literal["ok", "refused", "needs_key", "unreachable", "error"]


@dataclass(frozen=True)
class ModelList:
    status: ListStatus
    models: tuple[str, ...] = ()
    detail: str = ""
    fetched_at: str = ""
    """When the list was fetched (ISO 8601, UTC)."""
    windows: dict[str, int] = field(default_factory=dict)
    """Context windows the server documents per model, where it does."""


_LISTS: dict[str, tuple[str, ModelList]] = {}
"""provider -> (the key's digest it was fetched with, the list). In memory only."""


def _digest(key: str | None) -> str:
    return hashlib.sha256((key or "").encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def remember_list(provider: str, key: str | None, listing: ModelList) -> None:
    _LISTS[provider] = (_digest(key), listing)


def remembered_list(provider: str, key: str | None) -> ModelList | None:
    """The last list fetched for ``provider`` with this same key; ``None`` otherwise, so a refused
    key stops reading *Key refused* once it is changed."""
    entry = _LISTS.get(provider)
    if entry is None or entry[0] != _digest(key):
        return None
    return entry[1]


def forget_list(provider: str) -> None:
    _LISTS.pop(provider, None)


def current_key(provider: str) -> str | None:
    """The key a request to ``provider`` would carry now. Never logged."""
    key = provider.strip().lower()
    if key == "anthropic":
        return credentials.resolve_key("anthropic")
    if is_compatible(key):
        return endpoint(key).key
    return None


def fetch_models(provider: str) -> ModelList:
    """Ask ``provider``'s server for its models now, and remember the answer. Network."""
    from doc_assistant import compatible, llm

    key = provider.strip().lower()
    kind = kind_of(key)
    if kind == "claude":
        api_key = credentials.resolve_key("anthropic")
        listing = llm.list_claude_models(api_key) if api_key else ModelList("needs_key")
    elif kind == "ollama":
        reachable, models, detail = llm.ollama_probe()
        listing = ModelList(
            "ok" if reachable else "unreachable", models, detail or "", fetched_at=now_iso()
        )
    elif kind in ("preset", "server"):
        point = endpoint(key)
        if takes_key(key) and not point.key:
            listing = ModelList("needs_key", detail=f"{point.name} needs a key first.")
        else:
            listing = compatible.list_models(point.address, point.key, key_check=point.key_check)
    else:
        raise ValueError(f"unknown provider '{provider}'")
    remember_list(key, current_key(key), listing)
    log.info("model_list_fetched", provider=key, status=listing.status, count=len(listing.models))
    return listing
