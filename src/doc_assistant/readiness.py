"""First-run readiness: what this install still needs before it can answer a question.

The app has two things a new user must supply, and they are independent: **a generation provider**
(a Claude API key, or a local Ollama server with a model pulled) and **documents** (a folder to
index). Before ADR-034 both were discoverable only by failing — you asked a question and got a
transport error, or you opened Settings and read ``.env`` instructions. This module computes the
state instead, so the UI and the docs can say the same true thing.

Shape of the answer (deliberately not a boolean):

* :class:`ProviderReadiness` per provider — configured? reachable? which models? where the key
  comes from — because "not ready" has different fixes (paste a key / start Ollama / pull a model)
  and a single flag cannot carry the fix.
* :class:`SetupStep` for each thing left to do, in the order a user should do them, each with the
  action that resolves it. Getting started draws one row per step (ADR-034, amended 2026-11-01), so
  a step's words live here and the app never words one itself.

Rules this module keeps:

* **Never blocks.** It reports; nothing here refuses a request or changes an answer
  (inform-don't-block). An install with 0 documents is a *legitimate* state that reports honestly —
  the 0-document half of the robustness contract.
* **No key material.** Only ``key_source`` and a last-4 ``key_hint`` cross this boundary.
* **Probing is opt-in.** ``probe=False`` answers from local state alone (no network), which is what
  a test — and any caller on a hot path — wants.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

import structlog

from doc_assistant import app_settings, config, credentials, llm

log = structlog.get_logger(__name__)

PROVIDERS: tuple[str, ...] = ("anthropic", "ollama")
"""The providers the desktop app can switch between (mirrors ``llm.make_client``)."""

_ANTHROPIC_ADDRESS = "https://api.anthropic.com"
"""Where the SDK sends a request unless ``ANTHROPIC_BASE_URL`` says otherwise."""


@dataclass(frozen=True)
class ProviderReadiness:
    """Whether one provider could serve the next turn, and if not, what would fix it."""

    id: str
    paid: bool
    configured: bool
    """Credential present (or not needed). Local state only — never a network answer."""
    reachable: bool | None
    """``True``/``False`` from a probe, or ``None`` when not probed (or not probeable)."""
    ready: bool
    """Configured *and* not known-unreachable — i.e. worth selecting."""
    detail: str
    """One user-facing sentence: what is wrong, or what is right."""
    action: str | None = None
    """The next thing the user should do, phrased as an instruction. ``None`` when ready."""
    key_source: str | None = None
    """``"env"`` (``.env``/environment) or ``"app"`` (saved in Settings); ``None`` if no key."""
    key_hint: str | None = None
    """Last 4 characters of the live key, for display. Never the key."""
    models: tuple[str, ...] = ()
    """Models the probe found installed (Ollama). Empty when unknown or none."""
    name: str = ""
    """What the screens call the provider (``id`` is the code's word)."""
    group: str = ""
    """Where the text goes, as a list to sit in: ``"local"``, ``"cloud"`` or ``"server"``."""
    destination: str = ""
    """One user-facing sentence: where the questions and passages go."""


@dataclass(frozen=True)
class SetupStep:
    """One outstanding (or completed) first-run task: one row of Getting started."""

    id: str
    title: str
    detail: str
    done: bool
    status: str
    """The row's status, in the row's own words (*Engine online*, *Indexed*, *Not set*)."""
    action: str | None = None
    offline: bool = False
    """The chosen engine does not answer, which reads apart from a step nobody has done yet."""


@dataclass(frozen=True)
class SetupState:
    """The whole first-run picture: providers, corpus, and the steps left."""

    providers: tuple[ProviderReadiness, ...]
    active_provider: str
    active_model: str
    active_ready: bool
    chunk_count: int
    document_count: int
    ollama_host: str
    steps: tuple[SetupStep, ...] = field(default_factory=tuple)
    ready: bool = False
    """Every step done: a provider that can answer, and at least one indexed document."""


def _host_of(address: str) -> str:
    """The host in ``address``; a bare ``host:port`` counts, since ``OLLAMA_HOST`` may be one."""
    try:
        return urlsplit(address if "://" in address else f"//{address}").hostname or address
    except ValueError:  # a hand-set address with an unclosed IPv6 bracket
        return address


def _on_this_computer(host: str) -> bool:
    """Whether ``host`` is this machine. A name on the local network is not: the sentence built on
    this is a promise about where the user's text goes."""
    name = host.lower()
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _destination(address: str) -> str:
    """Where a provider at ``address`` sends the text, worked out from the address and not from
    the provider's name: an Ollama on another machine is not local."""
    host = _host_of(address)
    if _on_this_computer(host):
        return "Your questions and the passages found for them stay on this computer."
    return f"Your questions and the passages found for them go to {host}."


def _anthropic_readiness() -> ProviderReadiness:
    """Anthropic: a key is the whole question. No probe — a GET per settings read is not free
    of consequence (rate limits, an offline box waiting on a timeout), and key *presence* is the
    thing the user can act on. Validity is checked once, when the key is saved."""
    source = credentials.key_source("anthropic")
    configured = source is not None
    return ProviderReadiness(
        id="anthropic",
        name="Claude",
        group="cloud",
        destination=_destination(os.getenv("ANTHROPIC_BASE_URL") or _ANTHROPIC_ADDRESS),
        paid=True,
        configured=configured,
        reachable=None,
        ready=configured,
        detail=(
            f"Using the key from {'your .env file' if source == 'env' else 'this app'}."
            if configured
            else "No API key yet. Answers are metered by Anthropic."
        ),
        action=None if configured else "Paste an Anthropic API key under Claude.",
        key_source=source,
        key_hint=credentials.key_hint("anthropic"),
    )


def _ollama_readiness(*, probe: bool) -> ProviderReadiness:
    """Ollama: no credential, so readiness *is* reachability plus one installed model."""
    host = config.OLLAMA_HOST
    unprobed = ProviderReadiness(
        id="ollama",
        name="Ollama",
        group="local" if _on_this_computer(_host_of(host)) else "server",
        destination=_destination(host),
        paid=False,
        configured=True,
        reachable=None,
        ready=True,
        detail=f"Local server at {host} (not checked).",
    )
    if not probe:
        return unprobed
    reachable, models, detail = llm.ollama_probe()
    if not reachable:
        return replace(
            unprobed,
            reachable=False,
            ready=False,
            detail=detail or f"No Ollama server answering at {host}.",
            action="Install Ollama, then run `ollama serve` (it usually starts on its own).",
        )
    if not models:
        return replace(
            unprobed,
            reachable=True,
            ready=False,
            detail=detail or f"Ollama is running at {host} but has no models installed.",
            action="Pull a model, e.g. `ollama pull llama3.1:8b` (about 5 GB).",
        )
    return replace(
        unprobed,
        reachable=True,
        detail=f"{len(models)} model{'s' if len(models) != 1 else ''} installed, at no cost.",
        models=models,
    )


# What fixes a selection whose model the provider cannot serve, by provider (`_with_selection`).
_SELECTION_FIX = {
    "ollama": "Choose an installed model under Ollama and press Use for answers.",
    "anthropic": "Name a claude-… model under Claude and press Use for answers.",
}


def _with_selection(
    active: ProviderReadiness, model: str, by_id: dict[str, ProviderReadiness]
) -> ProviderReadiness:
    """``active``, marked not ready when its selected ``model`` cannot answer on it.

    A provider can be ready while the selection on it is not: the 0.7.0 walkthrough's sandbox had
    Ollama selected with a Claude model, and every question failed while setup looked done. Only
    certain cases count (``llm.model_mismatch``): a name the provider never serves, or a model a
    probed, answering Ollama does not have.
    """
    ollama = by_id.get("ollama")
    installed = ollama.models if ollama is not None and ollama.reachable else None
    mismatch = llm.model_mismatch(active.id, model, installed)
    if mismatch is None:
        return active
    return replace(active, ready=False, detail=mismatch, action=_SELECTION_FIX.get(active.id))


def provider_readiness(provider: str, *, probe: bool = True) -> ProviderReadiness:
    """Readiness for one provider. Unknown providers raise :class:`ValueError`."""
    key = provider.strip().lower()
    if key == "anthropic":
        return _anthropic_readiness()
    if key == "ollama":
        return _ollama_readiness(probe=probe)
    raise ValueError(f"unknown provider '{provider}' — valid options: {', '.join(PROVIDERS)}")


def _provider_step(
    active: ProviderReadiness, model: str, all_providers: tuple[ProviderReadiness, ...]
) -> SetupStep:
    """The provider step, phrased against the *active* selection.

    A provider the user is not using being unready is not a blocker, so the step tracks the active
    one — but when the active provider is unready and another one *is* ready, the detail says so,
    because switching is usually the faster fix than fixing the active one.

    *Engine online* means different things by provider: Ollama answered the probe; Claude is never
    probed (:func:`_anthropic_readiness`), so for it the words mean a key is in place. An Ollama
    nobody has asked yet does not get them.
    """
    if active.ready:
        unasked = active.id == "ollama" and active.reachable is None
        return SetupStep(
            id="provider",
            title="Model",
            detail=model,
            done=True,
            status="Not checked" if unasked else "Engine online",
        )
    detail = active.detail
    if active.id in PROVIDERS and not active.configured:
        detail = "No model chosen yet."
    alternative = next((p for p in all_providers if p.ready and p.id != active.id), None)
    if alternative is not None:
        detail = f"{detail.rstrip('.')}. {alternative.name} is ready, if you'd rather use that."
    offline = active.reachable is False
    return SetupStep(
        id="provider",
        title="Model",
        detail=detail,
        done=False,
        status="Engine offline" if offline else "Not set",
        action=active.action,
        offline=offline,
    )


def _documents_step(chunk_count: int, document_count: int) -> SetupStep:
    """The documents step. A count that could not be read arrives as 0 while chunks are there
    (``services.setup_state_dict``), and "0 documents indexed" beside *Indexed* would be false."""
    if chunk_count <= 0:
        return SetupStep(
            id="documents",
            title="Documents",
            detail="Nothing indexed yet.",
            done=False,
            status="Not set",
            action="Add documents from the Library.",
        )
    return SetupStep(
        id="documents",
        title="Documents",
        detail=(
            f"{document_count:,} document{'s' if document_count != 1 else ''} indexed"
            if document_count > 0
            else f"{chunk_count:,} chunks indexed"
        ),
        done=True,
        status="Indexed",
    )


def setup_state(*, chunk_count: int, document_count: int = 0, probe: bool = True) -> SetupState:
    """Compute the whole first-run picture.

    ``chunk_count``/``document_count`` are passed in rather than read here: the caller (the API
    shell) already holds the controller and the library session, and this module stays free of
    both — ``apps/`` is a thin renderer and this is the library function it renders.
    """
    providers = tuple(provider_readiness(p, probe=probe) for p in PROVIDERS)
    active_provider, active_model = app_settings.effective_llm()
    by_id = {p.id: p for p in providers}
    active = by_id.get(
        active_provider.lower(),
        ProviderReadiness(
            id=active_provider,
            name=active_provider,
            paid=active_provider.lower() in config.PAID_PROVIDERS,
            configured=False,
            reachable=None,
            ready=False,
            detail=f"'{active_provider}' is not a provider this build knows about.",
            action=f"Pick one of: {', '.join(PROVIDERS)}.",
        ),
    )
    active = _with_selection(active, active_model, by_id)
    steps = (
        _provider_step(active, active_model, providers),
        _documents_step(chunk_count, document_count),
    )
    state = SetupState(
        providers=providers,
        active_provider=active_provider,
        active_model=active_model,
        active_ready=active.ready,
        chunk_count=chunk_count,
        document_count=document_count,
        ollama_host=config.OLLAMA_HOST,
        steps=steps,
        ready=all(s.done for s in steps),
    )
    log.info(
        "setup_state_computed",
        ready=state.ready,
        active_provider=active_provider,
        active_ready=active.ready,
        chunk_count=chunk_count,
        probed=probe,
    )
    return state
