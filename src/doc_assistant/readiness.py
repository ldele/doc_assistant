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
  action that resolves it.

Rules this module keeps:

* **Never blocks.** It reports; nothing here refuses a request or changes an answer
  (inform-don't-block). An install with 0 documents is a *legitimate* state that reports honestly —
  the 0-document half of the robustness contract.
* **No key material.** Only ``key_source`` and a last-4 ``key_hint`` cross this boundary.
* **Probing is opt-in.** ``probe=False`` answers from local state alone (no network), which is what
  a test — and any caller on a hot path — wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from doc_assistant import app_settings, config, credentials, llm

log = structlog.get_logger(__name__)

PROVIDERS: tuple[str, ...] = ("anthropic", "ollama")
"""The providers the desktop app can switch between (mirrors ``llm.make_client``)."""


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


@dataclass(frozen=True)
class SetupStep:
    """One outstanding (or completed) first-run task."""

    id: str
    title: str
    detail: str
    done: bool
    action: str | None = None


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


def _anthropic_readiness() -> ProviderReadiness:
    """Anthropic: a key is the whole question. No probe — a GET per settings read is not free
    of consequence (rate limits, an offline box waiting on a timeout), and key *presence* is the
    thing the user can act on. Validity is checked once, when the key is saved."""
    source = credentials.key_source("anthropic")
    configured = source is not None
    return ProviderReadiness(
        id="anthropic",
        paid=True,
        configured=configured,
        reachable=None,
        ready=configured,
        detail=(
            f"Using the key from {'your .env file' if source == 'env' else 'this app'}."
            if configured
            else "No API key yet. Answers are metered by Anthropic."
        ),
        action=None if configured else "Paste an Anthropic API key below.",
        key_source=source,
        key_hint=credentials.key_hint("anthropic"),
    )


def _ollama_readiness(*, probe: bool) -> ProviderReadiness:
    """Ollama: no credential, so readiness *is* reachability plus one installed model."""
    host = config.OLLAMA_HOST
    if not probe:
        return ProviderReadiness(
            id="ollama",
            paid=False,
            configured=True,
            reachable=None,
            ready=True,
            detail=f"Local server at {host} (not checked).",
        )
    reachable, models, detail = llm.ollama_probe()
    if not reachable:
        return ProviderReadiness(
            id="ollama",
            paid=False,
            configured=True,
            reachable=False,
            ready=False,
            detail=detail or f"No Ollama server answering at {host}.",
            action="Install Ollama, then run `ollama serve` (it usually starts on its own).",
        )
    if not models:
        return ProviderReadiness(
            id="ollama",
            paid=False,
            configured=True,
            reachable=True,
            ready=False,
            detail=detail or f"Ollama is running at {host} but has no models installed.",
            action="Pull a model, e.g. `ollama pull llama3.1:8b` (about 5 GB).",
        )
    return ProviderReadiness(
        id="ollama",
        paid=False,
        configured=True,
        reachable=True,
        ready=True,
        detail=f"{len(models)} model{'s' if len(models) != 1 else ''} installed, running locally "
        f"at no cost.",
        models=models,
    )


def provider_readiness(provider: str, *, probe: bool = True) -> ProviderReadiness:
    """Readiness for one provider. Unknown providers raise :class:`ValueError`."""
    key = provider.strip().lower()
    if key == "anthropic":
        return _anthropic_readiness()
    if key == "ollama":
        return _ollama_readiness(probe=probe)
    raise ValueError(f"unknown provider '{provider}' — valid options: {', '.join(PROVIDERS)}")


def _provider_step(
    active: ProviderReadiness, all_providers: tuple[ProviderReadiness, ...]
) -> SetupStep:
    """The provider step, phrased against the *active* selection.

    A provider the user is not using being unready is not a blocker, so the step tracks the active
    one — but when the active provider is unready and another one *is* ready, the detail says so,
    because switching is usually the faster fix than fixing the active one.
    """
    if active.ready:
        return SetupStep(
            id="provider",
            title="Answer engine",
            detail=f"{active.id} / {active.detail[0].lower()}{active.detail[1:]}"
            if active.detail
            else active.id,
            done=True,
        )
    alternative = next((p for p in all_providers if p.ready and p.id != active.id), None)
    detail = active.detail
    if alternative is not None:
        detail = f"{detail} {alternative.id.capitalize()} is ready, if you'd rather use that."
    return SetupStep(
        id="provider",
        title="Choose an answer engine",
        detail=detail,
        done=False,
        action=active.action,
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
            paid=active_provider.lower() in config.PAID_PROVIDERS,
            configured=False,
            reachable=None,
            ready=False,
            detail=f"'{active_provider}' is not a provider this build knows about.",
            action=f"Pick one of: {', '.join(PROVIDERS)}.",
        ),
    )
    corpus_done = chunk_count > 0
    steps = (
        _provider_step(active, providers),
        SetupStep(
            id="documents",
            title="Your documents" if corpus_done else "Add your documents",
            detail=(
                f"{document_count} document{'s' if document_count != 1 else ''} indexed "
                f"({chunk_count:,} chunks)."
                if corpus_done
                else "Nothing indexed yet. Point the app at a folder of PDFs, EPUBs, "
                "HTML, DOCX or Markdown."
            ),
            done=corpus_done,
            action=None if corpus_done else "Choose a folder below, then index it.",
        ),
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
