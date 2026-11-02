"""First-run readiness: what this install still needs before it can answer a question.

The app has two things a new user must supply, and they are independent: **a generation provider**
(a Claude API key, a local Ollama server with a model pulled, or since ADR-060 any
OpenAI-compatible preset or server of your own) and **documents** (a folder to index). Before
ADR-034 both were discoverable only by failing — you asked a question and got a transport error, or
you opened Settings and read ``.env`` instructions. This module computes the state instead, so the
UI and the docs can say the same true thing.

Shape of the answer (deliberately not a boolean):

* :class:`ProviderReadiness` per provider — configured? reachable? which models? where the key
  comes from? where the text goes and who bills (ADR-060's two facts, kept apart)? — because "not
  ready" has different fixes (paste a key / start a server / pull a model) and a single flag
  cannot carry the fix.
* :class:`SetupStep` for each thing left to do, in the order a user should do them, each with the
  action that resolves it. Getting started draws one row per step (ADR-034, amended 2026-11-01), so
  a step's words live here and the app never words one itself.

Rules this module keeps:

* **Never blocks.** It reports; nothing here refuses a request or changes an answer
  (inform-don't-block). An install with 0 documents is a *legitimate* state that reports honestly —
  the 0-document half of the robustness contract.
* **No key material.** Only ``key_source`` and a last-4 ``key_hint`` cross this boundary.
* **Probing is opt-in.** ``probe=False`` answers from local state alone (no network), which is what
  a test — and any caller on a hot path — wants. A probe asks only servers on this computer, all at
  once, within :data:`llm.PROBE_TIMEOUT_SECONDS`; a cloud server is asked only when its form is
  opened (``providers.fetch_models``), and what it said is read back from memory here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import datetime
from urllib.parse import urlsplit

import structlog

from doc_assistant import app_settings, config, credentials, llm, providers

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ProviderReadiness:
    """Whether one provider could serve the next turn, and if not, what would fix it."""

    id: str
    paid: bool
    """Billed, or not known: what the CLI credit guard stops on (ADR-060's amendment)."""
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
    """Models the server reported, from a probe or the last list fetched. Empty when unknown."""
    name: str = ""
    """What the screens call the provider (``id`` is the code's word)."""
    group: str = ""
    """Where the text goes, as a list to sit in: ``"local"``, ``"cloud"`` or ``"server"``."""
    destination: str = ""
    """One user-facing sentence: where the questions and passages go."""
    kind: str = ""
    """``claude``, ``ollama``, ``preset`` or ``server``: which form the panel draws."""
    address: str = ""
    """Where the requests go. A preset's is fixed; a server's is the user's."""
    reach: str = ""
    """``this_computer``, ``local_network`` or ``elsewhere``."""
    billing: str = ""
    """Who bills, as the panel says it (*Billed by Groq*, *Billing not known*, ...)."""
    billing_kind: str = ""
    """``vendor``, ``nobody``, ``unknown`` or ``your_word``."""
    can_tick: bool = False
    """Whether *This server does not bill me* applies to this provider."""
    takes_key: bool = False
    """Whether it cannot answer without a key (a server's key is optional)."""
    key_page: str | None = None
    """Where to get a key."""
    key_status: str | None = None
    """``accepted`` or ``refused`` when a model list checked the live key; ``None`` otherwise."""
    models_fetched_at: str | None = None
    """When ``models`` was fetched from the server (ISO 8601), if it was."""
    windows: dict[str, int] = field(default_factory=dict)
    """Context windows the server's list documents, by model."""
    remote: dict[str, str] = field(default_factory=dict)
    """Ollama: models served from elsewhere, with the host they run on."""
    notes: tuple[str, ...] = ()
    """Lines the form carries about this provider: the window, forwarding, an open address."""
    sends: tuple[str, ...] = ()
    """Everything an answer sends to it."""


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
    active_name: str = ""
    """What the screens call the provider answering now."""
    active_destination: str = ""
    """Where the text for the model answering now goes, worked out per model (ADR-060)."""
    active_billing: str = ""
    """Who bills the model answering now."""


def _base(pid: str) -> ProviderReadiness:
    """The fields every provider row carries, from the two facts; the rest is filled by caller."""
    facts = providers.describe(pid)
    return ProviderReadiness(
        id=pid,
        paid=facts.billing.guard_trips,
        configured=True,
        reachable=None,
        ready=True,
        detail="",
        name=providers.display_name(pid),
        kind=providers.kind_of(pid) or "",
        address=providers.address_of(pid),
        reach=facts.route.reach,
        destination=facts.route.sentence,
        billing=facts.billing.line,
        billing_kind=facts.billing.kind,
        can_tick=providers.can_tick(pid),
        takes_key=providers.takes_key(pid),
        sends=providers.sends(),
    )


def _clock(iso: str) -> str:
    """``14:02`` in this computer's time, from an ISO timestamp."""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%H:%M")
    except ValueError:
        return iso


def _accepted(name: str, listing: providers.ModelList, billed_by: str | None) -> str:
    count = len(listing.models)
    found = f"{count} model{'s' if count != 1 else ''}"
    if billed_by:
        return (
            f"Key accepted at {_clock(listing.fetched_at)}. {found}. "
            f"{billed_by}'s balance is not checked."
        )
    return f"Reached it at {_clock(listing.fetched_at)}. {found} found."


def _anthropic_readiness() -> ProviderReadiness:
    """Anthropic: a key is the whole question until its model list is fetched (opening its form,
    saving a key). No probe — a GET per settings read is not free of consequence (rate limits, an
    offline box waiting on a timeout). A key its list refused stops reading *Ready* (ADR-060)."""
    source = credentials.key_source("anthropic")
    configured = source is not None
    listing = providers.remembered_list("anthropic", credentials.resolve_key("anthropic"))
    base = replace(
        _base("anthropic"),
        group="cloud",
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
        key_page="https://console.anthropic.com/settings/keys",
    )
    if not configured or listing is None:
        return base
    if listing.status == "refused":
        return replace(
            base,
            ready=False,
            key_status="refused",
            detail="Anthropic refused this key.",
            action="Paste a new Anthropic API key under Claude.",
        )
    if listing.status == "ok":
        return replace(
            base,
            key_status="accepted",
            models=listing.models,
            models_fetched_at=listing.fetched_at,
            detail=_accepted("Claude", listing, "Anthropic"),
        )
    return base


_Probe = tuple[bool, tuple[str, ...], str | None]


def _ollama_readiness(*, probe: bool, probed: _Probe | None = None) -> ProviderReadiness:
    """Ollama: no credential, so readiness *is* reachability plus one installed model.
    ``probed`` is a probe already run (``_all_readiness`` runs them at once)."""
    host = config.OLLAMA_HOST
    common = _base("ollama")
    here = common.reach == "this_computer"
    unprobed = replace(
        common,
        group="local" if here else "server",
        detail=f"Local server at {host} (not checked).",
        notes=() if here else (providers.FORWARDS_NOTE,),
    )
    if not probe:
        return unprobed
    reachable, models, detail = probed if probed is not None else llm.ollama_probe()
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
    remote = {m: r for m in models if (r := llm.ollama_remote_host(m))}
    shown = {m: providers.host_of(r) or r for m, r in remote.items()}
    local = len(models) - len(remote)
    return replace(
        unprobed,
        reachable=True,
        detail=(
            f"{len(models)} model{'s' if len(models) != 1 else ''} installed, at no cost."
            if not remote
            else f"{local} model{'s' if local != 1 else ''} on this computer, "
            f"{len(remote)} run by Ollama elsewhere and billed by Ollama."
        ),
        models=models,
        remote=shown,
    )


def _compatible_readiness(pid: str, listing: providers.ModelList | None) -> ProviderReadiness:
    """A preset or a server of your own (ADR-060). ``listing`` is a probe's answer, or the list
    the server gave last time its form asked; ``None`` when nobody has asked yet."""
    point = providers.endpoint(pid)
    common = _base(pid)
    kind = common.kind
    on_this_computer = common.reach == "this_computer"
    found = providers.preset(pid)
    billed_by = found.billed_by if found is not None else None
    needs_key = common.takes_key
    configured = point.key is not None or not needs_key
    group = "server" if kind == "server" else ("local" if on_this_computer else "cloud")
    notes: list[str] = []
    if kind == "server" or (on_this_computer and kind == "preset"):
        notes.append(providers.window_note())
    if kind == "server":
        notes.append(providers.FORWARDS_NOTE)
    if urlsplit(point.address).scheme == "http" and not on_this_computer:
        notes.append("This address is not encrypted: the text crosses the network readable.")
    base = replace(
        common,
        group=group,
        configured=configured,
        ready=configured,
        key_source="app" if point.key else None,
        key_hint=credentials.bound_key_hint(pid, point.address),
        key_page=found.key_page if found is not None else None,
        notes=tuple(notes),
    )
    host = providers.host_of(point.address) or point.address
    at = urlsplit(point.address).netloc or host  # host:port, for "a server at …"
    if listing is None:
        if not configured:
            return replace(
                base,
                detail=f"No API key yet. Billed by {billed_by}." if billed_by else "No key yet.",
                action=f"Paste a {point.name} API key under {point.name}.",
            )
        if point.key is not None:
            return replace(base, detail="Key saved; not checked yet.")
        if kind == "server":
            return replace(base, detail=f"Not checked yet. Press Test connection to reach {at}.")
        return replace(base, detail=f"Looks for a server at {at} (not checked).")
    if listing.status == "ok":
        return replace(
            base,
            models=listing.models,
            models_fetched_at=listing.fetched_at or None,
            reachable=True,
            windows=dict(listing.windows),
            key_status="accepted" if point.key else None,
            detail=_accepted(point.name, listing, billed_by if point.key else None),
        )
    if listing.status == "refused":
        return replace(
            base,
            reachable=True,
            ready=False,
            key_status="refused",
            detail=f"{point.name} refused this key.",
            action=f"Paste a new key under {point.name}.",
        )
    if listing.status == "needs_key":
        return replace(
            base,
            reachable=True,
            ready=False,
            detail=listing.detail or f"{point.name} needs a key.",
            action=f"Add the key under {point.name}.",
        )
    if listing.status == "unreachable":
        if not on_this_computer and kind == "preset":
            # A cloud server that did not answer one GET is no evidence against the key.
            return replace(
                base, detail=f"Could not reach {host} to check the key. Saved.", reachable=None
            )
        return replace(
            base,
            reachable=False,
            ready=False,
            detail=f"Not running at {at}." if on_this_computer else listing.detail,
            action=(
                f"Start {point.name}'s server, then press Re-check."
                if kind == "preset"
                else "Check that the server is running, then press Test connection."
            ),
        )
    return replace(
        base,
        reachable=True,
        ready=False,
        detail=listing.detail,
        action=f"Check {point.name}'s address.",
    )


def _probe_compatible(pid: str) -> providers.ModelList:
    """A probe of one compatible server on this computer, remembered as its last list."""
    import httpx

    from doc_assistant import compatible

    point = providers.endpoint(pid)
    timeout = httpx.Timeout(llm.PROBE_TIMEOUT_SECONDS, connect=compatible.LOCAL_CONNECT_SECONDS)
    listing = compatible.list_models(point.address, point.key, timeout=timeout)
    providers.remember_list(pid, point.key, listing)
    return listing


def _local_compatibles() -> list[str]:
    """The presets and servers on this computer: the ones a probe may ask (choice 8)."""
    return [
        pid
        for pid in providers.answer_provider_ids()
        if providers.is_compatible(pid)
        and providers.reach_of(providers.host_of(providers.address_of(pid))) == "this_computer"
    ]


def provider_readiness(provider: str, *, probe: bool = True) -> ProviderReadiness:
    """Readiness for one provider. Unknown providers raise :class:`ValueError`."""
    key = provider.strip().lower()
    if key == "anthropic":
        return _anthropic_readiness()
    if key == "ollama":
        return _ollama_readiness(probe=probe)
    if providers.is_compatible(key):
        if probe and key in _local_compatibles():
            return _compatible_readiness(key, _probe_compatible(key))
        point = providers.endpoint(key)
        return _compatible_readiness(key, providers.remembered_list(key, point.key))
    raise ValueError(
        f"unknown provider '{provider}' — valid options: "
        f"{', '.join(providers.answer_provider_ids())}"
    )


def _all_readiness(*, probe: bool) -> tuple[ProviderReadiness, ...]:
    """Every provider, with the local probes run at once so they share one time budget.

    A compatible server's probe that has not answered by the end of the budget reads as not
    running, and the panel does not wait for it: on Windows a port nobody listens on is not
    refused at once, the connection is retried, and LM Studio's probe took 4 s on this machine
    (2026-11-02) where Ollama's took milliseconds. The late thread finishes on its own.
    """
    ids = providers.answer_provider_ids()
    if not probe:
        return tuple(provider_readiness(p, probe=False) for p in ids)
    local = _local_compatibles()
    pool = ThreadPoolExecutor(max_workers=1 + len(local))
    try:
        ollama = pool.submit(llm.ollama_probe)
        lists = {pid: pool.submit(_probe_compatible, pid) for pid in local}
        wait(list(lists.values()), timeout=llm.PROBE_TIMEOUT_SECONDS)
        late = providers.ModelList(
            "unreachable", detail=f"No answer within {llm.PROBE_TIMEOUT_SECONDS:.0f} s."
        )
        listed = {pid: f.result() if f.done() else late for pid, f in lists.items()}
        ollama_result = ollama.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    rows: list[ProviderReadiness] = []
    for pid in ids:
        if pid == "ollama":
            rows.append(_ollama_readiness(probe=True, probed=ollama_result))
        elif pid in listed:
            rows.append(_compatible_readiness(pid, listed[pid]))
        else:
            rows.append(provider_readiness(pid, probe=False))
    return tuple(rows)


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
    probed, answering Ollama does not have. A compatible server's list is not one of them: a
    model it does not list is named on its form and still saved (ADR-060, choice 8).
    """
    ollama = by_id.get("ollama")
    installed = ollama.models if ollama is not None and ollama.reachable else None
    mismatch = llm.model_mismatch(active.id, model, installed)
    if mismatch is None:
        return active
    return replace(active, ready=False, detail=mismatch, action=_SELECTION_FIX.get(active.id))


def _provider_step(
    active: ProviderReadiness, model: str, all_providers: tuple[ProviderReadiness, ...]
) -> SetupStep:
    """The provider step, phrased against the *active* selection.

    A provider the user is not using being unready is not a blocker, so the step tracks the active
    one — but when the active provider is unready and another one *is* ready, the detail says so,
    because switching is usually the faster fix than fixing the active one.

    *Engine online* means different things by provider: a server on this computer answered the
    probe; a cloud provider is never probed (:func:`_anthropic_readiness`), so for it the words
    mean a key is in place and not refused. A local server nobody has asked yet does not get them.
    """
    if active.ready:
        unasked = active.reachable is None and (
            active.id == "ollama" or (active.kind != "claude" and active.reach == "this_computer")
        )
        return SetupStep(
            id="provider",
            title="Model",
            detail=model,
            done=True,
            status="Not checked" if unasked else "Engine online",
        )
    detail = active.detail
    if active.id in providers.answer_provider_ids() and not active.configured:
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
    rows = _all_readiness(probe=probe)
    active_provider, active_model = app_settings.effective_llm()
    by_id = {p.id: p for p in rows}
    key = active_provider.strip().lower()
    known = key in by_id
    active = by_id.get(
        key,
        ProviderReadiness(
            id=active_provider,
            name=active_provider,
            paid=True,
            configured=False,
            reachable=None,
            ready=False,
            detail=f"'{active_provider}' is not a provider this build knows about.",
            action="Choose a provider under Provider & model.",
        ),
    )
    active = _with_selection(active, active_model, by_id)
    steps = (
        _provider_step(active, active_model, rows),
        _documents_step(chunk_count, document_count),
    )
    facts = providers.describe(key, active_model) if known else None
    state = SetupState(
        providers=rows,
        active_provider=active_provider,
        active_model=active_model,
        active_ready=active.ready,
        chunk_count=chunk_count,
        document_count=document_count,
        ollama_host=config.OLLAMA_HOST,
        steps=steps,
        ready=all(s.done for s in steps),
        active_name=active.name,
        active_destination=facts.route.sentence if facts else "",
        active_billing=facts.billing.line if facts else "",
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
