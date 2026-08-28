"""Whether two eval runs measured the same thing — and which of their scores may be compared.

**The failure this exists for.** On 2026-08-15 a five-trial run of the private set scored
``contains_all`` **0.822** against the control's **0.777** and read as a 6% pipeline improvement.
It was a model swap: that run inherited Anthropic Haiku where the control had generated on
``llama3.1:8b``. The tell was sitting in the same table — ``citation_overlap`` reproduced *to the
digit* (0.9363 vs 0.936), because citations come from the **retrieved documents** and no LLM ever
touches them. One number moved, one did not, and the pair localises the change to generation
(RG-029).

That reasoning is right, and it has been written out by hand at least twice — in RG-029, and again
as the generator caveat at the top of ``chunking_sweep_private_2026-08-08.md``. A hand-written
caveat only protects the reader who happens to open the right file. This module turns it into a
check.

**The model: a score depends on a prefix of the pipeline.** Cases feed the index, the index feeds
retrieval, retrieval feeds generation. A scorer that reads only what was *retrieved*
(``citation_overlap``, ``figure_retrieval``) is untouched by anything downstream of retrieval — so
it survives a generator swap, which is exactly why it is the signal to trust when the generator
moved. A scorer that reads the *answer* (``contains_all``, ``embedding_similarity``, ``llm_judge``,
``exact_match``) depends on every stage. So one differing setting does not invalidate a comparison
wholesale: it invalidates the scorers at or below the stage it belongs to, and an honest report
says which.

**Three states, never two** — the rule the update check (ADR-044) and the index-composition keys
both follow. A setting is ``SAME``, ``DIFFERENT`` or ``UNKNOWN``, and ``UNKNOWN`` is the common
case rather than the corner one: of the 75 runs in the live store, **not one** records its
generator or its corpus, because those keys did not exist until 2026-08-15 and 2026-08-17. A
comparison of two historical runs is therefore *unknown*, not *fine* — and the point of this
module is that it now says so instead of printing two means side by side.

**What this deliberately does not do.** It never infers a setting a run did not record — not from
the run's ``note`` prose, not from its timestamp, not from a sibling run. The five annotated Haiku
trials carry their generator in ``note`` precisely because back-filling ``config_json`` would make
an inference indistinguishable from a recording, which is the defect RG-029 exists to prevent. An
unrecorded setting reads ``UNKNOWN`` here, permanently.

Generic by construction: pure mappings and strings, no ``doc_assistant`` import, so it travels with
the harness when it is lifted (ADR-003 Decision 8).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Stage(Enum):
    """A pipeline stage, ordered outside-in. A score depends on a *prefix* of these."""

    CASES = "cases"
    INDEX = "index"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"


#: Depth of each stage. A scorer reading stage S is invalidated by a difference at any stage whose
#: depth is <= depth(S) — everything upstream of what it reads.
_DEPTH: dict[Stage, int] = {
    Stage.CASES: 0,
    Stage.INDEX: 1,
    Stage.RETRIEVAL: 2,
    Stage.GENERATION: 3,
}


class State(Enum):
    """What is known about one setting across two runs."""

    SAME = "same"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


class Status(Enum):
    """The verdict on one scorer's numbers across two runs.

    ``NOT_COMPARABLE`` outranks ``UNKNOWN`` outranks ``COMPARABLE``: knowing that a run measured
    something else is a stronger statement than not knowing, and either beats a clean bill.
    """

    COMPARABLE = "comparable"
    UNKNOWN = "unknown"
    NOT_COMPARABLE = "not comparable"


_STATUS_RANK: dict[Status, int] = {
    Status.COMPARABLE: 0,
    Status.UNKNOWN: 1,
    Status.NOT_COMPARABLE: 2,
}

#: Which stage each recorded setting belongs to. A key absent from this table is not a
#: comparability dimension; :func:`unclassified_keys` finds the ones that fell through, and a guard
#: test pins that every key ``run_defining_settings()`` emits is classified — so adding a
#: run-defining setting without teaching this table fails the suite instead of silently widening
#: the blind spot.
SETTING_STAGE: dict[str, Stage] = {
    # Cases — the questions themselves. Upstream of everything.
    "n_cases": Stage.CASES,
    # Index — which documents retrieval could reach (RG-021). BM25/IDF statistics and the vector
    # neighbourhood are corpus-global, so this moves every downstream number.
    "index_doc_digest": Stage.INDEX,
    "index_doc_count": Stage.INDEX,
    # Retrieval — the ingest geometry plus the locked retrieval settings.
    "parent_chunk_size": Stage.RETRIEVAL,
    "parent_chunk_overlap": Stage.RETRIEVAL,
    "child_chunk_size": Stage.RETRIEVAL,
    "child_chunk_overlap": Stage.RETRIEVAL,
    "baseline_chunk_size": Stage.RETRIEVAL,
    "baseline_chunk_overlap": Stage.RETRIEVAL,
    "embedding_model": Stage.RETRIEVAL,
    "use_parent_child": Stage.RETRIEVAL,
    "use_multi_query": Stage.RETRIEVAL,
    "top_k": Stage.RETRIEVAL,
    "candidate_k": Stage.RETRIEVAL,
    "bm25_weight": Stage.RETRIEVAL,
    "rerank_candidate_cap": Stage.RETRIEVAL,
    # Generation — which LLM wrote the answers.
    "llm_provider": Stage.GENERATION,
    "llm_model": Stage.GENERATION,
    # Scoring instruments — not a stage of the system under test, but a scorer that grades with a
    # different model measures differently. Placed at GENERATION so only answer-reading scorers
    # can be affected, and reached solely through SCORER_INSTRUMENT.
    "judge_provider": Stage.GENERATION,
    "judge_model": Stage.GENERATION,
}

#: Bookkeeping recorded alongside the settings. Not comparability dimensions: ``trial_index`` and
#: ``n_trials`` differ *by design* between the trials of one experiment, and ``scorers`` lists
#: which scorers ran rather than describing what was measured — a run that ran fewer scorers
#: is not a different experiment, it simply has fewer numbers to offer.
IGNORED_KEYS: frozenset[str] = frozenset({"trial_index", "n_trials", "scorers"})

#: The deepest stage each scorer reads. Derived from what the scorer actually consumes in
#: ``scorers.py``, not from its name: ``citation_overlap`` and ``figure_retrieval`` read the
#: *retrieved* documents (``output.citations`` and ``output.raw["retrieved"]``), which the adapter
#: fills from ``pipeline.retrieve`` before a single token is generated. Everything else reads
#: ``output.answer``.
SCORER_DEPENDS_ON: dict[str, Stage] = {
    "citation_overlap": Stage.RETRIEVAL,
    "figure_retrieval": Stage.RETRIEVAL,
    "contains_all": Stage.GENERATION,
    "exact_match": Stage.GENERATION,
    "embedding_similarity": Stage.GENERATION,
    "llm_judge": Stage.GENERATION,
}

#: Settings a scorer depends on *in addition to* its stage prefix — its own instrument. The judge
#: is a different model from the generator, so two runs can share a generator and still have been
#: graded by different judges. ``embedding_similarity`` needs no entry: its instrument is
#: ``embedding_model``, already inside its retrieval prefix because it is the same embedder.
SCORER_INSTRUMENT: dict[str, tuple[str, ...]] = {
    "llm_judge": ("judge_provider", "judge_model"),
}


@dataclass(frozen=True)
class Difference:
    """What is known about one setting across the two runs."""

    key: str
    stage: Stage
    state: State
    value_a: Any = None
    value_b: Any = None
    #: Set on UNKNOWN: which side failed to record it, in words.
    detail: str = ""

    def describe(self) -> str:
        if self.state is State.DIFFERENT:
            return f"{self.key}: {self.value_a!r} -> {self.value_b!r}"
        if self.state is State.UNKNOWN:
            return f"{self.key}: {self.detail}"
        return f"{self.key}: {self.value_a!r}"


def _summarise(diffs: Sequence[Difference], limit: int = 2) -> str:
    """Up to ``limit`` differences spelled out, then a count of the rest."""
    shown = "; ".join(d.describe() for d in diffs[:limit])
    extra = len(diffs) - limit
    return f"{shown} (+{extra} more)" if extra > 0 else shown


@dataclass(frozen=True)
class ScorerVerdict:
    """Whether one scorer's numbers may be read against each other, and why not if not."""

    scorer_name: str
    status: Status
    blocking: tuple[Difference, ...] = ()
    unknown: tuple[Difference, ...] = ()

    @property
    def reason(self) -> str:
        """One line, deliberately bounded.

        A verdict on a historical run can rest on eighteen unrecorded settings, and eighteen of
        them in a table cell is a wall nobody reads — the full list is a section of its own in the
        report. Named settings come first because :func:`compare_configs` orders by stage depth,
        so the two shown are the most upstream, and upstream is where the consequence is largest.
        """
        if self.status is Status.NOT_COMPARABLE:
            return _summarise(self.blocking)
        if self.status is Status.UNKNOWN:
            keys = ", ".join(d.key for d in self.unknown[:2])
            more = " and others" if len(self.unknown) > 2 else ""
            return f"{len(self.unknown)} setting(s) not recorded, incl. {keys}{more}"
        return "every setting it depends on is recorded and identical"


@dataclass(frozen=True)
class Comparison:
    """The whole picture: every setting's state, plus one verdict per scorer."""

    differences: tuple[Difference, ...]
    verdicts: tuple[ScorerVerdict, ...]
    #: Settings the caller declared as the experiment's independent variable.
    varying: tuple[str, ...] = ()
    #: Declared-varying settings that did **not** actually differ — the KI-41 shape. Empty unless
    #: ``varying`` was given, and deliberately separate from :attr:`status`: two arms that varied
    #: nothing are perfectly comparable *and* the experiment between them is void.
    ineffective_variation: tuple[Difference, ...] = ()

    @property
    def status(self) -> Status:
        """The worst verdict across the scorers — what a caller should act on.

        No scorers at all reads ``UNKNOWN``, never ``COMPARABLE``: two runs with nothing in common
        to compare have not been shown to agree about anything.
        """
        if not self.verdicts:
            return Status.UNKNOWN
        return max((v.status for v in self.verdicts), key=lambda s: _STATUS_RANK[s])

    def by_state(self, state: State) -> tuple[Difference, ...]:
        return tuple(d for d in self.differences if d.state is state)


def _difference(key: str, stage: Stage, a: Mapping[str, Any], b: Mapping[str, Any]) -> Difference:
    """Classify one key across two configs, keeping "absent" apart from "equal"."""
    in_a, in_b = key in a, key in b
    if not in_a and not in_b:
        return Difference(key, stage, State.UNKNOWN, detail="recorded by neither run")
    if not in_a:
        return Difference(key, stage, State.UNKNOWN, value_b=b[key], detail="not recorded by A")
    if not in_b:
        return Difference(key, stage, State.UNKNOWN, value_a=a[key], detail="not recorded by B")
    va, vb = a[key], b[key]
    state = State.SAME if va == vb else State.DIFFERENT
    return Difference(key, stage, state, value_a=va, value_b=vb)


def compare_configs(
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    *,
    extra_keys: Mapping[str, Stage] | None = None,
) -> tuple[Difference, ...]:
    """One :class:`Difference` per known setting, whether or not either run recorded it.

    Absent keys are reported rather than skipped, and that is the whole point. A comparison of two
    2026-06 runs has to *say* that neither pinned its corpus or its generator, because the silence
    is the finding — dropping those rows would reproduce exactly the table those runs printed
    before anyone knew the numbers were incomparable.

    ``extra_keys`` lets a caller classify settings this table does not know: a project-specific
    knob, or the case-set identity the CLI derives from the stored per-case rows.
    """
    stages = {**SETTING_STAGE, **(extra_keys or {})}
    return tuple(
        _difference(key, stage, config_a, config_b)
        for key, stage in sorted(stages.items(), key=lambda kv: (_DEPTH[kv[1]], kv[0]))
    )


def scorer_verdict(
    scorer_name: str,
    differences: Iterable[Difference],
    *,
    varying: frozenset[str] = frozenset(),
) -> ScorerVerdict:
    """Judge one scorer against the settings it actually depends on.

    An unrecognised scorer is assumed to depend on **everything** — the deepest stage *and* every
    scoring instrument. A scorer nobody has classified must not be waved through as comparable.

    ``varying`` names the experiment's independent variable(s). A difference there is the *point*
    of the comparison, not an objection to it: a chunking sweep exists to read two arms that
    differ in chunk size against each other. Everything else still blocks — which is the useful
    half, because the sweep's real risk is that something *besides* the grid moved.
    """
    known = scorer_name in SCORER_DEPENDS_ON
    depth = _DEPTH[SCORER_DEPENDS_ON.get(scorer_name, Stage.GENERATION)]
    # Instrument keys live at GENERATION depth, so the stage prefix alone would hand every
    # answer-reading scorer the *judge's* model. A scorer answers only for its own instrument —
    # except an unclassified one, which answers for all of them, because that is what assuming
    # the worst means here.
    every_instrument = {key for keys in SCORER_INSTRUMENT.values() for key in keys}
    mine = set(SCORER_INSTRUMENT.get(scorer_name, ())) if known else every_instrument
    relevant = [
        d
        for d in differences
        if d.key in mine or (_DEPTH[d.stage] <= depth and d.key not in every_instrument)
    ]
    blocking = tuple(d for d in relevant if d.state is State.DIFFERENT and d.key not in varying)
    unknown = tuple(d for d in relevant if d.state is State.UNKNOWN and d.key not in varying)
    if blocking:
        status = Status.NOT_COMPARABLE
    elif unknown:
        status = Status.UNKNOWN
    else:
        status = Status.COMPARABLE
    return ScorerVerdict(scorer_name, status, blocking, unknown)


def compare(
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    scorer_names: Sequence[str],
    *,
    extra_keys: Mapping[str, Stage] | None = None,
    extra_differences: Sequence[Difference] = (),
    varying: Sequence[str] = (),
) -> Comparison:
    """Full comparison: every setting's state, and a verdict per scorer.

    ``extra_differences`` carries facts the configs cannot express — the CLI passes the real
    case-set identity, computed from the stored per-case rows, because equal ``n_cases`` is not
    proof that two runs asked the same questions.

    ``varying`` declares the experiment's independent variable, which turns this from "are these
    the same run?" into the question a sweep actually asks: **did exactly the intended thing
    change?** Two failures fall out of that, and they are opposite. A difference *outside*
    ``varying`` still blocks — the arm moved something nobody meant to move. A declared-varying
    setting that came back identical lands in :attr:`Comparison.ineffective_variation` — which is
    KI-41 exactly: the 2026-06-06 chunking sweep drove its grid through environment variables
    ``.env`` silently overwrote, so six arms re-ingested one configuration and the result read as
    "no config beats the default". `sweep_chunking`'s preflight now catches that *before* a run;
    this catches it afterwards, from the record, where a preflight-less runner cannot hide it.
    """
    varying_keys = frozenset(varying)
    differences = compare_configs(config_a, config_b, extra_keys=extra_keys) + tuple(
        extra_differences
    )
    verdicts = tuple(
        scorer_verdict(name, differences, varying=varying_keys)
        for name in sorted(set(scorer_names))
    )
    ineffective = tuple(
        d for d in differences if d.key in varying_keys and d.state is not State.DIFFERENT
    )
    return Comparison(
        differences=differences,
        verdicts=verdicts,
        varying=tuple(sorted(varying_keys)),
        ineffective_variation=ineffective,
    )


def unclassified_keys(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Recorded keys this module can neither place in a stage nor knowingly ignore.

    The blind-spot detector. A run-defining setting added to the record but not to
    :data:`SETTING_STAGE` would otherwise be excluded from every verdict in silence, and the
    comparison would read cleaner than the evidence supports.
    """
    return tuple(sorted(k for k in config if k not in SETTING_STAGE and k not in IGNORED_KEYS))
