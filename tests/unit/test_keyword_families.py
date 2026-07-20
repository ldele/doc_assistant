"""Unit tests for keyword-family detection — pure core, no model/DB (feature-tag-families.md)."""

from __future__ import annotations

import pytest

from doc_assistant.knowledge.keyword_families import (
    FamilyProposal,
    _edit_distance,
    _edit_similarity,
    _stem_candidates,
    detect_family_proposals,
)

# --- Tier 1: morphology ---------------------------------------------------------------------- #


def test_stem_strips_simple_plural() -> None:
    assert "llm" in _stem_candidates("llms")
    assert "connectome" in _stem_candidates("connectomes")


def test_stem_leaves_short_words_and_ss_us_is_endings_alone() -> None:
    assert _stem_candidates("bm25") == {"bm25"}
    assert _stem_candidates("class") == {"class"}
    assert _stem_candidates("bus") == {"bus"}
    assert _stem_candidates("axis") == {"axis"}


def test_stem_handles_ies_and_sibilant_es() -> None:
    assert "taxonomy" in _stem_candidates("taxonomies")
    assert "box" in _stem_candidates("boxes")


def test_tier1_groups_llm_and_llms() -> None:
    proposals = detect_family_proposals(["llm", "llms", "bm25"])
    assert len(proposals) == 1
    p = proposals[0]
    assert p.tier == "morphological"
    assert p.canonical == "llm"
    assert p.members == ("llms",)
    assert p.confidence == 1.0


def test_tier1_groups_connectome_and_connectomes() -> None:
    proposals = detect_family_proposals(["connectome", "connectomes"])
    assert len(proposals) == 1
    assert proposals[0] == FamilyProposal(
        canonical="connectome", members=("connectomes",), tier="morphological", confidence=1.0
    )


def test_no_proposal_for_a_singleton_stem() -> None:
    assert detect_family_proposals(["bm25", "chatgpt"]) == []


def test_tier1_canonical_prefers_shorter_then_alphabetical() -> None:
    proposals = detect_family_proposals(["boxes", "box"])
    assert proposals[0].canonical == "box"
    assert proposals[0].members == ("boxes",)


def test_duplicate_input_names_are_deduped() -> None:
    proposals = detect_family_proposals(["llm", "llm", "llms"])
    assert proposals[0].members == ("llms",)


# --- Tier 2: embedding (toy vectors — no real bge load) --------------------------------------- #


def _toy_embed(pairs: dict[str, list[float]]):
    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [pairs[t] for t in texts]

    return embed_fn


def test_tier2_groups_close_vectors_above_threshold() -> None:
    embed_fn = _toy_embed(
        {
            "connectome": [1.0, 0.0],
            "connectomics": [0.99, 0.14],  # cosine ~0.99, close
            "chatgpt": [0.0, 1.0],  # orthogonal, unrelated
        }
    )
    proposals = detect_family_proposals(
        ["connectome", "connectomics", "chatgpt"], embed_fn=embed_fn, embedding_threshold=0.9
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.tier == "embedding"
    assert p.canonical == "connectome"
    assert p.members == ("connectomics",)
    assert 0.9 < p.confidence <= 1.0


def test_tier2_no_proposal_below_threshold() -> None:
    embed_fn = _toy_embed({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert detect_family_proposals(["a", "b"], embed_fn=embed_fn, embedding_threshold=0.5) == []


def test_tier2_transitive_chain_becomes_one_group() -> None:
    # a~b and b~c both clear the threshold; a and c alone would not — union-find still merges them.
    embed_fn = _toy_embed(
        {
            "a": [1.0, 0.0, 0.0],
            "b": [0.9, 0.436, 0.0],  # cos(a,b) ~0.9
            "c": [0.81, 0.436, 0.391],  # cos(b,c) ~0.9-ish, cos(a,c) lower
        }
    )
    proposals = detect_family_proposals(
        ["a", "b", "c"], embed_fn=embed_fn, embedding_threshold=0.85
    )
    assert len(proposals) == 1
    assert set(proposals[0].members) | {proposals[0].canonical} == {"a", "b", "c"}


def test_tier2_skipped_without_embed_fn() -> None:
    assert detect_family_proposals(["connectome", "connectomics"]) == []


def test_tier1_consumed_names_excluded_from_tier2() -> None:
    # "llm"/"llms" already form a Tier-1 group; even a fake embed_fn that would cluster everything
    # together must not re-propose them under Tier 2.
    def embed_all_same(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    proposals = detect_family_proposals(
        ["llm", "llms", "chatgpt", "gpt"], embed_fn=embed_all_same, embedding_threshold=0.5
    )
    tiers = {p.tier for p in proposals}
    assert tiers == {"morphological", "embedding"}
    morph = next(p for p in proposals if p.tier == "morphological")
    embed = next(p for p in proposals if p.tier == "embedding")
    assert set(morph.members) | {morph.canonical} == {"llm", "llms"}
    assert set(embed.members) | {embed.canonical} == {"chatgpt", "gpt"}


# --- edit-distance helpers (used as Tier 2's supporting signal) ------------------------------- #


def test_edit_distance_and_similarity() -> None:
    assert _edit_distance("llm", "llm") == 0
    assert _edit_distance("", "abc") == 3
    assert _edit_distance("kitten", "sitting") == 3
    assert _edit_similarity("llm", "llm") == 1.0
    assert 0.0 <= _edit_similarity("connectome", "connectomics") < 1.0


# --- PR-2.5 D4: the sibilant `-es` rule over-stripped, so common plurals never matched --------- #
# `w.endswith(("ses","xes","zes","ches","shes")) -> w[:-2]` is right for `boxes`->`box` and
# `classes`->`class`, but wrong for every word whose singular already ends in `e`
# (`database`/`databases`, `size`/`sizes`, `cache`/`caches`). Structurally the two readings are
# indistinguishable without a lexicon, so both stems are now candidates and a match on **either**
# groups the pair. The existing tests picked `boxes`/`taxonomies` — the two inputs the old rules
# got right — which is why this hid.


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("database", "databases"),
        ("size", "sizes"),
        ("cache", "caches"),
        ("response", "responses"),
        ("interface", "interfaces"),
        ("box", "boxes"),  # the case the old rule already handled — must not regress
        ("class", "classes"),
        ("taxonomy", "taxonomies"),
        ("llm", "llms"),
    ],
)
def test_singular_and_plural_share_a_stem_candidate(singular: str, plural: str) -> None:
    assert _stem_candidates(singular) & _stem_candidates(plural), (
        f"{singular!r} and {plural!r} must group structurally"
    )


def test_detect_proposes_a_family_for_an_e_final_plural(singular_plural_corpus: list[str]) -> None:
    """The end-to-end shape of D4: a `confidence=1.0` structural match, not a Tier-2 fuzzy one."""
    proposals = detect_family_proposals(singular_plural_corpus)

    assert [p.canonical for p in proposals] == ["database"]
    assert proposals[0].members == ("databases",)
    assert proposals[0].tier == "morphological"
    assert proposals[0].confidence == 1.0


@pytest.fixture
def singular_plural_corpus() -> list[str]:
    return ["database", "databases", "retrieval"]


@pytest.mark.parametrize("word", ["bm25", "class", "bus", "axis", "not", "cas"])
def test_short_and_ss_us_is_words_keep_only_themselves(word: str) -> None:
    """The conservatism guard: no candidate may be produced that a *shorter* real word could
    collide with. `notes`->{notes,note} must never reach `not`; `cases`->{cases,cas,case} is the
    accepted residual risk (it needs a real keyword equal to an over-stripped stem)."""
    assert _stem_candidates(word) == {word}
