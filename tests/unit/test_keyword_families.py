"""Unit tests for keyword-family detection — pure core, no model/DB (feature-tag-families.md)."""

from __future__ import annotations

from doc_assistant.keyword_families import (
    FamilyProposal,
    _edit_distance,
    _edit_similarity,
    _stem,
    detect_family_proposals,
)

# --- Tier 1: morphology ---------------------------------------------------------------------- #


def test_stem_strips_simple_plural() -> None:
    assert _stem("llms") == "llm"
    assert _stem("connectomes") == "connectome"


def test_stem_leaves_short_words_and_ss_us_is_endings_alone() -> None:
    assert _stem("bm25") == "bm25"
    assert _stem("class") == "class"
    assert _stem("bus") == "bus"
    assert _stem("axis") == "axis"


def test_stem_handles_ies_and_sibilant_es() -> None:
    assert _stem("taxonomies") == "taxonomy"
    assert _stem("boxes") == "box"


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
