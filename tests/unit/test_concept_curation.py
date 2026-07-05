"""Guard tests for the vocabulary curation core (pure functions only — no DB, no LLM)."""

from __future__ import annotations

from doc_assistant.concept_curation import (
    build_classify_messages,
    is_artifact,
    parse_noise_indices,
    plan_merges,
)

# ============================================================
# is_artifact — deterministic noise filter
# ============================================================


def test_artifact_flags_pure_digit_tokens():
    assert is_artifact("2015 volume")
    assert is_artifact("10 3389 fpos")
    assert is_artifact("111 publication date")


def test_artifact_flags_single_char_tokens_and_tiny_labels():
    assert is_artifact("bert s")
    assert is_artifact("o relevance")
    assert is_artifact("ab")  # <= 2 non-space chars


def test_artifact_keeps_real_concepts_incl_alnum_terms():
    for good in ["dense retrieval", "deeplabcut", "res2net-50", "gpt-4", "deep brain stimulation"]:
        assert not is_artifact(good), good


# ============================================================
# parse_noise_indices — tolerant JSON
# ============================================================


def test_parse_noise_object_and_range_filtering():
    assert parse_noise_indices('{"noise": [0, 2, 9]}', 3) == {0, 2}  # 9 out of range
    assert parse_noise_indices('```json\n{"noise": [1]}\n```', 3) == {1}
    assert parse_noise_indices("[0, 1]", 3) == {0, 1}  # bare list tolerated


def test_parse_noise_empty_and_garbage():
    assert parse_noise_indices('{"noise": []}', 5) == set()
    assert parse_noise_indices("the model refused", 5) == set()  # failure -> keep all


def test_build_classify_messages_numbers_terms():
    msgs = build_classify_messages(["alpha", "beta"])
    assert msgs[0]["role"] == "system"
    assert "[0] alpha" in msgs[1]["content"] and "[1] beta" in msgs[1]["content"]


# ============================================================
# plan_merges — union-find, survivor = more docs
# ============================================================


def test_merge_survivor_is_the_higher_doc_count():
    labels = {"a": "text embedding", "b": "text embeddings"}
    docs = {"a": 5, "b": 1}
    plans = plan_merges([("a", "b", 0.97)], docs, labels)
    assert len(plans) == 1
    assert plans[0].keep_id == "a" and plans[0].drop_id == "b"


def test_merge_is_transitive_single_root():
    # a~b and b~c collapse to one root; two drops, one keep.
    labels = {"a": "sts", "b": "semantic textual similarity", "c": "semantic textual"}
    docs = {"a": 3, "b": 2, "c": 1}
    plans = plan_merges([("a", "b", 0.95), ("b", "c", 0.93)], docs, labels)
    keeps = {p.keep_id for p in plans}
    drops = {p.drop_id for p in plans}
    assert keeps == {"a"} and drops == {"b", "c"}


def test_merge_ignores_pairs_already_in_same_cluster():
    labels = {"a": "x", "b": "y", "c": "z"}
    docs = {"a": 9, "b": 1, "c": 1}
    plans = plan_merges([("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", 0.9)], docs, labels)
    assert {p.drop_id for p in plans} == {"b", "c"} and {p.keep_id for p in plans} == {"a"}
