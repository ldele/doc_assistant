"""Guard the verified-10 public benchmark regime (ADR-024 opens; 2026-07-20).

`tests/eval/cases.public.yaml` and every committed baseline in
`tests/eval/baselines/` were authored against exactly 10 corpus papers. Extra
corpus documents are retrieval distractors that change benchmark difficulty, so
the *default* download selection must stay exactly those 10 — the demo
collection (Sutskever->Carmack reading list) joins only via ``--demo``.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from scripts.download_corpus import MANIFEST, _selected


@pytest.fixture(scope="module")
def manifest_documents() -> list[dict[str, Any]]:
    documents = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]
    assert isinstance(documents, list)
    return documents


def test_default_selection_is_the_verified_10(
    manifest_documents: list[dict[str, Any]],
) -> None:
    selected = _selected(manifest_documents, include_demo=False)
    assert len(selected) == 10
    assert all(doc.get("referenced_by_eval") is True for doc in selected)
    assert all(doc.get("collection", "eval") == "eval" for doc in selected)


def test_demo_selection_is_a_strict_superset(
    manifest_documents: list[dict[str, Any]],
) -> None:
    default = {d["filename"] for d in _selected(manifest_documents, include_demo=False)}
    everything = {d["filename"] for d in _selected(manifest_documents, include_demo=True)}
    assert default < everything
    assert len(everything) == len(manifest_documents)


def test_demo_entries_stay_out_of_the_eval_regime(
    manifest_documents: list[dict[str, Any]],
) -> None:
    demo = [d for d in manifest_documents if d.get("collection", "eval") == "demo"]
    assert demo, "demo collection missing from the manifest"
    for doc in demo:
        assert doc["referenced_by_eval"] is False, doc["filename"]


def test_every_entry_is_pinned_and_fetchable(
    manifest_documents: list[dict[str, Any]],
) -> None:
    for doc in manifest_documents:
        name = doc["filename"]
        assert doc["tier"] == "arxiv", name
        assert doc["direct_pdf"] is True, name
        assert doc["url"].startswith("https://arxiv.org/pdf/"), name
        assert len(doc["sha256"]) == 64, name
        assert doc["bytes"] > 0, name


def test_filenames_are_unique(manifest_documents: list[dict[str, Any]]) -> None:
    names = [d["filename"] for d in manifest_documents]
    assert len(names) == len(set(names))
