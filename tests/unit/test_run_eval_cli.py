"""Guard tests for `scripts/run_eval.py`'s pure helpers — the generator and the corpus.

Both exist because of the same defect class: **the run record did not pin what the run
measured.** RG-029 found runs that never said which LLM wrote their answers, so a model swap
read as a pipeline win; RG-021 is its sibling on the retrieval side — nothing recorded which
documents the index held, so a run over a polluted corpus is silently incomparable to a
baseline. The fix in both cases is a key in `config_json`, and these tests pin the two helpers
that produce them.

`resolve_generator` additionally guards the *cost* half: `--provider` is the only thing that can
override `.env`'s all-Anthropic default, and a provider given without a model would otherwise
inherit the model name of the provider it replaced.

Pure functions only — no pipeline, no models, no store.
"""

from __future__ import annotations

import pytest
from scripts.run_eval import index_composition, resolve_generator

from doc_assistant import sparse_index


class TestResolveGenerator:
    @pytest.fixture(autouse=True)
    def _configured_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A known `.env`-shaped default: the paid pairing that caused the 2026-08-15 leak.

        Patched on `scripts.run_eval`, which holds its own binding of both names (a `from ...
        import` is a separate binding — patching `config` would not be seen here).
        """
        monkeypatch.setattr("scripts.run_eval.LLM_PROVIDER", "anthropic")
        monkeypatch.setattr("scripts.run_eval.LLM_MODEL", "claude-haiku-4-5-20251001")

    def test_no_flags_inherits_the_configured_pair(self) -> None:
        """Unchanged behaviour for a bare run — the flag adds a choice, it does not move the
        default. Moving the default is a policy change; this is a control."""
        assert resolve_generator(None, None) == ("anthropic", "claude-haiku-4-5-20251001")

    def test_provider_with_model_is_taken_verbatim(self) -> None:
        assert resolve_generator("ollama", "llama3.1:8b") == ("ollama", "llama3.1:8b")

    def test_changing_provider_without_a_model_is_refused(self) -> None:
        """The failure this prevents is expensive to discover: Ollama would be handed
        `claude-haiku-4-5-…`, and it would fail once per case, minutes into a run."""
        with pytest.raises(ValueError) as excinfo:
            resolve_generator("ollama", None)

        message = str(excinfo.value)
        assert "--model" in message  # says how to fix it
        assert "claude-haiku-4-5-20251001" in message  # and which model it refused to inherit

    def test_naming_the_configured_provider_without_a_model_is_fine(self) -> None:
        """`--provider anthropic` on an anthropic install is a statement of intent, not a change,
        so the configured model still pairs correctly."""
        assert resolve_generator("anthropic", None) == ("anthropic", "claude-haiku-4-5-20251001")

    def test_model_alone_keeps_the_configured_provider(self) -> None:
        assert resolve_generator(None, "claude-sonnet-4-5") == ("anthropic", "claude-sonnet-4-5")


class TestIndexComposition:
    def test_records_the_count_and_a_digest(self) -> None:
        recorded = index_composition({"aaa", "bbb", "ccc"})

        assert recorded["index_doc_count"] == 3
        assert recorded["index_doc_digest"] == sparse_index.doc_set_digest(["aaa", "bbb", "ccc"])

    def test_two_runs_over_different_corpora_are_distinguishable(self) -> None:
        """The RG-021 property: one extra document in the index must be visible in the record.

        This is the demo-collection case — BM25/IDF statistics are corpus-global, so the extra
        document moves scores that a reader would otherwise attribute to the pipeline.
        """
        clean = index_composition({"aaa", "bbb"})
        polluted = index_composition({"aaa", "bbb", "demo"})

        assert clean["index_doc_digest"] != polluted["index_doc_digest"]
        assert clean["index_doc_count"] != polluted["index_doc_count"]

    def test_an_empty_corpus_is_a_composition_not_an_absence(self) -> None:
        """0 documents is a real, comparable answer (the robustness contract's floor state)."""
        recorded = index_composition(set())

        assert recorded["index_doc_count"] == 0
        assert recorded["index_doc_digest"] == sparse_index.doc_set_digest([])
        assert recorded["index_doc_digest"] is not None

    def test_unknown_composition_records_nulls_rather_than_omitting_the_keys(self) -> None:
        """ "I could not tell" and "0 documents" are different facts, and neither is "no key".

        A missing key is indistinguishable from a run written before RG-021 existed, so the
        unknown case says so explicitly — the same three-states rule the update check follows.
        """
        recorded = index_composition(None)

        assert recorded == {"index_doc_count": None, "index_doc_digest": None}
        assert "index_doc_count" in recorded
