"""A6's guard tests (`docs/specs/chat-record-lookups.md` § Guard tests): record questions are
answered from the library's records by rule or command, with no model, and recorded as such.

Offline: a temp SQLite library holding the 28 public papers' records as the eval's data home
records them (`tests/eval/corpus_manifest.yaml`, with the two values the records cases name as
recorded: GPipe's later title, and no title for the MDL tutorial). No model, no vector store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc_assistant import chat_controller, record_lookups
from doc_assistant.chat_controller import ChatController, Result, Session
from doc_assistant.commands import help_message
from doc_assistant.db.models import (
    AnswerRecord,
    Concept,
    ConceptAlias,
    ConceptDefinition,
    Document,
    Folder,
)
from doc_assistant.db.session import session_scope
from doc_assistant.read_surface import get_conversation
from doc_assistant.record_lookups import RECORDS_NOTE, answer_command, answer_record_question

EVAL = Path(__file__).resolve().parents[1] / "eval"

#: What the eval data home recorded where it differs from the manifest (cases.records.yaml).
_RECORDED = {
    "gpipe_huang_2018.pdf": {"title": "GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism"},
    "mdl_tutorial_grunwald_2004.pdf": {"title": None},
    "mpnn_gilmer_2017.pdf": {
        "authors": "Justin Gilmer Samuel S. Schoenholz Patrick F. Riley Oriol Vinyals "
        "George E. Dahl"
    },
}

#: Each records case and the lookup its rule must reach.
_EXPECTED_LOOKUP = {
    "records_document_count": "library_summary",
    "records_broken_documents": "list_documents",
    "records_folder_count": "list_documents",
    "records_folder_list": "list_documents",
    "records_document_year": "get_document",
    "records_document_authors": "get_document",
    "records_document_title_as_recorded": "get_document",
    "records_missing_title": "list_documents",
}


def _cases(name: str) -> list[dict[str, Any]]:
    return list(yaml.safe_load((EVAL / name).read_text(encoding="utf-8")))


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    monkeypatch.setattr(chat_controller.app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    yield tmp_path
    engine.dispose()


@pytest.fixture
def demo_library(temp_db: Path) -> dict[str, str]:
    """The 28 public papers, the 18 demo ones in the 'Demo corpus' folder. Returns name -> id."""
    manifest = yaml.safe_load((EVAL / "corpus_manifest.yaml").read_text(encoding="utf-8"))
    ids: dict[str, str] = {}
    with session_scope() as session:
        folder = Folder(name="Demo corpus")
        session.add(folder)
        for entry in manifest["documents"]:
            name = entry["filename"]
            fields: dict[str, Any] = {
                "filename": name,
                "source_original": f"C:/lib/{name}",
                "doc_hash": f"h-{name}",
                "format": "pdf",
                "title": entry["title"],
                "authors": entry.get("authors"),
                "year": entry.get("year"),
                "chunk_count": 90,
                "page_count": 12,
                "extraction_health": "healthy",
            }
            fields.update(_RECORDED.get(name, {}))
            doc = Document(**fields)
            if entry.get("collection") == "demo":
                doc.folders.append(folder)
            session.add(doc)
            session.flush()
            ids[name] = str(doc.id)
    return ids


@pytest.fixture
def concepts(temp_db: Path) -> str:
    """One concept taken on (with forms and a chosen definition) and one term."""
    with session_scope() as session:
        c = Concept(label="Residual learning", kind="concept", graph_include=True)
        term = Concept(label="pipeline", kind="term", graph_include=False)
        session.add_all([c, term])
        session.flush()
        session.add_all(
            [
                ConceptAlias(concept_id=c.id, alias="residual mapping", breadth="exact"),
                ConceptAlias(concept_id=c.id, alias="shortcut", breadth="broad"),
                ConceptDefinition(
                    concept_id=c.id,
                    text="Layers fit a residual function with reference to their input.",
                    source="user",
                    provenance_key="u",
                    status="chosen",
                ),
            ]
        )
        return str(c.id)


# --- the eight records cases ---------------------------------------------------------------- #


@pytest.mark.parametrize("case", _cases("cases.records.yaml"), ids=lambda c: c["id"])
def test_each_records_case_reaches_its_lookup_and_holds_its_values(
    demo_library: dict[str, str], case: dict[str, Any]
) -> None:
    answered = answer_record_question(case["query"])
    assert answered is not None, "a records question must not fall through to retrieval"
    assert answered.lookup == _EXPECTED_LOOKUP[case["id"]]
    text = answered.text.lower()
    missing = [s for s in case["expected_substrings"] if s.lower() not in text]
    assert not missing, f"{missing} not in: {answered.text}"


def test_a_folder_count_is_the_folders_not_the_librarys(demo_library: dict[str, str]) -> None:
    """KI-95: asked about a folder, the count is the folder's (18), never the library's (28)."""
    answered = answer_record_question("How many documents are in the Demo corpus folder?")
    assert answered is not None
    assert "18" in answered.text and "28" not in answered.text


# --- no rule catches a content question ----------------------------------------------------- #


def _content_queries() -> list[str]:
    out: list[str] = []
    for name in (
        "cases.public.yaml",
        "cases.conversation.yaml",
        "cases.absent.yaml",
        "cases.hostile.yaml",
    ):
        for case in _cases(name):
            out += [q for q in [case.get("query"), *(case.get("turns") or [])] if q]
            out += list(case.get("prior_turns") or [])
    return out


def test_the_content_case_files_hold_the_questions_this_guard_reads() -> None:
    assert len(_content_queries()) >= 18 + 8 + 8 + 8


@pytest.mark.parametrize("query", _content_queries())
def test_no_rule_catches_a_content_question(demo_library: dict[str, str], query: str) -> None:
    """Every question of the single, conversation, absent and hostile files goes to retrieval,
    over a library holding every title those questions name."""
    assert answer_record_question(query) is None


def test_a_topical_word_sends_a_folder_question_to_retrieval(demo_library: dict[str, str]) -> None:
    query = "Which documents in the Demo corpus folder are about attention?"
    assert answer_record_question(query) is None
    assert answer_record_question("Who wrote the paper about attention?") is None


# --- finding a named document --------------------------------------------------------------- #


def test_several_title_matches_are_listed_and_the_answer_asks(
    demo_library: dict[str, str],
) -> None:
    answered = answer_record_question("What year is the Residual paper from?")
    assert answered is not None and answered.asked
    assert (
        "resnet_he_2015.pdf" in answered.text and "identity_mappings_he_2016.pdf" in answered.text
    )
    assert "Which one" in answered.text
    record = answered.as_record("rule")
    assert record["asked"] is True and record["rows"] == 2
    assert set(record["ids"]) == {
        demo_library["resnet_he_2015.pdf"],
        demo_library["identity_mappings_he_2016.pdf"],
    }


def test_no_title_match_goes_to_retrieval(demo_library: dict[str, str]) -> None:
    assert answer_record_question("What year is the Capsule Networks paper from?") is None


def test_every_content_word_must_match_and_case_and_punctuation_do_not(
    demo_library: dict[str, str],
) -> None:
    found = record_lookups.find_documents("GPIPE: easy-scaling")
    assert [d.filename for d in found] == ["gpipe_huang_2018.pdf"]
    assert record_lookups.find_documents("GPipe giant") == [], "the manifest's old title is not it"


def test_an_untitled_document_is_found_by_its_filename_stem(demo_library: dict[str, str]) -> None:
    found = record_lookups.find_documents("the mdl tutorial")
    assert [d.filename for d in found] == ["mdl_tutorial_grunwald_2004.pdf"]
    answered = answer_record_question("What is the title of the MDL tutorial document?")
    assert answered is not None and "No title is recorded" in answered.text


# --- an empty library ----------------------------------------------------------------------- #


def test_every_lookup_answers_honestly_on_an_empty_library(temp_db: Path) -> None:
    def text(q: str) -> str:
        got = answer_record_question(q)
        assert got is not None, q
        return got.text

    assert "0 documents" in text("How many documents are in my library?")
    assert "No broken documents" in text("Are there any broken documents in the library?")
    assert "empty" in text("What is my latest document?")
    assert "no folders yet" in text("How many documents are in the Demo corpus folder?")
    assert "no folders yet" in text("Which documents are in the Demo corpus folder?")
    assert "no folders yet" in text("Which folders do I have?")
    assert "no documents yet" in text("Which documents in my library have no title recorded?")
    assert "No concept has been taken on" in text("Which concepts have I taken on?")
    assert "not a concept you have taken on" in text("What is the concept attention?")
    # A named document with nothing to match goes to retrieval, which says the library is empty.
    assert answer_record_question("What year is the Pointer Networks paper from?") is None
    for cmd, arg, said in [
        ("folder", "", "no folders yet"),
        ("folder", "demo corpus", "no folders yet"),
        ("concept", "", "No concept has been taken on"),
        ("concept", "attention", "not a concept you have taken on"),
        ("document", "pointer networks", "No document is recorded under that name"),
    ]:
        got = answer_command(cmd, arg)
        assert got is not None and said in got.text, (cmd, arg)


def test_an_unknown_folder_says_so_and_lists_the_folders(demo_library: dict[str, str]) -> None:
    answered = answer_record_question("Which documents are in the Thesis folder?")
    assert answered is not None
    assert "No folder is named “Thesis”" in answered.text and "Demo corpus" in answered.text


def test_a_field_that_is_not_recorded_is_said_to_be_not_recorded(temp_db: Path) -> None:
    with session_scope() as session:
        session.add(
            Document(
                filename="mystery.pdf",
                source_original="C:/lib/mystery.pdf",
                doc_hash="h-m",
                format="pdf",
                title="A Mystery Paper",
            )
        )
    answered = answer_record_question("What year is the Mystery paper from?")
    assert answered is not None and "No year is recorded for `mystery.pdf`" in answered.text
    assert "**Authors:** not recorded" in answered.text


# --- what a rule-answered turn leaves behind ------------------------------------------------ #


class _NoModel:
    """A pipeline that fails the test if a rule-answered turn reaches it."""

    provider, model = "ollama", "llama3.1:8b"

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"a rule-answered turn touched the pipeline: {name}")


def _turn(controller: ChatController, session: Session, text: str) -> Any:
    results = [e.result for e in controller.handle_message(session, text) if isinstance(e, Result)]
    assert len(results) == 1
    return results[0]


def test_a_rule_answered_turn_writes_a_record_and_no_label(demo_library: dict[str, str]) -> None:
    controller = ChatController(rag=_NoModel())  # type: ignore[arg-type]
    session = Session()
    result = _turn(
        controller, session, "What year is the Pointer Networks paper in my library from?"
    )
    assert result.route == "library" and result.sources == []
    assert result.answer.endswith(RECORDS_NOTE)
    assert result.record_id is not None
    with session_scope() as s:
        row = s.get(AnswerRecord, result.record_id)
        assert row is not None
        assert json.loads(row.retrieved_chunks_json) == []
        assert row.model_name is None and row.token_input is None and row.token_output is None
        assert row.session_id == session.session_id
        recorded = json.loads(row.records_json or "null")
    assert recorded == {
        "lookup": "get_document",
        "args": {"name": "the Pointer Networks paper", "field": "year"},
        "ids": [demo_library["pointer_networks_vinyals_2015.pdf"]],
        "rows": 1,
        "via": "rule",
    }
    conversation = get_conversation(session.session_id)
    assert conversation is not None and len(conversation.turns) == 1
    assert conversation.turns[0].answer == result.answer and conversation.turns[0].cited == []
    assert conversation.labels.documents == [] and conversation.labels.folders == []
    assert session.history[-1]["content"] == result.answer


def test_a_retrieval_turn_records_no_lookup(temp_db: Path) -> None:
    from doc_assistant.provenance import record_answer

    rid = record_answer(query="q", answer="a", retrieved_chunks=[])
    with session_scope() as s:
        row = s.get(AnswerRecord, rid)
        assert row is not None and row.records_json is None


def test_records_json_is_added_to_an_old_answer_records_table(tmp_path: Path) -> None:
    from sqlalchemy import create_engine, inspect, text

    from doc_assistant.db.migrations import _apply_additive_columns
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", future=True)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE answer_records DROP COLUMN records_json"))
    added = _apply_additive_columns(engine)
    assert "answer_records.records_json" in added
    assert "records_json" in {c["name"] for c in inspect(engine).get_columns("answer_records")}
    assert _apply_additive_columns(engine) == [], "idempotent"
    engine.dispose()


# --- the commands --------------------------------------------------------------------------- #


def _same_lookup(command: Any, rule: Any) -> None:
    assert command is not None and rule is not None
    c, r = command.as_record("command"), rule.as_record("rule")
    assert (c["lookup"], c["ids"], c["rows"]) == (r["lookup"], r["ids"], r["rows"])


def test_folder_calls_the_folder_rules_lookup(demo_library: dict[str, str]) -> None:
    command = answer_command("folder", "demo corpus")
    rule = answer_record_question("Which documents are in the Demo corpus folder?")
    _same_lookup(command, rule)
    assert command is not None and rule is not None and command.text == rule.text


def test_concept_calls_the_concept_rules_lookup(concepts: str) -> None:
    command = answer_command("concept", "residual mapping")
    rule = answer_record_question("What is the concept residual mapping?")
    _same_lookup(command, rule)
    assert command is not None and command.ids == [concepts]
    assert "Layers fit a residual function" in command.text and "shortcut" in command.text


def test_concept_on_a_term_says_it_is_not_a_concept(concepts: str) -> None:
    answered = answer_command("concept", "pipeline")
    assert answered is not None and answered.ids == []
    assert "“pipeline” is not a concept you have taken on" in answered.text


def test_document_by_title_calls_the_document_rules_lookup(demo_library: dict[str, str]) -> None:
    command = answer_command("document", "pointer networks")
    rule = answer_record_question("Who wrote the Pointer Networks paper?")
    _same_lookup(command, rule)


def test_document_by_id_prefix_keeps_todays_details_view(demo_library: dict[str, str]) -> None:
    prefix = demo_library["pointer_networks_vinyals_2015.pdf"][:8]
    assert answer_command("document", prefix) is None, "an id prefix runs the existing command"


def test_a_lookup_command_is_recorded_as_a_rule_answered_turn(
    demo_library: dict[str, str],
) -> None:
    controller = ChatController(rag=_NoModel())  # type: ignore[arg-type]
    session = Session()
    result = _turn(controller, session, "/folder Demo corpus")
    assert result.route == "library" and result.record_id is not None
    with session_scope() as s:
        row = s.get(AnswerRecord, result.record_id)
        assert row is not None and row.records_json is not None
        recorded = json.loads(row.records_json)
    assert recorded["lookup"] == "list_documents" and recorded["via"] == "command"
    assert recorded["rows"] == 18 and len(recorded["ids"]) == 18


def test_help_lists_the_lookup_commands() -> None:
    text = help_message()
    assert "`/folder <name>`" in text and "`/concept <name>`" in text
    assert "`/document <id or title>`" in text


# --- the note ------------------------------------------------------------------------------- #


def test_the_records_note_wording_is_pinned() -> None:
    assert RECORDS_NOTE == (
        "_From the library's records: no passage was read and no model was asked._"
    )


def test_every_lookup_answer_ends_with_the_note(demo_library: dict[str, str]) -> None:
    for case in _cases("cases.records.yaml"):
        answered = answer_record_question(case["query"])
        assert answered is not None and answered.text.endswith(RECORDS_NOTE), case["id"]
