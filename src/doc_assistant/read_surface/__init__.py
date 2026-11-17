"""The read surface: nine read-only tools over the library (ADR-066, session A3).

What an outside AI program will call through the engine's MCP endpoint (A4), and what the chat's
record lookups will call (A6). All logic lives here; the endpoint and the chat are thin callers.

- `search_library`, `get_passage` — ranked passages for a question, and one passage with the
  text around it, fenced and checked (`search`).
- `list_folders`, `list_documents`, `get_document`, `list_concepts`, `get_concept` — records
  (`records`).
- `list_conversations`, `get_conversation` — past conversations with derived labels
  (`conversations`).

Read-only and model-free: nothing here writes, calls a model or bills one. Every result names
its source and carries `SURFACE_VERSION` (`shapes`); every list is honest at 0 documents.
"""

from doc_assistant.read_surface.conversations import get_conversation, list_conversations
from doc_assistant.read_surface.records import (
    get_concept,
    get_document,
    list_concepts,
    list_documents,
    list_folders,
)
from doc_assistant.read_surface.search import Retrieve, get_passage, search_library
from doc_assistant.read_surface.shapes import SURFACE_VERSION

#: The tools: ADR-066's seven in its order, then the two the user added on 2026-11-17 after the
#: stand-in client trial (`get_passage`, `list_concepts`).
TOOLS: tuple[str, ...] = (
    "search_library",
    "list_folders",
    "list_documents",
    "get_document",
    "list_conversations",
    "get_conversation",
    "get_concept",
    "get_passage",
    "list_concepts",
)

__all__ = [
    "SURFACE_VERSION",
    "TOOLS",
    "Retrieve",
    "get_concept",
    "get_conversation",
    "get_document",
    "get_passage",
    "list_concepts",
    "list_conversations",
    "list_documents",
    "list_folders",
    "search_library",
]
