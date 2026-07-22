"""Domain routers for the FastAPI shell (APIRouter split).

Each module exposes a ``router = APIRouter()`` that ``main.create_app`` includes. Routers read
per-request state via ``request.app.state`` and shared helpers from ``apps.api.services`` — none
import from ``apps.api.main`` (that would be a cycle). One domain per module:

- ``health`` — liveness + live model identity.
- ``chat`` — the answer/turn surface: chat SSE, A/B compare, claim adjudication, export, and the
  per-answer source/figure artifact lookups.
- ``conversations`` — the history sidebar (read + management flags).
- ``library`` — documents, folders, keyword families (the Library browser's write + read paths).
- ``concepts`` — the concept-graph read model, the gap list + triage, and the rebuild trigger.
- ``settings`` — the user-settable runtime settings (source dir, provider switch, epistemics).
- ``sources`` — selective ingestion: the registry scan/patch + the ingest job.
"""
