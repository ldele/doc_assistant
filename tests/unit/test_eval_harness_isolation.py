"""The eval harness must stay extractable — importing it pulls in NO app wiring.

ADR-003 Decision 8 / spec DoD #5: ``doc_assistant.eval.runner`` may use
``structlog.get_logger`` but must not import the ``logging_config`` seam or any app
module (pipeline, chat_controller, the UI shells). The harness is designed to be lifted
into a standalone repo (`tests/eval/TESTING.md`), so a stray app import would couple it
back to the application. Run in a subprocess so a clean import graph is measured,
unaffected by whatever the rest of the test session already imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_FORBIDDEN = (
    "doc_assistant.logging_config",
    "doc_assistant.config",
    "doc_assistant.pipeline",
    "doc_assistant.chat_controller",
    "chainlit",
    "fastapi",
)


def test_eval_runner_imports_without_app_wiring() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = ",".join(repr(m) for m in _FORBIDDEN)
    code = (
        "import sys\n"
        "import doc_assistant.eval.runner  # noqa: F401\n"
        f"leaked = [m for m in ({forbidden},) if m in sys.modules]\n"
        "assert not leaked, f'eval harness leaked app imports: {leaked}'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr or result.stdout
