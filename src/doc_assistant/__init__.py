"""doc_assistant — the backend library (product name: Provenote, ADR-012).

Only the version constant lives here. Importing anything heavier at package import time would
make every CLI runner pay for it.
"""

#: The running version, and the reason it is a literal rather than read from package metadata:
#: the frozen sidecar has no ``dist-info`` to read, and a version the app *guessed* would make
#: the update check (ADR-044) compare against a lie. This is the sixth of the six places a
#: release bumps — ``scripts/release_preflight.py`` and ``tests/unit/test_version.py`` both
#: refuse to let it drift from ``pyproject.toml``. See ``docs/RELEASE.md`` §1.
__version__ = "0.5.1"
