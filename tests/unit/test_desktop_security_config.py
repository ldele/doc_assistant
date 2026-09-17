"""Guard the config facts the app's security rests on.

The desktop renders LLM output through ``marked`` into ``{@html}``
(``apps/desktop/src/lib/chat/Markdown.svelte``). Three walls contain it: DOMPurify on that one
sink (S-3), the webview's Content-Security-Policy in ``tauri.conf.json`` — sent by the dev server
too, since Tauri injects it only into the built app — and a capability file that grants the page
almost nothing (one sidecar spawn, one file-open dialog). Each is an edit nobody would think of as
a security change, and neither ``svelte-check`` nor ``node:test`` reads the JSON or the Vite
config — so a loosened policy, an unsanitised sink or a broadened capability would ship silently.
These tests make that a red CI run.

Threat model and the reasoning: ``docs/security.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAURI_CONF = ROOT / "apps" / "desktop" / "src-tauri" / "tauri.conf.json"
CAPABILITIES = ROOT / "apps" / "desktop" / "src-tauri" / "capabilities" / "default.json"
DESKTOP_SRC = ROOT / "apps" / "desktop" / "src"
VITE_CONFIG = ROOT / "apps" / "desktop" / "vite.config.ts"

# The sidecar's fixed loopback origin. If the port ever moves, this changes with it — on purpose:
# the CSP is the only thing stopping injected markup from talking to anything else.
SIDECAR_ORIGIN = "http://127.0.0.1:8001"


def _csp_directives(csp: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in csp.split(";"):
        tokens = part.split()
        if tokens:
            out[tokens[0]] = tokens[1:]
    return out


def test_csp_is_set_and_never_unsafe() -> None:
    conf = json.loads(TAURI_CONF.read_text(encoding="utf-8"))
    csp = conf["app"]["security"]["csp"]
    assert isinstance(csp, str) and csp.strip(), "CSP is null or empty — nothing contains {@html}"
    assert "'unsafe-inline'" not in csp, "'unsafe-inline' would let injected <script>/onerror= run"
    assert "'unsafe-eval'" not in csp
    directives = _csp_directives(csp)
    assert directives.get("default-src") == ["'self'"], directives.get("default-src")
    # Every network directive that names a host names only the sidecar.
    for name in ("connect-src", "img-src"):
        hosts = [t for t in directives.get(name, []) if t.startswith("http")]
        assert hosts == [SIDECAR_ORIGIN], f"{name} reaches beyond the sidecar: {hosts}"


def test_capabilities_are_exactly_the_documented_set() -> None:
    cap = json.loads(CAPABILITIES.read_text(encoding="utf-8"))
    assert cap["windows"] == ["main"]
    names: set[str] = set()
    sidecars: list[dict[str, object]] = []
    for perm in cap["permissions"]:
        if isinstance(perm, str):
            names.add(perm)
        else:
            names.add(str(perm["identifier"]))
            sidecars.extend(perm.get("allow", []))
    # Add a permission here only together with the docs/security.md row that justifies it.
    assert names == {"core:default", "shell:allow-execute", "dialog:allow-open"}, names
    assert sidecars == [{"name": "binaries/doc-assistant-api", "sidecar": True}], sidecars


def test_no_remote_ipc_and_no_updater() -> None:
    conf = json.loads(TAURI_CONF.read_text(encoding="utf-8"))
    text = TAURI_CONF.read_text(encoding="utf-8")
    assert "dangerousRemoteDomainIpcAccess" not in text
    assert "remote" not in conf["app"]["security"]
    # ADR-044: the app tells you a new version exists; it never installs one.
    assert "updater" not in json.dumps(conf.get("plugins", {})).lower()


def _code_lines(text: str) -> list[str]:
    """Lines that are not comments — a comment may name ``{@html}`` without being a sink."""
    return [
        line
        for line in text.splitlines()
        if not line.strip().startswith(("//", "*", "/*", "<!--"))
    ]


def test_every_html_sink_renders_sanitised_markup() -> None:
    """S-3. One ``{@html}`` in the app, and it renders DOMPurify output. A second sink, or this one
    losing its sanitiser, is the S2 finding back."""
    sinks: dict[str, list[str]] = {}
    for path in DESKTOP_SRC.rglob("*.svelte"):
        code = "\n".join(_code_lines(path.read_text(encoding="utf-8")))
        found = re.findall(r"\{@html\s+([A-Za-z_]\w*)\s*\}", code)
        if found:
            sinks[path.relative_to(DESKTOP_SRC).as_posix()] = found
    assert sinks == {"lib/chat/Markdown.svelte": ["html"]}, sinks
    source = (DESKTOP_SRC / "lib" / "chat" / "Markdown.svelte").read_text(encoding="utf-8")
    assert re.search(r"const html = \$derived\(DOMPurify\.sanitize\(", source), (
        "the {@html} variable must be DOMPurify output"
    )


def test_the_dev_server_sends_the_production_policy() -> None:
    """S-3. ``tauri dev`` on desktop loads the Vite server directly and Tauri injects the CSP only
    into assets it serves, so without this header the dev loop runs with no policy at all
    (``devCsp`` is read on the same injection path and would not help). The header must be built
    from the production string — see ``src/lib/core/devCsp.ts`` and its tests."""
    config = VITE_CONFIG.read_text(encoding="utf-8")
    assert "'Content-Security-Policy': devCsp(tauriConf.app.security.csp" in config
    assert "src-tauri/tauri.conf.json" in config
