"""Tests for the update check (`doc_assistant.update_check`, ADR-044).

No socket is opened anywhere in this file: `_fetch_latest_release` is monkeypatched, which is the
only place the module touches the network. The behaviours worth pinning are the *honesty* ones —
a failed check must never read as "up to date", and a pre-release must never be offered to a
stable install.
"""

from __future__ import annotations

import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from doc_assistant import __version__, update_check

# --- version parsing / comparison (pure, no I/O) ------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.5.0", (0, 5, 0, 1, "")),
        ("v0.5.0", (0, 5, 0, 1, "")),
        ("  v1.2.3  ", (1, 2, 3, 1, "")),
        ("0.6", (0, 6, 0, 1, "")),
        ("0.6.0-rc1", (0, 6, 0, 0, "rc1")),
    ],
)
def test_parse_version_accepts_the_shapes_we_tag(
    raw: str, expected: tuple[int, int, int, int, str]
) -> None:
    assert update_check.parse_version(raw) == expected


@pytest.mark.parametrize("raw", ["", "latest", "nightly", "v", "1.x.0", "release-2026"])
def test_parse_version_returns_none_rather_than_raising(raw: str) -> None:
    """An unparseable tag is a reason to say `unknown`, not to crash the app."""
    assert update_check.parse_version(raw) is None


@pytest.mark.parametrize(
    ("candidate", "current", "newer"),
    [
        ("0.6.0", "0.5.0", True),
        ("0.5.1", "0.5.0", True),
        ("1.0.0", "0.9.9", True),
        ("0.5.0", "0.5.0", False),
        ("0.4.2", "0.5.0", False),
        ("0.10.0", "0.9.0", True),  # numeric, not lexicographic
    ],
)
def test_is_newer(candidate: str, current: str, newer: bool) -> None:
    assert update_check.is_newer(candidate, current) is newer


def test_a_prerelease_is_older_than_its_own_release() -> None:
    """So a stable install is never pointed at an rc, and an rc is told the release wins."""
    assert update_check.is_newer("0.6.0-rc1", "0.6.0") is False
    assert update_check.is_newer("0.6.0", "0.6.0-rc1") is True


def test_is_newer_is_false_when_either_side_is_unparseable() -> None:
    """The safe direction: a missed notification costs less than one chasing a release that
    does not exist."""
    assert update_check.is_newer("garbage", "0.5.0") is False
    assert update_check.is_newer("0.6.0", "garbage") is False


# --- the check itself ---------------------------------------------------------------------


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    def fake(url: str = "") -> Any:
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(update_check, "_fetch_latest_release", fake)


def test_check_reports_an_available_update(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(
        monkeypatch,
        {"tag_name": "v99.0.0", "html_url": "https://github.com/ldele/doc_assistant/releases/x"},
    )
    status = update_check.check_now()
    assert status.state == "update_available"
    assert status.latest_version == "99.0.0"
    assert status.current_version == __version__
    assert status.release_url == "https://github.com/ldele/doc_assistant/releases/x"
    assert status.checked_at is not None


def test_check_reports_current_when_the_latest_release_is_this_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, {"tag_name": f"v{__version__}"})
    assert update_check.check_now().state == "current"


def test_offline_is_unknown_and_never_current(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single most important behaviour in this module: a check that could not run must not
    report "up to date" (ADR-044)."""
    _patch_fetch(monkeypatch, urllib.error.URLError("no route to host"))
    status = update_check.check_now()
    assert status.state == "unknown"
    assert status.latest_version is None
    assert status.reason == "could not reach the update server"


def test_a_404_says_there_is_nothing_published_to_compare_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repo whose tags have no release objects cut — the release-process coupling ADR-044
    calls out. It reads differently from a network failure because it is a different problem."""
    _patch_fetch(
        monkeypatch,
        urllib.error.HTTPError("u", 404, "Not Found", {}, None),  # type: ignore[arg-type]
    )
    status = update_check.check_now()
    assert status.state == "unknown"
    assert status.reason == "no published release to compare against"


def test_a_server_error_is_unknown_with_its_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(
        monkeypatch,
        urllib.error.HTTPError("u", 503, "nope", {}, None),  # type: ignore[arg-type]
    )
    status = update_check.check_now()
    assert status.state == "unknown"
    assert status.reason is not None
    assert "503" in status.reason


@pytest.mark.parametrize("payload", [{}, {"tag_name": None}, {"tag_name": "nightly"}])
def test_an_unreadable_tag_is_unknown_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    _patch_fetch(monkeypatch, payload)
    assert update_check.check_now().state == "unknown"


def test_a_non_https_release_url_falls_back_to_the_known_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response decides the link the user clicks, so it does not get to name any scheme
    or host it likes."""
    _patch_fetch(monkeypatch, {"tag_name": "v99.0.0", "html_url": "javascript:alert(1)"})
    assert update_check.check_now().release_url == update_check.RELEASES_PAGE_URL


def test_check_never_raises_on_a_malformed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch, ValueError("unexpected response shape"))
    assert update_check.check_now().state == "unknown"


@pytest.mark.parametrize("url", ["file:///C:/Windows/win.ini", "http://example.com", "ftp://x/y"])
def test_the_fetcher_refuses_any_non_https_url(url: str) -> None:
    """`urlopen` would happily open `file://`, which turns "check for updates" into a local file
    read. The scheme is enforced, not assumed — no network is reached for any of these."""
    with pytest.raises(ValueError, match="non-https"):
        update_check._fetch_latest_release(url)


def test_a_refused_scheme_surfaces_as_unknown_not_a_crash() -> None:
    """And the refusal travels the same fail-safe path as any other error."""
    assert update_check.check_now("http://example.com").state == "unknown"


# --- the daily rate limit -----------------------------------------------------------------


def test_due_when_never_checked() -> None:
    assert update_check.due_for_check(None) is True


def test_not_due_immediately_after_a_check() -> None:
    assert update_check.due_for_check(datetime.now(timezone.utc).isoformat()) is False


def test_due_again_after_the_interval() -> None:
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    assert update_check.due_for_check(old) is True


def test_an_unreadable_stamp_costs_one_request_not_permanent_silence() -> None:
    assert update_check.due_for_check("not-a-timestamp") is True


def test_a_naive_stamp_is_read_as_utc_rather_than_crashing() -> None:
    """Older settings.json files could hold a naive timestamp; comparing one to an aware `now`
    raises TypeError, which would take out the whole check."""
    naive = (datetime.now(timezone.utc) - timedelta(hours=25)).replace(tzinfo=None).isoformat()
    assert update_check.due_for_check(naive) is True
