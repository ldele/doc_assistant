"""Tell the user a newer release exists — and nothing more (ADR-044).

The app ships as an installer with no store and no package manager behind it, so an install is
frozen at its shipped version unless something tells the user otherwise. This module is that
something: one unauthenticated HTTPS GET to the public GitHub Releases API, a version comparison,
and a link. **It never downloads, writes, executes or elevates anything** — delivery is explicitly
out of scope (ADR-044, and it needs a code-signing decision before it is even designable).

Three rules this module exists to keep:

* **Three states, never two.** A failed check is ``unknown``, never ``current``. Reporting "you are
  up to date" because the network was down is the one failure mode that would make the feature
  worse than not having it.
* **Fail-safe.** Every network, parse and decode error is caught and becomes ``unknown`` with a
  reason. A version check must not be able to break the app it is checking.
* **Nothing about the user leaves the machine.** No corpus, no queries, no titles, no install id,
  no telemetry — a GET to a public endpoint, with a ``User-Agent`` naming the app and its version
  because GitHub's API requires one.

Automatic checking is opt-in and runs at most once a day; a *manual* check always runs, because an
explicit press is its own consent (ADR-044). The caller decides which it is by passing ``force``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

# `timezone.utc`, not `datetime.UTC`: the latter is 3.11+ and this package declares >=3.10.
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import structlog

from doc_assistant import __version__

log = structlog.get_logger(__name__)

# The public repository this build checks against. Not user-configurable: pointing an update
# banner at an arbitrary host is a way to get someone to install something they did not choose.
GITHUB_OWNER = "ldele"
GITHUB_REPO = "doc_assistant"
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"

#: Bounded on purpose — this runs at startup and must never hold anything up (ADR-044).
TIMEOUT_SECONDS = 5.0
#: At most one automatic check per day. A manual check ignores this.
CHECK_INTERVAL = timedelta(hours=24)

UpdateState = Literal["current", "update_available", "unknown"]

_VERSION_RE = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?(?:[-+.](?P<pre>.+))?$"
)


@dataclass(frozen=True)
class UpdateStatus:
    """The outcome of a check — or of *not* having checked.

    ``state`` is the field the UI keys on and it has three values, not two: ``unknown`` covers
    both "never checked" and "the check failed", which are the same thing to a user deciding
    whether to go look. ``reason`` says which, in plain words, and is only set for ``unknown``.
    """

    state: UpdateState
    current_version: str
    latest_version: str | None = None
    release_url: str = RELEASES_PAGE_URL
    checked_at: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Wire shape (mirrored by ``apps/api/models/updates.py``)."""
        return {
            "state": self.state,
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "release_url": self.release_url,
            "checked_at": self.checked_at,
            "reason": self.reason,
        }


def parse_version(raw: str) -> tuple[int, int, int, int, str] | None:
    """Parse ``0.5.0`` / ``v0.5.0`` / ``0.6.0-rc1`` into a comparable tuple, or ``None``.

    The 4th element ranks a pre-release *below* its own release (``0`` vs ``1``), so ``0.6.0-rc1``
    sorts before ``0.6.0``. That is what stops a stable install from being offered an rc and an rc
    install from being told it is current. Returns ``None`` rather than raising for anything
    unrecognisable — an unparseable tag is a reason to say ``unknown``, not to crash.
    """
    m = _VERSION_RE.match(raw.strip())
    if m is None:
        return None
    pre = m.group("pre")
    return (
        int(m.group("major")),
        int(m.group("minor")),
        int(m.group("patch") or 0),
        0 if pre else 1,
        pre or "",
    )


def is_newer(candidate: str, current: str) -> bool:
    """Whether ``candidate`` is a strictly newer version than ``current``.

    ``False`` when either side is unparseable — the safe direction, since the cost of a missed
    notification is a user on an old version, and the cost of a false one is a user chasing a
    release that does not exist.
    """
    a, b = parse_version(candidate), parse_version(current)
    if a is None or b is None:
        return False
    return a > b


def _fetch_latest_release(url: str = RELEASES_API_URL) -> dict[str, Any]:
    """GET the latest-release object. Raises — the caller turns failures into ``unknown``.

    ``releases/latest`` excludes pre-releases and drafts server-side, which is half of why
    pre-releases are never offered (:func:`parse_version` is the other half).
    """
    # Enforce the scheme rather than assert it in a comment. `url` defaults to a module constant,
    # but it is a parameter (the tests pass one), and `urlopen` will happily open `file://` —
    # which would turn a "check for updates" into a local file read. This is also what makes the
    # `nosec B310` below honest: the scheme is checked, not assumed.
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https update URL: {url!r}")
    request = urllib.request.Request(
        url,
        headers={
            # GitHub's API requires a User-Agent. This is the whole of what we disclose beyond
            # the connection itself: the app name and the version asking (ADR-044).
            "User-Agent": f"Provenote/{__version__}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("unexpected response shape")
    return payload


def check_now(url: str = RELEASES_API_URL) -> UpdateStatus:
    """Perform one check against the network and report the outcome. Never raises.

    Every failure path lands on ``unknown`` with a reason the UI can show verbatim, because the
    honest sentence when the check fails is "could not check", not "up to date".
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        payload = _fetch_latest_release(url)
    except urllib.error.HTTPError as e:
        # A 404 is the normal state for a repo whose tags have no *release objects* cut — the
        # release-process coupling ADR-044 calls out. Say so plainly rather than "failed".
        reason = (
            "no published release to compare against"
            if e.code == 404
            else f"the update server answered {e.code}"
        )
        log.info("update_check_failed", status=e.code, reason=reason)
        return UpdateStatus("unknown", __version__, checked_at=now, reason=reason)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.info("update_check_offline", error=str(e))
        return UpdateStatus(
            "unknown", __version__, checked_at=now, reason="could not reach the update server"
        )
    except (ValueError, UnicodeDecodeError) as e:
        log.warning("update_check_unreadable", error=str(e))
        return UpdateStatus(
            "unknown",
            __version__,
            checked_at=now,
            reason="the update server sent something unreadable",
        )

    tag = payload.get("tag_name")
    if not isinstance(tag, str) or parse_version(tag) is None:
        log.warning("update_check_unparseable_tag", tag=tag)
        return UpdateStatus(
            "unknown",
            __version__,
            checked_at=now,
            reason="the latest release has no readable version",
        )

    latest = tag.lstrip("v")
    page = payload.get("html_url")
    url_out = page if isinstance(page, str) and page.startswith("https://") else RELEASES_PAGE_URL
    if is_newer(latest, __version__):
        log.info("update_available", current=__version__, latest=latest)
        return UpdateStatus("update_available", __version__, latest, url_out, checked_at=now)
    return UpdateStatus("current", __version__, latest, url_out, checked_at=now)


def due_for_check(last_checked: str | None, *, interval: timedelta = CHECK_INTERVAL) -> bool:
    """Whether an automatic check is due — ``True`` when never checked or the stamp is unreadable.

    Unreadable rather than "assume recent": a corrupted stamp should cost one extra request, not
    silence the feature permanently.
    """
    if not last_checked:
        return True
    try:
        stamp = datetime.fromisoformat(last_checked)
    except ValueError:
        return True
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - stamp >= interval
