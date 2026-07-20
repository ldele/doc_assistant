"""Fetch the public evaluation corpus from arXiv (reproducibility).

The public corpus is the literature behind this project's own methods (RAG,
dense retrieval, sentence embeddings, the BGE/SPECTER2 embedders, BERT
re-ranking, ColBERT, HyDE, LLM-as-a-judge, AI Usage Cards). Every paper is on
arXiv, so this script reads ``tests/eval/corpus_manifest.yaml`` and downloads
each pinned-version PDF straight from arXiv into ``data/sources/``.

Nothing is re-hosted in the repo: arXiv's non-exclusive license permits
downloading, not redistribution. Referencing + downloading keeps the project in
the clear regardless of any single paper's license.

Each download is checked against the manifest ``sha256`` when present. A
mismatch is a warning, not a failure — arXiv occasionally re-renders a PDF,
changing the bytes without changing the content.

The script also supports a forward-compatible ``committed: true`` flag (copy a
bundled file from ``tests/eval/corpus/`` instead of downloading) for any future
CC-licensed paper that can be shipped in-repo. The current manifest commits
nothing.

The manifest carries two collections. The **eval** collection (the default, and
the only one fetched without a flag) is the verified-10 benchmark corpus that
``tests/eval/cases.public.yaml`` and every committed baseline in
``tests/eval/baselines/`` were authored against — it must stay closed, because
extra corpus documents are retrieval distractors that change benchmark
difficulty. The **demo** collection (``collection: demo``, fetched only with
``--demo``) adds classic deep-learning papers from the rumoured
Sutskever->Carmack reading list (30papers.com) for exploring the app on a
bigger, richer corpus; it is never part of the benchmark regime.

Usage::

    python -m scripts.download_corpus               # eval corpus (10 PDFs) -> data/sources/
    python -m scripts.download_corpus --demo        # eval + demo collections (28 PDFs)
    python -m scripts.download_corpus --verify-only # checksum what's already on disk
    python -m scripts.download_corpus --dry-run     # print the plan, fetch nothing
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from doc_assistant.config import DOCS_PATH, PROJECT_ROOT

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MANIFEST = PROJECT_ROOT / "tests" / "eval" / "corpus_manifest.yaml"
COMMITTED_DIR = PROJECT_ROOT / "tests" / "eval" / "corpus"
_UA = {"User-Agent": "doc_assistant-corpus-fetch/1.0 (reproducibility script)"}


def _selected(documents: list[dict[str, Any]], *, include_demo: bool) -> list[dict[str, Any]]:
    """The manifest entries a run operates on.

    An entry with no ``collection`` field belongs to the eval corpus (the
    pre-demo manifest carried no such field), so the default selection is
    exactly the verified-10 benchmark regime.
    """
    if include_demo:
        return list(documents)
    return [d for d in documents if d.get("collection", "eval") == "eval"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(path: Path, expected: str | None) -> bool:
    if not expected:
        print("    ..  no sha256 in manifest; skipping check")
        return True
    if _sha256(path) == expected:
        print("    ok  sha256 matches")
        return True
    print("    !!  sha256 MISMATCH (likely an arXiv re-render; content probably fine)")
    return False


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    !!  download failed: {e}")
        return False
    if not data.startswith(b"%PDF-"):
        print("    !!  response is not a PDF (got a landing page?)")
        return False
    dest.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=str,
        default=str(DOCS_PATH),
        help="Where to assemble the corpus (default %(default)s)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Checksum files already in --dest against the manifest; fetch nothing",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without copying or downloading"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Also include the demo collection (classic DL papers, 30papers.com); "
        "the default fetches only the verified-10 eval corpus",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["documents"]
    documents = _selected(manifest, include_demo=args.demo)
    excluded = len(manifest) - len(documents)

    if not args.dry_run and not args.verify_only:
        dest.mkdir(parents=True, exist_ok=True)

    placed = downloaded = mismatched = failed = 0

    for doc in documents:
        name = doc["filename"]
        target = dest / name

        if args.verify_only:
            if target.exists():
                print(f"[verify] {name}")
                if not _check(target, doc.get("sha256")):
                    mismatched += 1
            continue

        # forward-compat: a bundled, in-repo (e.g. CC-licensed) paper
        if doc.get("committed"):
            src = COMMITTED_DIR / name
            print(f"[committed] {name}")
            if args.dry_run:
                print(f"    would copy {src} -> {target}")
                placed += 1
                continue
            if not src.exists():
                print(f"    !!  missing from repo: {src}")
                failed += 1
                continue
            shutil.copy2(src, target)
            placed += 1
            if not _check(target, doc.get("sha256")):
                mismatched += 1
            continue

        url = doc.get("url")
        print(f"[arxiv] {name}")
        if not url or not doc.get("direct_pdf"):
            print(f"    !!  no direct-PDF url; fetch manually: {doc.get('abstract_url') or url}")
            failed += 1
            continue
        if args.dry_run:
            print(f"    would download {url}")
            downloaded += 1
            continue
        if downloaded:
            time.sleep(3)  # arXiv politeness between fetches
        if _download(url, target):
            downloaded += 1
            if not _check(target, doc.get("sha256")):
                mismatched += 1
        else:
            print(f"        try manually: {doc.get('abstract_url') or url}")
            failed += 1

    print("\n--- summary ---")
    print(f"  downloaded       : {downloaded}")
    print(f"  copied (in-repo) : {placed}")
    print(f"  checksum mismatch: {mismatched}")
    print(f"  failed           : {failed}")
    if excluded:
        print(f"  (demo collection not selected: {excluded} papers — add --demo to include them)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
