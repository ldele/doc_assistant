"""Self-organizing wiki / synthesis layer (Phase 7 / Feature 6).

A derived, human-readable markdown layer *on top of* the RAG corpus: cluster the
library into topics, emit one note per topic (LLM summary + tags + ``[[links]]``
+ source citations), and write them as Obsidian-compatible sidecar files under
``WIKI_DIR``. The Karpathy LLM-wiki pattern, applied as an additive index — not a
replacement for retrieval.

Why this exists: over a library too big to read directly, a synthesis layer is a
cheap browsable index, and — critically — it makes **knowledge gaps computable**:
a topic note with two thin citations and no links is a *structural* gap signal,
not an LLM opinion (Feature 6b; the substrate Phase 7 gap-detection needs).

Architecture (Enrichment-Layer Pattern): post-ingest, idempotent, **sidecar**
markdown — never mutates the chunk store. Notes are *regenerated* from the
current doc-similarity graph + provenance, so they stay in sync; a content change
re-clusters and the drift report flags what moved (6c).

Design split mirrors the other enrichment modules: a pure core (clustering, gap
signals, link derivation, markdown rendering, manifest diff — all unit-testable
with no DB/LLM) behind a thin impure layer (load the graph, sample chunks,
summarize via the provider protocol, orchestrate the build).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import structlog

from doc_assistant.config import (
    WIKI_CHUNK_SAMPLE,
    WIKI_DIR,
    WIKI_MIN_CITATIONS,
    WIKI_MIN_SIMILARITY,
    WIKI_USE_CONCEPT_COMMUNITIES,
)

log = structlog.get_logger(__name__)

MANIFEST_NAME = ".manifest.json"


# ============================================================
# Data classes
# ============================================================


@dataclass
class DocRef:
    """One member document of a topic cluster."""

    doc_id: str
    doc_hash: str
    filename: str
    title: str | None = None
    year: int | None = None
    keywords: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.title or self.filename


@dataclass
class SimEdge:
    """A directed doc-similarity edge (from the ``DocSimilarity`` sidecar)."""

    source: str
    target: str
    score: float


@dataclass
class WikiGapSignals:
    """Structural knowledge-gap markers for one topic note (Feature 6b).

    Reuses the *idea* of ``provenance.compute_confidence_signals`` (thinness,
    single-source) but over a note's source-document set rather than one answer's
    retrieved chunks. No LLM judgement — these are countable structural facts.
    """

    n_sources: int
    n_links: int
    citation_thin: bool
    single_source: bool
    no_links: bool

    def any(self) -> bool:
        return self.citation_thin or self.single_source or self.no_links

    @property
    def reasons(self) -> list[str]:
        out: list[str] = []
        if self.single_source:
            out.append("single-source")
        elif self.citation_thin:
            out.append("citation-thin")
        if self.no_links:
            out.append("isolated (no links)")
        return out


@dataclass
class TopicNote:
    """One topic note — the full payload for rendering one wiki file."""

    topic_id: str
    title: str
    docs: list[DocRef]
    summary: str
    tags: list[str]
    links: list[tuple[str, str]]  # (target_topic_id, target_title)
    gap: WikiGapSignals

    @property
    def source_hashes(self) -> list[str]:
        return sorted(d.doc_hash for d in self.docs)


# ============================================================
# Pure core — clustering, gaps, links (no I/O)
# ============================================================


def cluster_documents(
    doc_ids: list[str],
    edges: list[SimEdge],
    *,
    min_similarity: float = WIKI_MIN_SIMILARITY,
) -> list[list[str]]:
    """Connected-components clustering over edges at/above ``min_similarity``.

    Union-find over the similarity graph: two docs share a topic when an edge
    between them clears the threshold. Singletons (a doc with no qualifying edge)
    form their own one-member topic. Deterministic: clusters sorted size-desc then
    by first member; members sorted. No new dependency (the roadmap's
    NetworkX/Leiden is Feature 7's concern, not this).
    """
    parent = {d: d for d in doc_ids}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        if e.score >= min_similarity and e.source in parent and e.target in parent:
            union(e.source, e.target)

    groups: dict[str, list[str]] = defaultdict(list)
    for d in doc_ids:
        groups[find(d)].append(d)
    clusters = [sorted(members) for members in groups.values()]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


def topic_id_for(doc_hashes: list[str]) -> str:
    """Stable topic id = short hash of the sorted member doc_hashes.

    Membership-derived (pre-LLM) so filenames are idempotent and a content change
    that re-clusters yields a new id — which the drift report surfaces as a
    removed+added pair.
    """
    payload = ",".join(sorted(doc_hashes))
    return "topic-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def compute_gap_signals(
    n_sources: int, n_links: int, *, min_citations: int = WIKI_MIN_CITATIONS
) -> WikiGapSignals:
    """Structural gap markers for a topic (pure). See ``WikiGapSignals``."""
    return WikiGapSignals(
        n_sources=n_sources,
        n_links=n_links,
        citation_thin=n_sources < min_citations,
        single_source=n_sources == 1,
        no_links=n_links == 0,
    )


def compute_links(clusters: list[list[str]], edges: list[SimEdge]) -> dict[int, set[int]]:
    """Map each cluster index → indices of clusters it links to.

    A link is a stored similarity edge crossing two clusters. (Edges at/above the
    clustering threshold were merged into one cluster, so a *cross-cluster* edge is
    by construction a weaker-but-real relation — exactly a ``[[link]]``.)
    """
    cluster_of: dict[str, int] = {}
    for i, members in enumerate(clusters):
        for d in members:
            cluster_of[d] = i
    links: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        ci, cj = cluster_of.get(e.source), cluster_of.get(e.target)
        if ci is not None and cj is not None and ci != cj:
            links[ci].add(cj)
            links[cj].add(ci)
    return links


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase kebab slug (for tags / alias readability)."""
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "untitled"


def fallback_title(docs: list[DocRef]) -> str:
    """Topic title when no LLM ran: the dominant keyword, else the first doc's label."""
    keywords = Counter(k for d in docs for k in d.keywords if k.strip())
    if keywords:
        return keywords.most_common(1)[0][0].title()
    return docs[0].label if docs else "Untitled topic"


# ============================================================
# Pure core — markdown rendering (Obsidian-compatible, 6d)
# ============================================================


def _yaml_list(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(v) for v in values) + "]"


def render_note_markdown(note: TopicNote) -> str:
    """Render one ``TopicNote`` to Obsidian-compatible markdown.

    YAML frontmatter (``aliases``/``tags``/``sources``/``gap``) + an H1 + the
    summary + a Related section of ``[[topic-id|Title]]`` wikilinks + a Sources
    list + any structural gap markers. Deterministic — no timestamps in the body
    (drift is tracked by the manifest, not by mtime).
    """
    fm = [
        "---",
        f"topic_id: {note.topic_id}",
        f"aliases: {_yaml_list([note.title])}",
        f"tags: {_yaml_list(note.tags)}",
        f"sources: {_yaml_list([d.filename for d in note.docs])}",
    ]
    if note.gap.any():
        fm.append(f"gap: {_yaml_list(note.gap.reasons)}")
    fm.append("---")

    body = [f"# {note.title}", "", note.summary.strip() or "_No summary generated._"]

    if note.links:
        body += ["", "## Related"]
        body += [f"- [[{tid}|{title}]]" for tid, title in note.links]

    body += ["", "## Sources"]
    for d in note.docs:
        year = f" ({d.year})" if d.year else ""
        # Avoid "filename — `filename`" when the doc has no extracted title.
        if d.title and d.title != d.filename:
            body.append(f"- {d.title}{year} — `{d.filename}`")
        else:
            body.append(f"- `{d.filename}`{year}")

    if note.gap.any():
        body += ["", f"> **Knowledge-gap signals:** {', '.join(note.gap.reasons)}."]

    return "\n".join([*fm, "", *body]) + "\n"


# ============================================================
# Pure core — manifest & drift (6c)
# ============================================================


def build_manifest(notes: list[TopicNote]) -> dict[str, list[str]]:
    """topic_id → sorted source doc_hashes (the drift fingerprint)."""
    return {n.topic_id: n.source_hashes for n in notes}


@dataclass
class WikiDrift:
    """What changed between two wiki builds."""

    added: list[str]
    removed: list[str]

    def any(self) -> bool:
        return bool(self.added or self.removed)


def diff_manifests(old: dict[str, list[str]], new: dict[str, list[str]]) -> WikiDrift:
    """Topics added / removed since the last build.

    Because ``topic_id`` is a hash of its source set, a topic whose underlying
    sources changed appears as the old id *removed* and a new id *added* — exactly
    "flag notes whose sources changed". Stable topics appear in neither list.
    """
    return WikiDrift(
        added=sorted(set(new) - set(old)),
        removed=sorted(set(old) - set(new)),
    )


# ============================================================
# Impure layer — DB / Chroma / LLM
# ============================================================


def load_doc_graph(embedding_model: str | None = None) -> tuple[list[DocRef], list[SimEdge]]:
    """Load non-archived documents + their similarity edges from SQLite."""
    from sqlalchemy import select

    from doc_assistant.db.models import DocSimilarity, Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        docs = [
            DocRef(
                doc_id=str(d.id),
                doc_hash=d.doc_hash,
                filename=d.filename,
                title=d.title,
                year=d.year,
                keywords=[k.name for k in d.keywords],
            )
            for d in session.execute(
                select(Document).where(Document.is_archived.is_(False))
            ).scalars()
        ]
        edge_stmt = select(
            DocSimilarity.source_document_id,
            DocSimilarity.target_document_id,
            DocSimilarity.score,
        )
        if embedding_model is not None:
            edge_stmt = edge_stmt.where(DocSimilarity.embedding_model == embedding_model)
        edges = [
            SimEdge(source=str(s), target=str(t), score=float(sc))
            for s, t, sc in session.execute(edge_stmt).all()
        ]
    return docs, edges


def load_communities(
    docs: list[DocRef], *, graph_dir: Path | None = None
) -> list[list[str]] | None:
    """Document clusters from the Feature 7 concept-graph sidecar — or ``None``.

    The threshold-free replacement for ``cluster_documents``: reads
    ``CONCEPT_GRAPH_DIR/graph.json`` (Louvain communities, written by
    ``build_concept_graph``) plus the per-doc extraction cache, and returns the
    document grouping ``concept_graph.doc_clusters_from_graph`` derives from it —
    docs grouped by the concept-community they most belong to. Same return shape as
    ``cluster_documents`` (``list[list[doc_id]]``), so it drops straight into
    ``_assemble_notes``.

    Returns ``None`` (→ the caller falls back to absolute-cosine clustering) when
    the graph can't be trusted for the *current* corpus:

    * ``graph.json`` is absent or unreadable, or
    * **stale** — some non-archived document has no cached extraction for its
      *current* ``doc_hash`` (a content change since the last ``build_concept_graph
      --apply`` re-keys the cache, so the missing entry reads as stale).

    Read-only: never extracts, never calls an LLM, never writes the sidecar.
    """
    from doc_assistant import concept_graph as cg
    from doc_assistant.config import CONCEPT_GRAPH_DIR

    root = graph_dir or CONCEPT_GRAPH_DIR
    graph_path = root / cg.GRAPH_NAME
    if not graph_path.exists():
        log.info("concept_graph_absent", path=str(graph_path), hint="using cosine clustering")
        return None
    try:
        graph = cg.graph_from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
    except Exception as e:
        log.warning("concept_graph_unreadable", error=str(e), hint="using cosine clustering")
        return None

    extractions: list[cg.DocExtraction] = []
    missing: list[str] = []
    for d in docs:
        ex = cg.load_cached_extraction(root, d.doc_hash)
        if ex is None:
            missing.append(d.filename)
            continue
        # The cache stores the doc_id at extraction time; realign to the current PK
        # so the returned clusters reference live doc_ids regardless of any reassign.
        ex.doc_id = d.doc_id
        extractions.append(ex)
    if missing:
        log.info(
            "concept_graph_stale",
            missing=len(missing),
            total=len(docs),
            examples=", ".join(missing[:3]),
            hint="run `build_concept_graph --apply`; using cosine clustering",
        )
        return None

    return cg.doc_clusters_from_graph(graph, extractions)


def sample_chunks(doc_ids: list[str], *, per_doc: int = WIKI_CHUNK_SAMPLE) -> dict[str, list[str]]:
    """Sample up to ``per_doc`` chunk excerpts per document from the baseline store."""
    from doc_assistant.config import CHROMA_PATH
    from doc_assistant.embeddings import get_collection_name

    try:
        import chromadb
    except ImportError:  # pragma: no cover - dep present in dev env
        return {}

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        coll = client.get_collection(get_collection_name())
    except Exception:
        log.warning("no_baseline_collection", hint="run ingest first; wiki summaries will be thin")
        return {}

    out: dict[str, list[str]] = {}
    for doc_id in doc_ids:
        data = coll.get(where={"document_id": doc_id}, include=["documents"], limit=per_doc)
        texts = [str(t)[:300] for t in (data.get("documents") or []) if t]
        if texts:
            out[doc_id] = texts
    return out


_SUMMARY_PROMPT = """You are writing one topic note for a research wiki, summarising a \
CLUSTER of related papers. Use ONLY the material provided — titles, keywords, and \
excerpts. Do not invent findings.

PAPERS IN THIS TOPIC:
{material}

Return JSON only, no prose, no markdown fence:
{{"title": "<a short topic title, 2-6 words>", \
"summary": "<2-4 sentences on what this topic is about and what the papers contribute>", \
"tags": ["<3-6 lowercase topic tags>"]}}"""


def _format_material(docs: list[DocRef], chunk_samples: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for d in docs:
        bits = [f"- {d.label}" + (f" ({d.year})" if d.year else "")]
        if d.keywords:
            bits.append(f"  keywords: {', '.join(d.keywords[:8])}")
        for ex in chunk_samples.get(d.doc_id, [])[:2]:
            bits.append(f"  excerpt: {ex}")
        parts.append("\n".join(bits))
    return "\n".join(parts)


def _extract_json(text: str) -> str:
    """Best-effort: strip a markdown fence, else take the outermost {...} span."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            t = t[start : end + 1]
    return t


def summarize_cluster(
    docs: list[DocRef],
    chunk_samples: dict[str, list[str]],
    client: Any,
    *,
    max_tokens: int = 400,
) -> tuple[str, str, list[str]]:
    """LLM topic summary → ``(title, summary, tags)``. Falls back on any failure.

    ``client`` is an ``llm.LLMClient`` (provider-configurable, local-capable).
    A transport/parse failure degrades to a derived title + empty summary rather
    than aborting the build — a thin note is still a useful gap signal.
    """
    prompt = _SUMMARY_PROMPT.format(material=_format_material(docs, chunk_samples))
    try:
        raw = client.complete(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens
        )
        parsed = json.loads(_extract_json(raw))
        title = str(parsed.get("title") or "").strip() or fallback_title(docs)
        summary = str(parsed.get("summary") or "").strip()
        tags_raw = parsed.get("tags") or []
        tags = [slugify(str(t)) for t in tags_raw if str(t).strip()][:6]
        return title, summary, tags
    except Exception as e:
        log.warning("wiki_summary_failed", error=str(e), hint="using fallback")
        return fallback_title(docs), "", []


@dataclass
class WikiBuildResult:
    """What a ``build_wiki`` run produced."""

    notes: list[TopicNote]
    drift: WikiDrift
    written: int
    removed_files: int
    applied: bool
    #: Which clustering primitive produced the topics this run — ``"concept-graph"``
    #: (Feature 7 communities) or ``"cosine-threshold"`` (the absolute-cosine
    #: fallback). Lets the CLI report which path ran, since the concept path silently
    #: falls back when the sidecar is absent/stale.
    clustering: str = "cosine-threshold"


def _assemble_notes(
    docs: list[DocRef],
    edges: list[SimEdge],
    *,
    min_similarity: float,
    min_citations: int,
    summarize: Any | None,
    per_doc_chunks: int,
    concept_clusters: list[list[str]] | None = None,
) -> list[TopicNote]:
    """Cluster + (optionally) summarise into ``TopicNote``s. ``summarize`` None = dry-run.

    ``concept_clusters`` (the Feature 7 community grouping, when present) replaces
    the absolute-cosine ``cluster_documents`` — the re-point's one behavioural
    branch. ``[[links]]`` still derive from the stored similarity edges crossing the
    chosen clusters, so a corpus with no ``DocSimilarity`` edges simply yields no
    links (see decisions.md → Deferred Improvements for the concept-edge follow-up).
    """
    by_id = {d.doc_id: d for d in docs}
    clusters = (
        concept_clusters
        if concept_clusters is not None
        else cluster_documents([d.doc_id for d in docs], edges, min_similarity=min_similarity)
    )
    links_by_idx = compute_links(clusters, edges)

    # Pre-compute each cluster's topic_id + title so links can resolve titles.
    members = [[by_id[i] for i in c if i in by_id] for c in clusters]
    topic_ids = [topic_id_for([d.doc_hash for d in m]) for m in members]

    titles: list[str] = []
    summaries: list[str] = []
    tagsets: list[list[str]] = []
    for m in members:
        if summarize is not None:
            samples = sample_chunks([d.doc_id for d in m], per_doc=per_doc_chunks)
            title, summary, tags = summarize(m, samples)
        else:
            title, summary, tags = fallback_title(m), "", []
        titles.append(title)
        summaries.append(summary)
        tagsets.append(tags)

    notes: list[TopicNote] = []
    for i, m in enumerate(members):
        link_refs = sorted(links_by_idx.get(i, set()))
        notes.append(
            TopicNote(
                topic_id=topic_ids[i],
                title=titles[i],
                docs=m,
                summary=summaries[i],
                tags=tagsets[i],
                links=[(topic_ids[j], titles[j]) for j in link_refs],
                gap=compute_gap_signals(len(m), len(link_refs), min_citations=min_citations),
            )
        )
    return notes


def build_wiki(
    *,
    apply: bool,
    force: bool = False,
    client: Any | None = None,
    min_similarity: float = WIKI_MIN_SIMILARITY,
    min_citations: int = WIKI_MIN_CITATIONS,
    per_doc_chunks: int = WIKI_CHUNK_SAMPLE,
    embedding_model: str | None = None,
    use_concept_communities: bool = WIKI_USE_CONCEPT_COMMUNITIES,
    wiki_dir: Path | None = None,
    graph_dir: Path | None = None,
) -> WikiBuildResult:
    """Cluster the library into topic notes; write the sidecar on ``apply``.

    Dry-run (``apply=False`` or no ``client``) clusters + derives titles with no
    LLM call. ``apply`` with a ``client`` summarises each topic, writes
    ``WIKI_DIR/{topic_id}.md`` + the manifest, reports drift, and removes orphan
    notes. Never touches the chunk store.

    Clustering primitive: ``use_concept_communities`` (default off) groups documents
    by the Feature 7 concept-graph communities instead of the absolute-cosine
    threshold — but only if that sidecar is present and fresh; otherwise it silently
    falls back to ``cluster_documents`` so the run never fails for a missing graph.
    """
    root = wiki_dir or WIKI_DIR
    docs, edges = load_doc_graph(embedding_model)

    concept_clusters = (
        load_communities(docs, graph_dir=graph_dir) if use_concept_communities else None
    )
    clustering = "concept-graph" if concept_clusters is not None else "cosine-threshold"

    summarize = (
        partial(summarize_cluster, client=client) if (apply and client is not None) else None
    )

    notes = _assemble_notes(
        docs,
        edges,
        min_similarity=min_similarity,
        min_citations=min_citations,
        summarize=summarize,
        per_doc_chunks=per_doc_chunks,
        concept_clusters=concept_clusters,
    )

    old_manifest = _read_manifest(root)
    new_manifest = build_manifest(notes)
    drift = diff_manifests(old_manifest, new_manifest)

    written = removed = 0
    if apply:
        root.mkdir(parents=True, exist_ok=True)
        if force:
            removed += _clear_notes(root)
        for note in notes:
            (root / f"{note.topic_id}.md").write_text(render_note_markdown(note), encoding="utf-8")
            written += 1
        # Drop notes whose topic no longer exists (re-cluster orphans).
        for stale in drift.removed:
            stale_path = root / f"{stale}.md"
            if stale_path.exists():
                stale_path.unlink()
                removed += 1
        _write_manifest(root, new_manifest)

    return WikiBuildResult(
        notes=notes,
        drift=drift,
        written=written,
        removed_files=removed,
        applied=apply,
        clustering=clustering,
    )


def _read_manifest(root: Path) -> dict[str, list[str]]:
    path = root / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): [str(h) for h in v] for k, v in data.items()}
    except Exception:
        return {}


def _write_manifest(root: Path, manifest: dict[str, list[str]]) -> None:
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _clear_notes(root: Path) -> int:
    """Delete every generated note file (not the manifest). Returns the count."""
    if not root.exists():
        return 0
    count = 0
    for p in root.glob("topic-*.md"):
        p.unlink()
        count += 1
    return count
