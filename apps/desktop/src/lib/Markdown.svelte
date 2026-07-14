<script lang="ts">
  import { marked } from 'marked'

  // Content is our own backend's markdown (the answer + pre-rendered blocks) — trusted,
  // local, single-user. A hardened build would run it through DOMPurify.
  let {
    source = '',
    onCitationClick,
    activeCitationN = null,
  }: {
    source?: string
    onCitationClick?: (n: number) => void
    activeCitationN?: number | null
  } = $props()
  const html = $derived(marked.parse(source, { async: false }))

  let el = $state<HTMLDivElement | null>(null)

  // Re-linkify after every render of {@html html}, and re-sync the active highlight whenever
  // either changes. linkifyCitations is idempotent (skips text already inside a .citation
  // button) so re-running it on an activeCitationN-only change is a no-op walk, not a rebuild.
  $effect(() => {
    void html
    if (!el) return
    linkifyCitations(el)
    syncActiveCitation(el, activeCitationN)
  })

  // One delegated listener attached imperatively (not a template onclick) — the button set
  // churns every time linkifyCitations re-runs, so binding it once per mounted container,
  // rather than per button, is both cheaper and correct.
  $effect(() => {
    if (!el) return
    const node = el
    node.addEventListener('click', onClick)
    return () => node.removeEventListener('click', onClick)
  })

  function hasAncestorMatching(node: Node, root: HTMLElement, pred: (e: HTMLElement) => boolean): boolean {
    let p: Node | null = node.parentNode
    while (p && p !== root) {
      if (p instanceof HTMLElement && pred(p)) return true
      p = p.parentNode
    }
    return false
  }

  // Citation forms we recognise: canonical [2] AND the non-canonical-but-unambiguous ones the
  // model emits — [Source 2], [Sources 2, 4], [2, 4], [2 and 4] — matching the backend parser
  // (synthesis.cited_source_numbers). Presentation only: each resolved number renders as a clean
  // clickable [n]; the source markdown is never rewritten. Non-global (no lastIndex state).
  const CITE_BODY = String.raw`\[\s*(?:sources?|refs?)?\s*\d+(?:\s*(?:,|;|&|and)\s*\d+)*\s*\]`
  const CITE_ANYWHERE = new RegExp(CITE_BODY, 'i')
  const CITE_SPLIT = new RegExp(`(${CITE_BODY})`, 'i')
  const CITE_EXACT = new RegExp(`^${CITE_BODY}$`, 'i')

  // Turn citation markers into clickable buttons — but never inside a <code>/<pre> span (a
  // technical corpus can legitimately contain "[2]" as code, not a citation) and never by
  // touching raw HTML/attributes: only text nodes we already hold.
  function linkifyCitations(root: HTMLElement): void {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    const targets: Text[] = []
    let n: Node | null
    while ((n = walker.nextNode())) {
      if (!CITE_ANYWHERE.test(n.textContent ?? '')) continue
      if (hasAncestorMatching(n, root, (e) => e.tagName === 'CODE' || e.tagName === 'PRE')) continue
      if (hasAncestorMatching(n, root, (e) => e.classList.contains('citation'))) continue
      targets.push(n as Text)
    }
    for (const textNode of targets) {
      const frag = document.createDocumentFragment()
      for (const part of (textNode.textContent ?? '').split(CITE_SPLIT)) {
        if (part && CITE_EXACT.test(part)) {
          // One [n] button per source number in the token (e.g. "[Sources 2, 4]" → [2][4]).
          for (const num of part.match(/\d+/g) ?? []) {
            const btn = document.createElement('button')
            btn.type = 'button'
            btn.className = 'citation'
            btn.dataset.n = num
            btn.textContent = `[${num}]`
            frag.appendChild(btn)
          }
        } else if (part) {
          frag.appendChild(document.createTextNode(part))
        }
      }
      textNode.replaceWith(frag)
    }
  }

  function syncActiveCitation(root: HTMLElement, n: number | null): void {
    for (const btn of root.querySelectorAll<HTMLButtonElement>('.citation')) {
      btn.classList.toggle('active', n !== null && Number(btn.dataset.n) === n)
    }
  }

  // One delegated listener on the container root rather than one per button — the button set
  // changes every time linkifyCitations re-runs.
  function onClick(e: MouseEvent): void {
    const btn = (e.target as HTMLElement).closest('.citation') as HTMLElement | null
    if (!btn || !onCitationClick) return
    const n = Number(btn.dataset.n)
    if (Number.isFinite(n)) onCitationClick(n)
  }
</script>

<div class="md" bind:this={el}>{@html html}</div>

<style>
  /* Reading surface — Spectral serif for the answer prose (paper & ink; V1). Code/citations
     opt back out below. V2: cap the line length at --measure (~68ch), left-aligned, so prose reads
     at a comfortable measure while wider elements (source/provenance cards) keep the full column. */
  .md {
    font-family: var(--font-serif);
    line-height: 1.6;
    max-width: var(--measure);
  }
  .md :global(p) {
    margin: 0.4em 0;
  }
  .md :global(code) {
    background: var(--surface-2);
    padding: 0.1em 0.3em;
    border-radius: 4px;
    font-size: 0.9em;
    font-family: ui-monospace, 'Cascadia Code', 'Segoe UI Mono', Menlo, Consolas, monospace;
  }
  .md :global(h1),
  .md :global(h2),
  .md :global(h3) {
    margin: 0.6em 0 0.3em;
  }
  .md :global(.citation) {
    font: inherit;
    cursor: pointer;
    border: none;
    background: none;
    padding: 0;
    color: var(--accent);
  }
  .md :global(.citation.active) {
    background: var(--accent);
    color: var(--accent-fg);
    border-radius: 3px;
    padding: 0 0.15em;
  }
</style>
