<script lang="ts">
  import DOMPurify from 'dompurify'
  import { marked } from 'marked'

  // Extensionless on purpose: `svelte-check` rejects a '.ts' import extension without
  // allowImportingTsExtensions, while `node:test` (citations.test.ts) requires one. Vite
  // resolves this to citations.ts; the two consumers just spell it differently.
  import { CITE_ANYWHERE, CITE_EXACT, CITE_SPLIT } from './citations'

  // The answer is model output over your documents, so it can carry whatever markup a document
  // did: a quoted `<img onerror>` or a `[link](javascript:…)` (marked >= 14 no longer filters
  // that scheme). DOMPurify strips scripts, handlers and `javascript:` URLs before `{@html}`
  // (docs/security.md S2, step S-3); the CSP is the second wall, not the only one. Nothing a
  // markdown answer needs is a form control or a style block, so those go too — they are the
  // spoofing half of S2. Citation buttons are added after this, by linkifyCitations, on the DOM.
  const SANITIZE = { FORBID_TAGS: ['style', 'form', 'input', 'button', 'textarea', 'select'] }
  let {
    source = '',
    onCitationClick,
    activeCitationN = null,
  }: {
    source?: string
    onCitationClick?: (n: number) => void
    activeCitationN?: number | null
  } = $props()
  const html = $derived(DOMPurify.sanitize(marked.parse(source, { async: false }), SANITIZE))

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

  // The citation forms we recognise (canonical [2] plus [Source 2] / [Sources 2, 4] / [2, 4] /
  // [2 and 4]) now live in ./citations.ts — a plain module so `node:test` can run them, pinned
  // against the backend parser by tests/fixtures/citation_vectors.json (KI-35). Presentation
  // only: each resolved number renders as a clean clickable [n]; the markdown is never rewritten.

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
