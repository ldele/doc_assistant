<script lang="ts">
  import DOMPurify from 'dompurify'
  import { onDestroy } from 'svelte'
  import { marked } from 'marked'

  // Extensionless on purpose: `svelte-check` rejects a '.ts' import extension without
  // allowImportingTsExtensions, while `node:test` (citations.test.ts) requires one. Vite
  // resolves this to citations.ts; the two consumers just spell it differently.
  import { CITE_ANYWHERE, CITE_EXACT, CITE_SPLIT } from '../../lib/chat/citations'
  import { linkAllowed } from '../../lib/chat/links'
  import { locateClaims, type ClaimKind } from '../../lib/chat/claims'
  import { tooltip } from '../../lib/shell/tooltip'

  /** A claim to highlight (A8b, G4): its key (the claim's number), its colour and its text. */
  interface ClaimMark {
    key: string
    kind: ClaimKind
    text: string
    tip: string
  }

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
    allowedLinks = null,
    claims = [],
    activeClaimKey = null,
    onClaimClick,
  }: {
    source?: string
    onCitationClick?: (n: number) => void
    activeCitationN?: number | null
    /** A model's answer passes the web links its passages carry (A2); any other web link is
     *  shown as text. `null` (the default) is text the app wrote, whose links all stay. */
    allowedLinks?: readonly string[] | null
    /** The answer's claims to colour (A8b): matched to the text in order; one not found is left
     *  plain, and nothing is guessed. */
    claims?: ClaimMark[]
    activeClaimKey?: string | null
    onClaimClick?: (key: string) => void
  } = $props()
  const html = $derived(DOMPurify.sanitize(marked.parse(source, { async: false }), SANITIZE))

  let el = $state<HTMLDivElement | null>(null)

  // Re-linkify after every render of {@html html}, and re-sync the active highlight whenever
  // either changes. linkifyCitations is idempotent (skips text already inside a .citation
  // button) so re-running it on an activeCitationN-only change is a no-op walk, not a rebuild.
  $effect(() => {
    void html
    if (!el) return
    defuseLinks(el, allowedLinks)
    linkifyCitations(el)
    syncActiveCitation(el, activeCitationN)
  })

  // A8b: colour the claims once per render of the answer, then keep the open one marked. The
  // spans' hover text goes through the app's tooltip, as a button's does.
  let tips: { destroy?: () => void }[] = []
  $effect(() => {
    void html
    const marks = claims
    if (!el) return
    if (marks.length > 0 && !el.querySelector('.claim')) {
      for (const t of tips) t.destroy?.()
      tips = highlightClaims(el, marks)
    }
  })
  onDestroy(() => {
    for (const t of tips) t.destroy?.()
  })
  $effect(() => {
    void html
    void claims
    const key = activeClaimKey
    if (!el) return
    for (const span of el.querySelectorAll<HTMLElement>('.claim')) {
      span.classList.toggle('active', key !== null && span.dataset.claim === key)
    }
  })

  /** The answer's text as shown, citation buttons left out, and where each text node starts. */
  function flatText(root: HTMLElement): { flat: string; nodes: { node: Text; start: number }[] } {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    const nodes: { node: Text; start: number }[] = []
    let flat = ''
    let n: Node | null
    while ((n = walker.nextNode())) {
      if (hasAncestorMatching(n, root, (e) => e.classList.contains('citation'))) continue
      nodes.push({ node: n as Text, start: flat.length })
      flat += n.textContent ?? ''
    }
    return { flat, nodes }
  }

  function highlightClaims(root: HTMLElement, marks: ClaimMark[]): { destroy?: () => void }[] {
    const { flat, nodes } = flatText(root)
    const where = locateClaims(flat, marks.map((m) => m.text))
    // Per text node, the pieces to wrap, wrapped from its end so earlier offsets stay true.
    const pieces = new Map<Text, { from: number; to: number; mark: ClaimMark }[]>()
    where.forEach((range, i) => {
      if (!range) return
      for (const { node, start } of nodes) {
        const len = node.textContent?.length ?? 0
        const from = Math.max(range.start, start) - start
        const to = Math.min(range.end, start + len) - start
        if (to <= from) continue
        const list = pieces.get(node) ?? []
        list.push({ from, to, mark: marks[i] })
        pieces.set(node, list)
      }
    })
    const made: { destroy?: () => void }[] = []
    for (const [node, list] of pieces) {
      list.sort((a, b) => b.from - a.from)
      for (const { from, to, mark } of list) {
        if (!(node.textContent ?? '').slice(from, to).trim()) continue
        const tail = node.splitText(from)
        tail.splitText(to - from)
        const span = document.createElement('span')
        span.className = `claim ${mark.kind}`
        span.dataset.claim = mark.key
        span.tabIndex = 0
        span.setAttribute('role', 'button')
        tail.replaceWith(span)
        span.appendChild(tail)
        const t = tooltip(span, mark.tip)
        if (t) made.push(t)
      }
    }
    return made
  }

  // One delegated listener attached imperatively (not a template onclick) — the button set
  // churns every time linkifyCitations re-runs, so binding it once per mounted container,
  // rather than per button, is both cheaper and correct.
  $effect(() => {
    if (!el) return
    const node = el
    node.addEventListener('click', onClick)
    node.addEventListener('keydown', onKey)
    return () => {
      node.removeEventListener('click', onClick)
      node.removeEventListener('keydown', onKey)
    }
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

  // A2: a web link whose address no passage carries becomes plain text showing that address,
  // so a link a document told the model to write cannot be clicked (lib/chat/links.ts).
  function defuseLinks(root: HTMLElement, allowed: readonly string[] | null): void {
    for (const a of Array.from(root.querySelectorAll<HTMLAnchorElement>('a[href]'))) {
      const href = a.getAttribute('href') ?? ''
      if (linkAllowed(href, allowed)) continue
      const span = document.createElement('span')
      span.className = 'unsourced-link'
      const label = a.textContent ?? ''
      span.textContent = label && label !== href ? `${label} (${href})` : href
      const note = document.createElement('span')
      note.className = 'unsourced-note'
      note.textContent = ' · link not in your sources'
      span.appendChild(note)
      a.replaceWith(span)
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
    if (btn) {
      const n = Number(btn.dataset.n)
      if (onCitationClick && Number.isFinite(n)) onCitationClick(n)
      return
    }
    const claim = (e.target as HTMLElement).closest('.claim') as HTMLElement | null
    if (claim?.dataset.claim && onClaimClick) onClaimClick(claim.dataset.claim)
  }
  // A claim span is a button to the keyboard too: Enter or Space opens its passage.
  function onKey(e: KeyboardEvent): void {
    if (e.key !== 'Enter' && e.key !== ' ') return
    const claim = (e.target as HTMLElement).closest('.claim') as HTMLElement | null
    if (!claim?.dataset.claim || !onClaimClick) return
    e.preventDefault()
    onClaimClick(claim.dataset.claim)
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
  .md :global(.unsourced-link) {
    overflow-wrap: anywhere;
  }
  .md :global(.unsourced-note) {
    color: var(--fg-2);
    font-size: 0.85em;
  }
  /* A8b, G4: lilac = cites a well-ranked passage; amber = cites one that ranked low; the open
     one is the darker lilac. Tokens only, so both themes follow. */
  .md :global(.claim) {
    border-radius: 3px;
    cursor: pointer;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
  }
  .md :global(.claim.cited) {
    background: color-mix(in srgb, var(--accent) 14%, transparent);
  }
  .md :global(.claim.cited.active) {
    background: color-mix(in srgb, var(--accent) 34%, transparent);
  }
  .md :global(.claim.weak) {
    background: var(--mark-bg);
  }
  .md :global(.claim.weak.active) {
    outline: 1px solid var(--warn-fg);
  }
  .md :global(.claim:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .md :global(.citation.active) {
    background: var(--accent);
    color: var(--accent-fg);
    border-radius: 3px;
    padding: 0 0.15em;
  }
</style>
