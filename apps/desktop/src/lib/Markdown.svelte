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

  // Turn bracketed citation markers like "[2]" into clickable buttons — but never inside a
  // <code>/<pre> span (a technical corpus can legitimately contain "[2]" as code, not a
  // citation) and never by touching raw HTML/attributes: only text nodes we already hold.
  function linkifyCitations(root: HTMLElement): void {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    const targets: Text[] = []
    let n: Node | null
    while ((n = walker.nextNode())) {
      if (!/\[\d+\]/.test(n.textContent ?? '')) continue
      if (hasAncestorMatching(n, root, (e) => e.tagName === 'CODE' || e.tagName === 'PRE')) continue
      if (hasAncestorMatching(n, root, (e) => e.classList.contains('citation'))) continue
      targets.push(n as Text)
    }
    for (const textNode of targets) {
      const frag = document.createDocumentFragment()
      for (const part of (textNode.textContent ?? '').split(/(\[\d+\])/g)) {
        const m = /^\[(\d+)\]$/.exec(part)
        if (m) {
          const btn = document.createElement('button')
          btn.type = 'button'
          btn.className = 'citation'
          btn.dataset.n = m[1]
          btn.textContent = part
          frag.appendChild(btn)
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
  .md :global(p) {
    margin: 0.4em 0;
  }
  .md :global(code) {
    background: var(--surface-2);
    padding: 0.1em 0.3em;
    border-radius: 4px;
    font-size: 0.9em;
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
