<script lang="ts">
  // About Provenote — opened from the toolbar app menu (☰). Product blurb + the live corpus stats,
  // and the standing ANZSRC CC-BY 4.0 attribution the taxonomy seed owes (ADR-028). Read-only;
  // scrim + centered card + Esc-to-close, matching the other hand-rolled modals.
  import appMark from '../assets/brand/app-mark.png'
  import Icon from './Icon.svelte'

  let {
    chunks = null,
    model = null,
    embedding = null,
    onClose,
  }: {
    chunks?: number | null
    model?: string | null
    embedding?: string | null
    onClose: () => void
  } = $props()

  function onKey(e: KeyboardEvent): void {
    if (e.key === 'Escape') onClose()
  }
</script>

<svelte:window onkeydown={onKey} />
<div class="scrim" onclick={onClose} role="presentation"></div>
<div class="modal" role="dialog" aria-modal="true" aria-label="About Provenote">
  <div class="mhead">
    <div class="brand">
      <span class="mark"><img src={appMark} alt="" width="34" height="34" /></span>
      <span class="wordmark">proven<span class="wm-accent">ote</span></span>
    </div>
    <button class="iconbtn" onclick={onClose} aria-label="Close" type="button">
      <Icon name="x" size={16} />
    </button>
  </div>

  <p class="blurb">
    A local-first research assistant: it answers questions grounded in your own documents, with inline
    citations, provenance, and per-claim review — not a general-purpose chatbot.
  </p>

  {#if chunks !== null}
    <dl class="stats">
      <div><dt>Corpus</dt><dd>{chunks.toLocaleString()} chunks</dd></div>
      {#if model}<div><dt>Answer model</dt><dd>{model}</dd></div>{/if}
      {#if embedding}<div><dt>Embeddings</dt><dd>{embedding}</dd></div>{/if}
    </dl>
  {/if}

  <p class="attribution">
    Field taxonomy seeded from the ANZSRC 2020 classification (Australian and New Zealand Standard
    Research Classification), used under <strong>CC BY 4.0</strong>.
  </p>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: color-mix(in srgb, var(--fg) 32%, transparent);
    z-index: 40;
  }
  .modal {
    position: fixed;
    z-index: 41;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: min(92vw, 440px);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .mhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .mark {
    flex: none;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    overflow: hidden;
    display: inline-flex;
    box-shadow: var(--shadow-1);
  }
  .mark img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .wordmark {
    font-family: var(--font-serif);
    font-size: var(--text-title);
    color: var(--fg);
  }
  .wm-accent {
    color: var(--accent-wordmark);
  }
  .iconbtn {
    display: inline-flex;
    padding: 0.2rem;
    border: none;
    background: none;
    color: var(--fg-2);
    border-radius: 6px;
    cursor: pointer;
  }
  .iconbtn:hover {
    color: var(--fg);
    background: var(--surface-2);
  }
  .blurb {
    margin: 0;
    font-size: 0.85rem;
    color: var(--fg-2);
    line-height: 1.6;
  }
  .stats {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    padding: var(--space-3);
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
  }
  .stats div {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .stats dt {
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .stats dd {
    margin: 0;
    font-size: 0.78rem;
    color: var(--fg);
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .attribution {
    margin: 0;
    padding-top: var(--space-3);
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--fg-2);
    line-height: 1.5;
  }
</style>
