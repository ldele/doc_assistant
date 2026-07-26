<script lang="ts">
  // Bottom status bar (ambient, full-width): connection dot + corpus/model info. Thin and quiet so
  // it never competes with the chat composer sitting just above it.
  //
  // Zero props by design — it renders `shell.status` / `shell.health`, which the readiness gate in
  // App.svelte writes. Ambient status, not navigation.
  import { shell } from './shell.svelte'
</script>

<div class="statusbar" role="status" aria-live="polite">
  <span
    class="status-dot"
    class:ok={shell.status === 'ready'}
    class:wait={shell.status === 'connecting'}
    class:off={shell.status === 'down'}
    aria-hidden="true"
  ></span>
  {#if shell.status === 'ready' && shell.health}
    <span class="status-meta">
      {shell.health.chunk_count.toLocaleString()} chunks · {shell.health.model} · {shell.health.embedding_model}
    </span>
  {:else if shell.status === 'connecting'}
    <span class="status-meta">starting the engine…</span>
  {:else}
    <span class="status-meta err">backend unreachable. Run <code>just api</code></span>
  {/if}
</div>

<style>
  /* ---- bottom status bar (ambient) ---- */
  .statusbar {
    flex: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.9rem;
    border-top: 1px solid var(--border);
    background: var(--bg);
    min-height: 1.6rem;
  }
  .status-dot {
    flex: none;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--fg-2);
  }
  .status-dot.ok {
    background: var(--ok, #2e9e5b);
  }
  .status-dot.wait {
    background: var(--warn-fg);
  }
  .status-dot.off {
    background: var(--danger);
  }
  .status-meta {
    font-size: var(--text-meta);
    color: var(--fg-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .status-meta.err {
    color: var(--warn-fg);
  }
  .status-meta code {
    font-size: 0.92em;
  }
</style>
