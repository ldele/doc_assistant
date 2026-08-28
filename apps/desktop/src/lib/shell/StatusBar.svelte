<script lang="ts">
  // Bottom status bar (ambient, full-width): connection dot + corpus/model info. Thin and quiet so
  // it never competes with the chat composer sitting just above it.
  //
  // Zero props by design — it renders `shell.status` / `shell.health`, which the readiness gate in
  // App.svelte writes. Ambient status, not navigation.
  import { shell } from './shell.svelte'
  import IngestChip from './IngestChip.svelte'
</script>

<div class="statusbar" role="status" aria-live="polite">
  <span
    class="status-dot"
    class:ok={shell.status === 'ready'}
    class:wait={shell.status === 'connecting'}
    class:off={shell.status === 'down'}
    aria-hidden="true"
  ></span>
  <!-- KI-39: this used to read "backend unreachable. Run `just api`" — a task runner and a repo
       recipe that a tester who installed an .exe does not have. The app's ONLY failure message
       asked for something that cannot exist on the machine showing it. Each line below is
       actionable by the person reading it, and none of them claims the wait is over: the gate
       keeps polling through every one of these states. -->
  {#if shell.status === 'ready' && shell.health}
    <span class="status-meta">
      {shell.health.chunk_count.toLocaleString()} chunks · {shell.health.model} · {shell.health.embedding_model}
    </span>
  {:else if shell.startupPhase === 'connecting'}
    <span class="status-meta">starting the engine…</span>
  {:else if shell.startupPhase === 'slow'}
    <!-- Kept short on purpose: this bar is nowrap + ellipsis, so a long sentence is simply cut. -->
    <span class="status-meta" title="A first launch unpacks a large model bundle before the engine can answer.">
      starting the engine — a first launch can take a minute…
    </span>
  {:else}
    <span class="status-meta err" title="The app keeps retrying on its own. If the engine never arrives, restart the app.">
      still starting — retrying. Restart the app if it never arrives.
    </span>
  {/if}

  <!-- Right-hand end of the same bar: a run in flight, when there is one. Renders nothing at all
       when idle, so the bar keeps its quiet resting state. -->
  <IngestChip />
</div>

<style>
  /* ---- bottom status bar (ambient) ---- */
  .statusbar {
    flex: none;
    /* The ingest chip's detail panel anchors to this bar (it opens upward, since the bar is the
       app's last row), so the bar has to be the positioned ancestor. */
    position: relative;
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
</style>
