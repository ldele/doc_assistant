<script lang="ts">
  // "Where does this passage sit?" (ROADMAP 19) — the other half of a citation.
  //
  // A citation says *which* document. This says where in it: the passage shown in place, with the
  // text that surrounds it, and a position the card cannot otherwise give. It reads the offsets
  // ingest recorded (`char_start`/`char_end`), so nothing is re-derived per query.
  //
  // **It expands in place rather than opening a modal.** The reader is mid-answer, checking a
  // citation against its source; a dialog would take the answer off screen, which is the one thing
  // someone verifying a claim must not lose sight of.
  //
  // The source it shows is the *extracted markdown*, not the PDF — that is what the app holds, and
  // the panel says so rather than implying a page image. The document viewer is ROADMAP 18.
  import type { ChunkContext } from '../core/types'
  import { getChunkContext } from '../core/api'
  import Icon from '../shell/Icon.svelte'
  import { elision, tidy, whereLabel } from './chunkcontext'

  let { chunkKey }: { chunkKey: string } = $props()

  let open = $state(false)
  let loading = $state(false)
  let loaded = $state(false)
  let ctx = $state<ChunkContext | null>(null)
  let failed = $state<string | null>(null)

  const marks = $derived(elision(ctx))

  // Scroll the window to the passage, not to its surroundings.
  //
  // Without this the panel opens at the top of the `before` text and the cited passage is below
  // the fold — so the one thing the reader opened this to see is the one thing not on screen.
  //
  // **Deliberately not `scrollIntoView`.** That walks every scrollable ancestor: it dragged the
  // chat transcript and the source panel with it and left this box at `scrollTop` 21,792 on a
  // 225px window, with the highlight 11,000px out of view. Setting `scrollTop` from the two
  // rects touches this element and nothing else. A third of the way down, so the lines *before*
  // the passage stay visible — showing them is the entire point of a context view.
  let windowEl = $state<HTMLElement | null>(null)
  $effect(() => {
    if (!open || ctx === null || !windowEl) return
    const mark = windowEl.querySelector('mark')
    if (!mark) return
    const box = windowEl.getBoundingClientRect()
    const target = mark.getBoundingClientRect()
    windowEl.scrollTop = Math.max(
      0,
      windowEl.scrollTop + (target.top - box.top) - windowEl.clientHeight / 3,
    )
  })

  async function toggle(): Promise<void> {
    open = !open
    // Fetch once, on first open — a citation panel can hold ten of these and most are never
    // expanded. `loaded` rather than `ctx !== null` because "we looked and it cannot be placed"
    // is a real answer worth caching too.
    if (!open || loaded || loading) return
    loading = true
    try {
      ctx = await getChunkContext(chunkKey)
      loaded = true
    } catch (e) {
      failed = e instanceof Error ? e.message : String(e)
    } finally {
      loading = false
    }
  }
</script>

<button class="toggle" onclick={toggle} aria-expanded={open} type="button">
  <Icon name={open ? 'chevron-down' : 'chevron-right'} size={12} />
  In context
</button>

{#if open}
  <div class="ctx">
    {#if loading}
      <p class="note">Locating…</p>
    {:else if failed}
      <p class="note err" role="alert">{failed}</p>
    {:else if ctx === null}
      <!-- Honest, and rare: 3 chunks in 39,000 on the reference corpus. The span was never
           resolvable at ingest, so there is no position to show and none is invented. -->
      <p class="note">
        This passage can’t be placed in the source text — its position wasn’t resolvable when the
        document was indexed.
      </p>
    {:else}
      <p class="where">
        <Icon name="file-text" size={12} />
        {whereLabel(ctx)}
        <span class="muted">· in the extracted text of {ctx.filename}</span>
      </p>
      <div class="window" bind:this={windowEl}>
        {#if marks.before}<span class="elide">…</span>{/if}<span class="around"
          >{tidy(ctx.before)}</span
        ><mark>{tidy(ctx.text)}</mark><span class="around">{tidy(ctx.after)}</span>{#if marks.after}<span
            class="elide">…</span
          >{/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .toggle {
    font: inherit;
    font-size: 0.72rem;
    color: var(--fg-2);
    background: none;
    border: none;
    padding: 0.15rem 0;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
  }
  .toggle:hover {
    color: var(--fg);
  }
  .ctx {
    margin-top: 0.4rem;
    border-top: 1px solid var(--border);
    padding-top: 0.4rem;
  }
  .where {
    margin: 0 0 0.35rem;
    font-size: 0.72rem;
    color: var(--fg-2);
    display: flex;
    align-items: center;
    gap: 0.3rem;
    flex-wrap: wrap;
  }
  .window {
    font-size: 0.76rem;
    line-height: 1.5;
    max-height: 15rem;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 0.6rem;
  }
  /* The surroundings are context, not the answer — they recede so the cited passage reads first. */
  .around {
    color: var(--fg-2);
  }
  .elide {
    color: var(--fg-2);
  }
  mark {
    background: var(--warn-bg, color-mix(in srgb, var(--accent) 22%, transparent));
    color: inherit;
    border-radius: 3px;
    padding: 0.05rem 0.1rem;
  }
  .note {
    margin: 0;
    font-size: 0.74rem;
    color: var(--fg-2);
  }
  .note.err {
    color: var(--danger, #c0392b);
  }
  .muted {
    color: var(--fg-2);
  }
</style>
