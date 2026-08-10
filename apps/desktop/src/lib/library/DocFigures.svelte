<script lang="ts">
  // Per-document figure panel (Library L1b). Figures are addressed *separately from the text
  // chunks*: a figure is a different kind of object — an image with a caption — so it gets its
  // own panel rather than being interleaved into the chunk browser.
  //
  // The panel's job is not "show the images", it is to show **which figures the assistant can
  // actually see**. A figure enters retrieval only once it has a description, so each card
  // states that plainly and, when it is absent, why. Read-only and advisory: a load failure
  // degrades to one quiet line and never blocks the document view.
  import type { LibraryDocumentFigures } from '../core/types'
  import { getDocumentFigures } from '../core/api'
  import { API_BASE } from '../core/api/_base'

  let { docId }: { docId: string } = $props()

  let data = $state<LibraryDocumentFigures | null>(null)
  let error = $state<string | null>(null)

  // Last-write-wins token, mirroring DocConnections' own load guard: clicking through
  // documents quickly must not let a slow earlier response overwrite a newer one.
  let token = 0
  $effect(() => {
    const id = docId
    data = null
    error = null
    if (!id) return
    const mine = ++token
    void (async () => {
      try {
        const d = await getDocumentFigures(id)
        if (mine === token) data = d
      } catch (e) {
        if (mine === token) error = String(e)
      }
    })()
  })

  function imageUrl(figureId: string): string {
    return `${API_BASE}/api/figures/${encodeURIComponent(figureId)}`
  }
</script>

<section class="figures">
  <h3>
    Figures
    {#if data}
      <span class="count">
        {data.total}
        {#if data.total > 0}
          · {data.retrievable_count} searchable
        {/if}
      </span>
    {/if}
  </h3>

  {#if error}
    <p class="quiet">Figures unavailable.</p>
  {:else if data === null}
    <p class="quiet">Loading…</p>
  {:else if data.total === 0}
    <!-- An honest empty state: no figures detected is an ordinary outcome, not a failure. -->
    <p class="quiet">No figures detected in this document.</p>
  {:else}
    {#if data.missing_image_count > 0}
      <!-- Inform, don't block: the rows are still listed below. -->
      <p class="warn">
        {data.missing_image_count} figure{data.missing_image_count === 1 ? '' : 's'} lost their
        rendered image — re-run the figure extraction pass.
      </p>
    {/if}
    <ul class="grid">
      {#each data.figures as fig (fig.id)}
        <li class="card" class:dim={!fig.retrievable}>
          {#if fig.has_image}
            <img src={imageUrl(fig.id)} alt={fig.caption ?? `Figure on page ${fig.page}`} loading="lazy" />
          {:else}
            <div class="noimage">no image</div>
          {/if}
          <div class="meta">
            <span class="page">p{fig.page}</span>
            {#if fig.kind}<span class="kind">{fig.kind}</span>{/if}
            {#if fig.retrievable}
              <span class="badge ok" title="This figure is searchable — the assistant can find and cite it">searchable</span>
            {:else}
              <span class="badge off" title={fig.not_retrievable_reason ?? ''}>not searchable</span>
            {/if}
          </div>
          {#if fig.caption}<p class="caption">{fig.caption}</p>{/if}
          {#if !fig.retrievable && fig.not_retrievable_reason}
            <p class="reason">{fig.not_retrievable_reason}</p>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .figures {
    margin-top: 1.25rem;
  }
  h3 {
    font-size: 0.9rem;
    margin: 0 0 0.5rem;
  }
  .count {
    font-weight: 400;
    opacity: 0.65;
    font-size: 0.8rem;
  }
  .quiet {
    opacity: 0.6;
    font-size: 0.85rem;
  }
  .warn {
    font-size: 0.8rem;
    opacity: 0.85;
  }
  .grid {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.75rem;
  }
  .card {
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 6px;
    padding: 0.5rem;
    overflow: hidden;
  }
  /* Not-searchable figures stay visible but read as secondary — the point is that the
     difference is legible at a glance, not that they are hidden. */
  .card.dim {
    opacity: 0.72;
  }
  img {
    width: 100%;
    height: auto;
    max-height: 180px;
    object-fit: contain;
    background: #fff;
    border-radius: 3px;
  }
  .noimage {
    height: 72px;
    display: grid;
    place-items: center;
    font-size: 0.75rem;
    opacity: 0.5;
    border: 1px dashed var(--border, #3a3a3a);
    border-radius: 3px;
  }
  .meta {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.4rem;
    font-size: 0.72rem;
  }
  .page,
  .kind {
    opacity: 0.65;
  }
  .badge {
    margin-left: auto;
    padding: 0.05rem 0.35rem;
    border-radius: 3px;
    font-size: 0.68rem;
    white-space: nowrap;
  }
  .badge.ok {
    background: color-mix(in srgb, green 25%, transparent);
  }
  .badge.off {
    background: color-mix(in srgb, gray 25%, transparent);
  }
  .caption {
    font-size: 0.75rem;
    margin: 0.35rem 0 0;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .reason {
    font-size: 0.7rem;
    opacity: 0.62;
    margin: 0.3rem 0 0;
  }
</style>
