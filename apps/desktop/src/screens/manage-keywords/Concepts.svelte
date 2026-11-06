<script lang="ts">
  // The *Concepts* section of Manage keywords (ADR-054, ADR-018): what a concept is and how its
  // forms count, the one search box of the two lists (it filters Terms below too), and the concept
  // rows. Owns nothing — the view filters the list and hands each row in as `familyRow`, the one
  // both lists render. Its `id="mk-concepts"` is where the Graph tab's door lands.
  import type { Snippet } from 'svelte'
  import type { KeywordFamily } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'

  let {
    concepts,
    conceptsShown,
    famQuery = $bindable(),
    familyRow,
  }: {
    concepts: KeywordFamily[] // every concept (the graph vocabulary), for the count
    conceptsShown: KeywordFamily[] // the concepts the search matches
    famQuery: string // the search of both lists
    familyRow: Snippet<[KeywordFamily]>
  } = $props()
</script>

<section class="block" id="mk-concepts">
  <h3>Concepts ({concepts.length})</h3>
  <!-- ADR-018's curation, in the view that ADR named as its home, under ADR-054's words. Said
       plainly because the default is *out* and the consequence is a blank Graph page: a user
       who never finds this control has no way to know why their graph is empty. -->
  <p class="hint">
    A concept is a meaning you have taken on: it is on the concept graph and can have a
    definition. Mark each of its forms <strong>exact</strong> when the form always means the
    concept, or <strong>broad</strong> when it also means other things — a broad form's
    documents are listed beside the concept and not counted in it. A form with neither mark
    counts as exact.
  </p>
  <p class="hint">
    The graph applies a change when you rebuild it, and says it is behind until you do.
  </p>
  <div class="poolhead">
    <div class="searchrow small">
      <Icon name="search" size={13} />
      <input
        bind:value={famQuery}
        placeholder="Search concepts and terms"
        aria-label="Search concepts and terms"
      />
    </div>
  </div>
  <div class="famlist">
    {#each conceptsShown as f (f.id)}
      {@render familyRow(f)}
    {:else}
      {#if famQuery.trim() === ''}
        <p class="hint">
          No concepts yet. Take a term on with the <Icon name="waypoints" size={12} /> button on
          its row, then rebuild the graph.
        </p>
      {:else}
        <p class="hint">No concept matches “{famQuery.trim()}”.</p>
      {/if}
    {/each}
  </div>
</section>

<style>
  .poolhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    margin-bottom: 0.4rem;
  }
  .searchrow.small {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex: 1;
    min-width: 0;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm, 6px);
    padding: 0.2rem 0.45rem;
    color: var(--fg-2);
  }
  .searchrow.small input {
    flex: 1;
    min-width: 0;
    border: none;
    background: none;
    color: var(--fg);
    font: inherit;
    font-size: 0.8rem;
    outline: none;
  }
  .block h3 {
    margin: 0 0 0.3rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--fg-2);
  }
  .hint {
    margin: 0 0 0.5rem;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .famlist {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
</style>
