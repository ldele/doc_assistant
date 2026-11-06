<script lang="ts">
  // The *Terms* section of Manage keywords (ADR-054): the words the library uses that nobody has
  // taken on, a tally, the toggle that shows the unused ones, and the term rows. Owns nothing —
  // the view splits and filters the list (the search box is in Concepts) and hands each row in as
  // `familyRow`. The toggle is bound: the view also opens it, for a new empty family or a jump to
  // an unused row.
  import type { Snippet } from 'svelte'
  import type { KeywordFamily } from '../../lib/core/types'
  import Icon from '../../shell/Icon.svelte'

  let {
    terms,
    famSplit,
    termsShown,
    famQuery,
    showInherited = $bindable(),
    familyRow,
  }: {
    terms: KeywordFamily[] // every term, for the count
    famSplit: { real: KeywordFamily[]; inherited: KeywordFamily[] } // unused = no members, no docs
    termsShown: KeywordFamily[] // the terms the toggle and the search let through
    famQuery: string // the search, typed in Concepts
    showInherited: boolean
    familyRow: Snippet<[KeywordFamily]>
  } = $props()
</script>

<section class="block">
  <h3>Terms ({terms.length})</h3>
  <p class="hint">
    Words your library uses that you have not taken on. A term stays searchable and still works
    as a keyword filter; no definition, merge or field is stored for it. Take one on with the
    <Icon name="waypoints" size={12} /> button on its row.
  </p>
  {#if terms.length === 0}
    <p class="hint">
      No terms yet — keywords arrive with your documents, or create a family above.
    </p>
  {:else}
    <div class="poolhead">
      <p class="hint tally">
        {famSplit.real.filter((f) => f.aliases.length > 0).length} collapse synonyms ·
        {famSplit.real.filter((f) => f.aliases.length === 0).length} single-label
        {#if famSplit.inherited.length > 0}· {famSplit.inherited.length} unused, hidden (no
          members, no documents){/if}
      </p>
      {#if famSplit.inherited.length > 0}
        <button class="linkbtn" onclick={() => (showInherited = !showInherited)} type="button">
          {showInherited ? 'Hide' : 'Show'} unused ({famSplit.inherited.length})
        </button>
      {/if}
    </div>
    <div class="famlist">
      {#each termsShown as f (f.id)}
        {@render familyRow(f)}
      {:else}
        {#if famQuery.trim() === ''}
          <p class="hint">Every term here is unused — show them with the toggle above.</p>
        {:else}
          <p class="hint">No term matches “{famQuery.trim()}”.</p>
        {/if}
      {/each}
    </div>
  {/if}
</section>

<style>
  .poolhead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
    margin-bottom: 0.4rem;
  }
  .linkbtn {
    flex: none;
    border: none;
    background: none;
    padding: 0;
    font: inherit;
    font-size: 0.75rem;
    color: var(--fg-2);
    cursor: pointer;
  }
  .linkbtn:hover {
    color: var(--fg);
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
  .hint.tally {
    margin: 0;
  }
</style>
