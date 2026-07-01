<script lang="ts">
  import type { ClaimView, Decision } from './types'
  import { adjudicate } from './api'

  let { claims }: { claims: ClaimView[] } = $props()

  type Status = 'pending' | 'accepted' | 'rejected' | 'edited' | 'error'
  interface St {
    status: Status
    editing: boolean
    draft: string
    error?: string
  }
  const DEFAULT: St = { status: 'pending', editing: false, draft: '' }

  // Per-claim adjudication state, keyed by claim_id. Mutated only in handlers (never in
  // render), so reads use a defaulting getter.
  let states = $state<Record<string, St>>({})
  const st = (id: string): St => states[id] ?? DEFAULT

  async function decide(c: ClaimView, decision: Decision, edited = ''): Promise<void> {
    try {
      await adjudicate(c.claim_id, decision, edited || undefined)
      states[c.claim_id] = { ...st(c.claim_id), status: decision, editing: false }
    } catch (e) {
      states[c.claim_id] = { ...st(c.claim_id), status: 'error', error: String(e) }
    }
  }

  function startEdit(c: ClaimView): void {
    states[c.claim_id] = { status: 'pending', editing: true, draft: c.text }
  }
  function cancelEdit(c: ClaimView): void {
    states[c.claim_id] = { ...st(c.claim_id), editing: false }
  }

  function doneLabel(s: Status): string {
    if (s === 'accepted') return '✓ accepted'
    if (s === 'rejected') return '✗ rejected'
    return '✎ edited'
  }
</script>

{#if claims.length}
  <section class="claims">
    <h3>⚠ {claims.length} claim(s) to review <small>(evidence vs interpretation)</small></h3>
    {#each claims as c (c.claim_id)}
      {@const s = st(c.claim_id)}
      <div class="claim" class:resolved={s.status !== 'pending' && s.status !== 'error'}>
        <div class="claim-text">
          <span class="badge {c.badge === 'unsupported' ? 'bad' : 'weak'}">{c.badge}</span>
          <span>#{c.n} {c.text}</span>
        </div>
        {#if s.editing}
          <div class="edit">
            <textarea bind:value={s.draft} rows="2"></textarea>
            <div class="actions">
              <button class="ok" onclick={() => decide(c, 'edited', s.draft)}>Save</button>
              <button onclick={() => cancelEdit(c)}>Cancel</button>
            </div>
          </div>
        {:else if s.status === 'pending'}
          <div class="actions">
            <button class="ok" onclick={() => decide(c, 'accepted')}>✓ Accept</button>
            <button class="no" onclick={() => decide(c, 'rejected')}>✗ Reject</button>
            <button onclick={() => startEdit(c)}>✎ Edit</button>
          </div>
        {:else if s.status === 'error'}
          <span class="err">⚠ couldn’t record: {s.error}</span>
        {:else}
          <span class="done">{doneLabel(s.status)}</span>
        {/if}
      </div>
    {/each}
  </section>
{/if}

<style>
  .claims {
    margin-top: 0.8rem;
    border-top: 1px solid var(--border);
    padding-top: 0.6rem;
  }
  h3 {
    margin: 0 0 0.5rem;
    font-size: 0.95rem;
  }
  h3 small {
    color: var(--fg-2);
    font-weight: 400;
  }
  .claim {
    padding: 0.5rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: var(--surface);
  }
  .claim.resolved {
    opacity: 0.65;
  }
  .claim-text {
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
    font-size: 0.88rem;
  }
  .badge {
    flex: none;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    padding: 0.05rem 0.4rem;
    border-radius: 4px;
  }
  .badge.bad {
    background: var(--warn-bg);
    color: var(--warn-fg);
  }
  .badge.weak {
    background: var(--surface-2);
    color: var(--fg-2);
  }
  .actions {
    display: flex;
    gap: 0.4rem;
    margin-top: 0.4rem;
  }
  textarea {
    width: 100%;
    font: inherit;
    padding: 0.3rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
  }
  button {
    font: inherit;
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--fg);
    cursor: pointer;
  }
  button.ok {
    border-color: var(--ok-border);
    color: var(--ok-fg);
  }
  button.no {
    border-color: var(--warn-border);
    color: var(--warn-fg);
  }
  .done {
    font-size: 0.82rem;
    color: var(--fg-2);
  }
  .err {
    font-size: 0.82rem;
    color: var(--warn-fg);
  }
</style>
