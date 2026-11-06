<script lang="ts">
  // Settings > Provider & model > *Answering with*: the card on top of the tab, naming the model in
  // use and its provider, its status, where the text goes, who bills, and what to do when nothing
  // can answer yet. Draws `answeringWith()`'s reading of `/api/setup` (ADR-034: a verdict, not one
  // formed here); owns nothing.
  import type { AnsweringWith } from '../../lib/settings/setup'

  let { now }: { now: AnsweringWith } = $props()
</script>

<div class="card now">
  <div class="now-head">
    <div class="now-what">
      <span class="eyebrow">Answering with</span>
      {#if now.model}<strong class="model">{now.model} <span class="on">on {now.provider}</span></strong>{/if}
    </div>
    <span class="pill {now.tone}">{now.status}</span>
  </div>
  <p class="detail">{now.line}</p>
  {#if now.billing}<p class="detail">{now.billing}.</p>{/if}
  {#if now.action}<p class="detail fix">{now.action}</p>{/if}
</div>

<style>
  .card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.75rem;
    background: var(--surface);
  }
  .now-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .now-what {
    min-width: 0;
  }
  .eyebrow {
    display: block;
    font-size: 0.72rem;
    color: var(--fg-2);
  }
  .model {
    display: block;
    font-size: 0.95rem;
    overflow-wrap: anywhere;
  }
  .now .detail {
    margin: 0.3rem 0 0;
    overflow-wrap: anywhere;
  }
  .now .fix {
    color: var(--fg);
  }
  .pill {
    flex: none;
    margin-left: auto;
    font-size: 0.72rem;
    font-weight: 400;
    white-space: nowrap;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    border: 1px solid var(--border);
    color: var(--fg-2);
  }
  .pill.ok {
    color: var(--ok-fg);
    border-color: var(--ok-fg);
  }
  .pill.down {
    color: var(--danger);
    border-color: var(--danger);
  }
  .detail {
    margin: 0 0 0.35rem;
    font-size: 0.78rem;
    color: var(--fg-2);
  }
  .on {
    font-weight: 400;
    font-size: 0.8rem;
    color: var(--fg-2);
  }
</style>
