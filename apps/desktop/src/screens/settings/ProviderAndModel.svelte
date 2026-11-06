<script lang="ts">
  // Settings > Provider & model: the provider setup (`ProviderSetup.svelte`: which model answers,
  // the providers, the open one's form) and *Answer epistemics*, the saved default for the
  // contested / superseded chips on an answer's sources. Owns that default's request; Settings
  // holds its in-flight flag and error line, which it keeps while another section shows.
  import type { Settings } from '../../lib/core/types'
  import { setMarkersEnabled } from '../../lib/core/api'
  import ProviderSetup from './ProviderSetup.svelte'

  let {
    settings = $bindable(),
    markersBusy = $bindable(),
    markersError = $bindable(),
    onProviderChanged,
  }: {
    settings: Settings
    markersBusy: boolean
    markersError: string | null
    /** A provider or model was switched in there: re-read what depends on it. */
    onProviderChanged: () => void
  } = $props()

  // ADR-027 D2 (E3) — the persisted answer-layer epistemics default. Distinct from the RAG
  // sandbox's session-only override (Retrieval): this one survives a restart and becomes the
  // sandbox's baseline. On failure the checkbox snaps back to the server state (no lying UI).
  async function applyMarkersDefault(enabled: boolean): Promise<void> {
    if (markersBusy) return
    markersBusy = true
    markersError = null
    try {
      settings = await setMarkersEnabled(enabled)
    } catch (e) {
      markersError = String(e)
    } finally {
      markersBusy = false
    }
  }
</script>

<ProviderSetup {onProviderChanged} />

<section>
  <h3>Answer epistemics <span class="muted">(experimental)</span></h3>
  <p class="hint">
    Whether corpus epistemics (contested / superseded chips) may appear on an answer's
    sources. Saved as your default; the per-source evaluation strip below answers is
    always shown either way.
  </p>
  <!-- Off by default, and said plainly rather than left as a silent default
       (REVIEW 2026-08-12 §2b R3 · KI-33 · ADR-041). A user turning this on deserves to know
       what they are turning on; a user leaving it off deserves to know they are not missing
       a measurement. -->
  <p class="hint">
    <strong>Known limitation:</strong> these chips come from a stance pass that judges a topic
    <em>without reading the document</em>, has no “neutral” verdict, and can change its answer
    when the same pair appears in a different order. They are a prompt to go and look, not a
    finding. Off by default until that is rebuilt on evidence.
  </p>
  <label class="switch-row">
    <input
      type="checkbox"
      checked={settings.epistemics_markers_enabled}
      disabled={markersBusy}
      onchange={(e) => void applyMarkersDefault((e.target as HTMLInputElement).checked)}
    />
    Epistemics chips in answers <span class="muted">(saved default)</span>
  </label>
  <div aria-live="polite">
    {#if markersError}
      <p class="err" role="alert">{markersError}</p>
    {/if}
  </div>
</section>

<style>
  section {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--border);
  }
  section:last-child {
    border-bottom: none;
  }
  h3 {
    margin: 0 0 0.6rem;
    font-size: 0.95rem;
  }
  label {
    display: block;
    font-size: 0.8rem;
    color: var(--fg-2);
    margin-bottom: 0.3rem;
  }
  input {
    width: 100%;
    font: inherit;
    font-size: 0.85rem;
    padding: 0.45rem 0.55rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--fg);
  }
  input:disabled {
    opacity: 0.6;
  }
  .hint {
    font-size: 0.76rem;
    color: var(--fg-2);
    margin: 0.4rem 0 0;
  }
  .muted {
    color: var(--fg-2);
    font-weight: 400;
    font-size: 0.82rem;
  }
  .err {
    color: var(--warn-fg);
    font-size: 0.82rem;
    margin: 0.5rem 0 0;
  }
  section label {
    margin-top: 0.7rem;
  }
  section label:first-of-type {
    margin-top: 0;
  }
  .switch-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--fg);
    cursor: pointer;
  }
  .switch-row input {
    width: auto;
    accent-color: var(--accent);
  }
</style>
