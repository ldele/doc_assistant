// App-shell state — the chrome that belongs to no single domain.
//
// A `.svelte.ts` rune module, and deliberately a **leaf**: it imports nothing from any sibling
// state module. That is what makes it safe for the pane components (step 5 phase 2) to import
// directly instead of receiving twenty-odd props each, and it is why the cross-domain readers can
// stay in App.svelte without an import cycle.
//
// What stays in App.svelte and must NOT move here:
//   · the nav-history `$effect` — it reads `shell.mode` *and* library collection/docId *and*
//     graph selection, so it is genuinely three domains wide;
//   · the readiness gate — it writes `health`/`status` here but also kicks conversations + folders;
//   · `selectMode` — it lazy-loads documents / families / folders / the graph, i.e. four domains;
//   · the chat-scope guard — reads folders, writes chat state.
// Those are orchestration. Moving them would trade coupling you can see for coupling you cannot.

import type { Health, SetupState } from '../core/types'
import type { StartupPhase } from './startup'

export type Mode = 'chat' | 'library' | 'graph'
export type ConnectionStatus = 'connecting' | 'ready' | 'down'

export const shell = $state({
  /** Which workspace the toolbar tabs select. Swapping it leaves chat state untouched. */
  mode: 'chat' as Mode,

  /** Mobile off-canvas drawer. Half the app closes it on selection, hence shell-owned and not a
   *  `prefs` value — it is transient navigation state, never persisted. */
  sidebarOpen: false,

  // Toolbar "more" surface + the two info modals it opens.
  appMenuOpen: false,
  showShortcuts: false,
  showAbout: false,

  /** Settings drawer. Reachable from the app menu and from its own gear (a fast path). */
  showSettings: false,

  // Global-search overlay (Cmd/Ctrl-K). App owns the query and derives the results via the pure,
  // tested `search.ts`; GlobalSearch just renders.
  searchOpen: false,
  searchQuery: '',

  /** Backend health, polled by the readiness gate in App.svelte. `null` until the first
   *  successful poll — the status bar renders "starting the engine…" until then. */
  health: null as Health | null,
  status: 'connecting' as ConnectionStatus,

  /** How long the backend has been silent, in words (KI-39). `'down'` above is now a *display*
   *  state, never a terminal one: the gate keeps polling through `'stalled'`, so a backend that
   *  arrives late still lands the app in `'ready'` without a relaunch. Timing + thresholds live in
   *  the pure, tested `startup.ts`. */
  startupPhase: 'connecting' as StartupPhase,

  /** First-run readiness (ADR-034), pulled once the backend answers and re-pulled after a
   *  provider/corpus change. `null` while unknown — the chat pane shows no setup banner until
   *  the answer is in, because guessing would flash "not set up" at a set-up install. */
  setup: null as SetupState | null,
})
