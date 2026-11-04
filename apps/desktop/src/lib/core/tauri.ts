// The Tauri boundary (AD1/W0). The ONLY module allowed to touch `window.__TAURI__`.
//
// Why a global and not an npm package: `withGlobalTauri: true` in `tauri.conf.json` injects Tauri's
// own JS bundle at `window.__TAURI__`, and the dialog plugin injects itself into the same object.
// That keeps the frontend a deliberate 1-dep artifact (`marked`). Evidence and the two runtime
// assertions live in `docs/specs/feature-add-documents.md` W0.
//
// **`window.__TAURI__` exists only inside the Tauri window.** The dev/test loop for this app runs
// in a plain browser (Vite on 1420/5731), where it is `undefined`. Every export here is therefore
// written to *degrade*, never to throw: `isTauri()` is false, subscribing is a no-op returning a
// no-op unsubscribe, and the picker resolves to `null`. Callers branch on `isTauri()` to explain
// themselves in the UI; they never need a try/catch.

/** Payload of `tauri://drag-drop`, per `tauri/src/manager/window.rs` (`paths` + `position`). */
interface DragDropPayload {
  paths?: string[]
  position?: { x: number; y: number }
}

interface TauriEventApi {
  listen: <T>(event: string, handler: (e: { payload: T }) => void) => Promise<() => void>
}

interface TauriDialogApi {
  open: (opts: {
    multiple?: boolean
    directory?: boolean
    title?: string
  }) => Promise<string | string[] | null>
}

interface TauriCoreApi {
  invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>
}

interface TauriGlobal {
  core?: TauriCoreApi
  event?: TauriEventApi
  dialog?: TauriDialogApi
}

function api(): TauriGlobal | null {
  const g = (globalThis as { __TAURI__?: TauriGlobal }).__TAURI__
  return g ?? null
}

/** True only inside the Tauri window. Drives what the UI offers rather than what it risks. */
export function isTauri(): boolean {
  return api() !== null
}

/** True when the picker is actually reachable — the plugin is registered *and* permitted. */
export function canPickFiles(): boolean {
  return typeof api()?.dialog?.open === 'function'
}

/** True when drag-drop can be subscribed to. Separate from `canPickFiles`: one can exist without
 *  the other (the event ships with core, the dialog is a plugin that must be registered). */
export function canReceiveDrops(): boolean {
  return typeof api()?.event?.listen === 'function'
}

/**
 * Subscribe to files dropped on the window. Returns an unsubscribe function — **always**, so the
 * caller's cleanup is unconditional even in a browser where nothing was subscribed.
 *
 * Tauri intercepts the OS drag before the DOM sees it (`dragDropEnabled` defaults true, and its
 * own docs say disabling it is *required* for HTML5 drag-and-drop on Windows), so this event is
 * the only route to real paths. HTML5 `drop` handlers would give `File` objects with no path.
 */
export function onFilesDropped(handler: (paths: string[]) => void): () => void {
  const listen = api()?.event?.listen
  if (!listen) return () => {}

  let stop: (() => void) | null = null
  let cancelled = false

  void listen<DragDropPayload>('tauri://drag-drop', (e) => {
    const paths = e.payload?.paths
    // A drag that carries no paths (text, a URL, an image from a browser) fires the same event.
    // Ignoring it is the honest read: nothing addable arrived.
    if (paths && paths.length > 0) handler(paths)
  }).then(
    (unlisten) => {
      if (cancelled) unlisten()
      else stop = unlisten
    },
    () => {
      /* subscription refused (capability missing) — the UI already degrades via canReceiveDrops */
    },
  )

  return () => {
    cancelled = true
    stop?.()
    stop = null
  }
}

/** Subscribe to drag-enter / drag-leave so the drop target can show itself. Same degradation. */
export function onDragHover(handler: (over: boolean) => void): () => void {
  const listen = api()?.event?.listen
  if (!listen) return () => {}

  const stops: Array<() => void> = []
  let cancelled = false
  const sub = (event: string, over: boolean) =>
    void listen(event, () => handler(over)).then(
      (unlisten) => (cancelled ? unlisten() : stops.push(unlisten)),
      () => {},
    )

  sub('tauri://drag-enter', true)
  sub('tauri://drag-leave', false)
  // A completed drop ends the hover as well; the drop handler does the real work.
  sub('tauri://drag-drop', false)

  return () => {
    cancelled = true
    for (const s of stops) s()
    stops.length = 0
  }
}

let tokenOnce: Promise<string | null> | null = null

/**
 * The launch token the shell minted for this run (security S-6), or `null` in a browser. Asked
 * once; a refused command also resolves to `null`, and the API's 401 then names the problem.
 */
export function launchToken(): Promise<string | null> {
  if (tokenOnce) return tokenOnce
  const invoke = api()?.core?.invoke
  tokenOnce = invoke
    ? invoke<string>('api_token').then(
        (t) => (typeof t === 'string' && t ? t : null),
        () => null,
      )
    : Promise.resolve(null)
  return tokenOnce
}

/**
 * Open the OS file or folder picker. Resolves to the chosen paths, or `null` when the user
 * cancelled **or** when there is no picker (browser). Callers must not distinguish those two by
 * catching — check `canPickFiles()` first if the difference matters to the UI.
 */
export async function pickPaths(
  opts: { directory?: boolean; multiple?: boolean; title?: string } = {},
): Promise<string[] | null> {
  const open = api()?.dialog?.open
  if (!open) return null
  try {
    const chosen = await open({
      // Defaults to true so every pre-CS1 caller is unchanged. Settings passes false: it is
      // choosing *the* library folder, and a picker that lets you select three of them while the
      // caller silently keeps the first is a dialog that lied about what it was asking.
      multiple: opts.multiple ?? true,
      directory: opts.directory ?? false,
      title:
        opts.title ?? (opts.directory ? 'Choose a folder of documents' : 'Choose documents'),
    })
    if (chosen === null) return null
    return Array.isArray(chosen) ? chosen : [chosen]
  } catch {
    // A refused permission or a cancelled native dialog both land here. Neither is an app error.
    return null
  }
}

/** The library the backend opens (ADR-061), as the shell remembers it. */
export interface LibraryState {
  /** A backend the shell can restart. False in development, where the control is not offered. */
  bundled: boolean
  /** The open library folder, or null for the app's own. */
  data_home: string | null
  /** The library open before the last switch, or null; where *Go back* returns. */
  previous: string | null
}

/** The shell's library state, or null in a browser (no shell, nothing to switch). */
export async function libraryState(): Promise<LibraryState | null> {
  const invoke = api()?.core?.invoke
  if (!invoke) return null
  try {
    return await invoke<LibraryState>('library_state')
  } catch {
    return null
  }
}

/**
 * Open a library folder, or the app's own with null: the shell remembers it and restarts the
 * backend on it. Resolves once the new backend has been started, not once it answers; the caller
 * reloads the window and the start-up gate waits for it. Rejects with the shell's sentence.
 */
export async function openLibrary(dataHome: string | null): Promise<void> {
  const invoke = api()?.core?.invoke
  if (!invoke) throw new Error('Libraries can be switched in the installed app only.')
  await invoke<void>('open_library', { dataHome })
}

const SWITCH_KEY = 'provenote-library-switch'

/**
 * Remember, across the reload a switch needs, which library was just asked for, so the start-up
 * gate can offer *Go back* if it never answers (ADR-061 choice 7). Session storage: the marker
 * belongs to this window's switch and dies with it. Storage that refuses is not an error.
 */
export function markLibrarySwitch(dataHome: string | null): void {
  try {
    sessionStorage.setItem(SWITCH_KEY, dataHome ?? '')
  } catch {
    // No storage: a stalled switch then shows the ordinary start-up message.
  }
}

/** The library a switch just asked for ('' for the app's own), or null when none is pending. */
export function pendingLibrarySwitch(): string | null {
  try {
    return sessionStorage.getItem(SWITCH_KEY)
  } catch {
    return null
  }
}

export function clearLibrarySwitch(): void {
  try {
    sessionStorage.removeItem(SWITCH_KEY)
  } catch {
    // Nothing to clear.
  }
}
