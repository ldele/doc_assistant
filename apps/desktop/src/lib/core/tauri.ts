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

interface TauriGlobal {
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

/**
 * Open the OS file or folder picker. Resolves to the chosen paths, or `null` when the user
 * cancelled **or** when there is no picker (browser). Callers must not distinguish those two by
 * catching — check `canPickFiles()` first if the difference matters to the UI.
 */
export async function pickPaths(opts: { directory?: boolean } = {}): Promise<string[] | null> {
  const open = api()?.dialog?.open
  if (!open) return null
  try {
    const chosen = await open({
      multiple: true,
      directory: opts.directory ?? false,
      title: opts.directory ? 'Choose a folder of documents' : 'Choose documents',
    })
    if (chosen === null) return null
    return Array.isArray(chosen) ? chosen : [chosen]
  } catch {
    // A refused permission or a cancelled native dialog both land here. Neither is an app error.
    return null
  }
}
