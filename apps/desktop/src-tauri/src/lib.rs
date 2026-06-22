//! doc_assistant desktop shell. Hosts the Svelte webview; all logic stays in the Python
//! core behind the FastAPI/SSE boundary (PR-M2). PR-M4 freezes that backend (PyInstaller)
//! and ships it as a Tauri **sidecar** (`bundle.externalBin` in tauri.conf.json), spawned
//! here on startup. The frontend's readiness gate polls `/api/health` until it is warm.

use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Spawn the bundled FastAPI sidecar. In dev there is no frozen binary (run the
            // backend separately with `just api`), so a missing sidecar is non-fatal.
            match app.shell().sidecar("doc-assistant-api") {
                Ok(command) => {
                    let (mut rx, _child) = command.spawn().expect("failed to spawn the API sidecar");
                    // Drain the sidecar's stderr to the host console for debugging.
                    tauri::async_runtime::spawn(async move {
                        while let Some(event) = rx.recv().await {
                            if let CommandEvent::Stderr(line) = event {
                                eprintln!("[sidecar] {}", String::from_utf8_lossy(&line));
                            }
                        }
                    });
                }
                Err(e) => eprintln!("[sidecar] not bundled (dev mode — use `just api`): {e}"),
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
