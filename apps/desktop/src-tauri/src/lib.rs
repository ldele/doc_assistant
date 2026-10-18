//! doc_assistant desktop shell. Hosts the Svelte webview; all logic stays in the Python
//! core behind the FastAPI/SSE boundary (PR-M2). PR-M4 freezes that backend (PyInstaller)
//! and ships it as a Tauri **sidecar** (`bundle.externalBin` in tauri.conf.json), spawned
//! here on startup. The frontend's readiness gate polls `/api/health` until it is warm.

use tauri::Manager;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

/// Security S-6 (docs/security.md S1b): the token this launch's API requires on every request
/// that can change something. The sidecar reads it from `DOC_API_TOKEN`; the window asks for it
/// with the `api_token` command and sends it in `X-Doc-Api-Token`.
struct LaunchToken(String);

#[tauri::command]
fn api_token(token: tauri::State<'_, LaunchToken>) -> String {
    token.0.clone()
}

/// A fresh random token per launch (32 bytes from the OS, hex). A `DOC_API_TOKEN` already in the
/// environment is used instead: the release check (`scripts/rg012/`) launches the installed app
/// and drives its API from outside the window, so it has to know the token. Whoever can set this
/// process's environment already runs as the user.
fn launch_token() -> String {
    if let Ok(preset) = std::env::var("DOC_API_TOKEN") {
        if !preset.trim().is_empty() {
            return preset.trim().to_string();
        }
    }
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("the OS random source failed; refusing to start unprotected");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        // AD1: file/folder picker for Add documents. Reachable from JS only as
        // `window.__TAURI__.dialog` (withGlobalTauri), and only `open` is permitted.
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![api_token])
        .setup(|app| {
            let token = launch_token();
            // Spawn the bundled FastAPI sidecar. In dev there is no frozen binary (run the
            // backend separately with `just api`, which checks no token unless one is set), so a
            // missing sidecar is non-fatal.
            match app.shell().sidecar("doc-assistant-api") {
                Ok(command) => {
                    let (mut rx, _child) = command
                        .env("DOC_API_TOKEN", &token)
                        .spawn()
                        .expect("failed to spawn the API sidecar");
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
            app.manage(LaunchToken(token));
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
