//! doc_assistant desktop shell. Hosts the Svelte webview; all logic stays in the Python
//! core behind the FastAPI/SSE boundary (PR-M2). PR-M4 freezes that backend (PyInstaller)
//! and ships it as a Tauri **sidecar** (`bundle.externalBin` in tauri.conf.json), spawned
//! here on startup. The frontend's readiness gate polls `/api/health` until it is warm.

use std::sync::Mutex;

use tauri::{Manager, RunEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// The API sidecar this launch started, ended when the app exits (KI-71). The shell plugin ends
/// only the children the window spawns, not one spawned here, so until 0.7.0 every closed window
/// left its backend holding port 8001 with that launch's token: the next launch's window read
/// through it and had every write refused (S-6).
struct Sidecar(Mutex<Option<CommandChild>>);

/// End the sidecar and everything it started. The frozen sidecar is a one-file launcher that
/// runs the real server as its own child, so ending the launcher alone can leave the server. On
/// Windows `taskkill /T` ends the tree, run from System32 by full path and without a console
/// window (checklist §6: no console window, ever). The backend also exits by itself once this
/// process is gone (`DOC_PARENT_PID`, `apps/api/parent_watch.py`), for a crash where this never runs.
fn end_sidecar(child: CommandChild) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let taskkill = std::path::Path::new(&system_root)
            .join("System32")
            .join("taskkill.exe");
        let ended = std::process::Command::new(taskkill)
            .args(["/PID", &child.pid().to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .creation_flags(CREATE_NO_WINDOW)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ended {
            return;
        }
    }
    let _ = child.kill();
}

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
    getrandom::fill(&mut bytes)
        .expect("the OS random source failed; refusing to start unprotected");
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
                    let (mut rx, child) = command
                        .env("DOC_API_TOKEN", &token)
                        // The backend exits when this process does (KI-71).
                        .env("DOC_PARENT_PID", std::process::id().to_string())
                        .spawn()
                        .expect("failed to spawn the API sidecar");
                    app.manage(Sidecar(Mutex::new(Some(child))));
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
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                // No Sidecar in dev (`just api` runs the backend); nothing to end then.
                if let Some(sidecar) = app.try_state::<Sidecar>() {
                    if let Some(child) = sidecar.0.lock().ok().and_then(|mut c| c.take()) {
                        end_sidecar(child);
                    }
                }
            }
        });
}
