//! doc_assistant desktop shell. Hosts the Svelte webview; all logic stays in the Python
//! core behind the FastAPI/SSE boundary (PR-M2). PR-M4 freezes that backend (PyInstaller);
//! since KI-73 it ships as a **folder** (`bundle.resources` in tauri.conf.json, unpacked once
//! by the installer) and is spawned here on startup by path. The frontend's readiness gate
//! polls `/api/health` until it is warm.

use std::path::PathBuf;
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

/// The frozen backend's executable inside the bundle: `<resources>/backend/doc-assistant-api`
/// (`scripts/build_sidecar.py` puts the one-dir build there). `None` in dev, where there is no
/// frozen build and the backend runs separately (`just api`).
fn bundled_backend(app: &tauri::AppHandle) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "doc-assistant-api.exe"
    } else {
        "doc-assistant-api"
    };
    let exe = app.path().resource_dir().ok()?.join("backend").join(name);
    exe.is_file().then_some(exe)
}

/// Which library folder the backend opens (ADR-061 choice 2), kept in the shell's own config
/// folder as `library.json`. `data_home: None` is the app's own library, which the backend finds
/// by itself; a folder is handed to it as `DOC_DATA_DIR`, the variable it reads first
/// (`config.py`). `previous` is where *Go back* returns after a switch that failed (choice 7).
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
struct LibraryChoice {
    data_home: Option<String>,
    previous: Option<String>,
}

fn choice_file(app: &tauri::AppHandle) -> Option<PathBuf> {
    Some(app.path().app_config_dir().ok()?.join("library.json"))
}

/// The remembered choice. A missing or unreadable file is the app's own library: a broken
/// setting must never stop the app from starting.
fn read_choice(app: &tauri::AppHandle) -> LibraryChoice {
    choice_file(app)
        .and_then(|f| std::fs::read_to_string(f).ok())
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn write_choice(app: &tauri::AppHandle, choice: &LibraryChoice) -> Result<(), String> {
    let file = choice_file(app).ok_or("no config folder for this app")?;
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(choice).map_err(|e| e.to_string())?;
    std::fs::write(file, text).map_err(|e| e.to_string())
}

/// Start the backend on the chosen library and keep its handle, so exit and a switch can end it.
fn spawn_backend(app: &tauri::AppHandle, exe: PathBuf, token: &str) -> Result<(), String> {
    let mut command = app
        .shell()
        .command(exe)
        .env("DOC_API_TOKEN", token)
        // The backend exits when this process does (KI-71).
        .env("DOC_PARENT_PID", std::process::id().to_string());
    if let Some(home) = read_choice(app).data_home {
        command = command.env("DOC_DATA_DIR", home);
    }
    let (mut rx, child) = command.spawn().map_err(|e| e.to_string())?;
    match app.try_state::<Sidecar>() {
        Some(sidecar) => {
            if let Ok(mut slot) = sidecar.0.lock() {
                *slot = Some(child);
            }
        }
        None => {
            app.manage(Sidecar(Mutex::new(Some(child))));
        }
    }
    // Drain the sidecar's stderr to the host console for debugging.
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let CommandEvent::Stderr(line) = event {
                eprintln!("[sidecar] {}", String::from_utf8_lossy(&line));
            }
        }
    });
    Ok(())
}

/// Wait until nothing listens on the backend's port, so the next backend can bind it. A switch
/// ends the old backend first; Windows can hold the port a moment after the process is gone.
fn wait_for_port(port: u16, timeout: std::time::Duration) {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// The backend's port (`apps/api/__main__.py`), fixed in the frontend and the CSP too.
const API_PORT: u16 = 8001;

/// What the window needs to show the library control: whether there is a backend to restart
/// (not in dev, ADR-061 choice 9), the library open now (`None`: the app's own), and the one
/// before it.
#[derive(serde::Serialize)]
struct LibraryState {
    bundled: bool,
    data_home: Option<String>,
    previous: Option<String>,
}

#[tauri::command]
fn library_state(app: tauri::AppHandle) -> LibraryState {
    let choice = read_choice(&app);
    LibraryState {
        bundled: bundled_backend(&app).is_some(),
        data_home: choice.data_home,
        previous: choice.previous,
    }
}

/// Open a library folder, or the app's own with `None`: remember it, then restart the backend
/// on it. The folder was checked by the backend first (`/api/libraries/inspect`); this only
/// refuses a folder with no `library.db`, so a stray call cannot point the app at nothing.
#[tauri::command]
async fn open_library(
    app: tauri::AppHandle,
    token: tauri::State<'_, LaunchToken>,
    data_home: Option<String>,
) -> Result<(), String> {
    let exe =
        bundled_backend(&app).ok_or("there is no bundled backend to restart (development)")?;
    if let Some(home) = &data_home {
        if !std::path::Path::new(home).join("library.db").is_file() {
            return Err(format!("{home} is not a library: it holds no library.db"));
        }
    }
    let current = read_choice(&app).data_home;
    if current == data_home {
        return Ok(());
    }
    write_choice(
        &app,
        &LibraryChoice {
            data_home,
            previous: current,
        },
    )?;
    if let Some(sidecar) = app.try_state::<Sidecar>() {
        let old = sidecar.0.lock().ok().and_then(|mut c| c.take());
        if let Some(child) = old {
            end_sidecar(child);
        }
    }
    wait_for_port(API_PORT, std::time::Duration::from_secs(10));
    spawn_backend(&app, exe, &token.0)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        // AD1: file/folder picker for Add documents. Reachable from JS only as
        // `window.__TAURI__.dialog` (withGlobalTauri), and only `open` is permitted.
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            api_token,
            library_state,
            open_library
        ])
        .setup(|app| {
            let token = launch_token();
            let handle = app.handle().clone();
            // Spawn the bundled FastAPI backend, on the library the person chose (ADR-061). In
            // dev there is no frozen build (run the backend separately with `just api`, which
            // checks no token unless one is set), so a missing one is non-fatal. The shell plugin
            // starts it without a console window.
            match bundled_backend(&handle) {
                Some(exe) => {
                    spawn_backend(&handle, exe, &token).expect("failed to spawn the API sidecar")
                }
                None => eprintln!("[sidecar] not bundled (dev mode — use `just api`)"),
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
