//! doc_assistant desktop shell. Hosts the Svelte webview; all logic stays in the Python
//! core behind the FastAPI/SSE boundary (PR-M2). In dev the backend runs separately
//! (`just api`); PR-M4 bundles it as a PyInstaller sidecar and spawns it on startup via
//! `tauri-plugin-shell`.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
