; Provenote's installer hooks, wired as tauri.conf.json bundle.windows.nsis.installerHooks so the
; template next door stays Tauri's own.
;
; KI-80: 0.6.0 shipped its engine as one file at the install root and left it running after its
; window closed (KI-71). Installed over that, this version finds port 8001 taken and stays on
; "still starting", and the old 1.5 GB file stays for good: no later installer or uninstaller
; lists it. Force-ending that engine leaves its 2.4 GB unpacked copy in TEMP, so that goes too.
;
; POSTINSTALL, not PREINSTALL: CheckIfAppIsRunning has closed any window by then, so no engine
; that an open window depends on is ended.
!macro NSIS_HOOK_POSTINSTALL
  Push $0
  Push $1

  nsExec::Exec '"$SYSDIR\taskkill.exe" /F /T /IM doc-assistant-api.exe'
  Pop $0 ; 128 when none is running, the usual case
  Sleep 1000

  Delete "$INSTDIR\doc-assistant-api.exe"

  ClearErrors
  FindFirst $0 $1 "$TEMP\_MEI*"
  ${DoWhile} $1 != ""
    ; Any PyInstaller program unpacks to a folder named like this: only ours holds this model.
    ${If} ${FileExists} "$TEMP\$1\hf_cache\hub\models--BAAI--bge-reranker-base\*.*"
      RMDir /r "$TEMP\$1"
    ${EndIf}
    FindNext $0 $1
  ${Loop}
  FindClose $0

  Pop $1
  Pop $0
!macroend
