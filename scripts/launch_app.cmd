@echo off
rem Double-click launcher for the doc_assistant dev app — delegates to launch_app.ps1.
rem (Right-click > Send to > Desktop to make a shortcut.)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_app.ps1"
if errorlevel 1 pause
