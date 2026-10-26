# Wrapper for Windows Task Scheduler. Usage: powershell -File scripts\run_ingestion.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "$root\.venv\Scripts\python.exe" -m src.scheduled_ingest
exit $LASTEXITCODE
