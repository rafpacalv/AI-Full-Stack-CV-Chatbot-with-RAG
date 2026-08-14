<#
    Start the CV Screener: FastAPI backend + Streamlit UI.

    .\run.ps1            # start both
    .\run.ps1 -ApiOnly   # backend only
#>
param(
    [switch]$ApiOnly,
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "No virtualenv found at venv\. Create it with:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\python.exe -m pip install -r requirements.txt -e ."
    exit 1
}

# Ollama must be up: everything (chat, embeddings, extraction) runs on it.
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 3 | Out-Null
    Write-Host "Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "Ollama is not reachable on http://localhost:11434" -ForegroundColor Red
    Write-Host "Start it with 'ollama serve', then run this script again."
    exit 1
}

if (-not (Test-Path (Join-Path $root "data\index\embeddings.npy"))) {
    Write-Host "No search index found. Build it with:" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\python.exe -m cvscreener.ingest.index"
    exit 1
}

Write-Host "Starting API on http://127.0.0.1:$ApiPort ..." -ForegroundColor Cyan
$api = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "cvscreener.api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
    -WorkingDirectory $root -PassThru -NoNewWindow

Start-Sleep -Seconds 4

if ($ApiOnly) {
    Write-Host "API running (PID $($api.Id)). Docs: http://127.0.0.1:$ApiPort/docs" -ForegroundColor Green
    Wait-Process -Id $api.Id
    exit 0
}

Write-Host "Starting Streamlit on http://localhost:$UiPort ..." -ForegroundColor Cyan
try {
    & $python -m streamlit run (Join-Path $root "ui\app.py") --server.port $UiPort
} finally {
    if (-not $api.HasExited) {
        Write-Host "`nStopping API (PID $($api.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
}
