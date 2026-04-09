$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @(
    (Join-Path $projectRoot "venv\Scripts\python.exe"),
    (Join-Path $projectRoot ".venv\Scripts\python.exe")
)

$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $pythonExe) {
    Write-Error "Could not find a project virtual environment. Expected venv\\Scripts\\python.exe or .venv\\Scripts\\python.exe."
}

& $pythonExe -m streamlit run (Join-Path $projectRoot "app.py")
