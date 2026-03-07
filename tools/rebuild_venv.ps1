param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Test-Python([string]$exe) {
    if (-not $exe) { return $false }
    if (-not (Test-Path $exe)) { return $false }
    try {
        $v = & $exe --version 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$candidates = @()
if ($PythonExe) {
    $candidates += $PythonExe
}
$candidates += "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$candidates += "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"

$selected = ""
foreach ($c in $candidates) {
    if (Test-Python $c) {
        $selected = $c
        break
    }
}

if (-not $selected) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $resolved = (& py -3.11 -c "import sys; print(sys.executable)").Trim()
            if (Test-Python $resolved) {
                $selected = $resolved
            }
        } catch {
        }
    }
}

if (-not $selected) {
    throw "No working Python 3.11 interpreter found. Install python.org CPython first, then rerun: powershell -ExecutionPolicy Bypass -File tools\rebuild_venv.ps1 -PythonExe <full-path-to-python.exe>"
}

Write-Host "[venv] using interpreter: $selected"

& $selected -m venv --clear .venv
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create .venv"
}

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    throw "Missing .venv python: $venvPy"
}

& $venvPy -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip/setuptools/wheel"
}

if (Test-Path (Join-Path $root "requirements.txt")) {
    & $venvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install requirements.txt"
    }
}

$pyVer = (& $venvPy --version).Trim()
$meta = @(
    "timestamp=" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
    "base_interpreter=$selected",
    "venv_python=$venvPy",
    "version=$pyVer"
)
Set-Content -Path (Join-Path $root ".venv\_bootstrap.txt") -Value $meta -Encoding utf8

Write-Host "[venv] ready: $venvPy"
Write-Host "[venv] python version: $pyVer"
Write-Host "[venv] marker: .venv\\_bootstrap.txt"
