param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPython = Join-Path $scriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw "Backend Python not found: $backendPython"
}

$backendRoot = $scriptRoot.Replace("'", "''")
$backendPythonEscaped = $backendPython.Replace("'", "''")

$backendCommand = @"
Set-Location -LiteralPath '$backendRoot'
& '$backendPythonEscaped' -m uvicorn src.web.app:app --reload --host 0.0.0.0 --port 8000
"@.Trim()

Write-Host 'Starting backend UI: http://127.0.0.1:8000' -ForegroundColor Cyan

if ($DryRun) {
    Write-Host ''
    Write-Host '[DryRun] Backend command:'
    Write-Host $backendCommand
    return
}

$backendProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoExit',
    '-Command',
    $backendCommand
) -PassThru

Write-Host "Backend PID: $($backendProcess.Id)" -ForegroundColor Green
Write-Host 'If a new window does not appear, check your PowerShell execution policy or terminal settings.' -ForegroundColor Yellow
