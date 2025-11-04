param(
    [Parameter(Mandatory = $true)]
    [string]$Timerange,
    [string]$Timeframe = "5m",
    [double]$Stake = 100
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ComposeCommand {
    if (Get-Command 'docker' -ErrorAction SilentlyContinue) {
        try {
            docker compose version | Out-Null
            return @('docker', 'compose')
        } catch {
            # fall back to docker-compose check
        }
    }
    if (Get-Command 'docker-compose' -ErrorAction SilentlyContinue) {
        return @('docker-compose')
    }
    throw 'docker compose command not found. Install Docker and retry.'
}

$composeParts = Get-ComposeCommand
$exe = $composeParts[0]
$composeArgs = @()
if ($composeParts.Count -gt 1) {
    $composeArgs += $composeParts[1..($composeParts.Count - 1)]
}

$arguments = @(
    'run','--rm','freqtrade','backtesting',
    '-c','user_data/configs/v29_backtest.json',
    '-s','TaxBrainV29','-i', $Timeframe,
    '--timerange', $Timerange,
    '--stake-amount', $Stake
)

Write-Host "Running backtest: $exe $($composeArgs + $arguments -join ' ')"
& $exe @($composeArgs + $arguments)

$resultsDir = Join-Path 'user_data' 'backtest_results'
$targetDir = Join-Path 'user_data' 'runs/last'
if (!(Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir | Out-Null
}

$files = Get-ChildItem -Path $resultsDir -Filter '*.json' -ErrorAction SilentlyContinue
if (!$files) {
    Write-Warning "No backtest result files found in $resultsDir"
} else {
    foreach ($file in $files) {
        Copy-Item $file.FullName -Destination $targetDir -Force
    }
    Write-Host "Copied $($files.Count) result file(s) to $targetDir"
}
