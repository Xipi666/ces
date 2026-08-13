[CmdletBinding()]
param(
    [string]$Python = "E:\Anaconda\envs\T3Time\python.exe",
    [string]$DataDir = "",
    [string]$LogDir = "",
    [int]$GpuId = 0,
    [int]$Epochs = 100,
    [int]$BatchSize = 32,
    [int[]]$Seeds = @(2024, 2025, 2026)
)

$ErrorActionPreference = "Stop"
$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
}
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $scriptRoot "dataset"
}
if ([string]::IsNullOrWhiteSpace($LogDir)) {
$LogDir = Join-Path $scriptRoot "logs\etth2_no_carm"
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python interpreter not found: $Python"
}
$mainScript = Join-Path $scriptRoot "ces_hmoe_ettdataset.py"
if (-not (Test-Path -LiteralPath $mainScript -PathType Leaf)) {
    throw "Non-CARM script not found: $mainScript"
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "Dataset directory not found: $DataDir"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$predLens = @(24, 96, 192, 336, 720)
$total = $Seeds.Count * $predLens.Count
$count = 0
$failCount = 0

function Get-LastRegexValue {
    param([string]$Path, [string]$Pattern, [int]$Group = 1)
    $matches = Select-String -LiteralPath $Path -Pattern $Pattern
    if ($null -eq $matches) { return "-" }
    $match = [regex]::Match(($matches | Select-Object -Last 1).Line, $Pattern)
    if (-not $match.Success) { return "-" }
    return $match.Groups[$Group].Value
}

foreach ($runSeed in $Seeds) {
    $summaryPath = Join-Path $LogDir ("{0}.tsv" -f $runSeed)
    Set-Content -LiteralPath $summaryPath -Encoding UTF8 -Value @(
        "pred_len`tseed`tbest_epoch`tbest_val_mse`ttest_mse`ttest_mae`ttrend_weight`tperiodic_weight`tramp_weight`ttrend_dominant`tperiodic_dominant`tramp_dominant`tstatus"
    )

    foreach ($predLen in $predLens) {
    $count++
    $logPath = Join-Path $LogDir ("ETTh2_seed{0}_pl{1}_no_carm.log" -f $runSeed, $predLen)
    Write-Host "[$count/$total] ETTh2 | seq_len=96 | pred_len=$predLen | mode=BASE | seed=$runSeed"

    $argumentList = @(
        "-u", $mainScript,
        "--dataset", "ETTh2",
        "--data_dir", $DataDir,
        "--seq_len", "96",
        "--pred_len", "$predLen",
        "--epochs", "$Epochs",
        "--batch_size", "$BatchSize",
        "--lr", "3e-5",
        "--stage2_lr", "1e-5",
        "--finetune_lr", "1e-5",
        "--patience", "15",
        "--min_epochs", "15",
        "--seed", "$runSeed",
        "--device", "auto",
        "--balance_weight", "0.01",
        "--horizon_balance_weight", "0.001"
    )

    $previousCuda = $env:CUDA_VISIBLE_DEVICES
    $env:CUDA_VISIBLE_DEVICES = "$GpuId"
    try {
        & $Python @argumentList 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        if ($null -eq $previousCuda) { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
        else { $env:CUDA_VISIBLE_DEVICES = $previousCuda }
    }

    if ($exitCode -eq 0) {
        $bestEpoch = Get-LastRegexValue $logPath 'Best Val MSE: [^ ]+ at epoch ([0-9]+)'
        $bestValMse = Get-LastRegexValue $logPath 'Best Val MSE: ([0-9.eE+-]+) at epoch'
        $testMse = Get-LastRegexValue $logPath "Test:.*'mse': ([0-9.eE+-]+)"
        $testMae = Get-LastRegexValue $logPath "Test:.*'mae': ([0-9.eE+-]+)"
        $trendWeight = Get-LastRegexValue $logPath "Test:.*'trend_weight': ([0-9.eE+-]+)"
        $periodicWeight = Get-LastRegexValue $logPath "Test:.*'periodic_weight': ([0-9.eE+-]+)"
        $rampWeight = Get-LastRegexValue $logPath "Test:.*'ramp_weight': ([0-9.eE+-]+)"
        $trendDominant = Get-LastRegexValue $logPath "Test:.*'trend_dominant': ([0-9.eE+-]+)"
        $periodicDominant = Get-LastRegexValue $logPath "Test:.*'periodic_dominant': ([0-9.eE+-]+)"
        $rampDominant = Get-LastRegexValue $logPath "Test:.*'ramp_dominant': ([0-9.eE+-]+)"
        Add-Content -LiteralPath $summaryPath -Encoding UTF8 -Value (
            "$predLen`t$runSeed`t$bestEpoch`t$([math]::Round([double]$bestValMse, 3).ToString('0.000'))`t$([math]::Round([double]$testMse, 3).ToString('0.000'))`t$([math]::Round([double]$testMae, 3).ToString('0.000'))`t$([math]::Round([double]$trendWeight, 3).ToString('0.000'))`t$([math]::Round([double]$periodicWeight, 3).ToString('0.000'))`t$([math]::Round([double]$rampWeight, 3).ToString('0.000'))`t$([math]::Round([double]$trendDominant, 3).ToString('0.000'))`t$([math]::Round([double]$periodicDominant, 3).ToString('0.000'))`t$([math]::Round([double]$rampDominant, 3).ToString('0.000'))`tok"
        )
        Write-Host "  completed | log=$logPath"
    }
    else {
        $failCount++
        Add-Content -LiteralPath $summaryPath -Encoding UTF8 -Value "$predLen`t$runSeed`t-`t-`t-`t-`t-`t-`t-`t-`t-`t-`tfailed:$exitCode"
        Write-Warning "  FAILED (exit=$exitCode) | log=$logPath"
    }
    }
}

Write-Host ""
Write-Host "Summary:"
Get-Content -LiteralPath $summaryPath | ForEach-Object { Write-Host $_ }
Write-Host ("Finished: {0}/{1} succeeded; {2} failed" -f ($total - $failCount), $total, $failCount)
if ($failCount -gt 0) { exit 1 }
