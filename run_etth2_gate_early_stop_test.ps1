[CmdletBinding()]
param(
    [string]$Python = "E:\Anaconda\envs\T3Time\python.exe",
    [string]$DataDir = "",
    [string]$LogDir = "",
    [int]$GpuId = 0,
    [int]$Epochs = 100,
    [int]$OofEpochs = 100,
    [int]$GateEpochs = 80,
    [int]$GatePatience = 15,
    [int]$GateMinEpochs = 15,
    [int]$BatchSize = 32,
    [int]$Seed = 2024
)

$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
}
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path -Path $scriptRoot -ChildPath "dataset"
}
if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path -Path $scriptRoot -ChildPath "logs\etth2_gate_early_stop"
}

if ($GatePatience -lt 1 -or $GateMinEpochs -lt 1 -or $GateEpochs -lt 1) {
    throw "Gate epoch and patience values must be positive."
}
if ($GateMinEpochs -gt $GateEpochs) {
    throw "GateMinEpochs cannot be greater than GateEpochs."
}
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python interpreter not found: $Python"
}

$mainScript = Join-Path -Path $scriptRoot -ChildPath "ces_hmoe_carm_ettdataset.py"
if (-not (Test-Path -LiteralPath $mainScript -PathType Leaf)) {
    throw "CARM script not found: $mainScript"
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "Dataset directory not found: $DataDir"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$summaryPath = Join-Path $LogDir "summary.tsv"
$header = @(
    "mode", "pred_len", "seed", "best_epoch", "stop_epoch",
    "best_val_mse", "last_val_mse", "diagnosis", "test_mse",
    "test_base_mse", "trend_weight", "periodic_weight", "ramp_weight",
    "trend_dominant", "periodic_dominant", "ramp_dominant", "status"
) -join "`t"
Set-Content -LiteralPath $summaryPath -Value $header -Encoding UTF8

$predLens = @(24, 96, 192, 336, 720)
$modes = @("early_stop", "full_search")
$total = $predLens.Count * $modes.Count
$count = 0
$failCount = 0

Write-Host "ETTh2 gate early-stopping test: $total experiments"
Write-Host "dataset=ETTh2 seq_len=96 lr=0.00001 gate_lr=0.0001 pred_lens=$($predLens -join ',')"
Write-Host "early_stop: gate_epochs=$GateEpochs gate_patience=$GatePatience gate_min_epochs=$GateMinEpochs"
Write-Host "full_search: gate_epochs=$GateEpochs gate_patience=$GateEpochs gate_min_epochs=$GateEpochs"
Write-Host "summary=$summaryPath"

function Get-LastRegexValue {
    param(
        [string]$Path,
        [string]$Pattern,
        [int]$Group = 1
    )

    $matches = Select-String -LiteralPath $Path -Pattern $Pattern
    if ($null -eq $matches) {
        return "-"
    }
    $last = $matches | Select-Object -Last 1
    $match = [regex]::Match($last.Line, $Pattern)
    if (-not $match.Success) {
        return "-"
    }
    return $match.Groups[$Group].Value
}

function Add-SummaryRow {
    param(
        [string]$Mode,
        [int]$PredLen,
        [int]$RunSeed,
        [string]$LogPath,
        [string]$Status
    )

    $bestEpoch = Get-LastRegexValue $LogPath 'Gate selection result: best_epoch=([0-9]+)'
    $stopEpoch = Get-LastRegexValue $LogPath 'Gate select early stopping at epoch ([0-9]+)'
    $bestValMse = Get-LastRegexValue $LogPath 'Gate selection result: best_epoch=[0-9]+ best_val_mse=([^ ]+)'
    $lastValMse = Get-LastRegexValue $LogPath 'last_val_mse=([^ ]+) diagnosis='
    $diagnosis = Get-LastRegexValue $LogPath 'diagnosis=([^ ]+)'
    $testMse = Get-LastRegexValue $LogPath "Selected Test:.*'mse': ([0-9.eE+-]+)"
    $testBaseMse = Get-LastRegexValue $LogPath "Selected Test:.*'base_mse': ([0-9.eE+-]+)"
    $trendWeight = Get-LastRegexValue $LogPath 'Test expert usage:.*trend_weight=([0-9.eE+-]+)'
    $periodicWeight = Get-LastRegexValue $LogPath 'Test expert usage:.*periodic_weight=([0-9.eE+-]+)'
    $rampWeight = Get-LastRegexValue $LogPath 'Test expert usage:.*ramp_weight=([0-9.eE+-]+)'
    $trendDominant = Get-LastRegexValue $LogPath 'Test expert usage:.*trend_dominant=([0-9.eE+-]+)'
    $periodicDominant = Get-LastRegexValue $LogPath 'Test expert usage:.*periodic_dominant=([0-9.eE+-]+)'
    $rampDominant = Get-LastRegexValue $LogPath 'Test expert usage:.*ramp_dominant=([0-9.eE+-]+)'

    $row = @(
        $Mode, $PredLen, $RunSeed, $bestEpoch, $stopEpoch,
        $bestValMse, $lastValMse, $diagnosis, $testMse,
        $testBaseMse, $trendWeight, $periodicWeight, $rampWeight,
        $trendDominant, $periodicDominant, $rampDominant, $Status
    ) -join "`t"
    Add-Content -LiteralPath $summaryPath -Value $row -Encoding UTF8
}

foreach ($index in 0..($predLens.Count - 1)) {
    $predLen = $predLens[$index]
    # Keep the seed fixed across prediction lengths so result differences are
    # not mixed with a changing random initialization.
    $runSeed = $Seed

    foreach ($mode in $modes) {
        $count++
        if ($mode -eq "early_stop") {
            $runPatience = $GatePatience
            $runMinEpochs = $GateMinEpochs
        }
        else {
            # Full search runs all gate epochs but still reports the OOF5 best epoch.
            $runPatience = $GateEpochs
            $runMinEpochs = $GateEpochs
        }

        $logPath = Join-Path $LogDir ("ETTh2_pl{0}_{1}.log" -f $predLen, $mode)
        Write-Host "[$count/$total] ETTh2 | seq_len=96 | pred_len=$predLen | mode=$mode | seed=$runSeed"

        $argumentList = @(
            "-u", $mainScript,
            "--dataset", "ETTh2",
            "--data_dir", $DataDir,
            "--seq_len", "96",
            "--pred_len", "$predLen",
            "--epochs", "$Epochs",
            "--oof_epochs", "$OofEpochs",
            "--oof_folds", "5",
            "--gate_epochs", "$GateEpochs",
            "--gate_patience", "$runPatience",
            "--gate_min_epochs", "$runMinEpochs",
            "--batch_size", "$BatchSize",
            "--lr", "1e-5",
            "--gate_lr", "1e-4",
            "--patience", "15",
            "--min_epochs", "15",
            "--top_k", "8",
            "--key_points", "32",
            "--seed", "$runSeed",
            "--device", "auto",
            "--balance_weight", "0.01"
        )

        $previousCuda = $env:CUDA_VISIBLE_DEVICES
        $env:CUDA_VISIBLE_DEVICES = "$GpuId"
        try {
            & $Python @argumentList 2>&1 | Tee-Object -FilePath $logPath
            $exitCode = $LASTEXITCODE
        }
        finally {
            if ($null -eq $previousCuda) {
                Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
            }
            else {
                $env:CUDA_VISIBLE_DEVICES = $previousCuda
            }
        }

        if ($exitCode -eq 0) {
            Add-SummaryRow $mode $predLen $runSeed $logPath "ok"
            Write-Host "  completed | log=$logPath"
        }
        else {
            $failCount++
            Add-SummaryRow $mode $predLen $runSeed $logPath ("failed:{0}" -f $exitCode)
            Write-Warning "  FAILED (exit=$exitCode) | log=$logPath"
        }
    }
}

Write-Host ""
Write-Host "Summary:"
Get-Content -LiteralPath $summaryPath | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "Compare early_stop and full_search for each pred_len in $summaryPath."
Write-Host "Early stopping is useful when it stops before GateEpochs without worsening test_mse."
Write-Host ("Finished: {0}/{1} succeeded; {2} failed" -f ($total - $failCount), $total, $failCount)

if ($failCount -gt 0) {
    exit 1
}
