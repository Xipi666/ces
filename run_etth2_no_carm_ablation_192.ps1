[CmdletBinding()]
param(
    [string]$Python = "E:\Anaconda\envs\T3Time\python.exe",
    [string]$DataDir = "",
    [string]$LogDir = "",
    [int]$GpuId = 0,
    [int]$Epochs = 100,
    [int]$BatchSize = 32,
    [int]$Stage1Epochs = 30,
    [int]$Stage2Epochs = 30,
    [int]$MinEpochs = 15,
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
    $LogDir = Join-Path $scriptRoot "logs\etth2_no_carm_ablation_192"
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
$predLen = 192
$modes = @("A", "B")
$total = $modes.Count * $Seeds.Count
$count = 0
$failCount = 0

$header = "mode`tstage2_lr`thorizon_balance_weight`tpred_len`tseed`tbest_epoch`tbest_val_mse`ttest_mse`ttest_mae`ttrend_weight`tperiodic_weight`tramp_weight`ttrend_dominant`tperiodic_dominant`tramp_dominant`tstatus"
foreach ($mode in $modes) {
    Set-Content -LiteralPath (Join-Path $LogDir ("{0}.tsv" -f $mode)) -Encoding UTF8 -Value $header
}

function Get-LastRegexValue {
    param([string]$Path, [string]$Pattern, [int]$Group = 1)
    $matches = Select-String -LiteralPath $Path -Pattern $Pattern
    if ($null -eq $matches) { return "-" }
    $match = [regex]::Match(($matches | Select-Object -Last 1).Line, $Pattern)
    if (-not $match.Success) { return "-" }
    return $match.Groups[$Group].Value
}

function Format-ThreeDecimals {
    param([string]$Value)
    if ($Value -eq "-") { return "-" }
    return ([math]::Round([double]$Value, 3)).ToString("0.000")
}

Write-Host "ETTh2 no-CARM ablation: pred_len=$predLen, experiments=$total"
Write-Host "A: stage2_lr=1e-5, horizon_balance_weight=0"
Write-Host "B: stage2_lr=3e-5, horizon_balance_weight=0.001"
Write-Host "Seeds: $($Seeds -join ', ')"

foreach ($mode in $modes) {
    if ($mode -eq "A") {
        $stage2Lr = "1e-5"
        $horizonBalanceWeight = "0"
    }
    else {
        $stage2Lr = "3e-5"
        $horizonBalanceWeight = "0.001"
    }

    $summaryPath = Join-Path $LogDir ("{0}.tsv" -f $mode)
    foreach ($runSeed in $Seeds) {
        $count++
        $logPath = Join-Path $LogDir ("ETTh2_ablation_{0}_seed{1}_pl{2}.log" -f $mode, $runSeed, $predLen)
        Write-Host "[$count/$total] mode=$mode | pred_len=$predLen | seed=$runSeed | stage2_lr=$stage2Lr | horizon_balance_weight=$horizonBalanceWeight"

        $argumentList = @(
            "-u", $mainScript,
            "--dataset", "ETTh2",
            "--data_dir", $DataDir,
            "--seq_len", "96",
            "--pred_len", "$predLen",
            "--epochs", "$Epochs",
            "--stage1_epochs", "$Stage1Epochs",
            "--stage2_epochs", "$Stage2Epochs",
            "--stage2_patience", "5",
            "--stage3_patience", "10",
            "--batch_size", "$BatchSize",
            "--lr", "3e-5",
            "--stage2_lr", $stage2Lr,
            "--finetune_lr", "1e-5",
            "--patience", "15",
            "--min_epochs", "$MinEpochs",
            "--seed", "$runSeed",
            "--device", "auto",
            "--balance_weight", "0.01",
            "--horizon_balance_weight", $horizonBalanceWeight
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
            $bestEpoch = Get-LastRegexValue $logPath 'Best Val MSE: [^ ]+ at epoch ([0-9]+)'
            $bestValMse = Format-ThreeDecimals (Get-LastRegexValue $logPath 'Best Val MSE: ([0-9.eE+-]+) at epoch')
            $testMse = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'mse': ([0-9.eE+-]+)")
            $testMae = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'mae': ([0-9.eE+-]+)")
            $trendWeight = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'trend_weight': ([0-9.eE+-]+)")
            $periodicWeight = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'periodic_weight': ([0-9.eE+-]+)")
            $rampWeight = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'ramp_weight': ([0-9.eE+-]+)")
            $trendDominant = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'trend_dominant': ([0-9.eE+-]+)")
            $periodicDominant = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'periodic_dominant': ([0-9.eE+-]+)")
            $rampDominant = Format-ThreeDecimals (Get-LastRegexValue $logPath "Test:.*'ramp_dominant': ([0-9.eE+-]+)")
            $row = @(
                $mode, $stage2Lr, $horizonBalanceWeight, $predLen, $runSeed,
                $bestEpoch, $bestValMse, $testMse, $testMae,
                $trendWeight, $periodicWeight, $rampWeight,
                $trendDominant, $periodicDominant, $rampDominant, "ok"
            ) -join "`t"
            Add-Content -LiteralPath $summaryPath -Encoding UTF8 -Value $row
        }
        else {
            $failCount++
            $row = @(
                $mode, $stage2Lr, $horizonBalanceWeight, $predLen, $runSeed,
                "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
                "failed:$exitCode"
            ) -join "`t"
            Add-Content -LiteralPath $summaryPath -Encoding UTF8 -Value $row
            Write-Warning "FAILED (exit=$exitCode) | log=$logPath"
        }
    }
}

Write-Host ""
Write-Host "A summary:"
Get-Content -LiteralPath (Join-Path $LogDir "A.tsv") | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "B summary:"
Get-Content -LiteralPath (Join-Path $LogDir "B.tsv") | ForEach-Object { Write-Host $_ }
Write-Host ("Finished: {0}/{1} succeeded; {2} failed" -f ($total - $failCount), $total, $failCount)

if ($failCount -gt 0) { exit 1 }
