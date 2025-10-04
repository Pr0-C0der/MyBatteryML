# Requires: Python environment with batteryml installed and data_chemistries prepared

$ErrorActionPreference = "Stop"

# Root
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$module = "batteryml.chemistry_data_analysis.cycle_plotter_combined"

function Invoke-ChemistryCombined {
    param(
        [Parameter(Mandatory = $true)][string]$Chem,
        [string]$Hint,
        [int]$MAWindow = 0,
        [ValidateSet('none', 'ma', 'median')][string]$Smoothing = 'none',
        [double]$RemoveAfterPercentile = [double]::NaN
    )
    $dataPath = Join-Path $repoRoot (Join-Path "data_chemistries" $Chem)
    if (!(Test-Path $dataPath)) {
        Write-Warning "Data path not found: $dataPath (skipping)"
        return
    }
    $outPath = Join-Path $repoRoot (Join-Path "chemistry_cycle_plots_combined" $Chem)
    New-Item -ItemType Directory -Force -Path $outPath | Out-Null
    Write-Host "=== Combined plots for: $Chem (MA=$MAWindow, Smoothing=$Smoothing) ==="
    $percentArg = @()
    if ($RemoveAfterPercentile -ne [double]::NaN) { $percentArg = @("--remove_after_percentile", $RemoveAfterPercentile) }

    if ($Hint) {
        python -m $module --data_path "$dataPath" --output_dir "$outPath" --dataset_hint $Hint --ma_window $MAWindow --smoothing $Smoothing @percentArg --verbose | Write-Host
    }
    else {
        python -m $module --data_path "$dataPath" --output_dir "$outPath" --ma_window $MAWindow --smoothing $Smoothing @percentArg --verbose | Write-Host
    }
}

# Hardcoded runs (example: enable smoothing with MA=5)
Invoke-ChemistryCombined -Chem "lfp" -MAWindow 5 -Smoothing ma -RemoveAfterPercentile 90
Invoke-ChemistryCombined -Chem "lco" -Hint "CALCE" -MAWindow 5 -Smoothing ma -RemoveAfterPercentile 90
Invoke-ChemistryCombined -Chem "nca" -MAWindow 5 -Smoothing ma -RemoveAfterPercentile 90
Invoke-ChemistryCombined -Chem "nmc" -MAWindow 5 -Smoothing ma -RemoveAfterPercentile 90
Invoke-ChemistryCombined -Chem "mixed_nmc_lco" -Hint "HNEI" -MAWindow 5 -Smoothing ma -RemoveAfterPercentile 90

Write-Host "All chemistry combined plots completed."


