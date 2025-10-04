# Requires: Python environment with batteryml installed and data_chemistries prepared

$ErrorActionPreference = "Stop"

# Root directories
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# Tools (chemistry-aware)
$cyclePlotModule = "batteryml.chemistry_data_analysis.cycle_plotter_mod"
$corrModule = "batteryml.chemistry_data_analysis.correlation_mod"

# Helper to run a chemistry with optional dataset hint
function Invoke-Chemistry {
    param(
        [Parameter(Mandatory = $true)][string]$Chem,
        [string]$Hint
    )
    $dataPath = Join-Path $repoRoot (Join-Path "data_chemistries" $Chem)
    if (!(Test-Path $dataPath)) {
        Write-Warning "Data path not found: $dataPath (skipping)"
        return
    }
    $cycleOut = Join-Path $repoRoot (Join-Path "chemistry_cycle_plots" $Chem)
    $corrOut = Join-Path $repoRoot (Join-Path "chemistry_correlation_analysis" $Chem)
    New-Item -ItemType Directory -Force -Path $cycleOut | Out-Null
    New-Item -ItemType Directory -Force -Path $corrOut | Out-Null
    Write-Host "=== Processing chemistry: $Chem ==="
    if ($Hint) {
        python -m $cyclePlotModule --data_path "$dataPath" --output_dir "$cycleOut" --cycle_gap 500 --dataset_hint $Hint --verbose | Write-Host
        python -m $corrModule --data_path "$dataPath" --output_dir "$corrOut" --dataset_hint $Hint --verbose | Write-Host
    }
    else {
        python -m $cyclePlotModule --data_path "$dataPath" --output_dir "$cycleOut" --cycle_gap 500 --verbose | Write-Host
        python -m $corrModule --data_path "$dataPath" --output_dir "$corrOut" --verbose | Write-Host
    }
}

# Hardcoded runs per chemistry
Invoke-Chemistry -Chem "lfp"
Invoke-Chemistry -Chem "lco" -Hint "CALCE"
Invoke-Chemistry -Chem "nca"
Invoke-Chemistry -Chem "nmc"
Invoke-Chemistry -Chem "mixed_nmc_lco" -Hint "HNEI"

Write-Host "All chemistry plots completed."


