# Requires: Python environment with batteryml installed and data_chemistries prepared

$ErrorActionPreference = "Stop"

# Root directory
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# Module (chemistry-aware statistical features)
$statModule = "batteryml.chemistry_data_analysis.stat_features"

# Helper to run a chemistry with optional dataset hint
function Invoke-ChemistryStats {
    param(
        [Parameter(Mandatory = $true)][string]$Chem,
        [string]$Hint
    )
    $dataPath = Join-Path $repoRoot (Join-Path "data_chemistries" $Chem)
    if (!(Test-Path $dataPath)) {
        Write-Warning "Data path not found: $dataPath (skipping)"
        return
    }
    $outPath = Join-Path $repoRoot (Join-Path "chemistry_stat_features" $Chem)
    New-Item -ItemType Directory -Force -Path $outPath | Out-Null
    Write-Host "=== Processing chemistry (stats): $Chem ==="
    if ($Hint) {
        python -m $statModule --data_path "$dataPath" --output_dir "$outPath" --dataset_hint $Hint --verbose | Write-Host
    }
    else {
        python -m $statModule --data_path "$dataPath" --output_dir "$outPath" --verbose | Write-Host
    }
}

# Hardcoded runs per chemistry
Invoke-ChemistryStats -Chem "lfp"
Invoke-ChemistryStats -Chem "lco" -Hint "CALCE"
Invoke-ChemistryStats -Chem "nca"
Invoke-ChemistryStats -Chem "nmc"
Invoke-ChemistryStats -Chem "mixed_nmc_lco" -Hint "HNEI"

Write-Host "All chemistry stats plots completed."


