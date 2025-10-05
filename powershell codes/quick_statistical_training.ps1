# Quick Statistical Feature Training Script
# Simple script for quick testing and development

Write-Host "Quick Statistical Feature Training" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Default parameters for quick testing
$Dataset = "MATR"
$CycleLimit = 50
$NFeatures = 10
$DataDir = "data/preprocessed"
$OutputDir = "statistical_training_results"

Write-Host "Running with default parameters:" -ForegroundColor Cyan
Write-Host "  Dataset: $Dataset" -ForegroundColor White
Write-Host "  Cycle Limit: $CycleLimit" -ForegroundColor White
Write-Host "  Features: $NFeatures" -ForegroundColor White
Write-Host "  Data Dir: $DataDir" -ForegroundColor White
Write-Host "  Output Dir: $OutputDir" -ForegroundColor White
Write-Host ""

# Check if script exists
$ScriptPath = "batteryml/chemistry_data_analysis/statistical_analysis/statistical_feature_training_v2.py"
if (-not (Test-Path $ScriptPath)) {
    Write-Host "Error: Training script not found: $ScriptPath" -ForegroundColor Red
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Run the training
$Command = "python $ScriptPath $Dataset --cycle_limit $CycleLimit --n_features $NFeatures --data_dir `"$DataDir`" --output_dir `"$OutputDir`""

Write-Host "Executing: $Command" -ForegroundColor Cyan
Write-Host ""

try {
    Invoke-Expression $Command
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Training completed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "`n❌ Training failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
}
catch {
    Write-Host "`n❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nDone!" -ForegroundColor Green
