# PowerShell script to run statistical feature training

param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    
    [int]$NFeatures = 15,
    [int]$CycleLimit = 100,
    [string]$DataDir = "data/processed",
    [string]$Output = "",
    [int]$Width = 12,
    [int]$Height = 8,
    [switch]$VerboseOutput
)

# Set default output path if not provided
if ([string]::IsNullOrEmpty($Output)) {
    $Output = "statistical_training_results"
}

# Set default cycle limit message
if ($CycleLimit -eq 0) {
    $CycleMessage = "All cycles"
}
else {
    $CycleMessage = "First $CycleLimit cycles"
}

Write-Host "Running Statistical Feature Training for RUL Prediction..." -ForegroundColor Green
Write-Host "Dataset: $DatasetName" -ForegroundColor Cyan
Write-Host "Cycle Limit: $CycleMessage" -ForegroundColor Cyan
Write-Host "Top Features: $NFeatures" -ForegroundColor Cyan
Write-Host "Figure Size: ${Width}x${Height}" -ForegroundColor Cyan
Write-Host "Output Directory: $Output" -ForegroundColor Cyan

# Check if data directory exists
if (-not (Test-Path $DataDir)) {
    Write-Host "Error: Data directory not found at $DataDir" -ForegroundColor Red
    exit 1
}

# Check if dataset directory exists
$datasetPath = Join-Path $DataDir $DatasetName
if (-not (Test-Path $datasetPath)) {
    Write-Host "Error: Dataset directory not found at $datasetPath" -ForegroundColor Red
    Write-Host "Available datasets:" -ForegroundColor Yellow
    Get-ChildItem $DataDir -Directory | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Yellow }
    exit 1
}

# Run the Python script
$pythonArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/statistical_feature_training.py",
    $DatasetName,
    "--n_features", $NFeatures,
    "--cycle_limit", $CycleLimit,
    "--data_dir", $DataDir,
    "--output", $Output,
    "--figsize", $Width, $Height
)

if ($VerboseOutput) {
    $pythonArgs += "--verbose"
}

Write-Host "`nExecuting command:" -ForegroundColor Yellow
Write-Host "python $($pythonArgs -join ' ')" -ForegroundColor Gray

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSuccessfully completed statistical feature training!" -ForegroundColor Green
        Write-Host "Results saved to: $Output" -ForegroundColor Cyan
        
        # List generated files in current directory (plots are saved there)
        Write-Host "`nGenerated files:" -ForegroundColor Yellow
        $plotFiles = @(
            "feature_importance_$DatasetName.png",
            "predictions_$DatasetName.png"
        )
        
        foreach ($file in $plotFiles) {
            if (Test-Path $file) {
                Write-Host "  - $file" -ForegroundColor Yellow
            }
        }
        
        Write-Host "`nTraining Summary:" -ForegroundColor Green
        Write-Host "  - Trained on log(RUL) for optimal performance" -ForegroundColor White
        Write-Host "  - Evaluated on actual RUL for interpretability" -ForegroundColor White
        Write-Host "  - Used battery-level aggregated features" -ForegroundColor White
        Write-Host "  - Selected top $NFeatures features with highest correlation" -ForegroundColor White
    }
    else {
        Write-Host "Error during training" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error running Python script: $_" -ForegroundColor Red
    exit 1
}