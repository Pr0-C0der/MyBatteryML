# PowerShell script to run statistical feature training

param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    
    [int]$CycleLimit = 0,  # 0 means use all cycles
    [int]$NFeatures = 15,
    [double]$TestSize = 0.3,
    [string]$DataDir = "data/preprocessed",
    [string]$OutputDir = "statistical_training_results",
    [int]$RandomState = 42,
    [switch]$VerboseOutput
)

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
Write-Host "Test Size: $TestSize (70/30 split)" -ForegroundColor Cyan
Write-Host "Output Directory: $OutputDir" -ForegroundColor Cyan

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
    "--test_size", $TestSize,
    "--data_dir", $DataDir,
    "--output_dir", $OutputDir,
    "--random_state", $RandomState
)

if ($CycleLimit -gt 0) {
    $pythonArgs += "--cycle_limit", $CycleLimit
}

if ($VerboseOutput) {
    $pythonArgs += "--verbose"
}

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully completed statistical feature training!" -ForegroundColor Green
        Write-Host "Results saved to: $OutputDir" -ForegroundColor Cyan
        
        # List generated files
        if (Test-Path $OutputDir) {
            Write-Host "Generated files:" -ForegroundColor Yellow
            Get-ChildItem $OutputDir -File | ForEach-Object { 
                Write-Host "  - $($_.Name)" -ForegroundColor Yellow 
            }
        }
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
