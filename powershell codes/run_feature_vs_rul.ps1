# PowerShell script to run feature vs RUL analysis

param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    
    [string]$Feature = "charge_cycle_length",
    [string]$DataDir = "data",
    [string]$Output = "",
    [int]$Width = 12,
    [int]$Height = 8,
    [switch]$Verbose
)

# Set default output path if not provided
if ($Output -eq "") {
    $Output = "feature_vs_rul_${DatasetName}_${Feature}.png"
}

Write-Host "Running Feature vs RUL Analysis..." -ForegroundColor Green
Write-Host "Dataset: $DatasetName" -ForegroundColor Cyan
Write-Host "Feature: $Feature" -ForegroundColor Cyan
Write-Host "Output: $Output" -ForegroundColor Cyan

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
    "batteryml/chemistry_data_analysis/statistical_analysis/feature_vs_rul.py",
    $DatasetName,
    "--feature", $Feature,
    "--data_dir", $DataDir,
    "--output", $Output,
    "--figsize", $Width, $Height
)

if ($Verbose) {
    $pythonArgs += "--verbose"
}

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully created feature vs RUL plot!" -ForegroundColor Green
        Write-Host "Plot saved to: $Output" -ForegroundColor Cyan
    }
    else {
        Write-Host "Error creating plot" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error running Python script: $_" -ForegroundColor Red
    exit 1
}
