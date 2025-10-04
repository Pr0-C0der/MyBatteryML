# PowerShell script to run feature-RUL correlation analysis

param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    
    [string]$Feature = "discharge_capacity",
    [int]$Cycle = 100,
    [string]$DataDir = "data",
    [string]$Output = "",
    [int]$Width = 12,
    [int]$Height = 8,
    [string[]]$Measures = @("mean", "variance", "median", "kurtosis", "skewness", "min", "max"),
    [int[]]$MultiCycle = @(),
    [string[]]$AllDatasets = @(),
    [switch]$Verbose
)

# Set default output path if not provided
if ($Output -eq "") {
    if ($MultiCycle.Count -gt 0) {
        $Output = "correlation_multi_cycle_${Feature}_${DatasetName}.png"
    }
    elseif ($AllDatasets.Count -gt 0) {
        $Output = "feature_rul_correlation_${Feature}_cycle_${Cycle}"
    }
    else {
        $Output = "correlation_${Feature}_cycle_${Cycle}_${DatasetName}.png"
    }
}

Write-Host "Running Feature-RUL Correlation Analysis..." -ForegroundColor Green
Write-Host "Dataset: $DatasetName" -ForegroundColor Cyan
Write-Host "Feature: $Feature" -ForegroundColor Cyan
Write-Host "Cycle: $Cycle" -ForegroundColor Cyan
Write-Host "Output: $Output" -ForegroundColor Cyan

if ($MultiCycle.Count -gt 0) {
    Write-Host "Mode: Multi-cycle analysis (cycles: $($MultiCycle -join ', '))" -ForegroundColor Yellow
}
elseif ($AllDatasets.Count -gt 0) {
    Write-Host "Mode: All datasets analysis (datasets: $($AllDatasets -join ', '))" -ForegroundColor Yellow
}
else {
    Write-Host "Mode: Single dataset analysis" -ForegroundColor Yellow
}

Write-Host "Statistical measures: $($Measures -join ', ')" -ForegroundColor Magenta

# Check if data directory exists
if (-not (Test-Path $DataDir)) {
    Write-Host "Error: Data directory not found at $DataDir" -ForegroundColor Red
    exit 1
}

# Check if dataset directory exists (only for single dataset mode)
if ($AllDatasets.Count -eq 0 -and $MultiCycle.Count -eq 0) {
    $datasetPath = Join-Path $DataDir $DatasetName
    if (-not (Test-Path $datasetPath)) {
        Write-Host "Error: Dataset directory not found at $datasetPath" -ForegroundColor Red
        Write-Host "Available datasets:" -ForegroundColor Yellow
        Get-ChildItem $DataDir -Directory | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Yellow }
        exit 1
    }
}

# Run the Python script
$pythonArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/feature_rul_correlation.py",
    $DatasetName,
    "--feature", $Feature,
    "--cycle", $Cycle,
    "--data_dir", $DataDir,
    "--output", $Output,
    "--figsize", $Width, $Height,
    "--measures"
) + $Measures

if ($MultiCycle.Count -gt 0) {
    $pythonArgs += "--multi_cycle"
    $pythonArgs += $MultiCycle
}

if ($AllDatasets.Count -gt 0) {
    $pythonArgs += "--all_datasets"
    $pythonArgs += $AllDatasets
}

if ($Verbose) {
    $pythonArgs += "--verbose"
}

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully created feature-RUL correlation analysis plot(s)!" -ForegroundColor Green
        Write-Host "Plot(s) saved to: $Output" -ForegroundColor Cyan
    }
    else {
        Write-Host "Error creating plot(s)" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error running Python script: $_" -ForegroundColor Red
    exit 1
}
