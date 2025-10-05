# Common Statistical Correlations Analysis Script
# This script finds statistical features with high correlations (> 0.5) with log(RUL)
# that are common across multiple datasets (MATR, CALCE, HNEI, OX, RWTH, SNL, HUST, UL_PUR)

param(
    [string]$DataPath = "data/preprocessed",
    [string]$OutputDir = "common_correlations_results",
    [double]$CorrelationThreshold = 0.5,
    [int]$MinDatasets = 3,
    [switch]$Verbose = $false
)

Write-Host "Common Statistical Correlations Analysis" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green
Write-Host "Data Path: $DataPath" -ForegroundColor Yellow
Write-Host "Output Directory: $OutputDir" -ForegroundColor Yellow
Write-Host "Correlation Threshold: $CorrelationThreshold" -ForegroundColor Yellow
Write-Host "Minimum Datasets: $MinDatasets" -ForegroundColor Yellow
Write-Host ""

# Check if data path exists
if (-not (Test-Path $DataPath)) {
    Write-Host "Error: Data path '$DataPath' does not exist!" -ForegroundColor Red
    exit 1
}

# Check for dataset directories
$ExpectedDatasets = @("MATR", "CALCE", "HNEI", "OX", "RWTH", "SNL", "HUST", "UL_PUR")
$AvailableDatasets = @()

Write-Host "Checking for dataset directories..." -ForegroundColor Cyan
foreach ($Dataset in $ExpectedDatasets) {
    $DatasetPath = Join-Path $DataPath $Dataset
    if (Test-Path $DatasetPath) {
        $PklCount = (Get-ChildItem -Path $DatasetPath -Filter "*.pkl" -File).Count
        if ($PklCount -gt 0) {
            $AvailableDatasets += $Dataset
            Write-Host "  ✓ $Dataset : $PklCount PKL files" -ForegroundColor Green
        }
        else {
            Write-Host "  ✗ $Dataset : No PKL files found" -ForegroundColor Red
        }
    }
    else {
        Write-Host "  ✗ $Dataset : Directory not found" -ForegroundColor Red
    }
}

if ($AvailableDatasets.Count -eq 0) {
    Write-Host "Error: No datasets found with PKL files!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Found $($AvailableDatasets.Count) datasets with data:" -ForegroundColor Green
Write-Host ($AvailableDatasets -join ", ") -ForegroundColor Cyan
Write-Host ""

# Build command arguments
$CommandArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/find_common_statistical_correlations.py"
    "--data_path", $DataPath
    "--output_dir", $OutputDir
    "--correlation_threshold", $CorrelationThreshold
    "--min_datasets", $MinDatasets
)

if ($Verbose) {
    $CommandArgs += "--verbose"
}

Write-Host "Running common correlations analysis..." -ForegroundColor Yellow
Write-Host "Command: python $($CommandArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    $StartTime = Get-Date
    python @CommandArgs
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    Write-Host ""
    Write-Host "Analysis completed in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    
    # Show results summary
    if (Test-Path "$OutputDir/common_features_summary.csv") {
        Write-Host ""
        Write-Host "Common Features Summary:" -ForegroundColor Green
        Write-Host "======================" -ForegroundColor Green
        
        try {
            $CommonFeatures = Import-Csv "$OutputDir/common_features_summary.csv"
            $CommonFeatures | Select-Object -First 10 | Format-Table -AutoSize
        }
        catch {
            Write-Host "Could not display common features summary" -ForegroundColor Yellow
        }
    }
    
    # Show correlation summary
    if (Test-Path "$OutputDir/correlation_summary.csv") {
        Write-Host ""
        Write-Host "Correlation Summary (Top 10):" -ForegroundColor Green
        Write-Host "============================" -ForegroundColor Green
        
        try {
            $CorrelationSummary = Import-Csv "$OutputDir/correlation_summary.csv"
            $CorrelationSummary | Select-Object -First 10 | Format-Table -AutoSize
        }
        catch {
            Write-Host "Could not display correlation summary" -ForegroundColor Yellow
        }
    }
    
    # List output files
    Write-Host ""
    Write-Host "Output files created:" -ForegroundColor Green
    Write-Host "===================" -ForegroundColor Green
    
    if (Test-Path $OutputDir) {
        $OutputFiles = Get-ChildItem -Path $OutputDir -File
        foreach ($File in $OutputFiles) {
            Write-Host "  $($File.Name)" -ForegroundColor Cyan
        }
    }
    
} # End of try block
catch {
    Write-Host "Error running analysis: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Common correlations analysis completed successfully!" -ForegroundColor Green
Write-Host "Results saved to: $OutputDir" -ForegroundColor Yellow
