# Quick Common Correlations Analysis Test
# This script runs a quick test of the common correlations analysis

param(
    [string]$DataPath = "data/preprocessed",
    [switch]$Verbose = $false
)

Write-Host "Quick Common Correlations Analysis Test" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host "Data Path: $DataPath" -ForegroundColor Yellow
Write-Host ""

# Check if data path exists
if (-not (Test-Path $DataPath)) {
    Write-Host "Error: Data path '$DataPath' does not exist!" -ForegroundColor Red
    Write-Host "Please provide a valid data path." -ForegroundColor Yellow
    exit 1
}

# Check for available datasets
$ExpectedDatasets = @("MATR", "CALCE", "HNEI", "OX", "RWTH", "SNL", "HUST", "UL_PUR")
$AvailableDatasets = @()

Write-Host "Checking for available datasets..." -ForegroundColor Cyan
foreach ($Dataset in $ExpectedDatasets) {
    $DatasetPath = Join-Path $DataPath $Dataset
    if (Test-Path $DatasetPath) {
        $PklCount = (Get-ChildItem -Path $DatasetPath -Filter "*.pkl" -File).Count
        if ($PklCount -gt 0) {
            $AvailableDatasets += $Dataset
            Write-Host "  ✓ $Dataset : $PklCount PKL files" -ForegroundColor Green
        }
    }
}

if ($AvailableDatasets.Count -eq 0) {
    Write-Host "Error: No datasets found with PKL files!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Found $($AvailableDatasets.Count) datasets: $($AvailableDatasets -join ', ')" -ForegroundColor Green
Write-Host ""

# Build command for quick test (lower thresholds for faster execution)
$CommandArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/find_common_statistical_correlations.py"
    "--data_path", $DataPath
    "--output_dir", "quick_correlations_test_results"
    "--correlation_threshold", 0.3  # Lower threshold for more results
    "--min_datasets", 2  # Lower minimum for more results
)

if ($Verbose) {
    $CommandArgs += "--verbose"
}

Write-Host "Running quick test..." -ForegroundColor Yellow
Write-Host "Command: python $($CommandArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    $StartTime = Get-Date
    python @Args
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    Write-Host ""
    Write-Host "Quick test completed in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    
    # Show results if available
    if (Test-Path "quick_correlations_test_results/common_features_summary.csv") {
        Write-Host ""
        Write-Host "Common Features Found:" -ForegroundColor Green
        Write-Host "====================" -ForegroundColor Green
        
        try {
            $CommonFeatures = Import-Csv "quick_correlations_test_results/common_features_summary.csv"
            $CommonFeatures | Select-Object -First 10 | Format-Table -AutoSize
        } catch {
            Write-Host "Could not display common features" -ForegroundColor Yellow
        }
    }
    
    # List output files
    Write-Host ""
    Write-Host "Output files created:" -ForegroundColor Green
    if (Test-Path "quick_correlations_test_results") {
        $OutputFiles = Get-ChildItem -Path "quick_correlations_test_results" -File
        foreach ($File in $OutputFiles) {
            Write-Host "  $($File.Name)" -ForegroundColor Cyan
        }
    }
    
} catch {
    Write-Host "Error running quick test: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Quick test completed successfully!" -ForegroundColor Green
