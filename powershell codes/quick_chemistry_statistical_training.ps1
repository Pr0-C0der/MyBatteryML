# Quick Chemistry-Specific Statistical Feature Training
# This script runs a quick test of chemistry statistical training

param(
    [string]$ChemistryPath = "data/processed/LFP",  # Default to LFP chemistry
    [switch]$Verbose = $false
)

Write-Host "Quick Chemistry Statistical Training Test" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Chemistry: $ChemistryPath" -ForegroundColor Yellow
Write-Host ""

# Check if chemistry path exists
if (-not (Test-Path $ChemistryPath)) {
    Write-Host "Error: Chemistry path '$ChemistryPath' does not exist!" -ForegroundColor Red
    Write-Host "Please provide a valid chemistry folder path." -ForegroundColor Yellow
    exit 1
}

# Count PKL files
$PklCount = (Get-ChildItem -Path $ChemistryPath -Filter "*.pkl" -File).Count
Write-Host "Found $PklCount PKL files in chemistry folder" -ForegroundColor Cyan
Write-Host ""

if ($PklCount -eq 0) {
    Write-Host "No PKL files found in chemistry folder!" -ForegroundColor Red
    exit 1
}

# Build command
$CommandArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py"
    "--data_path", $ChemistryPath
    "--output_dir", "quick_chemistry_test_results"
    "--cycle_limit", 50
    "--smoothing", "none"
    "--verbose"
)

if ($Verbose) {
    $CommandArgs += "--verbose"
}

Write-Host "Running quick test..." -ForegroundColor Yellow
Write-Host "Command: python $($CommandArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    $StartTime = Get-Date
    python @CommandArgs
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    Write-Host ""
    Write-Host "Quick test completed in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    
    # Show results if available
    if (Test-Path "quick_chemistry_test_results/RMSE.csv") {
        Write-Host ""
        Write-Host "Results:" -ForegroundColor Green
        Import-Csv "quick_chemistry_test_results/RMSE.csv" | Format-Table -AutoSize
    }
    
} catch {
    Write-Host "Error running quick test: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Quick test completed successfully!" -ForegroundColor Green
