# Quick Chemistry Statistical Training Test
# This script tests the updated chemistry statistical training with manual features

param(
    [string]$ChemistryPath = "data/processed/LFP",
    [int]$CycleLimit = 100,
    [string]$Smoothing = "moving_mean",
    [int]$MaWindow = 5,
    [string[]]$ManualFeatures = @("charge_cycle_length", "avg_c_rate", "cycle_length"),
    [switch]$Verbose = $false
)

Write-Host "Quick Chemistry Statistical Training Test" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Chemistry Path: $ChemistryPath" -ForegroundColor Yellow
Write-Host "Smoothing: $Smoothing" -ForegroundColor Yellow
Write-Host "Base Features: $($ManualFeatures -join ', ')" -ForegroundColor Yellow
Write-Host "  (Each will be expanded with statistical measures: mean, std, min, max, median, q25, q75, skewness, kurtosis)" -ForegroundColor Gray
Write-Host ""

# Check if chemistry path exists
if (-not (Test-Path $ChemistryPath)) {
    Write-Host "Error: Chemistry path '$ChemistryPath' does not exist!" -ForegroundColor Red
    Write-Host "Please provide a valid chemistry path." -ForegroundColor Yellow
    exit 1
}

# Check for PKL files
$PklFiles = Get-ChildItem -Path $ChemistryPath -Filter "*.pkl" -File
if ($PklFiles.Count -eq 0) {
    Write-Host "Error: No PKL files found in '$ChemistryPath'" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($PklFiles.Count) PKL files in chemistry folder" -ForegroundColor Green
Write-Host ""

# Build command for quick test
$CommandArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py"
    "--data_path", $ChemistryPath
    "--output_dir", "quick_chemistry_test_results"
    "--smoothing", $Smoothing
    "--ma_window", $MaWindow
    "--manual_features"
) + $ManualFeatures

if ($Verbose) {
    $CommandArgs += "--verbose"
}

Write-Host "Running quick chemistry test..." -ForegroundColor Yellow
Write-Host "Command: python $($CommandArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    $StartTime = Get-Date
    python @CommandArgs
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    Write-Host ""
    Write-Host "Quick chemistry test completed in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    
    # Show results if available
    if (Test-Path "quick_chemistry_test_results") {
        Write-Host ""
        Write-Host "Output files created:" -ForegroundColor Green
        $OutputFiles = Get-ChildItem -Path "quick_chemistry_test_results" -Recurse -File
        foreach ($File in $OutputFiles) {
            Write-Host "  $($File.FullName.Replace((Get-Location).Path + '\', ''))" -ForegroundColor Cyan
        }
        
        # Show RMSE results if available
        $RmseFile = Get-ChildItem -Path "quick_chemistry_test_results" -Filter "RMSE.csv" -Recurse
        if ($RmseFile) {
            Write-Host ""
            Write-Host "RMSE Results:" -ForegroundColor Green
            Write-Host "============" -ForegroundColor Green
            try {
                $Results = Import-Csv $RmseFile.FullName
                $Results | Format-Table -AutoSize
            }
            catch {
                Write-Host "Could not display RMSE results" -ForegroundColor Yellow
            }
        }
    }
    
}
catch {
    Write-Host "Error running quick chemistry test: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Quick chemistry test completed successfully!" -ForegroundColor Green
Write-Host "Try different manual features or smoothing methods for experimentation." -ForegroundColor Yellow
