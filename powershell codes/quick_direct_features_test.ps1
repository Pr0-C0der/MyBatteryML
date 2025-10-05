# Quick Direct Statistical Features Test
# This script tests training with specific statistical features

param(
    [string]$ChemistryPath = "data/processed/LFP",
    [int]$CycleLimit = 100,
    [string[]]$DirectFeatures = @("mean_cycle_length", "median_charge_cycle_length"),
    [switch]$Verbose = $false
)

Write-Host "Quick Direct Statistical Features Test" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Chemistry Path: $ChemistryPath" -ForegroundColor Yellow
Write-Host "Cycle Limit: $CycleLimit" -ForegroundColor Yellow
Write-Host "Direct Features: $($DirectFeatures -join ', ')" -ForegroundColor Yellow
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
    "--output_dir", "quick_direct_features_test_results"
    "--cycle_limit", $CycleLimit
    "--direct_statistical_features"
) + $DirectFeatures

if ($Verbose) {
    $CommandArgs += "--verbose"
}

Write-Host "Running quick direct features test..." -ForegroundColor Yellow
Write-Host "Command: python $($CommandArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    $StartTime = Get-Date
    python @CommandArgs
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    Write-Host ""
    Write-Host "Quick direct features test completed in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    
    # Show results if available
    if (Test-Path "quick_direct_features_test_results") {
        Write-Host ""
        Write-Host "Output files created:" -ForegroundColor Green
        $OutputFiles = Get-ChildItem -Path "quick_direct_features_test_results" -Recurse -File
        foreach ($File in $OutputFiles) {
            Write-Host "  $($File.FullName.Replace((Get-Location).Path + '\', ''))" -ForegroundColor Cyan
        }
        
        # Show RMSE results if available
        $RmseFile = Get-ChildItem -Path "quick_direct_features_test_results" -Filter "RMSE.csv" -Recurse
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
    Write-Host "Error running quick direct features test: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Quick direct features test completed successfully!" -ForegroundColor Green
Write-Host "Try different combinations of statistical features for experimentation." -ForegroundColor Yellow
