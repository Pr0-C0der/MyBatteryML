# PowerShell script for Statistical Feature Training
# This script runs the statistical feature training for different datasets and parameters

param(
    [string]$Dataset = "MATR",
    [int]$CycleLimit = 100,
    [int]$NFeatures = 15,
    [double]$TestSize = 0.3,
    [string]$DataDir = "data/preprocessed",
    [string]$OutputDir = "statistical_training_results",
    [int]$RandomState = 42,
    [switch]$AllDatasets = $false,
    [switch]$Help = $false
)

# Display help if requested
if ($Help) {
    Write-Host "Statistical Feature Training PowerShell Script" -ForegroundColor Green
    Write-Host "=============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\run_statistical_feature_training.ps1 [parameters]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Cyan
    Write-Host "  -Dataset <string>     : Dataset name (MATR, CALCE, UL_PUR, etc.) [default: MATR]"
    Write-Host "  -CycleLimit <int>     : Maximum cycles to use [default: 100]"
    Write-Host "  -NFeatures <int>      : Number of top features to select [default: 15]"
    Write-Host "  -TestSize <double>    : Test set size (0.0-1.0) [default: 0.3]"
    Write-Host "  -DataDir <string>     : Data directory [default: data/preprocessed]"
    Write-Host "  -OutputDir <string>   : Output directory [default: statistical_training_results]"
    Write-Host "  -RandomState <int>    : Random state for reproducibility [default: 42]"
    Write-Host "  -AllDatasets          : Run training for all available datasets"
    Write-Host "  -Help                 : Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run_statistical_feature_training.ps1 -Dataset MATR -CycleLimit 50"
    Write-Host "  .\run_statistical_feature_training.ps1 -AllDatasets -NFeatures 20"
    Write-Host "  .\run_statistical_feature_training.ps1 -Dataset CALCE -CycleLimit 200 -NFeatures 10"
    exit 0
}

# Available datasets
$AvailableDatasets = @("MATR", "CALCE", "UL_PUR", "HUST", "RWTH", "OX", "SNL", "HNEI")

# Function to run training for a single dataset
function Run-Training {
    param(
        [string]$DatasetName,
        [int]$CycleLimit,
        [int]$NFeatures,
        [double]$TestSize,
        [string]$DataDir,
        [string]$OutputDir,
        [int]$RandomState
    )
    
    Write-Host "`n=========================================" -ForegroundColor Green
    Write-Host "Training for dataset: $DatasetName" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    
    # Check if dataset exists
    $DatasetPath = Join-Path $DataDir $DatasetName
    if (-not (Test-Path $DatasetPath)) {
        Write-Host "Warning: Dataset directory not found: $DatasetPath" -ForegroundColor Yellow
        Write-Host "Skipping $DatasetName..." -ForegroundColor Yellow
        return $false
    }
    
    # Check if PKL files exist
    $PklFiles = Get-ChildItem -Path $DatasetPath -Filter "*.pkl" -ErrorAction SilentlyContinue
    if ($PklFiles.Count -eq 0) {
        Write-Host "Warning: No PKL files found in $DatasetPath" -ForegroundColor Yellow
        Write-Host "Skipping $DatasetName..." -ForegroundColor Yellow
        return $false
    }
    
    Write-Host "Found $($PklFiles.Count) PKL files in $DatasetPath" -ForegroundColor Cyan
    
    # Run the training script
    $Command = "python batteryml/chemistry_data_analysis/statistical_analysis/statistical_feature_training_v2.py"
    $Command += " $DatasetName"
    $Command += " --cycle_limit $CycleLimit"
    $Command += " --n_features $NFeatures"
    $Command += " --test_size $TestSize"
    $Command += " --data_dir `"$DataDir`""
    $Command += " --output_dir `"$OutputDir`""
    $Command += " --random_state $RandomState"
    
    Write-Host "Running command: $Command" -ForegroundColor Cyan
    Write-Host ""
    
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✅ Training completed successfully for $DatasetName" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "`n❌ Training failed for $DatasetName (exit code: $LASTEXITCODE)" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "`n❌ Error running training for $DatasetName`: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to run training for all datasets
function Run-AllDatasets {
    param(
        [int]$CycleLimit,
        [int]$NFeatures,
        [double]$TestSize,
        [string]$DataDir,
        [string]$OutputDir,
        [int]$RandomState
    )
    
    Write-Host "Running training for all available datasets..." -ForegroundColor Green
    Write-Host "Available datasets: $($AvailableDatasets -join ', ')" -ForegroundColor Cyan
    
    $Results = @{}
    $SuccessCount = 0
    $TotalCount = $AvailableDatasets.Count
    
    foreach ($Dataset in $AvailableDatasets) {
        $Success = Run-Training -DatasetName $Dataset -CycleLimit $CycleLimit -NFeatures $NFeatures -TestSize $TestSize -DataDir $DataDir -OutputDir $OutputDir -RandomState $RandomState
        $Results[$Dataset] = $Success
        if ($Success) {
            $SuccessCount++
        }
    }
    
    # Summary
    Write-Host "`n=========================================" -ForegroundColor Green
    Write-Host "TRAINING SUMMARY" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "Total datasets: $TotalCount" -ForegroundColor Cyan
    Write-Host "Successful: $SuccessCount" -ForegroundColor Green
    Write-Host "Failed: $($TotalCount - $SuccessCount)" -ForegroundColor Red
    
    Write-Host "`nDetailed Results:" -ForegroundColor Cyan
    foreach ($Dataset in $AvailableDatasets) {
        $Status = if ($Results[$Dataset]) { "✅ Success" } else { "❌ Failed" }
        $Color = if ($Results[$Dataset]) { "Green" } else { "Red" }
        Write-Host "  $Dataset : $Status" -ForegroundColor $Color
    }
    
    # Show metric results summary
    Write-Host "`n=========================================" -ForegroundColor Green
    Write-Host "METRIC RESULTS SUMMARY" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    
    # RMSE Results
    if (Test-Path "$OutputDir/RMSE.csv") {
        Write-Host "`nRMSE Results:" -ForegroundColor Cyan
        Write-Host "============" -ForegroundColor Cyan
        
        try {
            $RMSE_Results = Import-Csv "$OutputDir/RMSE.csv"
            $RMSE_Results | Format-Table -AutoSize
        }
        catch {
            Write-Host "Could not display RMSE results summary" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "RMSE.csv not found in output directory" -ForegroundColor Yellow
    }

    # MAE Results
    if (Test-Path "$OutputDir/MAE.csv") {
        Write-Host "`nMAE Results:" -ForegroundColor Cyan
        Write-Host "===========" -ForegroundColor Cyan
        
        try {
            $MAE_Results = Import-Csv "$OutputDir/MAE.csv"
            $MAE_Results | Format-Table -AutoSize
        }
        catch {
            Write-Host "Could not display MAE results summary" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "MAE.csv not found in output directory" -ForegroundColor Yellow
    }

    # MAPE Results
    if (Test-Path "$OutputDir/MAPE.csv") {
        Write-Host "`nMAPE Results (%):" -ForegroundColor Cyan
        Write-Host "================" -ForegroundColor Cyan
        
        try {
            $MAPE_Results = Import-Csv "$OutputDir/MAPE.csv"
            $MAPE_Results | Format-Table -AutoSize
        }
        catch {
            Write-Host "Could not display MAPE results summary" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "MAPE.csv not found in output directory" -ForegroundColor Yellow
    }
}

# Main execution
Write-Host "Statistical Feature Training PowerShell Script" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Validate parameters
if ($CycleLimit -lt 1) {
    Write-Host "Error: CycleLimit must be greater than 0" -ForegroundColor Red
    exit 1
}

if ($NFeatures -lt 1) {
    Write-Host "Error: NFeatures must be greater than 0" -ForegroundColor Red
    exit 1
}

if ($TestSize -lt 0.0 -or $TestSize -gt 1.0) {
    Write-Host "Error: TestSize must be between 0.0 and 1.0" -ForegroundColor Red
    exit 1
}

# Check if data directory exists
if (-not (Test-Path $DataDir)) {
    Write-Host "Error: Data directory not found: $DataDir" -ForegroundColor Red
    Write-Host "Please check the path and try again." -ForegroundColor Yellow
    exit 1
}

# Check if Python script exists
$ScriptPath = "batteryml/chemistry_data_analysis/statistical_analysis/statistical_feature_training_v2.py"
if (-not (Test-Path $ScriptPath)) {
    Write-Host "Error: Training script not found: $ScriptPath" -ForegroundColor Red
    Write-Host "Please ensure you're running from the correct directory." -ForegroundColor Yellow
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Cyan
}

# Display configuration
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Dataset: $Dataset" -ForegroundColor White
Write-Host "  Cycle Limit: $CycleLimit" -ForegroundColor White
Write-Host "  Number of Features: $NFeatures" -ForegroundColor White
Write-Host "  Test Size: $TestSize" -ForegroundColor White
Write-Host "  Data Directory: $DataDir" -ForegroundColor White
Write-Host "  Output Directory: $OutputDir" -ForegroundColor White
Write-Host "  Random State: $RandomState" -ForegroundColor White
Write-Host "  Run All Datasets: $AllDatasets" -ForegroundColor White
Write-Host ""

# Run training
if ($AllDatasets) {
    Run-AllDatasets -CycleLimit $CycleLimit -NFeatures $NFeatures -TestSize $TestSize -DataDir $DataDir -OutputDir $OutputDir -RandomState $RandomState
}
else {
    # Validate single dataset
    if ($AvailableDatasets -notcontains $Dataset) {
        Write-Host "Warning: Dataset '$Dataset' not in known datasets list." -ForegroundColor Yellow
        Write-Host "Known datasets: $($AvailableDatasets -join ', ')" -ForegroundColor Yellow
        Write-Host "Proceeding anyway..." -ForegroundColor Yellow
    }
    
    $Success = Run-Training -DatasetName $Dataset -CycleLimit $CycleLimit -NFeatures $NFeatures -TestSize $TestSize -DataDir $DataDir -OutputDir $OutputDir -RandomState $RandomState
    
    if ($Success) {
        Write-Host "`n🎉 All training completed successfully!" -ForegroundColor Green
        
        # Show metric results summary for single dataset
        Write-Host "`n=========================================" -ForegroundColor Green
        Write-Host "METRIC RESULTS SUMMARY" -ForegroundColor Green
        Write-Host "=========================================" -ForegroundColor Green
        
        # RMSE Results
        if (Test-Path "$OutputDir/RMSE.csv") {
            Write-Host "`nRMSE Results:" -ForegroundColor Cyan
            Write-Host "============" -ForegroundColor Cyan
            
            try {
                $RMSE_Results = Import-Csv "$OutputDir/RMSE.csv"
                $RMSE_Results | Format-Table -AutoSize
            }
            catch {
                Write-Host "Could not display RMSE results summary" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "RMSE.csv not found in output directory" -ForegroundColor Yellow
        }

        # MAE Results
        if (Test-Path "$OutputDir/MAE.csv") {
            Write-Host "`nMAE Results:" -ForegroundColor Cyan
            Write-Host "===========" -ForegroundColor Cyan
            
            try {
                $MAE_Results = Import-Csv "$OutputDir/MAE.csv"
                $MAE_Results | Format-Table -AutoSize
            }
            catch {
                Write-Host "Could not display MAE results summary" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "MAE.csv not found in output directory" -ForegroundColor Yellow
        }

        # MAPE Results
        if (Test-Path "$OutputDir/MAPE.csv") {
            Write-Host "`nMAPE Results (%):" -ForegroundColor Cyan
            Write-Host "================" -ForegroundColor Cyan
            
            try {
                $MAPE_Results = Import-Csv "$OutputDir/MAPE.csv"
                $MAPE_Results | Format-Table -AutoSize
            }
            catch {
                Write-Host "Could not display MAPE results summary" -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "MAPE.csv not found in output directory" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "`n💥 Training failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nScript execution completed." -ForegroundColor Green
