# Comprehensive Cross-Chemistry Training Experiments
# This script runs multiple cross-chemistry training experiments

param(
    [string]$DataPath = "data_chemistries",
    [string]$OutputBaseDir = "cross_chemistry_experiments",
    [int]$CycleLimit = 0,
    [switch]$UseGPU = $false,
    [switch]$Tune = $false,
    [switch]$Verbose = $false,
    [switch]$Help = $false
)

# Display help if requested
if ($Help) {
    Write-Host "Comprehensive Cross-Chemistry Training Experiments" -ForegroundColor Green
    Write-Host "=================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "This script runs multiple cross-chemistry training experiments:" -ForegroundColor Yellow
    Write-Host "1. Train on LFP, test on NMC, NCA, LCO, Mixed" -ForegroundColor White
    Write-Host "2. Train on NMC, test on LFP, NCA, LCO, Mixed" -ForegroundColor White
    Write-Host "3. Train on NCA, test on LFP, NMC, LCO, Mixed" -ForegroundColor White
    Write-Host "4. Train on LCO, test on LFP, NMC, NCA, Mixed" -ForegroundColor White
    Write-Host "5. Train on Mixed, test on LFP, NMC, NCA, LCO" -ForegroundColor White
    Write-Host ""
    Write-Host "Usage: .\run_all_cross_chemistry_experiments.ps1 [parameters]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Cyan
    Write-Host "  -DataPath <string>      : Base path to chemistry directories [default: data_chemistries]"
    Write-Host "  -OutputBaseDir <string> : Base output directory [default: cross_chemistry_experiments]"
    Write-Host "  -CycleLimit <int>       : Limit analysis to first N cycles [default: 0 (all cycles)]"
    Write-Host "  -UseGPU                 : Use GPU acceleration"
    Write-Host "  -Tune                   : Enable hyperparameter tuning"
    Write-Host "  -Verbose                : Enable verbose logging"
    Write-Host "  -Help                   : Show this help message"
    Write-Host ""
    exit 0
}

# Check if data path exists
if (-not (Test-Path $DataPath)) {
    Write-Host "Error: Data path not found: $DataPath" -ForegroundColor Red
    Write-Host "Please ensure the chemistry directories exist." -ForegroundColor Yellow
    exit 1
}

# Define chemistry types
$ChemistryTypes = @("lfp", "nmc", "nca", "lco", "mixed_nmc_lco")

# Check which chemistry directories exist
$AvailableChemistries = @()
foreach ($Chem in $ChemistryTypes) {
    $ChemPath = Join-Path $DataPath $Chem
    if (Test-Path $ChemPath) {
        $AvailableChemistries += $Chem
        Write-Host "✓ Found chemistry: $Chem" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Chemistry directory not found: $Chem" -ForegroundColor Red
    }
}

if ($AvailableChemistries.Count -lt 2) {
    Write-Host "Error: Need at least 2 chemistry types for cross-chemistry training!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Available chemistries: $($AvailableChemistries -join ', ')" -ForegroundColor Cyan
Write-Host ""

# Create base output directory
if (-not (Test-Path $OutputBaseDir)) {
    New-Item -ItemType Directory -Path $OutputBaseDir -Force | Out-Null
    Write-Host "Created base output directory: $OutputBaseDir" -ForegroundColor Cyan
}

# Define experiments
$Experiments = @()
foreach ($TrainChem in $AvailableChemistries) {
    $TestChems = $AvailableChemistries | Where-Object { $_ -ne $TrainChem }
    if ($TestChems.Count -gt 0) {
        $Experiments += @{
            TrainChemistry  = $TrainChem
            TestChemistries = $TestChems
            OutputDir       = Join-Path $OutputBaseDir "train_$TrainChem"
        }
    }
}

Write-Host "Planned experiments:" -ForegroundColor Cyan
foreach ($Exp in $Experiments) {
    Write-Host "  Train: $($Exp.TrainChemistry) -> Test: $($Exp.TestChemistries -join ', ')" -ForegroundColor White
}
Write-Host ""

# Run experiments
$TotalExperiments = $Experiments.Count
$CurrentExperiment = 0
$StartTime = Get-Date

foreach ($Exp in $Experiments) {
    $CurrentExperiment++
    $TrainPath = Join-Path $DataPath $Exp.TrainChemistry
    $TestPaths = $Exp.TestChemistries | ForEach-Object { Join-Path $DataPath $_ }
    
    Write-Host "=" * 80 -ForegroundColor Yellow
    Write-Host "Experiment $CurrentExperiment/$TotalExperiments" -ForegroundColor Yellow
    Write-Host "Training on: $($Exp.TrainChemistry)" -ForegroundColor Yellow
    Write-Host "Testing on: $($Exp.TestChemistries -join ', ')" -ForegroundColor Yellow
    Write-Host "Output: $($Exp.OutputDir)" -ForegroundColor Yellow
    Write-Host "=" * 80 -ForegroundColor Yellow
    Write-Host ""
    
    # Build command arguments
    $CommandArgs = @(
        "python", "run_cross_chemistry_training.py"
        "--train_chemistry", $TrainPath
        "--test_chemistries"
    ) + $TestPaths + @(
        "--output_dir", $Exp.OutputDir
        "--smoothing", "none"
        "--ma_window", "5"
        "--cv_splits", "5"
    )
    
    if ($CycleLimit -gt 0) {
        $CommandArgs += @("--cycle_limit", $CycleLimit)
    }
    
    if ($UseGPU) {
        $CommandArgs += "--use_gpu"
    }
    
    if ($Tune) {
        $CommandArgs += "--tune"
    }
    
    if ($Verbose) {
        $CommandArgs += "--verbose"
    }
    
    try {
        $ExpStartTime = Get-Date
        Write-Host "Running: $($CommandArgs -join ' ')" -ForegroundColor Cyan
        Write-Host ""
        
        & $CommandArgs[0] $CommandArgs[1..($CommandArgs.Length - 1)]
        
        $ExpEndTime = Get-Date
        $ExpDuration = $ExpEndTime - $ExpStartTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ Experiment $CurrentExperiment completed successfully!" -ForegroundColor Green
            Write-Host "Duration: $($ExpDuration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan
            
            # Show results summary
            $RMSEFile = Join-Path $Exp.OutputDir "cross_chemistry_RMSE.csv"
            if (Test-Path $RMSEFile) {
                Write-Host ""
                Write-Host "RMSE Results for $($Exp.TrainChemistry) -> $($Exp.TestChemistries -join ', '):" -ForegroundColor Green
                try {
                    $RMSE_Results = Import-Csv $RMSEFile
                    $RMSE_Results | Format-Table -AutoSize
                }
                catch {
                    Write-Host "Could not display RMSE results" -ForegroundColor Yellow
                }
            }
        }
        else {
            Write-Host "❌ Experiment $CurrentExperiment failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ Error running experiment $CurrentExperiment`: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Progress: $CurrentExperiment/$TotalExperiments experiments completed" -ForegroundColor Cyan
    Write-Host ""
}

$EndTime = Get-Date
$TotalDuration = $EndTime - $StartTime

Write-Host "=" * 80 -ForegroundColor Green
Write-Host "ALL EXPERIMENTS COMPLETED!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "Total Duration: $($TotalDuration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan
Write-Host "Results saved to: $OutputBaseDir" -ForegroundColor Yellow
Write-Host ""

# Show summary of all results
Write-Host "Summary of all experiments:" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

foreach ($Exp in $Experiments) {
    $RMSEFile = Join-Path $Exp.OutputDir "cross_chemistry_RMSE.csv"
    $MAEFile = Join-Path $Exp.OutputDir "cross_chemistry_MAE.csv"
    $MAPEFile = Join-Path $Exp.OutputDir "cross_chemistry_MAPE.csv"
    
    Write-Host ""
    Write-Host "Train: $($Exp.TrainChemistry) -> Test: $($Exp.TestChemistries -join ', ')" -ForegroundColor Yellow
    Write-Host "Output: $($Exp.OutputDir)" -ForegroundColor White
    
    if (Test-Path $RMSEFile) {
        Write-Host "  ✓ RMSE results available" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ RMSE results missing" -ForegroundColor Red
    }
    
    if (Test-Path $MAEFile) {
        Write-Host "  ✓ MAE results available" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ MAE results missing" -ForegroundColor Red
    }
    
    if (Test-Path $MAPEFile) {
        Write-Host "  ✓ MAPE results available" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ MAPE results missing" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Script execution completed!" -ForegroundColor Green
