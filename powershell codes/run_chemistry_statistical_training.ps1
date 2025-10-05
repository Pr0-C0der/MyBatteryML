# Chemistry-Specific Statistical Feature Training Script
# This script runs statistical feature training for chemistry-specific datasets
# following the pattern from chemistry_training.py

param(
    [string]$ChemistryPath = "",
    [string]$OutputDir = "chemistry_statistical_results",
    [string]$DatasetHint = "",
    [int]$CycleLimit = 0,
    [string]$Smoothing = "none",
    [int]$MaWindow = 5,
    [string[]]$ManualFeatures = @(),
    [switch]$UseGPU = $false,
    [switch]$Tune = $false,
    [int]$CvSplits = 5,
    [double]$TrainTestRatio = 0.7,
    [switch]$Verbose = $false
)

# Set default chemistry paths if not provided
if ([string]::IsNullOrEmpty($ChemistryPath)) {
    $ChemistryPath = "data_chemistries"
}

Write-Host "Chemistry-Specific Statistical Feature Training" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Chemistry Path: $ChemistryPath" -ForegroundColor Yellow
Write-Host "Output Directory: $OutputDir" -ForegroundColor Yellow
Write-Host "Cycle Limit: $(if ($CycleLimit -gt 0) { $CycleLimit } else { 'None (all cycles)' })" -ForegroundColor Yellow
Write-Host "Smoothing: $Smoothing" -ForegroundColor Yellow
Write-Host "Base Features: $($ManualFeatures -join ', ')" -ForegroundColor Yellow
Write-Host "  (Each will be expanded with statistical measures: mean, std, min, max, median, q25, q75, skewness, kurtosis)" -ForegroundColor Gray
Write-Host "Use GPU: $UseGPU" -ForegroundColor Yellow
Write-Host "Hyperparameter Tuning: $Tune" -ForegroundColor Yellow
Write-Host ""

# Check if chemistry path exists
if (-not (Test-Path $ChemistryPath)) {
    Write-Host "Error: Chemistry path '$ChemistryPath' does not exist!" -ForegroundColor Red
    exit 1
}

# Get all chemistry folders
$ChemistryFolders = Get-ChildItem -Path $ChemistryPath -Directory | Where-Object { $_.Name -match "^(lfp|nmc|nca|lco|mixed_nmc_lco)$" }

if ($ChemistryFolders.Count -eq 0) {
    Write-Host "No chemistry folders found in '$ChemistryPath'" -ForegroundColor Red
    Write-Host "Expected folders: LFP, NMC, NCA, NCM, LCO, LMO, LTO, SILICON, GRAPHITE" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found chemistry folders:" -ForegroundColor Green
foreach ($folder in $ChemistryFolders) {
    $pklCount = (Get-ChildItem -Path $folder.FullName -Filter "*.pkl" -File).Count
    Write-Host "  $($folder.Name): $pklCount PKL files" -ForegroundColor Cyan
}
Write-Host ""

# Build command arguments
$CommandArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py"
    "--data_path", $ChemistryPath
    "--output_dir", $OutputDir
    "--smoothing", $Smoothing
    "--ma_window", $MaWindow
    "--cv_splits", $CvSplits
    "--train_test_ratio", $TrainTestRatio
)

if (-not [string]::IsNullOrEmpty($DatasetHint)) {
    $CommandArgs += @("--dataset_hint", $DatasetHint)
}

if ($CycleLimit -gt 0) {
    $CommandArgs += @("--cycle_limit", $CycleLimit)
}

if ($ManualFeatures.Count -gt 0) {
    $CommandArgs += @("--manual_features") + $ManualFeatures
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

# Run training for each chemistry
$TotalChemistries = $ChemistryFolders.Count
$CurrentChemistry = 0

foreach ($ChemistryFolder in $ChemistryFolders) {
    $CurrentChemistry++
    $ChemistryName = $ChemistryFolder.Name
    $ChemistryPath = $ChemistryFolder.FullName
    
    Write-Host "Training $ChemistryName ($CurrentChemistry/$TotalChemistries)" -ForegroundColor Magenta
    Write-Host "Path: $ChemistryPath" -ForegroundColor Gray
    Write-Host "----------------------------------------" -ForegroundColor Magenta
    
    # Update data path for this chemistry
    $ChemistryArgs = $CommandArgs.Clone()
    $ChemistryArgs[2] = $ChemistryPath
    
    try {
        # Run the training
        $StartTime = Get-Date
        python @ChemistryArgs
        
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        
        Write-Host "Completed $ChemistryName in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
        Write-Host ""
        
    }
    catch {
        Write-Host "Error training $ChemistryName : $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
    }
}

Write-Host "Chemistry-Specific Statistical Training Complete!" -ForegroundColor Green
Write-Host "Results saved to: $OutputDir" -ForegroundColor Yellow

# Show summary of results
if (Test-Path "$OutputDir/RMSE.csv") {
    Write-Host ""
    Write-Host "RMSE Results Summary:" -ForegroundColor Green
    Write-Host "===================" -ForegroundColor Green
    
    try {
        $Results = Import-Csv "$OutputDir/RMSE.csv"
        $Results | Format-Table -AutoSize
    }
    catch {
        Write-Host "Could not display results summary" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Training completed successfully!" -ForegroundColor Green
