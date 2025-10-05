# Chemistry-Specific Statistical Feature Training Script
# This script runs statistical feature training for chemistry-specific datasets
# following the pattern from chemistry_training.py

param(
    [string]$ChemistryPath = "",
    [string]$OutputDir = "chemistry_statistical_results",
    [string]$DatasetHint = "",
    [int]$CycleLimit = 100,
    [string]$Smoothing = "none",
    [int]$MaWindow = 5,
    [string[]]$Features = @(),
    [switch]$UseGPU = $false,
    [switch]$Tune = $false,
    [int]$CvSplits = 5,
    [double]$TrainTestRatio = 0.7,
    [switch]$Verbose = $false
)

# Set default chemistry paths if not provided
if ([string]::IsNullOrEmpty($ChemistryPath)) {
    $ChemistryPath = "data/processed"
}

Write-Host "Chemistry-Specific Statistical Feature Training" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Chemistry Path: $ChemistryPath" -ForegroundColor Yellow
Write-Host "Output Directory: $OutputDir" -ForegroundColor Yellow
Write-Host "Cycle Limit: $CycleLimit" -ForegroundColor Yellow
Write-Host "Smoothing: $Smoothing" -ForegroundColor Yellow
Write-Host "Use GPU: $UseGPU" -ForegroundColor Yellow
Write-Host "Hyperparameter Tuning: $Tune" -ForegroundColor Yellow
Write-Host ""

# Check if chemistry path exists
if (-not (Test-Path $ChemistryPath)) {
    Write-Host "Error: Chemistry path '$ChemistryPath' does not exist!" -ForegroundColor Red
    exit 1
}

# Get all chemistry folders
$ChemistryFolders = Get-ChildItem -Path $ChemistryPath -Directory | Where-Object { $_.Name -match "^(LFP|NMC|NCA|NCM|LCO|LMO|LTO|SILICON|GRAPHITE)$" }

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
$Args = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py"
    "--data_path", $ChemistryPath
    "--output_dir", $OutputDir
    "--cycle_limit", $CycleLimit
    "--smoothing", $Smoothing
    "--ma_window", $MaWindow
    "--cv_splits", $CvSplits
    "--train_test_ratio", $TrainTestRatio
)

if (-not [string]::IsNullOrEmpty($DatasetHint)) {
    $Args += @("--dataset_hint", $DatasetHint)
}

if ($Features.Count -gt 0) {
    $Args += @("--features") + $Features
}

if ($UseGPU) {
    $Args += "--use_gpu"
}

if ($Tune) {
    $Args += "--tune"
}

if ($Verbose) {
    $Args += "--verbose"
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
    $ChemistryArgs = $Args.Clone()
    $ChemistryArgs[1] = $ChemistryPath
    
    try {
        # Run the training
        $StartTime = Get-Date
        python @ChemistryArgs
        
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        
        Write-Host "Completed $ChemistryName in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
        Write-Host ""
        
    } catch {
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
    } catch {
        Write-Host "Could not display results summary" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Training completed successfully!" -ForegroundColor Green
