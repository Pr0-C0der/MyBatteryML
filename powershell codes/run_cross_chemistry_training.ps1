# Cross-Chemistry Training PowerShell Script
# This script trains models on one chemistry and tests on other chemistries

param(
    [string]$TrainChemistry = "data_chemistries/lfp",
    [string[]]$TestChemistries = @("data_chemistries/nmc", "data_chemistries/nca", "data_chemistries/lco", "data_chemistries/mixed_nmc_lco"),
    [string]$OutputDir = "cross_chemistry_results",
    [string]$DatasetHint = "",
    [int]$CycleLimit = 100,
    [string]$Smoothing = "hms",
    [int]$MaWindow = 5,
    [string[]]$Features = @("avg_charge_capacity", "max_charge_capacity", "charge_cycle_length", "discharge_cycle_length", "cycle_length", "max_charge_c_rate", "avg_discharge_c_rate"),
    [switch]$UseGPU = $false,
    [switch]$Tune = $false,
    [int]$CvSplits = 5,
    [switch]$Verbose = $false,
    [switch]$Help = $false
)

# Display help if requested
if ($Help) {
    Write-Host "Cross-Chemistry Training PowerShell Script" -ForegroundColor Green
    Write-Host "===========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\run_cross_chemistry_training.ps1 [parameters]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Cyan
    Write-Host "  -TrainChemistry <string>    : Path to training chemistry folder [required]"
    Write-Host "  -TestChemistries <array>    : List of test chemistry folder paths [required]"
    Write-Host "  -OutputDir <string>         : Output directory [default: cross_chemistry_results]"
    Write-Host "  -DatasetHint <string>       : Optional dataset name hint"
    Write-Host "  -CycleLimit <int>           : Limit analysis to first N cycles [default: 0 (all cycles)]"
    Write-Host "  -Smoothing <string>         : Smoothing method (none, ma, median, hms) [default: none]"
    Write-Host "  -MaWindow <int>             : Window size for smoothing [default: 5]"
    Write-Host "  -Features <array>           : Features to use [default: avg_charge_capacity, max_charge_capacity, etc.]"
    Write-Host "  -UseGPU                     : Use GPU acceleration"
    Write-Host "  -Tune                       : Enable hyperparameter tuning"
    Write-Host "  -CvSplits <int>             : Number of cross-validation splits [default: 5]"
    Write-Host "  -Verbose                    : Enable verbose logging"
    Write-Host "  -Help                       : Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run_cross_chemistry_training.ps1 -TrainChemistry 'data_chemistries/lfp' -TestChemistries @('data_chemistries/nmc', 'data_chemistries/nca')"
    Write-Host "  .\run_cross_chemistry_training.ps1 -TrainChemistry 'data_chemistries/lfp' -TestChemistries @('data_chemistries/nmc', 'data_chemistries/nca', 'data_chemistries/lco') -CycleLimit 100 -Verbose"
    Write-Host "  .\run_cross_chemistry_training.ps1 -TrainChemistry 'data_chemistries/nmc' -TestChemistries @('data_chemistries/lfp', 'data_chemistries/nca') -UseGPU -Tune"
    exit 0
}

# Validate parameters
if ([string]::IsNullOrEmpty($TrainChemistry)) {
    Write-Host "Error: TrainChemistry must be specified!" -ForegroundColor Red
    Write-Host "Use -Help to see usage examples." -ForegroundColor Yellow
    exit 1
}

if ($TestChemistries.Count -eq 0) {
    Write-Host "Error: At least one test chemistry must be specified!" -ForegroundColor Red
    Write-Host "Use -Help to see usage examples." -ForegroundColor Yellow
    exit 1
}

if ($CycleLimit -lt 0) {
    Write-Host "Error: CycleLimit must be 0 or greater (0 = all cycles)" -ForegroundColor Red
    exit 1
}

if ($MaWindow -lt 1) {
    Write-Host "Error: MaWindow must be 1 or greater" -ForegroundColor Red
    exit 1
}

# Check if chemistry directories exist
if (-not (Test-Path $TrainChemistry)) {
    Write-Host "Error: Training chemistry directory not found: $TrainChemistry" -ForegroundColor Red
    exit 1
}

$ValidTestChemistries = @()
foreach ($TestChem in $TestChemistries) {
    if (Test-Path $TestChem) {
        $ValidTestChemistries += $TestChem
        Write-Host "Found test chemistry: $TestChem" -ForegroundColor Green
    }
    else {
        Write-Host "Test chemistry directory not found: $TestChem" -ForegroundColor Red
    }
}

if ($ValidTestChemistries.Count -eq 0) {
    Write-Host "Error: No valid test chemistry directories found!" -ForegroundColor Red
    exit 1
}

if ($ValidTestChemistries.Count -lt $TestChemistries.Count) {
    Write-Host "Warning: Some test chemistry directories were not found. Proceeding with $($ValidTestChemistries.Count) valid directories." -ForegroundColor Yellow
}

# Check if Python script exists
$ScriptPath = "batteryml/chemistry_data_analysis/cross_chemistry_training.py"
if (-not (Test-Path $ScriptPath)) {
    Write-Host "Error: Python script not found: $ScriptPath" -ForegroundColor Red
    Write-Host "Please ensure you're running from the correct directory." -ForegroundColor Yellow
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Cyan
}

# Display configuration
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Training Chemistry: $TrainChemistry" -ForegroundColor White
Write-Host "  Test Chemistries: $($ValidTestChemistries -join ', ')" -ForegroundColor White
Write-Host "  Output Directory: $OutputDir" -ForegroundColor White
Write-Host "  Dataset Hint: $(if ($DatasetHint) { $DatasetHint } else { 'Auto-detect' })" -ForegroundColor White
Write-Host "  Cycle Limit: $(if ($CycleLimit -gt 0) { $CycleLimit } else { 'All cycles' })" -ForegroundColor White
Write-Host "  Smoothing: $Smoothing" -ForegroundColor White
Write-Host "  MA Window: $MaWindow" -ForegroundColor White
Write-Host "  Features: $($Features -join ', ')" -ForegroundColor White
Write-Host "  Use GPU: $UseGPU" -ForegroundColor White
Write-Host "  Tune: $Tune" -ForegroundColor White
Write-Host "  CV Splits: $CvSplits" -ForegroundColor White
Write-Host "  Verbose: $Verbose" -ForegroundColor White
Write-Host ""

# Build command arguments
$CommandArgs = @(
    "python", $ScriptPath
    "--train_chemistry", $TrainChemistry
    "--test_chemistries"
) + $ValidTestChemistries + @(
    "--output_dir", $OutputDir
    "--smoothing", $Smoothing
    "--ma_window", $MaWindow
    "--cv_splits", $CvSplits
)

if ($DatasetHint) {
    $CommandArgs += @("--dataset_hint", $DatasetHint)
}

if ($CycleLimit -gt 0) {
    $CommandArgs += @("--cycle_limit", $CycleLimit)
}

if ($UseGPU) {
    $CommandArgs += "--use_gpu"
}

if ($Tune) {
    $CommandArgs += "--tune"
}

if ($Features.Count -gt 0) {
    $CommandArgs += "--features"
    $CommandArgs += $Features
}

if ($Verbose) {
    $CommandArgs += "--verbose"
}


Write-Host "Running command: $($CommandArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""

try {
    # Run the Python script
    $StartTime = Get-Date
    & $CommandArgs[0] $CommandArgs[1..($CommandArgs.Length - 1)]
    
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Cross-chemistry training completed successfully!" -ForegroundColor Green
        Write-Host "Duration: $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan
        Write-Host "Results saved to: $OutputDir" -ForegroundColor Yellow
        
        # Show output files
        if (Test-Path $OutputDir) {
            Write-Host ""
            Write-Host "Generated files:" -ForegroundColor Cyan
            Get-ChildItem -Path $OutputDir -Filter "*.csv" | ForEach-Object {
                Write-Host "  - $($_.Name)" -ForegroundColor White
            }
        }
        
        # Show results summary if CSV files exist
        if (Test-Path "$OutputDir/cross_chemistry_RMSE.csv") {
            Write-Host ""
            Write-Host "RMSE Results Summary:" -ForegroundColor Green
            Write-Host "===================" -ForegroundColor Green
            try {
                $RMSE_Results = Import-Csv "$OutputDir/cross_chemistry_RMSE.csv"
                $RMSE_Results | Format-Table -AutoSize
            }
            catch {
                Write-Host "Could not display RMSE results summary" -ForegroundColor Yellow
            }
        }
        
        if (Test-Path "$OutputDir/cross_chemistry_MAE.csv") {
            Write-Host ""
            Write-Host "MAE Results Summary:" -ForegroundColor Green
            Write-Host "===================" -ForegroundColor Green
            try {
                $MAE_Results = Import-Csv "$OutputDir/cross_chemistry_MAE.csv"
                $MAE_Results | Format-Table -AutoSize
            }
            catch {
                Write-Host "Could not display MAE results summary" -ForegroundColor Yellow
            }
        }
        
        if (Test-Path "$OutputDir/cross_chemistry_MAPE.csv") {
            Write-Host ""
            Write-Host "MAPE Results Summary (%):" -ForegroundColor Green
            Write-Host "========================" -ForegroundColor Green
            try {
                $MAPE_Results = Import-Csv "$OutputDir/cross_chemistry_MAPE.csv"
                $MAPE_Results | Format-Table -AutoSize
            }
            catch {
                Write-Host "Could not display MAPE results summary" -ForegroundColor Yellow
            }
        }
    }
    else {
        Write-Host "Cross-chemistry training failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error running cross-chemistry training`: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Script execution completed!" -ForegroundColor Green
