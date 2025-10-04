# PowerShell script to run comprehensive correlation analysis
# This script generates correlation plots for all datasets and features

param(
    [string[]]$Datasets = @(),
    [string[]]$Features = @(),
    [int]$CycleLimit = 100,
    [ValidateSet("none", "hms", "ma", "median")]
    [string]$Smoothing = "hms",
    [int]$SmoothingWindow = 5,
    [string]$OutputDir = "statistical_correlation_analysis",
    [int]$Width = 12,
    [int]$Height = 8,
    [string]$DataDir = "data/preprocessed",
    [switch]$Verbose,
    [switch]$Quiet
)

# Set verbose based on parameters
$VerboseFlag = if ($Verbose -and -not $Quiet) { "-v" } else { "" }

# Build Python arguments
$pythonArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/comprehensive_correlation_analysis.py"
)

if ($Datasets.Count -gt 0) {
    $pythonArgs += "--datasets"
    $pythonArgs += $Datasets
}

if ($Features.Count -gt 0) {
    $pythonArgs += "--features"
    $pythonArgs += $Features
}

if ($CycleLimit -gt 0) {
    $pythonArgs += @("--cycle_limit", $CycleLimit)
}

if ($Smoothing -ne "none") {
    $pythonArgs += @("--smoothing", $Smoothing, "--smoothing_window", $SmoothingWindow)
}

$pythonArgs += @(
    "--output_dir", $OutputDir,
    "--figsize", $Width, $Height,
    "--data_dir", $DataDir
)

if ($VerboseFlag) {
    $pythonArgs += $VerboseFlag
}

# Display parameters
Write-Host "Comprehensive Correlation Analysis" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host "Datasets: $(if ($Datasets.Count -gt 0) { $Datasets -join ', ' } else { 'All available' })"
Write-Host "Features: $(if ($Features.Count -gt 0) { $Features -join ', ' } else { 'All available' })"
Write-Host "Cycle Limit: $(if ($CycleLimit -gt 0) { $CycleLimit } else { 'All cycles' })"
Write-Host "Smoothing: $Smoothing"
if ($Smoothing -ne "none") {
    Write-Host "Smoothing Window: $SmoothingWindow"
}
Write-Host "Output Directory: $OutputDir"
Write-Host "Figure Size: ${Width}x${Height}"
Write-Host "Data Directory: $DataDir"
Write-Host ""

# Run the Python script
Write-Host "Running comprehensive correlation analysis..." -ForegroundColor Yellow
Write-Host "Command: python $($pythonArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    python @pythonArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Analysis completed successfully!" -ForegroundColor Green
        Write-Host "Check the output directory: $OutputDir" -ForegroundColor Cyan
    }
    else {
        Write-Host ""
        Write-Host "Analysis failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host ""
    Write-Host "Error running analysis: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Script completed." -ForegroundColor Green
