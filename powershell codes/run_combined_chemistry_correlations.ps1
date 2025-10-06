# Combined Chemistry Correlation Analysis PowerShell Script
# This script creates combined correlation boxplots for multiple chemistries

param(
    [string[]]$ChemistryDirs = @(),
    [string]$OutputDir = "combined_chemistry_correlations",
    [string]$DatasetHint = "",
    [int]$CycleLimit = 0,
    [string]$Smoothing = "none",
    [int]$MaWindow = 5,
    [switch]$Verbose = $false,
    [switch]$Help = $false
)

# Display help if requested
if ($Help) {
    Write-Host "Combined Chemistry Correlation Analysis" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\run_combined_chemistry_correlations.ps1 [parameters]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Cyan
    Write-Host "  -ChemistryDirs <array>  : List of chemistry directory paths [required]"
    Write-Host "  -OutputDir <string>     : Output directory [default: combined_chemistry_correlations]"
    Write-Host "  -DatasetHint <string>   : Optional dataset name hint"
    Write-Host "  -CycleLimit <int>       : Limit analysis to first N cycles [default: 0 (all cycles)]"
    Write-Host "  -Smoothing <string>     : Smoothing method (none, ma, median, hms) [default: none]"
    Write-Host "  -MaWindow <int>         : Window size for smoothing [default: 5]"
    Write-Host "  -Verbose                : Enable verbose logging"
    Write-Host "  -Help                   : Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run_combined_chemistry_correlations.ps1 -ChemistryDirs @('data_chemistries/lfp', 'data_chemistries/nmc', 'data_chemistries/nca')"
    Write-Host "  .\run_combined_chemistry_correlations.ps1 -ChemistryDirs @('data_chemistries/lfp', 'data_chemistries/nmc') -CycleLimit 100 -Smoothing ma"
    Write-Host "  .\run_combined_chemistry_correlations.ps1 -ChemistryDirs @('data_chemistries/lfp', 'data_chemistries/nmc', 'data_chemistries/nca', 'data_chemistries/lco', 'data_chemistries/mixed_nmc_lco') -Verbose"
    exit 0
}

# Validate parameters
if ($ChemistryDirs.Count -eq 0) {
    Write-Host "Error: At least one chemistry directory must be specified!" -ForegroundColor Red
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
$ValidDirs = @()
foreach ($Dir in $ChemistryDirs) {
    if (Test-Path $Dir) {
        $ValidDirs += $Dir
        Write-Host "✓ Found chemistry directory: $Dir" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Chemistry directory not found: $Dir" -ForegroundColor Red
    }
}

if ($ValidDirs.Count -eq 0) {
    Write-Host "Error: No valid chemistry directories found!" -ForegroundColor Red
    exit 1
}

if ($ValidDirs.Count -lt $ChemistryDirs.Count) {
    Write-Host "Warning: Some chemistry directories were not found. Proceeding with $($ValidDirs.Count) valid directories." -ForegroundColor Yellow
}

# Check if Python script exists
$ScriptPath = "plot_combined_chemistry_correlations.py"
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
Write-Host "  Chemistry Directories: $($ValidDirs -join ', ')" -ForegroundColor White
Write-Host "  Output Directory: $OutputDir" -ForegroundColor White
Write-Host "  Dataset Hint: $(if ($DatasetHint) { $DatasetHint } else { 'Auto-detect' })" -ForegroundColor White
Write-Host "  Cycle Limit: $(if ($CycleLimit -gt 0) { $CycleLimit } else { 'All cycles' })" -ForegroundColor White
Write-Host "  Smoothing: $Smoothing" -ForegroundColor White
Write-Host "  MA Window: $MaWindow" -ForegroundColor White
Write-Host "  Verbose: $Verbose" -ForegroundColor White
Write-Host ""

# Build command arguments
$CommandArgs = @(
    "python", $ScriptPath
    "--chemistry_dirs"
) + $ValidDirs + @(
    "--output_dir", $OutputDir
    "--smoothing", $Smoothing
    "--ma_window", $MaWindow
)

if ($DatasetHint) {
    $CommandArgs += @("--dataset_hint", $DatasetHint)
}

if ($CycleLimit -gt 0) {
    $CommandArgs += @("--cycle_limit", $CycleLimit)
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
        Write-Host "✅ Combined correlation analysis completed successfully!" -ForegroundColor Green
        Write-Host "Duration: $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan
        Write-Host "Results saved to: $OutputDir" -ForegroundColor Yellow
        
        # Show output files
        if (Test-Path $OutputDir) {
            Write-Host ""
            Write-Host "Generated files:" -ForegroundColor Cyan
            Get-ChildItem -Path $OutputDir -Filter "*.png" | ForEach-Object {
                Write-Host "  - $($_.Name)" -ForegroundColor White
            }
        }
    }
    else {
        Write-Host "❌ Combined correlation analysis failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "❌ Error running combined correlation analysis`: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Script execution completed!" -ForegroundColor Green
