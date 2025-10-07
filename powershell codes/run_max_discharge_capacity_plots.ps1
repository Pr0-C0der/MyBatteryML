# Max Discharge Capacity Plotting PowerShell Script
# This script plots max discharge capacity vs cycle for each dataset

param(
    [string]$OutputDir = "max_discharge_capacity_plots",
    [switch]$Individual = $false,
    [switch]$Combined = $true,
    [int[]]$FigSize = @(12, 8),
    [string]$Smoothing = "none",
    [int]$Window = 5,
    [switch]$Verbose = $false,
    [switch]$Help = $false
)

# Display help if requested
if ($Help) {
    Write-Host "Max Discharge Capacity Plotting PowerShell Script" -ForegroundColor Green
    Write-Host "=================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\run_max_discharge_capacity_plots.ps1 [parameters]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Cyan
    Write-Host "  -OutputDir <string>    : Output directory for plots [default: max_discharge_capacity_plots]"
    Write-Host "  -Individual            : Create individual plots for each dataset"
    Write-Host "  -Combined              : Create combined plot for all datasets [default: true]"
    Write-Host "  -FigSize <array>       : Figure size (width height) [default: 12 8]"
    Write-Host "  -Smoothing <string>    : Smoothing method (none, ma, median, hms) [default: none]"
    Write-Host "  -Window <int>          : Smoothing window size [default: 5]"
    Write-Host "  -Verbose               : Enable verbose output"
    Write-Host "  -Help                  : Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run_max_discharge_capacity_plots.ps1"
    Write-Host "  .\run_max_discharge_capacity_plots.ps1 -Individual -Combined"
    Write-Host "  .\run_max_discharge_capacity_plots.ps1 -OutputDir 'my_plots' -FigSize @(14, 10) -Verbose"
    Write-Host "  .\run_max_discharge_capacity_plots.ps1 -Smoothing 'hms' -Window 7 -Verbose"
    Write-Host ""
    exit 0
}

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonScript = Join-Path $ProjectRoot "plot_max_discharge_capacity_by_dataset.py"

# Check if Python script exists
if (-not (Test-Path $PythonScript)) {
    Write-Host "❌ Python script not found: $PythonScript" -ForegroundColor Red
    exit 1
}

# Display configuration
Write-Host "Max Discharge Capacity Plotting Configuration" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Output Directory: $OutputDir" -ForegroundColor White
Write-Host "  Individual Plots: $Individual" -ForegroundColor White
Write-Host "  Combined Plot: $Combined" -ForegroundColor White
Write-Host "  Figure Size: $($FigSize -join ' x ')" -ForegroundColor White
Write-Host "  Smoothing: $Smoothing" -ForegroundColor White
Write-Host "  Window: $Window" -ForegroundColor White
Write-Host "  Verbose: $Verbose" -ForegroundColor White
Write-Host ""

# Build command arguments
$CommandArgs = @(
    "python", $PythonScript
    "--output_dir", $OutputDir
    "--figsize"
    $FigSize[0]
    $FigSize[1]
)

if ($Individual) {
    $CommandArgs += "--individual"
}

if ($Combined) {
    $CommandArgs += "--combined"
}

$CommandArgs += "--smoothing"
$CommandArgs += $Smoothing

$CommandArgs += "--window"
$CommandArgs += $Window

if ($Verbose) {
    $CommandArgs += "--verbose"
}

Write-Host "Running command: $($CommandArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location $ProjectRoot

# Run the Python script
try {
    & $CommandArgs[0] $CommandArgs[1..($CommandArgs.Length - 1)]
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Max discharge capacity plotting completed successfully!" -ForegroundColor Green
        
        # Check if output directory exists and show contents
        if (Test-Path $OutputDir) {
            Write-Host ""
            Write-Host "Generated files:" -ForegroundColor Cyan
            Get-ChildItem $OutputDir -Filter "*.png" | ForEach-Object {
                Write-Host "  - $($_.Name)" -ForegroundColor White
            }
        }
    }
    else {
        Write-Host ""
        Write-Host "❌ Script execution failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host ""
    Write-Host "❌ Error running max discharge capacity plotting: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Script completed." -ForegroundColor Green
