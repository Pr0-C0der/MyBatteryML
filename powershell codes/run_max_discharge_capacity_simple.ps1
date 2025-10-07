# Simple Max Discharge Capacity Plotting Script
# Run this from the project root directory

param(
    [string]$Smoothing = "none",
    [int]$Window = 5,
    [switch]$Verbose = $false
)

Write-Host "Max Discharge Capacity Plotting" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green
Write-Host ""

# Build command
$Command = "python plot_max_discharge_capacity_by_dataset.py --smoothing $Smoothing --window $Window"

if ($Verbose) {
    $Command += " --verbose"
}

Write-Host "Running: $Command" -ForegroundColor Cyan
Write-Host ""

# Run the command
try {
    Invoke-Expression $Command
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Plotting completed successfully!" -ForegroundColor Green
        Write-Host "Check the 'max_discharge_capacity_plots' folder for results." -ForegroundColor White
    }
    else {
        Write-Host ""
        Write-Host "❌ Script failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host ""
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
