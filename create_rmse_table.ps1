# PowerShell script to create RMSE table PNG from CSV files

param(
    [string]$CsvPath = "chemistry_training_results/RMSE.csv",
    [string]$OutputPath = "",
    [switch]$Verbose
)

# Set default output path if not provided
if ($OutputPath -eq "") {
    $OutputPath = $CsvPath -replace "\.csv$", ".png"
}

# Check if CSV file exists
if (-not (Test-Path $CsvPath)) {
    Write-Host "Error: CSV file not found at $CsvPath" -ForegroundColor Red
    Write-Host "Please run chemistry training first to generate RMSE.csv" -ForegroundColor Yellow
    exit 1
}

Write-Host "Creating RMSE table PNG..." -ForegroundColor Green
Write-Host "Input CSV: $CsvPath" -ForegroundColor Cyan
Write-Host "Output PNG: $OutputPath" -ForegroundColor Cyan

# Run the Python script
$pythonArgs = @("create_rmse_table.py", $CsvPath)
if ($OutputPath -ne "") {
    $pythonArgs += "--output", $OutputPath
}
if ($Verbose) {
    $pythonArgs += "--verbose"
}

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully created RMSE table PNG!" -ForegroundColor Green
    }
    else {
        Write-Host "Error creating PNG table" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error running Python script: $_" -ForegroundColor Red
    exit 1
}
