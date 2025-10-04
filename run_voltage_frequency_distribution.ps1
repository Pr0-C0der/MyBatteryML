# PowerShell script to run voltage frequency distribution analysis

param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    
    [int]$CycleLimit = 10,
    [string]$BatteryId = "",
    [string]$DataDir = "data",
    [string]$Output = "",
    [int]$Width = 12,
    [int]$Height = 8,
    [switch]$All,
    [switch]$Kde,
    [switch]$VerboseOutput
)

# Set default output path if not provided
if ($Output -eq "") {
    if ($All) {
        if ($Kde) {
            $Output = "voltage_kde_distribution_${DatasetName}"
        }
        else {
            $Output = "voltage_frequency_distribution_${DatasetName}"
        }
    }
    else {
        if ($Kde) {
            $Output = "voltage_kde_distribution_${DatasetName}_${BatteryId}.png"
        }
        else {
            $Output = "voltage_frequency_distribution_${DatasetName}_${BatteryId}.png"
        }
    }
}

Write-Host "Running Voltage Frequency Distribution Analysis..." -ForegroundColor Green
Write-Host "Dataset: $DatasetName" -ForegroundColor Cyan
Write-Host "Cycle Limit: $CycleLimit" -ForegroundColor Cyan
Write-Host "Output: $Output" -ForegroundColor Cyan

if ($All) {
    Write-Host "Mode: All batteries in dataset (separate PNG files)" -ForegroundColor Yellow
}
else {
    if ($BatteryId -ne "") {
        Write-Host "Battery ID: $BatteryId" -ForegroundColor Cyan
    }
    else {
        Write-Host "Battery ID: Auto-select first available" -ForegroundColor Yellow
    }
}

if ($Kde) {
    Write-Host "Method: KDE (Kernel Density Estimation)" -ForegroundColor Magenta
}
else {
    Write-Host "Method: Histogram" -ForegroundColor Magenta
}

# Check if data directory exists
if (-not (Test-Path $DataDir)) {
    Write-Host "Error: Data directory not found at $DataDir" -ForegroundColor Red
    exit 1
}

# Check if dataset directory exists
$datasetPath = Join-Path $DataDir $DatasetName
if (-not (Test-Path $datasetPath)) {
    Write-Host "Error: Dataset directory not found at $datasetPath" -ForegroundColor Red
    Write-Host "Available datasets:" -ForegroundColor Yellow
    Get-ChildItem $DataDir -Directory | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Yellow }
    exit 1
}

# Run the Python script
$pythonArgs = @(
    "batteryml/chemistry_data_analysis/statistical_analysis/voltage_frequency_distribution.py",
    $DatasetName,
    "--cycle_limit", $CycleLimit,
    "--data_dir", $DataDir,
    "--output", $Output,
    "--figsize", $Width, $Height
)

if ($BatteryId -ne "") {
    $pythonArgs += "--battery_id", $BatteryId
}

if ($All) {
    $pythonArgs += "--all"
}

if ($Kde) {
    $pythonArgs += "--kde"
}

if ($VerboseOutput) {
    $pythonArgs += "--verbose"
}

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully created voltage frequency distribution plot(s)!" -ForegroundColor Green
        Write-Host "Plot(s) saved to: $Output" -ForegroundColor Cyan
    }
    else {
        Write-Host "Error creating plot(s)" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error running Python script: $_" -ForegroundColor Red
    exit 1
}
