# PowerShell script to run normalized charge capacity vs voltage analysis

param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetName,
    
    [int]$CycleLimit = 10,
    [string]$BatteryId = "",
    [string]$DataDir = "data",
    [string]$Output = "",
    [int]$Width = 12,
    [int]$Height = 8,
    [switch]$MultipleBatteries,
    [int]$MaxBatteries = 3,
    [switch]$All,
    [switch]$VerboseOutput
)

# Set default output path if not provided
if ($Output -eq "") {
    if ($All) {
        $Output = "normalized_charge_capacity_voltage_all_${DatasetName}"
    }
    elseif ($MultipleBatteries) {
        $Output = "normalized_charge_capacity_voltage_multiple_${DatasetName}.png"
    }
    else {
        $Output = "normalized_charge_capacity_voltage_${DatasetName}.png"
    }
}

Write-Host "Running Normalized Charge Capacity vs Voltage Analysis..." -ForegroundColor Green
Write-Host "Dataset: $DatasetName" -ForegroundColor Cyan
Write-Host "Cycle Limit: $CycleLimit" -ForegroundColor Cyan
Write-Host "Output: $Output" -ForegroundColor Cyan

if ($All) {
    Write-Host "Mode: All batteries in dataset (separate PNG files)" -ForegroundColor Yellow
}
elseif ($MultipleBatteries) {
    Write-Host "Mode: Multiple batteries (max: $MaxBatteries)" -ForegroundColor Yellow
}
else {
    if ($BatteryId -ne "") {
        Write-Host "Battery ID: $BatteryId" -ForegroundColor Cyan
    }
    else {
        Write-Host "Battery ID: Auto-select first available" -ForegroundColor Yellow
    }
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
    "batteryml/chemistry_data_analysis/statistical_analysis/normalized_charge_capacity_voltage.py",
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
elseif ($MultipleBatteries) {
    $pythonArgs += "--multiple_batteries"
    $pythonArgs += "--max_batteries", $MaxBatteries
}

if ($VerboseOutput) {
    $pythonArgs += "--verbose"
}

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully created normalized charge capacity vs voltage plot!" -ForegroundColor Green
        Write-Host "Plot saved to: $Output" -ForegroundColor Cyan
    }
    else {
        Write-Host "Error creating plot" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error running Python script: $_" -ForegroundColor Red
    exit 1
}
