# PowerShell script to run incremental capacity analysis

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
    [switch]$ChargeDischarge,
    [int]$WindowSize = 5,
    [switch]$NoSmoothing,
    [switch]$Verbose
)

# Set default output path if not provided
if ($Output -eq "") {
    if ($All) {
        if ($ChargeDischarge) {
            $Output = "incremental_capacity_charge_discharge_${DatasetName}"
        }
        else {
            $Output = "incremental_capacity_analysis_${DatasetName}"
        }
    }
    else {
        if ($ChargeDischarge) {
            $Output = "incremental_capacity_charge_discharge_${DatasetName}_${BatteryId}.png"
        }
        else {
            $Output = "incremental_capacity_analysis_${DatasetName}_${BatteryId}.png"
        }
    }
}

Write-Host "Running Incremental Capacity Analysis (dQ/dV vs V)..." -ForegroundColor Green
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

if ($ChargeDischarge) {
    Write-Host "Mode: Charge and Discharge phases separately" -ForegroundColor Magenta
}
else {
    Write-Host "Mode: Combined analysis" -ForegroundColor Magenta
}

if ($NoSmoothing) {
    Write-Host "Smoothing: Disabled" -ForegroundColor Yellow
}
else {
    Write-Host "Smoothing: Enabled (window size: $WindowSize)" -ForegroundColor Yellow
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
    "batteryml/chemistry_data_analysis/statistical_analysis/incremental_capacity_analysis.py",
    $DatasetName,
    "--cycle_limit", $CycleLimit,
    "--data_dir", $DataDir,
    "--output", $Output,
    "--figsize", $Width, $Height,
    "--window_size", $WindowSize
)

if ($BatteryId -ne "") {
    $pythonArgs += "--battery_id", $BatteryId
}

if ($All) {
    $pythonArgs += "--all"
}

if ($ChargeDischarge) {
    $pythonArgs += "--charge_discharge"
}

if ($NoSmoothing) {
    $pythonArgs += "--no_smoothing"
}

if ($Verbose) {
    $pythonArgs += "--verbose"
}

try {
    python @pythonArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully created incremental capacity analysis plot(s)!" -ForegroundColor Green
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
