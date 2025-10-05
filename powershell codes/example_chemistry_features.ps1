# Example Chemistry Statistical Training with Feature Expansion
# This script demonstrates how base features are expanded into statistical measures

Write-Host "Chemistry Statistical Training - Feature Expansion Example" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Base Features (what you specify):" -ForegroundColor Yellow
$BaseFeatures = @("charge_cycle_length", "avg_c_rate", "cycle_length")
$BaseFeatures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }

Write-Host ""
Write-Host "Statistical Measures:" -ForegroundColor Yellow
$StatisticalMeasures = @("mean", "std", "min", "max", "median", "q25", "q75", "skewness", "kurtosis")
$StatisticalMeasures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }

Write-Host ""
Write-Host "Expanded Features (what gets used for training):" -ForegroundColor Yellow
foreach ($BaseFeature in $BaseFeatures) {
    foreach ($Measure in $StatisticalMeasures) {
        Write-Host "  - ${BaseFeature}_${Measure}" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Total Features: $($BaseFeatures.Count * $StatisticalMeasures.Count)" -ForegroundColor Magenta
Write-Host ""

Write-Host "Example Usage:" -ForegroundColor Yellow
Write-Host "=============" -ForegroundColor Yellow
Write-Host ""

Write-Host "1. Default features (14 base features):" -ForegroundColor Cyan
Write-Host "   .\powershell codes\run_chemistry_statistical_training.ps1 -ChemistryPath 'data/processed/LFP'" -ForegroundColor Gray
Write-Host "   -> Generates 14 * 9 = 126 statistical features" -ForegroundColor Gray
Write-Host ""

Write-Host "2. Custom base features (3 base features):" -ForegroundColor Cyan
Write-Host "   .\powershell codes\run_chemistry_statistical_training.ps1 -ChemistryPath 'data/processed/LFP' -ManualFeatures @('charge_cycle_length', 'avg_c_rate', 'cycle_length')" -ForegroundColor Gray
Write-Host "   -> Generates 3 * 9 = 27 statistical features" -ForegroundColor Gray
Write-Host ""

Write-Host "3. Single feature (1 base feature):" -ForegroundColor Cyan
Write-Host "   .\powershell codes\run_chemistry_statistical_training.ps1 -ChemistryPath 'data/processed/LFP' -ManualFeatures @('charge_cycle_length')" -ForegroundColor Gray
Write-Host "   -> Generates 1 * 9 = 9 statistical features" -ForegroundColor Gray
Write-Host ""

Write-Host "4. Quick test:" -ForegroundColor Cyan
Write-Host "   .\powershell codes\quick_chemistry_statistical_training_test.ps1" -ForegroundColor Gray
Write-Host ""

Write-Host "Available Base Features:" -ForegroundColor Yellow
Write-Host "=======================" -ForegroundColor Yellow
$AvailableFeatures = @(
    "avg_c_rate", "max_charge_capacity", "avg_discharge_capacity", "avg_charge_capacity",
    "charge_cycle_length", "discharge_cycle_length", "cycle_length",
    "power_during_charge_cycle", "power_during_discharge_cycle",
    "avg_charge_c_rate", "avg_discharge_c_rate", "charge_to_discharge_time_ratio",
    "avg_voltage", "avg_current"
)

$AvailableFeatures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }

Write-Host ""
Write-Host "Note: Each base feature will be expanded with all 9 statistical measures!" -ForegroundColor Magenta
