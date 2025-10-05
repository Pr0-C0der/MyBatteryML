# Example: Direct Statistical Features Training
# This script demonstrates how to train with specific statistical features like mean_cycle_length and median_charge_cycle_length

Write-Host "Direct Statistical Features Training Example" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Example 1: Train with only mean_cycle_length and median_charge_cycle_length" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Cyan
Write-Host "python batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py --data_path 'data/processed/LFP' --direct_statistical_features mean_cycle_length median_charge_cycle_length --verbose" -ForegroundColor Gray
Write-Host ""

Write-Host "Example 2: Train with specific statistical features from different base features" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Cyan
Write-Host "python batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py --data_path 'data/processed/LFP' --direct_statistical_features mean_cycle_length std_charge_cycle_length skewness_avg_c_rate --verbose" -ForegroundColor Gray
Write-Host ""

Write-Host "Example 3: Train with only mean features from multiple base features" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Cyan
Write-Host "python batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py --data_path 'data/processed/LFP' --direct_statistical_features mean_cycle_length mean_charge_cycle_length mean_avg_c_rate --verbose" -ForegroundColor Gray
Write-Host ""

Write-Host "Example 4: Train with cycle limit and specific features" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor Cyan
Write-Host "python batteryml/chemistry_data_analysis/statistical_analysis/chemistry_statistical_training.py --data_path 'data/processed/LFP' --cycle_limit 100 --direct_statistical_features mean_cycle_length median_charge_cycle_length --verbose" -ForegroundColor Gray
Write-Host ""

Write-Host "Available Statistical Measures:" -ForegroundColor Yellow
Write-Host "=============================" -ForegroundColor Yellow
$Measures = @("mean", "std", "min", "max", "median", "q25", "q75", "skewness", "kurtosis")
$Measures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }

Write-Host ""
Write-Host "Available Base Features:" -ForegroundColor Yellow
Write-Host "=======================" -ForegroundColor Yellow
$BaseFeatures = @(
    "avg_c_rate", "max_charge_capacity", "avg_discharge_capacity", "avg_charge_capacity",
    "charge_cycle_length", "discharge_cycle_length", "cycle_length",
    "power_during_charge_cycle", "power_during_discharge_cycle",
    "avg_charge_c_rate", "avg_discharge_c_rate", "charge_to_discharge_time_ratio",
    "avg_voltage", "avg_current"
)
$BaseFeatures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Cyan }

Write-Host ""
Write-Host "Example Statistical Feature Names:" -ForegroundColor Yellow
Write-Host "=================================" -ForegroundColor Yellow
Write-Host "  - mean_cycle_length" -ForegroundColor Green
Write-Host "  - median_charge_cycle_length" -ForegroundColor Green
Write-Host "  - std_avg_c_rate" -ForegroundColor Green
Write-Host "  - skewness_cycle_length" -ForegroundColor Green
Write-Host "  - kurtosis_charge_cycle_length" -ForegroundColor Green
Write-Host "  - q25_avg_voltage" -ForegroundColor Green
Write-Host "  - q75_avg_current" -ForegroundColor Green

Write-Host ""
Write-Host "Note: Direct statistical features override manual_features if both are provided!" -ForegroundColor Magenta
