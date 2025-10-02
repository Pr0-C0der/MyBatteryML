# Hardcoded runs for chemistry-specific RUL training (both battery-level and cycle-level)
# This script trains models for each chemistry using chemistry-specific feature extractors

# LFP chemistry (MATR + SNL cells) - 70% of MATR cells + 70% of SNL cells for training
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lfp --output_dir chemistry_training_results --cycle_limit 100 --window_size 10 --verbose

# LCO chemistry (CALCE cells) - 70% of CALCE cells for training
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lco --output_dir chemistry_training_results --cycle_limit 100 --window_size 10 --verbose

# NCA chemistry (UL_PUR + SNL cells) - 70% of UL_PUR cells + 70% of SNL cells for training
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nca --output_dir chemistry_training_results --cycle_limit 100 --window_size 10 --verbose

# NMC chemistry (SNL + RWTH cells) - 70% of SNL cells + 70% of RWTH cells for training
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nmc --output_dir chemistry_training_results --cycle_limit 100 --window_size 10 --verbose

# Mixed NMC-LCO chemistry (HNEI cells) - 70% of HNEI cells for training
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/mixed_nmc_lco --output_dir chemistry_training_results --cycle_limit 100 --window_size 10 --verbose

# Additional runs with different cycle limits for comparison
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lfp --output_dir chemistry_training_results_50 --cycle_limit 50 --window_size 10 --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lfp --output_dir chemistry_training_results_200 --cycle_limit 200 --window_size 10 --verbose

python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lco --output_dir chemistry_training_results_50 --cycle_limit 50 --window_size 10 --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lco --output_dir chemistry_training_results_200 --cycle_limit 200 --window_size 10 --verbose

python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nca --output_dir chemistry_training_results_50 --cycle_limit 50 --window_size 10 --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nca --output_dir chemistry_training_results_200 --cycle_limit 200 --window_size 10 --verbose

python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nmc --output_dir chemistry_training_results_50 --cycle_limit 50 --window_size 10 --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nmc --output_dir chemistry_training_results_200 --cycle_limit 200 --window_size 10 --verbose

python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/mixed_nmc_lco --output_dir chemistry_training_results_50 --cycle_limit 50 --window_size 10 --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/mixed_nmc_lco --output_dir chemistry_training_results_200 --cycle_limit 200 --window_size 10 --verbose

# Battery-level only training
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lfp --output_dir chemistry_training_results_battery_only --cycle_limit 100 --battery_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lco --output_dir chemistry_training_results_battery_only --cycle_limit 100 --battery_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nca --output_dir chemistry_training_results_battery_only --cycle_limit 100 --battery_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nmc --output_dir chemistry_training_results_battery_only --cycle_limit 100 --battery_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/mixed_nmc_lco --output_dir chemistry_training_results_battery_only --cycle_limit 100 --battery_level_only --verbose

# Cycle-level only training
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lfp --output_dir chemistry_training_results_cycle_only --cycle_limit 100 --window_size 10 --cycle_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lco --output_dir chemistry_training_results_cycle_only --cycle_limit 100 --window_size 10 --cycle_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nca --output_dir chemistry_training_results_cycle_only --cycle_limit 100 --window_size 10 --cycle_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nmc --output_dir chemistry_training_results_cycle_only --cycle_limit 100 --window_size 10 --cycle_level_only --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/mixed_nmc_lco --output_dir chemistry_training_results_cycle_only --cycle_limit 100 --window_size 10 --cycle_level_only --verbose

# With smoothing options
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lfp --output_dir chemistry_training_results_hms --cycle_limit 100 --window_size 10 --smoothing hms --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/lco --output_dir chemistry_training_results_hms --cycle_limit 100 --window_size 10 --smoothing hms --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nca --output_dir chemistry_training_results_hms --cycle_limit 100 --window_size 10 --smoothing hms --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/nmc --output_dir chemistry_training_results_hms --cycle_limit 100 --window_size 10 --smoothing hms --verbose
python -m batteryml.chemistry_data_analysis.chemistry_training --data_path data_chemistries/mixed_nmc_lco --output_dir chemistry_training_results_hms --cycle_limit 100 --window_size 10 --smoothing hms --verbose

Write-Host "Chemistry-specific RUL training completed for all chemistries!"
Write-Host "Check the following output directories:"
Write-Host "- chemistry_training_results/ (standard training)"
Write-Host "- chemistry_training_results_50/ (50 cycle limit)"
Write-Host "- chemistry_training_results_200/ (200 cycle limit)"
Write-Host "- chemistry_training_results_battery_only/ (battery-level only)"
Write-Host "- chemistry_training_results_cycle_only/ (cycle-level only)"
Write-Host "- chemistry_training_results_hms/ (with HMS smoothing)"
