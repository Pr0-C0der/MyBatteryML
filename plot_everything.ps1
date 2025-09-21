# Run ALL plots for ALL datasets (Windows PowerShell)
# - Regular cycle plots (incl. average C-rate, power, energy, coulombic efficiency)
# - Split charge/discharge plots + correlations (keeps full-battery RUL)
# - Full correlation analysis
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_all_plots.ps1

# Base data path (use preprocessed if available; change to data/processed if needed)
$BASE_DATA_PATH = "data/preprocessed"

# 1) Regular cycle plots for all datasets
python batteryml/data_analysis/run_cycle_plots.py --all --data_path $BASE_DATA_PATH --output_dir data_analysis_results

# 2) Split charge/discharge plots and correlations for all datasets
python batteryml/data_analysis/run_split_charge_discharge.py --all --base_data_path $BASE_DATA_PATH --keep_full_rul --output_dir data_analysis_split_charge_discharge

# 3) Full correlation analysis for all datasets
python batteryml/data_analysis/run_correlation_analysis.py --all --data_path $BASE_DATA_PATH --output_dir data_analysis_results