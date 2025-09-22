# MATR
python batteryml/data_analysis/run_cycle_plots.py --data_path data/preprocessed/MATR --output_dir data_analysis_results/MATR

# CALCE
python batteryml/data_analysis/run_cycle_plots.py --data_path data/preprocessed/CALCE --output_dir data_analysis_results/CALCE


# MATR
python batteryml/data_analysis/run_split_charge_discharge.py --data_path data/preprocessed/MATR --output_dir data_analysis_split_charge_discharge/MATR --keep_full_rul

# CALCE
python batteryml/data_analysis/run_split_charge_discharge.py --data_path data/preprocessed/CALCE --output_dir data_analysis_split_charge_discharge/CALCE --keep_full_rul