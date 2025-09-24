# MATR, default feature set, window size 100
python -m batteryml.training.train_rul_windows --dataset MATR --data_path data/preprocessed/MATR --output_dir rul_windows --window_size 100

# CALCE, custom features and window size 60
python -m batteryml.training.train_rul_windows --dataset CALCE --data_path data/preprocessed/CALCE --output_dir rul_windows --window_size 60 --features avg_c_rate max_discharge_capacity charge_cycle_length peak_cc_length cycle_length

# Use all available features
python -m batteryml.training.train_rul_windows --dataset MATR --data_path data/preprocessed/MATR --window_size 80 --features all