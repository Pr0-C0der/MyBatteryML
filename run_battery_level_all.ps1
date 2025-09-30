# Hardcoded runs for battery-level RUL with cycle_limit=100 for all datasets
# Mixed pools
python -m batteryml.training.train_rul_windows --dataset CRUH  --data_path data/preprocessed/CALCE data/preprocessed/RWTH data/preprocessed/UL_PUR data/preprocessed/HNEI --output_dir test_results/CRUH  --battery_level --cycle_limit 100 --features default
python -m batteryml.training.train_rul_windows --dataset CRUSH --data_path data/preprocessed/CALCE data/preprocessed/RWTH data/preprocessed/UL_PUR data/preprocessed/HNEI data/preprocessed/SNL --output_dir test_results/CRUSH --battery_level --cycle_limit 100 --features default
python -m batteryml.training.train_rul_windows --dataset MIX100 --data_path data/preprocessed/HUST data/preprocessed/MATR data/preprocessed/RWTH data/preprocessed/CALCE data/preprocessed/UL_PUR data/preprocessed/HNEI --output_dir test_results/MIX100 --battery_level --cycle_limit 100 --features default

# MATR splits and CLO (all use data/preprocessed/MATR)
python -m batteryml.training.train_rul_windows --dataset MATR1 --data_path data/preprocessed/MATR --output_dir test_results/MATR1 --battery_level --cycle_limit 100 --features default
python -m batteryml.training.train_rul_windows --dataset MATR2 --data_path data/preprocessed/MATR --output_dir test_results/MATR2 --battery_level --cycle_limit 100 --features default
python -m batteryml.training.train_rul_windows --dataset CLO   --data_path data/preprocessed/MATR --output_dir test_results/CLO   --battery_level --cycle_limit 100 --features default

# Single-source datasets
python -m batteryml.training.train_rul_windows --dataset HUST  --data_path data/preprocessed/HUST  --output_dir test_results/HUST  --battery_level --cycle_limit 100 --features default
python -m batteryml.training.train_rul_windows --dataset SNL   --data_path data/preprocessed/SNL   --output_dir test_results/SNL   --battery_level --cycle_limit 100 --features default



