# Hardcoded commands to plot feature-vs-cycle graphs for multiple datasets
# Uses default feature set, 30 random batteries, caps 100/200/500, lag_window=5

python -m batteryml.data_analysis.plot_feature_vs_cycle_random --dataset MATR  --data_path data/preprocessed/MATR  --feature default --n 30 --caps 100 200 500 --lag_window 5 --output_dir feature_vs_cycle_random --verbose
python -m batteryml.data_analysis.plot_feature_vs_cycle_random --dataset CALCE --data_path data/preprocessed/CALCE --feature default --n 30 --caps 100 200 500 --lag_window 5 --output_dir feature_vs_cycle_random --verbose
python -m batteryml.data_analysis.plot_feature_vs_cycle_random --dataset HUST  --data_path data/preprocessed/HUST  --feature default --n 30 --caps 100 200 500 --lag_window 5 --output_dir feature_vs_cycle_random --verbose
python -m batteryml.data_analysis.plot_feature_vs_cycle_random --dataset SNL   --data_path data/preprocessed/SNL   --feature default --n 30 --caps 100 200 500 --lag_window 5 --output_dir feature_vs_cycle_random --verbose

# Optional (if these directories exist)
python -m batteryml.data_analysis.plot_feature_vs_cycle_random --dataset RWTH  --data_path data/preprocessed/RWTH  --feature default --n 30 --caps 100 200 500 --lag_window 5 --output_dir feature_vs_cycle_random --verbose
python -m batteryml.data_analysis.plot_feature_vs_cycle_random --dataset UL_PUR --data_path data/preprocessed/UL_PUR --feature default --n 30 --caps 100 200 500 --lag_window 5 --output_dir feature_vs_cycle_random --verbose
python -m batteryml.data_analysis.plot_feature_vs_cycle_random --dataset HNEI  --data_path data/preprocessed/HNEI  --feature default --n 30 --caps 100 200 500 --lag_window 5 --output_dir feature_vs_cycle_random --verbose


