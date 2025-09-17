python batteryml/data_analysis/run_analysis.py --all --data_path data/processed --output_dir data_analysis_results
python batteryml/data_analysis/run_analysis.py --combined-plots CALCE --data_path data/processed/CALCE --output_dir data_analysis_results/CALCE
python batteryml/data_analysis/run_analysis.py --combined-plots HUST --data_path data/processed/HUST --output_dir data_analysis_results/HUST
python batteryml/data_analysis/run_analysis.py --combined-plots MATR --data_path data/processed/MATR --output_dir data_analysis_results/MATR
python batteryml/data_analysis/run_analysis.py --combined-plots SNL --data_path data/processed/SNL --output_dir data_analysis_results/SNL
python batteryml/data_analysis/run_analysis.py --combined-plots HNEI --data_path data/processed/HNEI --output_dir data_analysis_results/HNEI
python batteryml/data_analysis/run_analysis.py --combined-plots RWTH --data_path data/processed/RWTH --output_dir data_analysis_results/RWTH
python batteryml/data_analysis/run_analysis.py --combined-plots UL_PUR --data_path data/processed/UL_PUR --output_dir data_analysis_results/UL_PUR
python batteryml/data_analysis/run_analysis.py --combined-plots OX --data_path data/processed/OX --output_dir data_analysis_results/OX
python batteryml/data_analysis/run_correlation_analysis.py --all --data_path data/processed --output_dir data_analysis_results
python batteryml/data_analysis/run_cycle_plots.py --all --data_path data/processed --output_dir data_analysis_results