# # For Correlation Analysis
# python -m batteryml.data_analysis.correlation_mod --data_path data/preprocessed/MATR --output_dir correlation_analysis_mod --verbose

# # For Cycle Plotting
python -m batteryml.data_analysis.cycle_plotter_mod --data_path data/preprocessed/MATR --output_dir cycle_plots_mod --cycle_gap 100 --verbose

# # For Misc Plots
# python -m batteryml.data_analysis.misc_plots --data_path data/preprocessed/MATR --output_dir misc_plots_out --plot all --twin_cycles 0,last --first_last_features voltage current --verbose