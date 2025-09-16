@echo off
REM Batch script to run correlation analysis for all datasets

echo Starting correlation analysis for all datasets
echo.

python -m batteryml.data_analysis.run_correlation_analysis --all --data_path data/processed

echo.
echo Correlation analysis completed!
pause
