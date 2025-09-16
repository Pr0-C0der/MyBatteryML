@echo off
REM Batch script to run cycle plotting for all datasets
REM Usage: run_cycle_plots.bat [cycle_gap]

set CYCLE_GAP=%1
if "%CYCLE_GAP%"=="" set CYCLE_GAP=100

echo Starting cycle plotting for all datasets with cycle gap: %CYCLE_GAP%
echo.

python -m batteryml.data_analysis.run_cycle_plots --all --data_path data/processed --cycle_gap %CYCLE_GAP%

echo.
echo Cycle plotting completed!
pause
