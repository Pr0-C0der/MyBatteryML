@echo off
REM Battery Data Analysis Batch Script for Windows
REM This script provides an easy way to run battery data analysis

echo Battery Data Analysis Tool
echo ==========================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if processed data exists
if not exist "data\processed" (
    echo Error: Processed data directory not found
    echo Please ensure you have processed data in data\processed\
    echo You can process data using: batteryml preprocess ^<dataset^> ^<raw_path^> ^<processed_path^>
    pause
    exit /b 1
)

REM Install requirements if needed
echo Checking dependencies...
pip install -r batteryml\data_analysis\requirements.txt

REM Run the analysis
echo.
echo Starting analysis...
python analyze_datasets.py

echo.
echo Analysis complete!
echo Check the analysis_output directory for results
pause
