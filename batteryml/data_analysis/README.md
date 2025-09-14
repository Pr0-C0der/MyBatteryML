# BatteryML Data Analysis Module

This module provides comprehensive data analysis tools for battery datasets, including individual battery analysis, dataset-level statistics, and visualization utilities.

## Features

- **Individual Battery Analysis**: Analyze each battery pickle file separately to avoid memory issues
- **Dataset-Level Statistics**: Comprehensive statistics across all batteries in a dataset
- **Feature Analysis**: Extract and analyze all features with min, max, mean, median, std, quartiles
- **Visualization**: Create comprehensive plots and reports
- **Memory Efficient**: Process batteries one at a time to handle large datasets
- **Error Handling**: Robust error handling to prevent runtime errors

## Quick Start

### 1. Analyze a Single Dataset

```bash
# Analyze a specific dataset
python batteryml/data_analysis/analyze_datasets.py \
    --dataset_path data/processed/MATR \
    --output_dir results/MATR_analysis
```

### 2. Analyze All Datasets

```bash
# Analyze all datasets in a directory
python batteryml/data_analysis/analyze_datasets.py \
    --all_datasets \
    --data_root data/processed \
    --output_dir results
```

### 3. Quick Test (Limited Batteries)

```bash
# Analyze only first 10 batteries per dataset (for testing)
python batteryml/data_analysis/analyze_datasets.py \
    --all_datasets \
    --data_root data/processed \
    --output_dir results \
    --max_batteries 10
```

### 4. Fast Analysis (No Plots)

```bash
# Skip visualization for faster analysis
python batteryml/data_analysis/analyze_datasets.py \
    --all_datasets \
    --data_root data/processed \
    --output_dir results \
    --no_plots
```

## Analysis Output

The analysis generates:

1. **Console Output**: 
   - Total number of batteries per dataset
   - Feature statistics (min, max, mean, median, std, etc.)
   - Dataset summary statistics
   - Chemistry distribution
   - Cycle life analysis

2. **Saved Files**:
   - `{dataset_name}_summary.json`: Complete analysis results
   - `{dataset_name}_features.csv`: Feature statistics table
   - `{dataset_name}_overview.png`: Dataset overview plots
   - `{dataset_name}_features.png`: Feature distribution plots
   - `{dataset_name}_cycle_life.png`: Cycle life analysis plots
   - `{dataset_name}_capacity.png`: Capacity analysis plots

## Programmatic Usage

```python
from batteryml.data_analysis import DatasetAnalyzer, AnalysisVisualizer

# Analyze a dataset
analyzer = DatasetAnalyzer("data/processed/MATR")
summary_stats = analyzer.analyze_dataset()

# Print results
analyzer.print_dataset_summary()
analyzer.print_feature_summary()

# Get feature statistics table
features_df = analyzer.get_feature_summary_table()
print(features_df)

# Create visualizations
visualizer = AnalysisVisualizer()
visualizer.create_comprehensive_report(analyzer, "output_dir")
```

## Key Statistics Provided

### Dataset Level
- Total number of batteries
- Successfully analyzed batteries
- Average cycle life across all batteries
- Chemistry distribution
- Capacity statistics

### Feature Level
For each feature found in the dataset:
- **Count**: Number of non-null values
- **Min**: Minimum value
- **Max**: Maximum value
- **Mean**: Average value
- **Median**: Median value
- **Std**: Standard deviation
- **Q25**: 25th percentile
- **Q75**: 75th percentile

### Battery Level
- Cell ID and basic metadata
- Total cycles and cycle life
- Capacity statistics (discharge, charge, retention)
- Voltage, current, and temperature statistics
- Degradation analysis

## Error Handling

The analysis includes comprehensive error handling:
- Safe loading of pickle files (continues if one fails)
- NaN value handling in statistics
- Memory-efficient processing
- Progress indicators for long-running analyses
- Detailed error messages

## Requirements

Install additional requirements for data analysis:

```bash
pip install -r batteryml/data_analysis/requirements.txt
```

## Examples

### Example 1: Basic Dataset Analysis

```bash
python batteryml/data_analysis/analyze_datasets.py \
    --dataset_path data/processed/MATR \
    --output_dir results/MATR_analysis
```

Output:
```
================================================================================
ANALYZING DATASET: MATR
================================================================================
Found 180 battery files
Analyzing batteries: 100%|████████████| 180/180 [00:45<00:00,  3.98it/s]
Successfully analyzed 180 batteries

================================================================================
DATASET ANALYSIS SUMMARY: MATR
================================================================================
Total Batteries: 180
Successfully Analyzed: 180

--- DATASET STATISTICS ---
Total Cycles: 148320
Average Cycle Life: 823.45
Median Cycle Life: 750.00
Cycle Life Std: 368.12
Min Cycle Life: 150.00
Max Cycle Life: 2000.00
Average Nominal Capacity: 1.1000 Ah

--- CHEMISTRY DISTRIBUTION ---
LFP/graphite: 180 batteries

--- CYCLE LIFE DISTRIBUTION ---
Mean: 823.45
Median: 750.00
Std: 368.12
Min: 150.00
Max: 2000.00
Q25: 500.00
Q75: 1100.00
```

### Example 2: Feature Analysis

The analysis automatically extracts and analyzes all features found in the battery data, providing comprehensive statistics for each feature type.

## Troubleshooting

1. **Memory Issues**: Use `--max_batteries` to limit the number of batteries analyzed
2. **No Files Found**: Ensure the dataset path contains `.pkl` files
3. **Import Errors**: Make sure you're running from the correct directory and have installed requirements
4. **Plot Issues**: Use `--no_plots` if visualization libraries are not available

## Contributing

To add new analysis features:
1. Extend the `BatteryAnalyzer` class for individual battery analysis
2. Extend the `DatasetAnalyzer` class for dataset-level analysis
3. Add new visualization methods to `AnalysisVisualizer`
4. Update the main script to include new functionality
