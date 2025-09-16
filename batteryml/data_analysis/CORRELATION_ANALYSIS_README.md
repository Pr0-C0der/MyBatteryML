# Correlation Analysis for Battery Data

This module provides functionality to analyze correlations between battery features and Remaining Useful Life (RUL) by creating cycle-feature matrices and correlation heatmaps.

## Features

- **Dynamic Feature Detection**: Automatically detects available features for each dataset
- **Cycle-Feature Matrix**: Creates matrices with cycles as rows and features as columns
- **RUL Integration**: Calculates RUL for each cycle using the label annotator
- **Correlation Heatmaps**: Generates comprehensive correlation matrices
- **RUL-Specific Analysis**: Creates focused plots showing feature correlations with RUL
- **All Datasets Support**: Works with CALCE, HUST, MATR, SNL, HNEI, RWTH, UL_PUR, and OX datasets

## What It Does

1. **Creates Cycle-Feature Matrix**: For each battery, creates a matrix where:
   - Rows represent cycle numbers
   - Columns represent features (voltage, current, capacity, etc.)
   - Each cell contains the mean value of that feature for that cycle
   - Includes RUL column showing remaining cycles until End of Life

2. **Generates Correlation Analysis**:
   - Full correlation matrix heatmap showing relationships between all features
   - RUL-specific correlation plot showing which features correlate most with RUL
   - Saves matrices as CSV files for further analysis

## Usage

### Command Line Interface

#### Analyze correlations for a specific dataset:
```bash
python -m batteryml.data_analysis.run_correlation_analysis --dataset CALCE --data_path data/processed/CALCE
```

#### Analyze correlations for all datasets:
```bash
python -m batteryml.data_analysis.run_correlation_analysis --all --data_path data/processed
```

#### Using the batch script (Windows):
```cmd
run_correlation_analysis.bat
```

### Python API

```python
from batteryml.data_analysis.correlation_analyzer import CorrelationAnalyzer

# Create analyzer instance
analyzer = CorrelationAnalyzer("data/processed/CALCE", "calce_correlation_analysis")

# Run analysis
analyzer.analyze_dataset()
```

## Output Structure

The correlation analyzer creates the following directory structure:

```
output_dir/
├── heatmaps/
│   ├── battery1_correlation_heatmap.png    # Full correlation matrix
│   ├── battery1_rul_correlations.png       # RUL-specific correlations
│   ├── battery2_correlation_heatmap.png
│   ├── battery2_rul_correlations.png
│   └── ...
└── matrices/
    ├── battery1_cycle_feature_matrix.csv   # Cycle-feature matrix
    ├── battery2_cycle_feature_matrix.csv
    └── ...
```

## Matrix Structure

Each cycle-feature matrix contains:

| cycle_number | rul | voltage | current | capacity | temperature | ... |
|--------------|-----|---------|---------|----------|-------------|-----|
| 1            | 500 | 3.8     | -1.0    | 1.1      | 25.0        | ... |
| 2            | 499 | 3.7     | -1.0    | 1.09     | 25.1        | ... |
| 3            | 498 | 3.6     | -1.0    | 1.08     | 25.2        | ... |
| ...          | ... | ...     | ...     | ...      | ...         | ... |

Where:
- **cycle_number**: The cycle index
- **rul**: Remaining Useful Life (cycles until EOL)
- **voltage**: Mean voltage for that cycle
- **current**: Mean current for that cycle
- **capacity**: Mean discharge capacity for that cycle
- **temperature**: Mean temperature for that cycle
- **...**: Other detected features

## RUL Calculation

RUL is calculated using the RULLabelAnnotator with the following logic:
- **EOL Threshold**: 80% of nominal capacity (configurable)
- **Cycle RUL**: For each cycle, RUL = Total RUL - Cycle Index
- **Total RUL**: Cycles until capacity drops below EOL threshold

## Supported Features

The analyzer automatically detects and includes all available features:

### Common Features:
- **voltage**: Mean voltage per cycle
- **current**: Mean current per cycle
- **capacity**: Mean discharge capacity per cycle
- **temperature**: Mean temperature per cycle

### Dataset-Specific Features:
- **charge_capacity**: Mean charge capacity per cycle
- **internal_resistance**: Mean internal resistance per cycle
- **energy_charge**: Mean charge energy per cycle
- **energy_discharge**: Mean discharge energy per cycle
- **qdlin**: Mean Qdlin feature per cycle (MATR)
- **tdlin**: Mean Tdlin feature per cycle (MATR)

## Visualization Types

### 1. Correlation Heatmap
- Shows correlations between all features
- Uses red-blue color scheme (red = negative, blue = positive)
- Includes correlation coefficients as annotations
- Masks upper triangle to avoid redundancy

### 2. RUL Correlation Plot
- Horizontal bar chart showing feature correlations with RUL
- Sorted by absolute correlation strength
- Color-coded (red = negative, blue = positive)
- Shows correlation values on bars

## Examples

### Example 1: Analyze CALCE dataset
```bash
python -m batteryml.data_analysis.run_correlation_analysis --dataset CALCE --data_path data/processed/CALCE --output_dir calce_correlations
```

### Example 2: Analyze all datasets
```bash
python -m batteryml.data_analysis.run_correlation_analysis --all --data_path data/processed --output_dir all_correlations
```

### Example 3: Using Python API
```python
from batteryml.data_analysis.correlation_analyzer import CorrelationAnalyzer

# Create analyzer for MATR dataset
analyzer = CorrelationAnalyzer(
    data_path="data/processed/MATR",
    output_dir="matr_correlations"
)

# Run analysis
analyzer.analyze_dataset()

# Access detected features
print(f"Features: {analyzer.features}")
```

## Interpretation

### Correlation Values:
- **+1.0**: Perfect positive correlation
- **0.0**: No correlation
- **-1.0**: Perfect negative correlation
- **>0.7**: Strong positive correlation
- **<-0.7**: Strong negative correlation

### RUL Correlations:
- **Positive correlation with RUL**: Feature increases as battery ages (degradation indicator)
- **Negative correlation with RUL**: Feature decreases as battery ages (capacity fade indicator)

## Requirements

- Python 3.7+
- matplotlib
- seaborn
- numpy
- pandas
- tqdm
- batteryml package

## Notes

- The analyzer uses the RULLabelAnnotator with default EOL threshold of 80%
- Invalid data points (NaN) are filtered out before calculating means
- Matrices are saved as CSV files for further analysis
- Heatmaps are optimized for publication quality with proper fonts and styling
- Each battery is analyzed independently to show individual degradation patterns
