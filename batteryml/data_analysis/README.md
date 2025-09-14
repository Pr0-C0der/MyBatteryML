# Battery Data Analysis Module

This module provides comprehensive data analysis tools for battery datasets processed by BatteryML. It generates visualizations and statistical summaries for each dataset.

## Features

- **Individual Battery Analysis**: Analyze each battery pickle file separately to avoid memory issues
- **Comprehensive Visualizations**: Generate 5 types of plots for each battery
- **Dataset-Specific Analysis**: Custom analysis for each dataset type (CALCE, HUST, MATR, SNL, etc.)
- **Statistical Summaries**: Generate detailed statistics tables for each dataset
- **Memory Efficient**: Processes batteries one at a time to handle large datasets

## Generated Plots

### Individual Battery Plots

For each battery, the following plots are generated:

1. **Capacity Fade Curves**: Discharge capacity vs cycle number
2. **Voltage vs Capacity Curves**: Voltage profiles for selected cycles
3. **Charge vs Discharge Capacity**: QC vs QD scatter plots
4. **Current vs Time**: Current profiles for selected cycles
5. **Voltage vs Time**: Voltage profiles for selected cycles

### Combined Plots (New Feature)

After generating individual plots, combined plots are automatically generated for 20 randomly selected batteries:

1. **Combined Capacity Fade**: All selected batteries on one plot
2. **Combined Voltage vs Capacity**: Multiple batteries' voltage profiles
3. **Combined QC vs QD**: Scatter plot of all selected batteries
4. **Combined Current vs Time**: Current profiles for multiple batteries
5. **Combined Voltage vs Time**: Voltage profiles for multiple batteries
6. **Capacity Distribution**: Statistical distribution plots

## Output Structure

```
analysis_output/
├── plots/
│   ├── capacity_fade/          # Capacity fade plots
│   ├── voltage_capacity/       # Voltage vs capacity plots
│   ├── qc_qd/                  # Charge vs discharge capacity plots
│   ├── current_time/           # Current vs time plots
│   └── voltage_time/           # Voltage vs time plots
├── combined_plots/             # Combined plots for random selection
│   ├── combined_capacity_fade.png
│   ├── combined_voltage_capacity.png
│   ├── combined_qc_qd.png
│   ├── combined_current_time.png
│   ├── combined_voltage_time.png
│   └── capacity_distribution.png
├── dataset_summary.csv         # Overall dataset statistics
└── {dataset}_summary.txt       # Dataset-specific summary
```

## Usage

### Method 1: Simple Interface

Run the simple analysis script from the project root:

```bash
python analyze_datasets.py
```

This will:
1. Check for processed data in `data/processed/`
2. Show available datasets
3. Let you choose to analyze all datasets or a specific one

### Method 2: Command Line Interface

#### Analyze a specific dataset:

```bash
python -m batteryml.data_analysis.run_analysis --dataset CALCE --data_path data/processed/CALCE
```

#### Analyze all datasets:

```bash
python -m batteryml.data_analysis.run_analysis --all --data_path data/processed
```

#### With custom output directory:

```bash
python -m batteryml.data_analysis.run_analysis --dataset MATR --data_path data/processed/MATR --output_dir my_analysis
```

### Method 3: Programmatic Usage

```python
from batteryml.data_analysis import CALCEAnalyzer

# Create analyzer
analyzer = CALCEAnalyzer("data/processed/CALCE", "calce_analysis")

# Run analysis
analyzer.analyze_dataset()
```

## Dataset-Specific Features

### CALCE Dataset
- Analyzes CS (1.1 Ah) vs CX (1.35 Ah) cell types
- Voltage range analysis (2.7V - 4.2V)
- Cycle life distribution

### HUST Dataset
- Discharge rate group analysis (3-stage discharge protocols)
- Temperature range analysis
- Cell ID pattern analysis

### MATR Dataset
- Batch distribution analysis (b1, b2, b3, b4)
- Charge policy analysis
- Internal resistance statistics
- Qdlin feature analysis

### SNL Dataset
- Cathode material distribution (NMC, NCA, LFP)
- Temperature group analysis (15°C, 25°C, 35°C)
- SOC range analysis (0-100%, 20-80%, 40-60%)
- Discharge rate analysis

### Other Datasets
- HNEI: NMC_LCO cell analysis
- RWTH: Cell ID analysis
- UL_PUR: N15/N20 cell type analysis
- OX: General cell analysis

## Requirements

Install the required packages:

```bash
pip install -r batteryml/data_analysis/requirements.txt
```

Required packages:
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- tqdm >= 4.62.0

## Output Files

### Summary Tables (CSV)
- `dataset_summary.csv`: Comprehensive statistics including:
  - Total batteries
  - Cycle counts (mean, min, max)
  - Nominal capacity statistics
  - Material composition
  - Voltage ranges
  - Capacity fade rates
  - Cycle life statistics

### Dataset-Specific Summaries (TXT)
- `{dataset}_summary.txt`: Dataset-specific analysis including:
  - Cell type distributions
  - Protocol analysis
  - Feature-specific statistics
  - Temperature/voltage ranges

### Plots (PNG)
- High-resolution PNG files (300 DPI)
- Individual plots for each battery
- Organized in subdirectories by plot type
- Safe filenames (special characters replaced)

## Memory Management

The analysis is designed to be memory-efficient:
- **One Battery at a Time**: Processes individual pickle files to avoid memory issues
- **Incremental Statistics**: Collects statistics without loading all batteries into memory
- **Immediate Cleanup**: Closes plots and clears memory after each battery
- **Progress Tracking**: Shows progress with tqdm progress bars
- **Error Handling**: Continues analysis even if some batteries fail
- **Memory Efficient**: Can handle datasets with thousands of batteries without memory issues

## Error Handling

- Graceful handling of corrupted or invalid pickle files
- Continues analysis even if some batteries fail
- Detailed error messages for debugging
- Progress tracking with tqdm

## Examples

### Quick Start
```bash
# Make sure you have processed data
batteryml preprocess CALCE data/raw/CALCE data/processed/CALCE

# Run analysis
python analyze_datasets.py
```

### Advanced Usage
```bash
# Analyze specific dataset with custom output
python -m batteryml.data_analysis.run_analysis \
    --dataset MATR \
    --data_path data/processed/MATR \
    --output_dir detailed_matr_analysis
```

## Troubleshooting

1. **No data found**: Ensure you have processed data using `batteryml preprocess`
2. **Memory issues**: The analysis is designed to be memory-efficient, but very large datasets might still cause issues
3. **Plot generation errors**: Check that matplotlib backend is properly configured
4. **Import errors**: Ensure you're running from the project root directory

## Contributing

To add analysis for a new dataset:
1. Create a new analyzer class inheriting from `BaseDataAnalyzer`
2. Implement dataset-specific features in `analyze_{dataset}_specific_features()`
3. Add the analyzer to the `__init__.py` imports
4. Update the `run_analysis.py` analyzer mapping
