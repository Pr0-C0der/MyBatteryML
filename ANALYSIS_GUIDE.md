# Battery Data Analysis Guide

This guide explains how to use the battery data analysis tools to generate comprehensive visualizations and statistics for your processed battery datasets.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r batteryml/data_analysis/requirements.txt
```

### 2. Ensure You Have Processed Data

Make sure you have processed battery data in the `data/processed/` directory. If not, process your data first:

```bash
# Example: Process CALCE data
batteryml preprocess CALCE data/raw/CALCE data/processed/CALCE

# Example: Process MATR data  
batteryml preprocess MATR data/raw/MATR data/processed/MATR
```

### 3. Run Analysis

#### Option A: Simple Interface (Recommended)
```bash
python analyze_datasets.py
```

#### Option B: Command Line
```bash
# Analyze specific dataset
python -m batteryml.data_analysis.run_analysis --dataset CALCE --data_path data/processed/CALCE

# Analyze all datasets
python -m batteryml.data_analysis.run_analysis --all --data_path data/processed
```

## What Gets Generated

### 1. Plots for Each Battery

For every battery pickle file, 5 types of plots are generated:

- **Capacity Fade**: `{battery_id}_capacity_fade.png`
- **Voltage vs Capacity**: `{battery_id}_voltage_capacity.png`  
- **Charge vs Discharge Capacity**: `{battery_id}_qc_qd.png`
- **Current vs Time**: `{battery_id}_current_time.png`
- **Voltage vs Time**: `{battery_id}_voltage_time.png`

### 2. Summary Tables

- **Overall Statistics**: `dataset_summary.csv` - Comprehensive statistics table
- **Dataset-Specific**: `{dataset}_summary.txt` - Custom analysis for each dataset type

### 3. Directory Structure

```
analysis_output/
├── plots/
│   ├── capacity_fade/          # Capacity fade curves
│   ├── voltage_capacity/       # Voltage vs capacity curves  
│   ├── qc_qd/                  # Charge vs discharge capacity
│   ├── current_time/           # Current vs time profiles
│   └── voltage_time/           # Voltage vs time profiles
├── dataset_summary.csv         # Overall statistics
└── {dataset}_summary.txt       # Dataset-specific analysis
```

## Dataset-Specific Analysis

### CALCE Dataset
- **Cell Types**: CS (1.1 Ah) vs CX (1.35 Ah) analysis
- **Voltage Range**: 2.7V - 4.2V analysis
- **Cycle Life**: Distribution and statistics

### HUST Dataset  
- **Discharge Protocols**: 3-stage discharge rate analysis
- **Temperature**: Operating temperature ranges
- **Cell Patterns**: ID-based grouping

### MATR Dataset
- **Batches**: b1, b2, b3, b4 batch distribution
- **Charge Policies**: Protocol analysis
- **Internal Resistance**: Statistical analysis
- **Qdlin Features**: Differential capacity analysis

### SNL Dataset
- **Cathode Materials**: NMC, NCA, LFP distribution
- **Temperature Groups**: 15°C, 25°C, 35°C analysis
- **SOC Ranges**: 0-100%, 20-80%, 40-60% analysis
- **Discharge Rates**: Rate combination analysis

## Example Output

### Summary Table (CSV)
```csv
Feature,Value
Total Batteries,180
Average Cycles per Battery,823.5
Max Cycles,1500
Min Cycles,200
Nominal Capacity (Ah) - Mean,1.1
Nominal Capacity (Ah) - Std,0.0
Cathode Material - LFP,180
Anode Material - graphite,180
Voltage Range (V) - Mean,1.5
Cycle Life - Mean,823.5
Cycle Life - Std,368.2
```

### Dataset-Specific Summary (TXT)
```
MATR Dataset Analysis Summary
=============================

Batch Distribution:
  b1: 45 cells
  b2: 45 cells
  b3: 45 cells
  b4: 45 cells

Charge Policy Distribution:
  CC-CV: 120 cells
  Multi-stage: 60 cells

Internal Resistance (Ohm):
  Min: 0.0123
  Max: 0.0456
  Mean: 0.0234
  Std: 0.0056

Cycle Life Statistics:
  Mean: 823.5
  Min: 200
  Max: 1500
```

## Memory Management

The analysis is designed to handle large datasets efficiently:

- **One Battery at a Time**: Processes individual pickle files to avoid memory issues
- **Incremental Statistics**: Collects statistics without loading all batteries into memory
- **Immediate Cleanup**: Closes plots and clears memory after each battery
- **Progress Tracking**: Shows progress with tqdm progress bars
- **Error Handling**: Continues analysis even if some batteries fail
- **Memory Efficient**: Can handle datasets with thousands of batteries without memory issues

## Troubleshooting

### Common Issues

1. **"No battery files found"**
   - Ensure data is processed: `batteryml preprocess <dataset> <raw_path> <processed_path>`
   - Check that pickle files exist in the processed directory

2. **"Import errors"**
   - Install requirements: `pip install -r batteryml/data_analysis/requirements.txt`
   - Run from project root directory

3. **"Memory errors"**
   - The analysis is memory-efficient, but very large datasets might still cause issues
   - Consider analyzing subsets of data

4. **"Plot generation errors"**
   - Check matplotlib backend configuration
   - Ensure write permissions in output directory

### Debug Mode

For detailed error information, run with Python's verbose mode:

```bash
python -u analyze_datasets.py
```

## Advanced Usage

### Custom Analysis

You can create custom analyzers by extending `BaseDataAnalyzer`:

```python
from batteryml.data_analysis.base_analyzer import BaseDataAnalyzer

class CustomAnalyzer(BaseDataAnalyzer):
    def analyze_custom_features(self):
        # Your custom analysis here
        pass
    
    def analyze_dataset(self):
        super().analyze_dataset()
        self.analyze_custom_features()
```

### Programmatic Usage

```python
from batteryml.data_analysis import MATRAnalyzer

# Create analyzer
analyzer = MATRAnalyzer("data/processed/MATR", "my_analysis")

# Run analysis
analyzer.analyze_dataset()

# Access results
print(f"Analysis complete! Check {analyzer.output_dir}")
```

## Performance Tips

1. **Use SSD Storage**: Faster I/O for large datasets
2. **Sufficient RAM**: At least 8GB recommended for large datasets
3. **Parallel Processing**: Consider running different datasets in parallel
4. **Incremental Analysis**: Analyze subsets if memory is limited

## File Formats

- **Input**: Pickle files (`.pkl`) from BatteryML preprocessing
- **Output**: PNG plots (300 DPI) and CSV/TXT summaries
- **Compatibility**: Works with all BatteryML supported datasets

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generated error messages
3. Ensure all dependencies are installed
4. Verify data preprocessing was successful
