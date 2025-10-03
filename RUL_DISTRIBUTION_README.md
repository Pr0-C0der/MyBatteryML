# RUL Distribution Analysis

This module provides comprehensive analysis and visualization of Remaining Useful Life (RUL) distributions across different battery chemistries.

## Features

- **Individual Chemistry Analysis**: Analyze RUL distributions for each chemistry separately
- **Combined Visualization**: Create side-by-side comparisons of all chemistries
- **Statistical Summary**: Calculate mean, median, standard deviation, quartiles, and range
- **Multiple Plot Types**: Histograms, box plots, and statistical comparisons
- **Data Export**: Save detailed RUL data as CSV files
- **High-Quality Output**: Generate PNG plots with 300 DPI resolution

## Files

- `batteryml/chemistry_data_analysis/rul_distribution.py` - Main analysis module
- `analyze_rul_distributions.py` - Standalone script to analyze all chemistries
- `run_chemistry_rul_distributions.ps1` - PowerShell script for Windows
- `test_rul_distribution.py` - Test script to verify functionality

## Usage

### Quick Start

```bash
# Analyze all available chemistries
python analyze_rul_distributions.py
```

### Individual Chemistry Analysis

```bash
# Analyze a specific chemistry
python -m batteryml.chemistry_data_analysis.rul_distribution \
    --data_path data/datasets_requiring_access/UL_PUR \
    --output_dir chemistry_rul_distributions \
    --verbose
```

### PowerShell (Windows)

```powershell
# Run the PowerShell script
.\run_chemistry_rul_distributions.ps1
```

## Output Files

### Individual Chemistry Files
For each chemistry (e.g., `UL_PUR`):
- `chemistry_rul_distributions/UL_PUR/UL_PUR_rul_distribution.png` - Histogram plot
- `chemistry_rul_distributions/UL_PUR/UL_PUR_rul_data.csv` - Detailed RUL data

### Combined Analysis Files
- `chemistry_rul_distributions/all_chemistries_rul_distributions.png` - Side-by-side histograms
- `chemistry_rul_distributions/all_chemistries_rul_boxplot.png` - Box plot comparison

## Supported Chemistries

The analysis supports the following battery chemistries:
- **HNEI**: Mixed NMC-LCO chemistry
- **SNL**: Multiple chemistries (LFP, NCA, NMC)
- **UL_PUR**: NCA chemistry

## Dataset Detection

The system automatically detects the dataset type based on:
1. Cell ID patterns (e.g., "UL-PUR" in cell ID)
2. Reference/description metadata
3. Manual dataset hint parameter

## RUL Calculation

RUL is calculated using the `RULLabelAnnotator` which:
- Identifies the end-of-life threshold (typically 80% capacity retention)
- Calculates remaining cycles until end-of-life
- Handles various battery degradation patterns

## Statistical Metrics

For each chemistry, the analysis provides:
- **Count**: Number of batteries with valid RUL
- **Mean**: Average RUL across all batteries
- **Standard Deviation**: RUL variability
- **Min/Max**: RUL range
- **Median**: Middle value
- **Q25/Q75**: First and third quartiles

## Troubleshooting

### UL_PUR Dataset Issues

If UL_PUR batteries are not loading properly:

1. **Check file format**: Ensure `.pkl` files are valid BatteryData objects
2. **Verify preprocessing**: Run the UL_PUR preprocessor first
3. **Check cell IDs**: Ensure cell IDs contain "UL-PUR" or "UL_PUR" patterns
4. **Enable verbose mode**: Use `--verbose` flag to see detailed processing information

### Common Issues

- **No RUL data found**: Check if batteries have sufficient cycle data
- **Dataset detection failed**: Use `--dataset_hint` parameter
- **Memory issues**: Process chemistries individually for large datasets

## Example Output

```
RUL DISTRIBUTION ANALYSIS SUMMARY
============================================================

HNEI:
  Count: 14 batteries
  Mean RUL: 1250.5 cycles
  Std RUL: 234.2 cycles
  Median RUL: 1200.0 cycles
  Range: 800 - 1800 cycles
  Q25-Q75: 1050.0 - 1450.0 cycles

SNL:
  Count: 61 batteries
  Mean RUL: 980.3 cycles
  Std RUL: 312.7 cycles
  Median RUL: 950.0 cycles
  Range: 400 - 1600 cycles
  Q25-Q75: 750.0 - 1200.0 cycles

UL_PUR:
  Count: 10 batteries
  Mean RUL: 1100.2 cycles
  Std RUL: 189.4 cycles
  Median RUL: 1080.0 cycles
  Range: 850 - 1400 cycles
  Q25-Q75: 950.0 - 1250.0 cycles
```

## Dependencies

- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `seaborn` - Enhanced visualizations
- `pandas` - Data manipulation
- `tqdm` - Progress bars
- `batteryml` - Battery data processing

## License

This module is part of the BatteryML project and follows the same license terms.
