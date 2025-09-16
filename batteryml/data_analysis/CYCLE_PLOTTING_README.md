# Cycle Plotting for Battery Data Analysis

This module provides functionality to generate time-series plots for different battery features across multiple cycles with configurable cycle gaps.

## Features

- **Dynamic Feature Detection**: Automatically detects available features for each dataset
- **Relative Time Plotting**: Each cycle starts at t=0 for easy comparison
- **Multiple Features**: Plots all available features (voltage, current, capacity, temperature, internal resistance, energy, etc.)
- **Configurable Cycle Gap**: Select cycles at regular intervals (default: 100)
- **Intuitive Color Scheme**: Blue for early cycles, red for late cycles with colorbar
- **All Datasets Support**: Works with CALCE, HUST, MATR, SNL, HNEI, RWTH, UL_PUR, and OX datasets
- **PNG Output**: Saves plots as high-quality PNG files without displaying them

## Usage

### Command Line Interface

#### Plot cycles for a specific dataset:
```bash
python -m batteryml.data_analysis.run_cycle_plots --dataset CALCE --data_path data/processed/CALCE --cycle_gap 50
```

#### Plot cycles for all datasets:
```bash
python -m batteryml.data_analysis.run_cycle_plots --all --data_path data/processed --cycle_gap 100
```

#### Using the batch script (Windows):
```cmd
run_cycle_plots.bat 50
```

### Python API

```python
from batteryml.data_analysis.cycle_plotter import CyclePlotter

# Create plotter instance
plotter = CyclePlotter(
    data_path="data/processed/CALCE",
    output_dir="calce_cycle_plots",
    cycle_gap=100
)

# Generate plots for all batteries
plotter.plot_dataset_features()
```

## Output Structure

The cycle plotter creates the following directory structure:

```
output_dir/
├── voltage_vs_time/
│   ├── battery1_voltage_time.png
│   ├── battery2_voltage_time.png
│   └── ...
├── current_vs_time/
│   ├── battery1_current_time.png
│   ├── battery2_current_time.png
│   └── ...
├── capacity_vs_time/
│   ├── battery1_capacity_time.png
│   ├── battery2_capacity_time.png
│   └── ...
└── temperature_vs_time/
    ├── battery1_temperature_time.png
    ├── battery2_temperature_time.png
    └── ...
```

## Parameters

- **data_path**: Path to the processed battery data directory
- **output_dir**: Directory to save cycle plots (default: "cycle_plots")
- **cycle_gap**: Gap between cycles to plot (default: 100)

## Cycle Selection Logic

The plotter automatically selects cycles based on the cycle gap:

1. Always includes cycle 1 (index 0)
2. Adds cycles at regular intervals (cycle_gap, 2*cycle_gap, 3*cycle_gap, ...)
3. Always includes the last cycle if it's not already selected

For example, with cycle_gap=100 and 500 total cycles:
- Selected cycles: [0, 100, 200, 300, 400, 499]

## Relative Time Normalization

Each cycle is normalized to start at t=0:

- **Before**: Cycle 1: t=0 to t=10, Cycle 2: t=11 to t=25
- **After**: Cycle 1: t=0 to t=10, Cycle 2: t=0 to t=14

This makes it easy to compare the behavior of different cycles regardless of their absolute timing.

## Color Scheme

The plots use an intuitive color gradient to distinguish between early and late cycles:

- **Early Cycles**: Blue colors (easy to identify fresh battery behavior)
- **Late Cycles**: Red colors (easy to identify aged battery behavior)
- **Colorbar**: Shows the cycle progression from early to late cycles
- **Gradient**: Smooth transition from blue → yellow → red

This color scheme makes it immediately clear which cycles represent early battery life versus end-of-life behavior.

## Supported Features

The plotter automatically detects and plots all available features for each dataset:

### Common Features (Available in most datasets):
1. **Voltage vs Time**: Discharge voltage profiles across cycles
2. **Current vs Time**: Current profiles during discharge  
3. **Capacity vs Time**: Discharge capacity accumulation over time
4. **Temperature vs Time**: Temperature evolution during cycles

### Dataset-Specific Features:
- **Charge Capacity vs Time**: Charge capacity profiles (CALCE, MATR, OX, SNL, NEWARE)
- **Internal Resistance vs Time**: Internal resistance evolution (MATR, NEWARE)
- **Energy Charge vs Time**: Charge energy profiles (NEWARE)
- **Energy Discharge vs Time**: Discharge energy profiles (NEWARE)
- **Qdlin vs Time**: Qdlin feature evolution (MATR)
- **Tdlin vs Time**: Tdlin feature evolution (MATR)

### Feature Detection:
The plotter examines the first battery file in each dataset to automatically detect which features are available, ensuring that only relevant plots are generated for each dataset.

## Requirements

- Python 3.7+
- matplotlib
- numpy
- tqdm
- batteryml package

## Examples

### Example 1: Plot every 50th cycle for CALCE dataset
```bash
python -m batteryml.data_analysis.run_cycle_plots --dataset CALCE --data_path data/processed/CALCE --cycle_gap 50 --output_dir calce_cycle_plots_50
```

### Example 2: Plot every 200th cycle for all datasets
```bash
python -m batteryml.data_analysis.run_cycle_plots --all --data_path data/processed --cycle_gap 200 --output_dir all_cycle_plots_200
```

### Example 3: Using Python API
```python
from batteryml.data_analysis.cycle_plotter import CyclePlotter

# Create plotter for MATR dataset with 25-cycle gap
plotter = CyclePlotter(
    data_path="data/processed/MATR",
    output_dir="matr_cycle_plots",
    cycle_gap=25
)

# Generate all plots
plotter.plot_dataset_features()
```

## Notes

- The plotter uses matplotlib's 'Agg' backend to prevent display windows
- All plots are saved as high-resolution PNG files (300 DPI)
- Invalid data points (NaN, negative values where inappropriate) are filtered out
- Each plot includes a legend showing the cycle numbers
- Plots are optimized for publication quality with proper fonts and styling
