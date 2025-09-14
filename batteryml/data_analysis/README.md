# Battery Data Analysis Tools

This module provides comprehensive data analysis tools for battery datasets in the BatteryML framework.

## Features

- **Dataset Overview Analysis**: Total batteries per dataset, chemistry distribution, capacity and cycle life statistics
- **Feature Statistics**: Comprehensive analysis of all features including min, max, mean, median, standard deviation, and quartiles
- **Visualizations**: Interactive plots and charts for data exploration
- **Export Capabilities**: Save results as CSV files and summary reports
- **Streaming Analysis**: Memory-efficient processing that handles large datasets by processing files one at a time
- **Progress Tracking**: Beautiful tqdm progress bars for real-time monitoring

## Quick Start

### 1. Basic Usage

```python
from batteryml.data_analysis.analyzer import BatteryDataAnalyzer
from batteryml.data_analysis.visualization import BatteryDataVisualizer

# Initialize analyzer
analyzer = BatteryDataAnalyzer("path/to/battery/data")

# Run analysis
dataset_stats = analyzer.analyze_dataset_overview()
feature_stats = analyzer.analyze_features()

# Generate visualizations
visualizer = BatteryDataVisualizer("output/plots")
visualizer.save_all_plots(dataset_stats, feature_stats)
```

### 2. Command Line Usage

```bash
# Regular analysis (loads all data into memory)
python batteryml/data_analysis/run_analysis.py --data_path data/processed/MATR --output_dir results

# Streaming analysis (memory-efficient, processes one file at a time)
python batteryml/data_analysis/run_analysis.py --data_path data/processed/MATR --output_dir results --streaming

# Quick analysis without plots
python batteryml/data_analysis/run_analysis.py --data_path data/processed/MATR --no_plots

# Save detailed CSV files (regular analysis only)
python batteryml/data_analysis/run_analysis.py --data_path data/processed/MATR --save_csv
```

### 3. Simple Script Usage

```bash
# Regular analysis
python analyze_battery_data.py

# Streaming analysis (memory-efficient)
python analyze_battery_streaming.py

# Or import and use directly
from analyze_battery_data import analyze_dataset
analyze_dataset("data/processed/MATR", "matr_analysis")
```

### 4. Streaming Analysis (Memory-Efficient)

For large datasets or memory-constrained environments:

```python
from batteryml.data_analysis.streaming_analyzer import StreamingBatteryDataAnalyzer

# Initialize streaming analyzer
analyzer = StreamingBatteryDataAnalyzer("path/to/battery/data")

# Run complete analysis (processes one file at a time)
dataset_stats, feature_stats = analyzer.run_complete_analysis()

# Results are identical to regular analysis
print(f"Total batteries: {dataset_stats['total_batteries']}")
print(f"Features analyzed: {len(feature_stats)}")
```

## Analysis Components

### Dataset Overview
- Total number of batteries per dataset
- Battery chemistry distribution (LCO, LFP, NMC, NCA, etc.)
- Nominal capacity distribution
- Cycle life distribution
- Voltage and temperature ranges

### Feature Analysis
The analyzer extracts and analyzes features from:

#### Basic Properties
- Nominal capacity (Ah)
- Cycle count per battery

#### Capacity Features
- Charge capacity (Ah)
- Discharge capacity (Ah)
- Maximum discharge capacity per cycle

#### Voltage Features
- Voltage measurements (V)
- Minimum voltage per cycle
- Maximum voltage per cycle

#### Current Features
- Current measurements (A)
- Minimum current per cycle
- Maximum current per cycle

#### Temperature Features
- Temperature measurements (°C)
- Minimum temperature per cycle
- Maximum temperature per cycle

#### Time Features
- Time measurements (s)
- Cycle duration

### Statistics Provided
For each feature, the analyzer provides:
- **Count**: Number of valid measurements
- **Min**: Minimum value
- **Max**: Maximum value
- **Mean**: Average value
- **Median**: Median value
- **Standard Deviation**: Measure of variability
- **Q25**: 25th percentile
- **Q75**: 75th percentile

## Output Files

### Generated Files
- `dataset_overview.csv`: Dataset composition statistics
- `feature_statistics.csv`: Detailed feature statistics
- `summary_report.txt`: Human-readable summary report

### Visualization Files
- `dataset_overview.png`: Dataset composition charts
- `feature_statistics.png`: Feature distribution plots
- `feature_correlation.png`: Feature correlation heatmap
- `analysis_dashboard.png`: Comprehensive analysis dashboard

## Error Handling

The analysis tools include robust error handling:
- Graceful handling of missing or corrupted data files
- Safe statistics calculation with NaN value handling
- Comprehensive error messages and warnings
- Fallback options for incomplete datasets

## When to Use Streaming vs Regular Analysis

### **Use Streaming Analysis When:**
- ✅ Dataset has > 100 battery files
- ✅ Individual files are large (> 10MB each)
- ✅ Limited available RAM (< 8GB)
- ✅ Processing on cloud instances with memory constraints
- ✅ Want to monitor progress in real-time
- ✅ Need to process datasets of any size

### **Use Regular Analysis When:**
- ✅ Dataset has < 50 battery files
- ✅ Individual files are small (< 5MB each)
- ✅ Plenty of available RAM (> 16GB)
- ✅ Need fastest possible processing
- ✅ Want to keep all data in memory for further analysis

### **Performance Comparison:**
| Aspect | Regular Analysis | Streaming Analysis |
|--------|------------------|-------------------|
| **Memory Usage** | Loads ALL data at once | Processes ONE file at a time |
| **Speed** | Faster for small datasets | Slightly slower due to file I/O |
| **Scalability** | Limited by available RAM | Scales to any dataset size |
| **Progress Tracking** | Basic progress bars | Detailed progress for each step |
| **Results** | Identical | Identical |

## Dependencies

Additional dependencies required for analysis:
```
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.60.0
```

Install with:
```bash
pip install -r batteryml/data_analysis/requirements.txt
```

## Examples

### Example 1: Analyze MATR Dataset
```python
from batteryml.data_analysis.analyzer import BatteryDataAnalyzer

analyzer = BatteryDataAnalyzer("data/processed/MATR")
dataset_stats = analyzer.analyze_dataset_overview()
feature_stats = analyzer.analyze_features()

print(f"Total batteries: {dataset_stats['total_batteries']}")
print(f"Features analyzed: {len(feature_stats)}")
```

### Example 2: Custom Visualization
```python
from batteryml.data_analysis.visualization import BatteryDataVisualizer

visualizer = BatteryDataVisualizer("custom_plots")
visualizer.plot_dataset_overview(dataset_stats, save=True)
visualizer.plot_feature_statistics(feature_stats, save=True)
```

### Example 3: Batch Analysis
```python
datasets = ["MATR", "CALCE", "HUST", "SNL"]
for dataset in datasets:
    data_path = f"data/processed/{dataset}"
    output_dir = f"analysis_{dataset.lower()}"
    
    analyzer = BatteryDataAnalyzer(data_path)
    dataset_stats = analyzer.analyze_dataset_overview()
    feature_stats = analyzer.analyze_features()
    
    visualizer = BatteryDataVisualizer(f"{output_dir}/plots")
    visualizer.save_all_plots(dataset_stats, feature_stats)
    
    analyzer.save_analysis(output_dir)
```

## Troubleshooting

### Common Issues

1. **No .pkl files found**
   - Ensure data path contains .pkl files
   - Check file permissions

2. **Import errors**
   - Install required dependencies
   - Check Python path

3. **Memory issues with large datasets**
   - Use `--no_plots` flag for faster analysis
   - Process datasets in smaller batches

4. **Empty statistics**
   - Check data quality
   - Verify .pkl files contain valid BatteryData objects

### Getting Help

For issues or questions:
1. Check the error messages and warnings
2. Verify your data format matches BatteryData structure
3. Ensure all dependencies are installed
4. Check file paths and permissions
