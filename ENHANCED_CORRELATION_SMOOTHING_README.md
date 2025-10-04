# Enhanced Feature-RUL Correlation Analysis with Smoothing and Cycle Limits

## Overview

This enhanced correlation analysis system now includes **smoothing capabilities** and **cycle limit options** to provide more robust and flexible analysis of cycle-level features with battery-level RUL.

## New Features

### 1. **Smoothing Methods**
- **HMS Filter**: Hampel -> Median -> Savitzky-Golay (from chemistry_training.py)
- **Moving Average (MA)**: Simple moving average smoothing
- **Moving Median**: Median-based smoothing for outlier robustness

### 2. **Cycle Limit**
- **First N Cycles**: Analyze only the first N cycles of each battery
- **All Cycles**: Use all available cycles (default)
- **Flexible**: Can be combined with any smoothing method

### 3. **Battery-Level Only**
- **Removed cycle-level correlation**: Focus only on battery-level analysis
- **Simplified interface**: Cleaner, more focused functionality
- **Better performance**: Faster execution without cycle-level complexity

## Smoothing Methods Details

### **HMS Filter (Recommended)**
```python
# Three-step process:
# 1. Hampel filter: Remove outliers using median absolute deviation
# 2. Median filter: Apply median smoothing (kernel size 5)
# 3. Savitzky-Golay: Polynomial smoothing (window 101, order 3)
```

**Advantages:**
- **Outlier robust**: Hampel filter removes extreme values
- **Smooth trends**: Savitzky-Golay preserves important features
- **No parameters**: Automatically adjusts to data length

### **Moving Average**
```python
# Simple moving average with configurable window
window_size = 5  # Default
smoothed = moving_average(values, window_size)
```

**Advantages:**
- **Simple**: Easy to understand and implement
- **Fast**: Computationally efficient
- **Configurable**: Adjustable window size

### **Moving Median**
```python
# Median-based smoothing for outlier robustness
window_size = 5  # Default
smoothed = moving_median(values, window_size)
```

**Advantages:**
- **Outlier robust**: Median is less sensitive to outliers than mean
- **Preserves edges**: Better at preserving sharp changes
- **Configurable**: Adjustable window size

## Usage Examples

### **PowerShell Usage**

#### **Basic Analysis with Smoothing**
```powershell
# HMS smoothing (recommended)
.\run_feature_rul_correlation.ps1 -DatasetName "UL_PUR" -Feature "avg_voltage" -Method "aggregated" -Smoothing "hms"

# Moving average with custom window
.\run_feature_rul_correlation.ps1 -DatasetName "MATR" -Feature "cycle_length" -Method "aggregated" -Smoothing "ma" -SmoothingWindow 10

# Moving median smoothing
.\run_feature_rul_correlation.ps1 -DatasetName "CALCE" -Feature "max_discharge_capacity" -Method "aggregated" -Smoothing "median" -SmoothingWindow 7
```

#### **Cycle Limit Analysis**
```powershell
# Analyze only first 100 cycles
.\run_feature_rul_correlation.ps1 -DatasetName "UL_PUR" -Feature "avg_voltage" -Method "aggregated" -CycleLimit 100

# Combine cycle limit with smoothing
.\run_feature_rul_correlation.ps1 -DatasetName "MATR" -Feature "cycle_length" -Method "aggregated" -CycleLimit 150 -Smoothing "hms"
```

#### **Comprehensive Analysis**
```powershell
# Full analysis with HMS smoothing and cycle limit
.\run_feature_rul_correlation.ps1 -DatasetName "UL_PUR" -Feature "avg_voltage" -Method "aggregated" -CycleLimit 100 -Smoothing "hms" -Measures @("mean", "variance", "median", "kurtosis", "skewness", "min", "max")
```

### **Python Direct Usage**

#### **Basic Analysis**
```python
from batteryml.chemistry_data_analysis.statistical_analysis.feature_rul_correlation import FeatureRULCorrelationAnalyzer

# Initialize analyzer
analyzer = FeatureRULCorrelationAnalyzer("data")

# Load data
data = analyzer.load_battery_data("UL_PUR")
data = analyzer.calculate_rul_labels(data, "UL_PUR")

# HMS smoothing with cycle limit
correlations = analyzer.calculate_correlations(
    data, "avg_voltage", None, 
    ["mean", "variance", "median", "kurtosis", "skewness", "min", "max"], 
    "aggregated", cycle_limit=100, smoothing_method="hms", smoothing_window=5
)

# Create plot
analyzer.plot_correlation_diverging_bar(
    data, "avg_voltage", None, "UL_PUR", 
    ["mean", "variance", "median", "kurtosis", "skewness", "min", "max"], 
    "aggregated", 100, "hms", 5, "correlation_plot.png"
)
```

#### **Smoothing Comparison**
```python
# Compare different smoothing methods
smoothing_methods = ['none', 'hms', 'ma', 'median']

for smoothing in smoothing_methods:
    correlations = analyzer.calculate_correlations(
        data, "avg_voltage", None, statistical_measures, "aggregated",
        cycle_limit=100, 
        smoothing_method=smoothing if smoothing != 'none' else None,
        smoothing_window=5
    )
    
    print(f"{smoothing}: {correlations['mean']:.3f}")
```

## Parameter Reference

### **Smoothing Parameters**
- **`--smoothing`**: `none`, `hms`, `ma`, `median` (default: `none`)
- **`--smoothing_window`**: Window size for MA/median (default: 5, ignored for HMS)

### **Cycle Limit Parameters**
- **`--cycle_limit`**: Number of first cycles to analyze (default: all cycles)

### **Aggregation Parameters**
- **`--method`**: `single_cycle`, `aggregated` (default: `single_cycle`)

### **Statistical Measures**
- **`--measures`**: List of measures (default: `mean`, `variance`, `median`, `kurtosis`, `skewness`, `min`, `max`)

## When to Use Each Smoothing Method

### **HMS Filter (Recommended)**
- **Use when**: You want the most robust smoothing
- **Best for**: Noisy data with outliers
- **Advantages**: Automatic parameter adjustment, outlier removal
- **Example**: Battery data with measurement noise

### **Moving Average**
- **Use when**: You want simple, fast smoothing
- **Best for**: Data with consistent noise patterns
- **Advantages**: Fast, easy to understand
- **Example**: Smooth capacity fade curves

### **Moving Median**
- **Use when**: You have outliers but want simpler smoothing than HMS
- **Best for**: Data with occasional extreme values
- **Advantages**: Outlier robust, preserves sharp changes
- **Example**: Voltage data with occasional spikes

### **No Smoothing**
- **Use when**: You want to preserve all data characteristics
- **Best for**: Clean data or when you want to see raw patterns
- **Advantages**: No information loss
- **Example**: High-quality laboratory data

## Cycle Limit Benefits

### **1. Focus on Early Cycles**
- **Early degradation**: Focus on initial battery behavior
- **Consistent comparison**: Same number of cycles across batteries
- **Reduced noise**: Avoid late-cycle measurement issues

### **2. Computational Efficiency**
- **Faster processing**: Fewer cycles to process
- **Memory efficient**: Less data to store and process
- **Scalable**: Better performance with large datasets

### **3. Research Applications**
- **Early warning**: Identify early degradation indicators
- **Manufacturing QC**: Focus on initial battery performance
- **Model training**: Use consistent cycle ranges

## Output Examples

### **Diverging Bar Chart Features**
- **Title**: Shows smoothing method and cycle limit
- **X-axis**: Spearman correlation with log(RUL) (-1 to +1)
- **Y-axis**: Statistical measures (sorted by correlation strength)
- **Colors**: Red (negative), Blue (positive)
- **Values**: Exact correlation coefficients displayed

### **Example Titles**
- `Feature-RUL Correlation Analysis (Aggregated) - Feature: avg_voltage | All Cycles | Dataset: UL_PUR`
- `Feature-RUL Correlation Analysis (Aggregated) - Feature: cycle_length | First 100 Cycles | Dataset: MATR`
- `Feature-RUL Correlation Analysis (Aggregated) - Feature: max_discharge_capacity | All Cycles (HMS Smoothed) | Dataset: CALCE`

## Performance Considerations

### **Smoothing Performance**
- **HMS**: Slowest (3-step process), most robust
- **Moving Average**: Fastest, simplest
- **Moving Median**: Medium speed, good balance

### **Cycle Limit Performance**
- **100 cycles**: ~50% faster than all cycles
- **50 cycles**: ~75% faster than all cycles
- **Memory usage**: Proportional to cycle limit

### **Recommended Settings**
- **Research**: HMS smoothing, 100-200 cycle limit
- **Quick analysis**: No smoothing, 50-100 cycle limit
- **Production**: Moving average, 100 cycle limit

## File Structure

```
batteryml/chemistry_data_analysis/statistical_analysis/
├── feature_rul_correlation.py          # Enhanced main module
├── __init__.py                         # Package initialization
└── ...

run_feature_rul_correlation.ps1         # Enhanced PowerShell wrapper
example_enhanced_correlation_with_smoothing.py # Example usage
ENHANCED_CORRELATION_SMOOTHING_README.md # This documentation
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `scipy`: Statistical functions and smoothing
- `batteryml.chemistry_data_analysis.cycle_features`: Cycle feature extraction
- `batteryml.label.rul`: RUL calculation
- `batteryml.data.battery_data`: Battery data handling

## Example Output Files

The system generates PNG files with descriptive names:
- `correlation_avg_voltage_UL_PUR.png` (no smoothing)
- `correlation_avg_voltage_hms_UL_PUR.png` (HMS smoothing)
- `correlation_cycle_length_ma_100cycles_MATR.png` (MA smoothing, 100 cycles)
- `correlation_max_discharge_capacity_median_150cycles_CALCE.png` (median smoothing, 150 cycles)

This enhanced system provides a comprehensive solution for robust battery feature analysis with flexible smoothing and cycle limit options, making it an invaluable tool for battery research and development.
