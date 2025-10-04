# Enhanced Feature-RUL Correlation Analysis

## Overview

This enhanced correlation analysis system addresses the fundamental challenge of correlating cycle-level features with battery-level RUL (Remaining Useful Life). It implements the **aggregated approach** to properly handle the many-to-one relationship between cycle-level features and battery-level RUL.

## The Challenge

- **Battery Level**: One RUL value per battery (e.g., Battery A: RUL = 500 cycles)
- **Cycle Level**: Multiple feature values per battery (e.g., Battery A: 100 cycles, each with discharge capacity, voltage, etc.)
- **Goal**: Correlate cycle-level features with battery-level RUL

## Solution: Aggregated Approach

The aggregated approach solves this by:

1. **Extracting cycle features** for each cycle in a battery's lifetime
2. **Aggregating across all cycles** using statistical measures (mean, variance, median, etc.)
3. **Creating one value per battery** for each statistical measure
4. **Correlating with RUL** using Spearman correlation

## Key Features

### 1. **Cycle Features Integration**
- Integrates with `cycle_features.py` for comprehensive feature extraction
- Supports all dataset-specific feature extractors (MATR, CALCE, HUST, RWTH, OX, UL_PUR, etc.)
- Handles both raw features and engineered cycle features

### 2. **Statistical Measures**
- **Central tendency**: mean, median
- **Variability**: variance, std, range, IQR
- **Distribution shape**: kurtosis, skewness
- **Extremes**: min, max, q25, q75

### 3. **Aggregation Methods**
- **Single Cycle**: Use features from one specific cycle
- **Aggregated**: Aggregate features across all cycles per battery (recommended)

### 4. **Visualization**
- **Diverging bar charts**: Clear visualization of correlation strength and direction
- **Color coding**: Red for negative correlations, blue for positive
- **Value annotations**: Exact correlation coefficients displayed

## Available Cycle Features

### Basic Features
- `avg_voltage`: Average voltage over the entire cycle
- `avg_current`: Average current over the entire cycle
- `avg_c_rate`: Average C-rate over the entire cycle
- `cycle_length`: Total cycle duration

### Capacity Features
- `max_charge_capacity`: Maximum charge capacity observed
- `max_discharge_capacity`: Maximum discharge capacity observed
- `avg_charge_capacity`: Average charge capacity during charge window
- `avg_discharge_capacity`: Average discharge capacity during discharge window

### Time-based Features
- `charge_cycle_length`: Duration of charge segment
- `discharge_cycle_length`: Duration of discharge segment
- `charge_to_discharge_time_ratio`: Ratio of charge to discharge time

### C-rate Features
- `avg_charge_c_rate`: Average C-rate during charge
- `avg_discharge_c_rate`: Average C-rate during discharge
- `max_charge_c_rate`: Maximum instantaneous C-rate during charge

### Power Features
- `power_during_charge_cycle`: Power during charge segment
- `power_during_discharge_cycle`: Power during discharge segment

## Usage Examples

### PowerShell Usage

#### Basic Aggregated Analysis
```powershell
# Analyze cycle features using aggregated approach
.\run_feature_rul_correlation.ps1 -DatasetName "UL_PUR" -Feature "avg_voltage" -Method "aggregated"

# Analyze with custom statistical measures
.\run_feature_rul_correlation.ps1 -DatasetName "MATR" -Feature "cycle_length" -Method "aggregated" -Measures @("mean", "variance", "kurtosis", "skewness")
```

#### Single Cycle Analysis
```powershell
# Analyze features from a specific cycle
.\run_feature_rul_correlation.ps1 -DatasetName "CALCE" -Feature "avg_voltage" -Cycle 100 -Method "single_cycle"
```

#### Multi-Dataset Analysis
```powershell
# Analyze across multiple datasets
.\run_feature_rul_correlation.ps1 -DatasetName "UL_PUR" -AllDatasets @("UL_PUR", "MATR", "CALCE") -Feature "avg_voltage" -Method "aggregated"
```

### Python Direct Usage

#### Basic Analysis
```python
from batteryml.chemistry_data_analysis.statistical_analysis.feature_rul_correlation import FeatureRULCorrelationAnalyzer

# Initialize analyzer
analyzer = FeatureRULCorrelationAnalyzer("data")

# Load data
data = analyzer.load_battery_data("UL_PUR")
data = analyzer.calculate_rul_labels(data, "UL_PUR")

# Analyze cycle features with aggregated approach
correlations = analyzer.calculate_correlations(
    data, "avg_voltage", None, 
    ["mean", "variance", "median", "kurtosis", "skewness", "min", "max"], 
    "aggregated"
)

# Create plot
analyzer.plot_correlation_diverging_bar(
    data, "avg_voltage", None, "UL_PUR", 
    ["mean", "variance", "median", "kurtosis", "skewness", "min", "max"], 
    "aggregated", "correlation_plot.png"
)
```

#### Custom Features
```python
# Add custom cycle features
def extract_voltage_efficiency(cycle_data):
    discharge_data = cycle_data[cycle_data['current'] < 0]
    voltage = discharge_data['voltage'].dropna()
    return voltage.values

analyzer.register_feature_extractor('voltage_efficiency', extract_voltage_efficiency)

# Analyze custom feature
correlations = analyzer.calculate_correlations(
    data, "voltage_efficiency", None, 
    ["mean", "variance", "median"], 
    "aggregated"
)
```

## Output Interpretation

### Diverging Bar Chart
- **X-axis**: Spearman correlation with log(RUL) (-1 to +1)
- **Y-axis**: Statistical measures (sorted by correlation strength)
- **Colors**: Red (negative correlation), Blue (positive correlation)
- **Values**: Exact correlation coefficients displayed on bars

### Correlation Strength
- **|r| > 0.7**: Strong correlation
- **0.5 < |r| < 0.7**: Moderate correlation
- **0.3 < |r| < 0.5**: Weak correlation
- **|r| < 0.3**: Very weak or no correlation

### Statistical Measures Interpretation
- **Mean**: Average feature value across all cycles
- **Variance**: Variability of feature values across cycles
- **Median**: Middle value, robust to outliers
- **Kurtosis**: Peakness of distribution (high = sharp peaks)
- **Skewness**: Asymmetry of distribution (positive = right tail)
- **Min/Max**: Extreme values across cycles

## Benefits of Aggregated Approach

### 1. **Uses All Available Data**
- No information loss from using only one cycle
- More robust to cycle-to-cycle variations
- Better representation of battery behavior

### 2. **Handles Many-to-One Relationship**
- Properly aggregates multiple cycle values to one battery value
- Maintains statistical validity
- Enables meaningful correlation analysis

### 3. **Captures Temporal Patterns**
- Mean: Overall battery performance
- Variance: Consistency of performance
- Skewness: Degradation patterns
- Kurtosis: Stability characteristics

### 4. **Robust to Outliers**
- Spearman correlation is rank-based
- Less sensitive to extreme values
- More reliable for battery data

## Research Applications

### 1. **Feature Selection**
- Identify most predictive cycle features
- Compare feature importance across datasets
- Guide feature engineering efforts

### 2. **Battery Health Monitoring**
- Find early warning indicators
- Understand degradation patterns
- Develop prognostic models

### 3. **Cross-Dataset Validation**
- Ensure features work across different datasets
- Identify universal vs dataset-specific features
- Validate model generalizability

### 4. **Quality Control**
- Compare feature distributions across batteries
- Identify manufacturing variations
- Monitor battery performance consistency

## File Structure

```
batteryml/chemistry_data_analysis/statistical_analysis/
├── feature_rul_correlation.py          # Main analysis module
├── __init__.py                         # Package initialization
└── ...

run_feature_rul_correlation.ps1         # PowerShell wrapper
example_enhanced_correlation_analysis.py # Example usage
ENHANCED_CORRELATION_ANALYSIS_README.md  # This documentation
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `scipy`: Statistical functions
- `batteryml.chemistry_data_analysis.cycle_features`: Cycle feature extraction
- `batteryml.label.rul`: RUL calculation
- `batteryml.data.battery_data`: Battery data handling

## Example Output

The system generates PNG files showing diverging bar charts with:
- Feature name and dataset
- Aggregation method used
- Statistical measures ranked by correlation strength
- Color-coded correlation direction
- Exact correlation values
- Professional formatting for publication

This enhanced system provides a comprehensive solution for analyzing cycle-level features in relation to battery RUL, making it an invaluable tool for battery research and development.
