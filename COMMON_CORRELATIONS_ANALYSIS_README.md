# Common Statistical Correlations Analysis

This analysis finds statistical features that have high correlations (> 0.5) with log(RUL) that are common across multiple datasets (MATR, CALCE, HNEI, OX, RWTH, SNL, HUST, UL_PUR).

## Overview

The analysis processes all datasets and:
1. **Loads battery data** from each dataset
2. **Calculates statistical features** (mean, std, min, max, median, q25, q75) for each battery
3. **Computes correlations** with log(RUL) for each dataset
4. **Identifies common features** with high correlations across multiple datasets
5. **Generates visualizations** and summary reports

## Files

### Core Scripts
- `find_common_statistical_correlations.py` - Main analysis script
- `run_common_correlations_analysis.ps1` - PowerShell script for full analysis
- `quick_common_correlations_test.ps1` - Quick test script

### Output Files
- `correlation_summary.csv` - Detailed correlations across all datasets
- `common_features_summary.csv` - Summary of common high-correlation features
- `{DATASET}_correlations.csv` - Individual dataset correlation results
- `correlation_heatmap.png` - Heatmap visualization
- `feature_frequency.png` - Feature frequency across datasets

## Usage

### Quick Test
```powershell
.\powershell codes\quick_common_correlations_test.ps1 -DataPath "data/preprocessed" -Verbose
```

### Full Analysis
```powershell
.\powershell codes\run_common_correlations_analysis.ps1 -DataPath "data/preprocessed" -CorrelationThreshold 0.5 -MinDatasets 3
```

### Direct Python Usage
```bash
python batteryml/chemistry_data_analysis/statistical_analysis/find_common_statistical_correlations.py \
    --data_path "data/preprocessed" \
    --output_dir "common_correlations_results" \
    --correlation_threshold 0.5 \
    --min_datasets 3 \
    --verbose
```

## Parameters

- `--data_path`: Path to preprocessed data directory (default: "data/preprocessed")
- `--output_dir`: Output directory for results (default: "common_correlations_results")
- `--correlation_threshold`: Minimum absolute correlation threshold (default: 0.5)
- `--min_datasets`: Minimum number of datasets required for common features (default: 3)
- `--verbose`: Enable verbose logging

## Statistical Features Analyzed

The analysis calculates 7 statistical measures for each of 17 cycle features:

### Cycle Features
- `avg_c_rate` - Average C-rate
- `max_temperature` - Maximum temperature
- `max_discharge_capacity` - Maximum discharge capacity
- `max_charge_capacity` - Maximum charge capacity
- `avg_discharge_capacity` - Average discharge capacity
- `avg_charge_capacity` - Average charge capacity
- `charge_cycle_length` - Charge cycle length
- `discharge_cycle_length` - Discharge cycle length
- `peak_cv_length` - Peak CV length
- `cycle_length` - Total cycle length
- `power_during_charge_cycle` - Power during charge
- `power_during_discharge_cycle` - Power during discharge
- `avg_charge_c_rate` - Average charge C-rate
- `avg_discharge_c_rate` - Average discharge C-rate
- `charge_to_discharge_time_ratio` - Charge/discharge time ratio
- `avg_voltage` - Average voltage
- `avg_current` - Average current

### Statistical Measures
- `mean` - Mean value across cycles
- `std` - Standard deviation across cycles
- `min` - Minimum value across cycles
- `max` - Maximum value across cycles
- `median` - Median value across cycles
- `q25` - 25th percentile across cycles
- `q75` - 75th percentile across cycles

## Output Format

### Correlation Summary CSV
```csv
feature,dataset,correlation,abs_correlation,p_value,n_samples,dataset_count
avg_c_rate_mean,MATR,0.723,0.723,0.001,45,3
avg_c_rate_mean,CALCE,0.689,0.689,0.002,32,3
avg_c_rate_mean,HNEI,0.712,0.712,0.001,28,3
...
```

### Common Features Summary CSV
```csv
feature,dataset_count,datasets,avg_abs_correlation,max_abs_correlation,min_abs_correlation
avg_c_rate_mean,3,"MATR, CALCE, HNEI",0.708,0.723,0.689
max_temperature_std,4,"MATR, CALCE, HNEI, OX",0.654,0.712,0.598
...
```

## Visualizations

### Correlation Heatmap
- Shows correlations between statistical features and log(RUL) across datasets
- Color-coded: Red (negative), Blue (positive), White (no correlation)
- Sorted by average absolute correlation

### Feature Frequency Plot
- Shows how many datasets each feature appears in with high correlation
- Color-coded: Green (80%+ datasets), Orange (50%+ datasets), Red (<50% datasets)

## Expected Results

The analysis typically finds 20-50 statistical features with high correlations common across multiple datasets, such as:

1. **Capacity-related features**: `max_discharge_capacity_mean`, `avg_discharge_capacity_std`
2. **Rate-related features**: `avg_c_rate_mean`, `avg_discharge_c_rate_std`
3. **Temperature features**: `max_temperature_std`, `max_temperature_max`
4. **Cycle length features**: `cycle_length_mean`, `discharge_cycle_length_std`
5. **Power features**: `power_during_discharge_cycle_mean`

## Interpretation

### High Correlation Features
- **Positive correlation**: Feature increases as RUL decreases (degradation indicator)
- **Negative correlation**: Feature decreases as RUL decreases (capacity indicator)

### Common Features
- Features appearing in 3+ datasets are more robust and generalizable
- Features with consistent correlation direction across datasets are more reliable
- Features with high average correlation are stronger predictors

## Troubleshooting

### No Common Features Found
- Lower `correlation_threshold` (e.g., 0.3)
- Lower `min_datasets` (e.g., 2)
- Check if datasets have sufficient data

### Missing Datasets
- Verify dataset directories exist in `data/preprocessed/`
- Check for PKL files in each dataset directory
- Ensure dataset names match expected format

### Memory Issues
- Process datasets individually
- Reduce cycle limit if available
- Use smaller correlation threshold

## Integration with Training

The common features identified can be used to:
1. **Feature selection** for RUL prediction models
2. **Cross-dataset validation** of model performance
3. **Robust feature engineering** for new datasets
4. **Chemistry-specific analysis** using identified features

## Example Workflow

1. **Run analysis**: `.\powershell codes\run_common_correlations_analysis.ps1`
2. **Review results**: Check `common_features_summary.csv`
3. **Select features**: Choose top features for training
4. **Train models**: Use selected features in training scripts
5. **Validate**: Test on new datasets using common features
