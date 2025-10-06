# Cross-Chemistry Training for RUL Prediction

This module provides comprehensive cross-chemistry training capabilities for battery RUL (Remaining Useful Life) prediction. It trains models on one battery chemistry and evaluates their performance on different chemistries to assess generalization capabilities.

## 🎯 Overview

Cross-chemistry training helps answer important questions:
- How well do models trained on LFP batteries perform on NMC batteries?
- Can a model trained on one chemistry generalize to others?
- Which models are most robust across different battery types?

## 📁 Files

### Core Scripts
- **`cross_chemistry_training.py`** - Main Python script for cross-chemistry training
- **`run_cross_chemistry_training.ps1`** - PowerShell script for single experiments
- **`run_all_cross_chemistry_experiments.ps1`** - PowerShell script for comprehensive experiments

### Output Files
- **`cross_chemistry_RMSE.csv`** - Root Mean Square Error results
- **`cross_chemistry_MAE.csv`** - Mean Absolute Error results
- **`cross_chemistry_MAPE.csv`** - Mean Absolute Percentage Error results

## 🚀 Quick Start

### Single Experiment
```powershell
# Train on LFP, test on NMC and NCA
.\run_cross_chemistry_training.ps1 -TrainChemistry "data_chemistries/lfp" -TestChemistries @("data_chemistries/nmc", "data_chemistries/nca")
```

### Comprehensive Experiments
```powershell
# Run all possible cross-chemistry combinations
.\run_all_cross_chemistry_experiments.ps1
```

### Python Direct Usage
```python
from batteryml.chemistry_data_analysis.cross_chemistry_training import CrossChemistryTrainer

# Create trainer
trainer = CrossChemistryTrainer(
    train_chemistry_path="data_chemistries/lfp",
    test_chemistry_paths=["data_chemistries/nmc", "data_chemistries/nca"],
    output_dir="my_cross_chemistry_results",
    verbose=True
)

# Train and evaluate
results = trainer.train_and_evaluate()
```

## 🔧 Parameters

### Required Parameters
- **`--train_chemistry`** - Path to training chemistry folder (e.g., `data_chemistries/lfp`)
- **`--test_chemistries`** - List of test chemistry folder paths

### Optional Parameters
- **`--output_dir`** - Output directory for results (default: `cross_chemistry_results`)
- **`--dataset_hint`** - Optional dataset name hint for feature extraction
- **`--cycle_limit`** - Limit analysis to first N cycles (default: 0 = all cycles)
- **`--smoothing`** - Smoothing method: `none`, `ma`, `median`, `hms` (default: `none`)
- **`--ma_window`** - Window size for smoothing (default: 5)
- **`--use_gpu`** - Enable GPU acceleration
- **`--tune`** - Enable hyperparameter tuning
- **`--cv_splits`** - Number of cross-validation splits (default: 5)
- **`--verbose`** - Enable verbose logging

## 🤖 Supported Models

The cross-chemistry trainer includes all models from `chemistry_training.py`:

### Linear Models
- **Linear Regression** - Basic linear model
- **Ridge Regression** - L2 regularized linear model
- **Elastic Net** - L1+L2 regularized linear model

### Advanced Models
- **Support Vector Regression (SVR)** - Kernel-based regression
- **Random Forest** - Ensemble of decision trees
- **Multi-layer Perceptron (MLP)** - Neural network
- **XGBoost** - Gradient boosting (with GPU support)
- **Partial Least Squares (PLSR)** - Dimensionality reduction + regression
- **Principal Component Regression (PCR)** - PCA + linear regression

### GPU Acceleration
When `--use_gpu` is enabled:
- Uses cuML implementations for linear models, SVR, and Random Forest
- XGBoost uses GPU tree method
- Falls back to CPU implementations if GPU unavailable

## 📊 Output Format

### CSV Files
Each experiment generates three CSV files with the same structure:

```csv
Model,lfp,nmc,nca,lco,mixed_nmc_lco
linear_regression,45.23,52.18,48.91,51.34,49.67
ridge,44.87,51.92,48.45,50.98,49.23
random_forest,42.15,49.78,46.32,48.91,47.56
...
```

- **Rows**: Model names
- **Columns**: Test chemistry names
- **Values**: Metric values (RMSE, MAE, or MAPE)

### Results Structure
```python
{
    'RMSE': {
        'linear_regression': {'lfp': 45.23, 'nmc': 52.18, ...},
        'ridge': {'lfp': 44.87, 'nmc': 51.92, ...},
        ...
    },
    'MAE': { ... },
    'MAPE': { ... }
}
```

## 🔬 Experimental Design

### Comprehensive Experiments
The `run_all_cross_chemistry_experiments.ps1` script runs:

1. **Train LFP → Test NMC, NCA, LCO, Mixed**
2. **Train NMC → Test LFP, NCA, LCO, Mixed**
3. **Train NCA → Test LFP, NMC, LCO, Mixed**
4. **Train LCO → Test LFP, NMC, NCA, Mixed**
5. **Train Mixed → Test LFP, NMC, NCA, LCO**

### Analysis Insights
- **Within-chemistry performance**: How well models perform on their training chemistry
- **Cross-chemistry performance**: How well models generalize to other chemistries
- **Model robustness**: Which models maintain performance across chemistries
- **Chemistry similarity**: Which chemistries are most similar based on model performance

## 📈 Example Results

### RMSE Results (Lower is Better)
```
Model              lfp    nmc    nca    lco    mixed
linear_regression  45.23  52.18  48.91  51.34  49.67
ridge             44.87  51.92  48.45  50.98  49.23
random_forest     42.15  49.78  46.32  48.91  47.56
xgboost           41.89  49.23  45.98  48.67  47.12
```

### MAPE Results (Lower is Better, %)
```
Model              lfp    nmc    nca    lco    mixed
linear_regression  12.34  15.67  13.89  14.56  14.12
ridge             12.01  15.23  13.45  14.23  13.78
random_forest     11.23  14.56  12.78  13.45  13.01
xgboost           10.98  14.23  12.45  13.12  12.67
```

## 🛠️ Advanced Usage

### Custom Feature Selection
```python
# Modify the _prepare_training_data method to use specific features
def _prepare_training_data(self, selected_features=None):
    # ... existing code ...
    if selected_features:
        feature_cols = [col for col in df.columns if col in selected_features]
    # ... rest of method ...
```

### Custom Model Configuration
```python
def _build_models(self):
    models = {}
    # Add your custom models here
    models['my_custom_model'] = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('model', MyCustomRegressor())
    ])
    return models
```

### Batch Processing
```powershell
# Process multiple cycle limits
foreach ($limit in @(50, 100, 200, 500)) {
    .\run_cross_chemistry_training.ps1 -TrainChemistry "data_chemistries/lfp" -TestChemistries @("data_chemistries/nmc") -CycleLimit $limit -OutputDir "results_cycles_$limit"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **No battery files found**
   - Check that chemistry directories contain `.pkl` files
   - Verify file paths are correct

2. **Feature extraction errors**
   - Ensure dataset hint is correct for your data
   - Check that cycle data is properly formatted

3. **GPU errors**
   - Verify CUDA installation
   - Check GPU memory availability
   - Use `--use_gpu` only when GPU is available

4. **Memory issues**
   - Use `--cycle_limit` to reduce data size
   - Process chemistries individually instead of all at once

### Performance Tips

1. **Use GPU acceleration** when available
2. **Limit cycles** for initial experiments
3. **Enable hyperparameter tuning** for better results
4. **Use verbose mode** to monitor progress

## 📚 Related Documentation

- [Chemistry Training Guide](ANALYSIS_GUIDE.md)
- [Feature Extraction](batteryml/chemistry_data_analysis/cycle_features.py)
- [Correlation Analysis](batteryml/chemistry_data_analysis/correlation_mod.py)

## 🤝 Contributing

To add new models or features:
1. Modify `_build_models()` method
2. Update parameter validation
3. Test with different chemistry combinations
4. Update documentation

## 📄 License

This module is part of the MyBatteryML project and follows the same license terms.
