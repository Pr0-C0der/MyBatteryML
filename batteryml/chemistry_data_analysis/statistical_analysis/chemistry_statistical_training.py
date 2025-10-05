#!/usr/bin/env python3
"""
Chemistry-Specific Statistical Feature Training for RUL Prediction

This script performs chemistry-specific training using statistical features,
following the pattern from chemistry_training.py but with statistical feature extraction
instead of raw cycle features.

Key features:
- Works with chemistry folders (e.g., LFP, NMC, NCA) containing multiple datasets
- Creates chemistry-level train/test splits across datasets within the chemistry
- Uses statistical features (mean, std, min, max, etc.) aggregated per battery
- Supports manual feature selection for training
- Battery-level training only (cycle level removed for simplicity)
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
from sklearn.model_selection import train_test_split, GroupKFold, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.signal import medfilt, savgol_filter

# Optional imports for advanced models
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from batteryml.models.rul_predictors.cnn import CNNRULPredictor
    from batteryml.models.rul_predictors.lstm import LSTMRULPredictor
    from batteryml.models.rul_predictors.transformer import TransformerRULPredictor
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import cuml  # noqa: F401
    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

try:
    import torch  # noqa: F401
    import gpytorch  # noqa: F401
    _HAS_GPYTORCH = True
except Exception:
    _HAS_GPYTORCH = False

# Import the working correlation analyzer
from feature_rul_correlation import FeatureRULCorrelationAnalyzer
from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class

# Suppress warnings
warnings.filterwarnings('ignore')


class ChemistryStatisticalTrainer:
    """Chemistry-specific trainer for RUL prediction using statistical features.
    
    This trainer focuses on battery-level training with manual feature selection.
    Cycle-level training has been removed for simplicity.
    """

    def __init__(self, data_path: str, output_dir: str = 'chemistry_statistical_results', 
                 verbose: bool = False, dataset_hint: Optional[str] = None, 
                 cycle_limit: Optional[int] = None, smoothing: str = 'none', ma_window: int = 5, 
                 use_gpu: bool = False, train_test_ratio: float = 0.7, 
                 manual_features: Optional[List[str]] = None, 
                 direct_statistical_features: Optional[List[str]] = None):
        """
        Initialize chemistry statistical trainer.
        
        Args:
            cycle_limit: Limit analysis to first N cycles (useful for early-cycle prediction)
            manual_features: List of base cycle features (e.g., ['charge_cycle_length', 'avg_c_rate']).
                           Each feature will be expanded with all statistical measures 
                           (mean, std, min, max, median, q25, q75, skewness, kurtosis).
                           So 'charge_cycle_length' becomes 'mean_charge_cycle_length', 
                           'std_charge_cycle_length', etc.
            direct_statistical_features: List of specific statistical features (e.g., ['mean_cycle_length', 'median_charge_cycle_length']).
                                       If provided, this overrides manual_features and uses these exact features.
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.cycle_limit = cycle_limit
        self.smoothing = str(smoothing or 'none').lower()
        self.ma_window = int(ma_window) if int(ma_window) > 1 else 5
        self.use_gpu = bool(use_gpu)
        self.train_test_ratio = float(train_test_ratio)
        self.manual_features = manual_features
        self.direct_statistical_features = direct_statistical_features
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Chemistry name from data path
        self.chemistry_name = self.data_path.name
        
        # Initialize correlation analyzer
        self.correlation_analyzer = FeatureRULCorrelationAnalyzer(str(self.data_path.parent))
        self.rul_annotator = RULLabelAnnotator()
        
        # Statistical measures to calculate
        self.statistical_measures = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skewness', 'kurtosis']
        
        # Initialize RMSE tracking
        self.rmse_file = self.output_dir / "RMSE.csv"
        
        # Create output subdirectories (only battery level now)
        self.battery_level_dir = self.output_dir / self.chemistry_name / 'battery_level'
        self.battery_level_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle feature selection logic
        if self.direct_statistical_features is not None:
            # Use direct statistical features (e.g., ['mean_cycle_length', 'median_charge_cycle_length'])
            self.statistical_feature_names = self.direct_statistical_features
            # Extract base features from statistical feature names for cycle feature extraction
            self.base_features_for_extraction = []
            for stat_feature in self.direct_statistical_features:
                # Extract base feature name (everything before the last underscore)
                parts = stat_feature.split('_')
                if len(parts) >= 2:
                    base_feature = '_'.join(parts[:-1])  # Everything except the last part (statistical measure)
                    if base_feature not in self.base_features_for_extraction:
                        self.base_features_for_extraction.append(base_feature)
        else:
            # Default base feature names if none provided (these will be expanded with statistical measures)
            if self.manual_features is None:
                self.manual_features = [
                    'avg_c_rate', 'max_charge_capacity', 'avg_discharge_capacity', 'avg_charge_capacity', 
                    'charge_cycle_length', 'discharge_cycle_length', 'cycle_length', 
                    'power_during_charge_cycle', 'power_during_discharge_cycle',
                    'avg_charge_c_rate', 'avg_discharge_c_rate', 'charge_to_discharge_time_ratio',
                    'avg_voltage', 'avg_current'
                ]
            
            # Generate all statistical feature names from base features
            self.statistical_feature_names = []
            for base_feature in self.manual_features:
                for measure in self.statistical_measures:
                    self.statistical_feature_names.append(f"{base_feature}_{measure}")
            
            self.base_features_for_extraction = self.manual_features

    def _battery_files(self) -> List[Path]:
        """Get all battery files in the chemistry folder."""
        return sorted(self.data_path.glob('*.pkl'))

    def _infer_dataset_for_battery(self, battery: BatteryData) -> Optional[str]:
        """Infer dataset for a battery."""
        if self.dataset_hint:
            return str(self.dataset_hint).upper()
        
        # Try tokens in metadata - handle both UL_PUR and UL-PUR patterns
        tokens = ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'UL-PUR', 'HUST', 'OX']
        
        def _txt(x):
            try:
                return str(x).upper()
            except Exception:
                return ''
        
        for source in [battery.cell_id, getattr(battery, 'reference', ''), getattr(battery, 'description', '')]:
            s = _txt(source)
            for t in tokens:
                if t in s:
                    # Normalize UL-PUR to UL_PUR for consistency
                    return 'UL_PUR' if t in ['UL_PUR', 'UL-PUR'] else t
        return None

    @staticmethod
    def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
        """Moving average smoothing."""
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            kernel = np.ones(w, dtype=float)
            mask = np.isfinite(arr).astype(float)
            arr0 = np.nan_to_num(arr, nan=0.0)
            num = np.convolve(arr0, kernel, mode='same')
            den = np.convolve(mask, kernel, mode='same')
            out = num / np.maximum(den, 1e-8)
            out[den < 1e-8] = np.nan
            return out
        except Exception:
            return y

    @staticmethod
    def _moving_median(y: np.ndarray, window: int) -> np.ndarray:
        """Moving median smoothing."""
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            if w < 1:
                return arr
            pad = w // 2
            padded = np.pad(arr, (pad, pad), mode='edge')
            out = np.empty_like(arr)
            for i in range(arr.size):
                seg = padded[i:i + w]
                m = np.isfinite(seg)
                out[i] = np.nanmedian(seg[m]) if np.any(m) else np.nan
            return out
        except Exception:
            return y

    @staticmethod
    def _hampel_filter(y: np.ndarray, window_size: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
        """Hampel filter for outlier detection."""
        arr = np.asarray(y, dtype=float)
        n = arr.size
        if n == 0 or window_size < 3:
            return arr
        w = int(window_size)
        half = w // 2
        out = arr.copy()
        for i in range(n):
            l = max(0, i - half)
            r = min(n, i + half + 1)
            seg = arr[l:r]
            m = np.isfinite(seg)
            if not np.any(m):
                continue
            med = float(np.nanmedian(seg[m]))
            mad = float(np.nanmedian(np.abs(seg[m] - med)))
            if mad <= 0:
                continue
            thr = n_sigmas * 1.4826 * mad
            if np.isfinite(arr[i]) and abs(arr[i] - med) > thr:
                out[i] = med
        return out

    @staticmethod
    def _hms_filter(y: np.ndarray) -> np.ndarray:
        """HMS filter: Hampel -> Median -> Savitzky-Golay."""
        try:
            arr = np.asarray(y, dtype=float)
            if arr.size == 0:
                return arr
            # 1) Hampel
            h = ChemistryStatisticalTrainer._hampel_filter(arr, window_size=11, n_sigmas=3.0)
            # 2) Median filter (size=5)
            try:
                m = medfilt(h, kernel_size=5)
            except Exception:
                m = h
            # 3) Savitzky–Golay (window_length=101, polyorder=3), adjusted to length
            wl = 101
            if m.size < wl:
                wl = m.size if m.size % 2 == 1 else max(1, m.size - 1)
            if wl >= 5 and wl > 3:
                try:
                    s = savgol_filter(m, window_length=wl, polyorder=3, mode='interp')
                except Exception:
                    s = m
            else:
                s = m
            return s
        except Exception:
            return y

    def apply_smoothing(self, values: np.ndarray, method: str = None, window_size: int = 5) -> np.ndarray:
        """Apply smoothing to the values."""
        if method is None or method.lower() == 'none':
            return values
        
        try:
            if method == 'ma':
                return self._moving_average(values, window_size)
            elif method == 'median':
                return self._moving_median(values, window_size)
            elif method == 'hms':
                return self._hms_filter(values)
            else:
                print(f"Warning: Unknown smoothing method '{method}', returning original values")
                return values
        except Exception as e:
            print(f"Warning: Smoothing failed with method {method}: {e}")
            return values

    def _build_cycle_feature_table_extractor(self, battery: BatteryData, feature_names: List[str]) -> pd.DataFrame:
        """Build cycle feature table using chemistry-specific extractor."""
        ds = self._infer_dataset_for_battery(battery)
        if not ds:
            return pd.DataFrame()
        
        cls = get_extractor_class(ds)
        if cls is None:
            return pd.DataFrame()
        
        extractor = cls()
        rows: List[Dict[str, float]] = []
        
        # Apply cycle limit if specified
        cycles_to_process = battery.cycle_data
        if self.cycle_limit is not None and self.cycle_limit > 0:
            cycles_to_process = cycles_to_process[:self.cycle_limit]
        
        for c in cycles_to_process:
            row: Dict[str, float] = {'cycle_number': c.cycle_number}
            for name in feature_names:
                fn = getattr(extractor, name, None)
                if not callable(fn):
                    continue
                try:
                    val = fn(battery, c)
                    row[name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
                except Exception:
                    row[name] = np.nan
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Apply smoothing to feature columns (not cycle_number)
        if self.smoothing != 'none' and len(df) > 1:
            feature_cols = [col for col in df.columns if col != 'cycle_number']
            for col in feature_cols:
                if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    series = df[col].values
                    if self.smoothing == 'ma':
                        smoothed = self._moving_average(series, self.ma_window)
                    elif self.smoothing == 'median':
                        smoothed = self._moving_median(series, self.ma_window)
                    elif self.smoothing == 'hms':
                        smoothed = self._hms_filter(series)
                    else:
                        smoothed = series
                    df[col] = smoothed
        
        return df

    def _create_chemistry_train_test_split(self, seed: int = 42) -> Tuple[List[Path], List[Path]]:
        """Create chemistry-level train/test split: specified ratio of cells from each dataset for training, remainder for testing."""
        files = self._battery_files()
        if len(files) < 2:
            if self.verbose:
                print("Not enough files for train/test split")
            return [], []
        
        # Group files by dataset
        dataset_groups: Dict[str, List[Path]] = {}
        for f in tqdm(files, desc="Grouping batteries by dataset", unit="battery"):
            try:
                battery = BatteryData.load(f)
                dataset = self._infer_dataset_for_battery(battery)
                if dataset is None:
                    dataset = "unknown"
                if dataset not in dataset_groups:
                    dataset_groups[dataset] = []
                dataset_groups[dataset].append(f)
            except Exception:
                # If we can't load the battery, put it in unknown group
                if "unknown" not in dataset_groups:
                    dataset_groups["unknown"] = []
                dataset_groups["unknown"].append(f)
        
        if self.verbose:
            print(f"Found datasets in chemistry: {list(dataset_groups.keys())}")
            for dataset, files_list in dataset_groups.items():
                print(f"  {dataset}: {len(files_list)} cells")
        
        # Split each dataset group
        train_files: List[Path] = []
        test_files: List[Path] = []
        
        np.random.seed(seed)
        for dataset, files_list in dataset_groups.items():
            if len(files_list) < 2:
                # If only 1 cell, put it in training
                train_files.extend(files_list)
                if self.verbose:
                    print(f"  {dataset}: Only 1 cell, adding to training")
                continue
            
            # Shuffle files for this dataset
            files_shuffled = files_list.copy()
            np.random.shuffle(files_shuffled)
            
            # Calculate split point
            split_idx = int(self.train_test_ratio * len(files_shuffled))
            
            # Split
            train_files.extend(files_shuffled[:split_idx])
            test_files.extend(files_shuffled[split_idx:])
            
            if self.verbose:
                print(f"  {dataset}: {len(files_shuffled[:split_idx])} train, {len(files_shuffled[split_idx:])} test")
        
        if self.verbose:
            print(f"Total: {len(train_files)} train files, {len(test_files)} test files")
        
        return train_files, test_files

    def _calculate_statistical_features(self, battery_files: List[Path], feature_names: List[str]) -> pd.DataFrame:
        """Calculate statistical features for all batteries."""
        print("Calculating statistical features...")
        
        all_battery_data = []
        
        for f in tqdm(battery_files, desc="Processing batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load {f}: {e}")
                continue
            
            # Build cycle feature table using chemistry-specific extractor
            try:
                df = self._build_cycle_feature_table_extractor(battery, feature_names)
            except Exception:
                df = pd.DataFrame()
            
            if df.empty:
                continue
            
            # Calculate statistical measures for each feature
            battery_row = {'battery_id': battery.cell_id, 'dataset': self._infer_dataset_for_battery(battery)}
            
            if self.direct_statistical_features is not None:
                # Use direct statistical features
                for stat_feature in self.statistical_feature_names:
                    # Extract base feature and measure from statistical feature name
                    parts = stat_feature.split('_')
                    if len(parts) >= 2:
                        base_feature = '_'.join(parts[:-1])
                        measure = parts[-1]
                        
                        if base_feature in df.columns:
                            feature_values = df[base_feature].dropna().values
                            
                            if len(feature_values) > 0:
                                # Apply smoothing if specified
                                if self.smoothing != 'none':
                                    feature_values = self.apply_smoothing(feature_values, self.smoothing, self.ma_window)
                                
                                stat_value = self._calculate_statistical_measure(feature_values, measure)
                                battery_row[stat_feature] = stat_value
                            else:
                                battery_row[stat_feature] = np.nan
                        else:
                            battery_row[stat_feature] = np.nan
                    else:
                        battery_row[stat_feature] = np.nan
            else:
                # Use base features and expand with all statistical measures
                for feature in feature_names:
                    if feature in df.columns:
                        feature_values = df[feature].dropna().values
                        
                        if len(feature_values) > 0:
                            # Apply smoothing if specified
                            if self.smoothing != 'none':
                                feature_values = self.apply_smoothing(feature_values, self.smoothing, self.ma_window)
                            
                            for measure in self.statistical_measures:
                                stat_value = self._calculate_statistical_measure(feature_values, measure)
                                battery_row[f"{feature}_{measure}"] = stat_value
                        else:
                            # Fill with NaN if no valid values
                            for measure in self.statistical_measures:
                                battery_row[f"{feature}_{measure}"] = np.nan
                    else:
                        # Fill with NaN if feature not available
                        for measure in self.statistical_measures:
                            battery_row[f"{feature}_{measure}"] = np.nan
            
            # Calculate RUL
            try:
                rul_tensor = self.rul_annotator.process_cell(battery)
                total_rul = int(rul_tensor.item()) if np.isfinite(float(rul_tensor.item())) else 0
                battery_row['rul'] = total_rul
                battery_row['log_rul'] = np.log(total_rul + 1) if total_rul > 0 else np.nan
            except Exception:
                battery_row['rul'] = 0
                battery_row['log_rul'] = np.nan
            
            if battery_row['rul'] > 0:  # Only include batteries with valid RUL
                all_battery_data.append(battery_row)
        
        if not all_battery_data:
            return pd.DataFrame()
        
        result = pd.DataFrame(all_battery_data)
        print(f"Statistical features calculated for {len(result)} batteries")
        print(f"Output shape: {result.shape}")
        
        return result

    def _calculate_statistical_measure(self, values: np.ndarray, measure: str) -> float:
        """Calculate a statistical measure for given values."""
        try:
            if measure == 'mean':
                return np.mean(values)
            elif measure == 'std':
                return np.std(values)
            elif measure == 'min':
                return np.min(values)
            elif measure == 'max':
                return np.max(values)
            elif measure == 'median':
                return np.median(values)
            elif measure == 'q25':
                return np.percentile(values, 25)
            elif measure == 'q75':
                return np.percentile(values, 75)
            elif measure == 'skewness':
                from scipy.stats import skew
                return skew(values)
            elif measure == 'kurtosis':
                from scipy.stats import kurtosis
                return kurtosis(values)
            else:
                return np.nan
        except Exception:
            return np.nan

    def _build_models(self, use_gpu: bool = False) -> Dict[str, Pipeline]:
        """Build all available models for training."""
        models: Dict[str, Pipeline] = {}
        base_steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler())]

        # Determine if GPU is truly available
        gpu_available = False
        if use_gpu:
            try:
                import torch  # type: ignore
                gpu_available = bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())
            except Exception:
                gpu_available = False

        # Prefer cuML implementations only when GPU is available
        if gpu_available and _HAS_CUML:
            from cuml.linear_model import LinearRegression as cuLinearRegression, Ridge as cuRidge, ElasticNet as cuElasticNet
            from cuml.svm import SVR as cuSVR
            from cuml.ensemble import RandomForestRegressor as cuRF
            models['linear_regression'] = Pipeline(base_steps + [('model', cuLinearRegression())])
            models['ridge'] = Pipeline(base_steps + [('model', cuRidge())])
            models['elastic_net'] = Pipeline(base_steps + [('model', cuElasticNet())])
            models['svr'] = Pipeline(base_steps + [('model', cuSVR(kernel='rbf', C=10.0))])
            models['random_forest'] = Pipeline(base_steps + [('model', cuRF(n_estimators=40, random_state=42))])
        else:
            # Linear models (CPU)
            models['linear_regression'] = Pipeline(base_steps + [('model', LinearRegression())])
            models['ridge'] = Pipeline(base_steps + [('model', Ridge(alpha=1.0))])
            models['elastic_net'] = Pipeline(base_steps + [('model', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))])
            # Kernel (CPU)
            models['svr'] = Pipeline(base_steps + [('model', SVR(kernel='rbf', C=10.0, gamma='scale'))])
            # Trees (CPU)
            models['random_forest'] = Pipeline(base_steps + [('model', RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1))])
        
        # Shallow MLP
        models['mlp'] = Pipeline(base_steps + [('model', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', batch_size=256, max_iter=300, random_state=42))])
        
        # XGBoost
        if _HAS_XGB:
            xgb_device = 'cuda' if gpu_available else 'cpu'
            models['xgboost'] = Pipeline(base_steps + [('model', XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                device=xgb_device
            ))])
        
        # PLSR
        models['plsr'] = Pipeline(base_steps + [('model', PLSRegression(n_components=10))])
        
        # PCR (PCA + Linear)
        models['pcr'] = Pipeline(base_steps + [('model', Pipeline([('pca', PCA(n_components=20)), ('lr', LinearRegression())]))])

        return models

    def _save_rmse_results(self, results: Dict[str, float], model_type: str, features: List[str], 
                          window_size: int = None, tune: bool = False, cv_splits: int = 5):
        """Save RMSE results to CSV file with simple format: rows=models, columns=datasets."""
        # Round RMSE values to 2 decimal places
        rounded_results = {k: round(v, 2) for k, v in results.items()}
        
        # Create DataFrame with chemistry as column
        df = pd.DataFrame({self.chemistry_name: rounded_results})
        
        # Append to existing CSV or create new one
        if self.rmse_file.exists() and self.rmse_file.stat().st_size > 0:
            try:
                existing_df = pd.read_csv(self.rmse_file, index_col=0)
                # Merge with existing data, keeping all models and datasets
                combined_df = existing_df.join(df, how='outer')
                combined_df.to_csv(self.rmse_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # File exists but is empty or corrupted, create new one
                df.to_csv(self.rmse_file)
        else:
            df.to_csv(self.rmse_file)
        
        if self.verbose:
            print(f"RMSE results saved to {self.rmse_file}")

    def train_battery_level(self, tune: bool = False, cv_splits: int = 5) -> Dict[str, float]:
        """Train battery-level RUL prediction models using statistical features."""
        if self.verbose:
            print(f"Training battery-level statistical models for {self.chemistry_name}...")
            if self.direct_statistical_features is not None:
                print(f"Using direct statistical features: {self.direct_statistical_features}")
                print(f"Total features: {len(self.statistical_feature_names)}")
            else:
                print(f"Using base features: {self.manual_features}")
                print(f"Generating statistical measures: {self.statistical_measures}")
                print(f"Total statistical features: {len(self.statistical_feature_names)}")
        
        # Use base features for cycle feature extraction
        feature_names = self.base_features_for_extraction
        
        # Create chemistry-level train/test split
        train_files, test_files = self._create_chemistry_train_test_split()
        
        if not train_files:
            if self.verbose:
                print("No training files available")
            return {}
        
        # Calculate statistical features for training data
        train_data = self._calculate_statistical_features(train_files, feature_names)
        
        if train_data.empty:
            if self.verbose:
                print("No data available for battery-level training")
            return {}
        
        # Prepare training data
        feature_cols = [col for col in train_data.columns 
                       if col not in ['battery_id', 'dataset', 'rul', 'log_rul']]
        
        X_train = train_data[feature_cols].values
        y_train = train_data['log_rul'].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        if len(X_train) == 0:
            if self.verbose:
                print("No valid training data after removing NaNs")
            return {}
        
        if self.verbose:
            print(f"Battery-level training data: {X_train.shape}, Features: {len(feature_cols)}")
            print(f"Features used: {feature_cols[:5]}...")  # Show first 5 features
        
        # Prepare test data if available
        X_test, y_test = None, None
        if test_files:
            if self.verbose:
                print("Preparing test data...")
            test_data = self._calculate_statistical_features(test_files, feature_names)
            if not test_data.empty:
                X_test = test_data[feature_cols].values
                y_test = test_data['log_rul'].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
                X_test = X_test[valid_mask]
                y_test = y_test[valid_mask]
        
        # Build models
        models = self._build_models(use_gpu=self.use_gpu)
        
        # Train models
        results = {}
        for name, pipe in tqdm(models.items(), desc="Training battery-level models", unit="model"):
            if self.verbose:
                print(f"Training battery-level {name}...")
            
            try:
                est = pipe
                best_params = None
                
                if tune:
                    best_params = self._tune_hyperparameters(pipe, name, X_train, y_train, cv_splits)
                    if best_params:
                        est = clone(pipe).set_params(**best_params)
                
                # Train model
                est.fit(X_train, y_train)
                
                # Evaluate on test set if available
                if X_test is not None and X_test.size > 0:
                    y_pred = est.predict(X_test)
                    
                    # Convert to actual RUL for evaluation
                    y_test_actual = np.exp(y_test) - 1
                    y_pred_actual = np.exp(y_pred) - 1
                    
                    rmse = mean_squared_error(y_test_actual, y_pred_actual) ** 0.5
                    results[name] = rmse
                    if self.verbose:
                        print(f"Battery-level {name} RMSE: {rmse:.3f}")
                else:
                    results[name] = 0.0  # No test set available
                
            except Exception as e:
                if self.verbose:
                    print(f"Error training {name}: {e}")
                results[name] = np.inf
        
        # Save RMSE results to CSV
        self._save_rmse_results(results, "battery_level", feature_names, tune=tune, cv_splits=cv_splits)
        
        return results

    def _tune_hyperparameters(self, pipe: Pipeline, name: str, X_train: np.ndarray, 
                             y_train: np.ndarray, cv_splits: int) -> Optional[Dict]:
        """Tune hyperparameters for a model using cross-validation."""
        param_grids = {
            'linear_regression': {
                'model__fit_intercept': [True, False]
            },
            'ridge': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'model__fit_intercept': [True, False],
            },
            'elastic_net': {
                'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'model__max_iter': [5000, 10000, 50000],
            },
            'svr': {
                'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'model__C': [0.001, 0.01, 0.1, 1.0],
                'model__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'model__epsilon': [0.05, 0.1, 0.2],
                'model__degree': [2, 3, 4],
            },
            'random_forest': {
                'model__n_estimators': [200, 400, 800],
                'model__max_depth': [None, 10, 20, 40, 80],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2', None],
            },
            'xgboost': {
                'model__n_estimators': [300, 600],
                'model__max_depth': [4, 6, 8, 10, 20],
                'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
                'model__reg_alpha': [0.0, 0.001, 0.01],
                'model__reg_lambda': [1.0, 5.0, 10.0]
            },
            'mlp': {
                'model__hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
                'model__alpha': [1e-6, 1e-5, 1e-4],
                'model__learning_rate_init': [0.0005, 0.001, 0.01],
                'model__activation': ['relu', 'tanh'],
                'model__batch_size': [8, 32, 64, 128],
                'model__max_iter': [300, 900, 2000],
            },
            'plsr': {
                'model__n_components': [5, 10, 15, 20, 30],
            },
            'pcr': {
                'model__pca__n_components': [10, 20, 40, 60],
                'model__lr__fit_intercept': [True, False],
            }
        }
        
        if name not in param_grids:
            return None
        
        param_grid = param_grids[name]
        grid = list(ParameterGrid(param_grid))
        
        if not grid:
            return None
        
        print(f"  Tuning {name} with {len(grid)} configurations...")
        
        # Simple cross-validation
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=min(cv_splits, len(X_train)), shuffle=True, random_state=42)
        
        best_score = np.inf
        best_params = None
        
        for params in tqdm(grid, desc=f"Tuning {name}", leave=False):
            fold_rmses = []
            for train_idx, val_idx in cv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = clone(pipe).set_params(**params)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                
                # Convert to actual RUL for evaluation
                y_val_actual = np.exp(y_val) - 1
                pred_actual = np.exp(pred) - 1
                
                rmse = np.sqrt(mean_squared_error(y_val_actual, pred_actual))
                fold_rmses.append(rmse)
            
            mean_rmse = np.mean(fold_rmses)
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = params
        
        if best_params:
            print(f"  Best {name} params: {best_params} (RMSE: {best_score:.3f})")
        
        return best_params


def main():
    """Main function to run chemistry-specific statistical feature training."""
    parser = argparse.ArgumentParser(description='Chemistry-specific statistical feature training for RUL prediction')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to chemistry folder containing *.pkl files')
    parser.add_argument('--output_dir', type=str, default='chemistry_statistical_results', 
                       help='Output directory')
    parser.add_argument('--dataset_hint', type=str, default=None, 
                       help='Optional dataset name hint')
    parser.add_argument('--cycle_limit', type=int, default=None, 
                       help='Limit analysis to first N cycles (useful for early-cycle prediction)')
    parser.add_argument('--smoothing', type=str, default='none', 
                       choices=['none', 'ma', 'median', 'hms'], 
                       help='Smoothing method')
    parser.add_argument('--ma_window', type=int, default=5, 
                       help='Window size for moving average/median smoothing')
    parser.add_argument('--manual_features', type=str, nargs='+', default=None, 
                       help='Base cycle features to use (e.g., charge_cycle_length avg_c_rate). Each will be expanded with statistical measures (mean, std, min, max, median, q25, q75, skewness, kurtosis)')
    parser.add_argument('--direct_statistical_features', type=str, nargs='+', default=None, 
                       help='Direct statistical features to use (e.g., mean_cycle_length median_charge_cycle_length). Overrides manual_features if provided.')
    parser.add_argument('--use_gpu', action='store_true', 
                       help='Use GPU acceleration')
    parser.add_argument('--tune', action='store_true', 
                       help='Enable hyperparameter tuning')
    parser.add_argument('--cv_splits', type=int, default=5, 
                       help='Number of cross-validation splits')
    parser.add_argument('--train_test_ratio', type=float, default=0.7, 
                       help='Train/test split ratio (default: 0.7)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    trainer = ChemistryStatisticalTrainer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
        dataset_hint=args.dataset_hint,
        cycle_limit=args.cycle_limit,
        smoothing=args.smoothing,
        ma_window=args.ma_window,
        use_gpu=args.use_gpu,
        train_test_ratio=args.train_test_ratio,
        manual_features=args.manual_features,
        direct_statistical_features=args.direct_statistical_features
    )
    
    # Train battery-level models
    results = trainer.train_battery_level(args.tune, args.cv_splits)
    
    if args.verbose:
        print(f"Training completed for {trainer.chemistry_name}")
        print(f"Results: {results}")


if __name__ == '__main__':
    main()
