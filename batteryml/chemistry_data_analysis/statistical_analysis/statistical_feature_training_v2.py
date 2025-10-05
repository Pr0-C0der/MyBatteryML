#!/usr/bin/env python3
"""
Statistical Feature Training for RUL Prediction - Version 2

This script integrates with the existing working correlation analysis code
and adds machine learning training functionality.

Uses the proven FeatureRULCorrelationAnalyzer for data loading and correlation calculation.
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

# Suppress warnings
warnings.filterwarnings('ignore')


class StatisticalFeatureTrainerV2:
    """Trainer for RUL prediction using statistical features with proven correlation analysis."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the statistical feature trainer.
        
        Args:
            data_dir: Base directory containing the data
        """
        self.data_dir = data_dir
        self.correlation_analyzer = FeatureRULCorrelationAnalyzer(data_dir)
        self.model = None
        self.feature_names = None
        self.correlations = None
        
        # Statistical measures to calculate (same as working correlation analysis)
        self.statistical_measures = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
    
    @staticmethod
    def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
        """Moving average smoothing."""
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = max(1, int(window))
            # rolling mean
            result = np.full_like(arr, np.nan)
            for i in range(arr.size):
                start = max(0, i - w + 1)
                end = i + 1
                segment = arr[start:end]
                if np.any(np.isfinite(segment)):
                    result[i] = np.nanmean(segment)
            return result
        except Exception:
            return y
    
    @staticmethod
    def _moving_median(y: np.ndarray, window: int) -> np.ndarray:
        """Moving median smoothing."""
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = max(1, int(window))
            return medfilt(arr, kernel_size=min(w, arr.size))
        except Exception:
            return y
    
    @staticmethod
    def _hms_filter(y: np.ndarray, window: int) -> np.ndarray:
        """HMS filter: Hampel -> Median -> Savitzky-Golay."""
        try:
            arr = np.asarray(y, dtype=float)
            if arr.size == 0:
                return arr
            
            # 1) Hampel filter
            window_size = 11
            n_sigmas = 3.0
            half = window_size // 2
            col_h = arr.copy()
            for i in range(arr.size):
                l = max(0, i - half)
                r = min(arr.size, i + half + 1)
                seg = arr[l:r]
                fin = np.isfinite(seg)
                if not np.any(fin):
                    continue
                med = float(np.nanmedian(seg[fin]))
                mad = float(np.nanmedian(np.abs(seg[fin] - med)))
                if mad <= 0:
                    continue
                thr = n_sigmas * 1.4826 * mad
                if np.isfinite(arr[i]) and abs(arr[i] - med) > thr:
                    col_h[i] = med
            
            # 2) Median filter
            try:
                col_m = medfilt(col_h, kernel_size=5)
            except Exception:
                col_m = col_h
            
            # 3) Savitzky-Golay
            try:
                wl = 101
                if col_m.size < wl:
                    wl = col_m.size if col_m.size % 2 == 1 else max(1, col_m.size - 1)
                if wl >= 5:
                    col_s = savgol_filter(col_m, window_length=wl, polyorder=3, mode='interp')
                else:
                    col_s = col_m
            except Exception:
                col_s = col_m
            
            return col_s
        except Exception:
            return y
    
    def apply_smoothing(self, values: np.ndarray, method: str = None, window_size: int = 5) -> np.ndarray:
        """
        Apply smoothing to the values.
        
        Args:
            values: Input values
            method: Smoothing method ('hms', 'moving_mean', 'moving_median')
            window_size: Window size for smoothing (ignored for HMS)
            
        Returns:
            Smoothed values
        """
        if method is None or method.lower() == 'none':
            return values
        
        try:
            if method == 'moving_mean':
                return self._moving_average(values, window_size)
            elif method == 'moving_median':
                return self._moving_median(values, window_size)
            elif method == 'hms':
                return self._hms_filter(values, window_size)
            else:
                print(f"Warning: Unknown smoothing method '{method}', returning original values")
                return values
        except Exception as e:
            print(f"Warning: Smoothing failed with method {method}: {e}")
            return values
        
    def load_and_prepare_data(self, dataset_name: str, cycle_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load battery data and prepare it for training using the working correlation analyzer.
        
        Args:
            dataset_name: Name of the dataset
            cycle_limit: Maximum number of cycles to use (None for all)
            
        Returns:
            DataFrame with cycle features and RUL labels
        """
        print(f"Loading and preparing data for dataset: {dataset_name}")
        
        # Use the working correlation analyzer to load data
        data = self.correlation_analyzer.load_battery_data(dataset_name)
        print(f"Loaded {len(data)} cycle records from {data['battery_id'].nunique()} batteries")
        
        # Calculate RUL labels using the working method
        data_with_rul = self.correlation_analyzer.calculate_rul_labels(data, dataset_name)
        print(f"RUL calculated for {data_with_rul['battery_id'].nunique()} batteries")
        print(f"log_rul range: {data_with_rul['log_rul'].min():.3f} to {data_with_rul['log_rul'].max():.3f}")
        
        # Apply cycle limit if specified
        if cycle_limit is not None:
            original_count = len(data_with_rul)
            data_with_rul = data_with_rul[data_with_rul['cycle_number'] <= cycle_limit]
            print(f"Applied cycle limit {cycle_limit}: {original_count} -> {len(data_with_rul)} records")
        
        return data_with_rul
    
    def calculate_statistical_features(self, data: pd.DataFrame, smoothing_method: str = None, smoothing_window: int = 5) -> pd.DataFrame:
        """
        Calculate statistical features for each battery using the working correlation method.
        
        Args:
            data: DataFrame with cycle features and RUL labels
            smoothing_method: Smoothing method to apply ('hms', 'moving_mean', 'moving_median')
            smoothing_window: Window size for smoothing (ignored for HMS)
            
        Returns:
            DataFrame with statistical features (one row per battery)
        """
        print("Calculating statistical features...")
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in data.columns 
                       if col not in ['battery_id', 'cycle_number', 'log_rul', 'rul']]
        
        print(f"Processing {len(feature_cols)} features with {len(self.statistical_measures)} statistical measures")
        if smoothing_method and smoothing_method != 'none':
            print(f"Applying smoothing: {smoothing_method} (window={smoothing_window})")
        
        battery_stats = []
        
        for battery_id, battery_data in tqdm(data.groupby('battery_id'), desc="Processing batteries", unit="battery", leave=True):
            try:
                # Sort by cycle number for proper smoothing
                battery_data = battery_data.sort_values('cycle_number').reset_index(drop=True)
                
                # Calculate statistical measures for each feature
                battery_row = {'battery_id': battery_id}
                
                for feature in feature_cols:
                    feature_values = battery_data[feature].dropna().values
                    
                    if len(feature_values) > 0:
                        # Apply smoothing if specified
                        if smoothing_method and smoothing_method != 'none':
                            feature_values = self.apply_smoothing(feature_values, smoothing_method, smoothing_window)
                        
                        for measure in self.statistical_measures:
                            stat_value = self._calculate_statistical_measure(feature_values, measure)
                            battery_row[f"{feature}_{measure}"] = stat_value
                    else:
                        # Fill with NaN if no valid values
                        for measure in self.statistical_measures:
                            battery_row[f"{feature}_{measure}"] = np.nan
                
                # Add RUL information
                rul_values = battery_data['log_rul'].dropna().values
                if len(rul_values) > 0:
                    battery_row['log_rul'] = np.mean(rul_values)
                    battery_row['rul'] = np.exp(np.mean(rul_values)) - 1  # Convert back to actual RUL
                    battery_stats.append(battery_row)
                
            except Exception as e:
                print(f"Warning: Could not process battery {battery_id}: {e}")
                continue
        
        if not battery_stats:
            raise ValueError("No valid statistical features could be calculated")
        
        result = pd.DataFrame(battery_stats)
        print(f"Statistical features calculated for {len(result)} batteries")
        print(f"Output shape: {result.shape}")
        
        return result
    
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

        # Optional: GPyTorch SVGP (GPU scalable GP)
        if gpu_available and _HAS_GPYTORCH:
            class _SVGPRegressor:
                def __init__(self, inducing_points: int = 1024, batch_size: int = 2048, iters: int = 1500, lr: float = 1e-2):
                    self.m = int(inducing_points)
                    self.batch = int(batch_size)
                    self.iters = int(iters)
                    self.lr = float(lr)
                    self._predict_fn = None

                def fit(self, X, y):
                    import torch
                    import gpytorch
                    from torch.utils.data import TensorDataset, DataLoader

                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    X_t = torch.tensor(X, dtype=torch.float32, device=device)
                    y_t = torch.tensor(y, dtype=torch.float32, device=device)
                    y_mean = y_t.mean(); y_std = y_t.std().clamp_min(1e-6)
                    y_n = (y_t - y_mean) / y_std

                    class GPModel(gpytorch.models.ApproximateGP):
                        def __init__(self, inducing, d):
                            var = gpytorch.variational.VariationalStrategy(
                                self,
                                inducing,
                                gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0)),
                                learn_inducing_locations=True,
                            )
                            super().__init__(var)
                            self.mean_module = gpytorch.means.ConstantMean()
                            self.covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.RBFKernel(ard_num_dims=d)
                            )

                        def forward(self, x):
                            mean_x = self.mean_module(x)
                            covar_x = self.covar_module(x)
                            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

                    m = min(self.m, X_t.shape[0])
                    inducing = X_t[torch.randperm(X_t.shape[0])[:m]].contiguous()
                    model = GPModel(inducing, X_t.shape[1]).to(device)
                    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
                    model.train(); likelihood.train()
                    opt = torch.optim.Adam([
                        {'params': model.parameters()},
                        {'params': likelihood.parameters()}
                    ], lr=self.lr)
                    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_t.shape[0])

                    loader = DataLoader(TensorDataset(X_t, y_n), batch_size=self.batch, shuffle=True, drop_last=False)
                    for _ in tqdm(range(self.iters), desc='SVGP epochs', leave=False):
                        for xb, yb in tqdm(loader, desc='SVGP batches', leave=False):
                            opt.zero_grad(set_to_none=True)
                            out = model(xb)
                            loss = -mll(out, yb)
                            loss.backward()
                            opt.step()

                    model.eval(); likelihood.eval()

                    def _predict(Xtest):
                        Xt = torch.tensor(Xtest, dtype=torch.float32, device=device)
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            mean = likelihood(model(Xt)).mean
                        return (mean * y_std + y_mean).detach().cpu().numpy()

                    self._predict_fn = _predict
                    return self

                def predict(self, X):
                    if self._predict_fn is None:
                        raise RuntimeError('SVGP model not fitted')
                    return self._predict_fn(X)

            models['gpytorch_svgp'] = Pipeline(base_steps + [('model', _SVGPRegressor())])

        return models
    
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
            else:
                return np.nan
        except Exception:
            return np.nan
    
    def calculate_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between statistical features and RUL using the proven method.
        
        Args:
            data: DataFrame with statistical features
            
        Returns:
            DataFrame with correlation results
        """
        print("Calculating feature-RUL correlations...")
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in data.columns 
                       if col not in ['battery_id', 'log_rul', 'rul']]
        
        print(f"Calculating correlations for {len(feature_cols)} features")
        
        correlations = []
        
        for feature in tqdm(feature_cols, desc="Calculating correlations", unit="feature", leave=True):
            try:
                # Get valid data points
                valid_data = data[['log_rul', feature]].dropna()
                
                if len(valid_data) < 2:
                    continue
                
                # Check if feature has any variation
                if valid_data[feature].std() == 0:
                    continue
                
                # Calculate Spearman correlation using the proven method
                from scipy.stats import spearmanr
                correlation, _ = spearmanr(valid_data['log_rul'], valid_data[feature], nan_policy='omit')
                
                if not np.isnan(correlation):
                    correlations.append({
                        'feature': feature,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation),
                        'n_samples': len(valid_data)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {feature}: {e}")
                continue
        
        if not correlations:
            raise ValueError("No valid correlations could be calculated")
        
        correlation_df = pd.DataFrame(correlations)
        correlation_df = correlation_df.sort_values('abs_correlation', ascending=False)
        
        print(f"Calculated {len(correlations)} valid correlations")
        return correlation_df
    
    def select_top_features(self, correlation_df: pd.DataFrame, n_features: int = 15) -> List[str]:
        """
        Select top N features with highest absolute correlation to RUL.
        
        Args:
            correlation_df: DataFrame with correlation results
            n_features: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        print(f"Selecting top {n_features} features...")
        
        if correlation_df.empty:
            raise ValueError("No correlations available for feature selection")
        
        top_features = correlation_df.head(n_features)['feature'].tolist()
        
        print("Selected features:")
        for i, (_, row) in enumerate(correlation_df.head(n_features).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:40s} (correlation: {row['correlation']:6.3f}, n={row['n_samples']})")
        
        return top_features
    
    def prepare_training_data(self, data: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with selected features.
        
        Args:
            data: DataFrame with statistical features
            feature_names: List of selected feature names
            
        Returns:
            Tuple of (X, y) arrays
        """
        print("Preparing training data...")
        
        # Check if all features exist
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Prepare feature matrix
        X = data[feature_names].values
        y = data['log_rul'].values
        
        # Remove rows with NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Training data shape: {X.shape}")
        print(f"Target range: {y.min():.3f} to {y.max():.3f}")
        
        return X, y
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, 
                    use_gpu: bool = False, tune: bool = False, 
                    cv_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and evaluate them.
        
        Args:
            X_train: Training features
            y_train: Training labels (log RUL)
            X_test: Test features  
            y_test: Test labels (log RUL)
            use_gpu: Whether to use GPU acceleration
            tune: Whether to perform hyperparameter tuning
            cv_splits: Number of cross-validation splits for tuning
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        print("Training multiple models...")
        
        # Build all available models
        models = self._build_models(use_gpu=use_gpu)
        
        # Convert labels to actual RUL for evaluation
        y_train_actual = np.exp(y_train) - 1
        y_test_actual = np.exp(y_test) - 1
        
        results = {}
        
        for name, pipe in tqdm(models.items(), desc="Training models", unit="model", leave=True):
            print(f"\nTraining {name}...")
            
            try:
                # Hyperparameter tuning if requested
                best_params = None
                if tune:
                    best_params = self._tune_hyperparameters(pipe, name, X_train, y_train, cv_splits)
                    if best_params:
                        pipe = clone(pipe).set_params(**best_params)
                
                # Train model
                pipe.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = pipe.predict(X_train)
                y_test_pred = pipe.predict(X_test)
                
                # Convert predictions back to actual RUL
                y_train_pred_actual = np.exp(y_train_pred) - 1
                y_test_pred_actual = np.exp(y_test_pred) - 1
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
                test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
                train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
                test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
                train_mape = np.mean(np.abs((y_train_actual - y_train_pred_actual) / y_train_actual)) * 100
                test_mape = np.mean(np.abs((y_test_actual - y_test_pred_actual) / y_test_actual)) * 100
                train_r2 = r2_score(y_train_actual, y_train_pred_actual)
                test_r2 = r2_score(y_test_actual, y_test_pred_actual)
                
                # Store results
                results[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_mape': train_mape,
                    'test_mape': test_mape,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'y_test_actual': y_test_actual,
                    'y_test_pred_actual': y_test_pred_actual
                }
                
                print(f"  {name}: Test RMSE={test_rmse:.2f}, Test MAE={test_mae:.2f}, Test R²={test_r2:.3f}")
                
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                results[name] = {
                    'train_rmse': np.inf,
                    'test_rmse': np.inf,
                    'train_mae': np.inf,
                    'test_mae': np.inf,
                    'train_mape': np.inf,
                    'test_mape': np.inf,
                    'train_r2': -np.inf,
                    'test_r2': -np.inf,
                    'y_test_actual': y_test_actual,
                    'y_test_pred_actual': np.zeros_like(y_test_actual)
                }
        
        # Print summary
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10}")
        print("-" * 80)
        
        # Sort by test RMSE
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_rmse'])
        for name, metrics in sorted_results:
            if metrics['test_rmse'] != np.inf:
                print(f"{name:<20} {metrics['test_rmse']:<12.2f} {metrics['test_mae']:<12.2f} {metrics['test_r2']:<10.3f}")
            else:
                print(f"{name:<20} {'FAILED':<12} {'FAILED':<12} {'FAILED':<10}")
        
        print("="*80)
        
        # Store best model
        if sorted_results and sorted_results[0][1]['test_rmse'] != np.inf:
            best_name = sorted_results[0][0]
            self.model = models[best_name]
            print(f"\nBest model: {best_name}")
        
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
        
        # Simple cross-validation (since we don't have groups for statistical features)
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
    
    def plot_feature_importance(self, output_path: Optional[str] = None):
        """Plot feature importance from the trained model."""
        if self.model is None:
            print("No model trained yet!")
            return
        
        print("Plotting feature importance...")
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Feature Importance (Top Correlated Features)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Feature importance plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, 
                        output_path: Optional[str] = None):
        """Plot actual vs predicted RUL values."""
        if self.model is None:
            print("No model trained yet!")
            return
        
        print("Plotting predictions...")
        
        y_test_pred = self.model.predict(X_test)
        
        # Convert to actual RUL for plotting
        y_test_actual = np.exp(y_test)
        y_test_pred_actual = np.exp(y_test_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test_actual, y_test_pred_actual, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(y_test_actual.min(), y_test_pred_actual.min())
        max_val = max(y_test_actual.max(), y_test_pred_actual.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual RUL (cycles)')
        plt.ylabel('Predicted RUL (cycles)')
        plt.title('Actual vs Predicted RUL (Test Set)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_test_actual, y_test_pred_actual)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Predictions plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_predictions(self, y_test_actual: np.ndarray, y_test_pred_actual: np.ndarray, 
                        output_path: str):
        """Save detailed predictions to CSV file."""
        print("Saving detailed predictions...")
        
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'actual_rul': y_test_actual,
            'predicted_rul': y_test_pred_actual,
            'error': y_test_pred_actual - y_test_actual,
            'abs_error': np.abs(y_test_pred_actual - y_test_actual),
            'error_percentage': ((y_test_pred_actual - y_test_actual) / y_test_actual) * 100
        })
        
        # Add index
        predictions_df.index.name = 'sample_index'
        predictions_df = predictions_df.reset_index()
        
        # Save to CSV
        predictions_df.to_csv(output_path, index=False)
        print(f"Detailed predictions saved to: {output_path}")
        
        # Print summary statistics
        print(f"Saved {len(predictions_df)} predictions")
        print(f"Mean absolute error: {predictions_df['abs_error'].mean():.2f} cycles")
        print(f"Max absolute error: {predictions_df['abs_error'].max():.2f} cycles")
        print(f"Mean error percentage: {predictions_df['error_percentage'].abs().mean():.2f}%")
    
    def train_and_evaluate(self, dataset_name: str, cycle_limit: Optional[int] = None, 
                          n_features: int = 15, test_size: float = 0.3,
                          random_state: int = 42, smoothing_method: str = None,
                          smoothing_window: int = 5, use_gpu: bool = False,
                          tune: bool = False, cv_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            dataset_name: Name of the dataset
            cycle_limit: Maximum number of cycles to use
            n_features: Number of top features to select
            test_size: Test set size (0.3 for 70/30 split)
            random_state: Random state for reproducibility
            smoothing_method: Smoothing method ('hms', 'moving_mean', 'moving_median')
            smoothing_window: Window size for smoothing (ignored for HMS)
            use_gpu: Whether to use GPU acceleration
            tune: Whether to perform hyperparameter tuning
            cv_splits: Number of cross-validation splits for tuning
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        print(f"Starting statistical feature training for dataset: {dataset_name}")
        print("=" * 60)
        
        # Create progress bar for main pipeline steps
        pipeline_steps = [
            "Loading and preparing data",
            "Calculating statistical features", 
            "Calculating correlations",
            "Selecting top features",
            "Preparing training data",
            "Training multiple models"
        ]
        
        with tqdm(total=len(pipeline_steps), desc="Training Pipeline", unit="step", leave=True) as pbar:
            # Load and prepare data using the working correlation analyzer
            pbar.set_description("Loading and preparing data")
            data = self.load_and_prepare_data(dataset_name, cycle_limit)
            pbar.update(1)
            
            # Calculate statistical features with smoothing
            pbar.set_description("Calculating statistical features")
            statistical_data = self.calculate_statistical_features(data, smoothing_method, smoothing_window)
            pbar.update(1)
            
            # Calculate correlations and select features
            pbar.set_description("Calculating correlations")
            correlation_df = self.calculate_correlations(statistical_data)
            self.correlations = correlation_df
            pbar.update(1)
            
            pbar.set_description("Selecting top features")
            self.feature_names = self.select_top_features(correlation_df, n_features)
            pbar.update(1)
            
            # Prepare training data
            pbar.set_description("Preparing training data")
            X, y = self.prepare_training_data(statistical_data, self.feature_names)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"Training set size: {X_train.shape[0]}")
            print(f"Test set size: {X_test.shape[0]}")
            pbar.update(1)
            
            # Train multiple models
            pbar.set_description("Training multiple models")
            all_metrics = self.train_models(X_train, y_train, X_test, y_test, 
                                          use_gpu=use_gpu, tune=tune, cv_splits=cv_splits)
            pbar.update(1)
        
        return all_metrics


def main():
    """Main function to run statistical feature training."""
    parser = argparse.ArgumentParser(description='Train XGBoost on statistical features with highest RUL correlation')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--cycle_limit', '-c', type=int, default= 100, 
                       help='Maximum number of cycles to use (default: 100)')
    parser.add_argument('--n_features', '-n', type=int, default=15, 
                       help='Number of top features to select (default: 15)')
    parser.add_argument('--test_size', '-t', type=float, default=0.3, 
                       help='Test set size (default: 0.3 for 70/30 split)')
    parser.add_argument('--data_dir', '-d', default='data/preprocessed', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--output_dir', '-o', default='statistical_training_results', 
                       help='Output directory for plots (default: statistical_training_results)')
    parser.add_argument('--random_state', '-r', type=int, default=42, 
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--smoothing', choices=['none', 'hms', 'moving_mean', 'moving_median'], 
                       default='none', help='Smoothing method (default: none)')
    parser.add_argument('--smoothing_window', type=int, default=5, 
                       help='Window size for smoothing (ignored for HMS)')
    parser.add_argument('--gpu', action='store_true', 
                       help='Enable GPU acceleration for compatible models')
    parser.add_argument('--tune', action='store_true', 
                       help='Enable hyperparameter tuning with cross-validation')
    parser.add_argument('--cv_splits', type=int, default=5, 
                       help='Number of cross-validation splits for tuning (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize trainer
        trainer = StatisticalFeatureTrainerV2(args.data_dir)
        
        # Set smoothing method to None if 'none'
        smoothing_method = None if args.smoothing == 'none' else args.smoothing
        
        # Run training and evaluation
        all_metrics = trainer.train_and_evaluate(
            dataset_name=args.dataset_name,
            cycle_limit=args.cycle_limit,
            n_features=args.n_features,
            test_size=args.test_size,
            random_state=args.random_state,
            smoothing_method=smoothing_method,
            smoothing_window=args.smoothing_window,
            use_gpu=args.gpu,
            tune=args.tune,
            cv_splits=args.cv_splits
        )
        
        # Generate plots and save predictions
        if trainer.model is not None:
            print("Generating plots and saving predictions...")
            
            # Feature importance plot
            importance_path = output_dir / f"feature_importance_{args.dataset_name}.png"
            trainer.plot_feature_importance(str(importance_path))
            
            # Save results summary
            results_summary = []
            for model_name, model_metrics in all_metrics.items():
                if model_metrics['test_rmse'] != np.inf:
                    results_summary.append({
                        'model': model_name,
                        'test_rmse': model_metrics['test_rmse'],
                        'test_mae': model_metrics['test_mae'],
                        'test_r2': model_metrics['test_r2']
                    })
            
            if results_summary:
                results_df = pd.DataFrame(results_summary)
                results_df = results_df.sort_values('test_rmse')
                results_path = output_dir / f"model_comparison_{args.dataset_name}.csv"
                results_df.to_csv(results_path, index=False)
                print(f"Model comparison saved to: {results_path}")
            
            # Save detailed predictions for best model
            best_model_name = min(all_metrics.keys(), key=lambda k: all_metrics[k]['test_rmse'])
            best_metrics = all_metrics[best_model_name]
            
            if 'y_test_actual' in best_metrics and 'y_test_pred_actual' in best_metrics:
                predictions_csv_path = output_dir / f"detailed_predictions_{args.dataset_name}_{best_model_name}.csv"
                trainer.save_predictions(best_metrics['y_test_actual'], best_metrics['y_test_pred_actual'], 
                                       str(predictions_csv_path))
        
        print(f"\nResults saved to: {output_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
