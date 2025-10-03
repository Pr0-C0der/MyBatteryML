from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
import argparse

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.chemistry_data_analysis.cycle_features import (
    get_extractor_class,
    DatasetSpecificCycleFeatures,
)
from batteryml.training.train_rul_windows import (
    _available_feature_fns,
    _build_cycle_feature_table_extractor,
    _infer_dataset_for_battery,
    _prepare_dataset_windows,
    _prepare_dataset_battery_level,
    _build_models,
    _fit_label_transform,
    _inverse_label_transform,
    _apply_feature_processing
)
from batteryml.training.train_rul_baselines import build_train_test_lists
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ChemistryTrainer:
    """Trainer for chemistry-specific RUL prediction models."""

    def __init__(self, data_path: str, output_dir: str = 'chemistry_training_results', 
                 verbose: bool = False, dataset_hint: Optional[str] = None, 
                 cycle_limit: Optional[int] = None, smoothing: str = 'none', 
                 ma_window: int = 5, use_gpu: bool = False):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.cycle_limit = cycle_limit
        self.smoothing = str(smoothing or 'none').lower()
        self.ma_window = int(ma_window) if int(ma_window) > 1 else 5
        self.use_gpu = bool(use_gpu)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Chemistry name from data path
        self.chemistry_name = self.data_path.name
        
        # Create output subdirectories
        self.battery_level_dir = self.output_dir / self.chemistry_name / 'battery_level'
        self.cycle_level_dir = self.output_dir / self.chemistry_name / 'cycle_level'
        self.battery_level_dir.mkdir(parents=True, exist_ok=True)
        self.cycle_level_dir.mkdir(parents=True, exist_ok=True)

        self.rul_annotator = RULLabelAnnotator()

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
            from scipy.signal import medfilt, savgol_filter
            arr = np.asarray(y, dtype=float)
            if arr.size == 0:
                return arr
            # 1) Hampel
            h = ChemistryTrainer._hampel_filter(arr, window_size=11, n_sigmas=3.0)
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

    def _create_chemistry_train_test_split(self, train_test_ratio: float = 0.7, seed: int = 42) -> Tuple[List[Path], List[Path]]:
        """Create chemistry-level train/test split: 70% of cells from each dataset for training, 30% for testing."""
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
            split_idx = int(train_test_ratio * len(files_shuffled))
            
            # Split
            train_files.extend(files_shuffled[:split_idx])
            test_files.extend(files_shuffled[split_idx:])
            
            if self.verbose:
                print(f"  {dataset}: {len(files_shuffled[:split_idx])} train, {len(files_shuffled[split_idx:])} test")
        
        if self.verbose:
            print(f"Total: {len(train_files)} train files, {len(test_files)} test files")
        
        return train_files, test_files

    def train_battery_level(self, window_size: int = 10, features: Optional[List[str]] = None, 
                          tune: bool = False, cv_splits: int = 5) -> Dict[str, float]:
        """Train battery-level RUL prediction models."""
        if self.verbose:
            print(f"Training battery-level models for {self.chemistry_name}...")
        
        # Get feature functions
        feature_fns = _available_feature_fns()
        if features is None:
            feature_names = list(feature_fns.keys())
        else:
            feature_names = [f for f in features if f in feature_fns]
        
        # Create chemistry-level train/test split (70/30 per dataset within chemistry)
        train_files, test_files = self._create_chemistry_train_test_split(train_test_ratio=0.7)
        
        if not train_files:
            if self.verbose:
                print("No training files available")
            return {}
        
        # Prepare battery-level dataset
        X_train, y_train, g_train = self._prepare_dataset_battery_level(train_files, feature_fns, feature_names)
        
        if X_train.size == 0:
            if self.verbose:
                print("No data available for battery-level training")
            return {}
        
        if self.verbose:
            print(f"Battery-level training data: {X_train.shape}, Features: {len(feature_names)}")
        
        # Apply label transformation
        y_train_t, train_label_stats = _fit_label_transform(y_train)
        
        # Prepare test data once (if available)
        X_test, y_test = None, None
        if test_files:
            if self.verbose:
                print("Preparing test data...")
            X_test, y_test, _ = self._prepare_dataset_battery_level(test_files, feature_fns, feature_names)
        
        # Build models
        models = _build_models(use_gpu=self.use_gpu)
        
        # Train models
        results = {}
        for name, pipe in tqdm(models.items(), desc="Training battery-level models", unit="model"):
            if self.verbose:
                print(f"Training battery-level {name}...")
            
            try:
                est = pipe
                best_params = None
                
                if tune:
                    # Define parameter grids for hyperparameter tuning
                    if name == 'linear_regression':
                        param_grid = {
                            'model__fit_intercept': [True, False]
                        }
                    elif name == 'ridge':
                        param_grid = {
                            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                            'model__fit_intercept': [True, False],
                        }
                    elif name == 'elastic_net':
                        param_grid = {
                            'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                            'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                            'model__max_iter': [5000, 10000, 50000],
                        }
                    elif name == 'svr':
                        param_grid = {
                            'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                            'model__C': [0.001, 0.01, 0.1, 1.0],
                            'model__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                            'model__epsilon': [0.05, 0.1, 0.2],
                            'model__degree': [2, 3, 4],
                        }
                    elif name == 'random_forest':
                        param_grid = {
                            'model__n_estimators': [200, 400, 800],
                            'model__max_depth': [None, 10, 20, 40, 80],
                            'model__min_samples_split': [2, 5, 10],
                            'model__min_samples_leaf': [1, 2, 4],
                            'model__max_features': ['sqrt', 'log2', None],
                        }
                    elif name == 'xgboost':
                        param_grid = {
                            'model__n_estimators': [300, 600],
                            'model__max_depth': [4, 6, 8, 10, 20],
                            'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
                            'model__reg_alpha': [0.0, 0.001, 0.01],
                            'model__reg_lambda': [1.0, 5.0, 10.0]
                        }
                    elif name == 'mlp':
                        param_grid = {
                            'model__hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
                            'model__alpha': [1e-6, 1e-5, 1e-4],
                            'model__learning_rate_init': [0.0005, 0.001, 0.01],
                            'model__activation': ['relu', 'tanh'],
                            'model__batch_size': [8, 32, 64, 128],
                            'model__max_iter': [300, 900, 2000],
                        }
                    elif name == 'plsr':
                        param_grid = {
                            'model__n_components': [5, 10, 15, 20, 30],
                        }
                    elif name == 'pcr':
                        param_grid = {
                            'model__pca__n_components': [10, 20, 40, 60],
                            'model__lr__fit_intercept': [True, False],
                        }
                    else:
                        param_grid = {}

                    if param_grid:
                        # Group-aware CV to avoid leakage across batteries
                        unique_groups = np.unique(g_train)
                        n_splits = min(cv_splits, len(unique_groups)) if len(unique_groups) > 1 else 2
                        cv = GroupKFold(n_splits=n_splits)
                        grid = list(ParameterGrid(param_grid))
                        cv_results = []
                        best_score = np.inf
                        best_estimator = None
                        pbar = tqdm(grid, desc=f"Tuning {name} ({len(grid)} cfgs x {n_splits} folds)")
                        for params in pbar:
                            fold_rmses = []
                            fold_maes = []
                            for tr_idx, va_idx in cv.split(X_train, y_train, groups=g_train):
                                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                                # Fit label transform on training fold
                                y_tr_t, fold_stats = _fit_label_transform(y_tr)
                                model = clone(pipe)
                                model.set_params(**params)
                                model.fit(X_tr, y_tr_t)
                                pred_t = model.predict(X_va)
                                pred_t = np.asarray(pred_t, dtype=float)
                                pred_t = np.nan_to_num(pred_t, nan=0.0, posinf=0.0, neginf=0.0)
                                # Clamp transformed predictions to reasonable range before inverse transform
                                pred_t = np.clip(pred_t, a_min=-50.0, a_max=50.0)
                                pred = _inverse_label_transform(pred_t, fold_stats)
                                y_va_arr = np.asarray(y_va, dtype=float)
                                fold_maes.append(mean_absolute_error(y_va_arr, np.asarray(pred, dtype=float)))
                                fold_rmses.append(mean_squared_error(y_va_arr, np.asarray(pred, dtype=float)) ** 0.5)
                            mean_mae = float(np.mean(fold_maes)) if fold_maes else np.inf
                            mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else np.inf
                            cv_results.append({**{f"param:{k}": v for k, v in params.items()}, 'mean_MAE': mean_mae, 'mean_RMSE': mean_rmse})
                            pbar.set_postfix({"RMSE": f"{mean_rmse:.3f}"})
                            if mean_rmse < best_score:
                                best_score = mean_rmse
                                best_params = params
                                best_estimator = clone(pipe).set_params(**params)
                        if best_estimator is not None:
                            est = best_estimator
                        # Save CV results
                        cv_path = self.battery_level_dir / f"{self.chemistry_name}_{name}_battery_level_cv_results.csv"
                        pd.DataFrame(cv_results).to_csv(cv_path, index=False)
                        if self.verbose:
                            print(f"Best {name} params: {best_params} (RMSE: {best_score:.3f})")

                # Rebuild estimator with best params (if any) and fit on full train set
                if best_params is not None:
                    est = clone(pipe).set_params(**best_params)
                est.fit(X_train, y_train_t)
                
                # Evaluate on test set if available
                if X_test is not None and X_test.size > 0:
                    y_pred_t = est.predict(X_test)
                        y_pred_t = np.asarray(y_pred_t, dtype=float)
                        y_pred_t = np.nan_to_num(y_pred_t, nan=0.0, posinf=0.0, neginf=0.0)
                        y_pred_t = np.clip(y_pred_t, a_min=-50.0, a_max=50.0)
                        y_pred = _inverse_label_transform(y_pred_t, train_label_stats)
                        rmse = mean_squared_error(y_test, y_pred) ** 0.5
                        results[name] = rmse
                        if self.verbose:
                            print(f"Battery-level {name} RMSE: {rmse:.3f}")
                else:
                    results[name] = 0.0  # No test set available
                
                # Save model
                model_path = self.battery_level_dir / f"{self.chemistry_name}_{name}_battery_level_model.pkl"
                import joblib
                joblib.dump(est, model_path)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error training {name}: {e}")
                results[name] = np.inf
        
        return results

    def train_cycle_level(self, window_size: int = 10, features: Optional[List[str]] = None, 
                        tune: bool = False, cv_splits: int = 5) -> Dict[str, float]:
        """Train cycle-level RUL prediction models."""
        if self.verbose:
            print(f"Training cycle-level models for {self.chemistry_name}...")
        
        # Get feature functions
        feature_fns = _available_feature_fns()
        if features is None:
            feature_names = list(feature_fns.keys())
        else:
            feature_names = [f for f in features if f in feature_fns]
        
        # Create chemistry-level train/test split (70/30 per dataset within chemistry)
        train_files, test_files = self._create_chemistry_train_test_split(train_test_ratio=0.7)
        
        if not train_files:
            if self.verbose:
                print("No training files available")
            return {}
        
        # Prepare cycle-level dataset
        X_train, y_train, g_train = self._prepare_dataset_windows(train_files, feature_fns, feature_names, window_size)
        
        if X_train.size == 0:
            if self.verbose:
                print("No data available for cycle-level training")
            return {}
        
        if self.verbose:
            print(f"Cycle-level training data: {X_train.shape}, Features: {len(feature_names)} × window {window_size}")
        
        # Apply label transformation
        y_train_t, train_label_stats = _fit_label_transform(y_train)
        
        # Prepare test data once (if available)
        X_test, y_test = None, None
        if test_files:
            if self.verbose:
                print("Preparing test data...")
            X_test, y_test, _ = self._prepare_dataset_windows(test_files, feature_fns, feature_names, window_size)
        
        # Build models
        models = _build_models(use_gpu=self.use_gpu)
        
        # Train models
        results = {}
        for name, pipe in tqdm(models.items(), desc="Training cycle-level models", unit="model"):
            if self.verbose:
                print(f"Training cycle-level {name}...")
            
            try:
                est = pipe
                best_params = None
                
                if tune:
                    # Define parameter grids for hyperparameter tuning
                    if name == 'linear_regression':
                        param_grid = {
                            'model__fit_intercept': [True, False]
                        }
                    elif name == 'ridge':
                        param_grid = {
                            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                            'model__fit_intercept': [True, False],
                        }
                    elif name == 'elastic_net':
                        param_grid = {
                            'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                            'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                            'model__max_iter': [5000, 10000, 50000],
                        }
                    elif name == 'svr':
                        param_grid = {
                            'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                            'model__C': [0.001, 0.01, 0.1, 1.0],
                            'model__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                            'model__epsilon': [0.05, 0.1, 0.2],
                            'model__degree': [2, 3, 4],
                        }
                    elif name == 'random_forest':
                        param_grid = {
                            'model__n_estimators': [200, 400, 800],
                            'model__max_depth': [None, 10, 20, 40, 80],
                            'model__min_samples_split': [2, 5, 10],
                            'model__min_samples_leaf': [1, 2, 4],
                            'model__max_features': ['sqrt', 'log2', None],
                        }
                    elif name == 'xgboost':
                        param_grid = {
                            'model__n_estimators': [300, 600],
                            'model__max_depth': [4, 6, 8, 10, 20],
                            'model__learning_rate': [0.001, 0.01, 0.05, 0.1],
                            'model__reg_alpha': [0.0, 0.001, 0.01],
                            'model__reg_lambda': [1.0, 5.0, 10.0]
                        }
                    elif name == 'mlp':
                        param_grid = {
                            'model__hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
                            'model__alpha': [1e-6, 1e-5, 1e-4],
                            'model__learning_rate_init': [0.0005, 0.001, 0.01],
                            'model__activation': ['relu', 'tanh'],
                            'model__batch_size': [8, 32, 64, 128],
                            'model__max_iter': [300, 900, 2000],
                        }
                    elif name == 'plsr':
                        param_grid = {
                            'model__n_components': [5, 10, 15, 20, 30],
                        }
                    elif name == 'pcr':
                        param_grid = {
                            'model__pca__n_components': [10, 20, 40, 60],
                            'model__lr__fit_intercept': [True, False],
                        }
                    else:
                        param_grid = {}

                    if param_grid:
                        # Group-aware CV to avoid leakage across batteries
                        unique_groups = np.unique(g_train)
                        n_splits = min(cv_splits, len(unique_groups)) if len(unique_groups) > 1 else 2
                        cv = GroupKFold(n_splits=n_splits)
                        grid = list(ParameterGrid(param_grid))
                        cv_results = []
                        best_score = np.inf
                        best_estimator = None
                        pbar = tqdm(grid, desc=f"Tuning {name} ({len(grid)} cfgs x {n_splits} folds)")
                        for params in pbar:
                            fold_rmses = []
                            fold_maes = []
                            for tr_idx, va_idx in cv.split(X_train, y_train, groups=g_train):
                                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                                # Fit label transform on training fold
                                y_tr_t, fold_stats = _fit_label_transform(y_tr)
                                model = clone(pipe)
                                model.set_params(**params)
                                model.fit(X_tr, y_tr_t)
                                pred_t = model.predict(X_va)
                                pred_t = np.asarray(pred_t, dtype=float)
                                pred_t = np.nan_to_num(pred_t, nan=0.0, posinf=0.0, neginf=0.0)
                                # Clamp transformed predictions to reasonable range before inverse transform
                                pred_t = np.clip(pred_t, a_min=-50.0, a_max=50.0)
                                pred = _inverse_label_transform(pred_t, fold_stats)
                                y_va_arr = np.asarray(y_va, dtype=float)
                                fold_maes.append(mean_absolute_error(y_va_arr, np.asarray(pred, dtype=float)))
                                fold_rmses.append(mean_squared_error(y_va_arr, np.asarray(pred, dtype=float)) ** 0.5)
                            mean_mae = float(np.mean(fold_maes)) if fold_maes else np.inf
                            mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else np.inf
                            cv_results.append({**{f"param:{k}": v for k, v in params.items()}, 'mean_MAE': mean_mae, 'mean_RMSE': mean_rmse})
                            pbar.set_postfix({"RMSE": f"{mean_rmse:.3f}"})
                            if mean_rmse < best_score:
                                best_score = mean_rmse
                                best_params = params
                                best_estimator = clone(pipe).set_params(**params)
                        if best_estimator is not None:
                            est = best_estimator
                        # Save CV results
                        cv_path = self.cycle_level_dir / f"{self.chemistry_name}_{name}_cycle_level_cv_results.csv"
                        pd.DataFrame(cv_results).to_csv(cv_path, index=False)
                        if self.verbose:
                            print(f"Best {name} params: {best_params} (RMSE: {best_score:.3f})")

                # Rebuild estimator with best params (if any) and fit on full train set
                if best_params is not None:
                    est = clone(pipe).set_params(**best_params)
                est.fit(X_train, y_train_t)
                
                # Evaluate on test set if available
                if X_test is not None and X_test.size > 0:
                    y_pred_t = est.predict(X_test)
                        y_pred_t = np.asarray(y_pred_t, dtype=float)
                        y_pred_t = np.nan_to_num(y_pred_t, nan=0.0, posinf=0.0, neginf=0.0)
                        y_pred_t = np.clip(y_pred_t, a_min=-50.0, a_max=50.0)
                        y_pred = _inverse_label_transform(y_pred_t, train_label_stats)
                        rmse = mean_squared_error(y_test, y_pred) ** 0.5
                        results[name] = rmse
                        if self.verbose:
                            print(f"Cycle-level {name} RMSE: {rmse:.3f}")
                else:
                    results[name] = 0.0  # No test set available
                
                # Save model
                model_path = self.cycle_level_dir / f"{self.chemistry_name}_{name}_cycle_level_model.pkl"
                import joblib
                joblib.dump(est, model_path)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error training {name}: {e}")
                results[name] = np.inf
        
        return results

    def _prepare_dataset_battery_level(self, files: List[Path], feature_fns: Dict[str, callable], 
                                     feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare battery-level dataset."""
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        g_list: List[str] = []
        
        for f in tqdm(files, desc="Processing batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            
            # Build feature table using chemistry-specific extractor
            try:
                df = self._build_cycle_feature_table_extractor(battery, feature_names)
            except Exception:
                df = pd.DataFrame()
            
            if df.empty:
                continue
            
            # Calculate total RUL
            try:
                rul_tensor = self.rul_annotator.process_cell(battery)
                total_rul = int(rul_tensor.item()) if np.isfinite(float(rul_tensor.item())) else 0
            except Exception:
                total_rul = 0
            
            if total_rul <= 0:
                continue
            
            # Extract features (mean values across cycles)
            feature_values = []
            for name in feature_names:
                if name in df.columns:
                    val = df[name].mean()
                    feature_values.append(float(val) if np.isfinite(val) else np.nan)
                else:
                    feature_values.append(np.nan)
            
            X_list.append(np.array(feature_values))
            y_list.append(float(total_rul))
            g_list.append(battery.cell_id)
        
        if not X_list:
            return np.zeros((0, len(feature_names)), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=str)
        
        return np.array(X_list), np.array(y_list), np.array(g_list)

    def _prepare_dataset_windows(self, files: List[Path], feature_fns: Dict[str, callable], 
                               feature_names: List[str], window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare cycle-level dataset with windowing."""
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        g_list: List[str] = []
        
        for f in tqdm(files, desc="Processing batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            
            # Build feature table using chemistry-specific extractor
            try:
                df = self._build_cycle_feature_table_extractor(battery, feature_names)
            except Exception:
                df = pd.DataFrame()
            
            if df.empty:
                continue
            
            # Calculate total RUL
            try:
                rul_tensor = self.rul_annotator.process_cell(battery)
                total_rul = int(rul_tensor.item()) if np.isfinite(float(rul_tensor.item())) else 0
            except Exception:
                total_rul = 0
            
            if total_rul <= 0:
                continue
            
            # Create windows
            df_sorted = df.sort_values('cycle_number').reset_index(drop=True)
            num_cycles = len(df_sorted)
            last_start = num_cycles - (window_size + 1)
            
            if last_start < 0:
                continue
            
            # Extract feature columns
            feature_cols = [col for col in feature_names if col in df_sorted.columns]
            if not feature_cols:
                continue
            
            feat_mat = df_sorted[feature_cols].to_numpy()
            
            for k in range(0, last_start + 1):
                w = feat_mat[k:k + window_size, :]
                x = w.reshape(-1)  # flatten
                rul = max(0, total_rul - (k + window_size))
                
                X_list.append(x)
                y_list.append(float(rul))
                g_list.append(battery.cell_id)
        
        if not X_list:
            return np.zeros((0, len(feature_names) * window_size), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=str)
        
        return np.array(X_list), np.array(y_list), np.array(g_list)

    def train_both(self, window_size: int = 10, features: Optional[List[str]] = None, 
                  tune: bool = False, cv_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """Train both battery-level and cycle-level models."""
        results = {}
        
        # Train battery-level models
        battery_results = self.train_battery_level(window_size, features, tune, cv_splits)
        results['battery_level'] = battery_results
        
        # Train cycle-level models
        cycle_results = self.train_cycle_level(window_size, features, tune, cv_splits)
        results['cycle_level'] = cycle_results
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Chemistry-specific RUL training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to chemistry folder containing *.pkl files')
    parser.add_argument('--output_dir', type=str, default='chemistry_training_results', help='Output directory')
    parser.add_argument('--dataset_hint', type=str, default=None, help='Optional dataset name hint')
    parser.add_argument('--cycle_limit', type=int, default=None, help='Limit analysis to first N cycles')
    parser.add_argument('--smoothing', type=str, default='none', choices=['none', 'ma', 'median', 'hms'], help='Smoothing method')
    parser.add_argument('--ma_window', type=int, default=5, help='Window size for moving average/median smoothing')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for cycle-level training')
    parser.add_argument('--features', type=str, nargs='+', default=None, help='Features to use (default: all)')
    parser.add_argument('--battery_level_only', action='store_true', help='Only train battery-level models')
    parser.add_argument('--cycle_level_only', action='store_true', help='Only train cycle-level models')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--cv_splits', type=int, default=5, help='Number of cross-validation splits')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    trainer = ChemistryTrainer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        verbose=args.verbose,
        dataset_hint=args.dataset_hint,
        cycle_limit=args.cycle_limit,
        smoothing=args.smoothing,
        ma_window=args.ma_window,
        use_gpu=args.use_gpu
    )

    if args.battery_level_only:
        results = trainer.train_battery_level(args.window_size, args.features, args.tune, args.cv_splits)
    elif args.cycle_level_only:
        results = trainer.train_cycle_level(args.window_size, args.features, args.tune, args.cv_splits)
    else:
        results = trainer.train_both(args.window_size, args.features, args.tune, args.cv_splits)

    if args.verbose:
        print(f"Training completed for {trainer.chemistry_name}")
        print(f"Results: {results}")


if __name__ == '__main__':
    main()
