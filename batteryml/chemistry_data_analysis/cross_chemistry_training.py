#!/usr/bin/env python3
"""
Cross-Chemistry Training for RUL Prediction

This script trains models on one chemistry (e.g., LFP) and tests on other chemistries
to evaluate cross-chemistry generalization performance.
"""

import sys
from pathlib import Path

# Ensure project root (parent of 'batteryml') is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import argparse
from typing import List, Tuple, Dict, Optional
import warnings
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
from tqdm import tqdm

# Optional imports for advanced models
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import cuml  # noqa: F401
    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class
from batteryml.training.train_rul_windows import (
    _infer_dataset_for_battery,
    _build_cycle_feature_table_extractor,
    _build_models,
    _fit_label_transform,
    _inverse_label_transform
)

# Suppress warnings
warnings.filterwarnings('ignore')


class CrossChemistryTrainer:
    """Cross-chemistry trainer for RUL prediction models."""
    
    def __init__(self, train_chemistry_path: str, test_chemistry_paths: List[str], 
                 output_dir: str = 'cross_chemistry_results', verbose: bool = False, 
                 dataset_hint: Optional[str] = None, cycle_limit: Optional[int] = None, 
                 smoothing: str = 'none', ma_window: int = 5, use_gpu: bool = False, 
                 train_test_ratio: float = 0.7):
        """
        Initialize cross-chemistry trainer.
        
        Args:
            train_chemistry_path: Path to training chemistry folder (e.g., 'data_chemistries/lfp')
            test_chemistry_paths: List of paths to test chemistry folders
            output_dir: Output directory for results
            verbose: Enable verbose logging
            dataset_hint: Optional dataset name hint
            cycle_limit: Limit analysis to first N cycles
            smoothing: Smoothing method
            ma_window: Window size for smoothing
            use_gpu: Use GPU acceleration
            train_test_ratio: Train/test split ratio for training chemistry
        """
        self.train_chemistry_path = Path(train_chemistry_path)
        self.test_chemistry_paths = [Path(p) for p in test_chemistry_paths]
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.cycle_limit = cycle_limit
        self.smoothing = str(smoothing or 'none').lower()
        self.ma_window = int(ma_window) if int(ma_window) > 1 else 5
        self.use_gpu = bool(use_gpu)
        self.train_test_ratio = float(train_test_ratio)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemistry names
        self.train_chemistry_name = self.train_chemistry_path.name
        self.test_chemistry_names = [p.name for p in self.test_chemistry_paths]
        
        # Initialize RUL annotator
        self.rul_annotator = RULLabelAnnotator()
        
        # Initialize metric tracking files
        self.rmse_file = self.output_dir / "cross_chemistry_RMSE.csv"
        self.mae_file = self.output_dir / "cross_chemistry_MAE.csv"
        self.mape_file = self.output_dir / "cross_chemistry_MAPE.csv"
        
        if self.verbose:
            print(f"Cross-Chemistry Training Setup:")
            print(f"  Train Chemistry: {self.train_chemistry_name}")
            print(f"  Test Chemistries: {', '.join(self.test_chemistry_names)}")
            print(f"  Output Directory: {self.output_dir}")

    def _battery_files(self, chemistry_path: Path) -> List[Path]:
        """Get all battery files in a chemistry folder."""
        return sorted(chemistry_path.glob('*.pkl'))

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from the training chemistry using the same approach as chemistry_training.py."""
        if self.verbose:
            print(f"Preparing training data from {self.train_chemistry_name}...")
        
        train_files = self._battery_files(self.train_chemistry_path)
        if not train_files:
            raise ValueError(f"No battery files found in {self.train_chemistry_path}")
        
        if self.verbose:
            print(f"Found {len(train_files)} battery files")
        
        # Get feature names using the same approach as chemistry_training.py
        feature_names = self._get_available_features(train_files)
        if not feature_names:
            raise ValueError("No valid features found")
        
        if self.verbose:
            print(f"Using {len(feature_names)} features: {feature_names[:10]}...")
        
        # Process all batteries using the same approach as chemistry_training.py
        X_list = []
        y_list = []
        processed_batteries = 0
        
        for f in tqdm(train_files, desc="Processing training batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
                
                # Build feature table using the same method as chemistry_training.py
                df = _build_cycle_feature_table_extractor(battery, feature_names, self.dataset_hint)
                
                if df.empty:
                    if self.verbose:
                        print(f"Empty feature table for {f}")
                    continue
                
                # Calculate total RUL using the same approach
                try:
                    rul_tensor = self.rul_annotator.process_cell(battery)
                    total_rul = int(rul_tensor.item()) if np.isfinite(float(rul_tensor.item())) else 0
                except Exception:
                    total_rul = 0
                
                if total_rul <= 0:
                    if self.verbose:
                        print(f"Invalid RUL for {f}: {total_rul}")
                    continue
                
                # Extract features (mean values across cycles) - same as chemistry_training.py
                feature_values = []
                for name in feature_names:
                    if name in df.columns:
                        val = df[name].mean()
                        feature_values.append(float(val) if np.isfinite(val) else np.nan)
                    else:
                        feature_values.append(np.nan)
                
                # Check if we have valid features
                if not any(np.isfinite(feature_values)):
                    if self.verbose:
                        print(f"No valid features for {f}")
                    continue
                
                X_list.append(np.array(feature_values))
                y_list.append(float(total_rul))
                processed_batteries += 1
                
                if self.verbose:
                    print(f"Processed {f}: RUL={total_rul}, features={len([x for x in feature_values if np.isfinite(x)])}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not process {f}: {e}")
                continue
        
        if self.verbose:
            print(f"Processed {processed_batteries} batteries")
        
        if not X_list:
            raise ValueError("No valid training samples found")
        
        # Convert to arrays
        X_train = np.vstack(X_list)
        y_train = np.array(y_list)
        
        if self.verbose:
            print(f"Training data shape: {X_train.shape}")
            print(f"RUL range: {y_train.min():.1f}-{y_train.max():.1f}")
        
        return X_train, y_train, feature_names

    def _get_available_features(self, files: List[Path]) -> List[str]:
        """Get available features from the first valid battery, same approach as chemistry_training.py."""
        for f in files:
            try:
                battery = BatteryData.load(f)
                dataset = _infer_dataset_for_battery(self.dataset_hint, battery)
                if not dataset:
                    continue
                
                extractor_class = get_extractor_class(dataset)
                if extractor_class is None:
                    continue
                
                # Get all available features from the extractor
                extractor = extractor_class()
                feature_methods = [method for method in dir(extractor) 
                                 if not method.startswith('_') and callable(getattr(extractor, method))]
                
                if self.verbose:
                    print(f"Found {len(feature_methods)} features from dataset {dataset}")
                
                return feature_methods
                
            except Exception as e:
                if self.verbose:
                    print(f"Error getting features from {f}: {e}")
                continue
        
        return []

    def _prepare_test_data(self, chemistry_path: Path, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, str]:
        """Prepare test data from a test chemistry using the same approach as chemistry_training.py."""
        chemistry_name = chemistry_path.name
        if self.verbose:
            print(f"Preparing test data from {chemistry_name}...")
        
        test_files = self._battery_files(chemistry_path)
        if not test_files:
            return np.array([]), np.array([]), chemistry_name
        
        # Process all batteries using the same approach as chemistry_training.py
        X_list = []
        y_list = []
        
        for f in tqdm(test_files, desc=f"Processing {chemistry_name} batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
                
                # Build feature table using the same method as chemistry_training.py
                df = _build_cycle_feature_table_extractor(battery, feature_names, self.dataset_hint)
                
                if df.empty:
                    continue
                
                # Calculate total RUL using the same approach
                try:
                    rul_tensor = self.rul_annotator.process_cell(battery)
                    total_rul = int(rul_tensor.item()) if np.isfinite(float(rul_tensor.item())) else 0
                except Exception:
                    total_rul = 0
                
                if total_rul <= 0:
                    continue
                
                # Extract features (mean values across cycles) - same as chemistry_training.py
                feature_values = []
                for name in feature_names:
                    if name in df.columns:
                        val = df[name].mean()
                        feature_values.append(float(val) if np.isfinite(val) else np.nan)
                    else:
                        feature_values.append(np.nan)
                
                # Check if we have valid features
                if not any(np.isfinite(feature_values)):
                    continue
                
                X_list.append(np.array(feature_values))
                y_list.append(float(total_rul))
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not process {f}: {e}")
                continue
        
        if not X_list:
            return np.array([]), np.array([]), chemistry_name
        
        # Convert to arrays
        X_test = np.vstack(X_list)
        y_test = np.array(y_list)
        
        if self.verbose:
            print(f"Test data shape for {chemistry_name}: {X_test.shape}")
        
        return X_test, y_test, chemistry_name

    def _build_models(self) -> Dict[str, Pipeline]:
        """Build all available models for training using the same approach as chemistry_training.py."""
        return _build_models(use_gpu=self.use_gpu)

    def _save_metric_results(self, results: Dict[str, Dict[str, float]], metric_file: Path, metric_name: str):
        """Save metric results to CSV file."""
        # Convert nested dict to DataFrame
        df = pd.DataFrame(results).T
        df.index.name = 'Model'
        
        # Round values to 2 decimal places
        df = df.round(2)
        
        # Save to CSV
        df.to_csv(metric_file)
        
        if self.verbose:
            print(f"{metric_name} results saved to {metric_file}")

    def train_and_evaluate(self, tune: bool = False, cv_splits: int = 5) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Train models on training chemistry and evaluate on test chemistries."""
        if self.verbose:
            print("Starting cross-chemistry training and evaluation...")
        
        # Prepare training data
        X_train, y_train, feature_names = self._prepare_training_data()
        
        # Apply label transformation (same as chemistry_training.py)
        y_train_t, train_label_stats = _fit_label_transform(y_train)
        
        # Build models
        models = self._build_models()
        
        # Initialize results storage
        all_results = {
            'RMSE': {},
            'MAE': {},
            'MAPE': {}
        }
        
        # Train each model
        for name, pipe in tqdm(models.items(), desc="Training models", unit="model"):
            if self.verbose:
                print(f"Training {name}...")
            
            try:
                # Train model
                pipe.fit(X_train, y_train_t)
                
                # Initialize results for this model
                model_rmse = {}
                model_mae = {}
                model_mape = {}
                
                # Evaluate on each test chemistry
                for test_chemistry_path in self.test_chemistry_paths:
                    X_test, y_test, chemistry_name = self._prepare_test_data(test_chemistry_path, feature_names)
                    
                    if X_test.size == 0:
                        if self.verbose:
                            print(f"No test data available for {chemistry_name}")
                        model_rmse[chemistry_name] = np.nan
                        model_mae[chemistry_name] = np.nan
                        model_mape[chemistry_name] = np.nan
                        continue
                    
                    # Make predictions and inverse transform (same as chemistry_training.py)
                    y_pred_t = pipe.predict(X_test)
                    y_pred_t = np.asarray(y_pred_t, dtype=float)
                    y_pred_t = np.nan_to_num(y_pred_t, nan=0.0, posinf=0.0, neginf=0.0)
                    y_pred_t = np.clip(y_pred_t, a_min=-50.0, a_max=50.0)
                    y_pred = _inverse_label_transform(y_pred_t, train_label_stats)
                    
                    # Calculate metrics
                    rmse = mean_squared_error(y_test, y_pred) ** 0.5
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Calculate MAPE (Mean Absolute Percentage Error)
                    epsilon = 1e-8
                    mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
                    
                    model_rmse[chemistry_name] = rmse
                    model_mae[chemistry_name] = mae
                    model_mape[chemistry_name] = mape
                    
                    if self.verbose:
                        print(f"  {chemistry_name}: RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.3f}%")
                
                # Store results for this model
                all_results['RMSE'][name] = model_rmse
                all_results['MAE'][name] = model_mae
                all_results['MAPE'][name] = model_mape
                
            except Exception as e:
                if self.verbose:
                    print(f"Error training {name}: {e}")
                # Fill with NaN for all test chemistries
                for test_chemistry_path in self.test_chemistry_paths:
                    chemistry_name = test_chemistry_path.name
                    all_results['RMSE'][name] = {chemistry_name: np.nan}
                    all_results['MAE'][name] = {chemistry_name: np.nan}
                    all_results['MAPE'][name] = {chemistry_name: np.nan}
        
        # Save results to CSV files
        self._save_metric_results(all_results['RMSE'], self.rmse_file, "RMSE")
        self._save_metric_results(all_results['MAE'], self.mae_file, "MAE")
        self._save_metric_results(all_results['MAPE'], self.mape_file, "MAPE")
        
        if self.verbose:
            print("Cross-chemistry training and evaluation completed!")
            print(f"Results saved to: {self.output_dir}")
        
        return all_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Cross-chemistry RUL training and evaluation')
    parser.add_argument('--train_chemistry', type=str, required=True, 
                       help='Path to training chemistry folder (e.g., data_chemistries/lfp)')
    parser.add_argument('--test_chemistries', type=str, nargs='+', required=True,
                       help='Paths to test chemistry folders (e.g., data_chemistries/nmc data_chemistries/nca)')
    parser.add_argument('--output_dir', type=str, default='cross_chemistry_results',
                       help='Output directory for results')
    parser.add_argument('--dataset_hint', type=str, default=None,
                       help='Optional dataset name hint')
    parser.add_argument('--cycle_limit', type=int, default=None,
                       help='Limit analysis to first N cycles')
    parser.add_argument('--smoothing', type=str, default='none', 
                       choices=['none', 'ma', 'median', 'hms'],
                       help='Smoothing method for feature data')
    parser.add_argument('--ma_window', type=int, default=5,
                       help='Window size for moving average/median smoothing')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--cv_splits', type=int, default=5,
                       help='Number of cross-validation splits')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CrossChemistryTrainer(
        train_chemistry_path=args.train_chemistry,
        test_chemistry_paths=args.test_chemistries,
        output_dir=args.output_dir,
        verbose=args.verbose,
        dataset_hint=args.dataset_hint,
        cycle_limit=args.cycle_limit,
        smoothing=args.smoothing,
        ma_window=args.ma_window,
        use_gpu=args.use_gpu
    )
    
    # Train and evaluate
    results = trainer.train_and_evaluate(tune=args.tune, cv_splits=args.cv_splits)
    
    # Print summary
    print("\nCross-Chemistry Training Summary:")
    print("=" * 50)
    print(f"Train Chemistry: {trainer.train_chemistry_name}")
    print(f"Test Chemistries: {', '.join(trainer.test_chemistry_names)}")
    print(f"Results saved to: {args.output_dir}")
    
    # Show best performing model for each test chemistry
    for metric in ['RMSE', 'MAE', 'MAPE']:
        print(f"\nBest {metric} per test chemistry:")
        for test_chem in trainer.test_chemistry_names:
            best_model = None
            best_score = np.inf
            for model_name, model_results in results[metric].items():
                if test_chem in model_results and not np.isnan(model_results[test_chem]):
                    score = model_results[test_chem]
                    if score < best_score:
                        best_score = score
                        best_model = model_name
            if best_model:
                print(f"  {test_chem}: {best_model} ({best_score:.3f})")


if __name__ == '__main__':
    main()