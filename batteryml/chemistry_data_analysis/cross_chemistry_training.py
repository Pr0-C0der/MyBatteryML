#!/usr/bin/env python3
"""
Cross-Chemistry Training for RUL Prediction

This script trains models on one chemistry (e.g., LFP) and tests on other chemistries
to evaluate cross-chemistry generalization performance.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import sys
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

# Ensure project root (parent of 'batteryml') is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class

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

    def _build_cycle_feature_matrix(self, battery: BatteryData, extractor) -> pd.DataFrame:
        """Build cycle feature matrix for a battery."""
        data = []
        total_rul = self._compute_total_rul(battery)
        
        # Apply cycle limit if specified
        cycles_to_process = battery.cycle_data
        if self.cycle_limit is not None and self.cycle_limit > 0:
            cycles_to_process = cycles_to_process[:self.cycle_limit]
        
        for idx, c in enumerate(cycles_to_process):
            row = {
                'cycle_number': c.cycle_number,
                'rul': max(0, total_rul - idx)
            }
            
            if extractor is not None:
                # Get all available features from the extractor
                feature_methods = [method for method in dir(extractor) 
                                 if not method.startswith('_') and callable(getattr(extractor, method))]
                
                for method_name in feature_methods:
                    try:
                        method = getattr(extractor, method_name)
                        val = method(battery, c)
                        if val is not None and np.isfinite(val):
                            row[method_name] = float(val)
                        else:
                            row[method_name] = np.nan
                    except Exception:
                        row[method_name] = np.nan
            
            data.append(row)
        
        return pd.DataFrame(data)

    def _compute_total_rul(self, battery: BatteryData) -> int:
        """Compute total RUL for a battery."""
        try:
            rul_tensor = self.rul_annotator.process_cell(battery)
            v = int(rul_tensor.item())
            return v if np.isfinite(v) else 0
        except Exception:
            return 0

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from the training chemistry."""
        if self.verbose:
            print(f"Preparing training data from {self.train_chemistry_name}...")
        
        train_files = self._battery_files(self.train_chemistry_path)
        if not train_files:
            raise ValueError(f"No battery files found in {self.train_chemistry_path}")
        
        all_battery_data = []
        feature_names = set()
        
        for f in tqdm(train_files, desc="Processing training batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
                dataset = self._infer_dataset_for_battery(battery)
                if not dataset:
                    continue
                
                extractor_class = get_extractor_class(dataset)
                if extractor_class is None:
                    continue
                
                extractor = extractor_class()
                df = self._build_cycle_feature_matrix(battery, extractor)
                
                if not df.empty:
                    # Collect feature names
                    feature_names.update([col for col in df.columns if col not in ['cycle_number', 'rul']])
                    all_battery_data.append((battery.cell_id, df))
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not process {f}: {e}")
                continue
        
        if not all_battery_data:
            raise ValueError("No valid training data found")
        
        # Convert to arrays
        X_list = []
        y_list = []
        
        for cell_id, df in all_battery_data:
            # Use all available features
            feature_cols = [col for col in df.columns if col not in ['cycle_number', 'rul']]
            X = df[feature_cols].values
            y = df['rul'].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) > 0:
                X_list.append(X)
                y_list.append(y)
        
        if not X_list:
            raise ValueError("No valid training samples found")
        
        # Concatenate all data
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        
        if self.verbose:
            print(f"Training data shape: {X_train.shape}")
            print(f"Features: {len(feature_names)}")
        
        return X_train, y_train, list(feature_names)

    def _prepare_test_data(self, chemistry_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
        """Prepare test data from a test chemistry."""
        chemistry_name = chemistry_path.name
        if self.verbose:
            print(f"Preparing test data from {chemistry_name}...")
        
        test_files = self._battery_files(chemistry_path)
        if not test_files:
            return np.array([]), np.array([]), chemistry_name
        
        all_battery_data = []
        
        for f in tqdm(test_files, desc=f"Processing {chemistry_name} batteries", unit="battery"):
            try:
                battery = BatteryData.load(f)
                dataset = self._infer_dataset_for_battery(battery)
                if not dataset:
                    continue
                
                extractor_class = get_extractor_class(dataset)
                if extractor_class is None:
                    continue
                
                extractor = extractor_class()
                df = self._build_cycle_feature_matrix(battery, extractor)
                
                if not df.empty:
                    all_battery_data.append((battery.cell_id, df))
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not process {f}: {e}")
                continue
        
        if not all_battery_data:
            return np.array([]), np.array([]), chemistry_name
        
        # Convert to arrays
        X_list = []
        y_list = []
        
        for cell_id, df in all_battery_data:
            # Use same features as training
            feature_cols = [col for col in df.columns if col not in ['cycle_number', 'rul']]
            X = df[feature_cols].values
            y = df['rul'].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) > 0:
                X_list.append(X)
                y_list.append(y)
        
        if not X_list:
            return np.array([]), np.array([]), chemistry_name
        
        # Concatenate all data
        X_test = np.vstack(X_list)
        y_test = np.concatenate(y_list)
        
        if self.verbose:
            print(f"Test data shape for {chemistry_name}: {X_test.shape}")
        
        return X_test, y_test, chemistry_name

    def _build_models(self) -> Dict[str, Pipeline]:
        """Build all available models for training."""
        models = {}
        base_steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler())]

        # Determine if GPU is truly available
        gpu_available = False
        if self.use_gpu:
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
                pipe.fit(X_train, y_train)
                
                # Initialize results for this model
                model_rmse = {}
                model_mae = {}
                model_mape = {}
                
                # Evaluate on each test chemistry
                for test_chemistry_path in self.test_chemistry_paths:
                    X_test, y_test, chemistry_name = self._prepare_test_data(test_chemistry_path)
                    
                    if X_test.size == 0:
                        if self.verbose:
                            print(f"No test data available for {chemistry_name}")
                        model_rmse[chemistry_name] = np.nan
                        model_mae[chemistry_name] = np.nan
                        model_mape[chemistry_name] = np.nan
                        continue
                    
                    # Make predictions
                    y_pred = pipe.predict(X_test)
                    
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
            best_score = np.inf if metric != 'MAPE' else np.inf
            for model_name, model_results in results[metric].items():
                if test_chem in model_results and not np.isnan(model_results[test_chem]):
                    score = model_results[test_chem]
                    if (metric == 'MAPE' and score < best_score) or (metric != 'MAPE' and score < best_score):
                        best_score = score
                        best_model = model_name
            if best_model:
                print(f"  {test_chem}: {best_model} ({best_score:.3f})")


if __name__ == '__main__':
    main()
