# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import Memory

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from batteryml.data.battery_data import BatteryData
from batteryml.data_analysis.correlation_mod import (
    ModularCorrelationAnalyzer,
    build_default_analyzer,
)
from batteryml.train_test_split.MATR_split import MATRPrimaryTestTrainTestSplitter
from batteryml.train_test_split.random_split import RandomTrainTestSplitter
from batteryml.train_test_split.CRUH_split import CRUHTrainTestSplitter
from batteryml.train_test_split.CRUSH_split import CRUSHTrainTestSplitter
from batteryml.train_test_split.HUST_split import HUSTTrainTestSplitter
from batteryml.train_test_split.SNL_split import SNLTrainTestSplitter
from batteryml.train_test_split.MIX100_split import MIX100TrainTestSplitter


def build_train_test_lists(dataset: str, data_path: str, seed: int = 42) -> Tuple[List[Path], List[Path]]:
    data_dir = Path(data_path)
    if dataset.upper() == 'MATR':
        splitter = MATRPrimaryTestTrainTestSplitter(str(data_dir))
        train_files, test_files = splitter.split()
    elif dataset.upper() == 'CALCE':
        # Use a reproducible random split for CALCE
        splitter = RandomTrainTestSplitter(str(data_dir), seed=seed, train_test_split_ratio=0.7)
        train_files, test_files = splitter.split()
    elif dataset.upper() == 'CRUH':
        splitter = CRUHTrainTestSplitter(str(data_dir))
        train_files, test_files = splitter.split()
    elif dataset.upper() == 'CRUSH':
        splitter = CRUSHTrainTestSplitter(str(data_dir))
        train_files, test_files = splitter.split()
    elif dataset.upper() == 'HUST':
        splitter = HUSTTrainTestSplitter(str(data_dir))
        train_files, test_files = splitter.split()
    elif dataset.upper() == 'SNL':
        splitter = SNLTrainTestSplitter(str(data_dir))
        train_files, test_files = splitter.split()
    elif dataset.upper() == 'MIX100':
        splitter = MIX100TrainTestSplitter(str(data_dir))
        train_files, test_files = splitter.split()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected one of ['MATR', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100'].")
    return [Path(p) for p in train_files], [Path(p) for p in test_files]


def load_cycle_feature_matrix(analyzer: ModularCorrelationAnalyzer, file_path: Path) -> pd.DataFrame:
    battery = BatteryData.load(file_path)
    df = analyzer.build_cycle_feature_matrix(battery)
    # Tag with battery id for traceability
    df['battery_id'] = battery.cell_id
    return df


def combine_split_features(
    analyzer: ModularCorrelationAnalyzer,
    files: List[Path]
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(load_cycle_feature_matrix(analyzer, f))
        except Exception as e:
            print(f"Warning: skipping {f} due to error: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=0, ignore_index=True)
    return df


def prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    # Keep numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude label and non-feature columns
    drop_cols = {'rul', 'cycle_number'}
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    X = df[feature_cols].to_numpy()
    y = df['rul'].to_numpy()
    # Groups for CV: group by battery id if available, else by a placeholder
    if 'battery_id' in df.columns:
        groups = df['battery_id'].astype(str).to_numpy()
    else:
        groups = np.array(['all'] * len(df))
    return X, y, feature_cols, groups


def build_models(memory: Memory, seed: int = 42, use_gpu: bool = False) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    # Linear Regression: impute + scale + LR
    models['linear_regression'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ], memory=memory)
    # Random Forest: impute + RF
    models['random_forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1))
    ], memory=memory)
    # XGBoost (if available): impute + XGB
    if _HAS_XGB:
        models['xgboost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=-1,
                tree_method='hist',
                device=('cuda' if use_gpu else 'cpu')
            ))
        ], memory=memory)
    else:
        print("XGBoost not available; skipping.")
    return models


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    # MAPE: avoid division by zero
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None)))
    return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}


def run(dataset: str, data_path: str, output_dir: str, seed: int = 42, tune: bool = True, cv_splits: int = 5, use_gpu: bool = False):
    out_dir = Path(output_dir)
    (out_dir / dataset.upper()).mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'sk_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory(location=str(cache_dir), verbose=0)

    # Build split
    train_files, test_files = build_train_test_lists(dataset, data_path, seed)
    print(f"Found {len(train_files)} train and {len(test_files)} test batteries for {dataset}.")
    try:
        train_names = [p.stem for p in train_files]
        test_names = [p.stem for p in test_files]
        if train_names:
            print(f"Train batteries ({len(train_names)}): {', '.join(train_names)}")
        if test_names:
            print(f"Test batteries ({len(test_names)}): {', '.join(test_names)}")
    except Exception:
        pass

    # Analyzer to compute per-cycle feature matrices (includes derived features)
    analyzer = build_default_analyzer(data_path, output_dir=str(out_dir / f"{dataset.upper()}_tmp"), verbose=False)

    # Build train/test frames
    df_train = combine_split_features(analyzer, train_files)
    df_test = combine_split_features(analyzer, test_files)

    if df_train.empty or df_test.empty:
        print("No data available after feature extraction. Aborting.")
        return

    # Prepare X/y
    X_train, y_train, feature_cols, groups_train = prepare_xy(df_train)
    X_test, y_test, _, _ = prepare_xy(df_test)

    # Train and evaluate models
    models = build_models(memory, seed, use_gpu=use_gpu)
    results = []
    for name, pipe in models.items():
        print(f"Training {name} on {dataset}...")
        est = pipe
        best_params = None

        if tune:
            # Parameter grids
            if name == 'linear_regression':
                param_grid = {
                    'model__fit_intercept': [True, False]
                }
            elif name == 'random_forest':
                param_grid = {
                    'model__n_estimators': [200, 400, 800],
                    'model__max_depth': [None, 10, 20, 40],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            elif name == 'xgboost':
                param_grid = {
                    'model__n_estimators': [300, 600, 1000],
                    'model__max_depth': [4, 6, 8],
                    'model__learning_rate': [0.03, 0.05, 0.1],
                    'model__subsample': [0.8, 0.9, 1.0],
                    'model__colsample_bytree': [0.8, 0.9, 1.0]
                }
            else:
                param_grid = {}

            # Group-aware CV to avoid leakage across batteries
            unique_groups = np.unique(groups_train)
            n_splits = min(cv_splits, len(unique_groups)) if len(unique_groups) > 1 else 2
            cv = GroupKFold(n_splits=n_splits)
            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring='neg_mean_absolute_error',
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=1,
                return_train_score=False
            ) if param_grid else None

            if search is not None:
                search.fit(X_train, y_train, groups=groups_train)
                est = search.best_estimator_
                best_params = search.best_params_
                # Save CV results
                cv_path = out_dir / dataset.upper() / f"{name}_cv_results.csv"
                pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)
        if best_params is None:
            # Fit without search
            est.fit(X_train, y_train)

        y_pred = est.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        print(f"  {name} -> MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, MAPE={metrics['MAPE']:.3f}")
        row = {'model': name, **metrics}
        if best_params is not None:
            row.update({f"param:{k}": v for k, v in best_params.items()})
            # Save params JSON
            with open(out_dir / dataset.upper() / f"{name}_best_params.json", 'w') as f:
                json.dump(best_params, f, indent=2)
        results.append(row)

        # Optional: save feature importances for tree models
        try:
            model = est.named_steps.get('model') if hasattr(est, 'named_steps') else None
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                imp_path = out_dir / dataset.upper() / f"{name}_feature_importance.csv"
                importances.to_csv(imp_path, header=['importance'])
        except Exception:
            pass

    # Save metrics
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / dataset.upper() / 'metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Train/test baseline ML models for RUL prediction (per-cycle).')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'CALCE'], help='Dataset to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to processed data directory for the dataset')
    parser.add_argument('--output_dir', type=str, default='rul_baselines', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_tune', action='store_true', help='Disable hyperparameter tuning')
    parser.add_argument('--cv_splits', type=int, default=5, help='CV splits for hyperparameter tuning')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration for XGBoost if available')
    args = parser.parse_args()

    run(args.dataset, args.data_path, args.output_dir, seed=args.seed, tune=not args.no_tune, cv_splits=args.cv_splits, use_gpu=args.gpu)


if __name__ == '__main__':
    main()


