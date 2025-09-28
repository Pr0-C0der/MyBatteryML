from __future__ import annotations

"""
Train RUL models with sliding-window or battery-level features (new, simplified).

Goals:
- Keep the data flow linear and easy to debug
- Provide verbose printouts controlled by --verbose
- Support two modes:
  * window: sequential sliding windows (features over N cycles -> next-cycle RUL)
  * battery: one vector per battery from early cycles; label is either total life or RUL at cutoff
- Support early-cycle filtering via --min_cycle_index / --max_cycle_index
- No padding in battery-level mode (truncate to desired length)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.training.train_rul_baselines import build_train_test_lists
from batteryml.data_analysis import cycle_features as cf


# -----------------------------
# Feature selection utilities
# -----------------------------
def list_available_feature_functions() -> Dict[str, callable]:
    names = [
        'avg_c_rate', 'max_temperature', 'max_discharge_capacity', 'max_charge_capacity',
        'avg_discharge_capacity', 'avg_charge_capacity',
        'charge_cycle_length', 'discharge_cycle_length', 'peak_cv_length',
        'cycle_length', 'power_during_charge_cycle', 'power_during_discharge_cycle',
        'avg_charge_c_rate', 'avg_discharge_c_rate', 'charge_to_discharge_time_ratio'
    ]
    feats: Dict[str, callable] = {}
    for n in names:
        fn = getattr(cf, n, None)
        if callable(fn):
            feats[n] = fn
    return feats


def resolve_feature_functions(requested: Optional[List[str]], verbose: bool) -> Tuple[List[str], Dict[str, callable]]:
    all_fns = list_available_feature_functions()
    if not requested or requested == ['default']:
        # A safe, interpretable default
        feature_names = [
            'avg_c_rate', 'max_discharge_capacity', 'max_charge_capacity',
            'avg_discharge_capacity', 'avg_charge_capacity',
            'charge_cycle_length', 'discharge_cycle_length', 'cycle_length'
        ]
    elif requested == ['all']:
        feature_names = list(all_fns.keys())
    else:
        feature_names = [n for n in requested if n in all_fns]
        if not feature_names:
            if verbose:
                print("[warn] No valid feature names provided; falling back to default set.")
            feature_names = [
                'avg_c_rate', 'max_discharge_capacity', 'max_charge_capacity',
                'avg_discharge_capacity', 'avg_charge_capacity',
                'charge_cycle_length', 'discharge_cycle_length', 'cycle_length'
            ]
    feature_fns = {n: all_fns[n] for n in feature_names}
    if verbose:
        print(f"[features] Using {len(feature_names)} features: {', '.join(feature_names)}")
    return feature_names, feature_fns


# -----------------------------
# Data and label utilities
# -----------------------------
def compute_total_rul(battery: BatteryData) -> int:
    annot = RULLabelAnnotator()
    try:
        v = int(annot.process_cell(battery).item())
        return v if np.isfinite(v) else 0
    except Exception:
        return 0


def build_cycle_feature_table(battery: BatteryData, feature_fns: Dict[str, callable], verbose: bool) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for c in battery.cycle_data:
        row: Dict[str, float] = {'cycle_number': c.cycle_number}
        for name, fn in feature_fns.items():
            try:
                val = fn(battery, c)
                row[name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
            except Exception:
                row[name] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        print(f"  [table] cycles={len(df)} columns={list(df.columns)}")
    return df


def filter_cycles(df: pd.DataFrame, min_idx: Optional[int], max_idx: Optional[int], verbose: bool) -> pd.DataFrame:
    dfn = df.sort_values('cycle_number').reset_index(drop=True)
    if min_idx is not None or max_idx is not None:
        lo = -np.inf if min_idx is None else float(min_idx)
        hi = np.inf if max_idx is None else float(max_idx)
        dfn = dfn[(dfn['cycle_number'] >= lo) & (dfn['cycle_number'] <= hi)].reset_index(drop=True)
    if verbose:
        rng = (min_idx if min_idx is not None else '-') , (max_idx if max_idx is not None else '-')
        print(f"  [filter] range={rng} -> kept_cycles={len(dfn)}")
    return dfn


# -----------------------------
# Window builders
# -----------------------------
def make_windows_sequential(dfn: pd.DataFrame, feature_names: List[str], total_rul: int, window_size: int, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in feature_names if c in dfn.columns]
    if not cols:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)
    arr = dfn[cols].to_numpy()  # [C, F]
    C = arr.shape[0]
    last_start = C - (window_size + 1)
    if last_start < 0:
        return np.zeros((0, len(cols) * window_size), dtype=float), np.zeros((0,), dtype=float)
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for k in range(0, last_start + 1):
        w = arr[k:k + window_size, :]
        X_list.append(w.reshape(-1))
        y_list.append(float(max(0, total_rul - (k + window_size))))
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    if verbose:
        print(f"  [windows] seq windows={X.shape[0]} window_size={window_size} features={len(cols)}")
    return X, y


def make_vector_battery(dfn: pd.DataFrame,
                        feature_names: List[str],
                        total_rul: int,
                        window_size: Optional[int],
                        battery_label_mode: str,
                        drop_terminal_offset: bool,
                        verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    cols = [c for c in feature_names if c in dfn.columns]
    if not cols:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)
    arr = dfn[cols].to_numpy()  # [C, F]
    C = arr.shape[0]

    # Determine desired cycles to keep
    if window_size is None:
        desired = C
    else:
        desired = min(C, int(window_size))
    if drop_terminal_offset and desired > 1:
        desired -= 1

    if desired <= 0:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    w = arr[:desired, :]  # truncate, no padding
    x = w.reshape(1, -1)

    if battery_label_mode == 'total_life':
        y = np.array([float(total_rul)], dtype=float)
    else:
        # RUL at cutoff (next after the window) – require next step exists
        cutoff = desired
        y = np.array([float(max(0, total_rul - cutoff))], dtype=float)

    if verbose:
        print(f"  [battery] vec_len={x.shape[1]} cycles_used={desired} label_mode={battery_label_mode}")
    return x, y


# -----------------------------
# Dataset preparation
# -----------------------------
def prepare_dataset(files: List[Path],
                    feature_fns: Dict[str, callable],
                    feature_names: List[str],
                    mode: str,
                    window_size: Optional[int],
                    min_cycle_index: Optional[int],
                    max_cycle_index: Optional[int],
                    battery_label_mode: str,
                    drop_terminal_offset: bool,
                    verbose: bool,
                    progress_desc: str) -> Tuple[np.ndarray, np.ndarray]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for f in files:
        try:
            b = BatteryData.load(f)
        except Exception as e:
            if verbose:
                print(f"[warn] failed to load {f}: {e}")
            continue
        if verbose:
            print(f"[cell] {b.cell_id}")
        total_rul = compute_total_rul(b)
        df = build_cycle_feature_table(b, feature_fns, verbose)
        dfn = filter_cycles(df, min_cycle_index, max_cycle_index, verbose)

        if mode == 'window':
            if window_size is None:
                if verbose:
                    print("  [skip] window_size is None for window mode")
                continue
            Xw, yw = make_windows_sequential(dfn, feature_names, total_rul, int(window_size), verbose)
        else:
            Xw, yw = make_vector_battery(dfn, feature_names, total_rul, window_size, battery_label_mode, drop_terminal_offset, verbose)

        if Xw.size and yw.size:
            Xs.append(Xw)
            ys.append(yw)

    if not Xs:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    if verbose:
        print(f"[dataset] {progress_desc}: X={X.shape} y={y.shape}")
    return X, y


# -----------------------------
# Models
# -----------------------------
def build_models(use_gpu: bool, verbose: bool) -> Dict[str, Pipeline]:
    base_steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
    models: Dict[str, Pipeline] = {}
    # Linear models
    models['linear_regression'] = Pipeline(base_steps + [('model', LinearRegression())])
    models['ridge'] = Pipeline(base_steps + [('model', Ridge(alpha=1.0))])
    models['elastic_net'] = Pipeline(base_steps + [('model', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))])
    # Kernel
    models['svr_rbf'] = Pipeline(base_steps + [('model', SVR(kernel='rbf', C=10.0, gamma='scale'))])
    # Trees
    models['random_forest'] = Pipeline(base_steps + [('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
    # Shallow MLP
    models['mlp'] = Pipeline(base_steps + [('model', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', batch_size=256, max_iter=300, random_state=42))])
    # XGBoost
    if _HAS_XGB:
        models['xgboost'] = Pipeline(base_steps + [('model', XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1, tree_method='hist', device=('cuda' if use_gpu else 'cpu')
        ))])
    # PLSR (required by user preference)
    models['plsr'] = Pipeline(base_steps + [('model', PLSRegression(n_components=10))])
    # PCR (PCA + Linear) (required by user preference)
    models['pcr'] = Pipeline(base_steps + [('model', Pipeline([('pca', PCA(n_components=20)), ('lr', LinearRegression())]))])

    if verbose:
        print(f"[models] Built {len(models)} models: {', '.join(models.keys())}")
    return models


# -----------------------------
# Orchestration
# -----------------------------
def run(dataset: str,
        data_path: List[str] | str,
        output_dir: str,
        mode: str,
        window_size: Optional[int],
        features: Optional[List[str]],
        min_cycle_index: Optional[int],
        max_cycle_index: Optional[int],
        battery_label_mode: str,
        drop_terminal_offset: bool,
        use_gpu: bool,
        verbose: bool):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build split lists (reuses existing splitter logic)
    train_files, test_files = build_train_test_lists(dataset, data_path)
    if verbose:
        print(f"[split] train={len(train_files)} test={len(test_files)}")

    feature_names, feature_fns = resolve_feature_functions(features, verbose)

    # Prepare datasets
    X_train, y_train = prepare_dataset(
        train_files, feature_fns, feature_names, mode, window_size,
        min_cycle_index, max_cycle_index, battery_label_mode, drop_terminal_offset,
        verbose, progress_desc='train')
    X_test, y_test = prepare_dataset(
        test_files, feature_fns, feature_names, mode, window_size,
        min_cycle_index, max_cycle_index, battery_label_mode, drop_terminal_offset,
        verbose, progress_desc='test')

    if X_train.size == 0 or X_test.size == 0:
        print("No data available after feature preparation. Aborting.")
        return

    # Diagnostics: all-NaN columns in train
    try:
        num_feats = len(feature_names)
        width = int(X_train.shape[1])
        all_nan_cols = np.all(np.isnan(X_train), axis=0)
        if np.any(all_nan_cols):
            print("[info] Detected all-NaN columns in train (will be skipped by imputer):")
            if width % num_feats == 0:
                for idx in np.where(all_nan_cols)[0].tolist():
                    cycle_off = idx // num_feats
                    feat_idx = idx % num_feats
                    feat_name = feature_names[feat_idx] if 0 <= feat_idx < num_feats else f"idx_{feat_idx}"
                    print(f"  - col {idx}: feature='{feat_name}', cycle_offset={cycle_off}")
            else:
                print(f"  [warn] cannot map columns to (cycle,feature): width={width}, num_features={num_feats}")
    except Exception:
        pass

    print(f"Train X={X_train.shape} Test X={X_test.shape} Features={len(feature_names)} Mode={mode} Window={window_size}")

    # Train/evaluate
    models = build_models(use_gpu, verbose)
    rows = []
    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        rows.append({'model': name, 'MAE': float(mae), 'RMSE': float(rmse)})
        print(f"  {name}: MAE={mae:.3f} RMSE={rmse:.3f}")

    pd.DataFrame(rows).to_csv(out_dir / f"rul_results_{mode}_ws{window_size}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='RUL training (simplified, modular, verbose)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100'])
    parser.add_argument('--data_path', type=str, nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, default='rul_windows_new')
    parser.add_argument('--mode', type=str, choices=['window', 'battery'], default='window')
    parser.add_argument('--window_size', type=int, default=None, help='Window length; required for mode=window; optional for mode=battery (truncate)')
    parser.add_argument('--features', type=str, nargs='*', default=['default'])
    parser.add_argument('--min_cycle_index', type=int, default=0)
    parser.add_argument('--max_cycle_index', type=int, default=99)
    parser.add_argument('--battery_label_mode', type=str, choices=['rul_at_cutoff', 'total_life'], default='total_life')
    parser.add_argument('--drop_terminal_offset', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    run(
        dataset=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        mode=args.mode,
        window_size=args.window_size,
        features=args.features,
        min_cycle_index=args.min_cycle_index,
        max_cycle_index=args.max_cycle_index,
        battery_label_mode=args.battery_label_mode,
        drop_terminal_offset=args.drop_terminal_offset,
        use_gpu=args.gpu,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()


