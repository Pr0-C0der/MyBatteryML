from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from batteryml.data.battery_data import BatteryData
from batteryml.training.train_rul_baselines import build_train_test_lists
from batteryml.data_analysis import cycle_features as cf
from batteryml.label.rul import RULLabelAnnotator


def _available_feature_fns() -> Dict[str, callable]:
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


def _default_feature_names() -> List[str]:
    all_fns = _available_feature_fns()
    # Exclude any features related to temperature or internal resistance
    excluded_tokens = ('temperature', 'internal')
    names = [n for n in all_fns.keys() if not any(tok in n.lower() for tok in excluded_tokens)]

    print(f'Selected features: {sorted(names)}', flush=True)
    return sorted(names)


def _compute_total_rul(battery: BatteryData) -> int:
    annot = RULLabelAnnotator()
    try:
        v = int(annot.process_cell(battery).item())
        return v if np.isfinite(v) else 0
    except Exception:
        return 0


def _build_cycle_feature_table(battery: BatteryData, feature_fns: Dict[str, callable]) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def _filter_df_by_cycle_limit(df: pd.DataFrame, cycle_limit: int) -> pd.DataFrame:
    dfn = df.sort_values('cycle_number').reset_index(drop=True)
    return dfn[dfn['cycle_number'] <= int(cycle_limit)].reset_index(drop=True)


def _make_battery_vector(df: pd.DataFrame, selected_features: List[str], cycle_limit: int) -> np.ndarray:
    dfn = _filter_df_by_cycle_limit(df, cycle_limit)
    cols = [c for c in selected_features if c in dfn.columns]
    if not cols:
        return np.zeros((0,), dtype=float)
    K = int(cycle_limit)
    mat = dfn[cols].to_numpy()
    F = len(cols)
    if mat.shape[0] >= K:
        matK = mat[:K, :]
    else:
        pad = np.full((K - mat.shape[0], F), np.nan, dtype=float)
        matK = np.vstack([mat, pad])
    vec = matK.reshape(-1)
    # Replace non-finite with NaN for downstream imputation
    vec[~np.isfinite(vec)] = np.nan
    return vec


def _fit_label_transform(y: np.ndarray) -> tuple[np.ndarray, dict]:
    y = np.asarray(y, dtype=float)
    yt = np.log1p(np.clip(y, a_min=0.0, a_max=None))
    mu = float(np.mean(yt))
    sigma = float(np.std(yt)) if np.std(yt) > 1e-6 else 1e-6
    yt = (yt - mu) / sigma
    return yt, {'mu': mu, 'sigma': sigma}


def _inverse_label_transform(yt: np.ndarray, stats: dict) -> np.ndarray:
    mu = stats['mu']; sigma = stats['sigma']
    y_log = yt * sigma + mu
    # Clip to avoid overflow in expm1
    clip_k = 5.0
    y_log = np.clip(y_log, mu - clip_k * sigma, mu + clip_k * sigma)
    y = np.expm1(y_log)
    return np.clip(y, a_min=0.0, a_max=None)


def _precompute_feature_tables(files: List[Path], feature_fns: Dict[str, callable]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for f in tqdm(files, desc='Building per-battery feature tables'):
        try:
            b = BatteryData.load(f)
        except Exception:
            continue
        df = _build_cycle_feature_table(b, feature_fns)
        out[b.cell_id] = df
    return out


def _assemble_dataset(feature_tables: Dict[str, pd.DataFrame], files: List[Path], selected: List[str], cycle_limit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    g_list: List[str] = []
    for f in files:
        try:
            b = BatteryData.load(f)
        except Exception:
            continue
        df = feature_tables.get(b.cell_id)
        if df is None:
            continue
        vec = _make_battery_vector(df, selected, cycle_limit)
        if vec.size == 0:
            continue
        X_list.append(vec)
        y_list.append(float(_compute_total_rul(b)))
        g_list.append(b.cell_id)
    if not X_list:
        return np.zeros((0, len(selected) * int(cycle_limit)), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=object)
    X = np.vstack([x.reshape(1, -1) for x in X_list])
    # Replace any inf with NaN for imputation later
    X[~np.isfinite(X)] = np.nan
    y = np.array(y_list, dtype=float)
    g = np.array(g_list, dtype=object)
    return X, y, g


def forward_select_linear(dataset: str, data_path: List[str], cycle_limit: int, pool_features: List[str], cv_splits: int = 5, verbose: bool = True):
    train_files, test_files = build_train_test_lists(dataset, data_path)
    if verbose:
        print(f"Found {len(train_files)} train and {len(test_files)} test batteries for {dataset}", flush=True)
    if not train_files or not test_files:
        print("No train/test files found. Check --data_path.", flush=True)
        return

    all_fns = _available_feature_fns()
    feature_fns = {n: all_fns[n] for n in pool_features if n in all_fns}

    # Precompute feature tables for all pool features
    if verbose:
        print("Precomputing train feature tables...", flush=True)
    train_tables = _precompute_feature_tables(train_files, feature_fns)
    if verbose:
        print("Precomputing test feature tables...", flush=True)
    test_tables = _precompute_feature_tables(test_files, feature_fns)

    selected: List[str] = []
    remaining: List[str] = list(feature_fns.keys())
    best_overall_rmse = np.inf
    iter_idx = 0

    while remaining:
        iter_idx += 1
        # Choose best next feature via GroupKFold RMSE on train
        best_feat = None
        best_cv_rmse = np.inf
        for feat in remaining:
            trial_feats = selected + [feat]
            X_tr, y_tr, g_tr = _assemble_dataset(train_tables, train_files, trial_feats, cycle_limit)
            if X_tr.size == 0:
                continue
            # Label transform on whole training set (kept consistent per trial)
            y_tr_t, stats = _fit_label_transform(y_tr)
            cv = GroupKFold(n_splits=min(cv_splits, max(2, len(np.unique(g_tr)))))
            rmses = []
            for tr_idx, va_idx in cv.split(X_tr, y_tr_t, groups=g_tr):
                Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
                ytr_t, yva = y_tr_t[tr_idx], y_tr[va_idx]
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('model', LinearRegression()),
                ])
                model.fit(Xtr, ytr_t)
                pred_t = model.predict(Xva)
                pred = _inverse_label_transform(np.asarray(pred_t, dtype=float), stats)
                rmse = mean_squared_error(yva, pred) ** 0.5
                rmses.append(float(rmse))
            mean_rmse = float(np.mean(rmses)) if rmses else np.inf
            if mean_rmse < best_cv_rmse:
                best_cv_rmse = mean_rmse
                best_feat = feat

        if best_feat is None:
            if verbose:
                print("No further improvement; stopping.", flush=True)
            break

        # Add the best feature
        selected.append(best_feat)
        remaining.remove(best_feat)

        # Fit on full train with current selected, evaluate Test RMSE
        X_tr_full, y_tr_full, _ = _assemble_dataset(train_tables, train_files, selected, cycle_limit)
        y_tr_t_full, stats_full = _fit_label_transform(y_tr_full)
        model_full = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', LinearRegression()),
        ])
        model_full.fit(X_tr_full, y_tr_t_full)
        X_te, y_te, _ = _assemble_dataset(test_tables, test_files, selected, cycle_limit)
        if X_te.size == 0:
            print("No test features; skipping test RMSE.", flush=True)
            continue
        pred_t = model_full.predict(X_te)
        y_pred = _inverse_label_transform(np.asarray(pred_t, dtype=float), stats_full)
        test_rmse = mean_squared_error(y_te, y_pred) ** 0.5

        if verbose:
            print(f"Iter {iter_idx}: +{best_feat} | selected={len(selected)} | Test RMSE={test_rmse:.3f} | CV RMSE={best_cv_rmse:.3f}", flush=True)

        if test_rmse + 1e-6 < best_overall_rmse:
            best_overall_rmse = test_rmse
        else:
            # Optional early stop if test RMSE not improving
            pass

    if verbose:
        print(f"Done. Selected features ({len(selected)}): {selected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Forward feature selection for Linear Regression (battery-level)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'MATR1', 'MATR2', 'CLO', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100'], help='Dataset name')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='One or more paths: directories with .pkl or a file listing .pkl paths')
    parser.add_argument('--cycle_limit', type=int, default=100, help='Use only cycles <= this index for features')
    parser.add_argument('--features', type=str, nargs='*', default=['default'], help="Pool features: 'default', 'all', or explicit list")
    parser.add_argument('--cv_splits', type=int, default=5, help='GroupKFold splits for selection (default 5)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Build pool features
    all_fns = _available_feature_fns()
    if not args.features or args.features == ['default']:
        pool = _default_feature_names()
    elif args.features == ['all']:
        pool = list(all_fns.keys())
    else:
        pool = [f for f in args.features if f in all_fns]
        if not pool:
            pool = _default_feature_names()

    forward_select_linear(args.dataset, args.data_path, args.cycle_limit, pool, cv_splits=args.cv_splits, verbose=args.verbose)


if __name__ == '__main__':
    main()


