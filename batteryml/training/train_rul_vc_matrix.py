from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.base import clone

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

# Optional RAPIDS cuML
try:
    import cuml  # noqa: F401
    _HAS_CUML = True
except Exception:
    _HAS_CUML = False

# Optional GPyTorch
try:
    import torch  # noqa: F401
    import gpytorch  # noqa: F401
    _HAS_GPYTORCH = True
except Exception:
    _HAS_GPYTORCH = False

from batteryml.data.battery_data import BatteryData
from batteryml.label.rul import RULLabelAnnotator
from batteryml.training.train_rul_baselines import build_train_test_lists
from batteryml.feature.severson import get_Qdlin, smooth


def _compute_total_rul(battery: BatteryData) -> int:
    annot = RULLabelAnnotator()
    try:
        v = int(annot.process_cell(battery).item())
        return v if np.isfinite(v) else 0
    except Exception:
        return 0


def _make_battery_vc_vector(battery: BatteryData,
                            cycle_limit: int,
                            diff_base: int = 9,
                            smooth_diff: bool = True,
                            cycle_average: Optional[int] = None) -> np.ndarray:
    min_idx = 0
    max_idx = max(0, int(cycle_limit) - 1)
    num_cycles = len(battery.cycle_data)
    if num_cycles == 0 or min_idx >= num_cycles:
        return np.zeros((0,), dtype=float)

    # Clamp diff_base within available range
    db = int(np.clip(diff_base, min_idx, min(max_idx, num_cycles - 1)))
    try:
        base_qdlin = get_Qdlin(battery, battery.cycle_data[db], use_precalculated=False)
    except Exception:
        return np.zeros((0,), dtype=float)
    if smooth_diff:
        base_qdlin = smooth(base_qdlin)
    if cycle_average is not None:
        base_qdlin = base_qdlin[..., ::int(cycle_average)]

    rows: List[np.ndarray] = []
    for idx, c in enumerate(battery.cycle_data):
        if idx < min_idx:
            continue
        if idx > max_idx:
            break
        try:
            qd = get_Qdlin(battery, c, use_precalculated=False)
            if smooth_diff:
                qd = smooth(qd)
            if cycle_average is not None:
                qd = qd[..., ::int(cycle_average)]
            diff = qd - base_qdlin
            if smooth_diff:
                diff = smooth(diff)
            rows.append(np.asarray(diff, dtype=float))
        except Exception:
            continue

    if not rows:
        return np.zeros((0,), dtype=float)
    mat = np.stack(rows, axis=0)  # [cycles, interp]
    # Replace non-finite with zeros to stabilize
    mat[~np.isfinite(mat)] = 0.0
    return mat.reshape(-1)


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
    y = np.expm1(y_log)
    return np.clip(y, a_min=0.0, a_max=None)


def _prepare_dataset_battery_level(files: List[Path],
                                   cycle_limit: int,
                                   diff_base: int,
                                   smooth_diff: bool,
                                   cycle_average: Optional[int],
                                   progress_desc: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    g_list: List[str] = []
    for f in tqdm(files, desc=progress_desc):
        try:
            b = BatteryData.load(f)
        except Exception:
            continue
        vec = _make_battery_vc_vector(b, cycle_limit, diff_base=diff_base, smooth_diff=smooth_diff, cycle_average=cycle_average)
        if vec.size == 0:
            continue
        X_list.append(vec)
        y_list.append(float(_compute_total_rul(b)))
        g_list.append(b.cell_id)
    if not X_list:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=object)
    # Align feature length across batteries (pad with zeros if needed)
    max_len = max(x.size for x in X_list)
    X = np.vstack([np.pad(x, (0, max_len - x.size), mode='constant', constant_values=0.0).reshape(1, -1) for x in X_list])
    y = np.array(y_list, dtype=float)
    g = np.array(g_list, dtype=object)
    return X, y, g


def _build_models(use_gpu: bool = False) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    base_steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0.0)), ('scaler', StandardScaler())]

    gpu_available = False
    if use_gpu:
        try:
            import torch  # type: ignore
            gpu_available = bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())
        except Exception:
            gpu_available = False

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
        models['linear_regression'] = Pipeline(base_steps + [('model', LinearRegression())])
        models['ridge'] = Pipeline(base_steps + [('model', Ridge(alpha=1.0))])
        models['elastic_net'] = Pipeline(base_steps + [('model', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))])
        models['svr'] = Pipeline(base_steps + [('model', SVR(kernel='rbf', C=10.0, gamma='scale'))])
        models['random_forest'] = Pipeline(base_steps + [('model', RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1))])

    models['mlp'] = Pipeline(base_steps + [('model', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', batch_size=256, max_iter=300, random_state=42))])
    if _HAS_XGB:
        xgb_device = 'cuda' if gpu_available else 'cpu'
        models['xgboost'] = Pipeline(base_steps + [('model', XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, tree_method='hist', device=xgb_device))])
    models['plsr'] = Pipeline(base_steps + [('model', PLSRegression(n_components=10))])
    models['pcr'] = Pipeline(base_steps + [('model', Pipeline([('pca', PCA(n_components=20)), ('lr', LinearRegression())]))])

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


def run(dataset: str,
        data_path: List[str],
        output_dir: str,
        cycle_limit: int,
        diff_base: int = 9,
        vc_cycle_average: Optional[int] = None,
        use_gpu: bool = False,
        tune: bool = False,
        cv_splits: int = 5):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, test_files = build_train_test_lists(dataset, data_path)
    print(f"Found {len(train_files)} train and {len(test_files)} test batteries for {dataset}.")

    X_train, y_train, g_train = _prepare_dataset_battery_level(train_files, cycle_limit, diff_base, True, vc_cycle_average, 'Building train VC vectors')
    X_test, y_test, _ = _prepare_dataset_battery_level(test_files, cycle_limit, diff_base, True, vc_cycle_average, 'Building test VC vectors')

    if X_train.size == 0 or X_test.size == 0:
        print("No data available after VC feature building. Aborting.")
        return

    print(f"Train batteries: {X_train.shape}, Test batteries: {X_test.shape}, Feature dim: {X_train.shape[1] if X_train.size else 0}")

    # Label transforms
    y_train_t, train_label_stats = _fit_label_transform(y_train)

    models = _build_models(use_gpu=use_gpu)
    rows = []
    for name, pipe in models.items():
        print(f"Training {name}...")
        est = pipe
        best_params = None
        if tune:
            if name == 'linear_regression':
                param_grid = {'model__fit_intercept': [True, False]}
            elif name == 'ridge':
                param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0], 'model__fit_intercept': [True, False]}
            elif name == 'elastic_net':
                param_grid = {'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1], 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7], 'model__max_iter': [10000]}
            elif name == 'svr':
                param_grid = {'model__kernel': ['rbf', 'linear'], 'model__C': [0.01, 0.1, 1.0, 10.0], 'model__gamma': ['scale', 'auto'], 'model__epsilon': [0.05, 0.1]}
            elif name == 'random_forest':
                param_grid = {'model__n_estimators': [200, 400, 800], 'model__max_depth': [None, 10, 20, 40], 'model__min_samples_split': [2, 5], 'model__min_samples_leaf': [1, 2]}
            elif name == 'xgboost':
                param_grid = {'model__n_estimators': [300, 600], 'model__max_depth': [4, 6, 8], 'model__learning_rate': [0.01, 0.05, 0.1], 'model__subsample': [0.8, 1.0], 'model__colsample_bytree': [0.8, 1.0]}
            elif name == 'mlp':
                param_grid = {'model__hidden_layer_sizes': [(128, 64), (256, 128)], 'model__alpha': [1e-6, 1e-5, 1e-4], 'model__learning_rate_init': [0.0005, 0.001], 'model__activation': ['relu']}
            elif name == 'plsr':
                param_grid = {'model__n_components': [5, 10, 20]}
            elif name == 'pcr':
                param_grid = {'model__pca__n_components': [10, 20, 40], 'model__lr__fit_intercept': [True, False]}
            elif name == 'gpytorch_svgp':
                param_grid = {'model__inducing_points': [256, 512], 'model__batch_size': [512, 1024], 'model__iters': [100, 200], 'model__lr': [1e-2, 5e-3]}
            else:
                param_grid = {}

            if param_grid:
                unique_groups = np.unique(g_train)
                n_splits = min(cv_splits, len(unique_groups)) if len(unique_groups) > 1 else 2
                cv = GroupKFold(n_splits=n_splits)
                grid = list(ParameterGrid(param_grid))
                best_score = np.inf
                results = []
                pbar = tqdm(grid, desc=f"Tuning {name} ({len(grid)} cfgs x {n_splits} folds)")
                for params in pbar:
                    fold_rmses = []
                    for tr_idx, va_idx in cv.split(X_train, y_train, groups=g_train):
                        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                        y_tr_t, fold_stats = _fit_label_transform(y_tr)
                        model = clone(pipe)
                        model.set_params(**params)
                        model.fit(X_tr, y_tr_t)
                        pred_t = model.predict(X_va)
                        pred = _inverse_label_transform(np.asarray(pred_t, dtype=float), fold_stats)
                        fold_rmses.append(mean_squared_error(y_va, pred) ** 0.5)
                    mean_rmse = float(np.mean(fold_rmses)) if fold_rmses else np.inf
                    results.append({**{f"param:{k}": v for k, v in params.items()}, 'mean_RMSE': mean_rmse})
                    pbar.set_postfix({"RMSE": f"{mean_rmse:.3f}"})
                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_params = params
                        est = clone(pipe).set_params(**params)
                cv_path = out_dir / f"{dataset.upper()}_{name}_cv_results_vc.csv"
                pd.DataFrame(results).to_csv(cv_path, index=False)

        if best_params is not None:
            est = clone(pipe).set_params(**best_params)
        est.fit(X_train, y_train_t)
        y_pred_t = est.predict(X_test)
        y_pred = _inverse_label_transform(np.asarray(y_pred_t, dtype=float), {'mu': train_label_stats['mu'], 'sigma': train_label_stats['sigma']})
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        rows.append({'model': name, 'MAE': float(mae), 'RMSE': float(rmse)})
        print(f"  {name}: MAE={mae:.3f} RMSE={rmse:.3f}")

    suffix = f"_vc_cl{cycle_limit}"
    pd.DataFrame(rows).to_csv(out_dir / f"rul_metrics{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='RUL prediction using Voltage-Capacity matrix (battery-level)')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'MATR1', 'MATR2', 'CLO', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100'], help='Dataset name')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='One or more paths: directories with .pkl or a file listing .pkl paths')
    parser.add_argument('--output_dir', type=str, default='rul_vc', help='Output directory')
    parser.add_argument('--cycle_limit', type=int, default=100, help='Use cycles <= this index (early cycles)')
    parser.add_argument('--diff_base', type=int, default=9, help='Cycle index used as diff base (Severson)')
    parser.add_argument('--vc_cycle_average', type=int, default=None, help='Optional stride to downsample VC curve (e.g., 5)')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU where available')
    parser.add_argument('--tune', action='store_true', help='Enable grid search with GroupKFold')
    parser.add_argument('--cv_splits', type=int, default=5, help='Number of GroupKFold splits')
    args = parser.parse_args()

    run(args.dataset, args.data_path, args.output_dir, args.cycle_limit, diff_base=args.diff_base, vc_cycle_average=args.vc_cycle_average, use_gpu=args.gpu, tune=args.tune, cv_splits=args.cv_splits)


if __name__ == '__main__':
    main()


