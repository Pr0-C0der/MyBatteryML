from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from batteryml.data.battery_data import BatteryData
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
from batteryml.data_analysis import cycle_features as cf


def _safe_id(cell_id: str) -> str:
    return cell_id.replace('/', '_').replace('\\', '_')


def _get_feature_fn(name: str):
    fn = getattr(cf, name, None)
    if callable(fn):
        return fn
    raise ValueError(f"Unknown feature '{name}'. Available include e.g. avg_voltage, avg_current, avg_c_rate, max_discharge_capacity, max_charge_capacity, cycle_length, ...")


def _compute_feature_series(battery: BatteryData, feature_name: str):
    fn = _get_feature_fn(feature_name)
    xs: List[int] = []
    ys: List[float] = []
    for c in battery.cycle_data:
        try:
            val = fn(battery, c)
            if val is None:
                continue
            f = float(val)
            if np.isfinite(f):
                xs.append(getattr(c, 'cycle_number', len(xs)))
                ys.append(f)
        except Exception:
            continue
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(np.array(xs, dtype=float))
    x_arr = np.array(xs, dtype=float)[order]
    y_arr = np.array(ys, dtype=float)[order]
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[m], y_arr[m]


def _plot_single_cap(ax, x, y, start: int, end: int):
    # Plot the segment between (start, end]
    m = (x > start) & (x <= end)
    if not np.any(m):
        return False
    ax.plot(x[m], y[m], linewidth=1.6, color='tab:blue')
    return True


def _merge_images_horiz(paths: List[Path], out_path: Path) -> bool:
    if not _HAS_PIL:
        return False
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert('RGB'))
        except Exception:
            return False
    if not imgs:
        return False
    heights = [im.height for im in imgs]
    max_h = max(heights)
    widths = [im.width for im in imgs]
    total_w = sum(widths)
    canvas = Image.new('RGB', (total_w, max_h), color=(255, 255, 255))
    xoff = 0
    for im in imgs:
        # center vertically
        yoff = (max_h - im.height) // 2
        canvas.paste(im, (xoff, yoff))
        xoff += im.width
    try:
        canvas.save(out_path)
        return True
    except Exception:
        return False


def _default_feature_names() -> List[str]:
    # Match train_rul_windows default list
    return [
        'avg_c_rate', 'max_discharge_capacity', 'max_charge_capacity',
        'avg_discharge_capacity', 'avg_charge_capacity',
        'charge_cycle_length', 'discharge_cycle_length',
        'cycle_length'
    ]


def _moving_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or int(window) <= 1:
        return y
    w = int(window)
    try:
        kernel = np.ones(w, dtype=float) / float(w)
        return np.convolve(y, kernel, mode='same')
    except Exception:
        return y


def run(data_path: str,
        dataset: str,
        features: List[str],
        n_samples: int,
        output_dir: str,
        caps: List[int],
        seed: int,
        lag_window: int,
        verbose: bool):
    data_dir = Path(data_path)
    out_dir = Path(output_dir) / str(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob('*.pkl'))
    if not files:
        print(f"No .pkl files found under {data_dir}")
        return
    random.seed(seed)
    selected = random.sample(files, k=min(n_samples, len(files)))

    for f in selected:
        try:
            b = BatteryData.load(f)
        except Exception as e:
            if verbose:
                print(f"[skip] failed to load {f}: {e}")
            continue
        # Loop through requested features, creating per-feature folders
        for feature in features:
            x, y = _compute_feature_series(b, feature)
            if x.size == 0:
                if verbose:
                    print(f"[skip] {b.cell_id}: no data for feature '{feature}'")
                continue

            feat_dir = out_dir / feature
            merged_dir = feat_dir / 'merged'
            mm_dir = feat_dir / 'moving_mean'
            merged_dir.mkdir(parents=True, exist_ok=True)
            mm_dir.mkdir(parents=True, exist_ok=True)

            # Plot all caps on a single figure (no individual saves)
            prev = 0
            fig, ax = plt.subplots(figsize=(10, 6))
            any_ok = False
            for cap in caps:
                ok = _plot_single_cap(ax, x, y, prev, cap)
                any_ok = any_ok or ok
                prev = cap
            if any_ok:
                ax.set_xlabel('Cycle number')
                ax.set_ylabel(feature.replace('_', ' ').title())
                ax.set_title(f"{feature.replace('_', ' ').title()} vs Cycle — {b.cell_id}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                merged_path = merged_dir / f"{_safe_id(b.cell_id)}_{feature}_merged.png"
                fig.savefig(merged_path, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"[ok] saved {merged_path}")
            plt.close(fig)

            # Moving mean combined figure (no individual saves)
            y_sm = _moving_mean(y, lag_window)
            prev = 0
            fig, ax = plt.subplots(figsize=(10, 6))
            any_ok = False
            for cap in caps:
                ok = _plot_single_cap(ax, x, y_sm, prev, cap)
                any_ok = any_ok or ok
                prev = cap
            if any_ok:
                ax.set_xlabel('Cycle number')
                ax.set_ylabel(feature.replace('_', ' ').title())
                ax.set_title(f"{feature.replace('_', ' ').title()} (moving mean w={lag_window}) — {b.cell_id}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                mm_merged = mm_dir / f"{_safe_id(b.cell_id)}_{feature}_mm_w{lag_window}_merged.png"
                fig.savefig(mm_merged, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"[ok] saved (moving mean) {mm_merged}")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot feature vs cycle for random batteries with multiple cycle caps; separate per-cap images then merged (with optional moving mean).')
    parser.add_argument('--dataset', type=str, required=True, choices=['MATR', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100', 'MATR1', 'MATR2', 'CLO'], help='Dataset name (used only for naming/output)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed directory (contains *.pkl) for the chosen dataset')
    parser.add_argument('--feature', type=str, default='default', help="Feature name or 'default' to use the default set from training")
    parser.add_argument('--n', type=int, default=30, help='Number of random batteries to plot')
    parser.add_argument('--output_dir', type=str, default='feature_vs_cycle_random', help='Output directory for plots')
    parser.add_argument('--caps', type=int, nargs='*', default=[100, 200, 500], help='Cycle caps for contiguous segments (e.g., 100 200 500)')
    parser.add_argument('--lag_window', type=int, default=5, help='Moving mean window (default 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    # Determine feature list
    if args.feature is None or args.feature == 'default':
        features = _default_feature_names()
    else:
        features = [args.feature]
    run(args.data_path, args.dataset, features, args.n, args.output_dir, args.caps, args.seed, args.lag_window, args.verbose)


if __name__ == '__main__':
    main()


