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


def run(data_path: str,
        features: List[str],
        n_samples: int,
        output_dir: str,
        caps: List[int],
        seed: int,
        verbose: bool):
    data_dir = Path(data_path)
    out_dir = Path(output_dir)
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
            indiv_dir = feat_dir / 'individual_caps'
            merged_dir = feat_dir / 'merged'
            indiv_dir.mkdir(parents=True, exist_ok=True)
            merged_dir.mkdir(parents=True, exist_ok=True)

            saved_paths: List[Path] = []
            prev = 0
            for cap in caps:
                fig, ax = plt.subplots(figsize=(10, 6))
                ok = _plot_single_cap(ax, x, y, prev, cap)
                ax.set_xlabel('Cycle number')
                ax.set_ylabel(feature.replace('_', ' ').title())
                ax.set_title(f"{feature.replace('_', ' ').title()} vs Cycle — {b.cell_id} ({prev}–{cap})")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fname = indiv_dir / f"{_safe_id(b.cell_id)}_{feature}_range_{prev}_{cap}.png"
                if ok:
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    saved_paths.append(fname)
                plt.close(fig)
                if verbose:
                    print(f"[ok] saved {fname}")
                prev = cap

            # Merge per-feature images into one composite
            if saved_paths:
                merged_path = merged_dir / f"{_safe_id(b.cell_id)}_{feature}_merged.png"
                merged_ok = _merge_images_horiz(saved_paths, merged_path)
                if verbose and merged_ok:
                    print(f"[ok] merged -> {merged_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot feature vs cycle for random MATR batteries with multiple cycle caps; separate per-cap images then merged')
    parser.add_argument('--data_path', type=str, required=True, help='Path to MATR preprocessed directory (contains *.pkl)')
    parser.add_argument('--feature', type=str, default='default', help="Feature name or 'default' to use the default set from training")
    parser.add_argument('--n', type=int, default=30, help='Number of random batteries to plot')
    parser.add_argument('--output_dir', type=str, default='feature_vs_cycle_random', help='Output directory for plots')
    parser.add_argument('--caps', type=int, nargs='*', default=[100, 200, 500], help='Cycle caps to overlay (e.g., 100 200 500)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    # Determine feature list
    if args.feature is None or args.feature == 'default':
        features = _default_feature_names()
    else:
        features = [args.feature]
    run(args.data_path, features, args.n, args.output_dir, args.caps, args.seed, args.verbose)


if __name__ == '__main__':
    main()


