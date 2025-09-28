from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from batteryml.data.battery_data import BatteryData
from batteryml.training.train_rul_baselines import build_train_test_lists
from batteryml.data_analysis import cycle_features as cf


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


def scan_files(files: List[Path], feature_fns: Dict[str, callable], min_cycle: int | None, max_cycle: int | None):
    for f in files:
        try:
            b = BatteryData.load(f)
        except Exception as e:
            print(f"[warn] failed to load {f}: {e}")
            continue
        print(f"[battery] {b.cell_id}")
        for c in b.cycle_data:
            idx = getattr(c, 'cycle_number', None)
            if idx is None:
                continue
            if min_cycle is not None and idx < min_cycle:
                continue
            if max_cycle is not None and idx > max_cycle:
                continue
            missing: List[str] = []
            for name, fn in feature_fns.items():
                try:
                    v = fn(b, c)
                    ok = (v is not None) and np.isfinite(float(v))
                except Exception:
                    ok = False
                if not ok:
                    missing.append(name)
            if missing:
                print(f"  cycle {idx}: missing -> {', '.join(missing)}")


def main():
    p = argparse.ArgumentParser(description='Report missing per-cycle features per battery')
    p.add_argument('--dataset', type=str, required=True, choices=['MATR', 'CALCE', 'CRUH', 'CRUSH', 'HUST', 'SNL', 'MIX100'])
    p.add_argument('--data_path', type=str, nargs='+', required=True)
    p.add_argument('--min_cycle_index', type=int, default=None)
    p.add_argument('--max_cycle_index', type=int, default=None)
    p.add_argument('--features', type=str, nargs='*', default=['all'], help="'all' or list of feature names")
    args = p.parse_args()

    # Build file lists (use same split builder for convenience; we just concatenate)
    train_files, test_files = build_train_test_lists(args.dataset, args.data_path)
    files = list(train_files) + list(test_files)

    feats = _available_feature_fns()
    if args.features != ['all']:
        feats = {n: fn for n, fn in feats.items() if n in set(args.features)}
        if not feats:
            print('[error] no valid features selected')
            return

    scan_files(files, feats, args.min_cycle_index, args.max_cycle_index)


if __name__ == '__main__':
    main()


