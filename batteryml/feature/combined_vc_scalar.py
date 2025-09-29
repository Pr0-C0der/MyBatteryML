from __future__ import annotations

# Licensed under the MIT License.

import torch
import numpy as np
from typing import List, Optional, Dict, Callable

from batteryml.builders import FEATURE_EXTRACTORS
from batteryml.data.battery_data import BatteryData
from batteryml.feature.base import BaseFeatureExtractor
from batteryml.feature.severson import get_Qdlin, smooth
from batteryml.data_analysis import cycle_features as cf


def _is_finite_array(x: np.ndarray) -> bool:
    return np.isfinite(x).all() and x.size > 0


def _default_scalar_feature_fns() -> Dict[str, Callable[[BatteryData, object], Optional[float]]]:
    # Exclude temperature/internal-resistance by default
    names = [
        'avg_voltage', 'avg_current',
        'avg_c_rate', 'max_discharge_capacity', 'max_charge_capacity',
        'avg_discharge_capacity', 'avg_charge_capacity',
        'charge_cycle_length', 'discharge_cycle_length', 'peak_cv_length',
        'cycle_length', 'power_during_charge_cycle', 'power_during_discharge_cycle',
        'avg_charge_c_rate', 'avg_discharge_c_rate', 'charge_to_discharge_time_ratio',
    ]
    fns: Dict[str, Callable] = {}
    for n in names:
        fn = getattr(cf, n, None)
        if callable(fn):
            fns[n] = fn
    return fns


@FEATURE_EXTRACTORS.register()
class CombinedVCScalarFeatureExtractor(BaseFeatureExtractor):
    """Author-style feature extractor that concatenates per-cycle VC-diff vectors
    with our per-cycle scalar features for early cycles.

    Output shape per cell: [num_cycles_kept, vc_dim + num_scalar_features]
    - VC part follows Severson: diff against base cycle (diff_base), smoothed
    - Scalar part: computed per-cycle from cf.* functions
    - NaN/Inf are zero-filled at the end
    """

    def __init__(self,
                 interp_dim: int = 1000,
                 diff_base: int = 9,
                 min_cycle_index: int = 0,
                 max_cycle_index: int = 99,
                 use_precalculated_qdlin: bool = False,
                 smooth_enabled: bool = True,
                 cycle_average: Optional[int] = None,
                 scalar_features: Optional[List[str]] = None,
                 verbose: bool = False):
        super().__init__()
        self.interp_dim = int(interp_dim)
        self.min_cycle_index = int(min_cycle_index)
        self.max_cycle_index = int(max_cycle_index)
        self.use_precalculated_qdlin = bool(use_precalculated_qdlin)
        self.smooth_enabled = bool(smooth_enabled)
        self.cycle_average = cycle_average if cycle_average is None else int(cycle_average)
        self.verbose = bool(verbose)

        # Scalar features
        all_fns = _default_scalar_feature_fns()
        if scalar_features is None:
            self.scalar_order = list(all_fns.keys())
        else:
            self.scalar_order = [n for n in scalar_features if n in all_fns]
            if not self.scalar_order:
                self.scalar_order = list(all_fns.keys())
        self.scalar_fns = {n: all_fns[n] for n in self.scalar_order}

        # Bounds for diff_base
        if diff_base < self.min_cycle_index:
            diff_base = self.min_cycle_index
        if diff_base > self.max_cycle_index:
            diff_base = self.max_cycle_index
        self.diff_base = int(diff_base)

    def _compute_vc_row(self, cell: BatteryData, cycle_idx: int, base_qdlin: np.ndarray) -> Optional[np.ndarray]:
        try:
            qd = get_Qdlin(cell, cell.cycle_data[cycle_idx], self.use_precalculated_qdlin)
            if self.smooth_enabled:
                qd = smooth(qd)
            if self.cycle_average is not None:
                qd = qd[..., ::self.cycle_average]
            diff = qd - base_qdlin
            if self.smooth_enabled:
                diff = smooth(diff)
            row = np.asarray(diff, dtype=float)
            if row.ndim != 1:
                row = row.reshape(-1)
            return row
        except Exception:
            return None

    def _compute_scalar_row(self, cell: BatteryData, cycle_obj) -> np.ndarray:
        vals: List[float] = []
        for name in self.scalar_order:
            fn = self.scalar_fns.get(name)
            try:
                v = fn(cell, cycle_obj)
                f = float(v) if v is not None and np.isfinite(float(v)) else 0.0
            except Exception:
                f = 0.0
            vals.append(f)
        return np.array(vals, dtype=float)

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        # Build VC base
        try:
            base_qdlin = get_Qdlin(cell_data, cell_data.cycle_data[self.diff_base], self.use_precalculated_qdlin)
        except Exception:
            if self.verbose:
                print(f"[skip] {cell_data.cell_id}: failed base Qdlin")
            return torch.zeros((1, len(self.scalar_order)), dtype=torch.float32)
        if self.smooth_enabled:
            base_qdlin = smooth(base_qdlin)
        if self.cycle_average is not None:
            base_qdlin = base_qdlin[..., ::self.cycle_average]

        rows: List[np.ndarray] = []
        start = max(0, self.min_cycle_index)
        end = min(len(cell_data.cycle_data) - 1, self.max_cycle_index)
        for idx in range(start, end + 1):
            vc = self._compute_vc_row(cell_data, idx, base_qdlin)
            scalars = self._compute_scalar_row(cell_data, cell_data.cycle_data[idx])
            if vc is None:
                # If VC row missing, use zeros of base length
                vc = np.zeros_like(base_qdlin, dtype=float)
            # Concatenate per-cycle [vc | scalars]
            row = np.concatenate([vc, scalars], axis=0)
            # Zero-fill non-finite
            row[~np.isfinite(row)] = 0.0
            rows.append(row)

        if not rows:
            return torch.zeros((1, len(base_qdlin) + len(self.scalar_order)), dtype=torch.float32)

        mat = np.stack(rows, axis=0)
        mat = mat.astype(np.float32, copy=False)
        if self.verbose:
            print(f"[ok] {cell_data.cell_id}: combined shape={mat.shape}, scalars={len(self.scalar_order)}")
        return torch.from_numpy(mat)


