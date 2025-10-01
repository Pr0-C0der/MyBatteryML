from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from batteryml.data.battery_data import BatteryData
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class, DatasetSpecificCycleFeatures


def _safe_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    s = ''.join(('_' if ch in invalid else ch) for ch in str(name))
    s = s.strip().replace(' ', '_')
    s = ''.join(ch for ch in s if ch.isprintable())
    return s or 'unknown'


SeriesComputeFn = Callable[..., Tuple[np.ndarray, np.ndarray]]
# Plot function can optionally use the extractor for dataset-specific logic
PlotFn = Callable[[BatteryData, Dict[str, Tuple[np.ndarray, np.ndarray]], Path, Optional[DatasetSpecificCycleFeatures]], None]


@dataclass
class SeriesSpec:
    name: str
    compute: SeriesComputeFn
    depends_on: List[str] = field(default_factory=list)
    description: Optional[str] = None
    requires_extractor: bool = False


@dataclass
class PlotSpec:
    name: str
    uses_series: List[str]
    plot: PlotFn
    subdir: Optional[str] = None
    description: Optional[str] = None


class StatsPlotter:
    def __init__(self, data_path: str, output_dir: str = 'chemistry_stat_features', verbose: bool = False, dataset_hint: Optional[str] = None):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.series_specs: Dict[str, SeriesSpec] = {}
        self.plot_specs: List[PlotSpec] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chem_name = self.data_path.name
        self.out_root = self.output_dir / self.chem_name
        self.out_root.mkdir(parents=True, exist_ok=True)

    # Registration
    def register_series(self, spec: SeriesSpec):
        self.series_specs[spec.name] = spec

    def register_plot(self, spec: PlotSpec):
        self.plot_specs.append(spec)

    # Utilities
    def _battery_files(self) -> List[Path]:
        return sorted(self.data_path.glob('*.pkl'))

    # Dataset inference (like other chemistry modules)
    @staticmethod
    def _infer_dataset_from_path(p: Path) -> Optional[str]:
        name = p.as_posix().upper()
        candidates = ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'HUST', 'OX']
        for key in candidates:
            if f"/{key}/" in name or name.endswith(f"/{key}.PKL") or (f"_{key}_" in name) or (f"-{key}-" in name) or (f"/{key}_" in name):
                return key
        return None

    def _infer_dataset(self, src: Path, b: BatteryData) -> Optional[str]:
        if self.dataset_hint:
            return str(self.dataset_hint).upper()
        key = self._infer_dataset_from_path(src)
        if key:
            return key
        cid = str(getattr(b, 'cell_id', '')).upper()
        for key in ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'HUST', 'OX']:
            if key in cid:
                return key
        for attr in ['reference', 'description']:
            txt = str(getattr(b, attr, '')).upper()
            for key in ['MATR', 'MATR1', 'MATR2', 'CALCE', 'SNL', 'RWTH', 'HNEI', 'UL_PUR', 'HUST', 'OX']:
                if key in txt:
                    return key
        return None

    def _get_extractor(self, src: Path, b: BatteryData) -> Optional[DatasetSpecificCycleFeatures]:
        ds = self._infer_dataset(src, b)
        if ds is None:
            if self.verbose:
                print(f"[warn] unable to infer dataset for {src.name}; skipping extractor-dependent series")
            return None
        cls = get_extractor_class(ds)
        if cls is None:
            if self.verbose:
                print(f"[warn] no extractor class registered for dataset '{ds}'")
            return None
        return cls()

    @staticmethod
    def _save_png(fig_path: Path):
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
        finally:
            plt.close()

    # Pipeline
    def _compute_needed_series(self, b: BatteryData, required: List[str], extractor: Optional[DatasetSpecificCycleFeatures]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in required:
            if name not in self.series_specs:
                continue
            try:
                spec = self.series_specs[name]
                if spec.requires_extractor:
                    if extractor is None:
                        continue
                    xs, ys = spec.compute(b, extractor)
                else:
                    xs, ys = spec.compute(b)
                results[name] = (xs, ys)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] {b.cell_id}: series '{name}' failed: {e}")
        return results

    def plot_battery(self, b: BatteryData, src: Path):
        extractor = self._get_extractor(src, b)
        for spec in self.plot_specs:
            series_map = self._compute_needed_series(b, spec.uses_series, extractor)
            try:
                out_dir = self.out_root / (spec.subdir or spec.name)
                spec.plot(b, series_map, out_dir, extractor)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] {b.cell_id}: plot '{spec.name}' failed: {e}")

    def run(self):
        for f in self._battery_files():
            try:
                b = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            self.plot_battery(b, f)


# -----------------
# Default series
# -----------------
def series_discharge_capacity_final(b: BatteryData) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for c in b.cycle_data:
        xs.append(float(c.cycle_number))
        arr = getattr(c, 'discharge_capacity_in_Ah', None)
        if arr is None:
            ys.append(np.nan)
            continue
        arr = np.asarray(arr, dtype=float)
        m = np.isfinite(arr)
        ys.append(float(arr[m][-1]) if np.any(m) else np.nan)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def series_coulombic_efficiency(b: BatteryData, extractor: DatasetSpecificCycleFeatures) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ce: List[float] = []
    for c in b.cycle_data:
        xs.append(float(c.cycle_number))
        # Charge window
        cs, ce_idx = extractor.charge_window_indices(b, c)
        Qc = getattr(c, 'charge_capacity_in_Ah', None)
        if Qc is None:
            ce.append(np.nan)
            continue
        Qc_arr = np.asarray(Qc, dtype=float)
        nQ = Qc_arr.size
        cs = max(0, min(cs, max(0, nQ - 1)))
        ce_i = max(0, min(ce_idx, max(0, nQ - 1)))
        Qc_seg = Qc_arr[cs:ce_i + 1] if nQ else np.array([], dtype=float)
        Qc_seg = Qc_seg[np.isfinite(Qc_seg)] if Qc_seg.size else Qc_seg
        Qc_final = float(Qc_seg[-1]) if Qc_seg.size else np.nan
        # Discharge window
        ds, de = extractor.discharge_window_indices(b, c)
        Qd = getattr(c, 'discharge_capacity_in_Ah', None)
        if Qd is None:
            ce.append(np.nan)
            continue
        Qd_arr = np.asarray(Qd, dtype=float)
        nD = Qd_arr.size
        ds = max(0, min(ds, max(0, nD - 1)))
        de_i = max(0, min(de, max(0, nD - 1)))
        Qd_seg = Qd_arr[ds:de_i + 1] if nD else np.array([], dtype=float)
        Qd_seg = Qd_seg[np.isfinite(Qd_seg)] if Qd_seg.size else Qd_seg
        Qd_final = float(Qd_seg[-1]) if Qd_seg.size else np.nan
        # CE
        if np.isfinite(Qc_final) and np.isfinite(Qd_final) and Qc_final > 0:
            ce.append(float(Qd_final / Qc_final))
        else:
            ce.append(np.nan)
    return np.asarray(xs, dtype=float), np.asarray(ce, dtype=float)


# -----------------
# dQ/dV utilities
# -----------------
def _compute_dqdv_on_vgrid(V_raw: np.ndarray, Q_raw: np.ndarray, V_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # sort by V
    idx = np.argsort(V_raw)
    V = np.asarray(V_raw, float)[idx]
    Q = np.asarray(Q_raw, float)[idx]
    # interpolate Q(V)
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    f = interp1d(V, Q, kind='linear', bounds_error=False, fill_value=np.nan)
    Qg = f(V_grid)
    m = np.isfinite(Qg)
    if not np.any(m):
        return V_grid, np.full_like(V_grid, np.nan)
    # fill internal NaNs by linear interp on V_grid
    if np.any(~m):
        Qg[~m] = np.interp(V_grid[~m], V_grid[m], Qg[m])
    # smooth and differentiate
    wl = min(len(V_grid) - (1 - len(V_grid) % 2), 51)  # odd, <= len
    wl = max(5, wl if wl % 2 == 1 else wl - 1)
    Qg_s = savgol_filter(Qg, window_length=wl, polyorder=3, mode='interp')
    dQdV = savgol_filter(Qg_s, window_length=wl, polyorder=3, deriv=1,
                         delta=np.mean(np.diff(V_grid)), mode='interp')
    return V_grid, dQdV


def _series_dqdv_for_cycle(b: BatteryData, extractor: DatasetSpecificCycleFeatures, c, mode: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # mode: 'charge' or 'discharge'
    V_arr = getattr(c, 'voltage_in_V', None)
    Qc_arr = getattr(c, 'charge_capacity_in_Ah', None)
    Qd_arr = getattr(c, 'discharge_capacity_in_Ah', None)
    if V_arr is None or (Qc_arr is None and Qd_arr is None):
        return None
    V = np.asarray(V_arr, float)
    if mode == 'charge' and Qc_arr is not None:
        cs, ce = extractor.charge_window_indices(b, c)
        Q = np.asarray(Qc_arr, float)
        n = min(V.size, Q.size)
        if n < 5:
            return None
        cs = max(0, min(cs, n - 1)); ce = max(0, min(ce, n - 1))
        if ce <= cs:
            return None
        Vw = V[cs:ce + 1]
        Qw = Q[cs:ce + 1]
    elif mode == 'discharge' and Qd_arr is not None:
        ds, de = extractor.discharge_window_indices(b, c)
        Q = np.asarray(Qd_arr, float)
        n = min(V.size, Q.size)
        if n < 5:
            return None
        ds = max(0, min(ds, n - 1)); de = max(0, min(de, n - 1))
        if de <= ds:
            return None
        Vw = V[ds:de + 1]
        Qw = Q[ds:de + 1]
    else:
        return None
    # Build common V grid for this cycle
    vmin = float(np.nanmax([np.nanmin(Vw), np.nanmin(Vw)]))
    vmax = float(np.nanmin([np.nanmax(Vw), np.nanmax(Vw)]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None
    V_grid = np.linspace(vmin, vmax, 600)
    return _compute_dqdv_on_vgrid(Vw, Qw, V_grid)


def plot_dqdv_curves_with_peaks(b: BatteryData, series: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path, extractor: Optional[DatasetSpecificCycleFeatures]):
    if extractor is None:
        return
    from scipy.signal import find_peaks, peak_widths
    # Plot cycles at a fixed GAP (hardcoded)
    GAP = 500
    total = len(b.cycle_data)
    selected_idxs = list(range(0, total, GAP)) if total > 0 else []
    if not selected_idxs and total > 0:
        selected_idxs = [0]
    safe_id = _safe_filename(b.cell_id)

    # Discharge dQ/dV
    plt.figure(figsize=(10, 6))
    for idx in selected_idxs:
        c = b.cycle_data[idx]
        res = _series_dqdv_for_cycle(b, extractor, c, mode='discharge')
        if res is None:
            continue
        Vg, dQdV = res
        plt.plot(Vg, dQdV, alpha=0.6)
    plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah/V)')
    plt.title(f'dQ/dV Discharge (gap=500) — {b.cell_id}')
    plt.grid(True, alpha=0.3)
    StatsPlotter._save_png(out_dir / 'dqdv_discharge' / f'{safe_id}_dqdv_discharge.png')

    # Peaks on the first discharge curve (if available)
    if selected_idxs:
        c = b.cycle_data[selected_idxs[0]]
        res = _series_dqdv_for_cycle(b, extractor, c, mode="discharge")
        if res is not None:
            Vg, dQdV = res
            m = np.isfinite(dQdV)
            if np.any(m):
                y = dQdV[m]; x = Vg[m]
                peaks, props = find_peaks(y, prominence=np.nanmax(np.abs(y)) * 0.05 if np.any(np.isfinite(y)) else 0.05)
                _ = peak_widths(y, peaks, rel_height=0.5)
                plt.figure(figsize=(10, 6))
                plt.plot(x, y)
                if peaks.size:
                    plt.plot(x[peaks], y[peaks], 'ro')
                plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah/V)')
                plt.title(f'dQ/dV Discharge Peaks — {b.cell_id}')
                plt.grid(True, alpha=0.3)
                StatsPlotter._save_png(out_dir / 'dqdv_discharge' / f'{safe_id}_dqdv_discharge_peaks.png')

    # Charge dQ/dV
    plt.figure(figsize=(10, 6))
    for idx in selected_idxs:
        c = b.cycle_data[idx]
        res = _series_dqdv_for_cycle(b, extractor, c, mode='charge')
        if res is None:
            continue
        Vg, dQdV = res
        plt.plot(Vg, dQdV, alpha=0.6)
    plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah/V)')
    plt.title(f'dQ/dV Charge (gap=500) — {b.cell_id}')
    plt.grid(True, alpha=0.3)
    StatsPlotter._save_png(out_dir / 'dqdv_charge' / f'{safe_id}_dqdv_charge.png')

    # Peaks on the first charge curve (if available)
    if selected_idxs:
        c = b.cycle_data[selected_idxs[0]]
        res = _series_dqdv_for_cycle(b, extractor, c, mode="charge")
        if res is not None:
            Vg, dQdV = res
            m = np.isfinite(dQdV)
            if np.any(m):
                y = dQdV[m]; x = Vg[m]
                peaks, props = find_peaks(y, prominence=np.nanmax(np.abs(y)) * 0.05 if np.any(np.isfinite(y)) else 0.05)
                _ = peak_widths(y, peaks, rel_height=0.5)
                plt.figure(figsize=(10, 6))
                plt.plot(x, y)
                if peaks.size:
                    plt.plot(x[peaks], y[peaks], 'ro')
                plt.xlabel('Voltage (V)'); plt.ylabel('dQ/dV (Ah/V)')
                plt.title(f'dQ/dV Charge Peaks — {b.cell_id}')
                plt.grid(True, alpha=0.3)
                StatsPlotter._save_png(out_dir / 'dqdv_charge' / f'{safe_id}_dqdv_charge_peaks.png')


# -----------------
# Default plots
# -----------------
def plot_relative_capacity_fade_abs(b: BatteryData, series: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path):
    xs, cap = series['discharge_capacity_final']
    if xs.size == 0:
        return
    if cap.size == 0 or not np.isfinite(cap[0]):
        return
    rel_abs = np.abs(cap - cap[0])
    plt.figure(figsize=(10, 6))
    plt.plot(xs, rel_abs, marker='o', linewidth=1.6, alpha=0.9)
    plt.xlabel('Cycle Number')
    plt.ylabel('|cap(n) - cap(0)| (Ah)')
    plt.title(f'Relative Capacity Fade (abs) — {b.cell_id}')
    plt.grid(True, alpha=0.3)
    fname = _safe_filename(b.cell_id) + '_rel_capacity_abs.png'
    StatsPlotter._save_png(out_dir / fname)


def plot_capacity_retention_ratio(battery: BatteryData, series: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path, extractor: Optional[DatasetSpecificCycleFeatures] = None):
    xs, cap = series['discharge_capacity_final']
    if xs.size == 0:
        return
    if cap.size == 0 or not np.isfinite(cap[0]) or cap[0] == 0:
        return
    ratio = cap / cap[0]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ratio, marker='o', linewidth=1.6, alpha=0.9)
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity Retention Ratio = cap(n)/cap(0)')
    plt.title(f'Capacity Retention Ratio — {battery.cell_id}')
    plt.grid(True, alpha=0.3)
    fname = _safe_filename(battery.cell_id) + '_capacity_retention.png'
    StatsPlotter._save_png(out_dir / fname)


def _fit_first_500(xs: np.ndarray, ys: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float, float]], np.ndarray]:
    m = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[m]
    ys = ys[m]
    if xs.size < 2:
        return None, None, xs
    order = np.argsort(xs)
    xfit = xs[order][:500]
    yfit = ys[order][:500]
    if xfit.size < 2:
        return None, None, xfit
    p1 = None
    p2 = None
    try:
        p = np.polyfit(xfit, yfit, deg=1)
        p1 = (float(p[0]), float(p[1]))
    except Exception:
        pass
    try:
        p = np.polyfit(xfit, yfit, deg=2)
        p2 = (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        pass
    return p1, p2, xfit


def plot_discharge_capacity_with_fits(b: BatteryData, series: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path, extractor: Optional[DatasetSpecificCycleFeatures] = None):
    xs, cap = series['discharge_capacity_final']
    if xs.size == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(xs, cap, marker='o', linewidth=1.2, alpha=0.9, label='Discharge capacity (final)')
    p1, p2, xfit = _fit_first_500(xs, cap)
    if p1 is not None and xfit.size > 0:
        m, c = p1
        yline = m * xfit + c
        plt.plot(xfit, yline, color='green', linewidth=2.0, label=f'Linear fit (first 500) slope={m:.4g}')
    if p2 is not None and xfit.size > 0:
        a, b, c2 = p2
        yquad = a * xfit**2 + b * xfit + c2
        plt.plot(xfit, yquad, color='purple', linewidth=2.0, alpha=0.8, label=f'Quadratic fit (a={a:.3g})')
    plt.xlabel('Cycle Number')
    plt.ylabel('Discharge Capacity (Ah)')
    plt.title(f'Discharge Capacity vs Cycle — {b.cell_id}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    fname = _safe_filename(b.cell_id) + '_discharge_capacity.png'
    StatsPlotter._save_png(out_dir / fname)


def plot_coulombic_efficiency(b: BatteryData, series: Dict[str, Tuple[np.ndarray, np.ndarray]], out_dir: Path, extractor: Optional[DatasetSpecificCycleFeatures] = None):
    xs, vals = series['coulombic_efficiency']
    if xs.size == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(xs, vals, marker='o', linewidth=1.6, alpha=0.9)
    plt.xlabel('Cycle Number')
    plt.ylabel('Coulombic Efficiency (Qd_final / Qc_final)')
    plt.title(f'Coulombic Efficiency vs Cycle — {b.cell_id}')
    plt.grid(True, alpha=0.3)
    fname = _safe_filename(b.cell_id) + '_coulombic_efficiency.png'
    StatsPlotter._save_png(out_dir / fname)


def build_default_plotter(data_path: str, output_dir: str = 'chemistry_stat_features', verbose: bool = False, dataset_hint: Optional[str] = None) -> StatsPlotter:
    sp = StatsPlotter(data_path, output_dir, verbose=verbose, dataset_hint=dataset_hint)
    # Series
    sp.register_series(SeriesSpec('discharge_capacity_final', series_discharge_capacity_final, description='Final discharge capacity per cycle'))
    sp.register_series(SeriesSpec('coulombic_efficiency', series_coulombic_efficiency, description='Qd_final/Qc_final per cycle', requires_extractor=True))
    # Plots
    sp.register_plot(PlotSpec('relative_capacity_fade_abs', ['discharge_capacity_final'], plot_relative_capacity_fade_abs, subdir='relative_capacity_fade_abs'))
    sp.register_plot(PlotSpec('capacity_retention_ratio', ['discharge_capacity_final'], plot_capacity_retention_ratio, subdir='capacity_retention_ratio'))
    sp.register_plot(PlotSpec('discharge_capacity_with_fits', ['discharge_capacity_final'], plot_discharge_capacity_with_fits, subdir='discharge_capacity'))
    sp.register_plot(PlotSpec('coulombic_efficiency', ['coulombic_efficiency'], plot_coulombic_efficiency, subdir='coulombic_efficiency'))
    sp.register_plot(PlotSpec('dqdv_curves_with_peaks', [], plot_dqdv_curves_with_peaks, subdir='dqdv'))
    return sp


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Modular statistical capacity/cycle features per chemistry')
    parser.add_argument('--data_path', type=str, required=True, help='Path to chemistry folder containing *.pkl')
    parser.add_argument('--output_dir', type=str, default='chemistry_stat_features', help='Output directory for plots')
    parser.add_argument('--dataset_hint', type=str, default=None, help='Optional dataset name hint to override auto detection')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    plotter = build_default_plotter(args.data_path, args.output_dir, verbose=args.verbose, dataset_hint=args.dataset_hint)
    plotter.run()


if __name__ == '__main__':
    main()


