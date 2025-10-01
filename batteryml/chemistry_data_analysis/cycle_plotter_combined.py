from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from batteryml.data.battery_data import BatteryData
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class, DatasetSpecificCycleFeatures


@dataclass
class CombinedSpec:
    feature_name: str
    method_name: str
    ylabel: str


class ChemistryCombinedPlotter:

    def __init__(self, data_path: str, output_dir: str = "chemistry_cycle_plots_combined", verbose: bool = False, dataset_hint: Optional[str] = None, ma_window: int = 0):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)
        self.dataset_hint = dataset_hint
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chem_name = self.data_path.name
        self.ma_window = int(ma_window) if int(ma_window) > 1 else 0

        # Combined pairs: (feature vs max_discharge_capacity)
        self.specs: List[CombinedSpec] = [
            CombinedSpec('charge_cycle_length', 'charge_cycle_length', 'Charge Cycle Length (s)'),
            CombinedSpec('max_charge_capacity', 'max_charge_capacity', 'Max Charge Capacity (Ah)'),
            CombinedSpec('energy_during_charge_cycle', 'power_during_charge_cycle', 'Energy during Charge (W·s)'),
            CombinedSpec('avg_charge_c_rate', 'avg_charge_c_rate', 'Avg C-rate (charge)'),
            CombinedSpec('max_charge_c_rate', 'max_charge_c_rate', 'Max C-rate (charge)'),
        ]

    @staticmethod
    def _safe_filename(name: str) -> str:
        invalid = '<>:"/\\|?*'
        s = ''.join(('_' if ch in invalid else ch) for ch in str(name))
        s = s.strip().replace(' ', '_')
        s = ''.join(ch for ch in s if ch.isprintable())
        return s or 'unknown'

    @staticmethod
    def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            kernel = np.ones(w, dtype=float)
            mask = np.isfinite(arr).astype(float)
            arr0 = np.nan_to_num(arr, nan=0.0)
            num = np.convolve(arr0, kernel, mode='same')
            den = np.convolve(mask, kernel, mode='same')
            out = num / np.maximum(den, 1e-8)
            out[den < 1e-8] = np.nan
            return out
        except Exception:
            return y

    def _infer_dataset_from_path(self, p: Path) -> Optional[str]:
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
                print(f"[warn] unable to infer dataset for {src.name}; skipping combined plots")
            return None
        cls = get_extractor_class(ds)
        if cls is None:
            if self.verbose:
                print(f"[warn] no extractor class registered for dataset '{ds}'")
            return None
        return cls()

    def _battery_files(self) -> List[Path]:
        return sorted(self.data_path.glob('*.pkl'))

    def _collect_series(self, b: BatteryData, extractor: DatasetSpecificCycleFeatures, method_name: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        fn = getattr(extractor, method_name, None)
        if fn is None:
            return None
        xs: List[float] = []
        ys: List[float] = []
        for c in b.cycle_data:
            try:
                val = fn(b, c)
            except Exception:
                val = None
            if val is None:
                xs.append(float(getattr(c, 'cycle_number', len(xs))))
                ys.append(np.nan)
                continue
            try:
                fval = float(val)
            except Exception:
                fval = np.nan
            xs.append(float(getattr(c, 'cycle_number', len(xs))))
            ys.append(fval if np.isfinite(fval) else np.nan)
        if not xs:
            return None
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        m = np.isfinite(xs_arr)
        if not np.any(m):
            return None
        order = np.argsort(xs_arr[m])
        return xs_arr[m][order], ys_arr[m][order]

    def plot_battery(self, src: Path, b: BatteryData):
        extractor = self._get_extractor(src, b)
        if extractor is None:
            return
        # Base series: max_discharge_capacity
        base = self._collect_series(b, extractor, 'max_discharge_capacity')
        if base is None:
            return
        x_base, y_base = base
        y_base_sm = self._moving_average(y_base, self._MA_WINDOW)
        for spec in self.specs:
            other = self._collect_series(b, extractor, spec.method_name)
            if other is None:
                continue
            x_other, y_other = other
            # Align on common x (cycles)
            x = np.intersect1d(x_base, x_other)
            if x.size == 0:
                continue
            idx_b = np.isin(x_base, x)
            idx_o = np.isin(x_other, x)
            xb = x_base[idx_b]
            yb = self._moving_average(y_base[idx_b], self.ma_window) if self.ma_window else y_base[idx_b]
            yo = self._moving_average(y_other[idx_o], self.ma_window) if self.ma_window else y_other[idx_o]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(xb, yo, color='blue', linewidth=1.6, alpha=0.9, label=spec.ylabel)
            ax1.set_xlabel('Cycle Number')
            ax1.set_ylabel(spec.ylabel, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(xb, yb, color='red', linewidth=1.6, alpha=0.9, label='Max Discharge Capacity (Ah)')
            ax2.set_ylabel('Max Discharge Capacity (Ah)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            fig.suptitle(f'{spec.feature_name.replace("_", " ").title()} vs Max Discharge Capacity — {b.cell_id}')

            out_dir = self.output_dir / self.chem_name / 'feature_vs_cycle_graphs_combined' / spec.feature_name
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = self._safe_filename(b.cell_id)
            try:
                plt.tight_layout()
                fig.savefig(out_dir / f'{safe}_{spec.feature_name}_vs_max_discharge_capacity.png', dpi=300, bbox_inches='tight')
            except Exception:
                pass
            finally:
                plt.close(fig)

    def run(self):
        for f in self._battery_files():
            try:
                b = BatteryData.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"[warn] load failed: {f} ({e})")
                continue
            self.plot_battery(f, b)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chemistry-aware combined feature plots against max discharge capacity')
    parser.add_argument('--data_path', type=str, required=True, help='Path to directory containing *.pkl (chemistry subfolder)')
    parser.add_argument('--output_dir', type=str, default='chemistry_cycle_plots_combined', help='Output directory')
    parser.add_argument('--dataset_hint', type=str, default=None, help='Optional dataset name hint to override auto detection')
    parser.add_argument('--ma_window', type=int, default=0, help='Moving average window for smoothing (0/1 disables, >1 enables)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    plotter = ChemistryCombinedPlotter(args.data_path, args.output_dir, verbose=args.verbose, dataset_hint=args.dataset_hint, ma_window=args.ma_window)
    plotter.run()


if __name__ == '__main__':
    main()


