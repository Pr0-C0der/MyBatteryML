#!/usr/bin/env python3
# Licensed under the MIT License.

"""
Run cycle/time plots and correlation plots separately for charge and discharge
segments of each cycle. Results are saved under:

data_analysis_split_charge_discharge/<DATASET>/
  charge/
    cycle_plots/<feature>_vs_time/*.png
    correlation/{heatmaps,matrices}
  discharge/
    cycle_plots/<feature>_vs_time/*.png
    correlation/{heatmaps,matrices}
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple, Optional
from tqdm import tqdm

# Ensure project root is on sys.path when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batteryml.data.battery_data import BatteryData, CycleData
from batteryml.label.rul import RULLabelAnnotator


def split_cycle_by_voltage_peak(cycle: CycleData) -> Tuple[Optional[CycleData], Optional[CycleData]]:
    V = cycle.voltage_in_V
    t = cycle.time_in_s
    if V is None or t is None or len(V) == 0 or len(t) == 0:
        return None, None
    V_arr = np.array(V)
    # Find the last occurrence of the maximum voltage (from the end)
    vmax = np.nanmax(V_arr)
    peak_idxs = np.where(np.isclose(V_arr, vmax, equal_nan=False))[0]
    if len(peak_idxs) == 0:
        return None, None
    peak_idx = peak_idxs[-1]

    def slice_attr(x, i0, i1):
        if x is None:
            return None
        try:
            arr = np.array(x)
            if arr.shape[0] == len(V_arr):
                return arr[i0:i1].tolist()
        except Exception:
            pass
        return x

    # Charge: [0:peak_idx+1]
    ch = CycleData(
        cycle_number=cycle.cycle_number,
        voltage_in_V=slice_attr(cycle.voltage_in_V, 0, peak_idx + 1),
        current_in_A=slice_attr(cycle.current_in_A, 0, peak_idx + 1),
        charge_capacity_in_Ah=slice_attr(cycle.charge_capacity_in_Ah, 0, peak_idx + 1),
        discharge_capacity_in_Ah=slice_attr(cycle.discharge_capacity_in_Ah, 0, peak_idx + 1),
        time_in_s=slice_attr(cycle.time_in_s, 0, peak_idx + 1),
        temperature_in_C=slice_attr(cycle.temperature_in_C, 0, peak_idx + 1),
        internal_resistance_in_ohm=cycle.internal_resistance_in_ohm,
        **cycle.additional_data
    )
    # Discharge: [peak_idx:]
    dc = CycleData(
        cycle_number=cycle.cycle_number,
        voltage_in_V=slice_attr(cycle.voltage_in_V, peak_idx, len(V_arr)),
        current_in_A=slice_attr(cycle.current_in_A, peak_idx, len(V_arr)),
        charge_capacity_in_Ah=slice_attr(cycle.charge_capacity_in_Ah, peak_idx, len(V_arr)),
        discharge_capacity_in_Ah=slice_attr(cycle.discharge_capacity_in_Ah, peak_idx, len(V_arr)),
        time_in_s=slice_attr(cycle.time_in_s, peak_idx, len(V_arr)),
        temperature_in_C=slice_attr(cycle.temperature_in_C, peak_idx, len(V_arr)),
        internal_resistance_in_ohm=cycle.internal_resistance_in_ohm,
        **cycle.additional_data
    )
    return ch, dc


def split_battery(battery: BatteryData) -> Tuple[BatteryData, BatteryData]:
    charge_cycles: List[CycleData] = []
    discharge_cycles: List[CycleData] = []
    for c in battery.cycle_data:
        ch, dc = split_cycle_by_voltage_peak(c)
        if ch is not None and ch.time_in_s is not None and len(ch.time_in_s) > 1:
            charge_cycles.append(ch)
        if dc is not None and dc.time_in_s is not None and len(dc.time_in_s) > 1:
            discharge_cycles.append(dc)

    meta = dict(
        cell_id=battery.cell_id,
        form_factor=battery.form_factor,
        anode_material=battery.anode_material,
        cathode_material=battery.cathode_material,
        electrolyte_material=battery.electrolyte_material,
        nominal_capacity_in_Ah=battery.nominal_capacity_in_Ah,
        depth_of_charge=battery.depth_of_charge,
        depth_of_discharge=battery.depth_of_discharge,
        already_spent_cycles=battery.already_spent_cycles,
        charge_protocol=battery.charge_protocol,
        discharge_protocol=battery.discharge_protocol,
        max_voltage_limit_in_V=battery.max_voltage_limit_in_V,
        min_voltage_limit_in_V=battery.min_voltage_limit_in_V,
        max_current_limit_in_A=battery.max_current_limit_in_A,
        min_current_limit_in_A=battery.min_current_limit_in_A,
        reference=battery.reference,
        description=battery.description,
    )
    charge_battery = BatteryData(cycle_data=charge_cycles, **meta)
    discharge_battery = BatteryData(cycle_data=discharge_cycles, **meta)
    return charge_battery, discharge_battery


def feature_mapping():
    return {
        'voltage': ('voltage_in_V', 'Voltage (V)'),
        'current': ('current_in_A', 'Current (A)'),
        'capacity': ('discharge_capacity_in_Ah', 'Discharge Capacity (Ah)'),
        'charge_capacity': ('charge_capacity_in_Ah', 'Charge Capacity (Ah)'),
        'temperature': ('temperature_in_C', 'Temperature (°C)'),
        'internal_resistance': ('internal_resistance_in_ohm', 'Internal Resistance (Ω)'),
    }


def select_cycles(total_cycles: int, gap: int = 100) -> List[int]:
    if total_cycles <= 1:
        return [0]
    sel = [0]
    cur = gap
    while cur < total_cycles:
        sel.append(cur)
        cur += gap
    if sel[-1] != total_cycles - 1:
        sel.append(total_cycles - 1)
    return sel


def plot_battery(battery: BatteryData, out_dir: Path, title_suffix: str, cycle_gap: int = 100):
    feats = feature_mapping()
    out_dir.mkdir(parents=True, exist_ok=True)
    cycles = battery.cycle_data
    sel = select_cycles(len(cycles), cycle_gap)
    # Plot time-series features (exclude temperature and internal resistance)
    for key, (attr_name, ylabel) in feats.items():
        if key in ['temperature', 'internal_resistance']:
            continue
        fdir = out_dir / f"{key}_vs_time"
        fdir.mkdir(exist_ok=True)
        plt.figure(figsize=(12, 8))
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(sel)))
        for i, idx in enumerate(sel):
            if idx >= len(cycles):
                continue
            c = cycles[idx]
            if hasattr(c, attr_name) and c.time_in_s is not None:
                y = getattr(c, attr_name)
                x = c.time_in_s
                try:
                    y = np.array(y)
                    x = np.array(x)
                    if y.size > 0 and x.size > 0:
                        mask = (~np.isnan(y)) & (~np.isnan(x))
                        if attr_name in ['voltage_in_V','discharge_capacity_in_Ah','charge_capacity_in_Ah']:
                            mask = mask & (y > 0)
                        if np.any(mask):
                            plt.plot(x[mask], y[mask], color=colors[i], linewidth=1.5, alpha=0.8,
                                     label=f'Cycle {c.cycle_number}')
                except Exception:
                    pass
        plt.xlabel('Time (s)'); plt.ylabel(ylabel)
        plt.title(f'{key.title()} vs Time - {title_suffix}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fdir / f"{title_suffix}_{key}_time.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot scalar features vs cycle number
    plot_scalar_feature_vs_cycle(battery, 'temperature', out_dir)
    plot_scalar_feature_vs_cycle(battery, 'internal_resistance', out_dir)


def plot_scalar_feature_vs_cycle(battery: BatteryData, feature_key: str, out_dir: Path):
    feats = feature_mapping()
    if feature_key not in feats:
        return
    attr_name, ylabel = feats[feature_key]
    xs, ys = [], []
    for c in battery.cycle_data:
        if hasattr(c, attr_name):
            val = getattr(c, attr_name)
            if val is None:
                continue
            try:
                if np.isscalar(val):
                    y = float(val)
                    if not np.isnan(y):
                        xs.append(c.cycle_number)
                        ys.append(y)
                else:
                    arr = np.array(val)
                    arr = arr[~np.isnan(arr)]
                    if arr.size > 0:
                        xs.append(c.cycle_number)
                        ys.append(float(np.mean(arr)))
            except Exception:
                continue
    if len(xs) == 0:
        return
    fdir = out_dir / f"{feature_key}_vs_cycle"
    fdir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    order = np.argsort(np.array(xs))
    xs_sorted = np.array(xs)[order]
    ys_sorted = np.array(ys)[order]
    plt.plot(xs_sorted, ys_sorted, marker='o', linewidth=1.5, alpha=0.9)
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{feature_key.replace("_"," ").title()} vs Cycle Number - {battery.cell_id}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
    plt.savefig(fdir / f"{safe_id}_{feature_key}_cycle.png", dpi=300, bbox_inches='tight')
    plt.close()


def get_total_rul(battery: BatteryData, annotator: RULLabelAnnotator) -> int:
    try:
        rul_tensor = annotator.process_cell(battery)
        val = rul_tensor.item()
        return int(val) if not np.isnan(val) else 0
    except Exception:
        return 0


def build_cycle_feature_matrix(
    battery: BatteryData,
    annotator: RULLabelAnnotator,
    base_total_rul: Optional[int] = None,
    keep_full_rul: bool = True
) -> pd.DataFrame:
    matrix_data = []
    if keep_full_rul and base_total_rul is not None:
        total_rul = base_total_rul
    else:
        total_rul = get_total_rul(battery, annotator)

    feats = {
        'voltage': 'voltage_in_V',
        'current': 'current_in_A',
        'capacity': 'discharge_capacity_in_Ah',
        'charge_capacity': 'charge_capacity_in_Ah',
        'temperature': 'temperature_in_C',
        'internal_resistance': 'internal_resistance_in_ohm',
    }
    for idx, c in enumerate(battery.cycle_data):
        cn = c.cycle_number if getattr(c, 'cycle_number', None) is not None else idx
        row = {'cycle_number': cn, 'rul': max(0, total_rul - cn)}
        for fname, attr in feats.items():
            if hasattr(c, attr):
                val = getattr(c, attr)
                if val is None:
                    row[fname] = np.nan
                else:
                    if np.isscalar(val):
                        v = float(val)
                        row[fname] = v if not np.isnan(v) else np.nan
                    else:
                        try:
                            arr = np.array(val)
                            arr = arr[~np.isnan(arr)]
                            row[fname] = float(np.mean(arr)) if arr.size > 0 else np.nan
                        except Exception:
                            row[fname] = np.nan
            else:
                row[fname] = np.nan
        matrix_data.append(row)
    return pd.DataFrame(matrix_data)


def plot_correlations(
    battery: BatteryData,
    out_dir: Path,
    title_suffix: str,
    annotator: RULLabelAnnotator,
    base_total_rul: Optional[int] = None,
    keep_full_rul: bool = True
):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_cycle_feature_matrix(
        battery,
        annotator,
        base_total_rul=base_total_rul,
        keep_full_rul=keep_full_rul
    )
    # save matrix
    matrices = out_dir / 'matrices'
    heatmaps = out_dir / 'heatmaps'
    matrices.mkdir(exist_ok=True); heatmaps.mkdir(exist_ok=True)
    safe_id = battery.cell_id.replace('/', '_').replace('\\', '_')
    df.to_csv(matrices / f"{safe_id}_{title_suffix}_cycle_feature_matrix.csv", index=False)
    # heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f',
                cbar_kws={"shrink": .8}, annot_kws={'size': 8})
    plt.title(f'Feature Correlation Matrix - {title_suffix} - {battery.cell_id}', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(heatmaps / f"{safe_id}_{title_suffix}_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def run(data_path: str, output_root: str, keep_full_rul: bool = True):
    data_dir = Path(data_path)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    annotator = RULLabelAnnotator()

    files = list(data_dir.glob('*.pkl'))
    if not files:
        print(f"No battery files found in {data_dir}")
        return

    print(f"Processing {len(files)} batteries from {data_dir}")
    for f in tqdm(files, desc='Splitting and plotting'):
        try:
            b = BatteryData.load(f)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
        full_total_rul = get_total_rul(b, annotator)
        ch_b, dc_b = split_battery(b)
        # Per-battery output dirs
        ds_charge_plot = out_root / 'charge' / 'cycle_plots'
        ds_discharge_plot = out_root / 'discharge' / 'cycle_plots'
        plot_battery(ch_b, ds_charge_plot, title_suffix=f"charge_{b.cell_id}")
        plot_battery(dc_b, ds_discharge_plot, title_suffix=f"discharge_{b.cell_id}")
        # Correlations
        plot_correlations(
            ch_b,
            out_root / 'charge' / 'correlation',
            title_suffix='charge',
            annotator=annotator,
            base_total_rul=full_total_rul,
            keep_full_rul=keep_full_rul
        )
        plot_correlations(
            dc_b,
            out_root / 'discharge' / 'correlation',
            title_suffix='discharge',
            annotator=annotator,
            base_total_rul=full_total_rul,
            keep_full_rul=keep_full_rul
        )


def run_all(base_data_path: str, base_output_dir: str = 'data_analysis_split_charge_discharge', keep_full_rul: bool = True):
    base_path = Path(base_data_path)
    datasets = ['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX']
    print(f"Running split charge/discharge analysis for all datasets in {base_path}")
    for ds in datasets:
        ds_path = base_path / ds
        if not ds_path.exists():
            print(f"Skipping {ds}: not found at {ds_path}")
            continue
        out_root = Path(base_output_dir) / ds
        print(f"\n{'='*20} {ds} {'='*20}")
        run(str(ds_path), str(out_root), keep_full_rul=keep_full_rul)


def main():
    import argparse
    p = argparse.ArgumentParser(description='Split cycles by voltage peak and run plots/correlations for charge/discharge segments.')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_path', type=str, help='Path to processed data directory (e.g., data/processed/MATR)')
    group.add_argument('--all', action='store_true', help='Process all datasets under a base path')
    p.add_argument('--base_data_path', type=str, default='data/processed', help='Base path containing dataset folders (used with --all)')
    p.add_argument('--output_dir', type=str, required=False, default=None, help='Base output directory (default: data_analysis_split_charge_discharge/<DATASET>)')
    p.add_argument('--keep_full_rul', dest='keep_full_rul', action='store_true', default=True, help='Use full-battery RUL for both charge/discharge')
    p.add_argument('--recompute_rul', dest='keep_full_rul', action='store_false', help='Recompute RUL on each segment instead')
    args = p.parse_args()

    if args.all:
        out_root = args.output_dir or 'data_analysis_split_charge_discharge'
        run_all(args.base_data_path, out_root, keep_full_rul=args.keep_full_rul)
    else:
        data_dir = Path(args.data_path)
        if not data_dir.exists():
            print(f"Data path not found: {data_dir}")
            sys.exit(1)
        dataset_name = data_dir.name
        out_root = args.output_dir or str(Path('data_analysis_split_charge_discharge') / dataset_name)
        run(str(data_dir), out_root, keep_full_rul=args.keep_full_rul)


if __name__ == '__main__':
    main()


