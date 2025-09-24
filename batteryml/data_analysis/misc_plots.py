# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

from batteryml.data.battery_data import BatteryData


def plot_voltage_current_twin(battery: BatteryData, cycle_index: int, save_path: Path) -> bool:
    """Plot voltage (left y-axis) and current (right y-axis) vs relative time for a single cycle.

    Returns True if a plot was generated and saved, False otherwise.
    """
    if cycle_index < 0 or cycle_index >= len(battery.cycle_data):
        return False

    c = battery.cycle_data[cycle_index]
    if c.voltage_in_V is None or c.current_in_A is None or c.time_in_s is None:
        return False

    try:
        V = np.array(c.voltage_in_V)
        I = np.array(c.current_in_A)
        t = np.array(c.time_in_s)
        n = min(len(V), len(I), len(t))
        if n == 0:
            return False
        V, I, t = V[:n], I[:n], t[:n]
        mask = (~np.isnan(V)) & (~np.isnan(I)) & (~np.isnan(t))
        if not np.any(mask):
            return False
        V, I, t = V[mask], I[mask], t[mask]
        tr = t - t[0]

        fig, ax1 = plt.subplots(figsize=(12, 6))
        color_v = 'tab:blue'
        color_i = 'tab:red'
        ax1.set_xlabel('Relative Time (s)')
        ax1.set_ylabel('Voltage (V)', color=color_v)
        ax1.plot(tr, V, color=color_v, linewidth=1.6)
        ax1.tick_params(axis='y', labelcolor=color_v)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Current (A)', color=color_i)
        ax2.plot(tr, I, color=color_i, linewidth=1.2, alpha=0.9)
        ax2.tick_params(axis='y', labelcolor=color_i)

        plt.title(f'Voltage & Current vs Relative Time - {battery.cell_id} (Cycle {c.cycle_number})', fontsize=13)
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception:
        return False


def plot_first_last_overlay(
    battery: BatteryData,
    feature: str,
    attr: str,
    ylabel: str,
    save_path: Path
) -> bool:
    """Overlay first (blue) and last (red) cycle for a single feature vs relative time.

    Returns True if a plot was generated and saved, False otherwise.
    """
    if len(battery.cycle_data) == 0:
        return False
    idx_first = 0
    idx_last = len(battery.cycle_data) - 1

    def get_xy(c) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        vals = getattr(c, attr, None)
        t = getattr(c, 'time_in_s', None)
        if vals is None or t is None:
            return None, None
        try:
            y = np.array(vals)
            x = np.array(t)
            n = min(len(y), len(x))
            if n == 0:
                return None, None
            y, x = y[:n], x[:n]
            m = (~np.isnan(y)) & (~np.isnan(x))
            if not np.any(m):
                return None, None
            return x[m] - x[m][0], y[m]
        except Exception:
            return None, None

    c_first = battery.cycle_data[idx_first]
    c_last = battery.cycle_data[idx_last]
    x1, y1 = get_xy(c_first)
    x2, y2 = get_xy(c_last)
    if x1 is None or x2 is None:
        return False

    plt.figure(figsize=(12, 6))
    plt.plot(x1, y1, color='tab:blue', linewidth=1.6, label=f'Cycle {c_first.cycle_number}')
    plt.plot(x2, y2, color='tab:red', linewidth=1.6, label=f'Cycle {c_last.cycle_number}')
    plt.xlabel('Relative Time (s)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{feature.title()} (First vs Last Cycle) - {battery.cell_id}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


# -----------------
# Command-line entry
# -----------------
def _parse_twin_cycles(arg: str) -> List[str]:
    # Allow values like "0,last" or "10" etc. We'll resolve 'last' per battery later
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    return parts if parts else ['0', 'last']


def _safe_cell_id(cell_id: str) -> str:
    return cell_id.replace('/', '_').replace('\\', '_')


def _process_file(file_path: Path,
                  out_dir: Path,
                  do_twin: bool,
                  twin_cycles_spec: List[str],
                  do_first_last: bool,
                  include_voltage: bool,
                  include_current: bool,
                  verbose: bool) -> None:
    try:
        b = BatteryData.load(file_path)
    except Exception as e:
        if verbose:
            print(f"[warn] failed to load {file_path}: {e}")
        return

    safe_id = _safe_cell_id(b.cell_id)

    if do_twin:
        twin_dir = out_dir / 'twin_plots'
        twin_dir.mkdir(parents=True, exist_ok=True)
        # Resolve cycles per battery (handle 'last')
        for spec in twin_cycles_spec:
            if spec.lower() == 'last':
                idx = max(0, len(b.cycle_data) - 1)
            else:
                try:
                    idx = int(spec)
                except ValueError:
                    if verbose:
                        print(f"[skip] invalid cycle index '{spec}' for {safe_id}")
                    continue
            save_path = twin_dir / f"{safe_id}_cycle{idx}_v_i.png"
            ok = plot_voltage_current_twin(b, idx, save_path)
            if verbose:
                print(f"[{'ok' if ok else 'skip'}] twin {safe_id} cycle {idx} -> {save_path}")

    if do_first_last:
        fl_dir = out_dir / 'first_last_overlays'
        fl_dir.mkdir(parents=True, exist_ok=True)
        if include_voltage:
            save_v = fl_dir / f"{safe_id}_first_last_voltage.png"
            ok_v = plot_first_last_overlay(b, 'voltage', 'voltage_in_V', 'Voltage (V)', save_v)
            if verbose:
                print(f"[{'ok' if ok_v else 'skip'}] first_last voltage {safe_id} -> {save_v}")
        if include_current:
            save_i = fl_dir / f"{safe_id}_first_last_current.png"
            ok_i = plot_first_last_overlay(b, 'current', 'current_in_A', 'Current (A)', save_i)
            if verbose:
                print(f"[{'ok' if ok_i else 'skip'}] first_last current {safe_id} -> {save_i}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Miscellaneous battery plots (twin and first/last overlays)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to a preprocessed *.pkl file or a directory containing *.pkl files')
    parser.add_argument('--output_dir', type=str, default='misc_plots_out', help='Output directory')
    parser.add_argument('--plot', type=str, choices=['twin', 'first_last', 'all'], default='all',
                        help='Which plots to generate')
    parser.add_argument('--twin_cycles', type=str, default='0,last',
                        help="Comma-separated cycle indices for twin plot, use 'last' to include the final cycle (default: 0,last)")
    parser.add_argument('--first_last_features', type=str, nargs='*', choices=['voltage', 'current'], default=['voltage', 'current'],
                        help='Which features to include in first/last overlays (default: voltage current)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_twin = args.plot in ('twin', 'all')
    do_first_last = args.plot in ('first_last', 'all')
    twin_cycles_spec = _parse_twin_cycles(args.twin_cycles)
    include_voltage = 'voltage' in args.first_last_features
    include_current = 'current' in args.first_last_features

    files: List[Path]
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = sorted(data_path.glob('*.pkl'))
    else:
        print(f"[error] data_path does not exist: {data_path}")
        return

    if args.verbose:
        print(f"Found {len(files)} file(s) to process under {data_path}")

    for f in files:
        _process_file(
            f,
            out_dir,
            do_twin,
            twin_cycles_spec,
            do_first_last,
            include_voltage,
            include_current,
            args.verbose,
        )


if __name__ == '__main__':
    main()

