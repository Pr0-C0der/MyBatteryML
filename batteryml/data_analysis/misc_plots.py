# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

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


