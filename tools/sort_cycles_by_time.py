#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure project root is on sys.path so `batteryml` can be imported when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batteryml.data.battery_data import BatteryData, CycleData


def sort_cycle_by_time(cycle: CycleData) -> CycleData:
    """Return a new CycleData with arrays sorted by time ascending.

    Only arrays that align with time length are re-ordered. Scalars are kept.
    """
    if cycle.time_in_s is None:
        return cycle

    time = np.array(cycle.time_in_s)
    if time.size == 0 or np.all(np.isnan(time)):
        return cycle

    order = np.argsort(time)

    def reorder(x):
        if x is None:
            return None
        # Scalars unchanged
        if np.isscalar(x):
            return x
        arr = np.array(x)
        # Only reorder if shapes match time
        try:
            if arr.shape[0] == time.shape[0]:
                return arr[order].tolist()
        except Exception:
            pass
        return x

    return CycleData(
        cycle_number=cycle.cycle_number,
        voltage_in_V=reorder(cycle.voltage_in_V),
        current_in_A=reorder(cycle.current_in_A),
        charge_capacity_in_Ah=reorder(cycle.charge_capacity_in_Ah),
        discharge_capacity_in_Ah=reorder(cycle.discharge_capacity_in_Ah),
        time_in_s=reorder(cycle.time_in_s),
        temperature_in_C=reorder(cycle.temperature_in_C),
        internal_resistance_in_ohm=cycle.internal_resistance_in_ohm,
        **cycle.additional_data
    )


def process_file(src_path: Path, dst_path: Path):
    battery = BatteryData.load(src_path)
    sorted_cycles: List[CycleData] = []
    for c in battery.cycle_data:
        sorted_cycles.append(sort_cycle_by_time(c))

    cleaned = BatteryData(
        cell_id=battery.cell_id,
        cycle_data=sorted_cycles,
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

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.dump(dst_path)


def main():
    base_src = Path('data/processed')
    base_dst = Path('data/preprocessed')

    datasets = [p for p in base_src.iterdir() if p.is_dir()]
    for ds in datasets:
        dst_dir = base_dst / ds.name
        for pkl in ds.glob('*.pkl'):
            out = dst_dir / pkl.name
            process_file(pkl, out)


if __name__ == '__main__':
    main()


