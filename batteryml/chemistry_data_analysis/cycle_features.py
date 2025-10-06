from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type, Optional

import numpy as np
import pandas as pd

from batteryml.data.battery_data import BatteryData


# ---------------------------- Helpers ----------------------------

def _to_array(x) -> np.ndarray:
    try:
        arr = np.asarray(x)
    except Exception:
        return np.array([], dtype=float)
    return arr if arr.ndim == 1 else arr.reshape(-1)


def _first_min_index(arr: np.ndarray, start: int = 0) -> Optional[int]:
    n = int(arr.size)
    if start >= n:
        return None
    sub = arr[start:]
    finite = np.isfinite(sub)
    if not np.any(finite):
        return None
    vals = sub.copy()
    vals[~finite] = np.inf
    idx_local = int(np.argmin(vals))
    return start + idx_local


def _first_max_index(arr: np.ndarray, start: int = 0) -> Optional[int]:
    n = int(arr.size)
    if start >= n:
        return None
    sub = arr[start:]
    finite = np.isfinite(sub)
    if not np.any(finite):
        return None
    vals = sub.copy()
    vals[~finite] = -np.inf
    idx_local = int(np.argmax(vals))
    return start + idx_local


def _first_zero_index(arr: np.ndarray, start: int = 0, atol: float = 1e-3) -> Optional[int]:
    n = int(arr.size)
    if start >= n:
        return None
    for i in range(start, n):
        v = arr[i]
        if np.isfinite(v) and (abs(float(v)) <= atol or np.isclose(float(v), 0.0, atol=atol)):
            return i
    return None


class BaseCycleFeatures(ABC):
    """
    Base class declaring the six common, dataset-agnostic feature signatures.

    Implementations of these six should be shared across datasets. For now,
    only method signatures and docstrings are provided; logic will be added later.
    """

    # ---------- Common, dataset-agnostic features ----------
    def avg_voltage(self, battery: BatteryData, cycle) -> Optional[float]:
        """Return average voltage over the entire cycle."""
        V = getattr(cycle, 'voltage_in_V', None)
        if V is None:
            return None
        try:
            arr = np.array(V)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            v = float(np.mean(arr))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def avg_current(self, battery: BatteryData, cycle) -> Optional[float]:
        """Return average current over the entire cycle."""
        I = getattr(cycle, 'current_in_A', None)
        if I is None:
            return None
        try:
            arr = np.array(I)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            v = float(np.mean(arr))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def avg_c_rate(self, battery: BatteryData, cycle) -> Optional[float]:
        """Return average C-rate over the entire cycle."""
        I = getattr(cycle, 'current_in_A', None)
        C = battery.nominal_capacity_in_Ah or 0.0
        if I is None or not C:
            return None
        I = np.array(I)
        I = I[np.isfinite(I)]
        if I.size == 0:
            return None
        return float(np.mean(np.abs(I)) / C)

    def cycle_length(self, battery: BatteryData, cycle) -> Optional[float]:
        """Return total cycle duration (e.g., in seconds)."""
        t = getattr(cycle, 'time_in_s', None)
        if t is None:
            return None
        t_arr = np.array(t)
        t_arr = t_arr[np.isfinite(t_arr)]
        if t_arr.size == 0:
            return None
        v = float(t_arr[-1] - t_arr[0])
        return v if np.isfinite(v) and v >= 0 else None

    def max_charge_capacity(self, battery: BatteryData, cycle) -> Optional[float]:
        """Return maximum charge capacity observed during the cycle."""
        Qc = getattr(cycle, 'charge_capacity_in_Ah', None)
        if Qc is None:
            return None
        Qc = np.array(Qc)
        Qc = Qc[np.isfinite(Qc)]
        if Qc.size == 0:
            return None
        m = np.nanmax(Qc)
        return float(m) if np.isfinite(m) else None

    def max_discharge_capacity(self, battery: BatteryData, cycle) -> Optional[float]:
        """Return maximum discharge capacity observed during the cycle."""
        Qd = getattr(cycle, 'discharge_capacity_in_Ah', None)
        if Qd is None:
            return None
        Qd = np.array(Qd)
        Qd = Qd[np.isfinite(Qd)]
        if Qd.size == 0:
            return None
        m = np.nanmax(Qd)
        return float(m) if np.isfinite(m) else None


class DatasetSpecificCycleFeatures(BaseCycleFeatures, ABC):
    """
    Abstract class for dataset-specific feature extraction.

    Datasets differ in how charging and discharging are segmented, so the
    window index extraction must be provided by concrete subclasses.
    Dependent features (listed below) are computed using those windows.
    """

    # ---------- Dataset-specific windowing contracts ----------
    @abstractmethod
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        """Return (start_index, end_index) of the charge window within the cycle."""
        raise NotImplementedError

    @abstractmethod
    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        """Return (start_index, end_index) of the discharge window within the cycle."""
        raise NotImplementedError

    # ---------- Features dependent on charge/discharge windowing ----------
    def charge_cycle_length(self, battery: BatteryData, cycle) -> Optional[float]:
        """Duration of the charge segment based on dataset-specific window indices."""
        t = _to_array(getattr(cycle, 'time_in_s', []))
        if t.size == 0:
            return None
        n = int(t.size)
        cs, ce = self.charge_window_indices(battery, cycle)
        cs = max(0, min(cs, n - 1)); ce = max(0, min(ce, n - 1))
        if ce < cs:
            return None
        dt = float(t[ce] - t[cs])
        return dt if np.isfinite(dt) and dt >= 0 else None

    def discharge_cycle_length(self, battery: BatteryData, cycle) -> Optional[float]:
        """Duration of the discharge segment based on dataset-specific window indices."""
        t = _to_array(getattr(cycle, 'time_in_s', []))
        if t.size == 0:
            return None
        n = int(t.size)
        ds, de = self.discharge_window_indices(battery, cycle)
        ds = max(0, min(ds, n - 1)); de = max(0, min(de, n - 1))
        if de < ds:
            return None
        dt = float(t[de] - t[ds])
        return dt if np.isfinite(dt) and dt >= 0 else None
    def avg_charge_c_rate(self, battery: BatteryData, cycle) -> Optional[float]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        t = _to_array(getattr(cycle, 'time_in_s', []))
        C_nom = battery.nominal_capacity_in_Ah or 0.0
        if I.size == 0 or t.size == 0 or not C_nom:
            return None
        n = int(min(I.size, t.size))
        I = I[:n]; t = t[:n]
        cs, ce = self.charge_window_indices(battery, cycle)
        cs = max(0, min(cs, n - 1)); ce = max(0, min(ce, n - 1))
        if ce <= cs:
            return None
        Iw = I[cs:ce + 1]
        tw = t[cs:ce + 1]
        mfin = np.isfinite(Iw) & np.isfinite(tw)
        if np.count_nonzero(mfin) < 2:
            return None
        Iw = Iw[mfin]; tw = tw[mfin]
        q_as = float(np.trapz(Iw, tw))
        val = q_as / (C_nom * 3600.0)
        return float(val) if np.isfinite(val) else None

    def avg_discharge_c_rate(self, battery: BatteryData, cycle) -> Optional[float]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        t = _to_array(getattr(cycle, 'time_in_s', []))
        C_nom = battery.nominal_capacity_in_Ah or 0.0
        if I.size == 0 or t.size == 0 or not C_nom:
            return None
        n = int(min(I.size, t.size))
        I = I[:n]; t = t[:n]
        ds, de = self.discharge_window_indices(battery, cycle)
        ds = max(0, min(ds, n - 1)); de = max(0, min(de, n - 1))
        if de <= ds:
            return None
        Iw = np.abs(I[ds:de + 1])
        tw = t[ds:de + 1]
        mfin = np.isfinite(Iw) & np.isfinite(tw)
        if np.count_nonzero(mfin) < 2:
            return None
        Iw = Iw[mfin]; tw = tw[mfin]
        q_as = float(np.trapz(Iw, tw))
        val = q_as / (C_nom * 3600.0)
        return float(val) if np.isfinite(val) else None

    def max_charge_c_rate(self, battery: BatteryData, cycle) -> Optional[float]:
        """Maximum instantaneous C-rate during charge window: max(|I|)/C_nom."""
        I = _to_array(getattr(cycle, 'current_in_A', []))
        C_nom = battery.nominal_capacity_in_Ah or 0.0
        if I.size == 0 or not C_nom:
            return None
        n = int(I.size)
        cs, ce = self.charge_window_indices(battery, cycle)
        cs = max(0, min(cs, n - 1)); ce = max(0, min(ce, n - 1))
        if ce < cs:
            return None
        Iw = I[cs:ce + 1]
        Iw = Iw[np.isfinite(Iw)]
        if Iw.size == 0:
            return None
        val = float(np.nanmax(np.abs(Iw)) / C_nom)
        return val if np.isfinite(val) else None

    def max_discharge_c_rate(self, battery: BatteryData, cycle) -> Optional[float]:
        """Maximum instantaneous C-rate during discharge window: max(|I|)/C_nom."""
        I = _to_array(getattr(cycle, 'current_in_A', []))
        C_nom = battery.nominal_capacity_in_Ah or 0.0
        if I.size == 0 or not C_nom:
            return None
        n = int(I.size)
        ds, de = self.discharge_window_indices(battery, cycle)
        ds = max(0, min(ds, n - 1)); de = max(0, min(de, n - 1))
        if de < ds:
            return None
        Iw = I[ds:de + 1]
        Iw = Iw[np.isfinite(Iw)]
        if Iw.size == 0:
            return None
        val = float(np.nanmax(np.abs(Iw)) / C_nom)
        return val if np.isfinite(val) else None

    def avg_charge_capacity(self, battery: BatteryData, cycle) -> Optional[float]:
        Qc = _to_array(getattr(cycle, 'charge_capacity_in_Ah', []))
        t = _to_array(getattr(cycle, 'time_in_s', []))
        if Qc.size == 0 or t.size == 0:
            return None
        n = int(min(Qc.size, t.size))
        Qc = Qc[:n]
        cs, ce = self.charge_window_indices(battery, cycle)
        cs = max(0, min(cs, n - 1)); ce = max(0, min(ce, n - 1))
        if ce < cs:
            return None
        seg = Qc[cs:ce + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            return None
        v = float(np.mean(seg))
        return v if np.isfinite(v) else None

    def avg_discharge_capacity(self, battery: BatteryData, cycle) -> Optional[float]:
        Qd = _to_array(getattr(cycle, 'discharge_capacity_in_Ah', []))
        t = _to_array(getattr(cycle, 'time_in_s', []))
        if Qd.size == 0 or t.size == 0:
            return None
        n = int(min(Qd.size, t.size))
        Qd = Qd[:n]
        ds, de = self.discharge_window_indices(battery, cycle)
        ds = max(0, min(ds, n - 1)); de = max(0, min(de, n - 1))
        if de < ds:
            return None
        seg = Qd[ds:de + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            return None
        v = float(np.mean(seg))
        return v if np.isfinite(v) else None

    def power_during_charge_cycle(self, battery: BatteryData, cycle) -> Optional[float]:
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        I = _to_array(getattr(cycle, 'current_in_A', []))
        t = _to_array(getattr(cycle, 'time_in_s', []))
        n = int(min(V.size, I.size, t.size))
        if n < 2:
            return None
        V = V[:n]; I = I[:n]; t = t[:n]
        cs, ce = self.charge_window_indices(battery, cycle)
        cs = max(0, min(cs, n - 1)); ce = max(0, min(ce, n - 1))
        if ce <= cs:
            return None
        Pw = (V[cs:ce + 1] * I[cs:ce + 1]).astype(float)
        tw = t[cs:ce + 1].astype(float)
        mfin = np.isfinite(Pw) & np.isfinite(tw)
        if np.count_nonzero(mfin) < 2:
            return None
        Pw = Pw[mfin]; tw = tw[mfin]
        val = float(np.trapz(Pw, tw))
        return val if np.isfinite(val) else None

    def power_during_discharge_cycle(self, battery: BatteryData, cycle) -> Optional[float]:
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        I = _to_array(getattr(cycle, 'current_in_A', []))
        t = _to_array(getattr(cycle, 'time_in_s', []))
        n = int(min(V.size, I.size, t.size))
        if n < 2:
            return None
        V = V[:n]; I = I[:n]; t = t[:n]
        ds, de = self.discharge_window_indices(battery, cycle)
        ds = max(0, min(ds, n - 1)); de = max(0, min(de, n - 1))
        if de <= ds:
            return None
        Pw = (V[ds:de + 1] * I[ds:de + 1]).astype(float)
        tw = t[ds:de + 1].astype(float)
        mfin = np.isfinite(Pw) & np.isfinite(tw)
        if np.count_nonzero(mfin) < 2:
            return None
        Pw = Pw[mfin]; tw = tw[mfin]
        val = float(np.trapz(Pw, tw))
        return val if np.isfinite(val) else None

    def charge_to_discharge_time_ratio(self, battery: BatteryData, cycle) -> Optional[float]:
        t = _to_array(getattr(cycle, 'time_in_s', []))
        if t.size < 2:
            return None
        n = int(t.size)
        cs, ce = self.charge_window_indices(battery, cycle)
        ds, de = self.discharge_window_indices(battery, cycle)
        cs = max(0, min(cs, n - 1)); ce = max(0, min(ce, n - 1))
        ds = max(0, min(ds, n - 1)); de = max(0, min(de, n - 1))
        if ce <= cs or de <= ds:
            return None
        t_charge = float(t[ce] - t[cs])
        t_discharge = float(t[de] - t[ds])
        if not (np.isfinite(t_charge) and np.isfinite(t_discharge)):
            return None
        if t_discharge <= 0:
            return None
        val = t_charge / t_discharge
        return float(val) if np.isfinite(val) else None

    # Note: peak_cv_length intentionally omitted for now


# ---------------------------- Concrete dataset classes ----------------------------

class MATRFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for MATR dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        imin = _first_min_index(I, 0)
        end = max(0, (imin - 1) if imin is not None else n - 1)
        return 0, end

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(max(V.size, I.size))
        if n == 0:
            return 0, -1
        # Start just after charge window end
        c_start, c_end = self.charge_window_indices(battery, cycle)
        start = min(n - 1, max(0, c_end + 1))
        vmin = _first_min_index(V, start)
        end = int(vmin) if vmin is not None else (n - 1)
        return start, end


class CALCEFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for CALCE dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        imin = _first_min_index(I, 0)
        end = max(0, (imin - 1) if imin is not None else n - 1)
        return 0, end

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(max(V.size, I.size))
        if n == 0:
            return 0, -1
        c_start, c_end = self.charge_window_indices(battery, cycle)
        start = min(n - 1, max(0, c_end + 1))
        vmin = _first_min_index(V, start)
        end = int(vmin) if vmin is not None else (n - 1)
        return start, end


class SNLFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for SNL dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        # Search for first zero after an initial offset of 100 samples
        zero_idx = _first_zero_index(I, start=min(100, max(0, n - 1)))
        if zero_idx is None:
            zero_idx = _first_zero_index(I, start=0)
        end = int(zero_idx) if zero_idx is not None else (n - 1)
        return 0, max(0, end)

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        imin = _first_min_index(I, 0)
        start = int(imin) if imin is not None else 0
        zero_after = _first_zero_index(I, start=start + 1)
        end = int(zero_after) if zero_after is not None else (n - 1)
        return start, end


class RWTHFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for RWTH dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        n = int(V.size)
        if n == 0:
            return 0, -1
        vmin = _first_min_index(V, 0)
        end = int(vmin) if vmin is not None else (n - 1)
        return 0, end

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        n = int(max(I.size, V.size))
        if n == 0:
            return 0, -1
        cs, ce = self.charge_window_indices(battery, cycle)
        # Start just after end of charge when current reaches its maximum
        i_max_idx = _first_max_index(I, start=min(n - 1, max(0, ce + 1)))
        start = int(i_max_idx) if i_max_idx is not None else min(n - 1, max(0, ce + 1))
        # End when voltage reaches its first maximum after that
        v_max_idx = _first_max_index(V, start=start)
        end = int(v_max_idx) if v_max_idx is not None else (n - 1)
        return start, end


class HNEIFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for HNEI dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        imin = _first_min_index(I, 0)
        end = max(0, (imin - 1) if imin is not None else n - 1)
        return 0, end

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        n = int(V.size)
        if n == 0:
            return 0, -1
        c_start, c_end = self.charge_window_indices(battery, cycle)
        start = min(n - 1, max(0, c_end + 1))
        vmin = _first_min_index(V, start)
        end = int(vmin) if vmin is not None else (n - 1)
        return start, end


class ULPURFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for UL_PUR dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        # Search for first zero after an initial offset of 20 samples
        zero_idx = _first_zero_index(I, start=min(20, max(0, n - 1)))
        if zero_idx is None:
            zero_idx = _first_zero_index(I, start=0)
        end = int(zero_idx) if zero_idx is not None else (n - 1)
        return 0, max(0, end)

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        n = int(max(I.size, V.size))
        if n == 0:
            return 0, -1
        imin = _first_min_index(I, 0)
        start = int(imin) if imin is not None else 0
        vmin = _first_min_index(V, start)
        end = int(vmin) if vmin is not None else (n - 1)
        return start, end


class HUSTFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for HUST dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        imin = _first_min_index(I, 0)
        end = max(0, (imin - 1) if imin is not None else n - 1)
        return 0, end

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        I = _to_array(getattr(cycle, 'current_in_A', []))
        n = int(I.size)
        if n == 0:
            return 0, -1
        c_start, c_end = self.charge_window_indices(battery, cycle)
        start = min(n - 1, max(0, c_end + 1))
        end = n - 1
        return start, end


class OXFeatures(DatasetSpecificCycleFeatures):
    """Dataset-specific implementation for OX dataset."""
    def charge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        # Reverse logic: start just after first voltage minimum until end of cycle
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        n = int(V.size)
        if n == 0:
            return 0, -1
        vmin = _first_min_index(V, 0)
        start = int(vmin) + 1 if vmin is not None else 0
        start = min(max(0, start), n - 1)
        end = n - 1
        return start, end

    def discharge_window_indices(self, battery: BatteryData, cycle) -> Tuple[int, int]:
        # Reverse logic: from start until first voltage minimum
        V = _to_array(getattr(cycle, 'voltage_in_V', []))
        n = int(V.size)
        if n == 0:
            return 0, -1
        vmin = _first_min_index(V, 0)
        end = int(vmin) if vmin is not None else (n - 1)
        return 0, end


# ---------------------------- Registry / Factory ----------------------------

_DATASET_TO_CLASS: Dict[str, Type[DatasetSpecificCycleFeatures]] = {
    'MATR': MATRFeatures,
    'MATR1': MATRFeatures,
    'MATR2': MATRFeatures,
    'CALCE': CALCEFeatures,
    'SNL': SNLFeatures,
    'RWTH': RWTHFeatures,
    'HNEI': HNEIFeatures,
    'UL_PUR': ULPURFeatures,
    'HUST': HUSTFeatures,
    'OX': OXFeatures,
}


def get_extractor_class(dataset_name: str) -> Optional[Type[DatasetSpecificCycleFeatures]]:
    """Return the dataset-specific feature extractor class for a dataset key."""
    return _DATASET_TO_CLASS.get(str(dataset_name).upper())


def extract_cycle_features(battery: BatteryData, dataset_name: str) -> pd.DataFrame:
    """
    Extract cycle features for a battery using the appropriate dataset-specific extractor.
    
    Args:
        battery: BatteryData object
        dataset_name: Name of the dataset (e.g., 'MATR', 'CALCE', etc.)
        
    Returns:
        DataFrame with cycle features
    """
    
    # Get the appropriate extractor class
    extractor_class = get_extractor_class(dataset_name)
    if extractor_class is None:
        # Fall back to base features if no dataset-specific extractor
        extractor = BaseCycleFeatures()
    else:
        extractor = extractor_class()
    
    # Get all available feature methods
    feature_methods = [method for method in dir(extractor) 
                      if not method.startswith('_') and callable(getattr(extractor, method))]
    
    # Extract features for each cycle
    rows = []
    for cycle in battery.cycle_data:
        row = {'battery_id': battery.cell_id, 'cycle': cycle.cycle_number}
        
        for method_name in feature_methods:
            try:
                method = getattr(extractor, method_name)
                value = method(battery, cycle)
                if value is not None and np.isfinite(float(value)):
                    row[method_name] = float(value)
                else:
                    row[method_name] = np.nan
            except Exception:
                row[method_name] = np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


__all__ = [
    'BaseCycleFeatures',
    'DatasetSpecificCycleFeatures',
    'MATRFeatures', 'CALCEFeatures', 'SNLFeatures', 'RWTHFeatures', 'HNEIFeatures', 'ULPURFeatures', 'HUSTFeatures', 'OXFeatures',
    'get_extractor_class',
    'extract_cycle_features',
]


