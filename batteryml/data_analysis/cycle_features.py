from __future__ import annotations

import numpy as np
from typing import Optional

from batteryml.data.battery_data import BatteryData


def _align_and_mask(a, b):
    aa = np.array(a)
    bb = np.array(b)
    n = min(len(aa), len(bb))
    if n == 0:
        return None, None
    aa = aa[:n]
    bb = bb[:n]
    m = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(m):
        return None, None
    return aa[m], bb[m]


def avg_c_rate(b: BatteryData, c) -> Optional[float]:
    I = getattr(c, 'current_in_A', None)
    C = b.nominal_capacity_in_Ah or 0.0
    if I is None or not C:
        return None
    I = np.array(I)
    I = I[np.isfinite(I)]
    if I.size == 0:
        return None
    return float(np.mean(np.abs(I)) / C)


def max_temperature(b: BatteryData, c) -> Optional[float]:
    T = getattr(c, 'temperature_in_C', None)
    if T is None:
        return None
    T = np.array(T)
    T = T[np.isfinite(T)]
    if T.size == 0:
        return None
    m = np.nanmax(T)
    return float(m) if np.isfinite(m) else None


def max_discharge_capacity(b: BatteryData, c) -> Optional[float]:
    Qd = getattr(c, 'discharge_capacity_in_Ah', None)
    if Qd is None:
        return None
    Qd = np.array(Qd)
    Qd = Qd[np.isfinite(Qd)]
    if Qd.size == 0:
        return None
    m = np.nanmax(Qd)
    return float(m) if np.isfinite(m) else None


def max_charge_capacity(b: BatteryData, c) -> Optional[float]:
    Qc = getattr(c, 'charge_capacity_in_Ah', None)
    if Qc is None:
        return None
    Qc = np.array(Qc)
    Qc = Qc[np.isfinite(Qc)]
    if Qc.size == 0:
        return None
    m = np.nanmax(Qc)
    return float(m) if np.isfinite(m) else None


def avg_discharge_capacity(b: BatteryData, c) -> Optional[float]:
    Qd = getattr(c, 'discharge_capacity_in_Ah', None)
    if Qd is None:
        return None
    Qd = np.array(Qd)
    Qd = Qd[np.isfinite(Qd)]
    if Qd.size == 0:
        return None
    v = float(np.mean(Qd))
    return v if np.isfinite(v) else None


def avg_charge_capacity(b: BatteryData, c) -> Optional[float]:
    Qc = getattr(c, 'charge_capacity_in_Ah', None)
    if Qc is None:
        return None
    Qc = np.array(Qc)
    Qc = Qc[np.isfinite(Qc)]
    if Qc.size == 0:
        return None
    v = float(np.mean(Qc))
    return v if np.isfinite(v) else None


def charge_cycle_length(b: BatteryData, c) -> Optional[float]:
    t = getattr(c, 'time_in_s', None)
    if t is None:
        return None
    # Use shared window indices for consistency
    wnd = _charge_window_indices(b, c)
    if wnd is None:
        return None
    t_arr = np.array(t)
    t_arr = t_arr[np.isfinite(t_arr)]
    if t_arr.size == 0:
        return None
    start_idx = wnd.start or 0
    end_idx = wnd.stop - 1 if wnd.stop is not None else t_arr.size - 1
    if end_idx < start_idx or end_idx >= t_arr.size:
        return None
    val = float(t_arr[end_idx] - t_arr[start_idx])
    return val if np.isfinite(val) and val >= 0 else None


def discharge_cycle_length(b: BatteryData, c) -> Optional[float]:
    t = getattr(c, 'time_in_s', None)
    if t is None:
        return None
    wnd = _discharge_window_indices(b, c)
    if wnd is None:
        return None
    t_arr = np.array(t)
    t_arr = t_arr[np.isfinite(t_arr)]
    if t_arr.size == 0:
        return None
    start_idx = wnd.start or 0
    end_idx = t_arr.size - 1 if wnd.stop is None else (wnd.stop - 1)
    if end_idx < start_idx or end_idx >= t_arr.size:
        return None
    val = float(t_arr[end_idx] - t_arr[start_idx])
    return val if np.isfinite(val) and val >= 0 else None


def _last_peak_length(arr: np.ndarray, tt: np.ndarray) -> Optional[float]:
    if arr.size == 0 or tt.size == 0:
        return None
    vmax = np.nanmax(arr)
    close = np.isclose(arr, vmax, rtol=1e-3, atol=1e-6)
    if not np.any(close):
        return None
    last_idx = np.where(close)[0][-1]
    v = float(tt[last_idx] - tt[0])
    return v if np.isfinite(v) and v >= 0 else None


def peak_cc_length(b: BatteryData, c) -> Optional[float]:
    I = getattr(c, 'current_in_A', None)
    t = getattr(c, 'time_in_s', None)
    if I is None or t is None:
        return None
    I_arr, t_arr = _align_and_mask(I, t)
    if I_arr is None or t_arr is None:
        return None
    wnd = _charge_window_indices(b, c)
    if wnd is None:
        return None
    Iw = I_arr[wnd]
    tw = t_arr[wnd]
    n = Iw.size
    if n < 32:
        return None

    # Simple rolling slope with fixed window size (30 samples)
    win = 30
    slopes = []
    for k in range(0, n - win + 1):
        dt = float(tw[k + win - 1] - tw[k])
        if dt <= 0 or not np.isfinite(dt):
            slopes.append(0.0)
            continue
        m = float(Iw[k + win - 1] - Iw[k]) / dt
        slopes.append(m if np.isfinite(m) else 0.0)
    slopes = np.array(slopes, dtype=float)
    if slopes.size < 2:
        return None

    # Robust threshold on slope change to ignore huge jumps
    dsl = np.diff(slopes)
    med = float(np.median(dsl))
    mad = float(np.median(np.abs(dsl - med)))
    noise = (mad / 0.6745) if mad > 0 else float(np.std(dsl))
    huge_thresh = 6.0 * noise if np.isfinite(noise) and noise > 0 else np.inf

    # Find first start where slope turns negative (gradual decrease), skipping huge changes
    tC_idx = None
    for k in range(slopes.size):
        huge = (k > 0 and abs(dsl[k - 1]) > huge_thresh)
        prev_nonneg = (k == 0 or slopes[k - 1] >= 0)
        if (not huge) and prev_nonneg and (slopes[k] < 0):
            tC_idx = k
            break

    if tC_idx is None:
        return None

    # Map to time from start of charge window
    t_len = float(tw[tC_idx] - tw[0])
    # print(f"Peak CC Length: {t_len}")
    # print(f"Time Window: {tw[tC_idx]}")
    # print(f"Time Array: {t_arr[wnd]}")
    return t_len if np.isfinite(t_len) and t_len >= 0 else None


def peak_cv_length(b: BatteryData, c) -> Optional[float]:
    V = getattr(c, 'voltage_in_V', None)
    t = getattr(c, 'time_in_s', None)
    if V is None or t is None:
        return None
    V_arr, t_arr = _align_and_mask(V, t)
    if V_arr is None or t_arr is None:
        return None
    return _last_peak_length(V_arr, t_arr)


def cycle_length(b: BatteryData, c) -> Optional[float]:
    t = getattr(c, 'time_in_s', None)
    if t is None:
        return None
    t_arr = np.array(t)
    t_arr = t_arr[np.isfinite(t_arr)]
    if t_arr.size == 0:
        return None
    v = float(t_arr[-1] - t_arr[0])
    return v if np.isfinite(v) and v >= 0 else None


# ------------------------------
# Window helpers (indices on mask
# ------------------------------
def _charge_window_indices(b: BatteryData, c) -> Optional[slice]:
    I = getattr(c, 'current_in_A', None)
    t = getattr(c, 'time_in_s', None)
    if I is None or t is None:
        return None
    I_arr, t_arr = _align_and_mask(I, t)
    if I_arr is None or t_arr is None:
        return None
    Imin = np.nanmin(I_arr)
    peak_idxs = np.where(np.isclose(I_arr, Imin, equal_nan=False, atol=1e-3))[0]
    if peak_idxs.size == 0:
        return None
    peak_idx = int(peak_idxs[0])
    end_idx = peak_idx + 1 if (peak_idx + 1) < t_arr.size else peak_idx
    # inclusive end for np.trapz; slice uses end_idx+1
    return slice(0, end_idx + 1)


def _discharge_window_indices(b: BatteryData, c) -> Optional[slice]:
    I = getattr(c, 'current_in_A', None)
    t = getattr(c, 'time_in_s', None)
    if I is None or t is None:
        return None
    I_arr, t_arr = _align_and_mask(I, t)
    if I_arr is None or t_arr is None:
        return None
    Imin = np.nanmin(I_arr)
    peak_idxs = np.where(np.isclose(I_arr, Imin, equal_nan=False, atol=1e-3))[0]
    if peak_idxs.size == 0:
        return None
    peak_idx = int(peak_idxs[0])
    return slice(peak_idx, t_arr.size)


# -----------------------------------------------
# New requested features using windowed integration
# -----------------------------------------------
def power_during_charge_cycle(b: BatteryData, c) -> Optional[float]:
    V = getattr(c, 'voltage_in_V', None)
    I = getattr(c, 'current_in_A', None)
    t = getattr(c, 'time_in_s', None)
    if V is None or I is None or t is None:
        return None
    V_arr, t_arr = _align_and_mask(V, t)
    I_arr, _ = _align_and_mask(I, t)
    if V_arr is None or I_arr is None or t_arr is None:
        return None
    wnd = _charge_window_indices(b, c)
    if wnd is None:
        return None
    Pw = V_arr[wnd] * I_arr[wnd]
    tw = t_arr[wnd]
    if Pw.size < 2 or tw.size < 2:
        return None
    val = float(np.trapz(Pw, tw))
    return val if np.isfinite(val) else None


def power_during_discharge_cycle(b: BatteryData, c) -> Optional[float]:
    V = getattr(c, 'voltage_in_V', None)
    I = getattr(c, 'current_in_A', None)
    t = getattr(c, 'time_in_s', None)
    if V is None or I is None or t is None:
        return None
    V_arr, t_arr = _align_and_mask(V, t)
    I_arr, _ = _align_and_mask(I, t)
    if V_arr is None or I_arr is None or t_arr is None:
        return None
    wnd = _discharge_window_indices(b, c)
    if wnd is None:
        return None
    Pw = V_arr[wnd] * I_arr[wnd]
    tw = t_arr[wnd]
    if Pw.size < 2 or tw.size < 2:
        return None
    val = float(np.trapz(Pw, tw))
    return val if np.isfinite(val) else None


def avg_charge_c_rate(b: BatteryData, c) -> Optional[float]:
    I = getattr(c, 'current_in_A', None)
    t = getattr(c, 'time_in_s', None)
    C_nom = b.nominal_capacity_in_Ah or 0.0
    if I is None or t is None or not C_nom:
        return None
    I_arr, t_arr = _align_and_mask(I, t)
    if I_arr is None or t_arr is None:
        return None
    wnd = _charge_window_indices(b, c)
    if wnd is None:
        return None
    Iw = I_arr[wnd]
    tw = t_arr[wnd]
    if Iw.size < 2 or tw.size < 2:
        return None
    q_as = float(np.trapz(Iw, tw))  # A*s
    val = q_as / (C_nom * 3600.0)   # dimensionless average C-rate
    return val if np.isfinite(val) else None


def avg_discharge_c_rate(b: BatteryData, c) -> Optional[float]:
    I = getattr(c, 'current_in_A', None)
    t = getattr(c, 'time_in_s', None)
    C_nom = b.nominal_capacity_in_Ah or 0.0
    if I is None or t is None or not C_nom:
        return None
    I_arr, t_arr = _align_and_mask(I, t)
    if I_arr is None or t_arr is None:
        return None
    wnd = _discharge_window_indices(b, c)
    if wnd is None:
        return None
    Iw = np.abs(I_arr[wnd])
    tw = t_arr[wnd]
    if Iw.size < 2 or tw.size < 2:
        return None
    q_as = float(np.trapz(Iw, tw))  # A*s
    val = q_as / (C_nom * 3600.0)   # dimensionless average C-rate
    return val if np.isfinite(val) else None



def charge_to_discharge_time_ratio(b: BatteryData, c) -> Optional[float]:
    t = getattr(c, 'time_in_s', None)
    if t is None:
        return None
    t_arr = np.array(t)
    t_arr = t_arr[np.isfinite(t_arr)]
    if t_arr.size < 2:
        return None
    w_charge = _charge_window_indices(b, c)
    w_discharge = _discharge_window_indices(b, c)
    if w_charge is None or w_discharge is None:
        return None
    c_start = w_charge.start or 0
    c_end = (w_charge.stop - 1) if w_charge.stop is not None else (t_arr.size - 1)
    d_start = w_discharge.start or 0
    d_end = (w_discharge.stop - 1) if w_discharge.stop is not None else (t_arr.size - 1)
    if not (0 <= c_start <= c_end < t_arr.size):
        return None
    if not (0 <= d_start <= d_end < t_arr.size):
        return None
    t_charge = float(t_arr[c_end] - t_arr[c_start])
    t_discharge = float(t_arr[d_end] - t_arr[d_start])
    if not (np.isfinite(t_charge) and np.isfinite(t_discharge)):
        return None
    if t_discharge <= 0:
        return None
    val = t_charge / t_discharge
    return float(val) if np.isfinite(val) else None

