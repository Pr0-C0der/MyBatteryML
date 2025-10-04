#!/usr/bin/env python3
"""
Feature-RUL Correlation Analysis

This module provides functionality to plot correlation boxplots of statistical features
with log(Remaining Useful Life) using Spearman correlation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
import warnings
from scipy import stats
from scipy.stats import spearmanr
import itertools
from tqdm import tqdm

# Import necessary modules from the parent package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class, DatasetSpecificCycleFeatures
from batteryml.label.rul import RULLabelAnnotator
from batteryml.preprocess.base import BatteryPreprocessor
from batteryml.data.battery_data import BatteryData


class FeatureRULCorrelationAnalyzer:
    """
    Modular class for analyzing correlations between statistical features and log(RUL).
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Base directory containing the data
        """
        self.data_dir = Path(data_dir)
        self.feature_extractors = {}
        self.statistical_measures = {}
        self._register_default_features()
        self._register_default_statistical_measures()
    
    def _register_default_features(self):
        """Register default feature extractors."""
        self.feature_extractors = {
            'discharge_capacity': self._extract_discharge_capacity,
            'charge_capacity': self._extract_charge_capacity,
            'voltage': self._extract_voltage,
            'current': self._extract_current,
            'temperature': self._extract_temperature,
            'discharge_energy': self._extract_discharge_energy,
            'charge_energy': self._extract_charge_energy,
            'power': self._extract_power,
            'internal_resistance': self._extract_internal_resistance,
        }
        
        # Register cycle features
        self.cycle_feature_names = [
            'avg_voltage', 'avg_current', 'avg_c_rate', 'cycle_length',
            'max_charge_capacity', 'max_discharge_capacity',
            'charge_cycle_length', 'discharge_cycle_length',
            'avg_charge_c_rate', 'avg_discharge_c_rate',
            'max_charge_c_rate', 'avg_charge_capacity', 'avg_discharge_capacity',
            'power_during_charge_cycle', 'power_during_discharge_cycle',
            'charge_to_discharge_time_ratio'
        ]
    
    def _register_default_statistical_measures(self):
        """Register default statistical measures."""
        self.statistical_measures = {
            'mean': np.mean,
            'variance': np.var,
            'std': np.std,
            'median': np.median,
            'kurtosis': stats.kurtosis,
            'skewness': stats.skew,
            'min': np.min,
            'max': np.max,
            'range': lambda x: np.max(x) - np.min(x),
            'q25': lambda x: np.percentile(x, 25),
            'q75': lambda x: np.percentile(x, 75),
            'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        }
        
        # Register smoothing methods
        self.smoothing_methods = {
            'hms': self._hms_filter,
            'ma': self._moving_average,
            'median': self._moving_median,
        }
    
    def register_feature_extractor(self, name: str, extractor_func):
        """
        Register a new feature extractor.
        
        Args:
            name: Name of the feature
            extractor_func: Function that extracts the feature from cycle data
        """
        self.feature_extractors[name] = extractor_func
    
    def register_statistical_measure(self, name: str, measure_func):
        """
        Register a new statistical measure.
        
        Args:
            name: Name of the statistical measure
            measure_func: Function that calculates the measure
        """
        self.statistical_measures[name] = measure_func
    
    def register_smoothing_method(self, name: str, smoothing_func):
        """Register a new smoothing method."""
        self.smoothing_methods[name] = smoothing_func
    
    @staticmethod
    def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
        """Moving average smoothing."""
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            if w > arr.size:
                w = arr.size
            if w < 2:
                return arr
            padded = np.pad(arr, (w//2, w-1-w//2), mode='edge')
            out = np.empty_like(arr)
            for i in range(arr.size):
                seg = padded[i:i + w]
                m = np.isfinite(seg)
                out[i] = np.nanmean(seg[m]) if np.any(m) else np.nan
            return out
        except Exception:
            return y
    
    @staticmethod
    def _moving_median(y: np.ndarray, window: int) -> np.ndarray:
        """Moving median smoothing."""
        try:
            arr = np.asarray(y, dtype=float)
            if window is None or int(window) <= 1 or arr.size == 0:
                return arr
            w = int(window)
            if w > arr.size:
                w = arr.size
            if w < 2:
                return arr
            padded = np.pad(arr, (w//2, w-1-w//2), mode='edge')
            out = np.empty_like(arr)
            for i in range(arr.size):
                seg = padded[i:i + w]
                m = np.isfinite(seg)
                out[i] = np.nanmedian(seg[m]) if np.any(m) else np.nan
            return out
        except Exception:
            return y
    
    @staticmethod
    def _hampel_filter(y: np.ndarray, window_size: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
        """Hampel filter for outlier detection."""
        arr = np.asarray(y, dtype=float)
        n = arr.size
        if n == 0 or window_size < 3:
            return arr
        w = int(window_size)
        half = w // 2
        out = arr.copy()
        for i in range(n):
            l = max(0, i - half)
            r = min(n, i + half + 1)
            seg = arr[l:r]
            m = np.isfinite(seg)
            if not np.any(m):
                continue
            med = float(np.nanmedian(seg[m]))
            mad = float(np.nanmedian(np.abs(seg[m] - med)))
            if mad <= 0:
                continue
            thr = n_sigmas * 1.4826 * mad
            if np.isfinite(arr[i]) and abs(arr[i] - med) > thr:
                out[i] = med
        return out
    
    @staticmethod
    def _hms_filter(y: np.ndarray) -> np.ndarray:
        """HMS filter: Hampel -> Median -> Savitzky-Golay."""
        try:
            from scipy.signal import medfilt, savgol_filter
            arr = np.asarray(y, dtype=float)
            if arr.size == 0:
                return arr
            # 1) Hampel
            h = FeatureRULCorrelationAnalyzer._hampel_filter(arr, window_size=11, n_sigmas=3.0)
            # 2) Median filter (size=5)
            try:
                m = medfilt(h, kernel_size=5)
            except Exception:
                m = h
            # 3) Savitzky–Golay (window_length=101, polyorder=3), adjusted to length
            wl = 101
            if m.size < wl:
                wl = m.size if m.size % 2 == 1 else max(1, m.size - 1)
            if wl >= 5 and wl > 3:
                try:
                    s = savgol_filter(m, window_length=wl, polyorder=3, mode='interp')
                except Exception:
                    s = m
            else:
                s = m
            return s
        except Exception:
            return y
    
    def apply_smoothing(self, values: np.ndarray, method: str = None, window_size: int = 5) -> np.ndarray:
        """
        Apply smoothing to a series of values.
        
        Args:
            values: Array of values to smooth
            method: Smoothing method ('hms', 'ma', 'median')
            window_size: Window size for smoothing (ignored for HMS)
            
        Returns:
            Smoothed values
        """
        if method is None or method not in self.smoothing_methods:
            return values
        
        if len(values) == 0:
            return values
        
        try:
            if method == 'hms':
                return self.smoothing_methods[method](values)
            else:
                return self.smoothing_methods[method](values, window_size)
        except Exception as e:
            print(f"Warning: Smoothing failed with method {method}: {e}")
            return values
    
    def load_battery_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load battery data for the specified dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame containing battery data
        """
        data_path = self.data_dir / dataset_name
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_path}")
        
        # Load all battery files in the dataset
        battery_files = list(data_path.glob("*.csv"))
        if not battery_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        all_batteries = []
        for file_path in battery_files:
            try:
                battery_data = pd.read_csv(file_path)
                battery_data['battery_id'] = file_path.stem
                all_batteries.append(battery_data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not all_batteries:
            raise ValueError(f"No valid battery data found in {data_path}")
        
        return pd.concat(all_batteries, ignore_index=True)
    
    def calculate_rul_labels(self, data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Calculate RUL labels for the battery data.
        
        Args:
            data: Battery data DataFrame
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with RUL labels added
        """
        # Initialize RUL label annotator
        rul_annotator = RULLabelAnnotator()
        
        # Group by battery and calculate RUL
        data_with_rul = []
        for battery_id, battery_data in data.groupby('battery_id'):
            try:
                # Calculate RUL for this battery
                battery_data = battery_data.copy()
                battery_data = rul_annotator.annotate(battery_data, dataset_name)
                data_with_rul.append(battery_data)
            except Exception as e:
                print(f"Warning: Could not calculate RUL for battery {battery_id}: {e}")
                continue
        
        if not data_with_rul:
            raise ValueError("No valid RUL labels could be calculated")
        
        return pd.concat(data_with_rul, ignore_index=True)
    
    def _extract_discharge_capacity(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract discharge capacity from cycle data."""
        discharge_data = cycle_data[cycle_data['current'] < 0].copy()
        if discharge_data.empty:
            return np.array([])
        
        discharge_data = discharge_data.sort_values('time')
        cumulative_capacity = np.cumsum(discharge_data['current'] * discharge_data['time'].diff().fillna(0) / 3600)
        return cumulative_capacity.values
    
    def _extract_charge_capacity(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract charge capacity from cycle data."""
        charge_data = cycle_data[cycle_data['current'] > 0].copy()
        if charge_data.empty:
            return np.array([])
        
        charge_data = charge_data.sort_values('time')
        cumulative_capacity = np.cumsum(charge_data['current'] * charge_data['time'].diff().fillna(0) / 3600)
        return cumulative_capacity.values
    
    def _extract_voltage(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract voltage from cycle data."""
        return cycle_data['voltage'].dropna().values
    
    def _extract_current(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract current from cycle data."""
        return cycle_data['current'].dropna().values
    
    def _extract_temperature(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract temperature from cycle data."""
        if 'temperature' in cycle_data.columns:
            return cycle_data['temperature'].dropna().values
        return np.array([])
    
    def _extract_discharge_energy(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract discharge energy from cycle data."""
        discharge_data = cycle_data[cycle_data['current'] < 0].copy()
        if discharge_data.empty:
            return np.array([])
        
        discharge_data = discharge_data.sort_values('time')
        energy = np.cumsum(discharge_data['current'] * discharge_data['voltage'] * discharge_data['time'].diff().fillna(0) / 3600)
        return energy.values
    
    def _extract_charge_energy(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract charge energy from cycle data."""
        charge_data = cycle_data[cycle_data['current'] > 0].copy()
        if charge_data.empty:
            return np.array([])
        
        charge_data = charge_data.sort_values('time')
        energy = np.cumsum(charge_data['current'] * charge_data['voltage'] * charge_data['time'].diff().fillna(0) / 3600)
        return energy.values
    
    def _extract_power(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract power from cycle data."""
        power = cycle_data['current'] * cycle_data['voltage']
        return power.dropna().values
    
    def _extract_internal_resistance(self, cycle_data: pd.DataFrame) -> np.ndarray:
        """Extract internal resistance from cycle data."""
        # Simple internal resistance calculation: dV/dI
        current = cycle_data['current'].values
        voltage = cycle_data['voltage'].values
        
        if len(current) < 2:
            return np.array([])
        
        dV = np.diff(voltage)
        dI = np.diff(current)
        
        # Avoid division by zero
        valid_mask = dI != 0
        resistance = np.zeros_like(dI)
        resistance[valid_mask] = dV[valid_mask] / dI[valid_mask]
        
        return resistance[valid_mask]
    
    def extract_statistical_features(self, data: pd.DataFrame, 
                                   feature_name: str, 
                                   cycle_number: int,
                                   statistical_measures: List[str] = None) -> Dict[str, float]:
        """
        Extract statistical features for a specific cycle.
        
        Args:
            data: Battery data DataFrame
            feature_name: Name of the feature to extract
            cycle_number: Cycle number to extract features from
            statistical_measures: List of statistical measures to calculate
            
        Returns:
            Dictionary of statistical measures
        """
        if feature_name not in self.feature_extractors:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        if statistical_measures is None:
            statistical_measures = list(self.statistical_measures.keys())
        
        # Get cycle data
        cycle_data = data[data['cycle'] == cycle_number]
        if cycle_data.empty:
            return {measure: np.nan for measure in statistical_measures}
        
        # Extract feature values
        feature_values = self.feature_extractors[feature_name](cycle_data)
        
        if len(feature_values) == 0:
            return {measure: np.nan for measure in statistical_measures}
        
        # Calculate statistical measures
        results = {}
        for measure in statistical_measures:
            if measure in self.statistical_measures:
                try:
                    results[measure] = self.statistical_measures[measure](feature_values)
                except Exception as e:
                    print(f"Warning: Could not calculate {measure} for {feature_name}: {e}")
                    results[measure] = np.nan
            else:
                results[measure] = np.nan
        
        return results
    
    def extract_aggregated_features(self, data: pd.DataFrame, 
                                  feature_name: str, 
                                  statistical_measures: List[str] = None,
                                  cycle_limit: int = None,
                                  smoothing_method: str = None,
                                  smoothing_window: int = 5) -> Dict[str, float]:
        """
        Extract aggregated features across all cycles for a battery.
        
        This method handles the many-to-one relationship by aggregating cycle-level
        features to battery-level features.
        """
        if statistical_measures is None:
            statistical_measures = list(self.statistical_measures.keys())
        
        # Check if it's a cycle feature
        if feature_name in self.cycle_feature_names:
            return self._extract_aggregated_cycle_features(data, feature_name, statistical_measures, 
                                                          cycle_limit, smoothing_method, smoothing_window)
        
        # For raw features, use the existing approach
        if feature_name not in self.feature_extractors:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        # Get all cycles for this battery
        cycles = sorted(data['cycle'].unique())
        if cycle_limit is not None:
            cycles = cycles[:cycle_limit]
        
        if len(cycles) == 0:
            return {measure: np.nan for measure in statistical_measures}
        
        # Extract feature values from all cycles
        all_feature_values = []
        print(f"Extracting {feature_name} from {len(cycles)} cycles...")
        
        for cycle in tqdm(cycles, desc=f"Extracting {feature_name}", unit="cycle"):
            cycle_data = data[data['cycle'] == cycle]
            feature_values = self.feature_extractors[feature_name](cycle_data)
            if len(feature_values) > 0:
                all_feature_values.extend(feature_values)
        
        if len(all_feature_values) == 0:
            return {measure: np.nan for measure in statistical_measures}
        
        # Apply smoothing if specified
        if smoothing_method is not None:
            all_feature_values = self.apply_smoothing(np.array(all_feature_values), 
                                                     smoothing_method, smoothing_window)
        
        # Calculate statistical measures across all cycles
        results = {}
        for measure in statistical_measures:
            if measure in self.statistical_measures:
                try:
                    results[measure] = self.statistical_measures[measure](all_feature_values)
                except Exception as e:
                    print(f"Warning: Could not calculate {measure} for {feature_name}: {e}")
                    results[measure] = np.nan
            else:
                results[measure] = np.nan
        
        return results
    
    def _extract_aggregated_cycle_features(self, data: pd.DataFrame, 
                                         feature_name: str, 
                                         statistical_measures: List[str],
                                         cycle_limit: int = None,
                                         smoothing_method: str = None,
                                         smoothing_window: int = 5) -> Dict[str, float]:
        """Extract aggregated cycle features across all cycles for a battery."""
        # Get dataset name to determine feature extractor
        dataset_name = self._infer_dataset_from_data(data)
        if not dataset_name:
            return {measure: np.nan for measure in statistical_measures}
        
        # Get the appropriate feature extractor class
        extractor_class = get_extractor_class(dataset_name)
        if not extractor_class:
            print(f"Warning: No feature extractor found for dataset {dataset_name}")
            return {measure: np.nan for measure in statistical_measures}
        
        # Create feature extractor instance
        feature_extractor = extractor_class()
        
        # Get all cycles for this battery
        cycles = sorted(data['cycle'].unique())
        if cycle_limit is not None:
            cycles = cycles[:cycle_limit]
        
        if len(cycles) == 0:
            return {measure: np.nan for measure in statistical_measures}
        
        # Extract cycle feature values from all cycles
        cycle_feature_values = []
        print(f"Extracting {feature_name} from {len(cycles)} cycles...")
        
        for cycle in tqdm(cycles, desc=f"Extracting {feature_name}", unit="cycle"):
            cycle_data = data[data['cycle'] == cycle]
            if cycle_data.empty:
                continue
            
            # Convert cycle data to BatteryData format for feature extraction
            try:
                battery_data = self._convert_to_battery_data(cycle_data, dataset_name)
                cycle_obj = self._create_cycle_object(cycle_data)
                
                # Extract the specific cycle feature
                if hasattr(feature_extractor, feature_name):
                    feature_value = getattr(feature_extractor, feature_name)(battery_data, cycle_obj)
                    if feature_value is not None and not np.isnan(feature_value):
                        cycle_feature_values.append(feature_value)
            except Exception as e:
                print(f"Warning: Could not extract {feature_name} for cycle {cycle}: {e}")
                continue
        
        if len(cycle_feature_values) == 0:
            return {measure: np.nan for measure in statistical_measures}
        
        # Apply smoothing if specified
        if smoothing_method is not None:
            cycle_feature_values = self.apply_smoothing(np.array(cycle_feature_values), 
                                                       smoothing_method, smoothing_window)
        
        # Calculate statistical measures across all cycle feature values
        results = {}
        for measure in statistical_measures:
            if measure in self.statistical_measures:
                try:
                    results[measure] = self.statistical_measures[measure](cycle_feature_values)
                except Exception as e:
                    print(f"Warning: Could not calculate {measure} for {feature_name}: {e}")
                    results[measure] = np.nan
            else:
                results[measure] = np.nan
        
        return results
    
    def _infer_dataset_from_data(self, data: pd.DataFrame) -> Optional[str]:
        """Infer dataset name from battery data."""
        if 'battery_id' not in data.columns:
            return None
        
        # Get first battery ID to infer dataset
        first_battery_id = data['battery_id'].iloc[0]
        
        # Check for common dataset patterns
        if 'UL-PUR' in str(first_battery_id) or 'UL_PUR' in str(first_battery_id):
            return 'UL_PUR'
        elif 'MATR' in str(first_battery_id):
            return 'MATR'
        elif 'CALCE' in str(first_battery_id):
            return 'CALCE'
        elif 'HUST' in str(first_battery_id):
            return 'HUST'
        elif 'RWTH' in str(first_battery_id):
            return 'RWTH'
        elif 'OX' in str(first_battery_id):
            return 'OX'
        elif 'SNL' in str(first_battery_id):
            return 'SNL'
        elif 'HNEI' in str(first_battery_id):
            return 'HNEI'
        
        return None
    
    def _convert_to_battery_data(self, cycle_data: pd.DataFrame, dataset_name: str) -> BatteryData:
        """Convert cycle data to BatteryData format."""
        # This is a simplified conversion - in practice, you'd need to handle
        # the full BatteryData structure based on your data format
        battery_id = cycle_data['battery_id'].iloc[0]
        
        # Create a minimal BatteryData object
        # Note: This is a simplified version - you may need to adjust based on your actual data structure
        battery_data = BatteryData(
            cell_id=battery_id,
            dataset_name=dataset_name,
            nominal_capacity_in_Ah=2.0,  # Default value - should be extracted from metadata
            cycles=[]
        )
        
        return battery_data
    
    def _create_cycle_object(self, cycle_data: pd.DataFrame):
        """Create a cycle object for feature extraction."""
        # This is a simplified cycle object - you may need to adjust based on your actual data structure
        class Cycle:
            def __init__(self, data):
                self.voltage_in_V = data['voltage'].values if 'voltage' in data.columns else []
                self.current_in_A = data['current'].values if 'current' in data.columns else []
                self.time_in_s = data['time'].values if 'time' in data.columns else []
                self.charge_capacity_in_Ah = data['charge_capacity'].values if 'charge_capacity' in data.columns else []
                self.discharge_capacity_in_Ah = data['discharge_capacity'].values if 'discharge_capacity' in data.columns else []
                self.temperature = data['temperature'].values if 'temperature' in data.columns else []
        
        return Cycle(cycle_data)
    
    def calculate_correlations(self, data: pd.DataFrame, 
                             feature_name: str, 
                             statistical_measures: List[str] = None,
                             cycle_limit: int = None,
                             smoothing_method: str = None,
                             smoothing_window: int = 5) -> Dict[str, float]:
        """
        Calculate Spearman correlations between statistical features and log(RUL).
        
        Args:
            data: Battery data DataFrame with RUL labels
            feature_name: Name of the feature to analyze
            statistical_measures: List of statistical measures to calculate
            cycle_limit: Maximum number of cycles to use (None for all cycles)
            smoothing_method: Smoothing method to apply ('hms', 'ma', 'median')
            smoothing_window: Window size for smoothing (ignored for HMS)
            
        Returns:
            Dictionary of correlation coefficients
        """
        if statistical_measures is None:
            statistical_measures = list(self.statistical_measures.keys())
        
        return self._calculate_aggregated_correlations(data, feature_name, statistical_measures, 
                                                      cycle_limit, smoothing_method, smoothing_window)
    
    
    def _calculate_aggregated_correlations(self, data: pd.DataFrame, 
                                         feature_name: str, 
                                         statistical_measures: List[str],
                                         cycle_limit: int = None,
                                         smoothing_method: str = None,
                                         smoothing_window: int = 5) -> Dict[str, float]:
        """Calculate correlations using aggregated features across all cycles."""
        batteries = data['battery_id'].unique()
        
        feature_data = []
        rul_data = []
        
        print(f"Processing {len(batteries)} batteries for aggregated correlation analysis...")
        if cycle_limit:
            print(f"Using first {cycle_limit} cycles per battery")
        if smoothing_method:
            print(f"Applying {smoothing_method} smoothing")
        
        for battery_id in tqdm(batteries, desc="Processing batteries", unit="battery"):
            battery_data = data[data['battery_id'] == battery_id]
            battery_rul = battery_data['rul'].iloc[0] if 'rul' in battery_data.columns else np.nan
            
            if not np.isnan(battery_rul) and battery_rul > 0:
                features = self.extract_aggregated_features(battery_data, feature_name, statistical_measures,
                                                          cycle_limit, smoothing_method, smoothing_window)
                feature_data.append(features)
                rul_data.append(np.log(battery_rul))
        
        if len(feature_data) == 0:
            return {measure: np.nan for measure in statistical_measures}
        
        feature_df = pd.DataFrame(feature_data)
        rul_series = pd.Series(rul_data)
        
        return self._compute_correlations(feature_df, rul_series, statistical_measures)
    
    def _compute_correlations(self, feature_df: pd.DataFrame, 
                            rul_series: pd.Series, 
                            statistical_measures: List[str]) -> Dict[str, float]:
        """Compute correlations between features and RUL."""
        correlations = {}
        for measure in statistical_measures:
            if measure in feature_df.columns:
                try:
                    corr, p_value = spearmanr(feature_df[measure], rul_series, nan_policy='omit')
                    correlations[measure] = corr
                except Exception as e:
                    print(f"Warning: Could not calculate correlation for {measure}: {e}")
                    correlations[measure] = np.nan
            else:
                correlations[measure] = np.nan
        
        return correlations
    
    def plot_correlation_diverging_bar(self, data: pd.DataFrame, 
                                      feature_name: str, 
                                      dataset_name: str = "",
                                      statistical_measures: List[str] = None,
                                      cycle_limit: int = None,
                                      smoothing_method: str = None,
                                      smoothing_window: int = 5,
                                      output_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot diverging bar chart for statistical features with log(RUL).
        
        Args:
            data: Battery data DataFrame with RUL labels
            feature_name: Name of the feature to analyze
            dataset_name: Name of the dataset
            statistical_measures: List of statistical measures to calculate
            cycle_limit: Maximum number of cycles to use (None for all cycles)
            smoothing_method: Smoothing method to apply ('hms', 'ma', 'median')
            smoothing_window: Window size for smoothing (ignored for HMS)
            output_path: Path to save the plot (optional)
            figsize: Figure size
        """
        if statistical_measures is None:
            statistical_measures = list(self.statistical_measures.keys())
        
        # Calculate correlations
        correlations = self.calculate_correlations(data, feature_name, statistical_measures, 
                                                 cycle_limit, smoothing_method, smoothing_window)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Prepare data for diverging bar chart
        measures = []
        corr_values = []
        
        for measure, corr in correlations.items():
            if not np.isnan(corr):
                measures.append(measure)
                corr_values.append(corr)
        
        if len(corr_values) == 0:
            print("Warning: No valid correlations found")
            return
        
        # Sort by correlation value for better visualization
        sorted_data = sorted(zip(measures, corr_values), key=lambda x: x[1], reverse=True)
        measures, corr_values = zip(*sorted_data)
        
        # Create diverging bar chart
        colors = ['red' if x < 0 else 'blue' for x in corr_values]
        
        bars = plt.barh(range(len(measures)), corr_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        plt.yticks(range(len(measures)), measures)
        plt.xlabel('Spearman Correlation with log(RUL)', fontsize=12)
        plt.ylabel('Statistical Measures', fontsize=12)
        # Create title based on parameters
        cycle_info = f"First {cycle_limit} Cycles" if cycle_limit else "All Cycles"
        smoothing_info = f" ({smoothing_method.upper()} Smoothed)" if smoothing_method else ""
        
        title = f'Feature-RUL Correlation Analysis\nFeature: {feature_name} | {cycle_info} | Dataset: {dataset_name}{smoothing_info}'
        
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add vertical line at zero
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Add correlation values as text on bars
        for i, (bar, corr) in enumerate(zip(bars, corr_values)):
            width = bar.get_width()
            plt.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=10, fontweight='bold')
        
        # Add grid
        plt.grid(True, alpha=0.3, axis='x')
        
        # Set x-axis limits with some padding
        x_min, x_max = min(corr_values), max(corr_values)
        padding = (x_max - x_min) * 0.1
        plt.xlim(x_min - padding, x_max + padding)
        
        # Invert y-axis to show highest correlations at top
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    
    def analyze_all_datasets(self, dataset_names: List[str], 
                           feature_name: str, 
                           statistical_measures: List[str] = None,
                           cycle_limit: int = None,
                           smoothing_method: str = None,
                           smoothing_window: int = 5,
                           output_dir: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Analyze correlations for all specified datasets.
        
        Args:
            dataset_names: List of dataset names to analyze
            feature_name: Name of the feature to analyze
            statistical_measures: List of statistical measures to calculate
            cycle_limit: Maximum number of cycles to use (None for all cycles)
            smoothing_method: Smoothing method to apply ('hms', 'ma', 'median')
            smoothing_window: Window size for smoothing (ignored for HMS)
            output_dir: Directory to save the plots (optional)
            figsize: Figure size
        """
        if output_dir is None:
            output_dir = f"feature_rul_correlation_{feature_name}"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Analyzing {len(dataset_names)} datasets: {', '.join(dataset_names)}")
        
        for dataset_name in tqdm(dataset_names, desc="Processing datasets", unit="dataset"):
            try:
                print(f"\nAnalyzing dataset: {dataset_name}")
                
                # Load data
                data = self.load_battery_data(dataset_name)
                data = self.calculate_rul_labels(data, dataset_name)
                
                # Create plot
                plot_path = output_path / f"correlation_{dataset_name}.png"
                self.plot_correlation_diverging_bar(data, feature_name, 
                                                   dataset_name, statistical_measures, 
                                                   cycle_limit, smoothing_method, smoothing_window,
                                                   str(plot_path), figsize)
                
            except Exception as e:
                print(f"Error analyzing dataset {dataset_name}: {e}")
                continue
        
        print(f"\nAll plots saved to: {output_path}")


def main():
    """Main function to run the feature-RUL correlation analysis."""
    parser = argparse.ArgumentParser(description='Plot feature-RUL correlation analysis using diverging bar charts')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--feature', '-f', default='discharge_capacity', 
                       help='Feature to analyze (default: discharge_capacity)')
    parser.add_argument('--cycle_limit', type=int, 
                       help='Maximum number of cycles to use (default: all cycles)')
    parser.add_argument('--data_dir', '-d', default='data', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--measures', nargs='+', 
                       default=['mean', 'variance', 'median', 'kurtosis', 'skewness', 'min', 'max'],
                       help='Statistical measures to calculate')
    parser.add_argument('--cycle_limit', type=int, default=None, 
                       help='Limit analysis to first N cycles (default: all cycles)')
    parser.add_argument('--smoothing', choices=['none', 'hms', 'ma', 'median'], 
                       default='none', help='Smoothing method (default: none)')
    parser.add_argument('--smoothing_window', type=int, default=5, 
                       help='Window size for smoothing (ignored for HMS)')
    parser.add_argument('--all_datasets', nargs='+', 
                       help='Analyze all specified datasets')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = FeatureRULCorrelationAnalyzer(args.data_dir)
        
        if args.verbose:
            print(f"Available features: {list(analyzer.feature_extractors.keys())}")
            print(f"Available statistical measures: {list(analyzer.statistical_measures.keys())}")
        
        # Load data
        data = analyzer.load_battery_data(args.dataset_name)
        data = analyzer.calculate_rul_labels(data, args.dataset_name)
        
        if args.verbose:
            print(f"Loaded {len(data)} records from {data['battery_id'].nunique()} batteries")
        
        # Create output path if not provided
        output_path = args.output
        if output_path is None:
            output_path = f"correlation_{args.feature}_{args.dataset_name}.png"
        
        # Generate the plot
        if args.verbose:
            print("Generating plot...")
        
        if args.all_datasets:
            analyzer.analyze_all_datasets(args.all_datasets, args.feature, 
                                        args.measures, args.cycle_limit, 
                                        args.smoothing, args.smoothing_window,
                                        output_path, tuple(args.figsize))
        else:
            # Set smoothing method to None if 'none'
            smoothing_method = None if args.smoothing == 'none' else args.smoothing
            
            analyzer.plot_correlation_diverging_bar(data, args.feature, 
                                                  args.dataset_name, args.measures, 
                                                  args.cycle_limit, 
                                                  smoothing_method, args.smoothing_window,
                                                  output_path, tuple(args.figsize))
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
