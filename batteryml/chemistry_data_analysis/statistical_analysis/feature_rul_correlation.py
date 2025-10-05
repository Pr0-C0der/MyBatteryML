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
from batteryml.data.battery_data import BatteryData


class FeatureRULCorrelationAnalyzer:
    """
    Modular class for analyzing correlations between statistical features and log(RUL).
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the correlation analyzer.
        
        Args:
            data_dir: Base directory containing the data
        """
        self.data_dir = Path(data_dir)
        self.rul_annotator = RULLabelAnnotator()
        
        # Available cycle-level features (from cycle_features.py)
        self.cycle_feature_names = [
            # Common dataset-agnostic features (from BaseCycleFeatures)
            'avg_voltage',
            'avg_current', 
            'avg_c_rate',
            'cycle_length',
            'max_charge_capacity',
            'max_discharge_capacity',
            
            # Dataset-specific features (from DatasetSpecificCycleFeatures)
            'charge_cycle_length',
            'discharge_cycle_length',
            'avg_charge_c_rate',
            'avg_discharge_c_rate',
            'max_charge_c_rate',
            'avg_charge_capacity',
            'avg_discharge_capacity',
            'power_during_charge_cycle',
            'power_during_discharge_cycle',
            'charge_to_discharge_time_ratio'
        ]
        
        # Statistical measures to use
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
    def _hms_filter(y: np.ndarray, window: int = None) -> np.ndarray:
        """Hodrick-Prescott filter (simplified version)."""
        try:
            arr = np.asarray(y, dtype=float)
            if arr.size < 4:
                return arr
            # Simple moving average as placeholder for HP filter
            return FeatureRULCorrelationAnalyzer._moving_average(arr, min(5, arr.size // 2))
        except Exception:
            return y
    
    def apply_smoothing(self, values: np.ndarray, method: str = None, window_size: int = 5) -> np.ndarray:
        """
        Apply smoothing to the values.
        
        Args:
            values: Array of values to smooth
            method: Smoothing method ('hms', 'ma', 'median')
            window_size: Window size for smoothing (ignored for HMS)
            
        Returns:
            Smoothed values
        """
        if method is None or method not in self.smoothing_methods:
            return values
        
        try:
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
            DataFrame containing battery data with cycle features
        """
        data_path = self.data_dir / dataset_name
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_path}")
        
        # Load all battery files in the dataset (look for .pkl files)
        battery_files = list(data_path.glob("*.pkl"))
        if not battery_files:
            raise FileNotFoundError(f"No PKL files found in {data_path}")
        
        all_cycle_data = []
        
        for file_path in tqdm(battery_files, desc=f"Loading batteries from {dataset_name}", unit="battery", leave=True):
            try:
                # Load battery data using BatteryData.load()
                battery = BatteryData.load(file_path)
                
                # Extract cycle features using the proper extractor
                try:
                    # Try chemistry-specific extractor first
                    df = self._build_cycle_feature_table_extractor(battery, self.cycle_feature_names, dataset_name)
                except Exception:
                    # Fall back to basic feature extraction
                    df = self._build_cycle_feature_table_basic(battery)
                
                if not df.empty:
                    df['battery_id'] = battery.cell_id
                    all_cycle_data.append(df)
                    
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
        
        if not all_cycle_data:
            raise ValueError(f"No valid battery data found in {data_path}")
        
        return pd.concat(all_cycle_data, ignore_index=True)
    
    def _build_cycle_feature_table_extractor(self, battery: BatteryData, feature_names: List[str], dataset_hint: str) -> pd.DataFrame:
        """
        Build cycle feature table using chemistry-specific extractor.
        
        Args:
            battery: BatteryData object
            feature_names: List of feature names to extract
            dataset_hint: Dataset name hint for extractor selection
            
        Returns:
            DataFrame with cycle features
        """
        try:
            # Get the appropriate extractor class
            extractor_class = get_extractor_class(dataset_hint)
            if extractor_class is None:
                return pd.DataFrame()
            
            extractor = extractor_class()
            rows: List[Dict[str, float]] = []
            
            # Process each cycle
            for c in battery.cycle_data:
                row: Dict[str, float] = {'cycle_number': c.cycle_number}
                
                # Extract each requested feature
                for name in feature_names:
                    fn = getattr(extractor, name, None)
                    if not callable(fn):
                        continue
                    try:
                        val = fn(battery, c)
                        row[name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
                    except Exception:
                        row[name] = np.nan
                
                rows.append(row)
            
            return pd.DataFrame(rows)
                
        except Exception as e:
            print(f"Warning: Chemistry-specific extractor failed: {e}")
            return pd.DataFrame()
    
    def _build_cycle_feature_table_basic(self, battery: BatteryData) -> pd.DataFrame:
        """
        Build cycle feature table using basic feature extraction.
        
        Args:
            battery: BatteryData object
            
        Returns:
            DataFrame with basic cycle features
        """
        try:
            rows = []
            for cycle in battery.cycle_data:
                if not cycle.discharge_capacity_in_Ah or not cycle.charge_capacity_in_Ah:
                    continue
                    
                row = {
                    'cycle_number': cycle.cycle_number,
                    'avg_voltage': np.mean(cycle.voltage_in_V) if cycle.voltage_in_V is not None else np.nan,
                    'avg_current': np.mean(cycle.current_in_A) if cycle.current_in_A is not None else np.nan,
                    'cycle_length': len(cycle.voltage_in_V) if cycle.voltage_in_V is not None else 0,
                    'max_charge_capacity': cycle.charge_capacity_in_Ah,
                    'max_discharge_capacity': cycle.discharge_capacity_in_Ah,
                }
                rows.append(row)
            
            return pd.DataFrame(rows)
            
        except Exception as e:
            print(f"Warning: Basic feature extraction failed: {e}")
            return pd.DataFrame()
    
    def calculate_rul_labels(self, data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Calculate RUL labels for the battery data.
        
        Args:
            data: Battery data DataFrame with cycle features
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with RUL labels added
        """
        # Group by battery and calculate RUL
        data_with_rul = []
        battery_groups = list(data.groupby('battery_id'))
        
        for battery_id, battery_data in tqdm(battery_groups, desc="Calculating RUL labels", unit="battery", leave=True):
            try:
                # Calculate total RUL for this battery
                # Find the maximum cycle number for this battery
                max_cycle = battery_data['cycle_number'].max()
                
                # Calculate RUL for each cycle (remaining cycles until end of life)
                battery_data = battery_data.copy()
                battery_data['rul'] = max_cycle - battery_data['cycle_number']
                
                # Add log(RUL) for correlation analysis
                battery_data['log_rul'] = np.log(battery_data['rul'] + 1)  # +1 to avoid log(0)
                
                data_with_rul.append(battery_data)
                
            except Exception as e:
                print(f"Warning: Could not calculate RUL for battery {battery_id}: {e}")
                continue
        
        if not data_with_rul:
            raise ValueError("No valid RUL labels could be calculated")
        
        return pd.concat(data_with_rul, ignore_index=True)
    
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
        """Calculate correlations using cycle-level features."""
        # Filter data if cycle limit is specified
        if cycle_limit:
            data = data[data['cycle_number'] <= cycle_limit]
        
        # Check if feature exists in data
        if feature_name not in data.columns:
            print(f"Warning: Feature '{feature_name}' not found in data")
            return {measure: np.nan for measure in statistical_measures}
        
        # Remove rows with missing feature values
        data_clean = data.dropna(subset=[feature_name, 'log_rul'])
        
        if data_clean.empty:
            print(f"Warning: No valid data for feature '{feature_name}'")
            return {measure: np.nan for measure in statistical_measures}
        
        # Apply smoothing if specified
        if smoothing_method and smoothing_method != 'none':
            data_clean = data_clean.copy()
            data_clean[feature_name] = self.apply_smoothing(
                data_clean[feature_name].values, smoothing_method, smoothing_window
            )
        
        # Calculate statistical measures for each battery
        correlations = {}
        
        for measure in statistical_measures:
            if measure not in self.statistical_measures:
                correlations[measure] = np.nan
                continue
                
            try:
                # Group by battery and calculate the statistical measure
                battery_features = []
                battery_ruls = []
                
                for battery_id, battery_data in data_clean.groupby('battery_id'):
                    feature_values = battery_data[feature_name].values
                    rul_values = battery_data['log_rul'].values
                    
                    # Calculate the statistical measure for this battery
                    stat_value = self.statistical_measures[measure](feature_values)
                    
                    if not np.isnan(stat_value):
                        battery_features.append(stat_value)
                        # Use mean RUL for this battery
                        battery_ruls.append(np.mean(rul_values))
                
                if len(battery_features) > 1:
                    corr, _ = spearmanr(battery_features, battery_ruls, nan_policy='omit')
                    correlations[measure] = corr
                else:
                    correlations[measure] = np.nan
                    
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {measure}: {e}")
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


def main():
    """Main function to run feature-RUL correlation analysis."""
    parser = argparse.ArgumentParser(description='Feature-RUL correlation analysis')
    
    # Dataset and feature selection
    parser.add_argument('--dataset', required=True, 
                       help='Dataset name (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--feature', required=True, 
                       help='Feature name to analyze')
    parser.add_argument('--data_dir', default='data', 
                       help='Base directory containing the data')
    
    # Analysis parameters
    parser.add_argument('--cycle_limit', type=int, 
                       help='Maximum number of cycles to use (default: all cycles)')
    parser.add_argument('--smoothing', choices=['none', 'hms', 'ma', 'median'], 
                       default='none', help='Smoothing method (default: none)')
    parser.add_argument('--smoothing_window', type=int, default=5, 
                       help='Window size for smoothing (ignored for HMS)')
    
    # Output parameters
    parser.add_argument('--output', help='Output path for the plot (optional)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = FeatureRULCorrelationAnalyzer(args.data_dir)
        
        print(f"Loading data for dataset: {args.dataset}")
        data = analyzer.load_battery_data(args.dataset)
        print(f"Loaded {len(data)} records from {data['battery_id'].nunique()} batteries")
        
        print("Calculating RUL labels...")
        data = analyzer.calculate_rul_labels(data, args.dataset)
        print("RUL labels calculated successfully")
        
        # Set smoothing method to None if 'none'
        smoothing_method = None if args.smoothing == 'none' else args.smoothing
        
        print(f"Generating correlation plot for feature: {args.feature}")
        analyzer.plot_correlation_diverging_bar(
            data=data,
            feature_name=args.feature,
            dataset_name=args.dataset,
            cycle_limit=args.cycle_limit,
            smoothing_method=smoothing_method,
            smoothing_window=args.smoothing_window,
            output_path=args.output,
            figsize=tuple(args.figsize)
        )
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())