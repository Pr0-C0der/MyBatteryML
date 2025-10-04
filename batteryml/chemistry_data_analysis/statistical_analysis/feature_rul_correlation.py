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

# Import necessary modules from the parent package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from batteryml.chemistry_data_analysis.cycle_features import extract_cycle_features
from batteryml.label.rul import RULLabelAnnotator
from batteryml.preprocess.base import BatteryPreprocessor


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
    
    def calculate_correlations(self, data: pd.DataFrame, 
                             feature_name: str, 
                             cycle_number: int,
                             statistical_measures: List[str] = None) -> Dict[str, float]:
        """
        Calculate Spearman correlations between statistical features and log(RUL).
        
        Args:
            data: Battery data DataFrame with RUL labels
            feature_name: Name of the feature to analyze
            cycle_number: Cycle number to extract features from
            statistical_measures: List of statistical measures to calculate
            
        Returns:
            Dictionary of correlation coefficients
        """
        if statistical_measures is None:
            statistical_measures = list(self.statistical_measures.keys())
        
        # Get unique batteries
        batteries = data['battery_id'].unique()
        
        # Extract features and RUL for each battery
        feature_data = []
        rul_data = []
        
        for battery_id in batteries:
            battery_data = data[data['battery_id'] == battery_id]
            
            # Get RUL for this battery (use first cycle's RUL)
            battery_rul = battery_data['rul'].iloc[0] if 'rul' in battery_data.columns else np.nan
            
            if not np.isnan(battery_rul) and battery_rul > 0:
                # Extract statistical features
                features = self.extract_statistical_features(battery_data, feature_name, cycle_number, statistical_measures)
                
                feature_data.append(features)
                rul_data.append(np.log(battery_rul))
        
        if len(feature_data) == 0:
            return {measure: np.nan for measure in statistical_measures}
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_data)
        rul_series = pd.Series(rul_data)
        
        # Calculate correlations
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
    
    def plot_correlation_boxplot(self, data: pd.DataFrame, 
                                feature_name: str, 
                                cycle_number: int,
                                dataset_name: str,
                                statistical_measures: List[str] = None,
                                output_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot correlation boxplot for statistical features with log(RUL).
        
        Args:
            data: Battery data DataFrame with RUL labels
            feature_name: Name of the feature to analyze
            cycle_number: Cycle number to extract features from
            dataset_name: Name of the dataset
            statistical_measures: List of statistical measures to calculate
            output_path: Path to save the plot (optional)
            figsize: Figure size
        """
        if statistical_measures is None:
            statistical_measures = list(self.statistical_measures.keys())
        
        # Calculate correlations
        correlations = self.calculate_correlations(data, feature_name, cycle_number, statistical_measures)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Prepare data for boxplot
        measures = []
        corr_values = []
        
        for measure, corr in correlations.items():
            if not np.isnan(corr):
                measures.append(measure)
                corr_values.append(corr)
        
        if len(corr_values) == 0:
            print("Warning: No valid correlations found")
            return
        
        # Create boxplot
        plt.boxplot(corr_values, labels=measures)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Spearman Correlation with log(RUL)', fontsize=12)
        plt.title(f'Feature-RUL Correlation Analysis\nFeature: {feature_name} | Cycle: {cycle_number} | Dataset: {dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Add correlation values as text
        for i, (measure, corr) in enumerate(zip(measures, corr_values)):
            plt.text(i + 1, corr, f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_multi_cycle_correlation(self, data: pd.DataFrame, 
                                   feature_name: str, 
                                   cycle_numbers: List[int],
                                   dataset_name: str,
                                   statistical_measures: List[str] = None,
                                   output_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot correlation analysis across multiple cycles.
        
        Args:
            data: Battery data DataFrame with RUL labels
            feature_name: Name of the feature to analyze
            cycle_numbers: List of cycle numbers to analyze
            dataset_name: Name of the dataset
            statistical_measures: List of statistical measures to calculate
            output_path: Path to save the plot (optional)
            figsize: Figure size
        """
        if statistical_measures is None:
            statistical_measures = list(self.statistical_measures.keys())
        
        # Calculate correlations for each cycle
        all_correlations = []
        cycle_labels = []
        
        for cycle in cycle_numbers:
            correlations = self.calculate_correlations(data, feature_name, cycle, statistical_measures)
            all_correlations.append(correlations)
            cycle_labels.append(f'Cycle {cycle}')
        
        # Create DataFrame for easier plotting
        corr_df = pd.DataFrame(all_correlations, index=cycle_labels)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.3f', cbar_kws={'label': 'Spearman Correlation with log(RUL)'})
        
        plt.title(f'Feature-RUL Correlation Heatmap\nFeature: {feature_name} | Dataset: {dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Statistical Measures', fontsize=12)
        plt.ylabel('Cycles', fontsize=12)
        
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
                           cycle_number: int,
                           statistical_measures: List[str] = None,
                           output_dir: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Analyze correlations for all specified datasets.
        
        Args:
            dataset_names: List of dataset names to analyze
            feature_name: Name of the feature to analyze
            cycle_number: Cycle number to extract features from
            statistical_measures: List of statistical measures to calculate
            output_dir: Directory to save the plots (optional)
            figsize: Figure size
        """
        if output_dir is None:
            output_dir = f"feature_rul_correlation_{feature_name}_cycle_{cycle_number}"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for dataset_name in dataset_names:
            try:
                print(f"Analyzing dataset: {dataset_name}")
                
                # Load data
                data = self.load_battery_data(dataset_name)
                data = self.calculate_rul_labels(data, dataset_name)
                
                # Create plot
                plot_path = output_path / f"correlation_{dataset_name}.png"
                self.plot_correlation_boxplot(data, feature_name, cycle_number, 
                                            dataset_name, statistical_measures, 
                                            str(plot_path), figsize)
                
            except Exception as e:
                print(f"Error analyzing dataset {dataset_name}: {e}")
                continue
        
        print(f"All plots saved to: {output_path}")


def main():
    """Main function to run the feature-RUL correlation analysis."""
    parser = argparse.ArgumentParser(description='Plot feature-RUL correlation analysis')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--feature', '-f', default='discharge_capacity', 
                       help='Feature to analyze (default: discharge_capacity)')
    parser.add_argument('--cycle', '-c', type=int, default=100, 
                       help='Cycle number to analyze (default: 100)')
    parser.add_argument('--data_dir', '-d', default='data', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--measures', nargs='+', 
                       default=['mean', 'variance', 'median', 'kurtosis', 'skewness', 'min', 'max'],
                       help='Statistical measures to calculate')
    parser.add_argument('--multi_cycle', nargs='+', type=int, 
                       help='Analyze multiple cycles (e.g., --multi_cycle 50 100 150)')
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
            if args.multi_cycle:
                output_path = f"correlation_multi_cycle_{args.feature}_{args.dataset_name}.png"
            else:
                output_path = f"correlation_{args.feature}_cycle_{args.cycle}_{args.dataset_name}.png"
        
        # Generate the plot
        if args.verbose:
            print("Generating plot...")
        
        if args.multi_cycle:
            analyzer.plot_multi_cycle_correlation(data, args.feature, args.multi_cycle, 
                                               args.dataset_name, args.measures, 
                                               output_path, tuple(args.figsize))
        elif args.all_datasets:
            analyzer.analyze_all_datasets(args.all_datasets, args.feature, args.cycle, 
                                        args.measures, output_path, tuple(args.figsize))
        else:
            analyzer.plot_correlation_boxplot(data, args.feature, args.cycle, 
                                            args.dataset_name, args.measures, 
                                            output_path, tuple(args.figsize))
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
