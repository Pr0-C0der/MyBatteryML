#!/usr/bin/env python3
"""
Example script showing how to use the FeatureRULCorrelationAnalyzer
with custom features and statistical measures.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from batteryml.chemistry_data_analysis.statistical_analysis.feature_rul_correlation import FeatureRULCorrelationAnalyzer


def example_custom_features():
    """Example of adding custom features to the analyzer."""
    
    # Initialize analyzer
    analyzer = FeatureRULCorrelationAnalyzer("data")
    
    # Add custom feature extractors
    def extract_voltage_range(cycle_data):
        """Extract voltage range (max - min) for a cycle."""
        voltage = cycle_data['voltage'].dropna()
        if len(voltage) == 0:
            return np.array([])
        return np.array([voltage.max() - voltage.min()])
    
    def extract_capacity_fade(cycle_data):
        """Extract capacity fade (capacity at end - capacity at start)."""
        discharge_data = cycle_data[cycle_data['current'] < 0].copy()
        if discharge_data.empty:
            return np.array([])
        
        discharge_data = discharge_data.sort_values('time')
        cumulative_capacity = np.cumsum(discharge_data['current'] * discharge_data['time'].diff().fillna(0) / 3600)
        
        if len(cumulative_capacity) == 0:
            return np.array([])
        
        return np.array([cumulative_capacity[-1] - cumulative_capacity[0]])
    
    def extract_voltage_efficiency(cycle_data):
        """Extract voltage efficiency (average voltage during discharge)."""
        discharge_data = cycle_data[cycle_data['current'] < 0]
        if discharge_data.empty:
            return np.array([])
        
        voltage = discharge_data['voltage'].dropna()
        if len(voltage) == 0:
            return np.array([])
        
        return voltage.values
    
    # Register custom features
    analyzer.register_feature_extractor('voltage_range', extract_voltage_range)
    analyzer.register_feature_extractor('capacity_fade', extract_capacity_fade)
    analyzer.register_feature_extractor('voltage_efficiency', extract_voltage_efficiency)
    
    return analyzer


def example_custom_statistical_measures():
    """Example of adding custom statistical measures."""
    
    # Initialize analyzer
    analyzer = FeatureRULCorrelationAnalyzer("data")
    
    # Add custom statistical measures
    def coefficient_of_variation(x):
        """Calculate coefficient of variation (std/mean)."""
        if len(x) == 0 or np.mean(x) == 0:
            return np.nan
        return np.std(x) / np.mean(x)
    
    def energy_efficiency(x):
        """Calculate energy efficiency (useful for power features)."""
        if len(x) == 0:
            return np.nan
        return np.sum(x[x > 0]) / np.sum(np.abs(x)) if np.sum(np.abs(x)) != 0 else np.nan
    
    def peak_to_peak(x):
        """Calculate peak-to-peak value."""
        if len(x) == 0:
            return np.nan
        return np.max(x) - np.min(x)
    
    def rms(x):
        """Calculate root mean square."""
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))
    
    # Register custom statistical measures
    analyzer.register_statistical_measure('cv', coefficient_of_variation)
    analyzer.register_statistical_measure('energy_eff', energy_efficiency)
    analyzer.register_statistical_measure('peak_to_peak', peak_to_peak)
    analyzer.register_statistical_measure('rms', rms)
    
    return analyzer


def example_comprehensive_analysis():
    """Example of comprehensive analysis with custom features and measures."""
    
    # Create analyzer with custom features and measures
    analyzer = example_custom_features()
    analyzer = example_custom_statistical_measures()
    
    # Define datasets to analyze
    datasets = ['UL_PUR', 'MATR', 'CALCE', 'HUST', 'RWTH', 'OX']
    
    # Define features to analyze
    features = ['discharge_capacity', 'voltage', 'voltage_range', 'capacity_fade', 'voltage_efficiency']
    
    # Define statistical measures
    measures = ['mean', 'variance', 'median', 'kurtosis', 'skewness', 'min', 'max', 'cv', 'rms']
    
    # Define cycles to analyze
    cycles = [50, 100, 150, 200]
    
    print("Running comprehensive feature-RUL correlation analysis...")
    
    for dataset in datasets:
        try:
            print(f"\nAnalyzing dataset: {dataset}")
            
            # Load data
            data = analyzer.load_battery_data(dataset)
            data = analyzer.calculate_rul_labels(data, dataset)
            
            for feature in features:
                print(f"  Analyzing feature: {feature}")
                
                # Single cycle analysis
                output_path = f"correlation_{feature}_cycle_100_{dataset}.png"
                analyzer.plot_correlation_boxplot(data, feature, 100, dataset, measures, output_path)
                
                # Multi-cycle analysis
                output_path = f"correlation_multi_cycle_{feature}_{dataset}.png"
                analyzer.plot_multi_cycle_correlation(data, feature, cycles, dataset, measures, output_path)
                
        except Exception as e:
            print(f"Error analyzing dataset {dataset}: {e}")
            continue
    
    print("\nComprehensive analysis completed!")


def example_specific_analysis():
    """Example of specific analysis for a particular use case."""
    
    # Initialize analyzer
    analyzer = FeatureRULCorrelationAnalyzer("data")
    
    # Add custom features for battery health monitoring
    def extract_voltage_plateau_length(cycle_data):
        """Extract length of voltage plateau during discharge."""
        discharge_data = cycle_data[cycle_data['current'] < 0].copy()
        if discharge_data.empty:
            return np.array([])
        
        discharge_data = discharge_data.sort_values('time')
        voltage = discharge_data['voltage'].values
        
        # Find voltage plateau (where voltage change is minimal)
        voltage_diff = np.abs(np.diff(voltage))
        plateau_threshold = 0.01  # 10mV threshold
        plateau_mask = voltage_diff < plateau_threshold
        
        # Calculate plateau length
        plateau_length = np.sum(plateau_mask)
        return np.array([plateau_length])
    
    def extract_charge_time(cycle_data):
        """Extract total charge time for a cycle."""
        charge_data = cycle_data[cycle_data['current'] > 0]
        if charge_data.empty:
            return np.array([])
        
        charge_time = charge_data['time'].max() - charge_data['time'].min()
        return np.array([charge_time])
    
    # Register custom features
    analyzer.register_feature_extractor('voltage_plateau_length', extract_voltage_plateau_length)
    analyzer.register_feature_extractor('charge_time', extract_charge_time)
    
    # Define custom statistical measures for time series
    def trend_slope(x):
        """Calculate trend slope using linear regression."""
        if len(x) < 2:
            return np.nan
        
        x_vals = np.arange(len(x))
        slope, _ = np.polyfit(x_vals, x, 1)
        return slope
    
    def stability_index(x):
        """Calculate stability index (inverse of variance)."""
        if len(x) == 0:
            return np.nan
        variance = np.var(x)
        return 1 / variance if variance > 0 else np.nan
    
    # Register custom measures
    analyzer.register_statistical_measure('trend_slope', trend_slope)
    analyzer.register_statistical_measure('stability', stability_index)
    
    # Run analysis
    print("Running specific battery health analysis...")
    
    # Analyze specific dataset
    dataset = 'UL_PUR'
    data = analyzer.load_battery_data(dataset)
    data = analyzer.calculate_rul_labels(data, dataset)
    
    # Define measures for this analysis
    health_measures = ['mean', 'variance', 'trend_slope', 'stability', 'min', 'max']
    
    # Analyze voltage plateau length
    analyzer.plot_correlation_boxplot(data, 'voltage_plateau_length', 100, dataset, 
                                    health_measures, 'voltage_plateau_analysis.png')
    
    # Analyze charge time
    analyzer.plot_correlation_boxplot(data, 'charge_time', 100, dataset, 
                                    health_measures, 'charge_time_analysis.png')
    
    print("Specific analysis completed!")


if __name__ == '__main__':
    print("Feature-RUL Correlation Analysis Examples")
    print("=" * 50)
    
    # Run examples
    print("\n1. Custom Features Example:")
    example_custom_features()
    
    print("\n2. Custom Statistical Measures Example:")
    example_custom_statistical_measures()
    
    print("\n3. Comprehensive Analysis Example:")
    example_comprehensive_analysis()
    
    print("\n4. Specific Analysis Example:")
    example_specific_analysis()
    
    print("\nAll examples completed!")
