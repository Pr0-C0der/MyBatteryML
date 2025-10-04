#!/usr/bin/env python3
"""
Comprehensive Correlation Analysis for BatteryML Chemistry Data.

This script generates correlation plots for all datasets and features,
saving them in a structured directory format:
statistical_correlation_analysis/<dataset_name>/<feature_name>.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings
from tqdm import tqdm

# Import the correlation analyzer
from .feature_rul_correlation import FeatureRULCorrelationAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')

class ComprehensiveCorrelationAnalyzer:
    """Analyzer for comprehensive correlation analysis across all datasets and features."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the comprehensive correlation analyzer.
        
        Args:
            data_dir: Base directory containing the data
        """
        self.data_dir = data_dir
        self.correlation_analyzer = FeatureRULCorrelationAnalyzer(data_dir)
        
        # Available datasets
        self.available_datasets = [
            'UL_PUR', 'MATR', 'CALCE', 'HUST', 'RWTH', 'OX', 'SNL', 'HNEI'
        ]
        
        # Available cycle-level features (from cycle_features.py)
        # These features are properly aggregated across all cycles for each battery
        self.available_features = [
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
        self.statistical_measures = [
            'mean', 'variance', 'median', 'kurtosis', 'skewness', 'min', 'max'
        ]
    
    def analyze_all_combinations(self, 
                               datasets: List[str] = None,
                               features: List[str] = None,
                               cycle_limit: int = None,
                               smoothing_method: str = None,
                               smoothing_window: int = 5,
                               output_dir: str = "statistical_correlation_analysis",
                               figsize: Tuple[int, int] = (12, 8),
                               verbose: bool = True) -> None:
        """
        Analyze correlations for all combinations of datasets and features.
        
        Args:
            datasets: List of datasets to analyze (None for all available)
            features: List of features to analyze (None for all available)
            cycle_limit: Maximum number of cycles to use (None for all cycles)
            smoothing_method: Smoothing method to apply ('hms', 'ma', 'median')
            smoothing_window: Window size for smoothing (ignored for HMS)
            output_dir: Base directory to save plots
            figsize: Figure size for plots
            verbose: Whether to print progress information
        """
        if datasets is None:
            datasets = self.available_datasets
        if features is None:
            features = self.available_features
        
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if verbose:
            print(f"Analyzing {len(datasets)} datasets and {len(features)} features")
            print(f"Total combinations: {len(datasets) * len(features)}")
            print(f"Output directory: {output_path.absolute()}")
            print()
        
        # Track successful and failed analyses
        successful = 0
        failed = 0
        failed_combinations = []
        
        # Process each dataset
        for dataset_name in tqdm(datasets, desc="Processing datasets", unit="dataset"):
            if verbose:
                print(f"\nProcessing dataset: {dataset_name}")
            
            # Create dataset-specific output directory
            dataset_output_dir = output_path / dataset_name
            dataset_output_dir.mkdir(exist_ok=True)
            
            try:
                # Load data for this dataset
                data = self.correlation_analyzer.load_battery_data(dataset_name)
                data = self.correlation_analyzer.calculate_rul_labels(data, dataset_name)
                
                if verbose:
                    print(f"  Loaded {len(data)} records from {data['battery_id'].nunique()} batteries")
                
                # Process each feature for this dataset
                for feature_name in tqdm(features, desc=f"Processing {dataset_name} features", unit="feature", leave=False):
                    try:
                        # Create output path for this combination
                        feature_output_path = dataset_output_dir / f"{feature_name}.png"
                        
                        # Generate correlation plot
                        self.correlation_analyzer.plot_correlation_diverging_bar(
                            data=data,
                            feature_name=feature_name,
                            dataset_name=dataset_name,
                            statistical_measures=self.statistical_measures,
                            cycle_limit=cycle_limit,
                            smoothing_method=smoothing_method,
                            smoothing_window=smoothing_window,
                            output_path=str(feature_output_path),
                            figsize=figsize
                        )
                        
                        successful += 1
                        
                        if verbose and successful % 10 == 0:
                            print(f"    Generated {successful} plots so far...")
                            
                    except Exception as e:
                        failed += 1
                        failed_combinations.append(f"{dataset_name}/{feature_name}: {str(e)}")
                        
                        if verbose:
                            print(f"    Failed {dataset_name}/{feature_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                if verbose:
                    print(f"  Failed to load dataset {dataset_name}: {str(e)}")
                failed += len(features)
                for feature_name in features:
                    failed_combinations.append(f"{dataset_name}/{feature_name}: Dataset loading failed")
                continue
        
        # Print summary
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Successful analyses: {successful}")
        print(f"Failed analyses: {failed}")
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        print(f"Output directory: {output_path.absolute()}")
        
        if failed_combinations and verbose:
            print(f"\nFailed combinations:")
            for combination in failed_combinations[:10]:  # Show first 10 failures
                print(f"  - {combination}")
            if len(failed_combinations) > 10:
                print(f"  ... and {len(failed_combinations) - 10} more")
    
    def analyze_specific_combinations(self, 
                                    combinations: List[Tuple[str, str]],
                                    cycle_limit: int = None,
                                    smoothing_method: str = None,
                                    smoothing_window: int = 5,
                                    output_dir: str = "statistical_correlation_analysis",
                                    figsize: Tuple[int, int] = (12, 8),
                                    verbose: bool = True) -> None:
        """
        Analyze correlations for specific dataset-feature combinations.
        
        Args:
            combinations: List of (dataset_name, feature_name) tuples
            cycle_limit: Maximum number of cycles to use (None for all cycles)
            smoothing_method: Smoothing method to apply ('hms', 'ma', 'median')
            smoothing_window: Window size for smoothing (ignored for HMS)
            output_dir: Base directory to save plots
            figsize: Figure size for plots
            verbose: Whether to print progress information
        """
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if verbose:
            print(f"Analyzing {len(combinations)} specific combinations")
            print(f"Output directory: {output_path.absolute()}")
            print()
        
        # Track successful and failed analyses
        successful = 0
        failed = 0
        failed_combinations = []
        
        # Process each combination
        for dataset_name, feature_name in tqdm(combinations, desc="Processing combinations", unit="combination"):
            try:
                # Create dataset-specific output directory
                dataset_output_dir = output_path / dataset_name
                dataset_output_dir.mkdir(exist_ok=True)
                
                # Load data for this dataset
                data = self.correlation_analyzer.load_battery_data(dataset_name)
                data = self.correlation_analyzer.calculate_rul_labels(data, dataset_name)
                
                # Create output path for this combination
                feature_output_path = dataset_output_dir / f"{feature_name}.png"
                
                # Generate correlation plot
                self.correlation_analyzer.plot_correlation_diverging_bar(
                    data=data,
                    feature_name=feature_name,
                    dataset_name=dataset_name,
                    statistical_measures=self.statistical_measures,
                    cycle_limit=cycle_limit,
                    smoothing_method=smoothing_method,
                    smoothing_window=smoothing_window,
                    output_path=str(feature_output_path),
                    figsize=figsize
                )
                
                successful += 1
                
                if verbose:
                    print(f"✓ {dataset_name}/{feature_name}")
                    
            except Exception as e:
                failed += 1
                failed_combinations.append(f"{dataset_name}/{feature_name}: {str(e)}")
                
                if verbose:
                    print(f"✗ {dataset_name}/{feature_name}: {str(e)}")
                continue
        
        # Print summary
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Successful analyses: {successful}")
        print(f"Failed analyses: {failed}")
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
        print(f"Output directory: {output_path.absolute()}")
        
        if failed_combinations and verbose:
            print(f"\nFailed combinations:")
            for combination in failed_combinations:
                print(f"  - {combination}")


def main():
    """Main function to run comprehensive correlation analysis."""
    parser = argparse.ArgumentParser(description='Comprehensive correlation analysis for all datasets and features')
    
    # Dataset selection
    parser.add_argument('--datasets', nargs='+', 
                       help='Specific datasets to analyze (default: all available)')
    parser.add_argument('--features', nargs='+', 
                       help='Specific features to analyze (default: all available)')
    
    # Analysis parameters
    parser.add_argument('--cycle_limit', type=int, 
                       help='Maximum number of cycles to use (default: all cycles)')
    parser.add_argument('--smoothing', choices=['none', 'hms', 'ma', 'median'], 
                       default='none', help='Smoothing method (default: none)')
    parser.add_argument('--smoothing_window', type=int, default=5, 
                       help='Window size for smoothing (ignored for HMS)')
    
    # Output parameters
    parser.add_argument('--output_dir', default='statistical_correlation_analysis',
                       help='Output directory for plots (default: statistical_correlation_analysis)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    
    # Control parameters
    parser.add_argument('--data_dir', default='data', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Quiet output (opposite of verbose)')
    
    args = parser.parse_args()
    
    # Set verbose based on arguments
    verbose = args.verbose and not args.quiet
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveCorrelationAnalyzer(args.data_dir)
        
        if verbose:
            print("Comprehensive Correlation Analysis")
            print("=" * 40)
            print(f"Available datasets: {', '.join(analyzer.available_datasets)}")
            print(f"Available features: {len(analyzer.available_features)} features")
            print(f"Statistical measures: {', '.join(analyzer.statistical_measures)}")
            print()
        
        # Set smoothing method to None if 'none'
        smoothing_method = None if args.smoothing == 'none' else args.smoothing
        
        # Run analysis
        analyzer.analyze_all_combinations(
            datasets=args.datasets,
            features=args.features,
            cycle_limit=args.cycle_limit,
            smoothing_method=smoothing_method,
            smoothing_window=args.smoothing_window,
            output_dir=args.output_dir,
            figsize=tuple(args.figsize),
            verbose=verbose
        )
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
