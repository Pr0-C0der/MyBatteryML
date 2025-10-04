#!/usr/bin/env python3
"""
Feature vs RUL Analysis

This module provides functionality to analyze how features change over cycles
for batteries with different RUL values (highest, lowest, and median RUL).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

# Import necessary modules from the parent package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from batteryml.chemistry_data_analysis.cycle_features import extract_cycle_features
from batteryml.label.rul import RULLabelAnnotator
from batteryml.preprocess.base import BatteryPreprocessor


def load_battery_data(dataset_name: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Load battery data for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'UL_PUR', 'MATR', 'CALCE')
        data_dir: Base directory containing the data
        
    Returns:
        DataFrame containing battery data
    """
    data_path = Path(data_dir) / dataset_name
    
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


def calculate_rul_labels(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
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


def find_battery_rul_stats(data: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Find battery IDs with highest, lowest, and median RUL values.
    
    Args:
        data: DataFrame with RUL labels
        
    Returns:
        Tuple of (highest_rul_battery, lowest_rul_battery, median_rul_battery)
    """
    # Get RUL values for each battery (use first cycle's RUL as representative)
    battery_rul = data.groupby('battery_id')['rul'].first().sort_values(ascending=False)
    
    if len(battery_rul) < 3:
        raise ValueError("Need at least 3 batteries to find highest, lowest, and median RUL")
    
    highest_rul_battery = battery_rul.index[0]
    lowest_rul_battery = battery_rul.index[-1]
    median_rul_battery = battery_rul.index[len(battery_rul) // 2]
    
    return highest_rul_battery, lowest_rul_battery, median_rul_battery


def plot_feature_vs_cycle(data: pd.DataFrame, 
                         feature: str, 
                         dataset_name: str,
                         output_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot feature vs cycle for batteries with different RUL values.
    
    Args:
        data: DataFrame with battery data and RUL labels
        feature: Feature name to plot
        dataset_name: Name of the dataset
        output_path: Path to save the plot (optional)
        figsize: Figure size
    """
    # Find batteries with different RUL values
    try:
        highest_battery, lowest_battery, median_battery = find_battery_rul_stats(data)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Get RUL values for reference
    battery_rul = data.groupby('battery_id')['rul'].first()
    highest_rul = battery_rul[highest_battery]
    lowest_rul = battery_rul[lowest_battery]
    median_rul = battery_rul[median_battery]
    
    print(f"Battery with highest RUL: {highest_battery} (RUL = {highest_rul:.1f})")
    print(f"Battery with lowest RUL: {lowest_battery} (RUL = {lowest_rul:.1f})")
    print(f"Battery with median RUL: {median_battery} (RUL = {median_rul:.1f})")
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot for highest RUL battery (green)
    highest_data = data[data['battery_id'] == highest_battery].sort_values('cycle')
    if feature in highest_data.columns:
        plt.plot(highest_data['cycle'], highest_data[feature], 
                color='green', linewidth=2, label=f'Highest RUL ({highest_battery}, RUL={highest_rul:.1f})')
    
    # Plot for lowest RUL battery (red)
    lowest_data = data[data['battery_id'] == lowest_battery].sort_values('cycle')
    if feature in lowest_data.columns:
        plt.plot(lowest_data['cycle'], lowest_data[feature], 
                color='red', linewidth=2, label=f'Lowest RUL ({lowest_battery}, RUL={lowest_rul:.1f})')
    
    # Plot for median RUL battery (blue)
    median_data = data[data['battery_id'] == median_battery].sort_values('cycle')
    if feature in median_data.columns:
        plt.plot(median_data['cycle'], median_data[feature], 
                color='blue', linewidth=2, label=f'Median RUL ({median_battery}, RUL={median_rul:.1f})')
    
    # Customize the plot
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel(f'{feature.replace("_", " ").title()}', fontsize=12)
    plt.title(f'{feature.replace("_", " ").title()} vs Cycle for Different RUL Batteries\nDataset: {dataset_name}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to run the feature vs RUL analysis."""
    parser = argparse.ArgumentParser(description='Analyze feature vs cycle for batteries with different RUL values')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--feature', '-f', default='charge_cycle_length', 
                       help='Feature to analyze (default: charge_cycle_length)')
    parser.add_argument('--data_dir', '-d', default='data', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.verbose:
            print(f"Loading data for dataset: {args.dataset_name}")
            print(f"Analyzing feature: {args.feature}")
        
        # Load battery data
        data = load_battery_data(args.dataset_name, args.data_dir)
        
        if args.verbose:
            print(f"Loaded {len(data)} records from {data['battery_id'].nunique()} batteries")
        
        # Calculate RUL labels
        if args.verbose:
            print("Calculating RUL labels...")
        data_with_rul = calculate_rul_labels(data, args.dataset_name)
        
        # Check if feature exists
        if args.feature not in data_with_rul.columns:
            available_features = [col for col in data_with_rul.columns if col not in ['battery_id', 'cycle', 'rul']]
            print(f"Error: Feature '{args.feature}' not found in data.")
            print(f"Available features: {available_features[:10]}...")  # Show first 10 features
            return
        
        # Create output path if not provided
        output_path = args.output
        if output_path is None:
            output_path = f"feature_vs_rul_{args.dataset_name}_{args.feature}.png"
        
        # Generate the plot
        if args.verbose:
            print("Generating plot...")
        plot_feature_vs_cycle(data_with_rul, args.feature, args.dataset_name, 
                             output_path, tuple(args.figsize))
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
