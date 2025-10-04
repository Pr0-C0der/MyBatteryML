#!/usr/bin/env python3
"""
Normalized Charge Capacity vs Normalized Voltage Analysis

This module provides functionality to plot normalized charge capacity vs normalized voltage
for multiple cycles within a single plot, using color-coded curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Union
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


def normalize_charge_capacity_voltage(data: pd.DataFrame, battery_id: str) -> pd.DataFrame:
    """
    Normalize charge capacity and voltage for a specific battery.
    
    Args:
        data: Battery data DataFrame
        battery_id: ID of the battery to normalize
        
    Returns:
        DataFrame with normalized capacity and voltage
    """
    battery_data = data[data['battery_id'] == battery_id].copy()
    
    if battery_data.empty:
        raise ValueError(f"No data found for battery {battery_id}")
    
    # Get charge data only (positive current)
    charge_data = battery_data[battery_data['current'] > 0].copy()
    
    if charge_data.empty:
        raise ValueError(f"No charge data found for battery {battery_id}")
    
    # Calculate cumulative capacity (charge capacity)
    charge_data = charge_data.sort_values('time')
    charge_data['cumulative_capacity'] = np.cumsum(charge_data['current'] * charge_data['time'].diff().fillna(0) / 3600)  # Convert to Ah
    
    # Normalize capacity (divide by maximum capacity)
    max_capacity = charge_data['cumulative_capacity'].max()
    charge_data['normalized_capacity'] = charge_data['cumulative_capacity'] / max_capacity
    
    # Normalize voltage (divide by maximum voltage)
    max_voltage = charge_data['voltage'].max()
    charge_data['normalized_voltage'] = charge_data['voltage'] / max_voltage
    
    return charge_data


def plot_normalized_charge_capacity_voltage(data: pd.DataFrame, 
                                          dataset_name: str,
                                          cycle_limit: int = 10,
                                          battery_id: Optional[str] = None,
                                          output_path: Optional[str] = None,
                                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot normalized charge capacity vs normalized voltage for multiple cycles.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        cycle_limit: Maximum number of cycles to plot
        battery_id: Specific battery ID to plot (if None, uses first available)
        output_path: Path to save the plot (optional)
        figsize: Figure size
    """
    # Select battery if not specified
    if battery_id is None:
        available_batteries = data['battery_id'].unique()
        if len(available_batteries) == 0:
            raise ValueError("No batteries found in data")
        battery_id = available_batteries[0]
        print(f"Using battery: {battery_id}")
    
    # Get available cycles for this battery
    battery_data = data[data['battery_id'] == battery_id]
    available_cycles = sorted(battery_data['cycle'].unique())
    
    if len(available_cycles) == 0:
        raise ValueError(f"No cycles found for battery {battery_id}")
    
    # Limit cycles to specified number
    cycles_to_plot = available_cycles[:min(cycle_limit, len(available_cycles))]
    
    print(f"Plotting {len(cycles_to_plot)} cycles for battery {battery_id}")
    print(f"Cycles: {cycles_to_plot}")
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Generate colors using RdYlBu_r colormap
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(cycles_to_plot)))
    
    # Plot each cycle
    for i, cycle in enumerate(cycles_to_plot):
        try:
            # Get data for this cycle
            cycle_data = battery_data[battery_data['cycle'] == cycle]
            
            if cycle_data.empty:
                print(f"Warning: No data for cycle {cycle}")
                continue
            
            # Normalize capacity and voltage for this cycle
            normalized_data = normalize_charge_capacity_voltage(cycle_data, battery_id)
            
            if normalized_data.empty:
                print(f"Warning: No charge data for cycle {cycle}")
                continue
            
            # Plot normalized voltage vs normalized capacity
            plt.plot(normalized_data['normalized_voltage'], 
                    normalized_data['normalized_capacity'],
                    color=colors[i], 
                    linewidth=2, 
                    label=f'Cycle {cycle}',
                    alpha=0.8)
            
        except Exception as e:
            print(f"Warning: Could not plot cycle {cycle}: {e}")
            continue
    
    # Customize the plot
    plt.xlabel('Normalized Voltage', fontsize=12)
    plt.ylabel('Normalized Charge Capacity', fontsize=12)
    plt.title(f'Normalized Voltage vs Normalized Charge Capacity\nBattery: {battery_id} | Dataset: {dataset_name}', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add some styling
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_batteries(data: pd.DataFrame, 
                          dataset_name: str,
                          cycle_limit: int = 5,
                          max_batteries: int = 3,
                          output_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot normalized charge capacity vs voltage for multiple batteries.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        cycle_limit: Maximum number of cycles to plot per battery
        max_batteries: Maximum number of batteries to plot
        output_path: Path to save the plot (optional)
        figsize: Figure size
    """
    available_batteries = data['battery_id'].unique()[:max_batteries]
    
    if len(available_batteries) == 0:
        raise ValueError("No batteries found in data")
    
    print(f"Plotting {len(available_batteries)} batteries: {available_batteries}")
    
    # Create subplots
    fig, axes = plt.subplots(1, len(available_batteries), figsize=figsize)
    if len(available_batteries) == 1:
        axes = [axes]
    
    for battery_idx, battery_id in enumerate(available_batteries):
        ax = axes[battery_idx]
        
        # Get available cycles for this battery
        battery_data = data[data['battery_id'] == battery_id]
        available_cycles = sorted(battery_data['cycle'].unique())
        cycles_to_plot = available_cycles[:min(cycle_limit, len(available_cycles))]
        
        # Generate colors using RdYlBu_r colormap
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(cycles_to_plot)))
        
        # Plot each cycle
        for i, cycle in enumerate(cycles_to_plot):
            try:
                cycle_data = battery_data[battery_data['cycle'] == cycle]
                normalized_data = normalize_charge_capacity_voltage(cycle_data, battery_id)
                
                if not normalized_data.empty:
                    ax.plot(normalized_data['normalized_voltage'], 
                           normalized_data['normalized_capacity'],
                           color=colors[i], 
                           linewidth=2, 
                           label=f'Cycle {cycle}',
                           alpha=0.8)
            except Exception as e:
                print(f"Warning: Could not plot cycle {cycle} for battery {battery_id}: {e}")
                continue
        
        # Customize subplot
        ax.set_xlabel('Normalized Voltage', fontsize=10)
        ax.set_ylabel('Normalized Charge Capacity', fontsize=10)
        ax.set_title(f'Battery {battery_id}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Overall title
    fig.suptitle(f'Normalized Voltage vs Normalized Charge Capacity\nDataset: {dataset_name}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_batteries(data: pd.DataFrame, 
                      dataset_name: str,
                      cycle_limit: int = 5,
                      output_dir: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot normalized voltage vs charge capacity for all batteries in the dataset.
    Creates separate PNG files for each battery.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        cycle_limit: Maximum number of cycles to plot per battery
        output_dir: Directory to save the plots (optional)
        figsize: Figure size
    """
    available_batteries = data['battery_id'].unique()
    
    if len(available_batteries) == 0:
        raise ValueError("No batteries found in data")
    
    print(f"Plotting {len(available_batteries)} batteries: {list(available_batteries)}")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = f"normalized_charge_capacity_voltage_all_{dataset_name}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot each battery separately
    for battery_id in available_batteries:
        try:
            # Get available cycles for this battery
            battery_data = data[data['battery_id'] == battery_id]
            available_cycles = sorted(battery_data['cycle'].unique())
            cycles_to_plot = available_cycles[:min(cycle_limit, len(available_cycles))]
            
            # Create individual plot for this battery
            plt.figure(figsize=figsize)
            
            # Generate colors using RdYlBu_r colormap
            colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(cycles_to_plot)))
            
            # Plot each cycle
            for i, cycle in enumerate(cycles_to_plot):
                try:
                    cycle_data = battery_data[battery_data['cycle'] == cycle]
                    normalized_data = normalize_charge_capacity_voltage(cycle_data, battery_id)
                    
                    if not normalized_data.empty:
                        plt.plot(normalized_data['normalized_voltage'], 
                               normalized_data['normalized_capacity'],
                               color=colors[i], 
                               linewidth=2, 
                               label=f'Cycle {cycle}',
                               alpha=0.8)
                except Exception as e:
                    print(f"Warning: Could not plot cycle {cycle} for battery {battery_id}: {e}")
                    continue
            
            # Customize the plot
            plt.xlabel('Normalized Voltage', fontsize=12)
            plt.ylabel('Normalized Charge Capacity', fontsize=12)
            plt.title(f'Normalized Voltage vs Normalized Charge Capacity\nBattery: {battery_id} | Dataset: {dataset_name}', 
                     fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            # Save individual plot
            battery_output_path = output_path / f"normalized_charge_capacity_voltage_{battery_id}.png"
            plt.tight_layout()
            plt.savefig(battery_output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Saved plot for battery {battery_id}: {battery_output_path}")
            
        except Exception as e:
            print(f"Error plotting battery {battery_id}: {e}")
            continue
    
    print(f"All battery plots saved to: {output_path}")


def main():
    """Main function to run the normalized charge capacity vs voltage analysis."""
    parser = argparse.ArgumentParser(description='Plot normalized charge capacity vs normalized voltage')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--cycle_limit', '-c', type=int, default=10, 
                       help='Maximum number of cycles to plot (default: 10)')
    parser.add_argument('--battery_id', '-b', help='Specific battery ID to plot (optional)')
    parser.add_argument('--data_dir', '-d', default='data', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--multiple_batteries', action='store_true', 
                       help='Plot multiple batteries in subplots')
    parser.add_argument('--max_batteries', type=int, default=3, 
                       help='Maximum number of batteries to plot (default: 3)')
    parser.add_argument('--all', action='store_true', 
                       help='Plot all batteries in the dataset')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        if args.verbose:
            print(f"Loading data for dataset: {args.dataset_name}")
            print(f"Cycle limit: {args.cycle_limit}")
        
        # Load battery data
        data = load_battery_data(args.dataset_name, args.data_dir)
        
        if args.verbose:
            print(f"Loaded {len(data)} records from {data['battery_id'].nunique()} batteries")
        
        # Create output path if not provided
        output_path = args.output
        if output_path is None:
            if args.all:
                output_path = f"normalized_charge_capacity_voltage_all_{args.dataset_name}.png"
            elif args.multiple_batteries:
                output_path = f"normalized_charge_capacity_voltage_multiple_{args.dataset_name}.png"
            else:
                output_path = f"normalized_charge_capacity_voltage_{args.dataset_name}.png"
        
        # Generate the plot
        if args.verbose:
            print("Generating plot...")
        
        if args.all:
            plot_all_batteries(data, args.dataset_name, args.cycle_limit, 
                             output_path, tuple(args.figsize))
        elif args.multiple_batteries:
            plot_multiple_batteries(data, args.dataset_name, args.cycle_limit, 
                                  args.max_batteries, output_path, tuple(args.figsize))
        else:
            plot_normalized_charge_capacity_voltage(data, args.dataset_name, args.cycle_limit, 
                                                  args.battery_id, output_path, tuple(args.figsize))
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
