#!/usr/bin/env python3
"""
Voltage Frequency Distribution Analysis

This module provides functionality to plot frequency distribution of voltage
for all cycles in a single plot, with separate plots for each battery.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Union
import warnings
from scipy import stats

# Import necessary modules from the parent package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from batteryml.data.battery_data import BatteryData


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
    
    # Load all battery files in the dataset (look for .pkl files)
    battery_files = list(data_path.glob("*.pkl"))
    if not battery_files:
        raise FileNotFoundError(f"No PKL files found in {data_path}")
    
    all_cycle_data = []
    
    for file_path in battery_files:
        try:
            # Load battery data using BatteryData.load()
            battery = BatteryData.load(file_path)
            
            # Extract cycle data for this battery
            for cycle in battery.cycle_data:
                cycle_data = []
                for i in range(len(cycle.voltage_in_V)):
                    cycle_data.append({
                        'battery_id': battery.cell_id,
                        'cycle': cycle.cycle_number,
                        'time': cycle.time_in_s[i] if cycle.time_in_s is not None else i,
                        'voltage': cycle.voltage_in_V[i] if cycle.voltage_in_V is not None else np.nan,
                        'current': cycle.current_in_A[i] if cycle.current_in_A is not None else np.nan,
                        'temperature': cycle.temperature_in_C[i] if cycle.temperature_in_C is not None else np.nan,
                    })
                
                if cycle_data:
                    all_cycle_data.extend(cycle_data)
                    
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    if not all_cycle_data:
        raise ValueError(f"No valid battery data found in {data_path}")
    
    return pd.DataFrame(all_cycle_data)


def plot_voltage_frequency_distribution(data: pd.DataFrame, 
                                      dataset_name: str,
                                      battery_id: str,
                                      cycle_limit: int = 10,
                                      output_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot frequency distribution of voltage for all cycles of a specific battery.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        battery_id: ID of the battery to plot
        cycle_limit: Maximum number of cycles to plot
        output_path: Path to save the plot (optional)
        figsize: Figure size
    """
    # Get data for this battery
    battery_data = data[data['battery_id'] == battery_id]
    
    if battery_data.empty:
        raise ValueError(f"No data found for battery {battery_id}")
    
    # Get available cycles for this battery
    available_cycles = sorted(battery_data['cycle'].unique())
    cycles_to_plot = available_cycles[:min(cycle_limit, len(available_cycles))]
    
    print(f"Plotting voltage frequency distribution for battery {battery_id}")
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
            
            # Get voltage data for this cycle
            voltage_data = cycle_data['voltage'].dropna()
            
            if len(voltage_data) == 0:
                print(f"Warning: No voltage data for cycle {cycle}")
                continue
            
            # Create histogram for frequency distribution
            plt.hist(voltage_data, 
                    bins=50, 
                    alpha=0.6, 
                    color=colors[i], 
                    label=f'Cycle {cycle}',
                    density=True,  # Normalize to show frequency density
                    edgecolor='black',
                    linewidth=0.5)
            
        except Exception as e:
            print(f"Warning: Could not plot cycle {cycle}: {e}")
            continue
    
    # Customize the plot
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Frequency Density', fontsize=12)
    plt.title(f'Voltage Frequency Distribution - All Cycles\nBattery: {battery_id} | Dataset: {dataset_name}', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
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


def plot_all_batteries_voltage_distribution(data: pd.DataFrame, 
                                          dataset_name: str,
                                          cycle_limit: int = 10,
                                          output_dir: Optional[str] = None,
                                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot voltage frequency distribution for all batteries in the dataset.
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
    
    print(f"Plotting voltage frequency distribution for {len(available_batteries)} batteries: {list(available_batteries)}")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = f"voltage_frequency_distribution_{dataset_name}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot each battery separately
    for battery_id in available_batteries:
        try:
            battery_output_path = output_path / f"voltage_frequency_distribution_{battery_id}.png"
            plot_voltage_frequency_distribution(data, dataset_name, battery_id, 
                                              cycle_limit, str(battery_output_path), figsize)
            
        except Exception as e:
            print(f"Error plotting battery {battery_id}: {e}")
            continue
    
    print(f"All battery plots saved to: {output_path}")


def plot_voltage_distribution_kde(data: pd.DataFrame, 
                                dataset_name: str,
                                battery_id: str,
                                cycle_limit: int = 10,
                                output_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot voltage frequency distribution using KDE (Kernel Density Estimation) for smoother curves.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        battery_id: ID of the battery to plot
        cycle_limit: Maximum number of cycles to plot
        output_path: Path to save the plot (optional)
        figsize: Figure size
    """
    # Get data for this battery
    battery_data = data[data['battery_id'] == battery_id]
    
    if battery_data.empty:
        raise ValueError(f"No data found for battery {battery_id}")
    
    # Get available cycles for this battery
    available_cycles = sorted(battery_data['cycle'].unique())
    cycles_to_plot = available_cycles[:min(cycle_limit, len(available_cycles))]
    
    print(f"Plotting voltage KDE distribution for battery {battery_id}")
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
            
            # Get voltage data for this cycle
            voltage_data = cycle_data['voltage'].dropna()
            
            if len(voltage_data) == 0:
                print(f"Warning: No voltage data for cycle {cycle}")
                continue
            
            # Create KDE for smooth distribution curve
            kde = stats.gaussian_kde(voltage_data)
            voltage_range = np.linspace(voltage_data.min(), voltage_data.max(), 200)
            density = kde(voltage_range)
            
            plt.plot(voltage_range, density, 
                    color=colors[i], 
                    linewidth=2, 
                    label=f'Cycle {cycle}',
                    alpha=0.8)
            
        except Exception as e:
            print(f"Warning: Could not plot cycle {cycle}: {e}")
            continue
    
    # Customize the plot
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Voltage Distribution (KDE) - All Cycles\nBattery: {battery_id} | Dataset: {dataset_name}', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
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


def plot_all_batteries_voltage_kde(data: pd.DataFrame, 
                                 dataset_name: str,
                                 cycle_limit: int = 10,
                                 output_dir: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot voltage KDE distribution for all batteries in the dataset.
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
    
    print(f"Plotting voltage KDE distribution for {len(available_batteries)} batteries: {list(available_batteries)}")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = f"voltage_kde_distribution_{dataset_name}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot each battery separately
    for battery_id in available_batteries:
        try:
            battery_output_path = output_path / f"voltage_kde_distribution_{battery_id}.png"
            plot_voltage_distribution_kde(data, dataset_name, battery_id, 
                                        cycle_limit, str(battery_output_path), figsize)
            
        except Exception as e:
            print(f"Error plotting battery {battery_id}: {e}")
            continue
    
    print(f"All battery plots saved to: {output_path}")


def main():
    """Main function to run the voltage frequency distribution analysis."""
    parser = argparse.ArgumentParser(description='Plot voltage frequency distribution for all cycles')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., UL_PUR, MATR, CALCE)')
    parser.add_argument('--cycle_limit', '-c', type=int, default=10, 
                       help='Maximum number of cycles to plot (default: 10)')
    parser.add_argument('--battery_id', '-b', help='Specific battery ID to plot (optional)')
    parser.add_argument('--data_dir', '-d', default='data', 
                       help='Base directory containing the data (default: data)')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], 
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--all', action='store_true', 
                       help='Plot all batteries in the dataset')
    parser.add_argument('--kde', action='store_true', 
                       help='Use KDE (Kernel Density Estimation) for smoother curves')
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
                if args.kde:
                    output_path = f"voltage_kde_distribution_{args.dataset_name}"
                else:
                    output_path = f"voltage_frequency_distribution_{args.dataset_name}"
            else:
                if args.kde:
                    output_path = f"voltage_kde_distribution_{args.dataset_name}_{args.battery_id or 'single'}.png"
                else:
                    output_path = f"voltage_frequency_distribution_{args.dataset_name}_{args.battery_id or 'single'}.png"
        
        # Generate the plot
        if args.verbose:
            print("Generating plot...")
        
        if args.all:
            if args.kde:
                plot_all_batteries_voltage_kde(data, args.dataset_name, args.cycle_limit, 
                                             output_path, tuple(args.figsize))
            else:
                plot_all_batteries_voltage_distribution(data, args.dataset_name, args.cycle_limit, 
                                                      output_path, tuple(args.figsize))
        else:
            if args.battery_id:
                if args.kde:
                    plot_voltage_distribution_kde(data, args.dataset_name, args.battery_id, 
                                                args.cycle_limit, output_path, tuple(args.figsize))
                else:
                    plot_voltage_frequency_distribution(data, args.dataset_name, args.battery_id, 
                                                      args.cycle_limit, output_path, tuple(args.figsize))
            else:
                # Use first available battery
                available_batteries = data['battery_id'].unique()
                if len(available_batteries) == 0:
                    raise ValueError("No batteries found in data")
                battery_id = available_batteries[0]
                print(f"Using battery: {battery_id}")
                
                if args.kde:
                    plot_voltage_distribution_kde(data, args.dataset_name, battery_id, 
                                                args.cycle_limit, output_path, tuple(args.figsize))
                else:
                    plot_voltage_frequency_distribution(data, args.dataset_name, battery_id, 
                                                      args.cycle_limit, output_path, tuple(args.figsize))
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
