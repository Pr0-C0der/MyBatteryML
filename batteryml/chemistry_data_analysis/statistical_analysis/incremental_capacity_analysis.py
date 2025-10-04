#!/usr/bin/env python3
"""
Incremental Capacity Analysis (dQ/dV vs V)

This module provides functionality to plot incremental capacity analysis
for all cycles in a single plot, with separate plots for each battery.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Union
import warnings
from scipy import signal
from scipy.ndimage import gaussian_filter1d

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


def calculate_incremental_capacity(data: pd.DataFrame, 
                                 battery_id: str, 
                                 cycle: int,
                                 window_size: int = 5,
                                 smoothing: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate incremental capacity (dQ/dV) for a specific cycle.
    
    Args:
        data: Battery data DataFrame
        battery_id: ID of the battery
        cycle: Cycle number
        window_size: Window size for smoothing
        smoothing: Whether to apply Gaussian smoothing
        
    Returns:
        Tuple of (voltage, dQ_dV) arrays
    """
    # Get data for this cycle
    cycle_data = data[(data['battery_id'] == battery_id) & (data['cycle'] == cycle)].copy()
    
    if cycle_data.empty:
        return np.array([]), np.array([])
    
    # Sort by time to ensure proper order
    cycle_data = cycle_data.sort_values('time').reset_index(drop=True)
    
    # Calculate cumulative capacity (Ah)
    cycle_data['cumulative_capacity'] = np.cumsum(cycle_data['current'] * cycle_data['time'].diff().fillna(0) / 3600)
    
    # Get voltage and capacity data
    voltage = cycle_data['voltage'].values
    capacity = cycle_data['cumulative_capacity'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(voltage) | np.isnan(capacity))
    voltage = voltage[valid_mask]
    capacity = capacity[valid_mask]
    
    if len(voltage) < 2:
        return np.array([]), np.array([])
    
    # Calculate dQ/dV using finite differences
    dQ = np.diff(capacity)
    dV = np.diff(voltage)
    
    # Avoid division by zero
    valid_dV = dV != 0
    dQ_dV = np.zeros_like(dQ)
    dQ_dV[valid_dV] = dQ[valid_dV] / dV[valid_dV]
    
    # Use midpoints of voltage intervals
    voltage_mid = (voltage[:-1] + voltage[1:]) / 2
    
    # Apply smoothing if requested
    if smoothing and len(dQ_dV) > window_size:
        dQ_dV = gaussian_filter1d(dQ_dV, sigma=window_size/3)
    
    return voltage_mid, dQ_dV


def plot_incremental_capacity_analysis(data: pd.DataFrame, 
                                     dataset_name: str,
                                     battery_id: str,
                                     cycle_limit: int = 10,
                                     output_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (12, 8),
                                     window_size: int = 5,
                                     smoothing: bool = True) -> None:
    """
    Plot incremental capacity analysis (dQ/dV vs V) for all cycles of a specific battery.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        battery_id: ID of the battery to plot
        cycle_limit: Maximum number of cycles to plot
        output_path: Path to save the plot (optional)
        figsize: Figure size
        window_size: Window size for smoothing
        smoothing: Whether to apply Gaussian smoothing
    """
    # Get data for this battery
    battery_data = data[data['battery_id'] == battery_id]
    
    if battery_data.empty:
        raise ValueError(f"No data found for battery {battery_id}")
    
    # Get available cycles for this battery
    available_cycles = sorted(battery_data['cycle'].unique())
    cycles_to_plot = available_cycles[:min(cycle_limit, len(available_cycles))]
    
    print(f"Plotting incremental capacity analysis for battery {battery_id}")
    print(f"Cycles: {cycles_to_plot}")
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Generate colors using RdYlBu_r colormap
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(cycles_to_plot)))
    
    # Plot each cycle
    for i, cycle in enumerate(cycles_to_plot):
        try:
            # Calculate incremental capacity for this cycle
            voltage, dQ_dV = calculate_incremental_capacity(data, battery_id, cycle, window_size, smoothing)
            
            if len(voltage) == 0 or len(dQ_dV) == 0:
                print(f"Warning: No valid data for cycle {cycle}")
                continue
            
            # Remove extreme outliers for better visualization
            if len(dQ_dV) > 0:
                q75, q25 = np.percentile(dQ_dV, [75, 25])
                iqr = q75 - q25
                outlier_threshold = 3 * iqr
                valid_mask = np.abs(dQ_dV - np.median(dQ_dV)) <= outlier_threshold
                
                voltage_clean = voltage[valid_mask]
                dQ_dV_clean = dQ_dV[valid_mask]
                
                if len(voltage_clean) > 0:
                    plt.plot(voltage_clean, dQ_dV_clean, 
                            color=colors[i], 
                            linewidth=2, 
                            label=f'Cycle {cycle}',
                            alpha=0.8)
            
        except Exception as e:
            print(f"Warning: Could not plot cycle {cycle}: {e}")
            continue
    
    # Customize the plot
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('dQ/dV (Ah/V)', fontsize=12)
    plt.title(f'Incremental Capacity Analysis (dQ/dV vs V)\nBattery: {battery_id} | Dataset: {dataset_name}', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    plt.ylim(bottom=0)
    
    # Add some styling
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_batteries_incremental_capacity(data: pd.DataFrame, 
                                          dataset_name: str,
                                          cycle_limit: int = 10,
                                          output_dir: Optional[str] = None,
                                          figsize: Tuple[int, int] = (12, 8),
                                          window_size: int = 5,
                                          smoothing: bool = True) -> None:
    """
    Plot incremental capacity analysis for all batteries in the dataset.
    Creates separate PNG files for each battery.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        cycle_limit: Maximum number of cycles to plot per battery
        output_dir: Directory to save the plots (optional)
        figsize: Figure size
        window_size: Window size for smoothing
        smoothing: Whether to apply Gaussian smoothing
    """
    available_batteries = data['battery_id'].unique()
    
    if len(available_batteries) == 0:
        raise ValueError("No batteries found in data")
    
    print(f"Plotting incremental capacity analysis for {len(available_batteries)} batteries: {list(available_batteries)}")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = f"incremental_capacity_analysis_{dataset_name}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot each battery separately
    for battery_id in available_batteries:
        try:
            battery_output_path = output_path / f"incremental_capacity_analysis_{battery_id}.png"
            plot_incremental_capacity_analysis(data, dataset_name, battery_id, 
                                             cycle_limit, str(battery_output_path), figsize,
                                             window_size, smoothing)
            
        except Exception as e:
            print(f"Error plotting battery {battery_id}: {e}")
            continue
    
    print(f"All battery plots saved to: {output_path}")


def plot_incremental_capacity_charge_discharge(data: pd.DataFrame, 
                                             dataset_name: str,
                                             battery_id: str,
                                             cycle_limit: int = 10,
                                             output_path: Optional[str] = None,
                                             figsize: Tuple[int, int] = (15, 10),
                                             window_size: int = 5,
                                             smoothing: bool = True) -> None:
    """
    Plot incremental capacity analysis separately for charge and discharge phases.
    
    Args:
        data: DataFrame with battery data
        dataset_name: Name of the dataset
        battery_id: ID of the battery to plot
        cycle_limit: Maximum number of cycles to plot
        output_path: Path to save the plot (optional)
        figsize: Figure size
        window_size: Window size for smoothing
        smoothing: Whether to apply Gaussian smoothing
    """
    # Get data for this battery
    battery_data = data[data['battery_id'] == battery_id]
    
    if battery_data.empty:
        raise ValueError(f"No data found for battery {battery_id}")
    
    # Get available cycles for this battery
    available_cycles = sorted(battery_data['cycle'].unique())
    cycles_to_plot = available_cycles[:min(cycle_limit, len(available_cycles))]
    
    print(f"Plotting incremental capacity analysis (charge/discharge) for battery {battery_id}")
    print(f"Cycles: {cycles_to_plot}")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Generate colors using RdYlBu_r colormap
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(cycles_to_plot)))
    
    # Plot each cycle for both charge and discharge
    for i, cycle in enumerate(cycles_to_plot):
        try:
            # Get cycle data
            cycle_data = battery_data[battery_data['cycle'] == cycle].copy()
            cycle_data = cycle_data.sort_values('time').reset_index(drop=True)
            
            # Separate charge and discharge data
            charge_data = cycle_data[cycle_data['current'] > 0].copy()
            discharge_data = cycle_data[cycle_data['current'] < 0].copy()
            
            # Plot charge phase
            if not charge_data.empty:
                charge_data['cumulative_capacity'] = np.cumsum(charge_data['current'] * charge_data['time'].diff().fillna(0) / 3600)
                voltage_charge = charge_data['voltage'].values
                capacity_charge = charge_data['cumulative_capacity'].values
                
                valid_mask = ~(np.isnan(voltage_charge) | np.isnan(capacity_charge))
                voltage_charge = voltage_charge[valid_mask]
                capacity_charge = capacity_charge[valid_mask]
                
                if len(voltage_charge) > 1:
                    dQ = np.diff(capacity_charge)
                    dV = np.diff(voltage_charge)
                    valid_dV = dV != 0
                    dQ_dV = np.zeros_like(dQ)
                    dQ_dV[valid_dV] = dQ[valid_dV] / dV[valid_dV]
                    voltage_mid = (voltage_charge[:-1] + voltage_charge[1:]) / 2
                    
                    if smoothing and len(dQ_dV) > window_size:
                        dQ_dV = gaussian_filter1d(dQ_dV, sigma=window_size/3)
                    
                    # Remove outliers
                    if len(dQ_dV) > 0:
                        q75, q25 = np.percentile(dQ_dV, [75, 25])
                        iqr = q75 - q25
                        outlier_threshold = 3 * iqr
                        valid_mask = np.abs(dQ_dV - np.median(dQ_dV)) <= outlier_threshold
                        
                        voltage_clean = voltage_mid[valid_mask]
                        dQ_dV_clean = dQ_dV[valid_mask]
                        
                        if len(voltage_clean) > 0:
                            ax1.plot(voltage_clean, dQ_dV_clean, 
                                   color=colors[i], linewidth=2, 
                                   label=f'Cycle {cycle}', alpha=0.8)
            
            # Plot discharge phase
            if not discharge_data.empty:
                discharge_data['cumulative_capacity'] = np.cumsum(discharge_data['current'] * discharge_data['time'].diff().fillna(0) / 3600)
                voltage_discharge = discharge_data['voltage'].values
                capacity_discharge = discharge_data['cumulative_capacity'].values
                
                valid_mask = ~(np.isnan(voltage_discharge) | np.isnan(capacity_discharge))
                voltage_discharge = voltage_discharge[valid_mask]
                capacity_discharge = capacity_discharge[valid_mask]
                
                if len(voltage_discharge) > 1:
                    dQ = np.diff(capacity_discharge)
                    dV = np.diff(voltage_discharge)
                    valid_dV = dV != 0
                    dQ_dV = np.zeros_like(dQ)
                    dQ_dV[valid_dV] = dQ[valid_dV] / dV[valid_dV]
                    voltage_mid = (voltage_discharge[:-1] + voltage_discharge[1:]) / 2
                    
                    if smoothing and len(dQ_dV) > window_size:
                        dQ_dV = gaussian_filter1d(dQ_dV, sigma=window_size/3)
                    
                    # Remove outliers
                    if len(dQ_dV) > 0:
                        q75, q25 = np.percentile(dQ_dV, [75, 25])
                        iqr = q75 - q25
                        outlier_threshold = 3 * iqr
                        valid_mask = np.abs(dQ_dV - np.median(dQ_dV)) <= outlier_threshold
                        
                        voltage_clean = voltage_mid[valid_mask]
                        dQ_dV_clean = dQ_dV[valid_mask]
                        
                        if len(voltage_clean) > 0:
                            ax2.plot(voltage_clean, dQ_dV_clean, 
                                   color=colors[i], linewidth=2, 
                                   label=f'Cycle {cycle}', alpha=0.8)
            
        except Exception as e:
            print(f"Warning: Could not plot cycle {cycle}: {e}")
            continue
    
    # Customize subplots
    ax1.set_xlabel('Voltage (V)', fontsize=12)
    ax1.set_ylabel('dQ/dV (Ah/V)', fontsize=12)
    ax1.set_title('Charge Phase', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    ax2.set_xlabel('Voltage (V)', fontsize=12)
    ax2.set_ylabel('dQ/dV (Ah/V)', fontsize=12)
    ax2.set_title('Discharge Phase', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Overall title
    fig.suptitle(f'Incremental Capacity Analysis (dQ/dV vs V)\nBattery: {battery_id} | Dataset: {dataset_name}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to run the incremental capacity analysis."""
    parser = argparse.ArgumentParser(description='Plot incremental capacity analysis (dQ/dV vs V)')
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
    parser.add_argument('--charge_discharge', action='store_true', 
                       help='Plot charge and discharge phases separately')
    parser.add_argument('--window_size', type=int, default=5, 
                       help='Window size for smoothing (default: 5)')
    parser.add_argument('--no_smoothing', action='store_true', 
                       help='Disable Gaussian smoothing')
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
                if args.charge_discharge:
                    output_path = f"incremental_capacity_charge_discharge_{args.dataset_name}"
                else:
                    output_path = f"incremental_capacity_analysis_{args.dataset_name}"
            else:
                if args.charge_discharge:
                    output_path = f"incremental_capacity_charge_discharge_{args.dataset_name}_{args.battery_id or 'single'}.png"
                else:
                    output_path = f"incremental_capacity_analysis_{args.dataset_name}_{args.battery_id or 'single'}.png"
        
        # Generate the plot
        if args.verbose:
            print("Generating plot...")
        
        smoothing = not args.no_smoothing
        
        if args.all:
            if args.charge_discharge:
                # For charge/discharge mode with all batteries, create separate files
                available_batteries = data['battery_id'].unique()
                output_dir = Path(output_path)
                output_dir.mkdir(exist_ok=True)
                
                for battery_id in available_batteries:
                    try:
                        battery_output_path = output_dir / f"incremental_capacity_charge_discharge_{battery_id}.png"
                        plot_incremental_capacity_charge_discharge(data, args.dataset_name, battery_id, 
                                                                 args.cycle_limit, str(battery_output_path), 
                                                                 tuple(args.figsize), args.window_size, smoothing)
                    except Exception as e:
                        print(f"Error plotting battery {battery_id}: {e}")
                        continue
                
                print(f"All battery plots saved to: {output_dir}")
            else:
                plot_all_batteries_incremental_capacity(data, args.dataset_name, args.cycle_limit, 
                                                      output_path, tuple(args.figsize), 
                                                      args.window_size, smoothing)
        else:
            if args.battery_id:
                if args.charge_discharge:
                    plot_incremental_capacity_charge_discharge(data, args.dataset_name, args.battery_id, 
                                                             args.cycle_limit, output_path, 
                                                             tuple(args.figsize), args.window_size, smoothing)
                else:
                    plot_incremental_capacity_analysis(data, args.dataset_name, args.battery_id, 
                                                     args.cycle_limit, output_path, tuple(args.figsize),
                                                     args.window_size, smoothing)
            else:
                # Use first available battery
                available_batteries = data['battery_id'].unique()
                if len(available_batteries) == 0:
                    raise ValueError("No batteries found in data")
                battery_id = available_batteries[0]
                print(f"Using battery: {battery_id}")
                
                if args.charge_discharge:
                    plot_incremental_capacity_charge_discharge(data, args.dataset_name, battery_id, 
                                                             args.cycle_limit, output_path, 
                                                             tuple(args.figsize), args.window_size, smoothing)
                else:
                    plot_incremental_capacity_analysis(data, args.dataset_name, battery_id, 
                                                     args.cycle_limit, output_path, tuple(args.figsize),
                                                     args.window_size, smoothing)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
