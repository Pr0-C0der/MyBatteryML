#!/usr/bin/env python3
"""
Plot max discharge capacity vs cycle for each dataset (one random battery per dataset).
Uses sober color palette and clean styling.
"""

import sys
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from batteryml.data.battery_data import BatteryData
from batteryml.chemistry_data_analysis.cycle_features import get_extractor_class, BaseCycleFeatures


def get_available_datasets() -> List[Tuple[str, Path]]:
    """Get list of available datasets and their paths."""
    datasets = []
    
    # Check processed data directory
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for dataset_dir in processed_dir.iterdir():
            if dataset_dir.is_dir() and any(dataset_dir.glob("*.pkl")):
                datasets.append((dataset_dir.name, dataset_dir))
    
    # Check datasets_requiring_access
    access_dir = Path("data/datasets_requiring_access")
    if access_dir.exists():
        for dataset_dir in access_dir.iterdir():
            if dataset_dir.is_dir() and any(dataset_dir.glob("*.pkl")):
                datasets.append((dataset_dir.name, dataset_dir))
    
    return datasets


def get_random_battery(dataset_path: Path) -> Optional[BatteryData]:
    """Get a random battery from the dataset."""
    pkl_files = list(dataset_path.glob("*.pkl"))
    if not pkl_files:
        return None
    
    # Pick a random file
    random_file = random.choice(pkl_files)
    try:
        return BatteryData.load(random_file)
    except Exception as e:
        print(f"Error loading {random_file}: {e}")
        return None


def apply_smoothing(data: np.ndarray, method: str = 'none', window: int = 5) -> np.ndarray:
    """Apply smoothing to the data."""
    if method == 'none' or window <= 1:
        return data
    
    if method == 'ma':  # Moving average
        return moving_average(data, window)
    elif method == 'median':  # Moving median
        return moving_median(data, window)
    elif method == 'hms':  # Hampel filter + Moving average + Savitzky-Golay
        # First apply Hampel filter
        filtered = hampel_filter(data, window=window)
        # Then moving average
        smoothed = moving_average(filtered, window)
        # Finally Savitzky-Golay if enough points
        if len(smoothed) > window * 2:
            try:
                from scipy.signal import savgol_filter
                return savgol_filter(smoothed, min(window * 2 + 1, len(smoothed) - 1), 3)
            except ImportError:
                return smoothed
        return smoothed
    else:
        return data


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing."""
    if window <= 1 or len(data) == 0:
        return data
    
    try:
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        
        w = int(window)
        kernel = np.ones(w, dtype=float)
        # Handle NaNs robustly
        mask = np.isfinite(arr).astype(float)
        arr0 = np.nan_to_num(arr, nan=0.0)
        num = np.convolve(arr0, kernel, mode='same')
        den = np.convolve(mask, kernel, mode='same')
        out = num / np.maximum(den, 1e-8)
        out[den < 1e-8] = np.nan
        return out
    except Exception:
        return data


def moving_median(data: np.ndarray, window: int) -> np.ndarray:
    """Apply moving median smoothing."""
    if window <= 1 or len(data) == 0:
        return data
    
    try:
        from scipy.ndimage import median_filter
        return median_filter(data, size=window, mode='nearest')
    except ImportError:
        # Fallback to manual implementation
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            window_data = data[start:end]
            valid_data = window_data[np.isfinite(window_data)]
            if len(valid_data) > 0:
                result[i] = np.median(valid_data)
        return result


def hampel_filter(data: np.ndarray, window: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """Apply Hampel filter to remove outliers."""
    if window <= 1 or len(data) == 0:
        return data
    
    try:
        result = data.copy()
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            window_data = data[start:end]
            valid_data = window_data[np.isfinite(window_data)]
            
            if len(valid_data) > 2:
                median = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median))
                if mad > 0:
                    threshold = n_sigma * 1.4826 * mad  # 1.4826 is a constant for normal distribution
                    if abs(data[i] - median) > threshold:
                        result[i] = median
        
        return result
    except Exception:
        return data


def extract_max_discharge_capacity_data(battery: BatteryData, dataset_name: str, 
                                      smoothing: str = 'none', window: int = 5,
                                      normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Extract max discharge capacity vs cycle data for a battery."""
    # Get the appropriate extractor
    extractor_class = get_extractor_class(dataset_name)
    if extractor_class is None:
        extractor = BaseCycleFeatures()
    else:
        extractor = extractor_class()
    
    cycles = []
    capacities = []
    
    for cycle in battery.cycle_data:
        try:
            capacity = extractor.max_discharge_capacity(battery, cycle)
            if capacity is not None and np.isfinite(capacity):
                cycles.append(cycle.cycle_number)
                capacities.append(capacity)
        except Exception:
            continue
    
    cycles = np.array(cycles)
    capacities = np.array(capacities)
    
    # Apply smoothing if requested
    if smoothing != 'none' and len(capacities) > 0:
        capacities = apply_smoothing(capacities, smoothing, window)
    
    # Apply normalization if requested
    if normalize and len(capacities) > 0:
        # Normalize to first value (capacity retention)
        first_capacity = capacities[0]
        if first_capacity > 0:
            capacities = (capacities / first_capacity) * 100  # Convert to percentage
    
    return cycles, capacities


def plot_max_discharge_capacity_by_dataset(output_dir: str = "max_discharge_capacity_plots", 
                                          figsize: Tuple[int, int] = (12, 8),
                                          smoothing: str = 'none',
                                          window: int = 5,
                                          normalize: bool = False,
                                          verbose: bool = False):
    """Plot max discharge capacity vs cycle for each dataset."""
    
    # Set up sober color palette
    plt.style.use('default')
    sns.set_palette('husl')
    
    # Sober color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7', '#F77F00', '#FCBF49']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get available datasets
    datasets = get_available_datasets()
    if not datasets:
        print("No datasets found!")
        return
    
    if verbose:
        print(f"Found {len(datasets)} datasets: {[d[0] for d in datasets]}")
    
    # Create the main plot
    plt.figure(figsize=figsize)
    
    plotted_datasets = []
    
    for i, (dataset_name, dataset_path) in enumerate(datasets):
        if verbose:
            print(f"Processing dataset: {dataset_name}")
        
        # Get random battery
        battery = get_random_battery(dataset_path)
        if battery is None:
            if verbose:
                print(f"  No valid battery found for {dataset_name}")
            continue
        
        if verbose:
            print(f"  Using battery: {battery.cell_id}")
        
        # Extract data
        cycles, capacities = extract_max_discharge_capacity_data(battery, dataset_name, smoothing, window, normalize)
        
        if len(cycles) == 0:
            if verbose:
                print(f"  No valid capacity data for {dataset_name}")
            continue
        
        # Plot the data
        color = colors[i % len(colors)]
        plt.plot(cycles, capacities, 
                marker='o', 
                linewidth=1.5, 
                markersize=3,
                alpha=0.8,
                color=color,
                label=f"{dataset_name} ({battery.cell_id})")
        
        plotted_datasets.append(dataset_name)
        
        if verbose:
            print(f"  Plotted {len(cycles)} data points")
    
    if not plotted_datasets:
        print("No data to plot!")
        return
    
    # Customize the plot
    title = 'Max Discharge Capacity vs Cycle Number by Dataset'
    if smoothing != 'none':
        title += f' (Smoothed: {smoothing.upper()}, window={window})'
    if normalize:
        title += ' (Normalized)'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Cycle Number', fontsize=12)
    ylabel = 'Max Discharge Capacity (%)' if normalize else 'Max Discharge Capacity (Ah)'
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(False)  # Remove grid
    plt.legend(frameon=False, fontsize=10)  # Clean legend without frame
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = output_path / "max_discharge_capacity_by_dataset.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_file}")
    print(f"Plotted data for {len(plotted_datasets)} datasets: {plotted_datasets}")


def plot_individual_datasets(output_dir: str = 'max_discharge_capacity_plots', 
                            figsize: Tuple[int, int] = (10, 6),
                            smoothing: str = 'none',
                            window: int = 5,
                            normalize: bool = False,
                            verbose: bool = False):
    """Plot individual max discharge capacity plots for each dataset."""
    
    # Set up sober color palette
    plt.style.use('default')
    sns.set_palette('husl')
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get available datasets
    datasets = get_available_datasets()
    if not datasets:
        print("No datasets found!")
        return
    
    for dataset_name, dataset_path in datasets:
        if verbose:
            print(f"Processing dataset: {dataset_name}")
        
        # Get random battery
        battery = get_random_battery(dataset_path)
        if battery is None:
            if verbose:
                print(f"  No valid battery found for {dataset_name}")
            continue
        
        if verbose:
            print(f"  Using battery: {battery.cell_id}")
        
        # Extract data
        cycles, capacities = extract_max_discharge_capacity_data(battery, dataset_name, smoothing, window, normalize)
        
        if len(cycles) == 0:
            if verbose:
                print(f"  No valid capacity data for {dataset_name}")
            continue
        
        # Create individual plot
        plt.figure(figsize=figsize)
        plt.plot(cycles, capacities, 
                marker='o', 
                linewidth=1.5, 
                markersize=3,
                alpha=0.8,
                color='#2E86AB')
        
        title = f'Max Discharge Capacity vs Cycle - {dataset_name} ({battery.cell_id})'
        if smoothing != 'none':
            title += f' (Smoothed: {smoothing.upper()}, window={window})'
        if normalize:
            title += ' (Normalized)'
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Cycle Number', fontsize=12)
        ylabel = 'Max Discharge Capacity (%)' if normalize else 'Max Discharge Capacity (Ah)'
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(False)  # Remove grid
        
        # Adjust layout
        plt.tight_layout()
        
        # Save individual plot
        safe_name = dataset_name.replace('/', '_').replace('\\', '_')
        output_file = output_path / f"max_discharge_capacity_{safe_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"  Individual plot saved to: {output_file}")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot max discharge capacity vs cycle for each dataset')
    parser.add_argument('--output_dir', type=str, default='max_discharge_capacity_plots',
                       help='Output directory for plots')
    parser.add_argument('--individual', action='store_true',
                       help='Create individual plots for each dataset')
    parser.add_argument('--combined', action='store_true', default=True,
                       help='Create combined plot for all datasets')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                       help='Figure size (width height)')
    parser.add_argument('--smoothing', type=str, default='none',
                       choices=['none', 'ma', 'median', 'hms'],
                       help='Smoothing method (none, ma, median, hms) [default: none]')
    parser.add_argument('--window', type=int, default=5,
                       help='Smoothing window size [default: 5]')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize capacity to percentage (capacity retention)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    if args.combined:
        print("Creating combined plot...")
        plot_max_discharge_capacity_by_dataset(
            output_dir=args.output_dir,
            figsize=tuple(args.figsize),
            smoothing=args.smoothing,
            window=args.window,
            normalize=args.normalize,
            verbose=args.verbose
        )
    
    if args.individual:
        print("Creating individual plots...")
        plot_individual_datasets(
            output_dir=args.output_dir,
            figsize=tuple(args.figsize),
            smoothing=args.smoothing,
            window=args.window,
            normalize=args.normalize,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()
