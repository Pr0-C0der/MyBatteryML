# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
from tqdm import tqdm

from batteryml.data.battery_data import BatteryData


def load_battery_data(data_path: str) -> List[BatteryData]:
    """
    Load battery data from a directory containing pickle files.
    
    Args:
        data_path: Path to directory containing .pkl files
        
    Returns:
        List of BatteryData objects
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    battery_files = list(data_path.glob("*.pkl"))
    if not battery_files:
        raise ValueError(f"No .pkl files found in {data_path}")
    
    batteries = []
    failed_files = []
    
    print(f"Loading {len(battery_files)} battery data files...")
    for file_path in tqdm(battery_files, desc="Loading battery data", unit="files"):
        try:
            with open(file_path, 'rb') as f:
                battery = BatteryData.load(file_path)
                batteries.append(battery)
        except Exception as e:
            failed_files.append((file_path, str(e)))
            warnings.warn(f"Failed to load {file_path}: {e}")
    
    if failed_files:
        print(f"Warning: {len(failed_files)} files failed to load")
    
    return batteries


def get_battery_files(data_path: str) -> List[Path]:
    """
    Get list of battery pickle files without loading them.

    Args:
        data_path: Path to directory containing .pkl files

    Returns:
        List of Path objects to .pkl files

    Raises:
        FileNotFoundError: If data_path doesn't exist
        ValueError: If no .pkl files found
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    battery_files = list(data_path.glob("*.pkl"))
    if not battery_files:
        raise ValueError(f"No .pkl files found in {data_path}")
    
    return battery_files


def process_battery_file(file_path: Path) -> Optional[BatteryData]:
    """
    Process a single battery file and return the data.

    Args:
        file_path: Path to the .pkl file

    Returns:
        BatteryData object or None if loading failed
    """
    try:
        with open(file_path, 'rb') as f:
            battery = BatteryData.load(file_path)
            return battery
    except Exception as e:
        warnings.warn(f"Failed to load {file_path}: {e}")
        return None


def get_dataset_stats_streaming(data_path: str) -> Dict[str, Any]:
    """
    Get basic statistics about a dataset by processing files one at a time.
    This is memory-efficient for large datasets.

    Args:
        data_path: Path to directory containing .pkl files

    Returns:
        Dictionary containing dataset statistics
    """
    battery_files = get_battery_files(data_path)
    
    stats = {
        'total_batteries': 0,
        'datasets': {},
        'chemistries': {},
        'capacities': [],
        'cycle_lives': [],
        'voltages': [],
        'temperatures': []
    }
    
    print(f"Processing {len(battery_files)} battery files one at a time...")
    
    for file_path in tqdm(battery_files, desc="Processing battery files", unit="files"):
        battery = process_battery_file(file_path)
        if battery is None:
            continue
            
        stats['total_batteries'] += 1
        
        # Extract dataset name from cell_id (assume format: DATASET_CELLID)
        dataset_name = battery.cell_id.split('_')[0] if '_' in battery.cell_id else 'Unknown'
        if dataset_name not in stats['datasets']:
            stats['datasets'][dataset_name] = 0
        stats['datasets'][dataset_name] += 1
        
        # Extract chemistry information
        cathode = getattr(battery, 'cathode_material', 'Unknown')
        if cathode not in stats['chemistries']:
            stats['chemistries'][cathode] = 0
        stats['chemistries'][cathode] += 1
        
        # Collect capacity information
        if hasattr(battery, 'nominal_capacity_in_Ah') and battery.nominal_capacity_in_Ah:
            stats['capacities'].append(battery.nominal_capacity_in_Ah)
        
        # Collect cycle life information
        if battery.cycle_data:
            stats['cycle_lives'].append(len(battery.cycle_data))
        
        # Collect voltage and temperature data from cycles (sample to avoid memory issues)
        voltage_samples = []
        temp_samples = []
        for cycle in battery.cycle_data:
            if cycle.voltage_in_V and len(voltage_samples) < 1000:  # Limit samples
                voltage_samples.extend(cycle.voltage_in_V[:100])  # Take first 100 points
            if cycle.temperature_in_C and len(temp_samples) < 1000:  # Limit samples
                temp_samples.extend(cycle.temperature_in_C[:100])  # Take first 100 points
        
        stats['voltages'].extend(voltage_samples)
        stats['temperatures'].extend(temp_samples)
    
    # Convert lists to numpy arrays for easier statistics
    stats['capacities'] = np.array(stats['capacities']) if stats['capacities'] else np.array([])
    stats['cycle_lives'] = np.array(stats['cycle_lives']) if stats['cycle_lives'] else np.array([])
    stats['voltages'] = np.array(stats['voltages']) if stats['voltages'] else np.array([])
    stats['temperatures'] = np.array(stats['temperatures']) if stats['temperatures'] else np.array([])
    
    return stats


def get_dataset_stats(batteries: List[BatteryData]) -> Dict[str, Any]:
    """
    Get basic statistics about a dataset.
    
    Args:
        batteries: List of BatteryData objects
        
    Returns:
        Dictionary containing dataset statistics
    """
    if not batteries:
        return {}
    
    stats = {
        'total_batteries': len(batteries),
        'datasets': {},
        'chemistries': {},
        'capacities': [],
        'cycle_lives': [],
        'voltages': [],
        'temperatures': []
    }
    
    for battery in batteries:
        # Extract dataset name from cell_id (assume format: DATASET_CELLID)
        dataset_name = battery.cell_id.split('_')[0] if '_' in battery.cell_id else 'Unknown'
        if dataset_name not in stats['datasets']:
            stats['datasets'][dataset_name] = 0
        stats['datasets'][dataset_name] += 1
        
        # Extract chemistry information
        cathode = getattr(battery, 'cathode_material', 'Unknown')
        if cathode not in stats['chemistries']:
            stats['chemistries'][cathode] = 0
        stats['chemistries'][cathode] += 1
        
        # Collect capacity information
        if hasattr(battery, 'nominal_capacity_in_Ah') and battery.nominal_capacity_in_Ah:
            stats['capacities'].append(battery.nominal_capacity_in_Ah)
        
        # Collect cycle life information
        if battery.cycle_data:
            stats['cycle_lives'].append(len(battery.cycle_data))
        
        # Collect voltage and temperature data from cycles
        for cycle in battery.cycle_data:
            if cycle.voltage_in_V:
                stats['voltages'].extend(cycle.voltage_in_V)
            if cycle.temperature_in_C:
                stats['temperatures'].extend(cycle.temperature_in_C)
    
    # Convert lists to numpy arrays for easier statistics
    stats['capacities'] = np.array(stats['capacities']) if stats['capacities'] else np.array([])
    stats['cycle_lives'] = np.array(stats['cycle_lives']) if stats['cycle_lives'] else np.array([])
    stats['voltages'] = np.array(stats['voltages']) if stats['voltages'] else np.array([])
    stats['temperatures'] = np.array(stats['temperatures']) if stats['temperatures'] else np.array([])
    
    return stats


def safe_statistics(data: np.ndarray, name: str = "data") -> Dict[str, Any]:
    """
    Calculate safe statistics for an array, handling empty arrays and NaN values.
    
    Args:
        data: Input array
        name: Name of the data for error messages
        
    Returns:
        Dictionary containing statistics
    """
    if len(data) == 0:
        return {f"{name}_count": 0, f"{name}_min": np.nan, f"{name}_max": np.nan, 
                f"{name}_mean": np.nan, f"{name}_median": np.nan, f"{name}_std": np.nan}
    
    # Remove NaN and infinite values
    clean_data = data[~np.isnan(data) & np.isfinite(data)]
    
    if len(clean_data) == 0:
        return {f"{name}_count": len(data), f"{name}_min": np.nan, f"{name}_max": np.nan,
                f"{name}_mean": np.nan, f"{name}_median": np.nan, f"{name}_std": np.nan}
    
    return {
        f"{name}_count": len(clean_data),
        f"{name}_min": float(np.min(clean_data)),
        f"{name}_max": float(np.max(clean_data)),
        f"{name}_mean": float(np.mean(clean_data)),
        f"{name}_median": float(np.median(clean_data)),
        f"{name}_std": float(np.std(clean_data)),
        f"{name}_q25": float(np.percentile(clean_data, 25)),
        f"{name}_q75": float(np.percentile(clean_data, 75))
    }
