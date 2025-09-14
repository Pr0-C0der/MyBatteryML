# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

from batteryml.data.battery_data import BatteryData


class AnalysisUtils:
    """Utility functions for battery data analysis."""
    
    @staticmethod
    def safe_load_battery(file_path: str) -> Optional[BatteryData]:
        """
        Safely load a battery pickle file with error handling.
        
        Args:
            file_path: Path to the battery pickle file
            
        Returns:
            BatteryData object or None if loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                battery_data = BatteryData.load(file_path)
            return battery_data
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_battery_files(dataset_path: str) -> List[str]:
        """
        Get all battery pickle files from a dataset directory.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            List of battery file paths
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"Warning: Dataset path {dataset_path} does not exist")
            return []
        
        battery_files = list(dataset_path.glob("*.pkl"))
        return [str(f) for f in battery_files]
    
    @staticmethod
    def extract_cycle_statistics(cycle_data) -> Dict[str, Any]:
        """
        Extract statistics from cycle data.
        
        Args:
            cycle_data: List of CycleData objects
            
        Returns:
            Dictionary of cycle statistics
        """
        if not cycle_data:
            return {}
        
        stats = {
            'total_cycles': len(cycle_data),
            'cycle_numbers': [cycle.cycle_number for cycle in cycle_data]
        }
        
        # Extract discharge capacity statistics
        discharge_capacities = []
        charge_capacities = []
        voltages = []
        currents = []
        temperatures = []
        
        for cycle in cycle_data:
            if cycle.discharge_capacity_in_Ah:
                discharge_capacities.extend(cycle.discharge_capacity_in_Ah)
            if cycle.charge_capacity_in_Ah:
                charge_capacities.extend(cycle.charge_capacity_in_Ah)
            if cycle.voltage_in_V:
                voltages.extend(cycle.voltage_in_V)
            if cycle.current_in_A:
                currents.extend(cycle.current_in_A)
            if cycle.temperature_in_C:
                temperatures.extend(cycle.temperature_in_C)
        
        # Calculate statistics for each measurement type
        for data_type, values in [
            ('discharge_capacity', discharge_capacities),
            ('charge_capacity', charge_capacities),
            ('voltage', voltages),
            ('current', currents),
            ('temperature', temperatures)
        ]:
            if values:
                values = np.array(values)
                values = values[~np.isnan(values)]  # Remove NaN values
                if len(values) > 0:
                    stats[f'{data_type}_min'] = float(np.min(values))
                    stats[f'{data_type}_max'] = float(np.max(values))
                    stats[f'{data_type}_mean'] = float(np.mean(values))
                    stats[f'{data_type}_median'] = float(np.median(values))
                    stats[f'{data_type}_std'] = float(np.std(values))
                    stats[f'{data_type}_count'] = len(values)
                else:
                    stats[f'{data_type}_min'] = np.nan
                    stats[f'{data_type}_max'] = np.nan
                    stats[f'{data_type}_mean'] = np.nan
                    stats[f'{data_type}_median'] = np.nan
                    stats[f'{data_type}_std'] = np.nan
                    stats[f'{data_type}_count'] = 0
            else:
                stats[f'{data_type}_min'] = np.nan
                stats[f'{data_type}_max'] = np.nan
                stats[f'{data_type}_mean'] = np.nan
                stats[f'{data_type}_median'] = np.nan
                stats[f'{data_type}_std'] = np.nan
                stats[f'{data_type}_count'] = 0
        
        return stats
    
    @staticmethod
    def extract_battery_metadata(battery_data: BatteryData) -> Dict[str, Any]:
        """
        Extract metadata from battery data.
        
        Args:
            battery_data: BatteryData object
            
        Returns:
            Dictionary of battery metadata
        """
        metadata = {
            'cell_id': battery_data.cell_id,
            'form_factor': battery_data.form_factor,
            'anode_material': battery_data.anode_material,
            'cathode_material': battery_data.cathode_material,
            'electrolyte_material': battery_data.electrolyte_material,
            'nominal_capacity_in_Ah': battery_data.nominal_capacity_in_Ah,
            'depth_of_charge': battery_data.depth_of_charge,
            'depth_of_discharge': battery_data.depth_of_discharge,
            'already_spent_cycles': battery_data.already_spent_cycles,
            'max_voltage_limit_in_V': battery_data.max_voltage_limit_in_V,
            'min_voltage_limit_in_V': battery_data.min_voltage_limit_in_V,
            'max_current_limit_in_A': battery_data.max_current_limit_in_A,
            'min_current_limit_in_A': battery_data.min_current_limit_in_A,
            'reference': battery_data.reference,
            'description': battery_data.description
        }
        
        return metadata
    
    @staticmethod
    def calculate_cycle_life(battery_data: BatteryData, eol_threshold: float = 0.8) -> int:
        """
        Calculate cycle life (RUL) for a battery.
        
        Args:
            battery_data: BatteryData object
            eol_threshold: End-of-life threshold (fraction of nominal capacity)
            
        Returns:
            Cycle life (number of cycles until EOL)
        """
        if not battery_data.cycle_data or not battery_data.nominal_capacity_in_Ah:
            return 0
        
        eol_capacity = battery_data.nominal_capacity_in_Ah * eol_threshold
        
        for i, cycle in enumerate(battery_data.cycle_data):
            if cycle.discharge_capacity_in_Ah:
                max_discharge = max(cycle.discharge_capacity_in_Ah)
                if max_discharge <= eol_capacity:
                    return i + 1
        
        return len(battery_data.cycle_data)
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, returning default if denominator is zero.
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            default: Default value if division by zero
            
        Returns:
            Division result or default value
        """
        try:
            if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
                return default
            return numerator / denominator
        except (ZeroDivisionError, TypeError, ValueError):
            return default
