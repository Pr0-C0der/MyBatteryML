# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings
from tqdm import tqdm

from .utils import get_battery_files, process_battery_file, safe_statistics
from batteryml.data.battery_data import BatteryData


class StreamingBatteryDataAnalyzer:
    """
    Memory-efficient battery data analyzer that processes files one at a time.
    This is ideal for large datasets that don't fit in memory.
    """

    def __init__(self, data_path: str):
        """
        Initialize the streaming analyzer with data path.
        
        Args:
            data_path: Path to directory containing battery data files
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        self.dataset_stats: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Any] = {}
        self.dataset_name = self.data_path.name  # Extract dataset name from path
        
    def analyze_dataset_overview(self) -> Dict[str, Any]:
        """
        Analyze dataset overview using streaming approach.
        
        Returns:
            Dictionary containing dataset overview statistics
        """
        from .utils import get_dataset_stats_streaming
        
        self.dataset_stats = get_dataset_stats_streaming(str(self.data_path))
        
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"Total number of batteries: {self.dataset_stats['total_batteries']}")
        
        print("\nBatteries per dataset:")
        print("-" * 30)
        for dataset, count in self.dataset_stats['datasets'].items():
            print(f"{dataset:<20}: {count:>4} batteries")
        
        print("\nBatteries per chemistry:")
        print("-" * 30)
        for chemistry, count in self.dataset_stats['chemistries'].items():
            print(f"{chemistry:<20}: {count:>4} batteries")
        
        return self.dataset_stats
    
    def analyze_features_streaming(self) -> Dict[str, Any]:
        """
        Analyze features using streaming approach (one file at a time).
        
        Returns:
            Dictionary containing feature statistics
        """
        battery_files = get_battery_files(str(self.data_path))
        
        print("\n" + "="*60)
        print("FEATURE ANALYSIS")
        print("="*60)
        
        # Initialize feature collection dictionaries
        feature_data = {
            'nominal_capacity': [],
            'cycle_count': [],
            'cycle_number': [],
            'charge_capacity': [],
            'discharge_capacity': [],
            'max_discharge_capacity': [],
            'voltage': [],
            'min_voltage': [],
            'max_voltage': [],
            'current': [],
            'min_current': [],
            'max_current': [],
            'temperature': [],
            'min_temperature': [],
            'max_temperature': [],
            'time': [],
            'cycle_duration': []
        }
        
        print("Processing battery files for feature analysis...")
        
        for file_path in tqdm(battery_files, desc="Analyzing features", unit="files"):
            battery = process_battery_file(file_path)
            if battery is None:
                continue
            
            # Basic properties
            if hasattr(battery, 'nominal_capacity_in_Ah') and battery.nominal_capacity_in_Ah:
                feature_data['nominal_capacity'].append(battery.nominal_capacity_in_Ah)
            
            if battery.cycle_data:
                feature_data['cycle_count'].append(len(battery.cycle_data))
            
            # Process each cycle
            for cycle in battery.cycle_data:
                # Cycle number
                if hasattr(cycle, 'cycle_number') and cycle.cycle_number is not None:
                    feature_data['cycle_number'].append(cycle.cycle_number)
                
                # Capacity features
                if cycle.charge_capacity_in_Ah:
                    feature_data['charge_capacity'].extend(cycle.charge_capacity_in_Ah)
                
                if cycle.discharge_capacity_in_Ah:
                    feature_data['discharge_capacity'].extend(cycle.discharge_capacity_in_Ah)
                    feature_data['max_discharge_capacity'].append(max(cycle.discharge_capacity_in_Ah))
                
                # Voltage features
                if cycle.voltage_in_V:
                    voltages = np.array(cycle.voltage_in_V)
                    feature_data['voltage'].extend(voltages)
                    feature_data['min_voltage'].append(np.min(voltages))
                    feature_data['max_voltage'].append(np.max(voltages))
                
                # Current features
                if cycle.current_in_A:
                    currents = np.array(cycle.current_in_A)
                    feature_data['current'].extend(currents)
                    feature_data['min_current'].append(np.min(currents))
                    feature_data['max_current'].append(np.max(currents))
                
                # Temperature features
                if cycle.temperature_in_C:
                    temps = np.array(cycle.temperature_in_C)
                    feature_data['temperature'].extend(temps)
                    feature_data['min_temperature'].append(np.min(temps))
                    feature_data['max_temperature'].append(np.max(temps))
                
                # Time features
                if cycle.time_in_s:
                    times = np.array(cycle.time_in_s)
                    feature_data['time'].extend(times)
                    if len(times) > 1:
                        feature_data['cycle_duration'].append(times[-1] - times[0])
        
        # Calculate statistics for each feature
        self.feature_stats = {}
        
        # Basic properties
        self._analyze_basic_properties_streaming(feature_data)
        
        # Cycle features
        self._analyze_cycle_features_streaming(feature_data)
        
        # Capacity features
        self._analyze_capacity_features_streaming(feature_data)
        
        # Voltage features
        self._analyze_voltage_features_streaming(feature_data)
        
        # Current features
        self._analyze_current_features_streaming(feature_data)
        
        # Temperature features
        self._analyze_temperature_features_streaming(feature_data)
        
        # Time features
        self._analyze_time_features_streaming(feature_data)
        
        return self.feature_stats
    
    def _analyze_basic_properties_streaming(self, feature_data: Dict[str, List]) -> None:
        """Analyze basic battery properties from collected data."""
        print("\nBasic Battery Properties:")
        print("-" * 40)
        
        # Nominal capacity
        if feature_data['nominal_capacity']:
            cap_stats = safe_statistics(np.array(feature_data['nominal_capacity']), "nominal_capacity_Ah")
            self.feature_stats['nominal_capacity'] = cap_stats
            self._print_feature_stats("Nominal Capacity (Ah)", cap_stats)
        
        # Cycle count
        if feature_data['cycle_count']:
            cycle_stats = safe_statistics(np.array(feature_data['cycle_count']), "cycle_count")
            self.feature_stats['cycle_count'] = cycle_stats
            self._print_feature_stats("Cycle Count", cycle_stats)
    
    def _analyze_cycle_features_streaming(self, feature_data: Dict[str, List]) -> None:
        """Analyze cycle-level features from collected data."""
        print("\nCycle-Level Features:")
        print("-" * 40)
        
        # Cycle number
        if feature_data['cycle_number']:
            cycle_num_stats = safe_statistics(np.array(feature_data['cycle_number']), "cycle_number")
            self.feature_stats['cycle_number'] = cycle_num_stats
            self._print_feature_stats("Cycle Number", cycle_num_stats)
    
    def _analyze_capacity_features_streaming(self, feature_data: Dict[str, List]) -> None:
        """Analyze capacity-related features from collected data."""
        print("\nCapacity Features:")
        print("-" * 40)
        
        # Charge capacity
        if feature_data['charge_capacity']:
            charge_stats = safe_statistics(np.array(feature_data['charge_capacity']), "charge_capacity_Ah")
            self.feature_stats['charge_capacity'] = charge_stats
            self._print_feature_stats("Charge Capacity (Ah)", charge_stats)
        
        # Discharge capacity
        if feature_data['discharge_capacity']:
            discharge_stats = safe_statistics(np.array(feature_data['discharge_capacity']), "discharge_capacity_Ah")
            self.feature_stats['discharge_capacity'] = discharge_stats
            self._print_feature_stats("Discharge Capacity (Ah)", discharge_stats)
        
        # Max discharge capacity per cycle
        if feature_data['max_discharge_capacity']:
            max_discharge_stats = safe_statistics(np.array(feature_data['max_discharge_capacity']), "max_discharge_capacity_Ah")
            self.feature_stats['max_discharge_capacity'] = max_discharge_stats
            self._print_feature_stats("Max Discharge Capacity per Cycle (Ah)", max_discharge_stats)
    
    def _analyze_voltage_features_streaming(self, feature_data: Dict[str, List]) -> None:
        """Analyze voltage-related features from collected data."""
        print("\nVoltage Features:")
        print("-" * 40)
        
        # All voltage measurements
        if feature_data['voltage']:
            voltage_stats = safe_statistics(np.array(feature_data['voltage']), "voltage_V")
            self.feature_stats['voltage'] = voltage_stats
            self._print_feature_stats("Voltage (V)", voltage_stats)
        
        # Min voltage per cycle
        if feature_data['min_voltage']:
            min_voltage_stats = safe_statistics(np.array(feature_data['min_voltage']), "min_voltage_V")
            self.feature_stats['min_voltage'] = min_voltage_stats
            self._print_feature_stats("Min Voltage per Cycle (V)", min_voltage_stats)
        
        # Max voltage per cycle
        if feature_data['max_voltage']:
            max_voltage_stats = safe_statistics(np.array(feature_data['max_voltage']), "max_voltage_V")
            self.feature_stats['max_voltage'] = max_voltage_stats
            self._print_feature_stats("Max Voltage per Cycle (V)", max_voltage_stats)
    
    def _analyze_current_features_streaming(self, feature_data: Dict[str, List]) -> None:
        """Analyze current-related features from collected data."""
        print("\nCurrent Features:")
        print("-" * 40)
        
        # All current measurements
        if feature_data['current']:
            current_stats = safe_statistics(np.array(feature_data['current']), "current_A")
            self.feature_stats['current'] = current_stats
            self._print_feature_stats("Current (A)", current_stats)
        
        # Min current per cycle
        if feature_data['min_current']:
            min_current_stats = safe_statistics(np.array(feature_data['min_current']), "min_current_A")
            self.feature_stats['min_current'] = min_current_stats
            self._print_feature_stats("Min Current per Cycle (A)", min_current_stats)
        
        # Max current per cycle
        if feature_data['max_current']:
            max_current_stats = safe_statistics(np.array(feature_data['max_current']), "max_current_A")
            self.feature_stats['max_current'] = max_current_stats
            self._print_feature_stats("Max Current per Cycle (A)", max_current_stats)
    
    def _analyze_temperature_features_streaming(self, feature_data: Dict[str, List]) -> None:
        """Analyze temperature-related features from collected data."""
        print("\nTemperature Features:")
        print("-" * 40)
        
        # All temperature measurements
        if feature_data['temperature']:
            temp_stats = safe_statistics(np.array(feature_data['temperature']), "temperature_C")
            self.feature_stats['temperature'] = temp_stats
            self._print_feature_stats("Temperature (°C)", temp_stats)
        
        # Min temperature per cycle
        if feature_data['min_temperature']:
            min_temp_stats = safe_statistics(np.array(feature_data['min_temperature']), "min_temperature_C")
            self.feature_stats['min_temperature'] = min_temp_stats
            self._print_feature_stats("Min Temperature per Cycle (°C)", min_temp_stats)
        
        # Max temperature per cycle
        if feature_data['max_temperature']:
            max_temp_stats = safe_statistics(np.array(feature_data['max_temperature']), "max_temperature_C")
            self.feature_stats['max_temperature'] = max_temp_stats
            self._print_feature_stats("Max Temperature per Cycle (°C)", max_temp_stats)
    
    def _analyze_time_features_streaming(self, feature_data: Dict[str, List]) -> None:
        """Analyze time-related features from collected data."""
        print("\nTime Features:")
        print("-" * 40)
        
        # All time measurements
        if feature_data['time']:
            time_stats = safe_statistics(np.array(feature_data['time']), "time_s")
            self.feature_stats['time'] = time_stats
            self._print_feature_stats("Time (s)", time_stats)
        
        # Cycle duration
        if feature_data['cycle_duration']:
            duration_stats = safe_statistics(np.array(feature_data['cycle_duration']), "cycle_duration_s")
            self.feature_stats['cycle_duration'] = duration_stats
            self._print_feature_stats("Cycle Duration (s)", duration_stats)
    
    def _print_feature_stats(self, feature_name: str, stats: Dict[str, Any]) -> None:
        """Print formatted feature statistics."""
        print(f"\n{feature_name}:")
        
        if not stats:
            print("  No data available")
            return
            
        first_key = list(stats.keys())[0]
        key_prefix = first_key.split('_')[0]
        
        count = stats.get(f'{key_prefix}_count', 'N/A')
        min_val = stats.get(f'{key_prefix}_min', 'N/A')
        max_val = stats.get(f'{key_prefix}_max', 'N/A')
        mean_val = stats.get(f'{key_prefix}_mean', 'N/A')
        median_val = stats.get(f'{key_prefix}_median', 'N/A')
        std_val = stats.get(f'{key_prefix}_std', 'N/A')
        q25_val = stats.get(f'{key_prefix}_q25', 'N/A')
        q75_val = stats.get(f'{key_prefix}_q75', 'N/A')
        
        print(f"  Count:    {count:>8}")
        if isinstance(min_val, (int, float)) and not np.isnan(min_val):
            print(f"  Min:      {min_val:>8.4f}")
        else:
            print(f"  Min:      {'N/A':>8}")
        
        if isinstance(max_val, (int, float)) and not np.isnan(max_val):
            print(f"  Max:      {max_val:>8.4f}")
        else:
            print(f"  Max:      {'N/A':>8}")
        
        if isinstance(mean_val, (int, float)) and not np.isnan(mean_val):
            print(f"  Mean:     {mean_val:>8.4f}")
        else:
            print(f"  Mean:     {'N/A':>8}")
        
        if isinstance(median_val, (int, float)) and not np.isnan(median_val):
            print(f"  Median:   {median_val:>8.4f}")
        else:
            print(f"  Median:   {'N/A':>8}")
        
        if isinstance(std_val, (int, float)) and not np.isnan(std_val):
            print(f"  Std:      {std_val:>8.4f}")
        else:
            print(f"  Std:      {'N/A':>8}")
        
        if isinstance(q25_val, (int, float)) and not np.isnan(q25_val):
            print(f"  Q25:      {q25_val:>8.4f}")
        else:
            print(f"  Q25:      {'N/A':>8}")
        
        if isinstance(q75_val, (int, float)) and not np.isnan(q75_val):
            print(f"  Q75:      {q75_val:>8.4f}")
        else:
            print(f"  Q75:      {'N/A':>8}")
    
    def run_complete_analysis(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run complete analysis using streaming approach.
        
        Returns:
            Tuple of (dataset_stats, feature_stats)
        """
        print(f"Starting streaming analysis of {self.dataset_name} dataset...")
        print(f"Data path: {self.data_path}")
        
        # Analyze dataset overview
        dataset_stats = self.analyze_dataset_overview()
        
        # Analyze features
        feature_stats = self.analyze_features_streaming()
        
        print(f"\nAnalysis completed for {self.dataset_name} dataset!")
        print(f"Total features analyzed: {len(feature_stats)}")
        
        return dataset_stats, feature_stats

