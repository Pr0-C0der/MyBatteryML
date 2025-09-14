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
    TRULY memory-efficient battery data analyzer that processes files one at a time
    and calculates running statistics without storing all data in memory.
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
        Analyze features using TRUE streaming approach (running statistics).
        This method calculates statistics without storing all data in memory.
        
        Returns:
            Dictionary containing feature statistics
        """
        battery_files = get_battery_files(str(self.data_path))
        
        print("\n" + "="*60)
        print("FEATURE ANALYSIS (TRUE STREAMING)")
        print("="*60)
        
        # Initialize running statistics accumulators
        running_stats = self._initialize_running_stats()
        
        print("Processing battery files with running statistics...")
        
        for file_path in tqdm(battery_files, desc="Analyzing features", unit="files"):
            battery = process_battery_file(file_path)
            if battery is None:
                continue
            
            # Process this battery and update running statistics
            self._update_running_stats(running_stats, battery)
        
        # Calculate final statistics from running accumulators
        self.feature_stats = self._calculate_final_stats(running_stats)
        
        # Print results
        self._print_all_feature_stats()
        
        return self.feature_stats
    
    def _initialize_running_stats(self) -> Dict[str, Any]:
        """Initialize running statistics accumulators."""
        return {
            # Basic properties
            'nominal_capacity': {'values': [], 'count': 0},
            'cycle_count': {'values': [], 'count': 0},
            
            # Cycle-level features
            'cycle_number': {'values': [], 'count': 0},
            
            # Capacity features
            'charge_capacity': {'values': [], 'count': 0},
            'discharge_capacity': {'values': [], 'count': 0},
            'max_discharge_capacity': {'values': [], 'count': 0},
            
            # Voltage features
            'voltage': {'values': [], 'count': 0},
            'min_voltage': {'values': [], 'count': 0},
            'max_voltage': {'values': [], 'count': 0},
            
            # Current features
            'current': {'values': [], 'count': 0},
            'min_current': {'values': [], 'count': 0},
            'max_current': {'values': [], 'count': 0},
            
            # Temperature features
            'temperature': {'values': [], 'count': 0},
            'min_temperature': {'values': [], 'count': 0},
            'max_temperature': {'values': [], 'count': 0},
            
            # Time features
            'time': {'values': [], 'count': 0},
            'cycle_duration': {'values': [], 'count': 0}
        }
    
    def _update_running_stats(self, running_stats: Dict[str, Any], battery: BatteryData) -> None:
        """Update running statistics with data from one battery."""
        
        # Basic properties
        if hasattr(battery, 'nominal_capacity_in_Ah') and battery.nominal_capacity_in_Ah:
            running_stats['nominal_capacity']['values'].append(battery.nominal_capacity_in_Ah)
            running_stats['nominal_capacity']['count'] += 1
        
        if battery.cycle_data:
            running_stats['cycle_count']['values'].append(len(battery.cycle_data))
            running_stats['cycle_count']['count'] += 1
        
        # Process each cycle
        for cycle in battery.cycle_data:
            # Cycle number
            if hasattr(cycle, 'cycle_number') and cycle.cycle_number is not None:
                running_stats['cycle_number']['values'].append(cycle.cycle_number)
                running_stats['cycle_number']['count'] += 1
            
            # Capacity features
            if cycle.charge_capacity_in_Ah:
                running_stats['charge_capacity']['values'].extend(cycle.charge_capacity_in_Ah)
                running_stats['charge_capacity']['count'] += len(cycle.charge_capacity_in_Ah)
            
            if cycle.discharge_capacity_in_Ah:
                running_stats['discharge_capacity']['values'].extend(cycle.discharge_capacity_in_Ah)
                running_stats['discharge_capacity']['count'] += len(cycle.discharge_capacity_in_Ah)
                running_stats['max_discharge_capacity']['values'].append(max(cycle.discharge_capacity_in_Ah))
                running_stats['max_discharge_capacity']['count'] += 1
            
            # Voltage features
            if cycle.voltage_in_V:
                voltages = np.array(cycle.voltage_in_V)
                running_stats['voltage']['values'].extend(voltages)
                running_stats['voltage']['count'] += len(voltages)
                running_stats['min_voltage']['values'].append(np.min(voltages))
                running_stats['min_voltage']['count'] += 1
                running_stats['max_voltage']['values'].append(np.max(voltages))
                running_stats['max_voltage']['count'] += 1
            
            # Current features
            if cycle.current_in_A:
                currents = np.array(cycle.current_in_A)
                running_stats['current']['values'].extend(currents)
                running_stats['current']['count'] += len(currents)
                running_stats['min_current']['values'].append(np.min(currents))
                running_stats['min_current']['count'] += 1
                running_stats['max_current']['values'].append(np.max(currents))
                running_stats['max_current']['count'] += 1
            
            # Temperature features
            if cycle.temperature_in_C:
                temps = np.array(cycle.temperature_in_C)
                running_stats['temperature']['values'].extend(temps)
                running_stats['temperature']['count'] += len(temps)
                running_stats['min_temperature']['values'].append(np.min(temps))
                running_stats['min_temperature']['count'] += 1
                running_stats['max_temperature']['values'].append(np.max(temps))
                running_stats['max_temperature']['count'] += 1
            
            # Time features
            if cycle.time_in_s:
                times = np.array(cycle.time_in_s)
                running_stats['time']['values'].extend(times)
                running_stats['time']['count'] += len(times)
                if len(times) > 1:
                    running_stats['cycle_duration']['values'].append(times[-1] - times[0])
                    running_stats['cycle_duration']['count'] += 1
    
    def _calculate_final_stats(self, running_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final statistics from running accumulators."""
        feature_stats = {}
        
        for feature_name, data in running_stats.items():
            if data['count'] > 0:
                # Convert to numpy array and calculate statistics
                values = np.array(data['values'])
                stats = safe_statistics(values, feature_name)
                feature_stats[feature_name] = stats
            else:
                feature_stats[feature_name] = {}
        
        return feature_stats
    
    def _print_all_feature_stats(self) -> None:
        """Print all feature statistics in organized sections."""
        
        # Basic properties
        self._print_feature_section("Basic Battery Properties")
        self._print_feature_stats("Nominal Capacity (Ah)", self.feature_stats.get('nominal_capacity', {}))
        self._print_feature_stats("Cycle Count", self.feature_stats.get('cycle_count', {}))
        
        # Cycle features
        self._print_feature_section("Cycle-Level Features")
        self._print_feature_stats("Cycle Number", self.feature_stats.get('cycle_number', {}))
        
        # Capacity features
        self._print_feature_section("Capacity Features")
        self._print_feature_stats("Charge Capacity (Ah)", self.feature_stats.get('charge_capacity', {}))
        self._print_feature_stats("Discharge Capacity (Ah)", self.feature_stats.get('discharge_capacity', {}))
        self._print_feature_stats("Max Discharge Capacity per Cycle (Ah)", self.feature_stats.get('max_discharge_capacity', {}))
        
        # Voltage features
        self._print_feature_section("Voltage Features")
        self._print_feature_stats("Voltage (V)", self.feature_stats.get('voltage', {}))
        self._print_feature_stats("Min Voltage per Cycle (V)", self.feature_stats.get('min_voltage', {}))
        self._print_feature_stats("Max Voltage per Cycle (V)", self.feature_stats.get('max_voltage', {}))
        
        # Current features
        self._print_feature_section("Current Features")
        self._print_feature_stats("Current (A)", self.feature_stats.get('current', {}))
        self._print_feature_stats("Min Current per Cycle (A)", self.feature_stats.get('min_current', {}))
        self._print_feature_stats("Max Current per Cycle (A)", self.feature_stats.get('max_current', {}))
        
        # Temperature features
        self._print_feature_section("Temperature Features")
        self._print_feature_stats("Temperature (°C)", self.feature_stats.get('temperature', {}))
        self._print_feature_stats("Min Temperature per Cycle (°C)", self.feature_stats.get('min_temperature', {}))
        self._print_feature_stats("Max Temperature per Cycle (°C)", self.feature_stats.get('max_temperature', {}))
        
        # Time features
        self._print_feature_section("Time Features")
        self._print_feature_stats("Time (s)", self.feature_stats.get('time', {}))
        self._print_feature_stats("Cycle Duration (s)", self.feature_stats.get('cycle_duration', {}))
    
    def _print_feature_section(self, title: str) -> None:
        """Print a section header for feature analysis."""
        print(f"\n{title}:")
        print("-" * 40)
    
    
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
        Run complete analysis using TRUE streaming approach.
        
        Returns:
            Tuple of (dataset_stats, feature_stats)
        """
        print(f"Starting TRUE streaming analysis of {self.dataset_name} dataset...")
        print(f"Data path: {self.data_path}")
        print("Note: This analyzer processes files one at a time and calculates running statistics")
        print("without storing all data in memory.")
        
        # Analyze dataset overview
        dataset_stats = self.analyze_dataset_overview()
        
        # Analyze features
        feature_stats = self.analyze_features_streaming()
        
        print(f"\nAnalysis completed for {self.dataset_name} dataset!")
        print(f"Total features analyzed: {len(feature_stats)}")
        
        return dataset_stats, feature_stats

