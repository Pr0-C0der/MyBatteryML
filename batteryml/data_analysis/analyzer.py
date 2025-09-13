# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings

from .utils import load_battery_data, get_dataset_stats, safe_statistics
from batteryml.data.battery_data import BatteryData


class BatteryDataAnalyzer:
    """
    Comprehensive battery data analyzer for extracting statistics and insights.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data path.
        
        Args:
            data_path: Path to directory containing battery data files
        """
        self.data_path = Path(data_path)
        self.batteries: List[BatteryData] = []
        self.dataset_stats: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Any] = {}
        self.dataset_name = self.data_path.name  # Extract dataset name from path
        
    def load_data(self) -> None:
        """Load battery data from the specified path."""
        print(f"Loading battery data from: {self.data_path}")
        self.batteries = load_battery_data(self.data_path)
        print(f"Successfully loaded {len(self.batteries)} batteries")
        
    def analyze_dataset_overview(self) -> Dict[str, Any]:
        """
        Analyze dataset overview including total batteries per dataset.
        
        Returns:
            Dictionary containing dataset overview statistics
        """
        if not self.batteries:
            self.load_data()
        
        self.dataset_stats = get_dataset_stats(self.batteries)
        
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"Total number of batteries: {self.dataset_stats['total_batteries']}")
        
        print("\nBatteries per dataset:")
        print("-" * 30)
        for dataset, count in self.dataset_stats['datasets'].items():
            print(f"{dataset:20}: {count:4d} batteries")
        
        print("\nBatteries per chemistry:")
        print("-" * 30)
        for chemistry, count in self.dataset_stats['chemistries'].items():
            print(f"{chemistry:20}: {count:4d} batteries")
        
        return self.dataset_stats
    
    def analyze_features(self) -> Dict[str, Any]:
        """
        Analyze all features in the dataset and their statistics.
        
        Returns:
            Dictionary containing feature statistics
        """
        if not self.batteries:
            self.load_data()
        
        print("\n" + "="*60)
        print("FEATURE ANALYSIS")
        print("="*60)
        
        self.feature_stats = {}
        
        # Analyze basic battery properties
        self._analyze_basic_properties()
        
        # Analyze cycle data features
        self._analyze_cycle_features()
        
        # Analyze capacity features
        self._analyze_capacity_features()
        
        # Analyze voltage features
        self._analyze_voltage_features()
        
        # Analyze current features
        self._analyze_current_features()
        
        # Analyze temperature features
        self._analyze_temperature_features()
        
        # Analyze time features
        self._analyze_time_features()
        
        return self.feature_stats
    
    def _analyze_basic_properties(self) -> None:
        """Analyze basic battery properties."""
        print("\nBasic Battery Properties:")
        print("-" * 40)
        
        # Nominal capacity
        capacities = []
        for battery in self.batteries:
            if hasattr(battery, 'nominal_capacity_in_Ah') and battery.nominal_capacity_in_Ah:
                capacities.append(battery.nominal_capacity_in_Ah)
        
        if capacities:
            cap_stats = safe_statistics(np.array(capacities), "nominal_capacity_Ah")
            self.feature_stats['nominal_capacity'] = cap_stats
            self._print_feature_stats("Nominal Capacity (Ah)", cap_stats)
        
        # Cycle count
        cycle_counts = [len(battery.cycle_data) for battery in self.batteries if battery.cycle_data]
        if cycle_counts:
            cycle_stats = safe_statistics(np.array(cycle_counts), "cycle_count")
            self.feature_stats['cycle_count'] = cycle_stats
            self._print_feature_stats("Cycle Count", cycle_stats)
    
    def _analyze_cycle_features(self) -> None:
        """Analyze cycle-level features."""
        print("\nCycle-Level Features:")
        print("-" * 40)
        
        # Collect all cycle data
        all_cycles = []
        for battery in self.batteries:
            all_cycles.extend(battery.cycle_data)
        
        if not all_cycles:
            print("No cycle data available")
            return
        
        # Cycle numbers
        cycle_numbers = [cycle.cycle_number for cycle in all_cycles]
        if cycle_numbers:
            cycle_num_stats = safe_statistics(np.array(cycle_numbers), "cycle_number")
            self.feature_stats['cycle_number'] = cycle_num_stats
            self._print_feature_stats("Cycle Number", cycle_num_stats)
    
    def _analyze_capacity_features(self) -> None:
        """Analyze capacity-related features."""
        print("\nCapacity Features:")
        print("-" * 40)
        
        charge_capacities = []
        discharge_capacities = []
        max_discharge_capacities = []
        
        for battery in self.batteries:
            for cycle in battery.cycle_data:
                if cycle.charge_capacity_in_Ah:
                    charge_capacities.extend(cycle.charge_capacity_in_Ah)
                if cycle.discharge_capacity_in_Ah:
                    discharge_capacities.extend(cycle.discharge_capacity_in_Ah)
                    max_discharge_capacities.append(max(cycle.discharge_capacity_in_Ah))
        
        # Charge capacity
        if charge_capacities:
            charge_stats = safe_statistics(np.array(charge_capacities), "charge_capacity_Ah")
            self.feature_stats['charge_capacity'] = charge_stats
            self._print_feature_stats("Charge Capacity (Ah)", charge_stats)
        
        # Discharge capacity
        if discharge_capacities:
            discharge_stats = safe_statistics(np.array(discharge_capacities), "discharge_capacity_Ah")
            self.feature_stats['discharge_capacity'] = discharge_stats
            self._print_feature_stats("Discharge Capacity (Ah)", discharge_stats)
        
        # Max discharge capacity per cycle
        if max_discharge_capacities:
            max_discharge_stats = safe_statistics(np.array(max_discharge_capacities), "max_discharge_capacity_Ah")
            self.feature_stats['max_discharge_capacity'] = max_discharge_stats
            self._print_feature_stats("Max Discharge Capacity per Cycle (Ah)", max_discharge_stats)
    
    def _analyze_voltage_features(self) -> None:
        """Analyze voltage-related features."""
        print("\nVoltage Features:")
        print("-" * 40)
        
        all_voltages = []
        min_voltages = []
        max_voltages = []
        
        for battery in self.batteries:
            for cycle in battery.cycle_data:
                if cycle.voltage_in_V:
                    voltages = np.array(cycle.voltage_in_V)
                    all_voltages.extend(voltages)
                    min_voltages.append(np.min(voltages))
                    max_voltages.append(np.max(voltages))
        
        # All voltage measurements
        if all_voltages:
            voltage_stats = safe_statistics(np.array(all_voltages), "voltage_V")
            self.feature_stats['voltage'] = voltage_stats
            self._print_feature_stats("Voltage (V)", voltage_stats)
        
        # Min voltage per cycle
        if min_voltages:
            min_voltage_stats = safe_statistics(np.array(min_voltages), "min_voltage_V")
            self.feature_stats['min_voltage'] = min_voltage_stats
            self._print_feature_stats("Min Voltage per Cycle (V)", min_voltage_stats)
        
        # Max voltage per cycle
        if max_voltages:
            max_voltage_stats = safe_statistics(np.array(max_voltages), "max_voltage_V")
            self.feature_stats['max_voltage'] = max_voltage_stats
            self._print_feature_stats("Max Voltage per Cycle (V)", max_voltage_stats)
    
    def _analyze_current_features(self) -> None:
        """Analyze current-related features."""
        print("\nCurrent Features:")
        print("-" * 40)
        
        all_currents = []
        min_currents = []
        max_currents = []
        
        for battery in self.batteries:
            for cycle in battery.cycle_data:
                if cycle.current_in_A:
                    currents = np.array(cycle.current_in_A)
                    all_currents.extend(currents)
                    min_currents.append(np.min(currents))
                    max_currents.append(np.max(currents))
        
        # All current measurements
        if all_currents:
            current_stats = safe_statistics(np.array(all_currents), "current_A")
            self.feature_stats['current'] = current_stats
            self._print_feature_stats("Current (A)", current_stats)
        
        # Min current per cycle
        if min_currents:
            min_current_stats = safe_statistics(np.array(min_currents), "min_current_A")
            self.feature_stats['min_current'] = min_current_stats
            self._print_feature_stats("Min Current per Cycle (A)", min_current_stats)
        
        # Max current per cycle
        if max_currents:
            max_current_stats = safe_statistics(np.array(max_currents), "max_current_A")
            self.feature_stats['max_current'] = max_current_stats
            self._print_feature_stats("Max Current per Cycle (A)", max_current_stats)
    
    def _analyze_temperature_features(self) -> None:
        """Analyze temperature-related features."""
        print("\nTemperature Features:")
        print("-" * 40)
        
        all_temperatures = []
        min_temperatures = []
        max_temperatures = []
        
        for battery in self.batteries:
            for cycle in battery.cycle_data:
                if cycle.temperature_in_C:
                    temps = np.array(cycle.temperature_in_C)
                    all_temperatures.extend(temps)
                    min_temperatures.append(np.min(temps))
                    max_temperatures.append(np.max(temps))
        
        # All temperature measurements
        if all_temperatures:
            temp_stats = safe_statistics(np.array(all_temperatures), "temperature_C")
            self.feature_stats['temperature'] = temp_stats
            self._print_feature_stats("Temperature (°C)", temp_stats)
        
        # Min temperature per cycle
        if min_temperatures:
            min_temp_stats = safe_statistics(np.array(min_temperatures), "min_temperature_C")
            self.feature_stats['min_temperature'] = min_temp_stats
            self._print_feature_stats("Min Temperature per Cycle (°C)", min_temp_stats)
        
        # Max temperature per cycle
        if max_temperatures:
            max_temp_stats = safe_statistics(np.array(max_temperatures), "max_temperature_C")
            self.feature_stats['max_temperature'] = max_temp_stats
            self._print_feature_stats("Max Temperature per Cycle (°C)", max_temp_stats)
    
    def _analyze_time_features(self) -> None:
        """Analyze time-related features."""
        print("\nTime Features:")
        print("-" * 40)
        
        all_times = []
        cycle_durations = []
        
        for battery in self.batteries:
            for cycle in battery.cycle_data:
                if cycle.time_in_s:
                    times = np.array(cycle.time_in_s)
                    all_times.extend(times)
                    if len(times) > 1:
                        cycle_durations.append(times[-1] - times[0])
        
        # All time measurements
        if all_times:
            time_stats = safe_statistics(np.array(all_times), "time_s")
            self.feature_stats['time'] = time_stats
            self._print_feature_stats("Time (s)", time_stats)
        
        # Cycle duration
        if cycle_durations:
            duration_stats = safe_statistics(np.array(cycle_durations), "cycle_duration_s")
            self.feature_stats['cycle_duration'] = duration_stats
            self._print_feature_stats("Cycle Duration (s)", duration_stats)
    
    def _print_feature_stats(self, feature_name: str, stats: Dict[str, Any]) -> None:
        """Print formatted feature statistics."""
        print(f"\n{feature_name}:")
        
        # Get the key prefix from the first key in stats
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
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            String containing the summary report
        """
        if not self.dataset_stats:
            self.analyze_dataset_overview()
        if not self.feature_stats:
            self.analyze_features()
        
        report = []
        report.append("="*80)
        report.append("BATTERY DATA ANALYSIS SUMMARY REPORT")
        report.append("="*80)
        
        # Dataset overview
        report.append(f"\nDataset Overview:")
        report.append(f"Total Batteries: {self.dataset_stats['total_batteries']}")
        
        report.append(f"\nBatteries per Dataset:")
        for dataset, count in self.dataset_stats['datasets'].items():
            report.append(f"  {dataset}: {count}")
        
        report.append(f"\nBatteries per Chemistry:")
        for chemistry, count in self.dataset_stats['chemistries'].items():
            report.append(f"  {chemistry}: {count}")
        
        # Feature summary
        report.append(f"\nFeature Summary:")
        report.append(f"Total Features Analyzed: {len(self.feature_stats)}")
        
        for feature_name, stats in self.feature_stats.items():
            key_prefix = list(stats.keys())[0].split('_')[0]
            count = stats.get(f'{key_prefix}_count', 0)
            mean = stats.get(f'{key_prefix}_mean', np.nan)
            report.append(f"  {feature_name}: {count} values, mean={mean:.4f}")
        
        return "\n".join(report)
    
    def save_analysis(self, output_path: str) -> None:
        """
        Save analysis results to files.
        
        Args:
            output_path: Directory to save analysis results
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save dataset stats
        if self.dataset_stats:
            dataset_df = pd.DataFrame([
                {'dataset': dataset, 'count': count} 
                for dataset, count in self.dataset_stats['datasets'].items()
            ])
            dataset_df.to_csv(output_path / 'dataset_overview.csv', index=False)
        
        # Save feature stats
        if self.feature_stats:
            feature_data = []
            for feature_name, stats in self.feature_stats.items():
                key_prefix = list(stats.keys())[0].split('_')[0]
                feature_data.append({
                    'feature': feature_name,
                    'count': stats.get(f'{key_prefix}_count', 0),
                    'min': stats.get(f'{key_prefix}_min', np.nan),
                    'max': stats.get(f'{key_prefix}_max', np.nan),
                    'mean': stats.get(f'{key_prefix}_mean', np.nan),
                    'median': stats.get(f'{key_prefix}_median', np.nan),
                    'std': stats.get(f'{key_prefix}_std', np.nan),
                    'q25': stats.get(f'{key_prefix}_q25', np.nan),
                    'q75': stats.get(f'{key_prefix}_q75', np.nan)
                })
            
            feature_df = pd.DataFrame(feature_data)
            feature_df.to_csv(output_path / 'feature_statistics.csv', index=False)
        
        # Save summary report
        with open(output_path / 'summary_report.txt', 'w') as f:
            f.write(self.generate_summary_report())
        
        print(f"\nAnalysis results saved to: {output_path}")
