# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from batteryml.data.battery_data import BatteryData
from .utils import AnalysisUtils


class BatteryAnalyzer:
    """Analyzer for individual battery data."""
    
    def __init__(self, battery_data: BatteryData):
        """
        Initialize the battery analyzer.
        
        Args:
            battery_data: BatteryData object to analyze
        """
        self.battery_data = battery_data
        self.metadata = AnalysisUtils.extract_battery_metadata(battery_data)
        self.cycle_stats = AnalysisUtils.extract_cycle_statistics(battery_data.cycle_data)
        self.cycle_life = AnalysisUtils.calculate_cycle_life(battery_data)
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the battery.
        
        Returns:
            Dictionary containing basic battery information
        """
        return {
            'cell_id': self.metadata['cell_id'],
            'total_cycles': self.cycle_stats.get('total_cycles', 0),
            'cycle_life': self.cycle_life,
            'nominal_capacity': self.metadata['nominal_capacity_in_Ah'],
            'cathode_material': self.metadata['cathode_material'],
            'anode_material': self.metadata['anode_material'],
            'form_factor': self.metadata['form_factor']
        }
    
    def get_capacity_statistics(self) -> Dict[str, Any]:
        """
        Get capacity-related statistics.
        
        Returns:
            Dictionary containing capacity statistics
        """
        stats = {}
        
        # Discharge capacity statistics
        if 'discharge_capacity_mean' in self.cycle_stats:
            stats['discharge_capacity'] = {
                'min': self.cycle_stats.get('discharge_capacity_min', np.nan),
                'max': self.cycle_stats.get('discharge_capacity_max', np.nan),
                'mean': self.cycle_stats.get('discharge_capacity_mean', np.nan),
                'median': self.cycle_stats.get('discharge_capacity_median', np.nan),
                'std': self.cycle_stats.get('discharge_capacity_std', np.nan),
                'count': self.cycle_stats.get('discharge_capacity_count', 0)
            }
        
        # Charge capacity statistics
        if 'charge_capacity_mean' in self.cycle_stats:
            stats['charge_capacity'] = {
                'min': self.cycle_stats.get('charge_capacity_min', np.nan),
                'max': self.cycle_stats.get('charge_capacity_max', np.nan),
                'mean': self.cycle_stats.get('charge_capacity_mean', np.nan),
                'median': self.cycle_stats.get('charge_capacity_median', np.nan),
                'std': self.cycle_stats.get('charge_capacity_std', np.nan),
                'count': self.cycle_stats.get('charge_capacity_count', 0)
            }
        
        # Calculate capacity retention if possible
        if (self.metadata['nominal_capacity_in_Ah'] and 
            'discharge_capacity_mean' in self.cycle_stats and
            not np.isnan(self.cycle_stats['discharge_capacity_mean'])):
            
            capacity_retention = AnalysisUtils.safe_divide(
                self.cycle_stats['discharge_capacity_mean'],
                self.metadata['nominal_capacity_in_Ah']
            )
            stats['capacity_retention'] = capacity_retention
        
        return stats
    
    def get_voltage_statistics(self) -> Dict[str, Any]:
        """
        Get voltage-related statistics.
        
        Returns:
            Dictionary containing voltage statistics
        """
        if 'voltage_mean' not in self.cycle_stats:
            return {}
        
        return {
            'voltage': {
                'min': self.cycle_stats.get('voltage_min', np.nan),
                'max': self.cycle_stats.get('voltage_max', np.nan),
                'mean': self.cycle_stats.get('voltage_mean', np.nan),
                'median': self.cycle_stats.get('voltage_median', np.nan),
                'std': self.cycle_stats.get('voltage_std', np.nan),
                'count': self.cycle_stats.get('voltage_count', 0)
            }
        }
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """
        Get current-related statistics.
        
        Returns:
            Dictionary containing current statistics
        """
        if 'current_mean' not in self.cycle_stats:
            return {}
        
        return {
            'current': {
                'min': self.cycle_stats.get('current_min', np.nan),
                'max': self.cycle_stats.get('current_max', np.nan),
                'mean': self.cycle_stats.get('current_mean', np.nan),
                'median': self.cycle_stats.get('current_median', np.nan),
                'std': self.cycle_stats.get('current_std', np.nan),
                'count': self.cycle_stats.get('current_count', 0)
            }
        }
    
    def get_temperature_statistics(self) -> Dict[str, Any]:
        """
        Get temperature-related statistics.
        
        Returns:
            Dictionary containing temperature statistics
        """
        if 'temperature_mean' not in self.cycle_stats:
            return {}
        
        return {
            'temperature': {
                'min': self.cycle_stats.get('temperature_min', np.nan),
                'max': self.cycle_stats.get('temperature_max', np.nan),
                'mean': self.cycle_stats.get('temperature_mean', np.nan),
                'median': self.cycle_stats.get('temperature_median', np.nan),
                'std': self.cycle_stats.get('temperature_std', np.nan),
                'count': self.cycle_stats.get('temperature_count', 0)
            }
        }
    
    def get_cycle_life_analysis(self) -> Dict[str, Any]:
        """
        Get cycle life analysis.
        
        Returns:
            Dictionary containing cycle life analysis
        """
        analysis = {
            'total_cycles': self.cycle_stats.get('total_cycles', 0),
            'cycle_life': self.cycle_life,
            'cycles_remaining': max(0, self.cycle_life - self.cycle_stats.get('total_cycles', 0))
        }
        
        # Calculate degradation rate if we have enough data
        if (self.cycle_stats.get('total_cycles', 0) > 1 and 
            self.metadata['nominal_capacity_in_Ah'] and
            'discharge_capacity_mean' in self.cycle_stats):
            
            # Simple linear degradation rate calculation
            initial_capacity = self.metadata['nominal_capacity_in_Ah']
            current_capacity = self.cycle_stats['discharge_capacity_mean']
            
            if not np.isnan(initial_capacity) and not np.isnan(current_capacity):
                capacity_loss = initial_capacity - current_capacity
                degradation_rate = AnalysisUtils.safe_divide(
                    capacity_loss, 
                    self.cycle_stats.get('total_cycles', 1)
                )
                analysis['degradation_rate'] = degradation_rate
                analysis['capacity_loss_percent'] = AnalysisUtils.safe_divide(
                    capacity_loss * 100, 
                    initial_capacity
                )
        
        return analysis
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of the battery.
        
        Returns:
            Dictionary containing all analysis results
        """
        analysis = {
            'basic_info': self.get_basic_info(),
            'capacity_stats': self.get_capacity_statistics(),
            'voltage_stats': self.get_voltage_statistics(),
            'current_stats': self.get_current_statistics(),
            'temperature_stats': self.get_temperature_statistics(),
            'cycle_life_analysis': self.get_cycle_life_analysis()
        }
        
        return analysis
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert analysis results to a pandas DataFrame.
        
        Returns:
            DataFrame containing analysis results
        """
        analysis = self.get_comprehensive_analysis()
        
        # Flatten the nested dictionary
        flattened = {}
        
        # Basic info
        for key, value in analysis['basic_info'].items():
            flattened[f'basic_{key}'] = value
        
        # Capacity stats
        for category, stats in analysis['capacity_stats'].items():
            for stat_name, value in stats.items():
                flattened[f'capacity_{category}_{stat_name}'] = value
        
        # Voltage stats
        for category, stats in analysis['voltage_stats'].items():
            for stat_name, value in stats.items():
                flattened[f'voltage_{category}_{stat_name}'] = value
        
        # Current stats
        for category, stats in analysis['current_stats'].items():
            for stat_name, value in stats.items():
                flattened[f'current_{category}_{stat_name}'] = value
        
        # Temperature stats
        for category, stats in analysis['temperature_stats'].items():
            for stat_name, value in stats.items():
                flattened[f'temperature_{category}_{stat_name}'] = value
        
        # Cycle life analysis
        for key, value in analysis['cycle_life_analysis'].items():
            flattened[f'cycle_life_{key}'] = value
        
        return pd.DataFrame([flattened])
    
    def print_summary(self):
        """Print a summary of the battery analysis."""
        print(f"\n{'='*60}")
        print(f"BATTERY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        basic_info = self.get_basic_info()
        print(f"Cell ID: {basic_info['cell_id']}")
        print(f"Total Cycles: {basic_info['total_cycles']}")
        print(f"Cycle Life: {basic_info['cycle_life']}")
        print(f"Nominal Capacity: {basic_info['nominal_capacity']} Ah")
        print(f"Cathode Material: {basic_info['cathode_material']}")
        print(f"Anode Material: {basic_info['anode_material']}")
        print(f"Form Factor: {basic_info['form_factor']}")
        
        # Capacity statistics
        capacity_stats = self.get_capacity_statistics()
        if capacity_stats:
            print(f"\n--- CAPACITY STATISTICS ---")
            for category, stats in capacity_stats.items():
                if isinstance(stats, dict):
                    print(f"\n{category.upper()}:")
                    for stat_name, value in stats.items():
                        if not np.isnan(value):
                            print(f"  {stat_name}: {value:.4f}")
        
        # Cycle life analysis
        cycle_analysis = self.get_cycle_life_analysis()
        print(f"\n--- CYCLE LIFE ANALYSIS ---")
        for key, value in cycle_analysis.items():
            if not np.isnan(value):
                print(f"{key}: {value:.4f}")
        
        print(f"{'='*60}\n")
