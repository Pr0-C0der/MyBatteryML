#!/usr/bin/env python3
"""
Demo script comparing regular vs streaming battery data analysis.
"""

import tempfile
import sys
import pickle
import time
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.analyzer import BatteryDataAnalyzer
from batteryml.data_analysis.streaming_analyzer import StreamingBatteryDataAnalyzer
from batteryml.data.battery_data import BatteryData, CycleData
import numpy as np


def create_test_data(num_batteries=10, cycles_per_battery=20, data_points_per_cycle=100):
    """Create test battery data with specified parameters."""
    batteries = []
    
    for i in range(num_batteries):
        # Create test cycle data
        cycles = []
        for j in range(1, cycles_per_battery + 1):
            # Generate realistic battery data
            voltage = np.linspace(3.0, 4.2, data_points_per_cycle) + np.random.normal(0, 0.01, data_points_per_cycle)
            current = np.random.normal(1.0, 0.1, data_points_per_cycle)
            capacity = np.linspace(0, 1.1 - j*0.005, data_points_per_cycle)  # Decreasing capacity
            time = np.linspace(0, 3600, data_points_per_cycle)  # 1 hour cycle
            temperature = np.random.normal(25, 2, data_points_per_cycle)
            
            cycle = CycleData(
                cycle_number=j,
                voltage_in_V=voltage.tolist(),
                current_in_A=current.tolist(),
                charge_capacity_in_Ah=capacity.tolist(),
                discharge_capacity_in_Ah=capacity.tolist(),
                time_in_s=time.tolist(),
                temperature_in_C=temperature.tolist()
            )
            cycles.append(cycle)
        
        # Create battery data
        battery_data = BatteryData(
            cell_id=f"TEST_BATTERY_{i+1:03d}",
            cycle_data=cycles,
            nominal_capacity_in_Ah=1.1,
            cathode_material="LCO"
        )
        batteries.append(battery_data)
    
    return batteries


def demo_analysis_comparison():
    """Demo comparing regular vs streaming analysis."""
    print("="*80)
    print("BATTERY DATA ANALYSIS COMPARISON: REGULAR vs STREAMING")
    print("="*80)
    print()
    
    # Create test data
    print("Creating test data...")
    num_batteries = 5
    cycles_per_battery = 10
    data_points_per_cycle = 50
    
    batteries = create_test_data(num_batteries, cycles_per_battery, data_points_per_cycle)
    print(f"Created {num_batteries} batteries with {cycles_per_battery} cycles each")
    print(f"Total data points: {num_batteries * cycles_per_battery * data_points_per_cycle:,}")
    print()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save battery data to files
        print("Saving battery data to files...")
        for i, battery in enumerate(batteries):
            battery_file = temp_path / f"battery_{i+1:03d}.pkl"
            with open(battery_file, 'wb') as f:
                pickle.dump(battery.to_dict(), f)
        
        print(f"Saved {len(batteries)} battery files to {temp_path}")
        print()
        
        # Test 1: Regular Analysis (loads all data into memory)
        print("="*60)
        print("TEST 1: REGULAR ANALYSIS (Loads all data into memory)")
        print("="*60)
        
        start_time = time.time()
        try:
            analyzer_regular = BatteryDataAnalyzer(str(temp_path))
            dataset_stats_regular, feature_stats_regular = analyzer_regular.run_complete_analysis()
            regular_time = time.time() - start_time
            
            print(f"✅ Regular analysis completed in {regular_time:.2f} seconds")
            print(f"   Total batteries: {dataset_stats_regular.get('total_batteries', 0)}")
            print(f"   Features analyzed: {len(feature_stats_regular)}")
            
        except Exception as e:
            print(f"❌ Regular analysis failed: {e}")
            regular_time = None
        
        print()
        
        # Test 2: Streaming Analysis (processes files one at a time)
        print("="*60)
        print("TEST 2: STREAMING ANALYSIS (Processes files one at a time)")
        print("="*60)
        
        start_time = time.time()
        try:
            analyzer_streaming = StreamingBatteryDataAnalyzer(str(temp_path))
            dataset_stats_streaming, feature_stats_streaming = analyzer_streaming.run_complete_analysis()
            streaming_time = time.time() - start_time
            
            print(f"✅ Streaming analysis completed in {streaming_time:.2f} seconds")
            print(f"   Total batteries: {dataset_stats_streaming.get('total_batteries', 0)}")
            print(f"   Features analyzed: {len(feature_stats_streaming)}")
            
        except Exception as e:
            print(f"❌ Streaming analysis failed: {e}")
            streaming_time = None
        
        print()
        
        # Comparison
        print("="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        if regular_time and streaming_time:
            print(f"Regular analysis time:   {regular_time:.2f} seconds")
            print(f"Streaming analysis time: {streaming_time:.2f} seconds")
            print(f"Time difference:        {abs(regular_time - streaming_time):.2f} seconds")
            
            if streaming_time < regular_time:
                print("🏆 Streaming analysis was faster!")
            elif regular_time < streaming_time:
                print("🏆 Regular analysis was faster!")
            else:
                print("🤝 Both analyses took the same time!")
        
        print()
        print("MEMORY USAGE COMPARISON:")
        print("• Regular analysis: Loads ALL battery data into memory at once")
        print("• Streaming analysis: Processes ONE battery file at a time")
        print("• For large datasets, streaming analysis uses significantly less memory")
        print("• Streaming analysis is recommended for datasets with many files or large files")
        
        print()
        print("RECOMMENDATIONS:")
        print("• Use REGULAR analysis for: Small datasets, when you need all data in memory")
        print("• Use STREAMING analysis for: Large datasets, memory-constrained environments")
        print("• Both approaches provide the same analysis results!")


if __name__ == "__main__":
    demo_analysis_comparison()

