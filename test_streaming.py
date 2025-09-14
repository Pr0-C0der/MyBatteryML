#!/usr/bin/env python3
"""
Test script for streaming battery data analyzer.
"""

import tempfile
import sys
import pickle
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.streaming_analyzer import StreamingBatteryDataAnalyzer
from batteryml.data.battery_data import BatteryData, CycleData
import numpy as np


def test_streaming_analyzer():
    """Test the streaming analyzer with dummy data."""
    print("Testing Streaming Battery Data Analyzer")
    print("="*50)
    
    # Create temporary directory with test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test battery files
        print("Creating test battery data...")
        for i in range(3):
            # Create test cycle data
            cycles = []
            for j in range(1, 6):  # 5 cycles
                voltage = np.linspace(3.0, 4.2, 50) + np.random.normal(0, 0.01, 50)
                current = np.random.normal(1.0, 0.1, 50)
                capacity = np.linspace(0, 1.1 - j*0.01, 50)
                time = np.linspace(0, 1800, 50)
                temperature = np.random.normal(25, 2, 50)
                
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
                cell_id=f"TEST_BATTERY_{i+1}",
                cycle_data=cycles,
                nominal_capacity_in_Ah=1.1,
                cathode_material="LCO"
            )
            
            battery_file = temp_path / f"battery_{i+1}.pkl"
            with open(battery_file, 'wb') as f:
                pickle.dump(battery_data.to_dict(), f)
        
        print(f"Created 3 test battery files in {temp_path}")
        
        # Test streaming analyzer
        print("\nTesting streaming analyzer...")
        analyzer = StreamingBatteryDataAnalyzer(str(temp_path))
        
        # Run analysis
        dataset_stats, feature_stats = analyzer.run_complete_analysis()
        
        print(f"\nResults:")
        print(f"  Dataset: {analyzer.dataset_name}")
        print(f"  Total batteries: {dataset_stats.get('total_batteries', 0)}")
        print(f"  Features analyzed: {len(feature_stats)}")
        
        print("\n✅ Streaming analyzer test completed successfully!")


if __name__ == "__main__":
    test_streaming_analyzer()
