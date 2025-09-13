#!/usr/bin/env python3
"""
Test script for the battery data analysis tools.

This script tests the analysis functionality without requiring actual data files.
"""

import sys
import os
import tempfile
import pickle
from pathlib import Path
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from batteryml.data_analysis.analyzer import BatteryDataAnalyzer
    from batteryml.data_analysis.visualization import BatteryDataVisualizer
    from batteryml.data_analysis.utils import load_battery_data, get_dataset_stats
    print("✓ Successfully imported analysis modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def create_test_battery_data():
    """Create test battery data for testing purposes."""
    from batteryml.data.battery_data import BatteryData, CycleData, CyclingProtocol
    
    # Create test cycle data
    cycles = []
    for i in range(1, 11):  # 10 cycles
        # Generate realistic battery data
        voltage = np.linspace(3.0, 4.2, 100) + np.random.normal(0, 0.01, 100)
        current = np.random.normal(1.0, 0.1, 100)
        capacity = np.linspace(0, 1.1 - i*0.01, 100)  # Decreasing capacity
        time = np.linspace(0, 3600, 100)  # 1 hour cycle
        temperature = np.random.normal(25, 2, 100)
        
        cycle = CycleData(
            cycle_number=i,
            voltage_in_V=voltage.tolist(),
            current_in_A=current.tolist(),
            charge_capacity_in_Ah=capacity.tolist(),
            discharge_capacity_in_Ah=capacity.tolist(),
            time_in_s=time.tolist(),
            temperature_in_C=temperature.tolist()
        )
        cycles.append(cycle)
    
    # Create test battery
    battery = BatteryData(
        cell_id="TEST_001",
        cycle_data=cycles,
        nominal_capacity_in_Ah=1.1,
        cathode_material="LCO",
        anode_material="graphite",
        form_factor="cylindrical_18650"
    )
    
    return battery

def test_analysis_tools():
    """Test the analysis tools with synthetic data."""
    print("\n" + "="*60)
    print("TESTING BATTERY DATA ANALYSIS TOOLS")
    print("="*60)
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"Creating test data in: {temp_path}")
        
        # Create test battery data files
        test_batteries = []
        for i in range(5):  # Create 5 test batteries
            battery = create_test_battery_data()
            battery.cell_id = f"TEST_{i:03d}"
            battery.cathode_material = ["LCO", "LFP", "NMC", "NCA", "LCO"][i]
            
            # Save to pickle file
            battery_file = temp_path / f"test_battery_{i:03d}.pkl"
            with open(battery_file, 'wb') as f:
                pickle.dump(battery.to_dict(), f)
            
            test_batteries.append(battery)
        
        print(f"Created {len(test_batteries)} test battery files")
        
        # Test 1: Load data
        print("\n1. Testing data loading...")
        try:
            loaded_batteries = load_battery_data(str(temp_path))
            print(f"✓ Successfully loaded {len(loaded_batteries)} batteries")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
        
        # Test 2: Dataset statistics
        print("\n2. Testing dataset statistics...")
        try:
            stats = get_dataset_stats(loaded_batteries)
            print(f"✓ Dataset stats: {stats['total_batteries']} batteries")
            print(f"  - Chemistries: {list(stats['chemistries'].keys())}")
        except Exception as e:
            print(f"✗ Error getting dataset stats: {e}")
            return False
        
        # Test 3: Analyzer initialization
        print("\n3. Testing analyzer initialization...")
        try:
            analyzer = BatteryDataAnalyzer(str(temp_path))
            print("✓ Analyzer initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing analyzer: {e}")
            return False
        
        # Test 4: Dataset overview analysis
        print("\n4. Testing dataset overview analysis...")
        try:
            dataset_stats = analyzer.analyze_dataset_overview()
            print(f"✓ Dataset overview completed")
            print(f"  - Total batteries: {dataset_stats['total_batteries']}")
        except Exception as e:
            print(f"✗ Error in dataset overview: {e}")
            return False
        
        # Test 5: Feature analysis
        print("\n5. Testing feature analysis...")
        try:
            feature_stats = analyzer.analyze_features()
            print(f"✓ Feature analysis completed")
            print(f"  - Features analyzed: {len(feature_stats)}")
        except Exception as e:
            print(f"✗ Error in feature analysis: {e}")
            return False
        
        # Test 6: Visualization (without saving)
        print("\n6. Testing visualization tools...")
        try:
            visualizer = BatteryDataVisualizer(str(temp_path / "test_plots"), "test_dataset")
            print("✓ Visualizer initialized successfully")
            
            # Test plotting functions (without saving)
            visualizer.plot_dataset_overview(dataset_stats, save=False)
            print("✓ Dataset overview plot generated")
            
            visualizer.plot_feature_statistics(feature_stats, save=False)
            print("✓ Feature statistics plot generated")
            
        except Exception as e:
            print(f"✗ Error in visualization: {e}")
            return False
        
        # Test 7: Summary report
        print("\n7. Testing summary report generation...")
        try:
            summary = analyzer.generate_summary_report()
            print(f"✓ Summary report generated ({len(summary)} characters)")
        except Exception as e:
            print(f"✗ Error generating summary report: {e}")
            return False
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("="*60)
        return True

def main():
    """Main test function."""
    print("Battery Data Analysis Tools - Test Suite")
    print("=" * 50)
    
    # Check if required modules are available
    try:
        import matplotlib
        import seaborn
        import pandas
        print("✓ All required dependencies are available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install matplotlib seaborn pandas numpy")
        return False
    
    # Run tests
    success = test_analysis_tools()
    
    if success:
        print("\n🎉 All tests passed! The analysis tools are working correctly.")
        print("\nYou can now use the analysis tools with your battery data:")
        print("1. Place your .pkl battery data files in a directory")
        print("2. Run: python analyze_battery_data.py")
        print("3. Or use the command line: python batteryml/data_analysis/run_analysis.py --data_path your_data_path")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
