#!/usr/bin/env python3
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Test script for the data analysis module.
This script tests the analysis functionality to ensure no runtime errors.
"""

import sys
import os
from pathlib import Path
import tempfile
import warnings

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing imports...")
    
    try:
        from batteryml.data_analysis import BatteryAnalyzer, DatasetAnalyzer, AnalysisVisualizer, AnalysisUtils
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    try:
        from batteryml.data_analysis.utils import AnalysisUtils
        
        # Test safe_divide function
        result = AnalysisUtils.safe_divide(10, 2)
        assert result == 5.0, f"Expected 5.0, got {result}"
        
        result = AnalysisUtils.safe_divide(10, 0)
        assert result == 0.0, f"Expected 0.0, got {result}"
        
        result = AnalysisUtils.safe_divide(10, float('nan'))
        assert result == 0.0, f"Expected 0.0, got {result}"
        
        print("✓ Utility functions working correctly")
        return True
    except Exception as e:
        print(f"✗ Utility test error: {e}")
        return False

def test_battery_analyzer():
    """Test BatteryAnalyzer with mock data."""
    print("Testing BatteryAnalyzer...")
    
    try:
        from batteryml.data_analysis.battery_analyzer import BatteryAnalyzer
        from batteryml.data.battery_data import BatteryData, CycleData
        
        # Create mock battery data
        cycle_data = [
            CycleData(
                cycle_number=1,
                voltage_in_V=[3.0, 3.5, 4.0, 3.8, 3.2],
                current_in_A=[1.0, 1.0, 1.0, 1.0, 1.0],
                discharge_capacity_in_Ah=[0.0, 0.2, 0.5, 0.8, 1.0],
                charge_capacity_in_Ah=[0.0, 0.2, 0.5, 0.8, 1.0],
                temperature_in_C=[25.0, 25.5, 26.0, 25.8, 25.2]
            ),
            CycleData(
                cycle_number=2,
                voltage_in_V=[3.0, 3.4, 3.9, 3.7, 3.1],
                current_in_A=[1.0, 1.0, 1.0, 1.0, 1.0],
                discharge_capacity_in_Ah=[0.0, 0.19, 0.48, 0.77, 0.98],
                charge_capacity_in_Ah=[0.0, 0.19, 0.48, 0.77, 0.98],
                temperature_in_C=[25.0, 25.6, 26.1, 25.9, 25.3]
            )
        ]
        
        battery_data = BatteryData(
            cell_id="test_battery_001",
            cycle_data=cycle_data,
            nominal_capacity_in_Ah=1.0,
            cathode_material="LFP",
            anode_material="graphite",
            form_factor="cylindrical_18650"
        )
        
        # Test analyzer
        analyzer = BatteryAnalyzer(battery_data)
        
        # Test basic info
        basic_info = analyzer.get_basic_info()
        assert basic_info['cell_id'] == "test_battery_001"
        assert basic_info['total_cycles'] == 2
        assert basic_info['cathode_material'] == "LFP"
        
        # Test capacity statistics
        capacity_stats = analyzer.get_capacity_statistics()
        assert 'discharge_capacity' in capacity_stats
        
        # Test comprehensive analysis
        analysis = analyzer.get_comprehensive_analysis()
        assert 'basic_info' in analysis
        assert 'capacity_stats' in analysis
        
        # Test DataFrame conversion
        df = analyzer.to_dataframe()
        assert len(df) == 1
        assert 'basic_cell_id' in df.columns
        
        print("✓ BatteryAnalyzer working correctly")
        return True
    except Exception as e:
        print(f"✗ BatteryAnalyzer test error: {e}")
        return False

def test_dataset_analyzer():
    """Test DatasetAnalyzer with mock data."""
    print("Testing DatasetAnalyzer...")
    
    try:
        from batteryml.data_analysis.dataset_analyzer import DatasetAnalyzer
        from batteryml.data.battery_data import BatteryData, CycleData
        import tempfile
        import pickle
        
        # Create temporary directory with mock data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock battery files
            for i in range(3):
                cycle_data = [
                    CycleData(
                        cycle_number=1,
                        voltage_in_V=[3.0, 3.5, 4.0],
                        current_in_A=[1.0, 1.0, 1.0],
                        discharge_capacity_in_Ah=[0.0, 0.5, 1.0],
                        temperature_in_C=[25.0, 25.5, 26.0]
                    )
                ]
                
                battery_data = BatteryData(
                    cell_id=f"test_battery_{i:03d}",
                    cycle_data=cycle_data,
                    nominal_capacity_in_Ah=1.0,
                    cathode_material="LFP",
                    anode_material="graphite"
                )
                
                # Save battery data
                battery_file = temp_path / f"battery_{i:03d}.pkl"
                battery_data.dump(str(battery_file))
            
            # Test dataset analyzer
            analyzer = DatasetAnalyzer(str(temp_path))
            assert len(analyzer.battery_files) == 3
            
            # Test analysis
            summary_stats = analyzer.analyze_dataset(max_batteries=2, show_progress=False)
            assert summary_stats['total_batteries'] == 2
            assert summary_stats['successful_analyses'] == 2
            
            # Test feature summary
            features_df = analyzer.get_feature_summary_table()
            assert not features_df.empty
            
            print("✓ DatasetAnalyzer working correctly")
            return True
    except Exception as e:
        print(f"✗ DatasetAnalyzer test error: {e}")
        return False

def test_visualization():
    """Test visualization functions."""
    print("Testing visualization functions...")
    
    try:
        from batteryml.data_analysis.visualization import AnalysisVisualizer
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        visualizer = AnalysisVisualizer()
        
        # Test with mock data
        mock_summary = {
            'dataset_name': 'test_dataset',
            'total_batteries': 10,
            'successful_analyses': 10,
            'chemistry_distribution': {'LFP': 5, 'NMC': 5},
            'cycle_life_distribution': {
                'mean': 500.0,
                'median': 450.0,
                'std': 100.0,
                'min': 300.0,
                'max': 700.0
            },
            'dataset_stats': {
                'total_cycles': 5000,
                'avg_cycle_life': 500.0,
                'avg_nominal_capacity': 1.0
            }
        }
        
        # Test dataset overview (should not raise errors)
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer.plot_dataset_overview(mock_summary, save_path=Path(temp_dir) / "test.png")
        
        print("✓ Visualization functions working correctly")
        return True
    except Exception as e:
        print(f"✗ Visualization test error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running BatteryML Data Analysis Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_utils,
        test_battery_analyzer,
        test_dataset_analyzer,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The analysis module is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
