#!/usr/bin/env python3
"""
Test script for battery data analysis.
This script tests the analysis functionality with a small sample.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.base_analyzer import BaseDataAnalyzer


def test_base_analyzer():
    """Test the base analyzer functionality."""
    print("Testing BaseDataAnalyzer...")
    
    # Create a test analyzer
    analyzer = BaseDataAnalyzer("data/processed/MATR", "test_analysis")
    
    # Test directory creation
    assert analyzer.output_dir.exists(), "Output directory not created"
    assert analyzer.capacity_fade_dir.exists(), "Capacity fade directory not created"
    assert analyzer.voltage_capacity_dir.exists(), "Voltage capacity directory not created"
    assert analyzer.qc_qd_dir.exists(), "QC QD directory not created"
    assert analyzer.current_time_dir.exists(), "Current time directory not created"
    assert analyzer.voltage_time_dir.exists(), "Voltage time directory not created"
    
    print("✓ Directory structure created successfully")
    
    # Test file discovery
    battery_files = analyzer.get_battery_files()
    print(f"Found {len(battery_files)} battery files")
    
    if battery_files:
        # Test loading a single battery
        battery = analyzer.load_battery_data(battery_files[0])
        if battery:
            print(f"✓ Successfully loaded battery: {battery.cell_id}")
            
            # Test capacity fade calculation
            cycles, capacities = analyzer.calculate_capacity_fade(battery)
            print(f"✓ Capacity fade calculated: {len(cycles)} cycles, {len(capacities)} capacity values")
            
            # Test statistics calculation
            stats = analyzer.calculate_dataset_statistics([battery])
            print(f"✓ Statistics calculated: {stats['total_batteries']} batteries")
            
        else:
            print("⚠ Could not load battery data")
    else:
        print("⚠ No battery files found")
    
    print("Base analyzer test completed!")


def test_imports():
    """Test that all analyzers can be imported."""
    print("Testing imports...")
    
    try:
        from batteryml.data_analysis import (
            CALCEAnalyzer, HUSTAnalyzer, MATRAnalyzer, SNLAnalyzer,
            HNEIAnalyzer, RWTHAnalyzer, UL_PURAnalyzer, OXAnalyzer
        )
        print("✓ All analyzers imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Battery Data Analysis Test Suite")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("Import test failed. Exiting.")
        return
    
    # Test base analyzer
    test_base_analyzer()
    
    print("\n" + "=" * 40)
    print("All tests completed!")


if __name__ == "__main__":
    main()
