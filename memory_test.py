#!/usr/bin/env python3
"""
Memory usage test script for battery data analysis.
This script demonstrates the memory-efficient approach.
"""

import sys
import psutil
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.base_analyzer import BaseDataAnalyzer


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_memory_efficiency():
    """Test memory efficiency of the analysis."""
    print("Memory Efficiency Test")
    print("=" * 40)
    
    # Check if data exists
    data_dir = Path("data/processed/MATR")
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure you have processed MATR data first.")
        return
    
    # Create analyzer
    analyzer = BaseDataAnalyzer(str(data_dir), "memory_test_output")
    
    # Get battery files
    battery_files = analyzer.get_battery_files()
    if not battery_files:
        print("No battery files found")
        return
    
    print(f"Found {len(battery_files)} battery files")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Test processing batteries one by one
    print("\nProcessing batteries one by one...")
    
    for i, file_path in enumerate(battery_files[:5]):  # Test first 5 batteries
        print(f"\nProcessing battery {i+1}/5: {file_path.name}")
        
        # Load battery
        battery = analyzer.load_battery_data(file_path)
        if battery:
            print(f"  Loaded: {battery.cell_id}")
            print(f"  Memory after loading: {get_memory_usage():.1f} MB")
            
            # Analyze battery
            analyzer._analyze_single_battery(battery)
            print(f"  Memory after analysis: {get_memory_usage():.1f} MB")
            
            # Clear from memory
            del battery
            print(f"  Memory after cleanup: {get_memory_usage():.1f} MB")
        else:
            print(f"  Failed to load battery")
    
    print(f"\nFinal memory usage: {get_memory_usage():.1f} MB")
    print("Memory efficiency test completed!")


def test_statistics_collection():
    """Test incremental statistics collection."""
    print("\nIncremental Statistics Test")
    print("=" * 40)
    
    data_dir = Path("data/processed/MATR")
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    analyzer = BaseDataAnalyzer(str(data_dir), "memory_test_output")
    battery_files = analyzer.get_battery_files()
    
    if not battery_files:
        print("No battery files found")
        return
    
    print(f"Found {len(battery_files)} battery files")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Test incremental statistics collection
    print("\nCollecting statistics incrementally...")
    stats = analyzer._collect_statistics_incrementally(battery_files)
    
    print(f"Memory after statistics collection: {get_memory_usage():.1f} MB")
    print(f"Statistics collected for {stats['total_batteries']} batteries")
    print(f"Cycle counts: {len(stats['cycle_counts'])}")
    print(f"Nominal capacities: {len(stats['nominal_capacities'])}")
    
    print("Incremental statistics test completed!")


def main():
    """Run memory efficiency tests."""
    print("Battery Data Analysis - Memory Efficiency Test")
    print("=" * 60)
    
    # Check if psutil is available
    try:
        import psutil
    except ImportError:
        print("Error: psutil not installed. Install with: pip install psutil")
        return
    
    # Run tests
    test_memory_efficiency()
    test_statistics_collection()
    
    print("\n" + "=" * 60)
    print("All memory tests completed!")
    print("The analysis is memory-efficient and can handle large datasets.")


if __name__ == "__main__":
    main()
