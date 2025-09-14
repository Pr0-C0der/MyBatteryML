#!/usr/bin/env python3
"""
Simple script to run streaming battery data analysis.
This script processes battery pkl files one at a time without loading all data into memory.
"""

import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.streaming_analyzer import StreamingBatteryDataAnalyzer
from batteryml.data_analysis.visualization import BatteryDataVisualizer


def main():
    """Run streaming analysis on battery data."""
    print("="*80)
    print("STREAMING BATTERY DATA ANALYSIS")
    print("="*80)
    print("This analysis processes battery files one at a time for memory efficiency.")
    print()
    
    # Get data path from user or use default
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = input("Enter path to battery data directory: ").strip()
        if not data_path:
            print("No path provided. Exiting.")
            return
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Error: Path does not exist: {data_path}")
        return
    
    if not data_path.is_dir():
        print(f"Error: Path is not a directory: {data_path}")
        return
    
    # Check for pkl files
    pkl_files = list(data_path.glob("*.pkl"))
    if not pkl_files:
        print(f"Error: No .pkl files found in {data_path}")
        return
    
    print(f"Found {len(pkl_files)} .pkl files in {data_path}")
    print()
    
    try:
        # Initialize streaming analyzer
        print("Initializing streaming analyzer...")
        analyzer = StreamingBatteryDataAnalyzer(str(data_path))
        
        # Run complete analysis
        print("Running streaming analysis...")
        dataset_stats, feature_stats = analyzer.run_complete_analysis()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plots_dir = Path("batteryml/data_analysis/analysis_results")
        visualizer = BatteryDataVisualizer(str(plots_dir), analyzer.dataset_name)
        visualizer.save_all_plots(dataset_stats, feature_stats)
        
        print("\n" + "="*80)
        print("STREAMING ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Dataset: {analyzer.dataset_name}")
        print(f"Total batteries processed: {dataset_stats.get('total_batteries', 0)}")
        print(f"Features analyzed: {len(feature_stats)}")
        print(f"Plots saved to: {plots_dir / analyzer.dataset_name}")
        print()
        print("This analysis used memory-efficient streaming processing.")
        print("Each battery file was processed individually without loading all data at once.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

