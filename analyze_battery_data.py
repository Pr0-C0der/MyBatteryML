#!/usr/bin/env python3
"""
Simple script to analyze battery data using the BatteryML data analysis tools.

This script provides an easy way to analyze battery datasets and generate
comprehensive reports and visualizations.

Usage:
    python analyze_battery_data.py
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.analyzer import BatteryDataAnalyzer
from batteryml.data_analysis.visualization import BatteryDataVisualizer


def analyze_dataset(data_path: str, output_dir: str = "analysis_results"):
    """
    Analyze a battery dataset and generate comprehensive reports.
    
    Args:
        data_path: Path to directory containing .pkl battery data files
        output_dir: Directory to save analysis results
    """
    print("="*80)
    print("BATTERY DATA ANALYSIS")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    try:
        # Initialize analyzer
        print("\n1. Initializing analyzer...")
        analyzer = BatteryDataAnalyzer(data_path)
        
        # Run dataset overview analysis
        print("\n2. Analyzing dataset overview...")
        dataset_stats = analyzer.analyze_dataset_overview()
        
        # Run feature analysis
        print("\n3. Analyzing features...")
        feature_stats = analyzer.analyze_features()
        
        # Generate visualizations
        print("\n4. Generating visualizations...")
        # Create plots folder inside data_analysis folder
        plots_dir = "batteryml/data_analysis/analysis_results"
        visualizer = BatteryDataVisualizer(plots_dir, analyzer.dataset_name)
        visualizer.save_all_plots(dataset_stats, feature_stats)
        
        # Save detailed results
        print("\n5. Saving detailed results...")
        analyzer.save_analysis(output_dir)
        
        # Generate and save summary report
        print("\n6. Generating summary report...")
        summary_report = analyzer.generate_summary_report()
        
        with open(f"{output_dir}/summary_report.txt", "w") as f:
            f.write(summary_report)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Plots saved to: {output_dir}/plots")
        print(f"Summary report: {output_dir}/summary_report.txt")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Please check that:")
        print("1. The data path exists and contains .pkl files")
        print("2. The .pkl files are valid BatteryData objects")
        print("3. You have the required dependencies installed")
        return False


def main():
    """Main function with example usage."""
    print("Battery Data Analysis Tool")
    print("=" * 40)
    
    # Example data paths - modify these based on your setup
    example_paths = [
        "data/processed/MATR",
        "data/processed/CALCE", 
        "data/processed/HUST",
        "data/processed/SNL"
    ]
    
    print("\nExample data paths:")
    for i, path in enumerate(example_paths, 1):
        print(f"{i}. {path}")
    
    print("\nTo analyze your data:")
    print("1. Place your .pkl battery data files in a directory")
    print("2. Run: analyze_dataset('path/to/your/data', 'output_directory')")
    print("\nExample:")
    print("analyze_dataset('data/processed/MATR', 'matr_analysis')")
    
    # Check if any example paths exist
    for path in example_paths:
        if Path(path).exists():
            print(f"\nFound existing data at: {path}")
            print(f"Would you like to analyze it? (uncomment the line below)")
            print(f"# analyze_dataset('{path}', '{path}_analysis')")
            break
    else:
        print("\nNo example data paths found.")
        print("Please ensure you have battery data in .pkl format.")


if __name__ == "__main__":
    main()
