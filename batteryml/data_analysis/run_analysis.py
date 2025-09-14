#!/usr/bin/env python3
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Main script to run comprehensive battery data analysis.

This script analyzes battery datasets and provides:
1. Total number of batteries per dataset
2. Feature statistics (min, max, mean, median, etc.)
3. Comprehensive visualizations
4. Summary reports

Usage:
    python run_analysis.py --data_path /path/to/battery/data --output_dir /path/to/output
"""

import argparse
import sys
import os
from pathlib import Path
import warnings

# Add the parent directory to the path to import batteryml modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batteryml.data_analysis.analyzer import BatteryDataAnalyzer
from batteryml.data_analysis.streaming_analyzer import StreamingBatteryDataAnalyzer
from batteryml.data_analysis.visualization import BatteryDataVisualizer


def main():
    """Main function to run the battery data analysis."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Battery Data Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze MATR dataset
    python run_analysis.py --data_path data/processed/MATR --output_dir analysis_results
    
    # Analyze with custom settings
    python run_analysis.py --data_path data/processed/MATR --output_dir results --no_plots --save_csv
        """
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='Path to directory containing battery data (.pkl files)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='analysis_results',
        help='Directory to save analysis results (default: analysis_results)'
    )
    
    parser.add_argument(
        '--no_plots', 
        action='store_true',
        help='Skip generating plots (faster analysis)'
    )
    
    parser.add_argument(
        '--save_csv', 
        action='store_true',
        help='Save detailed CSV files with statistics'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--streaming', 
        action='store_true',
        help='Use streaming analysis (memory-efficient for large datasets)'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    if not data_path.is_dir():
        print(f"Error: Data path is not a directory: {data_path}")
        sys.exit(1)
    
    # Check for .pkl files
    pkl_files = list(data_path.glob("*.pkl"))
    if not pkl_files:
        print(f"Error: No .pkl files found in {data_path}")
        print("Please ensure the directory contains battery data files in .pkl format")
        sys.exit(1)
    
    print(f"Found {len(pkl_files)} .pkl files in {data_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        print("\n" + "="*60)
        if args.streaming:
            print("INITIALIZING STREAMING BATTERY DATA ANALYZER")
            print("(Memory-efficient processing - one file at a time)")
        else:
            print("INITIALIZING BATTERY DATA ANALYZER")
            print("(Loading all data into memory)")
        print("="*60)
        
        if args.streaming:
            analyzer = StreamingBatteryDataAnalyzer(str(data_path))
            # Run complete analysis with streaming
            dataset_stats, feature_stats = analyzer.run_complete_analysis()
            
            # Generate summary report
            print("\nGenerating summary report...")
            summary_report = f"""
BATTERY DATA ANALYSIS SUMMARY
============================
Dataset: {analyzer.dataset_name}
Total Batteries: {dataset_stats.get('total_batteries', 0)}
Features Analyzed: {len(feature_stats)}

Dataset Distribution:
{chr(10).join([f"  {k}: {v} batteries" for k, v in dataset_stats.get('datasets', {}).items()])}

Chemistry Distribution:
{chr(10).join([f"  {k}: {v} batteries" for k, v in dataset_stats.get('chemistries', {}).items()])}

Analysis completed using streaming approach (memory-efficient).
"""
            print(summary_report)
        else:
            analyzer = BatteryDataAnalyzer(str(data_path))
            
            # Run dataset overview analysis
            print("\nRunning dataset overview analysis...")
            dataset_stats = analyzer.analyze_dataset_overview()
            
            # Run feature analysis
            print("\nRunning feature analysis...")
            feature_stats = analyzer.analyze_features()
            
            # Generate summary report
            print("\nGenerating summary report...")
            summary_report = analyzer.generate_summary_report()
            print(summary_report)
        
        # Save analysis results
        if args.save_csv and not args.streaming:
            print("\nSaving detailed CSV files...")
            analyzer.save_analysis(str(output_dir))
        elif args.save_csv and args.streaming:
            print("\nCSV saving not available for streaming analysis (use regular analysis for CSV output)")
        
        # Generate visualizations
        if not args.no_plots:
            print("\nGenerating visualizations...")
            # Create plots folder inside data_analysis folder
            plots_dir = Path("batteryml/data_analysis/analysis_results")
            visualizer = BatteryDataVisualizer(str(plots_dir), analyzer.dataset_name)
            visualizer.save_all_plots(dataset_stats, feature_stats)
        else:
            print("\nSkipping plot generation (--no_plots flag set)")
        
        # Save summary report to file
        with open(output_dir / "summary_report.txt", "w") as f:
            f.write(summary_report)
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        
        if not args.no_plots:
            print(f"Plots saved to: {output_dir / 'plots'}")
        
        if args.save_csv:
            print(f"CSV files saved to: {output_dir}")
        
        print(f"Summary report: {output_dir / 'summary_report.txt'}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_quick_analysis(data_path: str, output_dir: str = "quick_analysis") -> None:
    """
    Run a quick analysis with default settings.
    
    Args:
        data_path: Path to battery data directory
        output_dir: Output directory for results
    """
    print("Running quick battery data analysis...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = BatteryDataAnalyzer(data_path)
        
        # Run analyses
        dataset_stats = analyzer.analyze_dataset_overview()
        feature_stats = analyzer.analyze_features()
        
        # Generate and save visualizations
        visualizer = BatteryDataVisualizer(str(output_path / "plots"))
        visualizer.save_all_plots(dataset_stats, feature_stats)
        
        # Save results
        analyzer.save_analysis(str(output_path))
        
        # Generate summary report
        summary_report = analyzer.generate_summary_report()
        with open(output_path / "summary_report.txt", "w") as f:
            f.write(summary_report)
        
        print(f"Quick analysis completed! Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during quick analysis: {str(e)}")
        raise


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    main()
