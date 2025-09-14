#!/usr/bin/env python3
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Standalone script to generate capacity vs cycles plots for battery datasets.

This script creates a merged capacity vs cycles graph showing all batteries
in a dataset on a single plot with individual curves and average trend.

Usage:
    python plot_capacity_vs_cycles.py --dataset_path /path/to/dataset --output_dir /path/to/output
    python plot_capacity_vs_cycles.py --all_datasets --data_root /path/to/data --output_dir /path/to/output
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import warnings

# Add the parent directory to the path so we can import batteryml modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batteryml.data_analysis import DatasetAnalyzer, AnalysisVisualizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def plot_capacity_vs_cycles_single_dataset(dataset_path: str, output_dir: str, 
                                         max_batteries: int = 100) -> bool:
    """
    Generate capacity vs cycles plot for a single dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        output_dir: Directory to save the plot
        max_batteries: Maximum number of batteries to plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*80}")
        print(f"GENERATING CAPACITY VS CYCLES PLOT: {Path(dataset_path).name}")
        print(f"{'='*80}")
        
        # Initialize dataset analyzer
        analyzer = DatasetAnalyzer(dataset_path)
        
        # Check if dataset exists and has files
        if not analyzer.battery_files:
            print(f"Warning: No battery files found in {dataset_path}")
            return False
        
        print(f"Found {len(analyzer.battery_files)} battery files")
        
        # Analyze the dataset (needed to get file paths)
        summary_stats = analyzer.analyze_dataset(max_batteries=max_batteries, show_progress=True)
        
        if not analyzer.analysis_results:
            print("No analysis results available")
            return False
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate the capacity vs cycles plot
        visualizer = AnalysisVisualizer()
        plot_path = output_path / f"{analyzer.dataset_name}_capacity_vs_cycles.png"
        
        print(f"\nGenerating capacity vs cycles plot...")
        visualizer.plot_capacity_vs_cycles(
            analyzer.analysis_results,
            save_path=str(plot_path),
            max_batteries_to_plot=max_batteries
        )
        
        print(f"\nCapacity vs cycles plot saved to: {plot_path}")
        return True
        
    except Exception as e:
        print(f"Error generating capacity vs cycles plot for {dataset_path}: {str(e)}")
        return False


def plot_capacity_vs_cycles_all_datasets(data_root: str, output_dir: str, 
                                       max_batteries_per_dataset: int = 100) -> None:
    """
    Generate capacity vs cycles plots for all datasets.
    
    Args:
        data_root: Root directory containing all datasets
        output_dir: Directory to save plots
        max_batteries_per_dataset: Maximum batteries to plot per dataset
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    if not data_root.exists():
        print(f"Error: Data root directory {data_root} does not exist")
        return
    
    # Find all dataset directories
    dataset_dirs = []
    for item in data_root.iterdir():
        if item.is_dir():
            # Check if it contains pickle files
            pkl_files = list(item.glob("*.pkl"))
            if pkl_files:
                dataset_dirs.append(item)
    
    if not dataset_dirs:
        print(f"No dataset directories with pickle files found in {data_root}")
        return
    
    print(f"Found {len(dataset_dirs)} dataset directories:")
    for dataset_dir in dataset_dirs:
        pkl_count = len(list(dataset_dir.glob("*.pkl")))
        print(f"  - {dataset_dir.name}: {pkl_count} files")
    
    # Generate plots for each dataset
    successful_plots = 0
    failed_plots = 0
    
    for dataset_dir in dataset_dirs:
        dataset_output = output_dir / dataset_dir.name
        success = plot_capacity_vs_cycles_single_dataset(
            str(dataset_dir), 
            str(dataset_output),
            max_batteries=max_batteries_per_dataset
        )
        
        if success:
            successful_plots += 1
        else:
            failed_plots += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"CAPACITY VS CYCLES PLOT GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets: {len(dataset_dirs)}")
    print(f"Successful plots: {successful_plots}")
    print(f"Failed plots: {failed_plots}")
    print(f"Plots saved to: {output_dir}")


def main():
    """Main function to run the capacity vs cycles plotting."""
    parser = argparse.ArgumentParser(
        description="Generate capacity vs cycles plots for battery datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plot for a single dataset
  python plot_capacity_vs_cycles.py --dataset_path data/processed/MATR --output_dir plots
  
  # Generate plots for all datasets
  python plot_capacity_vs_cycles.py --all_datasets --data_root data/processed --output_dir plots
  
  # Generate plots with limited batteries (for performance)
  python plot_capacity_vs_cycles.py --all_datasets --data_root data/processed --output_dir plots --max_batteries 50
        """
    )
    
    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_path', type=str, 
                      help='Path to a single dataset directory to plot')
    group.add_argument('--all_datasets', action='store_true',
                      help='Generate plots for all datasets in the data root directory')
    
    # Required arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the plots')
    
    # Optional arguments
    parser.add_argument('--data_root', type=str,
                       help='Root directory containing all datasets (required with --all_datasets)')
    parser.add_argument('--max_batteries', type=int, default=100,
                       help='Maximum number of batteries to plot per dataset (default: 100)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.all_datasets and not args.data_root:
        parser.error("--data_root is required when using --all_datasets")
    
    if args.dataset_path and not Path(args.dataset_path).exists():
        parser.error(f"Dataset path {args.dataset_path} does not exist")
    
    if args.data_root and not Path(args.data_root).exists():
        parser.error(f"Data root {args.data_root} does not exist")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.all_datasets:
        plot_capacity_vs_cycles_all_datasets(
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_batteries_per_dataset=args.max_batteries
        )
    else:
        success = plot_capacity_vs_cycles_single_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            max_batteries=args.max_batteries
        )
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
