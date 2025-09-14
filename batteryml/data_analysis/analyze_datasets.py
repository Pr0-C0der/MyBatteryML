#!/usr/bin/env python3
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Main script for analyzing battery datasets.

This script provides a comprehensive analysis of battery datasets, including:
1. Total number of batteries per dataset
2. Feature analysis with statistics (min, max, mean, median, etc.)
3. Visualization of key insights
4. Export of analysis results

Usage:
    python analyze_datasets.py --dataset_path /path/to/dataset --output_dir /path/to/output
    python analyze_datasets.py --all_datasets --data_root /path/to/data --output_dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import warnings

# Add the parent directory to the path so we can import batteryml modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batteryml.data_analysis import DatasetAnalyzer, AnalysisVisualizer
from batteryml.data_analysis.utils import AnalysisUtils

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def analyze_single_dataset(dataset_path: str, output_dir: str, 
                          max_batteries: Optional[int] = None,
                          create_plots: bool = True) -> bool:
    """
    Analyze a single dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        output_dir: Directory to save analysis results
        max_batteries: Maximum number of batteries to analyze (None for all)
        create_plots: Whether to create visualization plots
        
    Returns:
        True if analysis was successful, False otherwise
    """
    try:
        print(f"\n{'='*80}")
        print(f"ANALYZING DATASET: {Path(dataset_path).name}")
        print(f"{'='*80}")
        
        # Initialize dataset analyzer
        analyzer = DatasetAnalyzer(dataset_path)
        
        # Check if dataset exists and has files
        if not analyzer.battery_files:
            print(f"Warning: No battery files found in {dataset_path}")
            return False
        
        print(f"Found {len(analyzer.battery_files)} battery files")
        
        # Analyze the dataset
        summary_stats = analyzer.analyze_dataset(max_batteries=max_batteries)
        
        # Print summary
        analyzer.print_dataset_summary()
        
        # Print feature summary
        analyzer.print_feature_summary(top_n=20)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save analysis results
        analyzer.save_analysis_results(output_path)
        
        # Create visualizations if requested
        if create_plots:
            print("\nCreating visualizations...")
            visualizer = AnalysisVisualizer()
            visualizer.create_comprehensive_report(analyzer, output_path)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing dataset {dataset_path}: {str(e)}")
        return False


def analyze_all_datasets(data_root: str, output_dir: str, 
                        max_batteries_per_dataset: Optional[int] = None,
                        create_plots: bool = True) -> None:
    """
    Analyze all datasets in the data root directory.
    
    Args:
        data_root: Root directory containing all datasets
        output_dir: Directory to save analysis results
        max_batteries_per_dataset: Maximum batteries to analyze per dataset
        create_plots: Whether to create visualization plots
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
    
    # Analyze each dataset
    successful_analyses = 0
    failed_analyses = 0
    
    for dataset_dir in dataset_dirs:
        dataset_output = output_dir / dataset_dir.name
        success = analyze_single_dataset(
            str(dataset_dir), 
            str(dataset_output),
            max_batteries=max_batteries_per_dataset,
            create_plots=create_plots
        )
        
        if success:
            successful_analyses += 1
        else:
            failed_analyses += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets: {len(dataset_dirs)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")
    print(f"Results saved to: {output_dir}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze battery datasets for comprehensive insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single dataset
  python analyze_datasets.py --dataset_path data/processed/MATR --output_dir results/MATR_analysis
  
  # Analyze all datasets in a directory
  python analyze_datasets.py --all_datasets --data_root data/processed --output_dir results
  
  # Analyze with limited batteries per dataset (for testing)
  python analyze_datasets.py --all_datasets --data_root data/processed --output_dir results --max_batteries 10
  
  # Analyze without creating plots (faster)
  python analyze_datasets.py --all_datasets --data_root data/processed --output_dir results --no_plots
        """
    )
    
    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset_path', type=str, 
                      help='Path to a single dataset directory to analyze')
    group.add_argument('--all_datasets', action='store_true',
                      help='Analyze all datasets in the data root directory')
    
    # Required arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    
    # Optional arguments
    parser.add_argument('--data_root', type=str,
                       help='Root directory containing all datasets (required with --all_datasets)')
    parser.add_argument('--max_batteries', type=int, default=None,
                       help='Maximum number of batteries to analyze per dataset (None for all)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip creating visualization plots (faster analysis)')
    
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
    
    # Run analysis
    if args.all_datasets:
        analyze_all_datasets(
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_batteries_per_dataset=args.max_batteries,
            create_plots=not args.no_plots
        )
    else:
        success = analyze_single_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            max_batteries=args.max_batteries,
            create_plots=not args.no_plots
        )
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
