#!/usr/bin/env python3
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Main script to run battery data analysis for all datasets.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import batteryml modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from batteryml.data_analysis import (
    CALCEAnalyzer, HUSTAnalyzer, MATRAnalyzer, SNLAnalyzer,
    HNEIAnalyzer, RWTHAnalyzer, UL_PURAnalyzer, OXAnalyzer
)


def run_analysis(dataset_name: str, data_path: str, output_dir: str = None):
    """
    Run analysis for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (CALCE, HUST, MATR, SNL, HNEI, RWTH, UL_PUR, OX)
        data_path: Path to the processed data directory
        output_dir: Output directory for results (optional)
    """
    
    # Map dataset names to analyzers
    analyzers = {
        'CALCE': CALCEAnalyzer,
        'HUST': HUSTAnalyzer,
        'MATR': MATRAnalyzer,
        'SNL': SNLAnalyzer,
        'HNEI': HNEIAnalyzer,
        'RWTH': RWTHAnalyzer,
        'UL_PUR': UL_PURAnalyzer,
        'OX': OXAnalyzer
    }
    
    if dataset_name.upper() not in analyzers:
        print(f"Error: Unknown dataset '{dataset_name}'. Available datasets: {list(analyzers.keys())}")
        return False
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"{dataset_name.lower()}_analysis"
    
    print(f"Starting analysis for {dataset_name} dataset...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    try:
        # Create analyzer instance
        analyzer_class = analyzers[dataset_name.upper()]
        analyzer = analyzer_class(data_path, output_dir)
        
        # Run analysis
        analyzer.analyze_dataset()
        
        print(f"\n{dataset_name} analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during {dataset_name} analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_analyses(base_data_path: str, base_output_dir: str = "all_analysis"):
    """
    Run analysis for all available datasets.
    
    Args:
        base_data_path: Base path containing all dataset directories
        base_output_dir: Base output directory for all results
    """
    
    datasets = ['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX']
    base_path = Path(base_data_path)
    base_output = Path(base_output_dir)
    
    print(f"Running analysis for all datasets...")
    print(f"Base data path: {base_path}")
    print(f"Base output directory: {base_output}")
    print("=" * 60)
    
    results = {}
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        output_path = base_output / f"{dataset.lower()}_analysis"
        
        if dataset_path.exists():
            print(f"\n{'='*20} {dataset} {'='*20}")
            success = run_analysis(dataset, str(dataset_path), str(output_path))
            results[dataset] = success
        else:
            print(f"\n{'='*20} {dataset} {'='*20}")
            print(f"Warning: Dataset path not found: {dataset_path}")
            results[dataset] = False
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{dataset:10} : {status}")
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Battery Data Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a specific dataset
  python run_analysis.py --dataset CALCE --data_path data/processed/CALCE
  
  # Analyze all datasets
  python run_analysis.py --all --data_path data/processed
  
  # Analyze with custom output directory
  python run_analysis.py --dataset MATR --data_path data/processed/MATR --output_dir my_analysis
        """
    )
    
    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, 
                      choices=['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX'],
                      help='Specific dataset to analyze')
    group.add_argument('--all', action='store_true',
                      help='Analyze all available datasets')
    
    # Path arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    # Run analysis
    if args.all:
        results = run_all_analyses(str(data_path), args.output_dir or "all_analysis")
        
        # Exit with error code if any analysis failed
        if not all(results.values()):
            sys.exit(1)
    else:
        success = run_analysis(args.dataset, str(data_path), args.output_dir)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
