#!/usr/bin/env python3
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Main script to run correlation analysis for all datasets.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import batteryml modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from batteryml.data_analysis.correlation_analyzer import CorrelationAnalyzer


def run_correlation_analysis(dataset_name: str, data_path: str, output_dir: str = None):
    """
    Run correlation analysis for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (CALCE, HUST, MATR, SNL, HNEI, RWTH, UL_PUR, OX)
        data_path: Path to the processed data directory
        output_dir: Output directory for results (optional)
    """
    
    # Available datasets
    available_datasets = ['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX']
    
    if dataset_name.upper() not in available_datasets:
        print(f"Error: Unknown dataset '{dataset_name}'. Available datasets: {available_datasets}")
        return False
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"{dataset_name.lower()}_correlation_analysis"
    
    print(f"Starting correlation analysis for {dataset_name} dataset...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    try:
        # Create analyzer instance
        analyzer = CorrelationAnalyzer(data_path, output_dir)
        
        # Run correlation analysis
        analyzer.analyze_dataset()
        
        print(f"\n{dataset_name} correlation analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during {dataset_name} correlation analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_correlation_analysis(base_data_path: str, base_output_dir: str = "all_correlation_analysis"):
    """
    Run correlation analysis for all available datasets.
    
    Args:
        base_data_path: Base path containing all dataset directories
        base_output_dir: Base output directory for all results
    """
    
    datasets = ['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX']
    base_path = Path(base_data_path)
    base_output = Path(base_output_dir)
    
    print(f"Running correlation analysis for all datasets...")
    print(f"Base data path: {base_path}")
    print(f"Base output directory: {base_output}")
    print("=" * 60)
    
    results = {}
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        output_path = base_output / f"{dataset.lower()}_correlation_analysis"
        
        if dataset_path.exists():
            print(f"\n{'='*20} {dataset} {'='*20}")
            success = run_correlation_analysis(dataset, str(dataset_path), str(output_path))
            results[dataset] = success
        else:
            print(f"\n{'='*20} {dataset} {'='*20}")
            print(f"Warning: Dataset path not found: {dataset_path}")
            results[dataset] = False
    
    # Print summary
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
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
        description='Battery Correlation Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze correlations for a specific dataset
  python run_correlation_analysis.py --dataset CALCE --data_path data/processed/CALCE
  
  # Analyze correlations for all datasets
  python run_correlation_analysis.py --all --data_path data/processed
  
  # Analyze with custom output directory
  python run_correlation_analysis.py --dataset MATR --data_path data/processed/MATR --output_dir my_correlation_analysis
        """
    )
    
    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, 
                      choices=['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX'],
                      help='Specific dataset to analyze')
    group.add_argument('--all', action='store_true',
                      help='Analyze correlations for all available datasets')
    
    # Path arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for correlation analysis')
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    # Run correlation analysis
    if args.all:
        results = run_all_correlation_analysis(str(data_path), args.output_dir or "all_correlation_analysis")
        
        # Exit with error code if any analysis failed
        if not all(results.values()):
            sys.exit(1)
    else:
        success = run_correlation_analysis(args.dataset, str(data_path), args.output_dir)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
