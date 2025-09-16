#!/usr/bin/env python3
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Main script to run cycle plotting for all datasets.
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import batteryml modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from batteryml.data_analysis.cycle_plotter import CyclePlotter


def run_cycle_plots(dataset_name: str, data_path: str, output_dir: str = None, cycle_gap: int = 100):
    """
    Run cycle plotting for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (CALCE, HUST, MATR, SNL, HNEI, RWTH, UL_PUR, OX)
        data_path: Path to the processed data directory
        output_dir: Output directory for results (optional)
        cycle_gap: Gap between cycles to plot (default: 100)
    """
    
    # Available datasets
    available_datasets = ['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX']
    
    if dataset_name.upper() not in available_datasets:
        print(f"Error: Unknown dataset '{dataset_name}'. Available datasets: {available_datasets}")
        return False
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"{dataset_name.lower()}_cycle_plots"
    
    print(f"Starting cycle plotting for {dataset_name} dataset...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cycle gap: {cycle_gap}")
    print("-" * 50)
    
    try:
        # Create plotter instance
        plotter = CyclePlotter(data_path, output_dir, cycle_gap)
        
        # Run cycle plotting
        plotter.plot_dataset_features()
        
        print(f"\n{dataset_name} cycle plotting completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during {dataset_name} cycle plotting: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_cycle_plots(base_data_path: str, base_output_dir: str = "all_cycle_plots", cycle_gap: int = 100):
    """
    Run cycle plotting for all available datasets.
    
    Args:
        base_data_path: Base path containing all dataset directories
        base_output_dir: Base output directory for all results
        cycle_gap: Gap between cycles to plot (default: 100)
    """
    
    datasets = ['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX']
    base_path = Path(base_data_path)
    base_output = Path(base_output_dir)
    
    print(f"Running cycle plotting for all datasets...")
    print(f"Base data path: {base_path}")
    print(f"Base output directory: {base_output}")
    print(f"Cycle gap: {cycle_gap}")
    print("=" * 60)
    
    results = {}
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        output_path = base_output / f"{dataset.lower()}_cycle_plots"
        
        if dataset_path.exists():
            print(f"\n{'='*20} {dataset} {'='*20}")
            success = run_cycle_plots(dataset, str(dataset_path), str(output_path), cycle_gap)
            results[dataset] = success
        else:
            print(f"\n{'='*20} {dataset} {'='*20}")
            print(f"Warning: Dataset path not found: {dataset_path}")
            results[dataset] = False
    
    # Print summary
    print("\n" + "="*60)
    print("CYCLE PLOTTING SUMMARY")
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
        description='Battery Cycle Plotting Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot cycles for a specific dataset
  python run_cycle_plots.py --dataset CALCE --data_path data/processed/CALCE
  
  # Plot cycles for all datasets
  python run_cycle_plots.py --all --data_path data/processed
  
  # Plot cycles with custom gap and output directory
  python run_cycle_plots.py --dataset MATR --data_path data/processed/MATR --output_dir my_cycle_plots --cycle_gap 50
        """
    )
    
    # Dataset selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, 
                      choices=['CALCE', 'HUST', 'MATR', 'SNL', 'HNEI', 'RWTH', 'UL_PUR', 'OX'],
                      help='Specific dataset to plot')
    group.add_argument('--all', action='store_true',
                      help='Plot cycles for all available datasets')
    
    # Path arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for cycle plots')
    parser.add_argument('--cycle_gap', type=int, default=100,
                       help='Gap between cycles to plot (default: 100)')
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    # Run cycle plotting
    if args.all:
        results = run_all_cycle_plots(str(data_path), args.output_dir or "all_cycle_plots", args.cycle_gap)
        
        # Exit with error code if any plotting failed
        if not all(results.values()):
            sys.exit(1)
    else:
        success = run_cycle_plots(args.dataset, str(data_path), args.output_dir, args.cycle_gap)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
