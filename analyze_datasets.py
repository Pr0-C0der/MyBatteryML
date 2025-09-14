#!/usr/bin/env python3
"""
Simple script to analyze battery datasets.
This script provides an easy interface to run data analysis on processed battery datasets.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.run_analysis import run_analysis, run_all_analyses


def main():
    """Main function with simple interface."""
    print("Battery Data Analysis Tool")
    print("=" * 40)
    
    # Check if data directory exists
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure you have processed data in the 'data/processed' directory.")
        print("You can process data using: batteryml preprocess <dataset> <raw_path> <processed_path>")
        return
    
    # List available datasets
    available_datasets = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and any(subdir.glob("*.pkl")):
            available_datasets.append(subdir.name)
    
    if not available_datasets:
        print(f"No processed datasets found in {data_dir}")
        print("Available subdirectories:")
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                print(f"  - {subdir.name}")
        return
    
    print(f"Found processed datasets: {', '.join(available_datasets)}")
    print()
    
    # Ask user what to do
    print("What would you like to do?")
    print("1. Analyze all datasets")
    print("2. Analyze a specific dataset")
    print("3. Generate combined plots only")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nAnalyzing all datasets...")
            results = run_all_analyses(str(data_dir), "analysis_output")
            
            # Print summary
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE")
            print("="*50)
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            print(f"Successfully analyzed: {successful}/{total} datasets")
            
            if successful > 0:
                print(f"\nResults saved to: analysis_output/")
                print("Check the subdirectories for plots and summary files.")
            break
            
        elif choice == "2":
            print(f"\nAvailable datasets: {', '.join(available_datasets)}")
            dataset = input("Enter dataset name: ").strip().upper()
            
            if dataset in available_datasets:
                dataset_path = data_dir / dataset
                output_dir = f"{dataset.lower()}_analysis"
                
                print(f"\nAnalyzing {dataset} dataset...")
                success = run_analysis(dataset, str(dataset_path), output_dir)
                
                if success:
                    print(f"\n{dataset} analysis completed successfully!")
                    print(f"Results saved to: {output_dir}/")
                else:
                    print(f"\n{dataset} analysis failed. Check the error messages above.")
            else:
                print(f"Error: Dataset '{dataset}' not found in {data_dir}")
            break
            
        elif choice == "3":
            print(f"\nAvailable datasets: {', '.join(available_datasets)}")
            dataset = input("Enter dataset name for combined plots: ").strip().upper()
            
            if dataset in available_datasets:
                dataset_path = data_dir / dataset
                output_dir = f"{dataset.lower()}_combined_plots"
                
                print(f"\nGenerating combined plots for {dataset} dataset...")
                try:
                    from batteryml.data_analysis.combined_plots import CombinedPlotGenerator
                    from batteryml.data_analysis.base_analyzer import BaseDataAnalyzer
                    
                    # Create analyzer
                    analyzer = BaseDataAnalyzer(str(dataset_path), output_dir)
                    battery_files = analyzer.get_battery_files()
                    
                    if battery_files:
                        # Create combined plot generator
                        plot_generator = CombinedPlotGenerator(Path(output_dir), num_batteries=20)
                        plot_generator.generate_combined_plots(battery_files, analyzer)
                        
                        print(f"\n{dataset} combined plots completed successfully!")
                        print(f"Results saved to: {output_dir}/combined_plots/")
                    else:
                        print(f"No battery files found in {dataset_path}")
                except Exception as e:
                    print(f"Error generating combined plots: {e}")
            else:
                print(f"Error: Dataset '{dataset}' not found in {data_dir}")
            break
            
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
