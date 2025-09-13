#!/usr/bin/env python3
"""
Demonstration script showing the new folder structure for battery data analysis.

This script demonstrates how the analysis tools now save images in dataset-specific
folders within the data_analysis/analysis_results directory.
"""

import sys
import tempfile
import pickle
from pathlib import Path
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from batteryml.data_analysis.analyzer import BatteryDataAnalyzer
from batteryml.data_analysis.visualization import BatteryDataVisualizer
from batteryml.data.battery_data import BatteryData, CycleData


def create_demo_battery_data(dataset_name: str, battery_id: str):
    """Create demo battery data for testing."""
    cycles = []
    for i in range(1, 6):  # 5 cycles
        voltage = np.linspace(3.0, 4.2, 50) + np.random.normal(0, 0.01, 50)
        current = np.random.normal(1.0, 0.1, 50)
        capacity = np.linspace(0, 1.1 - i*0.02, 50)
        time = np.linspace(0, 1800, 50)
        temperature = np.random.normal(25, 2, 50)
        
        cycle = CycleData(
            cycle_number=i,
            voltage_in_V=voltage.tolist(),
            current_in_A=current.tolist(),
            charge_capacity_in_Ah=capacity.tolist(),
            discharge_capacity_in_Ah=capacity.tolist(),
            time_in_s=time.tolist(),
            temperature_in_C=temperature.tolist()
        )
        cycles.append(cycle)
    
    battery = BatteryData(
        cell_id=f"{dataset_name}_{battery_id}",
        cycle_data=cycles,
        nominal_capacity_in_Ah=1.1,
        cathode_material=["LCO", "LFP", "NMC", "NCA"][hash(battery_id) % 4],
        anode_material="graphite",
        form_factor="cylindrical_18650"
    )
    
    return battery


def demo_folder_structure():
    """Demonstrate the new folder structure."""
    print("="*80)
    print("BATTERY DATA ANALYSIS - FOLDER STRUCTURE DEMO")
    print("="*80)
    
    # Create temporary directories for different datasets
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple datasets
        datasets = {
            "MATR": ["battery_001", "battery_002", "battery_003"],
            "CALCE": ["cell_001", "cell_002"],
            "HUST": ["test_001", "test_002", "test_003", "test_004"]
        }
        
        print("\n1. Creating demo datasets...")
        for dataset_name, battery_ids in datasets.items():
            dataset_dir = temp_path / dataset_name
            dataset_dir.mkdir()
            
            for battery_id in battery_ids:
                battery = create_demo_battery_data(dataset_name, battery_id)
                battery_file = dataset_dir / f"{battery_id}.pkl"
                with open(battery_file, 'wb') as f:
                    pickle.dump(battery.to_dict(), f)
            
            print(f"   Created {dataset_name}: {len(battery_ids)} batteries")
        
        # Analyze each dataset
        print("\n2. Analyzing each dataset...")
        for dataset_name in datasets.keys():
            print(f"\n   Analyzing {dataset_name} dataset...")
            
            # Initialize analyzer
            analyzer = BatteryDataAnalyzer(str(temp_path / dataset_name))
            
            # Run analysis
            dataset_stats = analyzer.analyze_dataset_overview()
            feature_stats = analyzer.analyze_features()
            
            # Create visualizations in dataset-specific folder
            plots_dir = "batteryml/data_analysis/analysis_results"
            visualizer = BatteryDataVisualizer(plots_dir, dataset_name)
            visualizer.save_all_plots(dataset_stats, feature_stats)
            
            print(f"   ✓ {dataset_name} analysis completed")
        
        # Show the folder structure
        print("\n3. Generated folder structure:")
        print("   batteryml/data_analysis/analysis_results/")
        
        analysis_results_dir = Path("batteryml/data_analysis/analysis_results")
        if analysis_results_dir.exists():
            for dataset_dir in sorted(analysis_results_dir.iterdir()):
                if dataset_dir.is_dir():
                    print(f"   ├── {dataset_dir.name}/")
                    for plot_file in sorted(dataset_dir.glob("*.png")):
                        print(f"   │   ├── {plot_file.name}")
        else:
            print("   (No analysis results generated yet)")
        
        print("\n4. Summary:")
        print("   ✓ Images are saved in dataset-specific folders")
        print("   ✓ No plots are displayed (plt.close() used)")
        print("   ✓ Organized structure: analysis_results/DATASET_NAME/")
        print("   ✓ Each dataset gets its own folder with all plots")


if __name__ == "__main__":
    demo_folder_structure()


