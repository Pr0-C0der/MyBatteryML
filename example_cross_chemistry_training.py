#!/usr/bin/env python3
"""
Example script demonstrating cross-chemistry training for RUL prediction.

This script shows how to:
1. Train models on one chemistry (e.g., LFP)
2. Test on other chemistries (e.g., NMC, NCA)
3. Evaluate cross-chemistry generalization performance
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from batteryml.chemistry_data_analysis.cross_chemistry_training import CrossChemistryTrainer


def main():
    """Run a simple cross-chemistry training example."""
    
    print("Cross-Chemistry Training Example")
    print("=" * 40)
    
    # Define paths (adjust these to your actual data structure)
    train_chemistry = "data_chemistries/lfp"
    test_chemistries = [
        "data_chemistries/nmc",
        "data_chemistries/nca",
        "data_chemistries/lco"
    ]
    
    # Check if chemistry directories exist
    train_path = Path(train_chemistry)
    if not train_path.exists():
        print(f"❌ Training chemistry directory not found: {train_chemistry}")
        print("Please ensure the chemistry directories exist.")
        return
    
    missing_test_chemistries = []
    for test_chem in test_chemistries:
        if not Path(test_chem).exists():
            missing_test_chemistries.append(test_chem)
    
    if missing_test_chemistries:
        print(f"❌ Test chemistry directories not found: {missing_test_chemistries}")
        print("Available chemistries will be used.")
        test_chemistries = [chem for chem in test_chemistries if Path(chem).exists()]
    
    if not test_chemistries:
        print("❌ No valid test chemistry directories found!")
        return
    
    print(f"✓ Training chemistry: {train_chemistry}")
    print(f"✓ Test chemistries: {', '.join(test_chemistries)}")
    print()
    
    # Create trainer
    trainer = CrossChemistryTrainer(
        train_chemistry_path=train_chemistry,
        test_chemistry_paths=test_chemistries,
        output_dir="example_cross_chemistry_results",
        verbose=True,
        cycle_limit=100,  # Limit to first 100 cycles for faster execution
        use_gpu=False  # Set to True if you have GPU available
    )
    
    print("Starting cross-chemistry training...")
    print()
    
    try:
        # Train and evaluate
        results = trainer.train_and_evaluate()
        
        print("\n" + "=" * 50)
        print("CROSS-CHEMISTRY TRAINING COMPLETED!")
        print("=" * 50)
        
        # Display results summary
        print("\nResults Summary:")
        print("-" * 20)
        
        for metric_name, metric_results in results.items():
            print(f"\n{metric_name} Results:")
            print("-" * 15)
            
            # Find best performing model for each test chemistry
            for test_chem in test_chemistries:
                test_chem_name = Path(test_chem).name
                best_model = None
                best_score = float('inf')
                
                for model_name, model_results in metric_results.items():
                    if test_chem_name in model_results:
                        score = model_results[test_chem_name]
                        if not pd.isna(score) and score < best_score:
                            best_score = score
                            best_model = model_name
                
                if best_model:
                    print(f"  {test_chem_name}: {best_model} ({best_score:.3f})")
                else:
                    print(f"  {test_chem_name}: No valid results")
        
        print(f"\nDetailed results saved to: {trainer.output_dir}")
        print("Files generated:")
        print(f"  - {trainer.rmse_file.name}")
        print(f"  - {trainer.mae_file.name}")
        print(f"  - {trainer.mape_file.name}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Import pandas here to avoid issues if not available
    try:
        import pandas as pd
        main()
    except ImportError:
        print("❌ pandas is required but not installed.")
        print("Please install it with: pip install pandas")
        sys.exit(1)
