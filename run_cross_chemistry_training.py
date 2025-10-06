#!/usr/bin/env python3
"""
Launcher script for cross-chemistry training.
This script ensures the correct Python path is set before running the main training script.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the main script
if __name__ == '__main__':
    from batteryml.chemistry_data_analysis.cross_chemistry_training import main
    main()
