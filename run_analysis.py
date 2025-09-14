#!/usr/bin/env python3
"""
Convenience script to run battery dataset analysis.

This script provides easy access to the data analysis functionality.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the main analysis script
from batteryml.data_analysis.analyze_datasets import main

if __name__ == "__main__":
    main()
