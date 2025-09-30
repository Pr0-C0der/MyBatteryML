"""
chemistry_data_analysis
-----------------------

Utilities and scripts for analyzing batteries grouped by chemistry.

Expected input directory layout (external to this package):

data_chemistries/
    lfp/                # MATR, SNL cells
    lco/                # CALCE cells
    nca/                # UL_PUR, SNL cells
    nmc/                # SNL, RWTH cells
    mixed_nmc_lco/      # HNEI cells

This package will host loaders and analysis routines that operate on the
above groupings to produce per-chemistry plots, statistics, and reports.
"""

__all__: list[str] = []


