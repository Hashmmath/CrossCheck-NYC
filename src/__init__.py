"""
Brooklyn Crosswalk Detection QA System
======================================

A two-stage visual QA system for evaluating AI-derived crosswalk maps
from aerial imagery in Brooklyn, NYC.

Modules:
--------
- data: Data download and preprocessing utilities
- inference: Tile2Net model wrapper for inference
- evaluation: Metrics, calibration, and analysis tools
- utils: Geospatial and visualization utilities
"""

__version__ = "0.1.0"
__author__ = "ZebraSenseLads"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "outputs"