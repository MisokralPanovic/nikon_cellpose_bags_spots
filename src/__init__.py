"""
Spot Detection Pipeline

A pipeline for automated segmentation and spot detection in microscopy images.
Supports ND2 and CZI file formats with Cellpose segmentation and Spotiflow detection.
"""

__version__ = "0.1.0"
__author__ = "Michal Varga"

from process import run_pipeline
from config import load_config, setup_paths

__all__ = [
    "run_pipeline",
    "load_config",
    "setup_paths"
]