#!/usr/bin/env python3
"""
Spot Detection Pipeline - Main Entry Point

Run from experiment folder:
    cd /path/to/experiment_folder
    python scripts/spot_detection_pipeline/run_pipeline.py

Or make executable and run:
    chmod +x scripts/spot_detection_pipeline/run_pipeline.py
    ./scripts/spot_detection_pipeline/run_pipeline.py
"""

import sys
import logging
from pathlib import Path

# add package to path if running as script
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from spot_detector.config import load_config, setup_paths
from spot_detector.process import run_pipeline

def setup_logging(log_file: Path | None = None) -> None:
    """Configure logging to console and optionally to file.

    Args:
        log_file (Path, optional): Optional path to log file. If provided, logs are written to both console and file. Defaults to None.
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(exist_ok=True, parents=True)
        handlers.append(logging.FileHandler(log_file)) # type: ignore
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def main():
    """Main entry point for the pipeline."""
    
    experiment_folder = Path.cwd()
    config_path = Path(__file__).parent / 'config.yml'
    
    log_file = experiment_folder / 'processed_data' / 'pipeline.log'
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    try:
        pass
    
    except:
        pass
