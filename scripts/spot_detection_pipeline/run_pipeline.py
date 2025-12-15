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
        logger.info("="*60)
        logger.info("Spot Detection Pipeline")
        logger.info("="*60)
        logger.info(f"Experiment folder: {experiment_folder}")
        logger.info(f"Config file: {config_path}")
        
        logger.info("Loading configuration...")
        config = load_config(config_path)
        config = setup_paths(config, experiment_folder)
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Raw data: {config['paths']['raw_data']}")
        logger.info(f"Output: {config['paths']['processed_data']}")
        
        results = run_pipeline(config)
        
        logger.info("="*60)
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Analyzed {len(results)} ROIs")
        logger.info(f"Results saved to: {config['paths']['processed_data']}")
        logger.info("="*60)
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    
    except Exception as e:
        logger.error("="*60)
        logger.error(f"Pipeline failed with error: {type(e).__name__}: {e}")
        logger.error("="*60)
        logger.exception("Full traceback:")
        return 1

if __name__ == '__main__':
    sys.exit(main())