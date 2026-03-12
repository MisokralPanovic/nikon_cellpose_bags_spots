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

def setup_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    """Configure logging to console and optionally to file.

    Args:
        log_file (Path, optional): Optional path to log file. If provided, logs are written to both console and file. Defaults to None.
        verbose (bool): If True, show DEBUG level and library logs
    """
    handlers = [logging.StreamHandler()]
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s - %(message)s')
    )
    handlers.append(console_handler)
    
    if log_file:
        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler) # type: ignore
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Suppress noisy libraries in console (but keep in file)
    if not verbose:
        # Silence cellpose
        logging.getLogger('cellpose').setLevel(logging.WARNING)
        logging.getLogger('cellpose.core').setLevel(logging.WARNING)
        logging.getLogger('cellpose.models').setLevel(logging.WARNING)
        logging.getLogger('cellpose.transforms').setLevel(logging.WARNING)
        logging.getLogger('cellpose.dynamics').setLevel(logging.WARNING)
        
        # Silence spotiflow
        logging.getLogger('spotiflow').setLevel(logging.WARNING)
        logging.getLogger('spotiflow.model').setLevel(logging.WARNING)
        logging.getLogger('spotiflow.model.spotiflow').setLevel(logging.WARNING)
        
        # Silence matplotlib
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Only log detailed info to file, not console
    if log_file and not verbose:
        # Keep detailed logs in file
        for logger_name in ['cellpose', 'spotiflow']:
            logger = logging.getLogger(logger_name)
            # Remove console handler for these loggers
            logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
            # Add file handler
            logger.addHandler(file_handler) # type: ignore
        

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