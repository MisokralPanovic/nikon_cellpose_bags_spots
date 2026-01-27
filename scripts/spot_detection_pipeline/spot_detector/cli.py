#!/usr/bin/env python3
"""
Command-line interface for spot detection pipeline.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import logging
import textwrap

package_dir = Path(__file__).parent.parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from spot_detector.config import load_config, setup_paths
from spot_detector.process import run_pipeline

def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure logging.

    Args:
        verbose (bool, optional): If True, set log level to DEBUG. Defaults to False.
        log_file (Optional[Path], optional): Optional path to log file. Defaults to None.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(exist_ok=True, parents=True)
        handlers.append(logging.FileHandler(log_file)) # type: ignore
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Spot Detection Pipeline for Microscopy Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
            # Run from experiment folder with default config
            python -m spot_detector
                
            # Specify custom config file
            python -m spot_detector.cli --config custom_config.yml

            # Run with verbose logging
            python -m spot_detector.cli --verbose

            # Specify experiment directory
            python -m spot_detector.cli --experiment-dir /path/to/experiment""")
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=None,
        help='Path to configuration YAML file (default: config.yml in package directory)'
    )
    
    parser.add_argument(
        '--experiment-dir', '-e',
        type=Path,
        default=None,
        help='Path to experiment directory (default: current working directory)'
    )
    
    parser.add_argument(
        '--log-file', '-l',
        type=Path,
        default=None,
        help='Path to log file (default: processed_data/pipeline.log)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG level) logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Spot Detection Pipeline v0.1.0'
    )

    args = parser.parse_args()

    experiment_folder = args.experiment_dir if args.experiment_dir else Path.cwd()
    experiment_folder = experiment_folder.resolve()

    if args.config:
        config_path = args.config
    else:
        # Default to config.yml in package directory
        config_path = Path(__file__).parent.parent / 'config.yml'

    if args.log_file:
        log_file = args.log_file
    else:
        log_file = experiment_folder / 'processed_data' / 'pipeline.log'

    # Setup logging
    setup_logging(verbose=args.verbose, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("="*60)
        logger.info("Spot Detection Pipeline")
        logger.info("="*60)
        logger.info(f"Experiment folder: {experiment_folder}")
        logger.info(f"Config file: {config_path}")
        logger.info(f"Log file: {log_file}")
        
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(config_path)
        config = setup_paths(config, experiment_folder)
        
        # Run pipeline
        results = run_pipeline(config)
        
        logger.info("="*60)
        logger.info(f"✅ Pipeline completed successfully!")
        logger.info(f"✅ Analyzed {len(results)} ROIs")
        logger.info(f"✅ Results: {config['paths']['processed_data']}")
        logger.info("="*60)
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    
    except Exception as e:
        logger.error("="*60)
        logger.error(f"Pipeline failed:")
        logger.error(f" {type(e).__name__}: {e}")
        logger.error("="*60)
        if args.verbose:
            logger.exception("Fill traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())