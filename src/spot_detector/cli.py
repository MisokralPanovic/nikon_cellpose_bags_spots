from pathlib import Path
import argparse
import logging
from datetime import datetime

from spot_detector.config import load_config
from spot_detector.run_pipeline import run_pipeline


def configure_logging(log_dir: Path, level=logging.INFO) -> None:
    """Configure logging for the pipeline: file (DEBUG+) and console (level+) handlers on the root logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname) -8s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=level)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    """CLI entry point: load config from path, run pipeline."""
    parser = argparse.ArgumentParser(
        description="Run segmentation + spot detection pipeline"
    )
    parser.add_argument("config_path", type=Path, help="Path to config.yml")
    args = parser.parse_args()

    config = load_config(args.config_path)

    configure_logging(log_dir=Path(config["paths"]["out_dir"]) / "logs")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded config from {args.config_path}: {config}")

    run_pipeline(config=config)


if __name__ == "__main__":
    main()
