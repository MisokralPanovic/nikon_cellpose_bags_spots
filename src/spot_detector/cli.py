from pathlib import Path

from spot_detector.config import load_config
from spot_detector.run_pipeline import run_pipeline

def main():
    """CLI entry point: load config from path, run pipeline."""
    import argparse
    from spot_detector.config import load_config

    parser = argparse.ArgumentParser(description="Run segmentation + spot detection pipeline")
    parser.add_argument("config_path", type=Path, help="Path to config.yml")
    args = parser.parse_args()

    config = load_config(args.config_path)
    run_pipeline(config=config)


if __name__ == "__main__":
    main()