from pathlib import Path
import yaml

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file.

    Args:
        config (Path): Config path.

    Returns:
        dict: Config.    
    """
    
    with open(config_path) as f:
        return yaml.safe_load(f)