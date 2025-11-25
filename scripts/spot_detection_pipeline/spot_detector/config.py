from pathlib import Path
import logging
import yaml
import numpy as np

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file.

    Args:
        config (Path): _description_

    Returns:
        dict: _description_    
    """
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_paths(config: dict, experiment_folder: Path) -> dict:
    """_summary_

    Args:
        config (dict): _description_
        experiment_folder (Path): _description_

    Returns:
        dict: _description_
    """
    experiment_folder = experiment_folder.resolve()
    
    config['experiment']['folder'] = experiment_folder
    config['experiment']['name'] = experiment_folder.name # do I need this?
    
    # resolve paths
    config['paths']['raw_data'] = (experiment_folder / config['paths']['raw_data']).resolve()
    config['paths']['figures'] = (experiment_folder / config['paths']['figures']).resolve()
    config['paths']['processed_data'] = (experiment_folder / config['paths']['processed_data']).resolve()
    
    cellpose_models_path = experiment_folder / config['paths']['cellpose_models']
    config['paths']['cellpose_models'] = cellpose_models_path.resolve()
    
    # full cellpose models path
    model_name = config['segmentation']['model_name']
    config['paths']['model_file'] = (config['paths']['cellpose_models'] / model_name).resolve()
    
    # output directories
    config['paths']['figures'].mkdir(exist_ok=True, parents=True)
    config['paths']['processed_data'].mkdir(exist_ok=True, parents=True)
    
    if not config['paths']['raw_data'].exist():
        raise FileNotFoundError(f"Raw data folder not found: {config['paths']['raw_data']}")
    
    if not config['paths']['model_file'].exist():
        raise FileNotFoundError(f"Model file not found: {config['paths']['model_file']}")
    
    logger.info(f"Experiment: {config['experiment']['name']}")
    
    return config

def get_footprint(config: dict) -> np.ndarray:
    """Extract and convert footprint from config."""
    return np.array(config['detection']['footprint'], dtype=bool)

def get_channel_params(config: dict) -> dict:
    """Extract channel parameters."""
    return config['channels']

def get_segmentation_params(config: dict) -> dict:
    """Extract segmentation parameters."""
    return {
        'model_path': str(config['paths']['model_file']),
        'use_gpu': config['segmentation']['use_gpu'],
        'scale_factor': config['segmentation']['scale_factor'],
        'buffer_size': config['segmentation']['buffer_size']
    }
    
def get_detection_params(config: dict) -> dict:
    """Extract detection parameters."""
    return {'min_distance': config['detection']['min_distance']}

def get_analysis_params(config: dict) -> dict:
    """Extract analysis parameters."""
    return {
        'thickness_um': config['experiment']['thickness_um'],
        'min_roi_area_um2': config['analysis']['min_roi_area_um2']        
    }

def get_qc_params(config: dict) -> dict:
    """Extract qc parameters."""
    return {
        'save_figures': config['qc']['save_figures'],
        'dpi': config['qc']['dpi'],
        'figsize': tuple(config['qc']['figsize'])        
    }