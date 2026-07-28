import re
from dataclasses import dataclass
from cellpose import models
from spotiflow.model import Spotiflow
import logging


logger = logging.getLogger(__name__)


def parse_condition_from_name(filename_stem: str) -> str:
    """Extracts the base condition name from a filename stem.

    Examples:
        'Control_01'         -> 'Control'
        'Treated-DrugA_FOV3' -> 'Treated-DrugA'
        'WT_high_res'        -> 'WT_high_res'
    """
    return re.sub(r"[-_][^-_]*?\d+$", "", filename_stem)


@dataclass
class ModelBundle:
    """Container for Cellpose and Spotiflow models with validated loading

    Use ModelBundle.load() to construct.

    Attributes:
        cellpose (models.CellposeModel): Loaded CellposeModel instance.
        spotiflow (Spotiflow): Loaded Spotiflow instance, validated against pipeline mode.
    """

    cellpose: models.CellposeModel
    spotiflow: Spotiflow

    @classmethod
    def load(cls, config: dict, do_3d: bool) -> "ModelBundle":
        """Load and validate both models from config.

        Args:
            config (dict): Pipeline config dict containing cellpose_models_path and spotiflow_models_path.
            do_3d (bool): Whether the pipeline is running in 3D mode.

        Returns:
            ModelBundle: Validated ModelBundle.
        """
        cellpose = cls._load_cellpose(config=config)
        spotiflow = cls._load_spotiflow(config=config, do_3d=do_3d)
        return cls(cellpose=cellpose, spotiflow=spotiflow)

    @staticmethod
    def _load_cellpose(config: dict) -> models.CellposeModel:
        logger.debug("Loading Cellpose model...")
        return models.CellposeModel(
            gpu=config["segmentation"]["use_gpu"],
            pretrained_model=str(config["paths"]["cellpose_models_path"]),
        )

    @staticmethod
    def _load_spotiflow(config: dict, do_3d: bool) -> Spotiflow:
        logger.debug("Loading Spotiflow model...")
        model = ModelBundle._load_spotiflow_from_config(config=config, do_3d=do_3d)
        model = ModelBundle._validate_spotiflow_mode(model=model, do_3d=do_3d)
        return model

    @staticmethod
    def _load_spotiflow_from_config(config: dict, do_3d: bool) -> Spotiflow:
        """Attempts to load Spotiflow model from config path, falling back to pretrained default on failure."""
        try:
            model = Spotiflow.from_folder(str(config["paths"]["spotiflow_models_path"]))
            logger.info(
                f"Loaded custom Spotiflow model from: {config['paths']['spotiflow_models_path']}"
            )
            return model
        except Exception as e:
            fallback = "smfish_3d" if do_3d else "synth_complex"
            logger.warning(
                f"Custom model failed ({e}), falling back to {fallback}...",
                exc_info=True,
            )
            return Spotiflow.from_pretrained(fallback)

    @staticmethod
    def _validate_spotiflow_mode(model: Spotiflow, do_3d: bool) -> Spotiflow:
        """Checks model dimentionality matches pipeline mode, replacing with  pretrained default if not."""
        model_is_3d = model.config.is_3d
        if model_is_3d == do_3d:
            return model

        mode_str = "3D" if do_3d else "2D"
        fallback = "smfish_3d" if do_3d else "synth_complex"
        logger.warning(
            f"Mode conflict: model is {'3D' if model_is_3d else '2D'} "
            f"but pipeline is {mode_str}. Overriding with {fallback}..."
        )
        return Spotiflow.from_pretrained(fallback)
