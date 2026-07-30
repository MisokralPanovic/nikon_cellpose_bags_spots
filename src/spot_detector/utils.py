import logging
import re
from dataclasses import dataclass

from cellpose import models
from spotiflow.model import Spotiflow

from spot_detector.config import PipelineConfig

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
    def load(cls, config: PipelineConfig) -> "ModelBundle":
        """Load and validate both models from config.

        Args:
            config (PipelineConfig): Pipeline config PipelineConfig object containing cellpose_model_path and spotiflow_model_path.

        Returns:
            ModelBundle: Validated ModelBundle.
        """
        cellpose = cls._load_cellpose(config=config)
        spotiflow = cls._load_spotiflow(config=config)
        return cls(cellpose=cellpose, spotiflow=spotiflow)

    @staticmethod
    def _load_cellpose(config: PipelineConfig) -> models.CellposeModel:
        if not config.segmentation.use_default_model:
            logger.debug("Loading custom Cellpose model...")
            return models.CellposeModel(
                gpu=config.segmentation.use_gpu,
                pretrained_model=str(config.segmentation.cellpose_model_path),
            )

        if config.segmentation.cellpose_model_path is not None:
            logger.warning(
                f"Ignoring cellpose_model_path ({config.segmentation.cellpose_model_path}) "
                "since use_default_model is True"
            )
        logger.debug("Loading default Cellpose model...")
        return models.CellposeModel(gpu=config.segmentation.use_gpu)

    @staticmethod
    def _load_spotiflow(config: PipelineConfig) -> Spotiflow:
        logger.debug("Loading Spotiflow model...")
        model = ModelBundle._load_spotiflow_from_config(config=config)
        model = ModelBundle._validate_spotiflow_mode(model=model, config=config)
        return model

    @staticmethod
    def _load_spotiflow_from_config(config: PipelineConfig) -> Spotiflow:
        """Attempts to load Spotiflow model from config path, falling back to pretrained default on failure."""
        if not config.detection.use_default_model:
            try:
                model = Spotiflow.from_folder(
                    str(config.detection.spotiflow_model_path)
                )
                logger.info(
                    f"Loaded custom Spotiflow model from: {config.detection.spotiflow_model_path}"
                )
                return model
            except Exception as e:
                fallback = "smfish_3d" if config.mode.do_3d else "synth_complex"
                logger.warning(
                    f"Custom model failed ({e}), falling back to {fallback}...",
                    exc_info=True,
                )
                return Spotiflow.from_pretrained(fallback)
        if config.detection.spotiflow_model_path is not None:
            logger.warning(
                f"Ignoring spotiflow_model_path ({config.detection.spotiflow_model_path}) "
                "since use_default_model is True"
            )
        logger.debug("Loading default Spotiflow model...")
        fallback = "smfish_3d" if config.mode.do_3d else "synth_complex"
        return Spotiflow.from_pretrained(fallback)

    @staticmethod
    def _validate_spotiflow_mode(model: Spotiflow, config: PipelineConfig) -> Spotiflow:
        """Checks model dimentionality matches pipeline mode, replacing with  pretrained default if not."""
        model_is_3d = model.config.is_3d
        if model_is_3d == config.mode.do_3d:
            return model

        mode_str = "3D" if config.mode.do_3d else "2D"
        fallback = "smfish_3d" if config.mode.do_3d else "synth_complex"
        logger.warning(
            f"Mode conflict: model is {'3D' if model_is_3d else '2D'} "
            f"but pipeline is {mode_str}. Overriding with {fallback}..."
        )
        return Spotiflow.from_pretrained(fallback)
