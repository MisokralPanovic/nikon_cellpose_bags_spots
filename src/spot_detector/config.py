from pathlib import Path
import yaml
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    ConfigDict,
    FilePath,
    model_validator,
)
from typing import Annotated


class ModeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    do_3d: bool = False


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_data_dir: DirectoryPath
    out_dir: Path


class ChannelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    segmentation_image: Annotated[int, Field(ge=0)]
    spot_image: Annotated[int, Field(ge=0)]


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_default_model: bool = False
    cellpose_model_path: FilePath | None = None
    use_gpu: bool = False
    bin_factor: Annotated[int, Field(gt=0)] = (
        4  # maybe do that it has to be dividsible by something
    )
    stitch_threshold: Annotated[float, Field(ge=0, le=1)] = 0.4

    @model_validator(mode="after")
    def check_cellpose_path_set_unless_default(self) -> "SegmentationConfig":
        if not self.use_default_model and self.cellpose_model_path is None:
            raise ValueError(
                "segmentation.cellpose_model_path must be set unless segmentation.use_default_model is true"
            )
        return self


class DetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_default_model: bool = False
    spotiflow_model_path: DirectoryPath | None = None
    prob_thresh: Annotated[float, Field(ge=0, le=1)] = 0.3
    min_distance: Annotated[int, Field(gt=0)] = 1

    @model_validator(mode="after")
    def check_spotiflow_path_set_unless_default(self) -> "DetectionConfig":
        if not self.use_default_model and self.spotiflow_model_path is None:
            raise ValueError(
                "detection.spotiflow_model_path must be set unless detection.use_default_model is true"
            )
        return self


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: ModeConfig
    paths: PathsConfig
    channels: ChannelConfig
    segmentation: SegmentationConfig
    detection: DetectionConfig


def load_config(config_path: Path) -> PipelineConfig:
    """Load YAML configuration file.

    Args:
        config (Path): Config path.

    Returns:
        PipelineConfig: Pydantic typechecked config.
    """

    with open(config_path) as f:
        return PipelineConfig(**yaml.safe_load(f))
