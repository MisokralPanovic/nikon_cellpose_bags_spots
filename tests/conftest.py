import numpy as np
import pytest
from spot_detector.config import PipelineConfig


@pytest.fixture
def make_config(tmp_path):
    def _make(**overrides):
        raw_data_dir = tmp_path / "data"
        raw_data_dir.mkdir(exist_ok=True)

        cellpose_model_path = tmp_path / "cellpose_model.ckpt"
        cellpose_model_path.touch()

        spotiflow_model_path = tmp_path / "spotiflow_model"
        spotiflow_model_path.mkdir(exist_ok=True)
        base = {
            "mode": {"do_3d": False},
            "paths": {
                "raw_data_dir": str(raw_data_dir),
                "out_dir": str(tmp_path / "output"),
            },
            "channels": {"segmentation_image": 0, "spot_image": 1},
            "segmentation": {
                "use_default_model": False,
                "cellpose_model_path": str(cellpose_model_path),
                "use_gpu": True,
                "bin_factor": 4,
                "stitch_threshold": 0.4,
            },
            "detection": {
                "use_default_model": False,
                "spotiflow_model_path": str(spotiflow_model_path),
                "prob_thresh": 0.3,
                "min_distance": 1,
            },
        }
        merged = {**base, **overrides}
        return PipelineConfig(**merged)

    return _make


@pytest.fixture
def make_stack():
    def _make(shape):
        return np.random.rand(*shape).astype(np.float32)

    return _make
