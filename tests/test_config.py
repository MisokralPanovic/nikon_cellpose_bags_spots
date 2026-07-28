import yaml
import pytest
from spot_detector.config import (
    load_config,
    PathsConfig,
    ChannelConfig,
    SegmentationConfig,
    DetectionConfig,
    PipelineConfig,
)
from pydantic import ValidationError


@pytest.fixture
def full_config_dict(tmp_path):
    """Full config required for load_config()"""
    raw_data_dir = tmp_path / "data"
    raw_data_dir.mkdir()

    cellpose_model_path = tmp_path / "cellpose_model.ckpt"
    cellpose_model_path.touch()

    spotiflow_model_path = tmp_path / "spotiflow_model"
    spotiflow_model_path.mkdir()

    config_dic = {
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
    return config_dic


class TestYAML:
    def test_load_config_reads_yaml(self, tmp_path, full_config_dict):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump(full_config_dict))

        result = load_config(config_file)

        assert result.paths.raw_data_dir == tmp_path / "data"
        assert result.channels.spot_image == 1
        assert result.detection.use_default_model is False

    def test_load_config_raises_on_missing_file(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yml"

        with pytest.raises(FileNotFoundError):
            load_config(missing_path)

    def test_load_config_raises_on_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "bad_config.yml"
        config_file.write_text("paths: [unclosed\n  this is not valid: yaml:")

        with pytest.raises(Exception):
            load_config(config_file)


class TestDefaultModelsLoadLogic:
    def test_segmentation_loads_default_model_with_no_path(self, full_config_dict):
        full_config_dict["segmentation"]["use_default_model"] = True
        full_config_dict["segmentation"]["cellpose_model_path"] = None

        segmentation = SegmentationConfig(**full_config_dict["segmentation"])

        assert segmentation.cellpose_model_path is None

    def test_segmentation_raises_without_path_and_default_model(self, full_config_dict):
        full_config_dict["segmentation"]["use_default_model"] = False
        full_config_dict["segmentation"]["cellpose_model_path"] = None

        with pytest.raises(ValueError):
            SegmentationConfig(**full_config_dict["segmentation"])

    def test_segmentation_default_model_true_with_path(self, full_config_dict):
        full_config_dict["segmentation"]["use_default_model"] = True

        segmentation = SegmentationConfig(**full_config_dict["segmentation"])

        assert segmentation.cellpose_model_path is not None

    def test_detection_loads_default_model_with_no_path(self, full_config_dict):
        full_config_dict["detection"]["use_default_model"] = True
        full_config_dict["detection"]["spotiflow_model_path"] = None

        detection = DetectionConfig(**full_config_dict["detection"])

        assert detection.spotiflow_model_path is None

    def test_detection_raises_without_path_and_default_model(self, full_config_dict):
        full_config_dict["detection"]["use_default_model"] = False
        full_config_dict["detection"]["spotiflow_model_path"] = None

        with pytest.raises(ValueError):
            DetectionConfig(**full_config_dict["detection"])

    def test_detection_default_model_true_with_path(self, full_config_dict):
        full_config_dict["detection"]["use_default_model"] = True

        detection = DetectionConfig(**full_config_dict["detection"])

        assert detection.spotiflow_model_path is not None


class TestFieldConstraints:
    @pytest.mark.parametrize(
        "section, field, invalid_value",
        [
            ("channels", "segmentation_image", -1),
            ("channels", "spot_image", -1),
            ("segmentation", "bin_factor", 0),
            ("segmentation", "stitch_threshold", -0.1),
            ("segmentation", "stitch_threshold", 1.1),
            ("detection", "prob_thresh", -0.1),
            ("detection", "prob_thresh", 1.1),
            ("detection", "min_distance", 0),
        ],
    )
    def test_field_constraint_raises(
        self, full_config_dict, section, field, invalid_value
    ):
        full_config_dict[section][field] = invalid_value
        model_cls = {
            "channels": ChannelConfig,
            "segmentation": SegmentationConfig,
            "detection": DetectionConfig,
        }[section]

        with pytest.raises(ValidationError):
            model_cls(**full_config_dict[section])


class TestDefaults:
    @pytest.mark.parametrize(
        "section, field, expected_default",
        [
            ("mode", "do_3d", False),
            ("segmentation", "use_gpu", False),
            ("segmentation", "bin_factor", 4),
            ("segmentation", "stitch_threshold", 0.4),
            ("detection", "prob_thresh", 0.3),
            ("detection", "min_distance", 1),
        ],
    )
    def test_field_uses_default_when_omitted(
        self, full_config_dict, section, field, expected_default
    ):
        del full_config_dict[section][field]

        result = PipelineConfig(**full_config_dict)

        assert getattr(getattr(result, section), field) == expected_default


class TestPathValidation:
    @pytest.mark.parametrize(
        "section, field",
        [
            ("paths", "raw_data_dir"),
            ("segmentation", "cellpose_model_path"),
            ("detection", "spotiflow_model_path"),
        ],
    )
    def test_raises_on_nonexistent_path(
        self, tmp_path, full_config_dict, section, field
    ):
        full_config_dict[section][field] = str(tmp_path / "does_not_exist")

        with pytest.raises(ValidationError):
            PipelineConfig(**full_config_dict)

    @pytest.mark.parametrize(
        "section, field, wrong_kind",
        [
            ("paths", "raw_data_dir", "file"),  # expects a directory, given a file
            (
                "segmentation",
                "cellpose_model_path",
                "dir",
            ),  # expects a file, given a directory
            (
                "detection",
                "spotiflow_model_path",
                "file",
            ),  # expects a directory, given a file
        ],
    )
    def test_raises_on_wrong_type_of_path(
        self, tmp_path, full_config_dict, section, field, wrong_kind
    ):
        wrong_path = tmp_path / f"wrong_{field}"
        if wrong_kind == "file":
            wrong_path.touch()
        else:
            wrong_path.mkdir()

        full_config_dict[section][field] = str(wrong_path)
        model_cls = {
            "paths": PathsConfig,
            "segmentation": SegmentationConfig,
            "detection": DetectionConfig,
        }[section]

        with pytest.raises(ValidationError):
            model_cls(**full_config_dict[section])


class TestRequiredFields:
    def test_raises_with_missing_section(self, full_config_dict):
        del full_config_dict["paths"]

        with pytest.raises(ValueError):
            PipelineConfig(**full_config_dict)

    @pytest.mark.parametrize(
        "section, field",
        [
            ("paths", "raw_data_dir"),
            ("paths", "out_dir"),
            ("channels", "segmentation_image"),
            ("channels", "spot_image"),
        ],
    )
    def test_raises_with_missing_required_fields(
        self, full_config_dict, section, field
    ):
        del full_config_dict[section][field]

        with pytest.raises(ValueError):
            PipelineConfig(**full_config_dict)


class TestFrozen:
    @pytest.mark.parametrize(
        "section, field",
        [
            ("mode", "do_3d"),
            ("paths", "out_dir"),
            ("channels", "segmentation_image"),
            ("segmentation", "bin_factor"),
            ("detection", "prob_thresh"),
        ],
    )
    def test_mutating_nested_field_raises_error(self, full_config_dict, section, field):
        result = PipelineConfig(**full_config_dict)
        current_value = getattr(getattr(result, section), field)

        with pytest.raises(ValidationError):
            setattr(getattr(result, section), field, current_value)

    @pytest.mark.parametrize(
        "section", ["mode", "paths", "channels", "segmentation", "detection"]
    )
    def test_reassigning_section_raises_error(self, full_config_dict, section):
        result = PipelineConfig(**full_config_dict)
        current_value = getattr(result, section)

        with pytest.raises(ValidationError):
            setattr(result, section, current_value)
