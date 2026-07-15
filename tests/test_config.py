import pytest
from spot_detector.config import load_config


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("paths:\n raw_data_dir: data\n")

    result = load_config(config_file)

    assert result["paths"]["raw_data_dir"] == "data"


def test_load_config_handels_nested_structure(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "mode:\n  do_3d: true\n"
        "segmentation:\n  model_name: test_model\n  bin_factor: 4\n"
    )

    result = load_config(config_file)

    assert result["mode"]["do_3d"] is True
    assert result["segmentation"]["bin_factor"] == 4


def test_load_config_raises_on_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.yml"

    with pytest.raises(FileNotFoundError):
        load_config(missing_path)


def test_load_config_raises_on_invalid_yaml(tmp_path):
    config_file = tmp_path / "bad_config.yml"
    config_file.write_text("paths: [unclosed\n  this is not valid: yaml:")

    with pytest.raises(Exception):
        load_config(config_file)
