import pytest
from pytest_mock import MockerFixture

from spot_detector.cli import main


def test_main_loads_config_and_calls_run_pipeline(mocker: MockerFixture, tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("mode:\n  do_3d: false\npaths:\n  out_dir: output\n")

    mock_load_config = mocker.patch(
        "spot_detector.cli.load_config",
        return_value={"mode": {"do_3d": False}, "paths": {"out_dir": "output"}},
    )
    mocker.patch("spot_detector.cli.configure_logging")
    mock_run_pipeline = mocker.patch("spot_detector.cli.run_pipeline")

    mocker.patch("sys.argv", ["spot-detector", str(config_file)])
    main()

    mock_load_config.assert_called_once_with(config_file)
    mock_run_pipeline.assert_called_once_with(
        config={"mode": {"do_3d": False}, "paths": {"out_dir": "output"}}
    )


def test_main_requires_config_path_argument(mocker: MockerFixture):
    mocker.patch("sys.argv", ["spot-detector"])  # no argument given

    with pytest.raises(SystemExit):
        main()
