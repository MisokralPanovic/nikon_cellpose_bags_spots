import pytest
from pytest_mock import MockerFixture

from spot_detector.cli import main

# =====================================================================
# main()
# =====================================================================


def test_main_loads_config_and_calls_run_pipeline(
    mocker: MockerFixture, make_config, tmp_path
):
    config_file = tmp_path / "config.yml"
    config_file.touch()
    config = make_config()

    mock_load_config = mocker.patch(
        "spot_detector.cli.load_config",
        return_value=config,
    )
    mocker.patch("spot_detector.cli.configure_logging")
    mock_run_pipeline = mocker.patch("spot_detector.cli.run_pipeline")

    mocker.patch("sys.argv", ["spot-detector", str(config_file)])
    main()

    mock_load_config.assert_called_once_with(config_file)
    mock_run_pipeline.assert_called_once_with(config=config)


def test_main_requires_config_path_argument(mocker: MockerFixture):
    mocker.patch("sys.argv", ["spot-detector"])  # no argument given

    with pytest.raises(SystemExit):
        main()
