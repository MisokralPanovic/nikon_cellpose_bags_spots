import pytest
from pytest_mock import MockerFixture
from spot_detector.utils import ModelBundle


@pytest.fixture
def base_config():
    return {
        "paths": {
            "cellpose_models_path": "/fake/cellpose/path",
            "spotiflow_models_path": "/fake/spotiflow/path",
        }
    }


class TestLoadCellpose:
    def test_calls_cellpose_models_with_correct_path(
        self, mocker: MockerFixture, base_config
    ):
        mock_cellpose_cls = mocker.patch("spot_detector.utils.models.CellposeModel")
        ModelBundle._load_cellpose(config=base_config)
        mock_cellpose_cls.assert_called_once_with(
            gpu=True, pretrained_model="/fake/cellpose/path"
        )


class TestLoadSpotiflow:
    def test_loads_custom_model_successfully(self, mocker: MockerFixture, base_config):
        mock_model = mocker.MagicMock()
        mock_from_folder = mocker.patch(
            "spot_detector.utils.Spotiflow.from_folder", return_value=mock_model
        )
        mocker.patch.object(
            ModelBundle, "_validate_spotiflow_mode", return_value=mock_model
        )

        result = ModelBundle._load_spotiflow(config=base_config, do_3d=False)

        mock_from_folder.assert_called_once_with("/fake/spotiflow/path")
        assert result is mock_model

    def test_falls_back_to_pretrained_2d_when_custom_load_fails(
        self, mocker: MockerFixture, base_config
    ):
        mocker.patch(
            "spot_detector.utils.Spotiflow.from_folder",
            side_effect=Exception("not found"),
        )
        mock_fallback = mocker.patch("spot_detector.utils.Spotiflow.from_pretrained")
        mocker.patch.object(
            ModelBundle,
            "_validate_spotiflow_mode",
            side_effect=lambda model, do_3d: model,
        )

        ModelBundle._load_spotiflow(config=base_config, do_3d=False)

        mock_fallback.assert_called_once_with("synth_complex")

    def test_falls_back_to_pretrained_3d_when_custom_load_fails(
        self, mocker: MockerFixture, base_config
    ):
        mocker.patch(
            "spot_detector.utils.Spotiflow.from_folder",
            side_effect=Exception("not found"),
        )
        mock_fallback = mocker.patch("spot_detector.utils.Spotiflow.from_pretrained")
        mocker.patch.object(
            ModelBundle,
            "_validate_spotiflow_mode",
            side_effect=lambda model, do_3d: model,
        )

        ModelBundle._load_spotiflow(config=base_config, do_3d=True)

        mock_fallback.assert_called_once_with("smfish_3d")


class TestValidateSpotiflowMode:
    def test_returns_model_unchanged_when_mode_matches(self, mocker: MockerFixture):
        mock_model = mocker.MagicMock()
        mock_model.config.is_3d = True

        result = ModelBundle._validate_spotiflow_mode(model=mock_model, do_3d=True)

        assert result is mock_model

    def test_overrides_with_pretrained_when_mode_mismatches_2d_pipeline(
        self, mocker: MockerFixture
    ):
        mock_model = mocker.MagicMock()
        mock_model.config.is_3d = True  # model is 3D but pipeline wants 2D

        mock_fallback = mocker.patch("spot_detector.utils.Spotiflow.from_pretrained")
        ModelBundle._validate_spotiflow_mode(model=mock_model, do_3d=False)

        mock_fallback.assert_called_once_with("synth_complex")

    def test_overrides_with_pretrained_when_mode_mismatches_3d_pipeline(
        self, mocker: MockerFixture
    ):
        mock_model = mocker.MagicMock()
        mock_model.config.is_3d = False  # model is 2D but pipeline wants 3D

        mock_fallback = mocker.patch("spot_detector.utils.Spotiflow.from_pretrained")
        ModelBundle._validate_spotiflow_mode(model=mock_model, do_3d=True)

        mock_fallback.assert_called_once_with("smfish_3d")


class TestModelBundleLoad:
    def test_load_returns_bundle_with_both_models(
        self, mocker: MockerFixture, base_config
    ):
        mock_cellpose = mocker.MagicMock()
        mock_spotiflow = mocker.MagicMock()
        mocker.patch.object(ModelBundle, "_load_cellpose", return_value=mock_cellpose)
        mocker.patch.object(ModelBundle, "_load_spotiflow", return_value=mock_spotiflow)

        bundle = ModelBundle.load(config=base_config, do_3d=False)

        assert bundle.cellpose is mock_cellpose
        assert bundle.spotiflow is mock_spotiflow
