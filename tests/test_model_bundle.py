import pytest
from pytest_mock import MockerFixture

from spot_detector.utils import ModelBundle

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_from_pretrained(mocker):
    return mocker.patch("spot_detector.utils.Spotiflow.from_pretrained")


@pytest.fixture
def mock_validate_passtrhough(mocker):
    return mocker.patch.object(
        ModelBundle,
        "_validate_spotiflow_mode",
        side_effect=lambda model, config: model,
    )


# =====================================================================
# ModelBundle._load_cellpose
# =====================================================================


class TestLoadCellpose:
    def test_calls_cellpose_models_with_correct_path(
        self, mocker: MockerFixture, make_config
    ):
        config = make_config()
        mock_cellpose_cls = mocker.patch("spot_detector.utils.models.CellposeModel")

        ModelBundle._load_cellpose(config=config)

        mock_cellpose_cls.assert_called_once_with(
            gpu=config.segmentation.use_gpu,
            pretrained_model=str(config.segmentation.cellpose_model_path),
        )


# =====================================================================
# ModelBundle._load_spotiflow
# =====================================================================


class TestLoadSpotiflow:
    def test_loads_custom_model_successfully(
        self, mocker: MockerFixture, make_config, mock_validate_passtrhough
    ):
        config = make_config(mode={"do_3d": False})
        mock_model = mocker.MagicMock()
        mock_from_folder = mocker.patch(
            "spot_detector.utils.Spotiflow.from_folder", return_value=mock_model
        )
        result = ModelBundle._load_spotiflow(config=config)

        mock_from_folder.assert_called_once_with(
            str(config.detection.spotiflow_model_path)
        )
        assert result is mock_model

    def test_falls_back_to_pretrained_2d_when_custom_load_fails(
        self,
        mocker: MockerFixture,
        make_config,
        mock_from_pretrained,
        mock_validate_passtrhough,
    ):
        config = make_config(mode={"do_3d": False})
        mocker.patch(
            "spot_detector.utils.Spotiflow.from_folder",
            side_effect=Exception("not found"),
        )
        mock_fallback = mock_from_pretrained

        ModelBundle._load_spotiflow(config=config)

        mock_fallback.assert_called_once_with("synth_complex")

    def test_falls_back_to_pretrained_3d_when_custom_load_fails(
        self,
        mocker: MockerFixture,
        make_config,
        mock_from_pretrained,
        mock_validate_passtrhough,
    ):
        config = make_config(mode={"do_3d": True})
        mocker.patch(
            "spot_detector.utils.Spotiflow.from_folder",
            side_effect=Exception("not found"),
        )
        mock_fallback = mock_from_pretrained

        ModelBundle._load_spotiflow(config=config)

        mock_fallback.assert_called_once_with("smfish_3d")


# =====================================================================
# ModelBundle._validate_spotiflow_mode
# =====================================================================


class TestValidateSpotiflowMode:
    def test_returns_model_unchanged_when_mode_matches(
        self, mocker: MockerFixture, make_config
    ):
        mock_model = mocker.MagicMock()
        mock_model.config.is_3d = True

        result = ModelBundle._validate_spotiflow_mode(
            model=mock_model, config=make_config(mode={"do_3d": True})
        )

        assert result is mock_model

    def test_overrides_with_pretrained_when_mode_mismatches_2d_pipeline(
        self, mocker: MockerFixture, make_config, mock_from_pretrained
    ):
        mock_model = mocker.MagicMock()
        mock_model.config.is_3d = True  # model is 3D but pipeline wants 2D

        mock_fallback = mock_from_pretrained
        ModelBundle._validate_spotiflow_mode(
            model=mock_model, config=make_config(mode={"do_3d": False})
        )

        mock_fallback.assert_called_once_with("synth_complex")

    def test_overrides_with_pretrained_when_mode_mismatches_3d_pipeline(
        self, mocker: MockerFixture, make_config, mock_from_pretrained
    ):
        mock_model = mocker.MagicMock()
        mock_model.config.is_3d = False  # model is 2D but pipeline wants 3D

        mock_fallback = mock_from_pretrained
        ModelBundle._validate_spotiflow_mode(
            model=mock_model, config=make_config(mode={"do_3d": True})
        )

        mock_fallback.assert_called_once_with("smfish_3d")


# =====================================================================
# ModelBundle.load
# =====================================================================


class TestModelBundleLoad:
    def test_load_returns_bundle_with_both_models(
        self, mocker: MockerFixture, make_config
    ):
        mock_cellpose = mocker.MagicMock()
        mock_spotiflow = mocker.MagicMock()
        mocker.patch.object(ModelBundle, "_load_cellpose", return_value=mock_cellpose)
        mocker.patch.object(ModelBundle, "_load_spotiflow", return_value=mock_spotiflow)

        bundle = ModelBundle.load(config=make_config(mode={"do_3d": False}))

        assert bundle.cellpose is mock_cellpose
        assert bundle.spotiflow is mock_spotiflow
