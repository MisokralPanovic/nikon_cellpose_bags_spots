import numpy as np
import pytest
from pytest_mock import MockerFixture
from types import SimpleNamespace
from spot_detector.segmentation_detection import detect_spots_spotiflow

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_spotiflow(mocker: MockerFixture):
    fake_points = np.array([[5.0, 5.0], [12.0, 8.0]])
    fake_details = SimpleNamespace(flow=np.zeros((10, 10, 3)))

    model = mocker.MagicMock()
    model.predict.return_value = (fake_points, fake_details)
    return model


@pytest.fixture
def base_params():
    return {
        "prob_thresh": 0.5,
        "min_distance": 10,
        "do_3d": False,
    }


# =====================================================================
# detect_spots_spotiflow
# =====================================================================


class TestDetectSpotsSpotiflow:
    def test_calls_predict_with_expected_kwargs(
        self, mock_spotiflow, make_stack, base_params
    ):
        stack = make_stack((20, 20))
        detect_spots_spotiflow(
            spot_stack=stack, model_spotiflow=mock_spotiflow, **base_params
        )
        _, kwargs = mock_spotiflow.predict.call_args
        assert kwargs["prob_thresh"] == 0.5
        assert kwargs["min_distance"] == 10
        assert kwargs["verbose"] is False

    def test_returns_points_and_details(self, mock_spotiflow, make_stack, base_params):
        stack = make_stack((20, 20))
        points, details = detect_spots_spotiflow(
            spot_stack=stack, model_spotiflow=mock_spotiflow, **base_params
        )
        assert len(points) == 2
        assert hasattr(details, "flow")

    def test_2d_mode_max_projects_multiplane_stack(
        self, mock_spotiflow, make_stack, base_params
    ):
        stack = make_stack((3, 20, 20))
        # a (Z,Y,X) stack in 2D mode should be max-projected before predict
        detect_spots_spotiflow(
            spot_stack=stack, model_spotiflow=mock_spotiflow, **base_params
        )
        called_img = mock_spotiflow.predict.call_args[1]["img"]
        assert called_img.ndim == 2

    def test_3d_mode_keeps_full_stack(self, mock_spotiflow, make_stack, base_params):
        params = {**base_params, "do_3d": True}
        stack = make_stack((3, 20, 20))

        detect_spots_spotiflow(
            spot_stack=stack, model_spotiflow=mock_spotiflow, **params
        )
        called_img = mock_spotiflow.predict.call_args[1]["img"]
        assert called_img.ndim == 3

    def test_single_plane_2d_input_squeezed(
        self, mock_spotiflow, make_stack, base_params
    ):
        # shape (1, Y, X) in 2D mode should squeeze to (Y, X), not max-project
        stack = make_stack((1, 20, 20))
        detect_spots_spotiflow(
            spot_stack=stack, model_spotiflow=mock_spotiflow, **base_params
        )
        called_img = mock_spotiflow.predict.call_args[1]["img"]
        assert called_img.shape == (20, 20)
