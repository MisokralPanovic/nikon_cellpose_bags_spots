import numpy as np
import pytest
from pytest_mock import MockerFixture
from types import SimpleNamespace
from spot_detector.segmentation_detection import detect_spots_spotiflow

@pytest.fixture
def mock_spotiflow(mocker: MockerFixture):
    fake_points = np.array([[5.0, 5.0], [12.0, 8.0]])
    fake_details = SimpleNamespace(flow=np.zeros((10, 10, 3)))
    
    model = mocker.MagicMock()
    model.predict.return_value = (fake_points, fake_details)
    return model

@pytest.fixture
def stack_2d():
    """Standard 40x40 2D float32 input image."""
    return np.random.rand(20, 20).astype(np.float32)
    
@pytest.fixture
def stack_3d():
    """Standard 3x40x40 3D float32 image stack."""
    return np.random.rand(5, 20, 20).astype(np.float32)

@pytest.fixture
def base_params():
    return {
        "prob_thresh": 0.5,
        "min_distance": 10,
        "do_3d": False,     
    }
class TestDetectSpotsSpotiflow:
    
    def test_calls_predict_with_expected_kwargs(self, mock_spotiflow, stack_2d, base_params):
        detect_spots_spotiflow(
            spot_stack=stack_2d, model_spotiflow=mock_spotiflow,
            **base_params
        )
        _, kwargs = mock_spotiflow.predict.call_args
        assert kwargs["prob_thresh"] == 0.5
        assert kwargs["min_distance"] == 10
        assert kwargs["verbose"] is False
    
    def test_returns_points_and_details(self, mock_spotiflow, stack_2d, base_params):
        points, details = detect_spots_spotiflow(
            spot_stack=stack_2d, model_spotiflow=mock_spotiflow,
            **base_params
        )
        assert len(points) == 2
        assert hasattr(details, "flow")
        
    def test_2d_mode_max_projects_multiplane_stack(self, mock_spotiflow, stack_3d, base_params):
        # a (Z,Y,X) stack in 2D mode should be max-projected before predict
        detect_spots_spotiflow(
            spot_stack=stack_3d, model_spotiflow=mock_spotiflow,
            **base_params
        )
        called_img = mock_spotiflow.predict.call_args[1]["img"]
        assert called_img.ndim == 2
        
    def test_3d_mode_keeps_full_stack(self, mock_spotiflow, stack_3d, base_params):
        params = {**base_params, "do_3d": True}
        detect_spots_spotiflow(
            spot_stack=stack_3d, model_spotiflow=mock_spotiflow,
            **params
        )
        called_img = mock_spotiflow.predict.call_args[1]["img"]
        assert called_img.ndim == 3
        
    def test_single_plane_2d_input_squeezed(self, mock_spotiflow, base_params):
        # shape (1, Y, X) in 2D mode should squeeze to (Y, X), not max-project
        stack = np.random.rand(1, 20, 20).astype(np.float32)
        detect_spots_spotiflow(
            spot_stack=stack, model_spotiflow=mock_spotiflow,
            **base_params
        )
        called_img = mock_spotiflow.predict.call_args[1]["img"]
        assert called_img.shape == (20, 20)