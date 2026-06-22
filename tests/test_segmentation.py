import numpy as np
import pytest
from pytest_mock import MockerFixture
from spot_detector.segmentation_detection import segment_2d, segment_3d

@pytest.fixture
def mock_cellpose_2d(mocker: MockerFixture):
    """Fake CellposeModel whose eval() returns a small mask with one
    object touching the edge and one fully inside."""
    fake_masks = np.zeros((10,10), dtype=int)
    fake_masks[0:3, 0:3] = 1 # touches edge (0,0) -> should be removed
    fake_masks[4:7, 4:7] = 2 # interior -> should survive

    model = mocker.MagicMock()
    model.eval.return_value = (fake_masks, None, None)
    return model
    
@pytest.fixture
def mock_cellpose_3d(mocker: MockerFixture):
    """Fake CellposeModel for 3D: eval() returns one z-stack of masks."""
    fake_masks = np.zeros((3,10,10), dtype=int)
    fake_masks[:,0:3, 0:3] = 1 # touches edge (0,0) -> should be removed
    fake_masks[:,4:7, 4:7] = 2 # interior -> should survive

    model = mocker.MagicMock()
    model.eval.return_value = (fake_masks, None, None)
    return model

@pytest.fixture
def stack_2d():
    """Standard 40x40 2D float32 input image."""
    return np.random.rand(40, 40).astype(np.float32)
    
@pytest.fixture
def stack_3d():
    """Standard 3x40x40 3D float32 image stack."""
    return np.random.rand(3, 40, 40).astype(np.float32)

class TestSegment2D:
    def test_calls_model_eval(self, mock_cellpose_2d, stack_2d):
        segment_2d(bf_stack=stack_2d, model_cellpose=mock_cellpose_2d, factor=4)
        assert mock_cellpose_2d.eval.called
        
    def test_output_shape_matches_input_after_upscale(self, mock_cellpose_2d, stack_2d):
        # factor=4 downscales (40x40 -> 10x10), function should upscale back
        result = segment_2d(bf_stack=stack_2d, model_cellpose=mock_cellpose_2d, factor=4)
        assert result.shape == (40,40)
    
    def test_edge_touching_removal(self, mock_cellpose_2d, stack_2d):
        result = segment_2d(bf_stack=stack_2d, model_cellpose=mock_cellpose_2d, factor=4)
        # corner region (where the edge-touching object was) should be background
        assert result[0,0] == 0
        # interior region (where the surviving object was) should be foreground
        assert result[24, 24] != 0
        # exactly one object should remain after edge removal: background + 1 survivor
        assert len(np.unique(result)) == 2
        
    def test_3d_input_uses_stdev_projection(self, mock_cellpose_2d):
        stack = np.random.rand(5, 40, 40).astype(np.float32)
        segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)
        called_arg = mock_cellpose_2d.eval.call_args[0][0]
        assert called_arg.ndim == 2 # projected down to 2D before binning

class TestSegment3D:
    def test_calls_model_eval_with_3d_kwargs(self, mock_cellpose_3d, stack_3d):
        segment_3d(
            bf_stack=stack_3d, model_cellpose=mock_cellpose_3d, 
            factor=4, stitch_threshold=0.4
        )
        _, kwargs = mock_cellpose_3d.eval.call_args
        assert kwargs["do_3D"] is False
        assert kwargs["z_axis"] == 0
        assert kwargs["stitch_threshold"] == 0.4
    
    def test_output_shape(self, mock_cellpose_3d, stack_3d):
        result = segment_3d(
            bf_stack=stack_3d, model_cellpose=mock_cellpose_3d, 
            factor=4, stitch_threshold=0.4
        )
        assert result.shape == (3, 40, 40)
        
    def test_edge_clearing_applied_per_z_plane(self, mock_cellpose_3d, stack_3d):
        result = segment_3d(
            bf_stack=stack_3d, model_cellpose=mock_cellpose_3d, 
            factor=4, stitch_threshold=0.4
        )
        # edge-touching object (label 2, corner) removed on every z-plane
        for z in range(result.shape[0]):
            assert result[z, 0, 0] == 0