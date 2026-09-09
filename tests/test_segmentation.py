import logging

import numpy as np
import pytest
from pytest_mock import MockerFixture

from spot_detector.segmentation_detection import segment_2d, segment_3d

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_cellpose_2d(mocker: MockerFixture):
    """Fake CellposeModel: eval() returns a mask shaped like its input
    (as real Cellpose does), with a corner object (edge -> stripped) and
    a centered object (survives)."""

    def fake_eval(img, **kwargs):
        h, w = img.shape[-2:]
        m = np.zeros((h, w), dtype=int)
        m[0:2, 0:2] = 1  # corner -> edge touching
        m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 2  # centered -> interior
        return (m, None, None)

    model = mocker.MagicMock()
    model.eval.side_effect = fake_eval
    return model


@pytest.fixture
def mock_cellpose_3d(mocker: MockerFixture):
    """Fake CellposeModel for 3D: eval() returns a mask stack shaped like its input."""

    def fake_eval(img, **kwargs):
        z, h, w = img.shape
        m = np.zeros((z, h, w), dtype=int)
        m[:, 0:2, 0:2] = 1  # corner -> edge touching
        m[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 2  # centered -> interior
        return (m, None, None)

    model = mocker.MagicMock()
    model.eval.side_effect = fake_eval
    return model


# =====================================================================
# segment_2d
# =====================================================================


class TestSegment2D:
    def test_calls_model_eval(self, mock_cellpose_2d, make_stack):
        stack = make_stack((40, 40))
        segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)
        assert mock_cellpose_2d.eval.called

    def test_output_shape_divisible(self, mock_cellpose_2d, make_stack):
        stack = make_stack((40, 40))
        # factor=4 downscales (40x40 -> 10x10), function should upscale back and crop back to input shape
        result = segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)
        assert result.shape == (40, 40)

    def test_output_shape_non_divisible(self, mock_cellpose_2d, make_stack):
        stack = make_stack((41, 41))
        # 41 not divisible by bin_factor 4 -> block_reduce zero-pads -> upscaled mask overshoots to (44, 44) and must be cropped back.
        result = segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)
        assert result.shape == (41, 41)

    def test_non_square_non_divisible(self, mock_cellpose_2d, make_stack):
        stack = make_stack((41, 37))

        result = segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)
        assert result.shape == (41, 37)

    def test_edge_touching_removal(self, mock_cellpose_2d, make_stack):
        stack = make_stack((40, 40))
        result = segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)
        # corner region (where the edge-touching object was) should be background
        assert result[0, 0] == 0
        # interior region (where the surviving object was) should be foreground
        assert result[20, 20] != 0
        # exactly one object should remain after edge removal: background + 1 survivor
        assert len(np.unique(result)) == 2

    def test_3d_input_uses_stdev_projection(self, mock_cellpose_2d, make_stack):
        stack = make_stack((5, 40, 40))
        segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)
        called_arg = mock_cellpose_2d.eval.call_args[0][0]
        assert called_arg.ndim == 2  # projected down to 2D before binning

    # --- bin_factor / block_reduce zero-padding ---

    def test_logs_when_padding_needed(self, mock_cellpose_2d, make_stack, caplog):
        caplog.set_level(logging.DEBUG, logger="spot_detector.segmentation_detection")
        stack = make_stack((41, 41))
        segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)

        assert "not divisible by bin_factor" in caplog.text.lower()

    def test_no_log_when_divisible(self, mock_cellpose_2d, make_stack, caplog):
        caplog.set_level(logging.DEBUG, logger="spot_detector.segmentation_detection")
        stack = make_stack((40, 40))
        segment_2d(bf_stack=stack, model_cellpose=mock_cellpose_2d, factor=4)

        assert "not divisible by bin_factor" not in caplog.text.lower()


# =====================================================================
# segment_3d
# =====================================================================


class TestSegment3D:
    def test_calls_model_eval_with_3d_kwargs(self, mock_cellpose_3d, make_stack):
        stack = make_stack((3, 40, 40))
        segment_3d(
            bf_stack=stack,
            model_cellpose=mock_cellpose_3d,
            factor=4,
            stitch_threshold=0.4,
        )
        _, kwargs = mock_cellpose_3d.eval.call_args
        assert kwargs["do_3D"] is False
        assert kwargs["z_axis"] == 0
        assert kwargs["stitch_threshold"] == 0.4

    def test_output_shape_divisible(self, mock_cellpose_3d, make_stack):
        stack = make_stack((3, 40, 40))
        result = segment_3d(
            bf_stack=stack,
            model_cellpose=mock_cellpose_3d,
            factor=4,
            stitch_threshold=0.4,
        )
        assert result.shape == (3, 40, 40)

    def test_output_shape_non_divisible(self, mock_cellpose_3d, make_stack):
        stack = make_stack((3, 41, 41))
        result = segment_3d(
            bf_stack=stack,
            model_cellpose=mock_cellpose_3d,
            factor=4,
            stitch_threshold=0.4,
        )
        assert result.shape == (3, 41, 41)

    def test_edge_clearing_applied_per_z_plane(self, mock_cellpose_3d, make_stack):
        stack = make_stack((3, 40, 40))
        result = segment_3d(
            bf_stack=stack,
            model_cellpose=mock_cellpose_3d,
            factor=4,
            stitch_threshold=0.4,
        )
        # edge-touching object (label 1, corner) removed on every z-plane
        for z in range(result.shape[0]):
            assert result[z, 0, 0] == 0

    # --- bin_factor / block_reduce zero-padding ---

    def test_logs_when_padding_needed(self, mock_cellpose_3d, make_stack, caplog):
        caplog.set_level(logging.DEBUG, logger="spot_detector.segmentation_detection")
        stack = make_stack((3, 41, 41))
        segment_3d(
            bf_stack=stack,
            model_cellpose=mock_cellpose_3d,
            factor=4,
            stitch_threshold=0.4,
        )

        assert "not divisible by bin_factor" in caplog.text.lower()
